"""
Stonefish Bot -- The Main Bot Class
====================================
Orchestrates puzzle-based move selection with three-tier Maia detection,
adaptive depth, mate retry, conversion mode, and game state.

Two interfaces:
    choose_move(board)      -- Simple, backward-compatible with play_game()
    choose_move_full(board) -- Full, returns MoveResult for logging/chat
"""

import chess
from typing import Optional

from .config import StonefishConfig
from .game_state import StonefishGameState, MoveResult, OpponentMoveResult
from .scoring import select_stonefish_move, rank_opponent_move
from .maia import MaiaEngine


class StonefishBot:
    """The Stonefish chess bot.

    Plays puzzle-creating moves using three-tier Maia disagreement detection.
    "I don't expect you to play like an engine. I expect you to play like
    a better version of yourself."

    For local play (backward-compatible with play_game):
        bot = StonefishBot(engine, config)
        move = bot.choose_move(board)

    For Lichess (full data access):
        bot = StonefishBot(engine, config)
        bot.start_game(chess.WHITE)
        result = bot.choose_move_full(board)
        bot.notify_opponent_move(board, opponent_move)
    """

    def __init__(self, engine, config: Optional[StonefishConfig] = None,
                 maia: Optional[MaiaEngine] = None):
        self.engine = engine
        self.config = config or StonefishConfig()
        self._game_state: Optional[StonefishGameState] = None
        # Maia engine (created lazily if not provided)
        self._maia = maia

    @property
    def maia(self) -> MaiaEngine:
        if self._maia is None:
            self._maia = MaiaEngine(stockfish_engine=self.engine)
        return self._maia

    @property
    def name(self) -> str:
        return (f"Stonefish(floor={self.config.floor_rating},"
                f"stretch={self.config.stretch_rating},"
                f"reach={self.config.reach_rating})")

    @property
    def game_state(self) -> Optional[StonefishGameState]:
        return self._game_state

    def start_game(self, our_color: chess.Color):
        """Initialize per-game state. Call before the first move."""
        self._game_state = StonefishGameState(self.config, our_color)

    def end_game(self, board: Optional[chess.Board] = None) -> Optional[StonefishGameState]:
        """Clean up and return the game state for logging/analysis."""
        if board is not None and self._game_state is not None:
            self._infer_opponent_move(board)
        state = self._game_state
        self._game_state = None
        return state

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Simple interface: returns just the move.

        Auto-initializes game state if needed.
        Infers opponent's last move from board.move_stack.
        """
        if self._game_state is None:
            self.start_game(board.turn)

        self._infer_opponent_move(board)

        result = self.choose_move_full(board)
        return result.move

    def choose_move_full(self, board: chess.Board,
                         our_clock_seconds: float = None) -> MoveResult:
        """Full interface: returns MoveResult with all analysis data.

        Call start_game() first, or let choose_move() auto-init.
        """
        if self._game_state is None:
            self.start_game(board.turn)

        current_shallow = self._game_state.adaptive.current_depth
        move_number = board.fullmove_number

        result = select_stonefish_move(
            self.engine,
            board,
            self.config,
            current_shallow,
            move_number,
            our_clock_seconds=our_clock_seconds,
            maia=self.maia,
            game_state=self._game_state,
        )

        self._game_state.on_our_move(result)
        return result

    def notify_opponent_move(self, board: chess.Board, move: chess.Move):
        """Analyze the opponent's move and update state.

        Args:
            board: Position BEFORE the opponent's move (opponent to move)
            move: The move the opponent played
        """
        if self._game_state is None:
            return

        was_puzzle = self._game_state.last_was_puzzle
        puzzle_type = self._game_state.last_puzzle_type
        puzzle_floor_prob = self._game_state.last_puzzle_floor_prob

        rank, eval_cost, best_move_san = rank_opponent_move(
            self.engine,
            board,
            move,
            self.config.target_band,
            self.config.deep_depth,
        )

        found = rank < self.config.target_band if was_puzzle else False

        try:
            move_san = board.san(move)
        except Exception:
            move_san = move.uci()

        opp_result = OpponentMoveResult(
            move=move,
            move_san=move_san,
            move_rank=rank,
            eval_cost=round(eval_cost, 3),
            was_puzzle=was_puzzle,
            found_puzzle=found,
            best_response_san=best_move_san,
            puzzle_type=puzzle_type,
            puzzle_floor_prob=puzzle_floor_prob,
        )

        self._game_state.on_opponent_move(opp_result)
        return opp_result

    def _infer_opponent_move(self, board: chess.Board):
        """Detect the opponent's last move from board.move_stack."""
        if self._game_state is None:
            return

        actual_half_moves = len(board.move_stack)
        tracked = self._game_state._total_move_count

        if actual_half_moves <= tracked:
            return

        if actual_half_moves > tracked:
            opponent_move = board.move_stack[tracked]

            temp_board = chess.Board()
            for i in range(tracked):
                temp_board.push(board.move_stack[i])

            self.notify_opponent_move(temp_board, opponent_move)

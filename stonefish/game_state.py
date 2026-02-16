"""
Stonefish Game State — Per-Move Data Classes
=============================================
MoveResult captures everything about a Stonefish move decision.
OpponentMoveResult captures how the opponent responded.
StonefishGameState holds the mutable state for one game.
"""

import chess
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .adaptive import AdaptiveDepthTracker
from .config import StonefishConfig


@dataclass
class MoveResult:
    """Complete result from a single Stonefish move selection.

    This is the central data structure — the logger, chat engine, and game
    handler all consume it.
    """
    # Our move
    move: chess.Move
    move_san: str
    move_rank: int                              # 0 = played the top Stockfish move
    top_engine_move: str                        # SAN of Stockfish's #1 choice

    # Evaluations
    deep_eval: float                            # Eval from deep pass (our perspective, pawns)
    shallow_eval: float                         # Eval from shallow pass
    candidate_evals: List[Tuple[str, float]]    # [(move_san, eval), ...] all candidates

    # Nettlesomeness
    nettlesomeness_score: float
    gap: float                              # Move 1 vs move 2 gap (opponent's perspective)
    disagreement: int                       # Rank of deep best move at shallow depth (0 = obvious)

    # Criticality of the resulting position (for the opponent)
    was_flagged_critical: bool

    # Depth info
    current_shallow_depth: int
    deep_depth: int

    # Context
    move_number: int

    emergency_mode: bool = False   # True if played under time pressure

    # Progressive search info
    search_depth_found: int = 0            # 0=fallback, 1=immediate, 2=two-move chain, 3=three-move chain
    search_threshold_used: float = 0.0     # Gap threshold that was met (1.0 or 0.5)


@dataclass
class OpponentMoveResult:
    """Analysis of what the opponent played."""
    move: chess.Move
    move_san: str
    move_rank: int              # 0-indexed; >= 2*target_band if not in top moves
    eval_cost: float            # Pawns lost vs best move
    was_critical: bool          # Was this a flagged critical position?
    found_critical: bool        # Did they play a move in the good cluster?
    best_response_san: str = "" # SAN of the best move in this position


class StonefishGameState:
    """Mutable per-game state for one Stonefish game.

    Created at game start, destroyed at game end.
    Holds the adaptive depth tracker, move history, and criticality tracking.
    """

    def __init__(self, config: StonefishConfig, our_color: chess.Color):
        self.config = config
        self.our_color = our_color
        self.adaptive = AdaptiveDepthTracker(
            base_depth=config.base_depth,
            boost_on_miss=config.depth_boost_on_miss,
            boost_duration=config.boost_duration_moves,
            target_band=config.target_band,
        )
        self.move_results: List[MoveResult] = []
        self.opponent_results: List[OpponentMoveResult] = []
        self.last_was_critical: bool = False
        self._our_move_count: int = 0
        self._total_move_count: int = 0  # Half-moves we've tracked
        self._last_flagged_move: int = -100  # Move number of last flagged critical moment

    def on_our_move(self, result: MoveResult):
        """Record our move and update state.

        Applies critical cooldown: if we flagged a critical moment recently
        (within critical_cooldown_moves), suppress the flag on this move.
        """
        # Apply cooldown — suppress flag if too recent
        cooldown = self.config.critical_cooldown_moves
        if result.was_flagged_critical and cooldown > 0:
            moves_since_last = result.move_number - self._last_flagged_move
            if moves_since_last <= cooldown:
                result.was_flagged_critical = False

        # Track the flag for cooldown
        if result.was_flagged_critical:
            self._last_flagged_move = result.move_number

        self.move_results.append(result)
        self.last_was_critical = result.was_flagged_critical
        self._our_move_count += 1
        self._total_move_count += 1
        self.adaptive.tick()

    def on_opponent_move(self, result: OpponentMoveResult):
        """Record opponent's move and update adaptive depth."""
        self.opponent_results.append(result)
        # Pass the full-move number for tracking when the adjustment was triggered
        move_number = self._our_move_count
        self.adaptive.record_opponent_move(
            result.move_rank,
            result.was_critical,
            move_number,
        )
        self.last_was_critical = False
        self._total_move_count += 1

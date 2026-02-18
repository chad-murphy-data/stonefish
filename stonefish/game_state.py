"""
Stonefish Game State -- Per-Move Data Classes
=============================================
PuzzleResult captures puzzle detection output from the three-tier Maia system.
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
class PuzzleResult:
    """Result from the unified puzzle detector (three-tier Maia disagreement)."""
    # Core
    is_puzzle: bool
    puzzle_type: str                      # "negative", "positive", "mate"
    disagreement_type: str                # "floor_v_stretch", "floor_v_reach", "stretch_v_reach", "mate"
    score: float

    # Moves
    our_move: chess.Move                  # What Stonefish plays to create this puzzle
    our_move_san: str
    worse_move: chess.Move                # What Floor tier predicts (the mistake)
    worse_move_san: str
    better_move: chess.Move               # What Stretch/Reach tier predicts (the correct move)
    better_move_san: str

    # Eval
    eval_gap: float                       # Cost of playing worse_move vs better_move
    eval_cost_to_create: float            # What Stonefish sacrifices to create this puzzle

    # Maia distributions
    floor_prob_correct: float             # "X% of players at your level find this"
    stretch_prob_correct: float
    reach_prob_correct: float
    tiers_agreeing: int                   # How many tiers agree on the correct move (2 or 3)

    # Mate-specific
    is_mate: bool = False
    mate_distance: int = 0                # Mate in N (0 if not a mate puzzle)
    retries_remaining: int = 0

    # Search metadata
    search_depth: int = 0                 # 0=immediate, 1=one-ahead, 2=two-ahead
    move_number: int = 0


@dataclass
class MoveResult:
    """Complete result from a single Stonefish move selection."""
    # Core move info
    move: chess.Move
    move_san: str
    move_rank: int
    top_engine_move: str
    deep_eval: float
    move_number: int
    emergency_mode: bool = False

    # Puzzle info
    puzzle_found: bool = False
    puzzle_type: str = "none"             # "negative", "positive", "mate", "none"
    puzzle_score: float = 0.0
    puzzle_eval_gap: float = 0.0
    puzzle_floor_prob: float = 0.0        # "X% at your level find this"
    puzzle_disagreement: str = ""         # Which tiers disagreed
    puzzle_search_depth: int = 0
    puzzle_better_move_san: str = ""      # What they should play
    puzzle_worse_move_san: str = ""       # What Floor predicts they'd play

    # Mate-specific
    puzzle_is_mate: bool = False
    puzzle_mate_distance: int = 0
    puzzle_retries_remaining: int = 0

    # Screening stats
    positions_screened: int = 0
    disagreements_found: int = 0
    puzzles_after_validation: int = 0

    # Backward-compat / logging
    candidate_evals: List[Tuple[str, float]] = field(default_factory=list)
    current_shallow_depth: int = 6
    deep_depth: int = 12


@dataclass
class OpponentMoveResult:
    """Analysis of what the opponent played."""
    move: chess.Move
    move_san: str
    move_rank: int                        # 0-indexed rank among top moves
    eval_cost: float                      # Pawns lost vs best move
    was_puzzle: bool                      # Was this a flagged puzzle position?
    found_puzzle: bool                    # Did they find the correct move?
    best_response_san: str = ""           # SAN of the objectively best move
    puzzle_type: str = "none"             # "negative", "positive", "mate", "none"
    puzzle_floor_prob: float = 0.0        # "X% at your level find this"


class StonefishGameState:
    """Mutable per-game state for one Stonefish game.

    Created at game start, destroyed at game end.
    Holds the adaptive depth tracker, move history, puzzle tracking,
    mate retry state, and conversion mode state.
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
        self.last_was_puzzle: bool = False
        self.last_puzzle_type: str = "none"
        self.last_puzzle_floor_prob: float = 0.0
        self._our_move_count: int = 0
        self._total_move_count: int = 0
        self._last_flagged_move: int = -100

        # Puzzle tracking for adaptive threshold
        self.puzzles_found_so_far: int = 0
        self.puzzles_solved_by_opponent: int = 0

        # Conversion mode: after opponent misses, play Stockfish #1
        self.conversion_moves_remaining: int = 0

        # Mate retry state
        self.mate_retry_active: bool = False
        self.mate_distance: int = 0
        self.mate_retries_used: int = 0
        self.mate_retries_allowed: int = 0

    @property
    def total_puzzles_created(self) -> int:
        return sum(1 for o in self.opponent_results if o.was_puzzle)

    @property
    def total_puzzles_solved(self) -> int:
        return sum(1 for o in self.opponent_results if o.was_puzzle and o.found_puzzle)

    def on_our_move(self, result: MoveResult):
        """Record our move and update state."""
        # Apply cooldown
        cooldown = self.config.critical_cooldown_moves
        if result.puzzle_found and cooldown > 0:
            moves_since_last = result.move_number - self._last_flagged_move
            if moves_since_last <= cooldown:
                result.puzzle_found = False
                result.puzzle_type = "none"

        if result.puzzle_found:
            self._last_flagged_move = result.move_number
            self.puzzles_found_so_far += 1

        self.move_results.append(result)
        self.last_was_puzzle = result.puzzle_found
        self.last_puzzle_type = result.puzzle_type
        self.last_puzzle_floor_prob = result.puzzle_floor_prob
        self._our_move_count += 1
        self._total_move_count += 1
        self.adaptive.tick()

        # Decrement conversion counter
        if self.conversion_moves_remaining > 0:
            self.conversion_moves_remaining -= 1

    def on_opponent_move(self, result: OpponentMoveResult):
        """Record opponent's move and update adaptive depth + conversion."""
        self.opponent_results.append(result)
        move_number = self._our_move_count

        self.adaptive.record_opponent_move(
            result.move_rank,
            result.was_puzzle,
            move_number,
        )

        # If opponent missed a puzzle, enter conversion mode
        if result.was_puzzle and not result.found_puzzle:
            self.conversion_moves_remaining = self.config.conversion_moves

        # If opponent solved a puzzle, cancel conversion
        if result.was_puzzle and result.found_puzzle:
            self.conversion_moves_remaining = 0
            self.puzzles_solved_by_opponent += 1

        # Handle mate retry state
        if self.mate_retry_active and result.was_puzzle:
            if result.found_puzzle:
                # They found the mate -- clear retry state
                self.mate_retry_active = False
                self.mate_distance = 0
                self.mate_retries_used = 0
            else:
                # They missed -- increment retry counter
                self.mate_retries_used += 1
                if self.mate_retries_used >= self.mate_retries_allowed:
                    self.mate_retry_active = False

        self.last_was_puzzle = False
        self.last_puzzle_type = "none"
        self._total_move_count += 1

    def enter_mate_retry(self, mate_distance: int):
        """Enter mate retry mode when a mate puzzle is detected."""
        budget = self.config.mate_retry_budget
        # Find the retry count for this mate distance (or nearest higher)
        retries = 0
        for dist in sorted(budget.keys()):
            if mate_distance <= dist:
                retries = budget[dist]
                break
        if not retries and budget:
            retries = budget[max(budget.keys())]

        self.mate_retry_active = True
        self.mate_distance = mate_distance
        self.mate_retries_used = 0
        self.mate_retries_allowed = retries

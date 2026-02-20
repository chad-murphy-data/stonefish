"""
Stonefish Configuration
=======================
All tunable parameters in one place. Nothing is hardcoded elsewhere.

The new Maia-tier system replaces the old Level 1/2/3 + divergence detection.
Puzzles are defined as gaps between what the opponent would do (Floor tier)
and what a slightly better version of them would do (Stretch/Reach tiers).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class StonefishConfig:
    # --- Maia tiers ---
    floor_rating: int = 1500            # Maia at opponent's rating
    stretch_rating: Optional[int] = 1700  # Maia at +200-400; None = use Stockfish
    reach_rating: Optional[int] = 2000    # Maia at +500-800; None = use Stockfish

    # --- Puzzle detection ---
    puzzle_eval_threshold: float = 1.0    # Min eval gap (pawns) to confirm puzzle
    soft_puzzle_eval_threshold: float = 0.5  # Lowered threshold when puzzles are scarce
    soft_puzzle_move_threshold: int = 15     # Move number to start lowering threshold
    puzzles_before_softening: int = 1        # If fewer puzzles by soft_puzzle_move_threshold, soften

    # --- Maia rollout validation ---
    rollout_depth: int = 6                # Half-moves to simulate (6 = 3 full moves of Maia vs Maia)
    rollout_averaging: int = 1            # Number of rollouts to average (1 = single, 2-3 = more reliable)

    # --- Mate puzzles ---
    max_mate_depth: int = 4               # Max mate-in-N to flag as puzzle
    mate_retry_budget: Dict[int, int] = field(
        default_factory=lambda: {2: 0, 3: 1, 4: 1, 5: 1}
    )

    # --- Positive puzzles ---
    enable_positive_puzzles: bool = True
    positive_not_capturable_depth: int = 1  # Depth to check "not immediately exploitable"

    # --- Generosity ---
    puzzle_generosity: float = 0.5        # 0.0 = all negative, 1.0 = all positive

    # --- Move selection ---
    num_candidates: int = 6
    max_eval_cost: float = 1.8
    opening_book_moves: int = 5
    conversion_moves: int = 4             # Moves of Stockfish #1 after opponent misses

    # --- Losing position uncap ---
    losing_eval_threshold: float = -1.5
    losing_max_eval_cost: float = 2.0
    desperate_eval_threshold: float = -3.0
    desperate_max_eval_cost: float = 99.0

    # --- Search depth ---
    max_lookahead: int = 2                # 0=immediate only, 1=one-ahead, 2=two-ahead

    # --- Engine depths ---
    deep_depth: int = 12                  # Deep pass: accurate eval
    base_depth: int = 6                   # Shallow pass: playing strength

    # --- Adaptive depth (boost only -- no reward mode) ---
    target_band: int = 1
    depth_boost_on_miss: int = 5
    boost_duration_moves: int = 5

    # --- Critical cooldown ---
    critical_cooldown_moves: int = 3

    # --- Chat ---
    chat_enabled: bool = True

    # --- Lichess ---
    min_time_control_seconds: int = 600
    max_concurrent_games: int = 3

    # --- Emergency clock mode ---
    emergency_clock_seconds: int = 60
    emergency_depth: int = 4

    # --- Stockfish engine ---
    stockfish_threads: int = 4
    stockfish_hash_mb: int = 512

"""
Stonefish Configuration
=======================
All tunable parameters in one place. Nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass


@dataclass
class StonefishConfig:
    # --- Engine depths ---
    deep_depth: int = 12          # Deep pass: accurate nettlesomeness + criticality
    base_depth: int = 6           # Shallow pass: playing strength (for 1500 version)

    # --- Move selection ---
    num_candidates: int = 6       # How many of our candidate moves to evaluate
    max_eval_cost: float = 1.8    # Max pawns we'll sacrifice for nettlesomeness (base)

    # --- Losing position uncap ---
    losing_eval_threshold: float = -1.5      # Eval worse than this: widen eval cost
    losing_max_eval_cost: float = 2.0        # Max eval cost when losing
    desperate_eval_threshold: float = -3.0   # Eval worse than this: fully uncap
    desperate_max_eval_cost: float = 99.0    # Uncapped nettlesomeness when desperate

    # --- Gap-based scoring (find THE move) ---
    target_band: int = 1          # Only rank 0 counts as "found"
    gap_threshold: float = 1.0    # High threshold: min gap for first pass
    gap_threshold_low: float = 0.5 # Low threshold: min gap for second pass
    critical_cooldown_moves: int = 3     # Suppress flagging for N moves after a critical moment

    # --- Depth disagreement (counterintuitive move filter) ---
    shallow_comparison_depth: int = 2    # Depth for "what looks obvious" check
    disagreement_threshold: int = 3      # Deep best move must rank this or worse at shallow depth to flag
    disagreement_bonus_multiplier: float = 0.5  # Reward for moves that look worse than they are

    # --- Adaptive depth (boost only — no reward mode) ---
    depth_boost_on_miss: int = 5         # Added to base_depth when opponent misses
    boost_duration_moves: int = 5        # How long a miss-boost lasts

    # --- Chat ---
    chat_enabled: bool = True
    high_gap_multiplier: float = 1.5   # Gap >= threshold * this = "high" criticality tier

    # --- Lichess ---
    min_time_control_seconds: int = 600  # 10 minutes minimum
    max_concurrent_games: int = 3

    # --- Opening phase ---
    opening_book_moves: int = 5           # First N moves: just play Stockfish's top move (skip progressive search)
    opening_moves: int = 10              # First N moves get boosted nettlesomeness (legacy, kept for compat)
    opening_nettlesomeness_boost: float = 1.5  # Multiplier in opening

    # --- Progressive search ---
    max_search_depth: int = 3            # Max chain depth to search (1=immediate only, 2=two-move chains, 3=full)

    # --- Emergency clock mode ---
    emergency_clock_seconds: int = 60 # Below this, skip deep pass entirely
    emergency_depth: int = 4          # Depth for emergency moves

    # --- Stockfish engine ---
    stockfish_threads: int = 4
    stockfish_hash_mb: int = 512

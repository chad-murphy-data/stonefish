"""
Stonefish Elo Presets -- Tier-Based Configuration
==================================================
Each preset configures the three Maia tiers (Floor/Stretch/Reach),
mate depth, retry budgets, generosity, and adaptive thresholds
for a given opponent rating.

When stretch_rating or reach_rating is None, Stockfish's best move
is used instead of a Maia prediction for that tier.
"""


ELO_PRESETS = {
    500: {
        "floor_rating": 500,
        "stretch_rating": 900,
        "reach_rating": 1400,
        "puzzle_generosity": 1.0,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 15,
        "rollout_depth": 6,             # 3 full moves — positions diverge fast
        "max_mate_depth": 2,
        "mate_retry_budget": {2: 1, 3: 2, 4: 3, 5: 3},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 2,
    },

    750: {
        "floor_rating": 750,
        "stretch_rating": 1100,
        "reach_rating": 1500,
        "puzzle_generosity": 0.85,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 15,
        "rollout_depth": 6,
        "max_mate_depth": 2,
        "mate_retry_budget": {2: 1, 3: 2, 4: 2, 5: 3},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 3,
    },

    1000: {
        "floor_rating": 1000,
        "stretch_rating": 1300,
        "reach_rating": 1700,
        "puzzle_generosity": 0.7,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 15,
        "rollout_depth": 8,             # 4 full moves
        "max_mate_depth": 3,
        "mate_retry_budget": {2: 1, 3: 1, 4: 2, 5: 2},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 3,
    },

    1250: {
        "floor_rating": 1250,
        "stretch_rating": 1500,
        "reach_rating": 1800,
        "puzzle_generosity": 0.55,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 20,
        "rollout_depth": 8,
        "max_mate_depth": 3,
        "mate_retry_budget": {2: 0, 3: 1, 4: 1, 5: 2},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 4,
    },

    1500: {
        "floor_rating": 1500,
        "stretch_rating": 1700,
        "reach_rating": 2000,
        "puzzle_generosity": 0.4,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 20,
        "rollout_depth": 8,
        "max_mate_depth": 4,
        "mate_retry_budget": {2: 0, 3: 1, 4: 1, 5: 1},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 4,
    },

    1750: {
        "floor_rating": 1750,
        "stretch_rating": 2000,
        "reach_rating": None,           # Stockfish as reach
        "puzzle_generosity": 0.25,
        "puzzle_eval_threshold": 1.0,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 25,
        "rollout_depth": 10,            # 5 full moves — subtle differences need time
        "max_mate_depth": 5,
        "mate_retry_budget": {2: 0, 3: 0, 4: 1, 5: 1},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 4,
    },

    2000: {
        "floor_rating": 2000,
        "stretch_rating": None,         # Stockfish as stretch
        "reach_rating": None,           # Stockfish as reach
        "puzzle_generosity": 0.1,
        "puzzle_eval_threshold": 0.8,
        "soft_puzzle_eval_threshold": 0.5,
        "soft_puzzle_move_threshold": 25,
        "rollout_depth": 10,
        "max_mate_depth": 6,
        "mate_retry_budget": {2: 0, 3: 0, 4: 0, 5: 1},
        "enable_positive_puzzles": True,
        "positive_not_capturable_depth": 1,
        "conversion_moves": 5,
    },
}


MATE_DEPTH_BY_ELO = {
    500: 2,
    750: 2,
    1000: 3,
    1250: 3,
    1500: 4,
    1750: 5,
    2000: 6,
}


def get_nearest_preset(elo: int) -> dict:
    """Return the preset dict for the nearest available elo bracket."""
    brackets = sorted(ELO_PRESETS.keys())
    best = min(brackets, key=lambda b: abs(b - elo))
    return ELO_PRESETS[best]


def apply_preset(config, elo: int):
    """Apply an elo preset to a StonefishConfig instance in-place.

    Only sets fields that exist on the config dataclass; ignores unknown keys.
    """
    preset = get_nearest_preset(elo)
    for key, value in preset.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

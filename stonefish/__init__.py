"""
Stonefish -- a chess bot that teaches you to recognize critical moments
through Maia-tier puzzle detection.
"""

from .config import StonefishConfig
from .bot import StonefishBot
from .presets import ELO_PRESETS, get_nearest_preset, apply_preset
from .maia import MaiaEngine

__all__ = [
    "StonefishConfig",
    "StonefishBot",
    "ELO_PRESETS",
    "get_nearest_preset",
    "apply_preset",
    "MaiaEngine",
]

"""
Stonefish Adaptive Depth — The Consequence System
==================================================
Depth increases after the opponent misses critical moments (Stonefish presses).
When the opponent finds a critical moment, depth returns to baseline immediately.

No reward mode: the reward for solving a critical moment is the position you
earned and the chat message, not an artificial handicap from the bot.

This controls CANDIDATE GENERATION depth — the set of moves the bot can "see".
The deep pass always evaluates nettlesomeness at full depth, but only among
candidates visible at the current shallow depth. Higher depth = stronger play
(sees better moves), baseline depth = calibrated play for the rating band.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DepthAdjustment:
    """An active depth boost (miss only — no reward mode)."""
    delta: int               # +N for miss-boost
    remaining_moves: int     # Moves left before expiry
    trigger_move: int        # Move number that triggered this
    trigger_type: str        # "miss"


class AdaptiveDepthTracker:
    """Tracks adaptive depth adjustments based on opponent behavior.

    After each opponent move on a critical position, call record_opponent_move.
    After each Stonefish move, call tick to decrement the countdown.
    Read current_depth to get the current shallow pass depth.

    State machine (target_band=1, so "find" = rank 0 only):
        base → opponent misses critical → base + boost for N moves → base
        base → opponent finds critical → return to base immediately (no reward)
        Active boost → new miss → replaces old boost
        Active boost → opponent finds → cancel boost, return to base
    """

    def __init__(
        self,
        base_depth: int,
        boost_on_miss: int,
        boost_duration: int,
        target_band: int,
        # Legacy kwargs accepted but ignored (for backward compatibility)
        reward_on_find: int = 0,
        reward_duration: int = 0,
    ):
        self.base_depth = base_depth
        self.boost_on_miss = boost_on_miss
        self.boost_duration = boost_duration
        self.target_band = target_band
        self._adjustment: Optional[DepthAdjustment] = None

    @property
    def current_depth(self) -> int:
        """Current shallow depth = base + active adjustment delta."""
        if self._adjustment and self._adjustment.remaining_moves > 0:
            return max(1, self.base_depth + self._adjustment.delta)
        return self.base_depth

    @property
    def active_adjustment(self) -> Optional[DepthAdjustment]:
        """The currently active adjustment, if any."""
        if self._adjustment and self._adjustment.remaining_moves > 0:
            return self._adjustment
        return None

    def record_opponent_move(self, move_rank: int, was_critical: bool, move_number: int):
        """Update adaptive state after opponent plays.

        Only triggers on critical positions.
        Miss → boost depth (press harder).
        Find → return to baseline immediately (no reward/penalty).

        Args:
            move_rank: 0-indexed rank of opponent's move among top moves
            was_critical: Whether this position was flagged as critical
            move_number: Current full-move number
        """
        if not was_critical:
            return

        found_it = move_rank < self.target_band

        if found_it:
            # Opponent solved it — cancel any active boost, return to baseline
            self._adjustment = None
        else:
            # Opponent missed — boost depth (press harder)
            self._adjustment = DepthAdjustment(
                delta=self.boost_on_miss,
                remaining_moves=self.boost_duration,
                trigger_move=move_number,
                trigger_type="miss",
            )

    def tick(self):
        """Decrement the active adjustment countdown. Call once per Stonefish move."""
        if self._adjustment:
            self._adjustment.remaining_moves -= 1
            if self._adjustment.remaining_moves <= 0:
                self._adjustment = None

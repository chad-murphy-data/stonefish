"""
Stonefish Chat Engine -- Message Generation
=============================================
Generates in-game chat messages and post-game summaries from per-move data.
Templates are loaded from chat_templates.json for easy editing.

5 triggers, no other messages:
    1. Game start (onboarding) — always sends
    2. Stonefish plays critical move (pre-critical alert) — skipped in quiet mode
    3. Opponent finds critical moment — skipped in quiet mode
    4. Opponent misses critical moment — skipped in quiet mode
    5. Game end (post-game summary) — always sends, returns list of messages
"""

import json
import os
import random
from typing import Optional, List

from .config import StonefishConfig
from .game_state import MoveResult, OpponentMoveResult, StonefishGameState


class ChatEngine:
    """Generates chat messages based on per-move analysis data.

    Messages are selected from templates and personalized with game data.
    Supports quiet mode (toggled by opponent saying "quiet"/"talk" in chat).
    """

    def __init__(self, config: StonefishConfig, templates_path: Optional[str] = None):
        self.config = config
        self.quiet = False

        if templates_path is None:
            templates_path = os.path.join(
                os.path.dirname(__file__), "chat_templates.json"
            )

        with open(templates_path) as f:
            self.templates = json.load(f)

    def on_game_start(self) -> str:
        """Trigger 1: Return the onboarding message. Always sends."""
        return self._pick("onboarding")

    def on_our_move(self, move_result: MoveResult) -> Optional[str]:
        """Trigger 2: Pre-critical alert after Stonefish plays a critical move.

        Returns None if position not critical or quiet mode is on.
        Selects moderate or high tier based on gap vs gap_threshold * high_gap_multiplier.
        """
        if self.quiet or not self.config.chat_enabled:
            return None

        if not move_result.was_flagged_critical:
            return None

        high_threshold = self.config.gap_threshold * self.config.high_gap_multiplier
        if move_result.gap >= high_threshold:
            return self._pick("high_criticality")
        else:
            return self._pick("moderate_criticality")

    def on_opponent_move(self, opp_result: OpponentMoveResult) -> Optional[str]:
        """Triggers 3 & 4: Response after opponent plays on a critical position.

        Returns None if the position wasn't critical or quiet mode is on.
        Does NOT include pawn cost or reveal best move for misses.
        """
        if self.quiet or not self.config.chat_enabled:
            return None

        if not opp_result.was_critical:
            return None

        if opp_result.found_critical:
            return self._pick("found_critical")
        else:
            return self._pick("missed_critical")

    def generate_post_game_summary(self, game_state: StonefishGameState) -> List[str]:
        """Trigger 5: Generate post-game summary as a list of separate messages.

        Returns a list of strings — each should be sent as a separate chat message
        with a short delay between them. Always sends regardless of quiet mode.

        Messages:
            1. Opening line (based on solve rate)
            2-N. One message per missed critical moment with move breakdown
        """
        critical_moments = [
            (i, opp) for i, opp in enumerate(game_state.opponent_results)
            if opp.was_critical
        ]
        found_count = sum(1 for _, opp in critical_moments if opp.found_critical)
        critical_count = len(critical_moments)

        messages: List[str] = []

        if critical_count == 0:
            messages.append(self._pick("post_game_no_critical"))
            return messages

        # Opening line based on solve rate
        solve_rate = found_count / critical_count if critical_count > 0 else 0

        if solve_rate >= 0.6:
            header = self._pick_formatted(
                "post_game_high_solve",
                n_critical=critical_count,
                n_found=found_count,
            )
        elif solve_rate >= 0.3:
            header = self._pick_formatted(
                "post_game_mid_solve",
                n_critical=critical_count,
                n_found=found_count,
            )
        else:
            header = self._pick_formatted(
                "post_game_low_solve",
                n_critical=critical_count,
                n_found=found_count,
            )

        messages.append(header)

        # Per-missed-moment breakdowns (each as a separate message)
        for idx, opp in critical_moments:
            if opp.found_critical:
                continue  # Only break down misses

            sf_move = game_state.move_results[idx] if idx < len(game_state.move_results) else None
            if sf_move:
                detail = self._pick_formatted(
                    "post_game_miss_detail",
                    move_number=sf_move.move_number,
                    sf_move=sf_move.move_san,
                    opp_move=opp.move_san,
                    best_response=opp.best_response_san or "the top move",
                )
                if detail:
                    messages.append(detail)

        return messages

    def handle_incoming_chat(self, message: str):
        """Process incoming chat from the opponent. Toggles quiet mode."""
        lower = message.strip().lower()
        if lower == "quiet":
            self.quiet = True
        elif lower == "talk":
            self.quiet = False

    def _pick(self, category: str) -> str:
        """Pick a random template from the category."""
        variants = self.templates.get(category, [])
        if not variants:
            return ""
        return random.choice(variants)

    def _pick_formatted(self, category: str, **kwargs) -> str:
        """Pick a random template and format it with kwargs."""
        template = self._pick(category)
        if not template:
            return ""
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template

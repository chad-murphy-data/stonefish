"""
Stonefish Chat Engine -- Message Generation
=============================================
Generates in-game chat messages and post-game summaries from per-move data.
Templates are loaded from chat_templates.json for easy editing.

Triggers:
    1. Game start (onboarding) -- always sends
    2. Stonefish plays puzzle-creating move (pre-alert) -- skipped in quiet mode
    3. Opponent finds puzzle -- skipped in quiet mode
    4. Opponent misses puzzle -- skipped in quiet mode
    5. Mate retry hint -- skipped in quiet mode
    6. Game end (post-game summary) -- always sends
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
        """Trigger 2: Pre-puzzle alert after Stonefish plays a puzzle-creating move.

        Returns None if no puzzle or quiet mode is on.
        Selects template based on puzzle type.
        """
        if self.quiet or not self.config.chat_enabled:
            return None

        if not move_result.puzzle_found:
            return None

        ptype = move_result.puzzle_type
        pct = int(move_result.puzzle_floor_prob * 100)

        if ptype == "mate":
            return self._pick_formatted("mate_puzzle", pct=pct)
        elif ptype == "positive":
            return self._pick_formatted("positive_puzzle", pct=pct)
        elif ptype == "negative":
            return self._pick_formatted("negative_puzzle", pct=pct)

        return None

    def on_opponent_move(self, opp_result: OpponentMoveResult) -> Optional[str]:
        """Triggers 3 & 4: Response after opponent plays on a puzzle position."""
        if self.quiet or not self.config.chat_enabled:
            return None

        if not opp_result.was_puzzle:
            return None

        ptype = opp_result.puzzle_type
        pct = int(opp_result.puzzle_floor_prob * 100)

        if opp_result.found_puzzle:
            if ptype == "mate":
                return self._pick_formatted("found_mate", pct=pct)
            else:
                return self._pick_formatted("found_puzzle", pct=pct)
        else:
            if ptype == "mate":
                return self._pick("mate_escaped")
            elif ptype == "positive":
                return self._pick("missed_positive")
            else:
                return self._pick("missed_puzzle")

    def on_mate_retry(self) -> Optional[str]:
        """Trigger 5: Hint during mate retry mode."""
        if self.quiet or not self.config.chat_enabled:
            return None
        return self._pick("mate_retry")

    def generate_post_game_summary(self, game_state: StonefishGameState) -> List[str]:
        """Trigger 6: Generate post-game summary as a list of separate messages.

        Always sends regardless of quiet mode.
        """
        puzzle_moments = [
            (i, opp) for i, opp in enumerate(game_state.opponent_results)
            if opp.was_puzzle
        ]
        found_count = sum(1 for _, opp in puzzle_moments if opp.found_puzzle)
        puzzle_count = len(puzzle_moments)

        messages: List[str] = []

        if puzzle_count == 0:
            messages.append(self._pick("post_game_no_puzzles"))
            return messages

        solve_rate = found_count / puzzle_count if puzzle_count > 0 else 0

        if solve_rate >= 0.6:
            rating_msg = self._pick("post_game_rating_high")
        elif solve_rate >= 0.3:
            rating_msg = self._pick("post_game_rating_mid")
        else:
            rating_msg = self._pick("post_game_rating_low")

        header = self._pick_formatted(
            "post_game",
            n_found=found_count,
            n_total=puzzle_count,
            rating_msg=rating_msg,
        )
        messages.append(header)

        # Per-missed-puzzle breakdowns
        for idx, opp in puzzle_moments:
            if opp.found_puzzle:
                continue

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

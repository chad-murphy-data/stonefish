"""
Stonefish Logger -- Per-Move and Per-Game JSON Logging
======================================================
Records every detail of every game for QA, tuning, and analysis.
One JSON file per game: stonefish_game_{timestamp}_{opponent}.json
"""

import json
import os
from datetime import datetime
from typing import Optional, List

import chess

from .config import StonefishConfig
from .game_state import MoveResult, OpponentMoveResult


class StonefishLogger:
    """Logs per-move data and writes per-game JSON files.

    Usage:
        logger = StonefishLogger(output_dir="game_logs")
        logger.start_game(config, chess.WHITE, "maia5")

        # After each Stonefish move:
        logger.record_our_move(move_result)

        # After each opponent move:
        logger.record_opponent_move(opp_result)

        # At game end:
        filepath = logger.end_game("1-0", board)
    """

    def __init__(self, output_dir: str = "game_logs"):
        self.output_dir = output_dir
        self._game_data: Optional[dict] = None
        self._config: Optional[StonefishConfig] = None

    def start_game(
        self,
        config: StonefishConfig,
        our_color: chess.Color,
        opponent_name: str,
    ):
        """Initialize a new game log."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._config = config
        self._game_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "opponent": opponent_name,
                "our_color": "white" if our_color == chess.WHITE else "black",
                "base_depth": config.base_depth,
                "deep_depth": config.deep_depth,
                "target_band": config.target_band,
                "gap_threshold": config.gap_threshold,
                "gap_threshold_low": config.gap_threshold_low,
                "max_eval_cost": config.max_eval_cost,
                "depth_boost_on_miss": config.depth_boost_on_miss,
                "boost_duration_moves": config.boost_duration_moves,
                "losing_eval_threshold": config.losing_eval_threshold,
                "losing_max_eval_cost": config.losing_max_eval_cost,
                "desperate_eval_threshold": config.desperate_eval_threshold,
                "desperate_max_eval_cost": config.desperate_max_eval_cost,
                "critical_cooldown_moves": config.critical_cooldown_moves,
                "shallow_comparison_depth": config.shallow_comparison_depth,
                "disagreement_threshold": config.disagreement_threshold,
                "disagreement_bonus_multiplier": config.disagreement_bonus_multiplier,
                "emergency_clock_seconds": config.emergency_clock_seconds,
                "emergency_depth": config.emergency_depth,
            },
            "moves": [],
            "summary": None,
        }

    def record_our_move(self, result: MoveResult):
        """Record a Stonefish move with all analysis data."""
        if self._game_data is None:
            return

        entry = {
            "move_number": result.move_number,
            "side": "stonefish",
            "stonefish_move": result.move_san,
            "stonefish_move_uci": result.move.uci(),
            "stonefish_move_rank": result.move_rank,
            "top_engine_move": result.top_engine_move,
            "deep_eval": result.deep_eval,
            "shallow_eval": result.shallow_eval,
            "candidate_evals": result.candidate_evals,
            "nettlesomeness_score": result.nettlesomeness_score,
            "gap": result.gap,
            "disagreement": result.disagreement,
            "was_flagged": result.was_flagged_critical,
            "current_depth": result.current_shallow_depth,
            "deep_depth": result.deep_depth,
            "emergency_mode": result.emergency_mode,
            "search_depth_found": result.search_depth_found,
            "search_threshold_used": result.search_threshold_used,
        }
        self._game_data["moves"].append(entry)

    def record_opponent_move(self, opp_result: OpponentMoveResult):
        """Record the opponent's move analysis."""
        if self._game_data is None:
            return

        entry = {
            "side": "opponent",
            "opponent_move": opp_result.move_san,
            "opponent_move_uci": opp_result.move.uci() if opp_result.move else "",
            "opponent_move_rank": opp_result.move_rank,
            "opponent_eval_cost": opp_result.eval_cost,
            "was_critical": opp_result.was_critical,
            "found_critical": opp_result.found_critical,
            "best_response": opp_result.best_response_san,
        }
        self._game_data["moves"].append(entry)

    def end_game(
        self,
        result_str: str,
        board: Optional[chess.Board] = None,
    ) -> Optional[str]:
        """Compute summary, write JSON file, return filepath."""
        if self._game_data is None:
            return None

        # Collect per-side move data
        sf_moves = [m for m in self._game_data["moves"] if m.get("side") == "stonefish"]
        opp_moves = [m for m in self._game_data["moves"] if m.get("side") == "opponent"]

        # Critical moment stats
        critical_moments = [m for m in opp_moves if m.get("was_critical")]
        critical_found = [m for m in critical_moments if m.get("found_critical")]
        total_eval_cost = sum(
            m.get("opponent_eval_cost", 0)
            for m in critical_moments
            if not m.get("found_critical")
        )

        # Rank distribution
        rank_dist = {}
        for m in sf_moves:
            r = m["stonefish_move_rank"]
            rank_dist[str(r)] = rank_dist.get(str(r), 0) + 1

        # Trajectories
        eval_trajectory = [m["deep_eval"] for m in sf_moves]
        depth_trajectory = [m["current_depth"] for m in sf_moves]

        self._game_data["summary"] = {
            "result": result_str,
            "total_moves": len(sf_moves),
            "critical_moments_created": len(critical_moments),
            "critical_moments_found": len(critical_found),
            "solve_rate": (
                len(critical_found) / len(critical_moments)
                if critical_moments else 0.0
            ),
            "total_eval_cost_of_misses": round(total_eval_cost, 3),
            "stonefish_rank_distribution": rank_dist,
            "eval_trajectory": eval_trajectory,
            "depth_trajectory": depth_trajectory,
        }

        # Add final FEN if board provided
        if board is not None:
            self._game_data["summary"]["final_fen"] = board.fen()

        # Write file
        opponent = self._game_data["metadata"]["opponent"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stonefish_game_{ts}_{opponent}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self._game_data, f, indent=2, default=str)

        self._game_data = None
        self._config = None
        return filepath

"""
Stonefish Logger -- Per-Move and Per-Game JSON Logging
======================================================
Records every detail of every game for QA, tuning, and analysis.
One JSON file per game: stonefish_game_{timestamp}_{opponent}.json

Updated for the three-tier Maia puzzle detection system.
"""

import json
import os
from datetime import datetime
from typing import Optional, List

import chess

from .config import StonefishConfig
from .game_state import MoveResult, OpponentMoveResult


class StonefishLogger:
    """Logs per-move data and writes per-game JSON files."""

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
                "floor_rating": config.floor_rating,
                "stretch_rating": config.stretch_rating,
                "reach_rating": config.reach_rating,
                "puzzle_eval_threshold": config.puzzle_eval_threshold,
                "soft_puzzle_eval_threshold": config.soft_puzzle_eval_threshold,
                "soft_puzzle_move_threshold": config.soft_puzzle_move_threshold,
                "max_mate_depth": config.max_mate_depth,
                "puzzle_generosity": config.puzzle_generosity,
                "enable_positive_puzzles": config.enable_positive_puzzles,
                "max_lookahead": config.max_lookahead,
                "max_eval_cost": config.max_eval_cost,
                "base_depth": config.base_depth,
                "deep_depth": config.deep_depth,
                "conversion_moves": config.conversion_moves,
                "opening_book_moves": config.opening_book_moves,
                "emergency_clock_seconds": config.emergency_clock_seconds,
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
            "candidate_evals": result.candidate_evals,
            "emergency_mode": result.emergency_mode,
            # Puzzle data
            "puzzle_found": result.puzzle_found,
            "puzzle_type": result.puzzle_type,
            "puzzle_score": result.puzzle_score,
            "puzzle_eval_gap": result.puzzle_eval_gap,
            "puzzle_floor_prob": result.puzzle_floor_prob,
            "puzzle_disagreement": result.puzzle_disagreement,
            "puzzle_search_depth": result.puzzle_search_depth,
            "puzzle_better_move": result.puzzle_better_move_san,
            "puzzle_worse_move": result.puzzle_worse_move_san,
            "puzzle_is_mate": result.puzzle_is_mate,
            "puzzle_mate_distance": result.puzzle_mate_distance,
            # Screening stats
            "positions_screened": result.positions_screened,
            "disagreements_found": result.disagreements_found,
            "puzzles_after_validation": result.puzzles_after_validation,
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
            "was_puzzle": opp_result.was_puzzle,
            "found_puzzle": opp_result.found_puzzle,
            "best_response": opp_result.best_response_san,
            "puzzle_type": opp_result.puzzle_type,
            "puzzle_floor_prob": opp_result.puzzle_floor_prob,
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

        sf_moves = [m for m in self._game_data["moves"] if m.get("side") == "stonefish"]
        opp_moves = [m for m in self._game_data["moves"] if m.get("side") == "opponent"]

        # Puzzle stats
        puzzle_moments = [m for m in opp_moves if m.get("was_puzzle")]
        puzzles_found = [m for m in puzzle_moments if m.get("found_puzzle")]
        total_eval_cost = sum(
            m.get("opponent_eval_cost", 0)
            for m in puzzle_moments
            if not m.get("found_puzzle")
        )

        # Puzzle type breakdown
        puzzle_types = {}
        for m in sf_moves:
            if m.get("puzzle_found"):
                ptype = m.get("puzzle_type", "unknown")
                puzzle_types[ptype] = puzzle_types.get(ptype, 0) + 1

        # Rank distribution
        rank_dist = {}
        for m in sf_moves:
            r = m["stonefish_move_rank"]
            rank_dist[str(r)] = rank_dist.get(str(r), 0) + 1

        # Eval trajectory
        eval_trajectory = [m["deep_eval"] for m in sf_moves]

        # Mate puzzle stats
        mate_puzzles = sum(1 for m in sf_moves if m.get("puzzle_is_mate"))

        self._game_data["summary"] = {
            "result": result_str,
            "total_moves": len(sf_moves),
            "puzzles_created": len(puzzle_moments),
            "puzzles_solved": len(puzzles_found),
            "solve_rate": (
                len(puzzles_found) / len(puzzle_moments)
                if puzzle_moments else 0.0
            ),
            "total_eval_cost_of_misses": round(total_eval_cost, 3),
            "puzzle_type_breakdown": puzzle_types,
            "mate_puzzles": mate_puzzles,
            "stonefish_rank_distribution": rank_dist,
            "eval_trajectory": eval_trajectory,
        }

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

"""
Stonefish Maia Interface -- Probability-Distribution Move Predictions
=====================================================================
Wraps Maia model calls to return full probability distributions over moves.

Each tier call (Floor/Stretch/Reach) returns a MaiaPrediction with:
    - top_move: the move Maia thinks is most likely
    - distribution: dict mapping move -> probability
    - prob_for(move): quick lookup for a specific move's probability

When a tier's rating is None, Stockfish's best move is used instead
(with probability 1.0).

The actual Maia model integration depends on the deployment environment.
This module provides the interface and a Stockfish-based fallback.
"""

import chess
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


@dataclass
class MaiaPrediction:
    """Prediction from a single Maia tier."""
    top_move: chess.Move
    distribution: Dict[chess.Move, float] = field(default_factory=dict)
    rating: Optional[int] = None          # None means Stockfish was used

    def prob_for(self, move: chess.Move) -> float:
        """Return the probability of a specific move."""
        return self.distribution.get(move, 0.0)

    def top_n(self, n: int) -> List[chess.Move]:
        """Return the top N moves by probability."""
        sorted_moves = sorted(self.distribution.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in sorted_moves[:n]]


@dataclass
class ThreeTierPrediction:
    """Combined predictions from all three tiers."""
    floor: MaiaPrediction
    stretch: MaiaPrediction
    reach: MaiaPrediction


class MaiaEngine:
    """Interface for Maia model predictions.

    In production, this wraps actual Maia ONNX/TF models.
    For now, it uses Stockfish MultiPV at different depths to simulate
    rating-dependent play, or loads Maia models if available.

    The key contract: predict(board, rating) returns a MaiaPrediction
    with a full probability distribution over legal moves.
    """

    def __init__(self, stockfish_engine=None, maia_models=None,
                 max_simulated_depth=None, multi_pv=5):
        """
        Args:
            stockfish_engine: chess.engine.SimpleEngine for fallback/high-tier
            maia_models: Optional dict mapping rating -> loaded model
            max_simulated_depth: Cap on Stockfish depth for simulated Maia calls.
                None = no cap (production). Set to 5-6 for fast live viewing.
            multi_pv: Number of MultiPV lines for simulated predictions.
                5 = fast (viewer), 8 = richer distributions (Lichess).
        """
        self.engine = stockfish_engine
        self.maia_models = maia_models or {}
        self.max_simulated_depth = max_simulated_depth
        self.multi_pv = multi_pv

    def predict(self, board: chess.Board, rating: Optional[int]) -> MaiaPrediction:
        """Get a move prediction for a position at a given rating level.

        If rating is None, uses Stockfish's best move.
        If a Maia model is available for this rating, uses it.
        Otherwise, simulates via Stockfish depth scaling.
        """
        if rating is None:
            return self._predict_stockfish(board)

        if rating in self.maia_models:
            return self._predict_maia_model(board, rating)

        # Find nearest available Maia model
        if self.maia_models:
            nearest = min(self.maia_models.keys(), key=lambda r: abs(r - rating))
            if abs(nearest - rating) <= 200:
                return self._predict_maia_model(board, nearest)

        # Fallback: simulate with Stockfish depth scaling
        return self._predict_stockfish_simulated(board, rating)

    def predict_three_tier(
        self, board: chess.Board,
        floor_rating: int,
        stretch_rating: Optional[int],
        reach_rating: Optional[int],
    ) -> ThreeTierPrediction:
        """Get predictions from all three tiers for a position."""
        floor = self.predict(board, floor_rating)
        stretch = self.predict(board, stretch_rating)
        reach = self.predict(board, reach_rating)
        return ThreeTierPrediction(floor=floor, stretch=stretch, reach=reach)

    def _predict_stockfish(self, board: chess.Board) -> MaiaPrediction:
        """Use Stockfish's best move with probability 1.0."""
        if self.engine is None:
            raise RuntimeError("No Stockfish engine available")

        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine import get_top_moves

        # Terminal position guard
        if board.is_game_over():
            dummy_move = chess.Move.null()
            return MaiaPrediction(
                top_move=dummy_move,
                distribution={dummy_move: 1.0},
                rating=None,
            )

        sf_depth = 12
        if self.max_simulated_depth is not None:
            sf_depth = min(sf_depth, self.max_simulated_depth)
        top = get_top_moves(self.engine, board, num_moves=1, depth=sf_depth)
        if not top:
            # Fallback to any legal move
            move = list(board.legal_moves)[0]
            return MaiaPrediction(
                top_move=move,
                distribution={move: 1.0},
                rating=None,
            )

        move = top[0][0]
        return MaiaPrediction(
            top_move=move,
            distribution={move: 1.0},
            rating=None,
        )

    def _predict_maia_model(self, board: chess.Board, rating: int) -> MaiaPrediction:
        """Use an actual Maia model to get probability distribution.

        Maia models output a softmax distribution over all legal moves.
        This is the production path when models are loaded.
        """
        model = self.maia_models[rating]

        # The actual Maia model call depends on the model format.
        # Expected interface: model.predict(board) -> dict[move_uci, float]
        raw_probs = model.predict(board)

        distribution = {}
        for uci_str, prob in raw_probs.items():
            try:
                move = chess.Move.from_uci(uci_str)
                if move in board.legal_moves:
                    distribution[move] = prob
            except ValueError:
                continue

        if not distribution:
            # Model produced no valid moves -- fallback
            move = list(board.legal_moves)[0]
            return MaiaPrediction(
                top_move=move,
                distribution={move: 1.0},
                rating=rating,
            )

        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {m: p / total for m, p in distribution.items()}

        top_move = max(distribution, key=distribution.get)
        return MaiaPrediction(
            top_move=top_move,
            distribution=distribution,
            rating=rating,
        )

    @staticmethod
    def _flip_turn(board: chess.Board) -> chess.Board:
        """Return a copy of *board* with the side-to-move flipped via FEN surgery.

        This produces a legal-looking board from python-chess's perspective so
        that Stockfish's bestmove reply passes the UCI move-validation layer.
        Castling rights and en-passant are cleared (irrelevant for the shallow
        follow-up eval we need).
        """
        parts = board.fen().split()
        parts[1] = "w" if parts[1] == "b" else "b"
        parts[2] = "-"   # castling — avoid illegal-state complaints
        parts[3] = "-"   # en passant
        return chess.Board(" ".join(parts))

    def predict_solitaire(self, board: chess.Board, rating: int) -> MaiaPrediction:
        """Pick the move that sets up the best follow-up, assuming the opponent passes.

        Replicates solitaire chess thinking: "I go here, then I go here."
        Used to emulate ~500 ELO play patterns where opponent agency is ignored.

        The trick: after pushing a candidate move, flip side_to_move back to the
        original player before asking Maia what to do next. This mechanically
        reproduces Type 4 magical thinking — planning based on your own next move,
        not the opponent's reply.
        """
        if self.engine is None:
            raise RuntimeError("No Stockfish engine available")

        # Get top candidate moves for the current position
        first_pred = self.predict(board, rating)
        candidates = first_pred.top_n(5)

        if not candidates:
            return first_pred

        best_move = candidates[0]
        best_followup_score = -999.0

        for move in candidates:
            if move not in board.legal_moves:
                continue

            sim = board.copy()
            sim.push(move)

            # THE TRICK: don't hand the turn to the opponent — rebuild the
            # board with our colour to move so python-chess + Stockfish agree.
            sim = self._flip_turn(sim)

            # Ask Maia what WE would do next (opponent skipped)
            followup_pred = self.predict(sim, rating)
            followup_move = followup_pred.top_move

            if followup_move not in sim.legal_moves:
                continue

            # Score the position we'd reach after our imagined two-move sequence
            sim.push(followup_move)

            if sim.is_game_over():
                # Checkmate or stalemate — assign extreme scores
                score = 10000.0 if sim.is_checkmate() else 0.0
            else:
                try:
                    info = self.engine.analyse(sim, chess.engine.Limit(depth=6))
                    raw = info["score"].relative.score(mate_score=10000)
                    score = float(raw) if raw is not None else 0.0
                except Exception:
                    score = 0.0

            if score > best_followup_score:
                best_followup_score = score
                best_move = move

        return MaiaPrediction(
            top_move=best_move,
            distribution={best_move: 1.0},
            rating=rating,
        )

    def _predict_stockfish_simulated(self, board: chess.Board, rating: int) -> MaiaPrediction:
        """Simulate Maia-like predictions using Stockfish at scaled depth.

        Maps rating to Stockfish depth and uses MultiPV to build
        a rough probability distribution. This is an approximation --
        real Maia models trained on human games are much better at
        predicting actual human moves.

        Rating -> depth mapping (rough):
            500  -> depth 2
            750  -> depth 3
            1000 -> depth 4
            1250 -> depth 6
            1500 -> depth 8
            1750 -> depth 10
            2000 -> depth 12
        """
        if self.engine is None:
            raise RuntimeError("No Stockfish engine available")

        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine import get_top_moves

        # Map rating to simulated depth
        depth = max(2, min(12, 2 + (rating - 500) // 150))
        if self.max_simulated_depth is not None:
            depth = min(depth, self.max_simulated_depth)

        # MultiPV count: use self.multi_pv but cap to legal moves, min 1
        legal_count = len(list(board.legal_moves))
        if legal_count == 0:
            # Terminal position (checkmate/stalemate) -- return a dummy prediction
            # This can happen when puzzle detection pushes a move that ends the game
            dummy_move = chess.Move.null()
            return MaiaPrediction(
                top_move=dummy_move,
                distribution={dummy_move: 1.0},
                rating=rating,
            )
        num_moves = max(1, min(self.multi_pv, legal_count))
        top = get_top_moves(self.engine, board, num_moves=num_moves, depth=depth)

        if not top:
            move = list(board.legal_moves)[0]
            return MaiaPrediction(
                top_move=move,
                distribution={move: 1.0},
                rating=rating,
            )

        # Build a softmax-like distribution from eval differences
        # Better moves get higher probability, scaled by rating
        # Lower-rated players have more uniform distributions
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        evals = [(m, e * sign) for m, e in top]
        best_eval = evals[0][1]

        # Temperature: lower rating = higher temperature (more random)
        temperature = max(0.5, 3.0 - (rating / 800.0))

        import math
        raw_weights = {}
        for move, ev in evals:
            diff = best_eval - ev  # Positive = this move is worse
            raw_weights[move] = math.exp(-diff / temperature)

        total = sum(raw_weights.values())
        distribution = {m: w / total for m, w in raw_weights.items()}

        top_move = max(distribution, key=distribution.get)
        return MaiaPrediction(
            top_move=top_move,
            distribution=distribution,
            rating=rating,
        )

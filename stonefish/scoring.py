"""
Stonefish Scoring -- Unified Three-Tier Maia Puzzle Detector
=============================================================
Replaces the old Level 1/2/3 + divergence + positive puzzle system
with a single unified detector based on three-tier Maia disagreement.

Core idea: For a given opponent rating, run three Maia models:
    - Floor (opponent's rating): what they'd actually play
    - Stretch (+300-400): what a better version would play
    - Reach (+600-800): what a solidly stronger player would play

A puzzle exists when these tiers disagree AND Stockfish confirms
the disagreement matters (eval gap >= threshold).

Puzzle types (all unified under one detector):
    - Negative: Floor plays a losing move, Stretch/Reach finds the defense
    - Positive: Opponent could win something but Floor wouldn't find it
    - Mate: Stockfish shows forced mate, always a puzzle regardless of Maia
"""

import chess
import chess.engine
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import get_top_moves

from .config import StonefishConfig
from .maia import MaiaEngine, MaiaPrediction, ThreeTierPrediction


# ---------------------------------------------------------------------------
# Helper: smart net material difference filter
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def _count_material(board: chess.Board) -> float:
    """Count material balance from White's perspective."""
    total = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        if piece.color == chess.WHITE:
            total += value
        else:
            total -= value
    return total


# ---------------------------------------------------------------------------
# Helper: is_genuinely_hung — smart hung-piece detection
# ---------------------------------------------------------------------------

HUNG_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

HUNG_PIECE_FLOOR = 0.8   # Minimum net gain to count as genuinely hung
HUNG_PIECE_RATIO = 0.5   # Fraction of piece value required as net gain


def is_genuinely_hung(
    board: chess.Board,
    square: chess.Square,
    engine,
    trade_depth: int = 4,
    sf_depth: int = 6,
) -> bool:
    """Returns True only if a piece on `square` is genuinely free to take —
    i.e., capturing it and surviving the recapture sequence nets the
    capturing side at least max(HUNG_PIECE_FLOOR, piece_value * HUNG_PIECE_RATIO) pawns.

    Filters out:
    - Pieces that are protected (capture loses material after trades)
    - Pieces where the capture is a losing exchange
    - Pawns where a reasonable non-capture is nearly as good (GM positional ignore)

    Args:
        board:       Position to analyse (the side to move is the potential capturer)
        square:      Square of the potentially hung piece
        engine:      Stockfish engine instance
        trade_depth: Depth to resolve trade sequences (4 is enough for most exchanges)
        sf_depth:    Depth for the pawn-ignore non-capture baseline check
    """
    piece = board.piece_at(square)
    if piece is None:
        return False

    # Only pieces belonging to the opponent can be hung
    if piece.color == board.turn:
        return False

    captures = [m for m in board.legal_moves if m.to_square == square]
    if not captures:
        return False

    piece_value = HUNG_PIECE_VALUES.get(piece.piece_type, 1.0)
    min_gain = max(HUNG_PIECE_FLOOR, piece_value * HUNG_PIECE_RATIO)

    winning_captures = []

    for capture in captures:
        sim = board.copy()
        sim.push(capture)

        if sim.is_game_over():
            # Checkmate on capture — obviously winning
            if sim.is_checkmate():
                winning_captures.append((capture, 999.0))
            continue

        # Resolve the recapture sequence: play out up to trade_depth half-moves
        try:
            info = engine.analyse(sim, chess.engine.Limit(depth=trade_depth))
            pv = info.get("pv", [])
            for response in pv[:trade_depth]:
                if response in sim.legal_moves:
                    sim.push(response)
                    if sim.is_game_over():
                        break
                else:
                    break
        except Exception:
            pass

        net = _net_material_gain(board, sim, board.turn)
        if net >= min_gain:
            winning_captures.append((capture, net))

    if not winning_captures:
        return False

    # Pawn filter: if it's an opponent pawn, check whether not capturing is
    # a reasonable choice (within 0.3 pawns of the best capture).
    # This handles the GM "ignore the pawn for initiative" case.
    if piece.piece_type == chess.PAWN:
        best_capture_net = max(winning_captures, key=lambda x: x[1])[1]
        non_capture_baseline = _best_non_capture_eval(board, engine, sf_depth)
        if non_capture_baseline > -900:  # Valid baseline found
            # Convert best_capture_net to eval-space for comparison
            # (rough: net material gain ~ eval gain from capturing side's perspective)
            if abs(best_capture_net - non_capture_baseline) < 0.3:
                return False  # Pawn is "reasonable to ignore"

    return True


def _net_material_gain(
    board_before: chess.Board,
    board_after: chess.Board,
    capturing_side: chess.Color,
) -> float:
    """Material delta from capturing_side's perspective after a sequence resolves.
    Positive = capturing side gained material.
    """
    sign = 1.0 if capturing_side == chess.WHITE else -1.0
    before = _count_material(board_before) * sign
    after = _count_material(board_after) * sign
    return after - before


def _best_non_capture_eval(
    board: chess.Board,
    engine,
    depth: int,
) -> float:
    """Stockfish eval (in pawns, from side-to-move's perspective) of the best
    non-capture move. Used as a baseline for the pawn-ignore filter.

    Returns -999.0 if no non-capture moves exist or analysis fails.
    """
    non_captures = [m for m in board.legal_moves if not board.is_capture(m)]
    if not non_captures:
        return -999.0

    sign = 1.0 if board.turn == chess.WHITE else -1.0

    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=5)
        for line in result:
            if "pv" not in line:
                continue
            top_move = line["pv"][0]
            if not board.is_capture(top_move):
                raw = line["score"].white().score(mate_score=10000)
                if raw is not None:
                    return (raw / 100.0) * sign
    except Exception:
        pass

    return -999.0


# ---------------------------------------------------------------------------
# Helper: smart net material difference filter (existing)
# ---------------------------------------------------------------------------

def net_material_difference(board: chess.Board, move_a: chess.Move, move_b: chess.Move, engine) -> float:
    """Play out each move through obvious trade sequences, compare net material.

    Instead of just checking "are these captures?", plays each move + Stockfish's
    low-depth response to resolve trade sequences, then compares material balance.
    Returns the difference in pawns (positive = move_a leads to more material for
    the side to move).

    Cost: two Stockfish depth-3 calls (~10ms each) = ~20ms per disagreement.
    Fast path: if neither move is a capture/promotion and they go to the same
    square, skip the expensive analysis (net difference is always ~0).
    """
    # Fast path: quiet moves to the same square can't differ materially
    a_cap = board.is_capture(move_a)
    b_cap = board.is_capture(move_b)
    if (not a_cap and not b_cap
            and not move_a.promotion and not move_b.promotion
            and move_a.to_square == move_b.to_square):
        return 0.0

    sign = 1.0 if board.turn == chess.WHITE else -1.0

    def material_after_resolution(board, move):
        sim = board.copy()
        sim.push(move)
        if sim.is_game_over():
            return _count_material(sim) * sign

        # Let Stockfish play out the obvious response at low depth
        # Depth 3 is enough to see Nxd5 exd5 Bxd5 type sequences
        try:
            info = engine.analyse(sim, chess.engine.Limit(depth=3))
            pv = info.get("pv", [])
            for response_move in pv[:3]:
                if response_move in sim.legal_moves:
                    sim.push(response_move)
                    if sim.is_game_over():
                        break
                else:
                    break
        except Exception:
            pass

        return _count_material(sim) * sign

    mat_a = material_after_resolution(board, move_a)
    mat_b = material_after_resolution(board, move_b)
    return mat_a - mat_b


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _eval_after_move(engine, board: chess.Board, move: chess.Move, depth: int) -> float:
    """Return the eval in pawns from the side-to-move's perspective BEFORE the move.

    Pushes the move, runs Stockfish, pops the move.
    Positive = good for the player who is about to move (before push).
    """
    sign = 1.0 if board.turn == chess.WHITE else -1.0

    board.push(move)
    if board.is_game_over():
        is_mate = board.is_checkmate()
        board.pop()
        return -100.0 * sign if is_mate else 0.0
    top = get_top_moves(engine, board, num_moves=1, depth=depth)
    board.pop()

    if not top:
        return 0.0

    # top[0][1] is from white's perspective; we want from the mover's perspective
    return top[0][1] * sign


def _get_mate_score(engine, board: chess.Board, depth: int) -> Optional[int]:
    """Check if there's a forced mate in the position.

    Returns mate distance (positive = side to move can mate) or None.
    """
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = result.get("score")
    if score is None:
        return None

    relative = score.relative
    if relative.is_mate():
        return relative.mate()
    return None


def _maia_rollout_eval(
    board: chess.Board,
    first_move: chess.Move,
    maia: 'MaiaEngine',
    engine,
    floor_rating: int,
    rollout_depth: int,
) -> float:
    """Push first_move, then play out rollout_depth half-moves with Maia
    at floor_rating for both sides. Return Stockfish's eval of the
    final position from the perspective of the side that played first_move.

    This captures how initial differences compound with human-level play.
    A 0.1 pawn gap at move 1 might become 2.0 by move 4 because the
    "worse" line generates tactics that Maia-at-floor can't navigate.

    Speed optimization: rollout Maia calls use a low rating cap (min of
    floor_rating and 1100) so Stockfish simulated depth stays <= 6.
    The final eval uses depth 8 (sufficient after rollout noise reduction).
    """
    sim_board = board.copy()
    sim_board.push(first_move)

    # If the move ends the game, return a large eval (mate)
    if sim_board.is_game_over():
        if sim_board.is_checkmate():
            return -100.0  # Side to move is mated = great for us
        return 0.0  # Stalemate/draw

    # Cap the rollout rating to keep simulated Maia calls fast (depth <= 6).
    # We just need plausible human moves, not perfectly calibrated ones.
    rollout_rating = min(floor_rating, 1100)

    for _ in range(rollout_depth):
        if sim_board.is_game_over():
            break
        maia_pred = maia.predict(sim_board, rollout_rating)
        if maia_pred.top_move in sim_board.legal_moves:
            sim_board.push(maia_pred.top_move)
        else:
            break

    # Depth 8 is sufficient — the rollout provides noise reduction already.
    try:
        info = engine.analyse(sim_board, chess.engine.Limit(depth=8))
        score = info.get("score")
        if score is None:
            return 0.0
        relative = score.relative
        if relative.is_mate():
            return relative.mate() * 100.0  # Large value for mate
        cp = relative.score(mate_score=10000)
        if cp is None:
            return 0.0
        eval_score = cp / 100.0
    except Exception:
        return 0.0

    # The eval is from the perspective of the side to move in sim_board.
    # We want it from the perspective of the side that played first_move.
    # first_move was played by board.turn (before push). After push, it's
    # the opponent's turn. After rollout_depth moves, the side alternates.
    # If rollout_depth is even, sim_board.turn == opponent of first_move player.
    # If rollout_depth is odd, sim_board.turn == first_move player.
    # Since relative eval is from sim_board.turn's perspective, we may need to flip.
    first_move_is_white = board.turn == chess.WHITE
    sim_is_white_turn = sim_board.turn == chess.WHITE
    if first_move_is_white != sim_is_white_turn:
        eval_score = -eval_score

    return eval_score


def _is_capturable_in_1(engine, board: chess.Board, move: chess.Move) -> bool:
    """After we play 'move', can the opponent win material in one move?

    Used for the NOT-capturable-in-1 filter on positive puzzles.
    The board should have the opponent to move; 'move' is the opponent's move.
    After they play it, we check if the reply wins material.
    """
    board.push(move)
    if board.is_game_over():
        board.pop()
        return False
    info = engine.analyse(board, chess.engine.Limit(depth=1))
    score = info.get("score")
    board.pop()

    if score is None:
        return False

    relative = score.relative
    if relative.is_mate():
        return True
    cp = relative.score()
    if cp is None:
        return False
    return cp / 100.0 > 0.5


# ---------------------------------------------------------------------------
# Difficulty framing
# ---------------------------------------------------------------------------

def get_difficulty_label(floor_prob: float) -> str:
    """Map floor probability to a difficulty label."""
    if floor_prob < 0.05:
        return "very hard"
    elif floor_prob < 0.15:
        return "hard"
    elif floor_prob < 0.30:
        return "moderate"
    else:
        return "findable"


# ---------------------------------------------------------------------------
# Adaptive eval threshold
# ---------------------------------------------------------------------------

def get_eval_threshold(config: StonefishConfig, move_number: int, puzzles_found: int) -> float:
    """Get the eval threshold, lowering it if puzzles are scarce."""
    base = config.puzzle_eval_threshold

    if move_number >= config.soft_puzzle_move_threshold and puzzles_found < config.puzzles_before_softening:
        return config.soft_puzzle_eval_threshold
    if move_number >= config.soft_puzzle_move_threshold + 10 and puzzles_found < config.puzzles_before_softening + 1:
        return config.soft_puzzle_eval_threshold

    return base


# ---------------------------------------------------------------------------
# Core: Three-tier puzzle detection
# ---------------------------------------------------------------------------

def detect_puzzle(
    board: chess.Board,
    maia: MaiaEngine,
    engine,
    config: StonefishConfig,
    our_move: chess.Move,
    eval_cost_to_create: float,
    move_number: int,
    puzzles_found: int,
) -> Optional['PuzzleResult']:
    """Run the three-tier Maia detector on a position (opponent to move).

    Args:
        board: Position with the opponent to move (AFTER our_move was pushed)
        maia: MaiaEngine for tier predictions
        engine: Stockfish engine
        config: Stonefish configuration
        our_move: The move Stonefish played to reach this position
        eval_cost_to_create: Eval cost of choosing our_move over Stockfish #1
        move_number: Current full move number
        puzzles_found: Puzzles found so far in this game

    Returns:
        PuzzleResult if a puzzle is detected, None otherwise.
    """
    from .game_state import PuzzleResult

    # Guard: if the position is terminal, no puzzle can exist
    if board.is_game_over():
        return None

    # Step 1: Three Maia calls
    tiers = maia.predict_three_tier(
        board,
        config.floor_rating,
        config.stretch_rating,
        config.reach_rating,
    )

    floor_move = tiers.floor.top_move
    stretch_move = tiers.stretch.top_move
    reach_move = tiers.reach.top_move

    # Step 2: Check for disagreements
    disagreements = []
    if floor_move != stretch_move:
        disagreements.append(("floor_v_stretch", floor_move, stretch_move))
    if floor_move != reach_move:
        disagreements.append(("floor_v_reach", floor_move, reach_move))
    if stretch_move != reach_move:
        disagreements.append(("stretch_v_reach", stretch_move, reach_move))

    if not disagreements:
        return None

    # Step 3: Smart material first-cut filter
    # Play out each move + opponent's best recapture (Stockfish depth 3),
    # then compare NET material balance. Only flag as material-relevant if
    # the net material after trades resolve is actually different (>= 0.3 pawns).
    threshold = get_eval_threshold(config, move_number, puzzles_found)

    filtered = []
    for dtype, worse_move, better_move in disagreements:
        net_diff = net_material_difference(board, worse_move, better_move, engine)
        if abs(net_diff) >= 0.3:
            filtered.append((dtype, worse_move, better_move))

    if not filtered:
        # No real material differences — allow non-material puzzles past the
        # adaptive threshold move
        if move_number >= config.soft_puzzle_move_threshold:
            filtered = disagreements
        else:
            return None

    # Step 4: Maia rollout validation
    # Instead of checking the immediate Stockfish eval after one move, roll out
    # each line for several moves using Maia at Floor rating for both sides,
    # THEN check Stockfish eval. This captures how initial differences compound.
    best_puzzles = []
    rollout_depth = config.rollout_depth

    for dtype, worse_move, better_move in filtered:
        worse_eval = _maia_rollout_eval(
            board, worse_move, maia, engine,
            config.floor_rating, rollout_depth,
        )
        better_eval = _maia_rollout_eval(
            board, better_move, maia, engine,
            config.floor_rating, rollout_depth,
        )
        eval_gap = worse_eval - better_eval  # Positive = worse_move costs pawns

        # Optionally average multiple rollouts for reliability
        if config.rollout_averaging > 1:
            for _ in range(config.rollout_averaging - 1):
                worse_eval += _maia_rollout_eval(
                    board, worse_move, maia, engine,
                    config.floor_rating, rollout_depth,
                )
                better_eval += _maia_rollout_eval(
                    board, better_move, maia, engine,
                    config.floor_rating, rollout_depth,
                )
            worse_eval /= config.rollout_averaging
            better_eval /= config.rollout_averaging
            eval_gap = worse_eval - better_eval

        if eval_gap < threshold:
            continue

        floor_prob = tiers.floor.prob_for(better_move)
        stretch_prob = tiers.stretch.prob_for(better_move)
        reach_prob = tiers.reach.prob_for(better_move)

        # Count tiers agreeing on the correct move
        tiers_agreeing = sum(1 for m in [floor_move, stretch_move, reach_move] if m == better_move)

        worse_san = board.san(worse_move) if worse_move in board.legal_moves else worse_move.uci()
        better_san = board.san(better_move) if better_move in board.legal_moves else better_move.uci()

        our_move_san = our_move.uci()

        score = _score_puzzle_raw(
            eval_gap=eval_gap,
            floor_prob=floor_prob,
            tiers_agreeing=tiers_agreeing,
            is_positive=False,
            is_mate=False,
            generosity=config.puzzle_generosity,
            eval_cost_to_create=eval_cost_to_create,
        )

        best_puzzles.append(PuzzleResult(
            is_puzzle=True,
            puzzle_type="negative",
            disagreement_type=dtype,
            score=score,
            our_move=our_move,
            our_move_san=our_move_san,
            worse_move=worse_move,
            worse_move_san=worse_san,
            better_move=better_move,
            better_move_san=better_san,
            eval_gap=eval_gap,
            eval_cost_to_create=eval_cost_to_create,
            floor_prob_correct=floor_prob,
            stretch_prob_correct=stretch_prob,
            reach_prob_correct=reach_prob,
            tiers_agreeing=tiers_agreeing,
            move_number=move_number,
        ))

    if not best_puzzles:
        return None

    return max(best_puzzles, key=lambda p: p.score)


def detect_positive_puzzle(
    board_before: chess.Board,
    candidate_move: chess.Move,
    maia: MaiaEngine,
    engine,
    config: StonefishConfig,
    eval_cost_to_create: float,
    move_number: int,
    puzzles_found: int,
) -> Optional['PuzzleResult']:
    """Check if a candidate move creates a positive puzzle.

    A positive puzzle is a position where the opponent could win material
    but Floor tier wouldn't find the winning continuation.
    The NOT-capturable-in-1 filter applies.

    Args:
        board_before: Position BEFORE our move (our turn)
        candidate_move: The move we're considering
    """
    from .game_state import PuzzleResult

    if not config.enable_positive_puzzles:
        return None

    board_before.push(candidate_move)

    # Guard: if the position is terminal, no puzzle can exist
    if board_before.is_game_over():
        board_before.pop()
        return None

    # Run three tiers from opponent's perspective
    tiers = maia.predict_three_tier(
        board_before,
        config.floor_rating,
        config.stretch_rating,
        config.reach_rating,
    )

    floor_move = tiers.floor.top_move
    stretch_move = tiers.stretch.top_move
    reach_move = tiers.reach.top_move

    # For a positive puzzle: stretch/reach finds a winning move, floor doesn't
    if floor_move == stretch_move and floor_move == reach_move:
        board_before.pop()
        return None

    threshold = get_eval_threshold(config, move_number, puzzles_found)
    rollout_depth = config.rollout_depth

    best_puzzle = None

    for dtype, worse_move, better_move in [
        ("floor_v_stretch", floor_move, stretch_move),
        ("floor_v_reach", floor_move, reach_move),
    ]:
        if worse_move == better_move:
            continue

        worse_eval = _maia_rollout_eval(
            board_before, worse_move, maia, engine,
            config.floor_rating, rollout_depth,
        )
        better_eval = _maia_rollout_eval(
            board_before, better_move, maia, engine,
            config.floor_rating, rollout_depth,
        )
        eval_gap = better_eval - worse_eval

        if eval_gap < threshold:
            continue

        if _is_capturable_in_1(engine, board_before, better_move):
            continue

        floor_prob = tiers.floor.prob_for(better_move)
        stretch_prob = tiers.stretch.prob_for(better_move)
        reach_prob = tiers.reach.prob_for(better_move)

        tiers_agreeing = sum(1 for m in [floor_move, stretch_move, reach_move] if m == better_move)

        worse_san = board_before.san(worse_move) if worse_move in board_before.legal_moves else worse_move.uci()
        better_san = board_before.san(better_move) if better_move in board_before.legal_moves else better_move.uci()

        score = _score_puzzle_raw(
            eval_gap=eval_gap,
            floor_prob=floor_prob,
            tiers_agreeing=tiers_agreeing,
            is_positive=True,
            is_mate=False,
            generosity=config.puzzle_generosity,
            eval_cost_to_create=eval_cost_to_create,
        )

        puzzle = PuzzleResult(
            is_puzzle=True,
            puzzle_type="positive",
            disagreement_type=dtype,
            score=score,
            our_move=candidate_move,
            our_move_san=candidate_move.uci(),
            worse_move=worse_move,
            worse_move_san=worse_san,
            better_move=better_move,
            better_move_san=better_san,
            eval_gap=eval_gap,
            eval_cost_to_create=eval_cost_to_create,
            floor_prob_correct=floor_prob,
            stretch_prob_correct=stretch_prob,
            reach_prob_correct=reach_prob,
            tiers_agreeing=tiers_agreeing,
            move_number=move_number,
        )

        if best_puzzle is None or puzzle.score > best_puzzle.score:
            best_puzzle = puzzle

    board_before.pop()
    return best_puzzle


def detect_mate_puzzle(
    board: chess.Board,
    engine,
    config: StonefishConfig,
    our_move: chess.Move,
    eval_cost_to_create: float,
    move_number: int,
) -> Optional['PuzzleResult']:
    """Check if the position after our_move has a forced mate for the opponent.

    Mate puzzles are always flagged regardless of Maia predictions.
    """
    from .game_state import PuzzleResult

    board.push(our_move)

    # Guard: if the position is terminal (checkmate/stalemate), skip
    if board.is_game_over():
        board.pop()
        return None

    mate_dist = _get_mate_score(engine, board, depth=config.deep_depth)

    if mate_dist is None or mate_dist <= 0 or mate_dist > config.max_mate_depth:
        board.pop()
        return None

    top = get_top_moves(engine, board, num_moves=1, depth=config.deep_depth)

    if not top:
        board.pop()
        return None

    best_mate_move = top[0][0]
    mate_san = board.san(best_mate_move) if best_mate_move in board.legal_moves else best_mate_move.uci()

    board.pop()

    score = 5.0 + (config.max_mate_depth - mate_dist)

    return PuzzleResult(
        is_puzzle=True,
        puzzle_type="mate",
        disagreement_type="mate",
        score=score,
        our_move=our_move,
        our_move_san=our_move.uci(),
        worse_move=best_mate_move,
        worse_move_san="",
        better_move=best_mate_move,
        better_move_san=mate_san,
        eval_gap=10.0,
        eval_cost_to_create=eval_cost_to_create,
        floor_prob_correct=0.0,
        stretch_prob_correct=0.0,
        reach_prob_correct=0.0,
        tiers_agreeing=0,
        is_mate=True,
        mate_distance=mate_dist,
        move_number=move_number,
    )


# ---------------------------------------------------------------------------
# Mate-preserving move
# ---------------------------------------------------------------------------

def find_mate_preserving_move(
    board: chess.Board,
    engine,
    current_mate_distance: int,
) -> Optional[chess.Move]:
    """Find a move that keeps the opponent's mate alive without making it trivial.

    Board should be at Stonefish's turn. We need to find a move that
    preserves the opponent's forced mate against us.
    """
    candidates = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_game_over():
            board.pop()
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=current_mate_distance + 2))
        score = info.get("score")
        if score is not None:
            relative = score.relative
            if relative.is_mate() and relative.mate() > 0:
                new_distance = relative.mate()
                if new_distance > 1:  # Never allow mate in 1
                    candidates.append((move, new_distance))
        board.pop()

    if not candidates:
        return None

    candidates.sort(key=lambda x: abs(x[1] - current_mate_distance))
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Unified puzzle scoring
# ---------------------------------------------------------------------------

def _score_puzzle_raw(
    eval_gap: float,
    floor_prob: float,
    tiers_agreeing: int,
    is_positive: bool,
    is_mate: bool,
    generosity: float,
    eval_cost_to_create: float,
) -> float:
    """Score a puzzle candidate. Higher = better puzzle."""
    base_score = eval_gap

    if is_mate:
        base_score += 5.0

    if 0.05 <= floor_prob <= 0.20:
        base_score += 0.5

    if is_positive:
        base_score *= generosity

    base_score -= eval_cost_to_create

    if tiers_agreeing >= 2:
        base_score += 0.3

    return base_score


# ---------------------------------------------------------------------------
# Opponent move ranking
# ---------------------------------------------------------------------------

def rank_opponent_move(
    engine,
    board: chess.Board,
    opponent_move: chess.Move,
    target_band: int,
    depth: int,
) -> Tuple[int, float, str]:
    """Determine the rank, eval cost, and best move for the opponent's actual move.

    Returns:
        (rank, eval_cost, best_move_san)
    """
    num_check = max(2 * target_band, 6)
    top_moves = get_top_moves(engine, board, num_moves=num_check, depth=depth)

    if not top_moves:
        return (0, 0.0, "")

    opp_color = board.turn
    sign = 1.0 if opp_color == chess.WHITE else -1.0

    best_eval = top_moves[0][1] * sign
    try:
        best_move_san = board.san(top_moves[0][0])
    except Exception:
        best_move_san = top_moves[0][0].uci()

    for i, (move, eval_white) in enumerate(top_moves):
        if move == opponent_move:
            eval_for_opp = eval_white * sign
            eval_cost = best_eval - eval_for_opp
            return (i, max(0.0, eval_cost), best_move_san)

    worst_eval = top_moves[-1][1] * sign
    return (num_check, max(0.0, best_eval - worst_eval), best_move_san)


# ---------------------------------------------------------------------------
# Main entry point: select_stonefish_move
# ---------------------------------------------------------------------------

def select_stonefish_move(
    engine,
    board: chess.Board,
    config: StonefishConfig,
    current_shallow_depth: int,
    move_number: int,
    our_clock_seconds: float = None,
    maia: MaiaEngine = None,
    game_state=None,
):
    """Select Stonefish's move using the unified puzzle detection system.

    Move selection flow:
        1. Opening book -- first N moves play Stockfish #1
        2. Conversion mode -- after opponent misses puzzle, Stockfish #1 for K moves
        3. Mate retry mode -- play mate-preserving move if active
        4. Capture priority -- free captures worth > 0.5 pawns
        5. Puzzle search (negative, positive, mate at depth 0/1/2)
        6. Fallback -- Stockfish's #1 move
    """
    from .game_state import MoveResult
    import random as _random

    our_color = board.turn
    sign = 1.0 if our_color == chess.WHITE else -1.0

    puzzles_found = game_state.puzzles_found_so_far if game_state else 0

    if maia is None:
        maia = MaiaEngine(stockfish_engine=engine)

    # === EMERGENCY MODE ===
    emergency = (our_clock_seconds is not None
                 and our_clock_seconds < config.emergency_clock_seconds)

    if emergency:
        emg_candidates = get_top_moves(engine, board, num_moves=3, depth=config.emergency_depth)
        if not emg_candidates:
            fallback = _random.choice(list(board.legal_moves))
            emg_candidates = [(fallback, 0.0)]
        move, ev = emg_candidates[0]
        return MoveResult(
            move=move,
            move_san=board.san(move),
            move_rank=0,
            top_engine_move=board.san(move),
            deep_eval=round(ev * sign, 3),
            move_number=move_number,
            emergency_mode=True,
            candidate_evals=[(board.san(m), round(e, 3)) for m, e in emg_candidates],
            current_shallow_depth=config.emergency_depth,
            deep_depth=0,
        )

    # === CANDIDATE GENERATION ===
    candidates = get_top_moves(engine, board, num_moves=config.num_candidates,
                               depth=current_shallow_depth)

    if not candidates:
        fallback = _random.choice(list(board.legal_moves))
        return MoveResult(
            move=fallback,
            move_san=board.san(fallback),
            move_rank=0,
            top_engine_move=board.san(fallback),
            deep_eval=0.0,
            move_number=move_number,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
        )

    best_eval_for_us = candidates[0][1] * sign
    top_engine_move_san = board.san(candidates[0][0])
    candidate_evals_log = [(board.san(m), round(e, 3)) for m, e in candidates]

    # === 1. OPENING BOOK ===
    if move_number <= config.opening_book_moves:
        move = candidates[0][0]
        return MoveResult(
            move=move,
            move_san=board.san(move),
            move_rank=0,
            top_engine_move=top_engine_move_san,
            deep_eval=round(best_eval_for_us, 3),
            move_number=move_number,
            candidate_evals=candidate_evals_log,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
        )

    # === 2. CONVERSION MODE ===
    if game_state and game_state.conversion_moves_remaining > 0:
        move = candidates[0][0]
        return MoveResult(
            move=move,
            move_san=board.san(move),
            move_rank=0,
            top_engine_move=top_engine_move_san,
            deep_eval=round(best_eval_for_us, 3),
            move_number=move_number,
            candidate_evals=candidate_evals_log,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
        )

    # === 3. MATE RETRY MODE ===
    if game_state and game_state.mate_retry_active:
        mate_move = find_mate_preserving_move(board, engine, game_state.mate_distance)
        if mate_move:
            return MoveResult(
                move=mate_move,
                move_san=board.san(mate_move),
                move_rank=_rank_of_move(candidates, mate_move),
                top_engine_move=top_engine_move_san,
                deep_eval=round(best_eval_for_us, 3),
                move_number=move_number,
                puzzle_found=True,
                puzzle_type="mate",
                puzzle_is_mate=True,
                puzzle_mate_distance=game_state.mate_distance,
                puzzle_retries_remaining=game_state.mate_retries_allowed - game_state.mate_retries_used,
                candidate_evals=candidate_evals_log,
                current_shallow_depth=current_shallow_depth,
                deep_depth=config.deep_depth,
            )
        else:
            game_state.mate_retry_active = False

    # === 4. CAPTURE PRIORITY ===
    top_move = candidates[0][0]
    if board.is_capture(top_move) and len(candidates) >= 2:
        second_eval = candidates[1][1] * sign
        capture_gain = best_eval_for_us - second_eval
        if capture_gain > 0.5:
            return MoveResult(
                move=top_move,
                move_san=board.san(top_move),
                move_rank=0,
                top_engine_move=top_engine_move_san,
                deep_eval=round(best_eval_for_us, 3),
                move_number=move_number,
                candidate_evals=candidate_evals_log,
                current_shallow_depth=current_shallow_depth,
                deep_depth=config.deep_depth,
            )

    # === EFFECTIVE MAX EVAL COST ===
    effective_max_eval_cost = config.max_eval_cost
    if best_eval_for_us < config.desperate_eval_threshold:
        effective_max_eval_cost = config.desperate_max_eval_cost
    elif best_eval_for_us < config.losing_eval_threshold:
        effective_max_eval_cost = config.losing_max_eval_cost

    # === 5. PUZZLE SEARCH ===
    all_puzzles = []
    positions_screened = 0
    disagreements_found = 0

    # --- Step A: Negative puzzles (immediate) ---
    for rank, (cand_move, cand_eval_white) in enumerate(candidates):
        cand_eval = cand_eval_white * sign
        eval_cost = best_eval_for_us - cand_eval
        if eval_cost > effective_max_eval_cost:
            continue

        board.push(cand_move)

        if not board.is_game_over():
            positions_screened += 1

            puzzle = detect_puzzle(
                board, maia, engine, config,
                our_move=cand_move,
                eval_cost_to_create=max(0.0, eval_cost),
                move_number=move_number,
                puzzles_found=puzzles_found,
            )
        else:
            puzzle = None

        board.pop()

        if puzzle:
            puzzle.our_move_san = board.san(cand_move)
            puzzle.search_depth = 0
            disagreements_found += 1
            all_puzzles.append(puzzle)

    # --- Step B: Positive puzzles (immediate) ---
    if config.enable_positive_puzzles:
        for rank, (cand_move, cand_eval_white) in enumerate(candidates):
            cand_eval = cand_eval_white * sign
            eval_cost = best_eval_for_us - cand_eval
            if eval_cost > effective_max_eval_cost:
                continue

            puzzle = detect_positive_puzzle(
                board, cand_move, maia, engine, config,
                eval_cost_to_create=max(0.0, eval_cost),
                move_number=move_number,
                puzzles_found=puzzles_found,
            )

            if puzzle:
                puzzle.our_move_san = board.san(cand_move)
                puzzle.search_depth = 0
                all_puzzles.append(puzzle)

    # --- Step C: Mate puzzles ---
    for rank, (cand_move, cand_eval_white) in enumerate(candidates):
        cand_eval = cand_eval_white * sign
        eval_cost = best_eval_for_us - cand_eval
        if eval_cost > effective_max_eval_cost:
            continue

        puzzle = detect_mate_puzzle(
            board, engine, config,
            our_move=cand_move,
            eval_cost_to_create=max(0.0, eval_cost),
            move_number=move_number,
        )

        if puzzle:
            puzzle.our_move_san = board.san(cand_move)
            puzzle.search_depth = 0
            all_puzzles.append(puzzle)

    # --- Step D: One-move-ahead screening ---
    if not all_puzzles and config.max_lookahead >= 1:
        for rank, (cand_move, cand_eval_white) in enumerate(candidates[:3]):
            cand_eval = cand_eval_white * sign
            eval_cost = best_eval_for_us - cand_eval
            if eval_cost > effective_max_eval_cost:
                continue

            board.push(cand_move)

            floor_pred = maia.predict(board, config.floor_rating)
            floor_response = floor_pred.top_move

            if floor_response in board.legal_moves:
                board.push(floor_response)
                positions_screened += 1

                followups = get_top_moves(engine, board, num_moves=3, depth=config.deep_depth)

                for f_move, f_eval_white in followups:
                    board.push(f_move)

                    puzzle = detect_puzzle(
                        board, maia, engine, config,
                        our_move=f_move,
                        eval_cost_to_create=max(0.0, eval_cost),
                        move_number=move_number,
                        puzzles_found=puzzles_found,
                    )

                    board.pop()

                    if puzzle:
                        puzzle.our_move = cand_move
                        puzzle.our_move_san = board.san(cand_move) if cand_move in board.legal_moves else cand_move.uci()
                        puzzle.search_depth = 1
                        all_puzzles.append(puzzle)
                        break

                board.pop()  # undo floor_response

            board.pop()  # undo cand_move

    # --- Step E: Two-move-ahead screening ---
    if not all_puzzles and config.max_lookahead >= 2:
        for rank, (cand_move, cand_eval_white) in enumerate(candidates[:2]):
            cand_eval = cand_eval_white * sign
            eval_cost = best_eval_for_us - cand_eval
            if eval_cost > effective_max_eval_cost:
                continue

            board.push(cand_move)

            floor_pred = maia.predict(board, config.floor_rating)
            floor_resp1 = floor_pred.top_move

            if floor_resp1 not in board.legal_moves:
                board.pop()
                continue

            board.push(floor_resp1)

            our_followups = get_top_moves(engine, board, num_moves=2, depth=config.deep_depth)

            found_at_depth2 = False
            for y_move, _ in our_followups:
                board.push(y_move)

                floor_pred2 = maia.predict(board, config.floor_rating)
                floor_resp2 = floor_pred2.top_move

                if floor_resp2 in board.legal_moves:
                    board.push(floor_resp2)
                    positions_screened += 1

                    our_z = get_top_moves(engine, board, num_moves=2, depth=config.deep_depth)

                    for z_move, z_eval_white in our_z:
                        board.push(z_move)

                        puzzle = detect_puzzle(
                            board, maia, engine, config,
                            our_move=z_move,
                            eval_cost_to_create=max(0.0, eval_cost),
                            move_number=move_number,
                            puzzles_found=puzzles_found,
                        )

                        board.pop()

                        if puzzle:
                            puzzle.our_move = cand_move
                            puzzle.search_depth = 2
                            all_puzzles.append(puzzle)
                            found_at_depth2 = True
                            break

                    board.pop()  # undo floor_resp2

                board.pop()  # undo y_move

                if found_at_depth2:
                    break

            board.pop()  # undo floor_resp1
            board.pop()  # undo cand_move

            if found_at_depth2:
                break

    # === Step F: Score and select ===
    puzzles_after_validation = len(all_puzzles)

    if all_puzzles:
        best = max(all_puzzles, key=lambda p: p.score)

        puzzle_move = best.our_move
        puzzle_rank = _rank_of_move(candidates, puzzle_move)

        if best.is_mate and game_state:
            game_state.enter_mate_retry(best.mate_distance)

        return MoveResult(
            move=puzzle_move,
            move_san=board.san(puzzle_move) if puzzle_move in board.legal_moves else puzzle_move.uci(),
            move_rank=puzzle_rank,
            top_engine_move=top_engine_move_san,
            deep_eval=round(best_eval_for_us, 3),
            move_number=move_number,
            puzzle_found=True,
            puzzle_type=best.puzzle_type,
            puzzle_score=round(best.score, 3),
            puzzle_eval_gap=round(best.eval_gap, 3),
            puzzle_floor_prob=round(best.floor_prob_correct, 3),
            puzzle_disagreement=best.disagreement_type,
            puzzle_search_depth=best.search_depth,
            puzzle_better_move_san=best.better_move_san,
            puzzle_worse_move_san=best.worse_move_san,
            puzzle_is_mate=best.is_mate,
            puzzle_mate_distance=best.mate_distance,
            positions_screened=positions_screened,
            disagreements_found=disagreements_found,
            puzzles_after_validation=puzzles_after_validation,
            candidate_evals=candidate_evals_log,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
        )

    # === 6. FALLBACK ===
    move = candidates[0][0]
    return MoveResult(
        move=move,
        move_san=board.san(move),
        move_rank=0,
        top_engine_move=top_engine_move_san,
        deep_eval=round(best_eval_for_us, 3),
        move_number=move_number,
        positions_screened=positions_screened,
        disagreements_found=disagreements_found,
        puzzles_after_validation=0,
        candidate_evals=candidate_evals_log,
        current_shallow_depth=current_shallow_depth,
        deep_depth=config.deep_depth,
    )


def _rank_of_move(candidates, move):
    """Find the rank of a move in the candidate list."""
    for i, (m, _) in enumerate(candidates):
        if m == move:
            return i
    return len(candidates)

"""
Stonefish Scoring — Find THE Move + Depth Disagreement
=======================================================
Two filters determine a critical moment:
    1. GAP: The best opponent response is >= gap_threshold pawns better than
       the second-best. One move works, everything else hurts.
    2. DISAGREEMENT: The best move at deep depth (12) is ranked poorly at
       shallow depth (2). The right move isn't obvious — it requires
       calculation, not pattern recognition.

Both must be true. This filters out trivial recaptures and check escapes
while keeping genuinely hard positions where intuition fails.

Depth disagreement also drives move selection: Stonefish prefers candidates
that look bad at shallow depth but are actually fine at deep depth —
sacrifices that aren't really sacrifices, quiet moves that set traps.
"""

import chess
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import get_top_moves

from .config import StonefishConfig


@dataclass
class MoveScore:
    """Result of nettlesomeness evaluation for a single candidate move."""
    move: chess.Move
    move_san: str
    own_eval: float              # This candidate's eval (from our perspective, pawns)
    move_rank: int               # Rank among our candidates (0 = top Stockfish move)
    gap: float                   # eval(opponent move 1) - eval(opponent move 2) — higher = more nettlesome
    opponent_evals: List[Tuple[str, float]]  # All opponent response (move_san, eval) pairs
    nettlesomeness_score: float  # Final composite score used for move selection
    deep_best_opponent_move: Optional[chess.Move] = None  # Best opponent response at deep depth


@dataclass
class CriticalityResult:
    """Assessment of how critical a position is for the opponent."""
    is_critical: bool
    gap: float                   # The move 1 vs move 2 gap
    disagreement: int            # Rank of deep best move at shallow depth (0 = agrees)


def compute_move_score(
    engine,
    board: chess.Board,
    candidate_move: chess.Move,
    candidate_eval: float,
    candidate_rank: int,
    depth: int,
    num_responses: int = 6,
) -> MoveScore:
    """Evaluate a candidate move by measuring the gap in opponent's responses.

    After pushing candidate_move, gets the opponent's top responses at deep depth.
    The gap = eval(move 1) - eval(move 2) from opponent's perspective.
    Higher gap = only one move works = more nettlesome.

    Also returns the deep best opponent move for later disagreement checking.
    """
    our_color = board.turn
    opp_sign = -1.0 if our_color == chess.WHITE else 1.0

    move_san = board.san(candidate_move)
    board.push(candidate_move)

    # Get opponent's top responses at deep depth
    responses = get_top_moves(engine, board, num_moves=num_responses, depth=depth)

    board.pop()

    if len(responses) < 2:
        opp_evals = [(board.san(m) if board.is_legal(m) else str(m), e * opp_sign)
                     for m, e in responses] if responses else []
        deep_best = responses[0][0] if responses else None
        return MoveScore(
            move=candidate_move,
            move_san=move_san,
            own_eval=candidate_eval,
            move_rank=candidate_rank,
            gap=0.0,
            opponent_evals=opp_evals,
            nettlesomeness_score=0.0,
            deep_best_opponent_move=deep_best,
        )

    # Convert evals to opponent's perspective
    opp_evals_raw = [e * opp_sign for _, e in responses]

    # Build (san, eval) pairs for logging
    board.push(candidate_move)
    opp_eval_pairs = []
    for move, eval_white in responses:
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        opp_eval_pairs.append((san, eval_white * opp_sign))
    board.pop()

    # THE gap: how much worse is the second-best move compared to the best?
    gap = opp_evals_raw[0] - opp_evals_raw[1]

    return MoveScore(
        move=candidate_move,
        move_san=move_san,
        own_eval=candidate_eval,
        move_rank=candidate_rank,
        gap=gap,
        opponent_evals=opp_eval_pairs,
        nettlesomeness_score=gap,  # Will be adjusted by caller
        deep_best_opponent_move=responses[0][0],
    )


def compute_disagreement(
    engine,
    board: chess.Board,
    after_move: chess.Move,
    deep_best_opponent_move: chess.Move,
    shallow_depth: int,
    num_check: int = 6,
) -> int:
    """Compute depth disagreement for the opponent's position.

    After pushing after_move (Stonefish's candidate), evaluate the opponent's
    position at shallow_depth and find where the deep best move ranks.

    Returns the rank of deep_best_opponent_move at shallow depth.
    0 = shallow agrees with deep (obvious move), higher = more disagreement.
    """
    board.push(after_move)

    shallow_responses = get_top_moves(engine, board, num_moves=num_check, depth=shallow_depth)

    board.pop()

    if not shallow_responses:
        return 0

    for i, (move, _) in enumerate(shallow_responses):
        if move == deep_best_opponent_move:
            return i

    # Not found in top N at shallow depth — maximum disagreement
    return num_check


def assess_criticality(
    gap: float,
    disagreement: int,
    config: StonefishConfig,
) -> CriticalityResult:
    """Determine if a position is critical for the opponent.

    Critical = ALL of:
        1. Gap >= gap_threshold — real consequences
        2. Disagreement >= disagreement_threshold — right move isn't obvious
        3. (Cooldown is checked separately in game_state.py)
    """
    is_critical = (
        gap >= config.gap_threshold
        and disagreement >= config.disagreement_threshold
    )

    return CriticalityResult(
        is_critical=is_critical,
        gap=gap,
        disagreement=disagreement,
    )


def rank_opponent_move(
    engine,
    board: chess.Board,
    opponent_move: chess.Move,
    target_band: int,
    depth: int,
) -> Tuple[int, float, str]:
    """Determine the rank, eval cost, and best move for the opponent's actual move.

    Returns:
        (rank, eval_cost, best_move_san) where rank is 0-indexed among top moves,
        eval_cost is how many pawns they lost vs the best move, and
        best_move_san is the SAN of the objectively best move.
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


def _measure_gap_after_position(engine, board, depth, num_responses=6):
    """Measure the gap between opponent's best and second-best move in a position.

    board should already be set to the position where it's the opponent's turn.
    Returns (gap, deep_best_opponent_move) where gap is eval(move1) - eval(move2)
    from the opponent's perspective.
    """
    opp_color = board.turn
    opp_sign = 1.0 if opp_color == chess.WHITE else -1.0

    responses = get_top_moves(engine, board, num_moves=num_responses, depth=depth)

    if len(responses) < 2:
        deep_best = responses[0][0] if responses else None
        return (0.0, deep_best)

    opp_evals = [e * opp_sign for _, e in responses]
    gap = opp_evals[0] - opp_evals[1]
    return (gap, responses[0][0])


def _search_depth1(engine, board, candidates, sign, best_eval_for_us,
                   effective_max_eval_cost, config, gap_threshold):
    """Priority search at depth 1: does any candidate immediately create a hard position?

    For each of our candidate moves, push it, then measure the gap in the
    opponent's responses. If gap >= gap_threshold, we found a Concept 1 move.

    Returns list of (move, rank, eval_for_us, gap, deep_best_opp_move) sorted by gap desc,
    filtered to only those meeting the threshold.
    """
    results = []
    for rank, (move, eval_white) in enumerate(candidates):
        eval_for_us = eval_white * sign
        eval_cost = best_eval_for_us - eval_for_us
        if eval_cost > effective_max_eval_cost:
            continue

        board.push(move)
        gap, deep_best_opp = _measure_gap_after_position(
            engine, board, config.deep_depth)
        board.pop()

        if gap >= gap_threshold:
            results.append((move, rank, eval_for_us, gap, deep_best_opp))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def _search_depth2(engine, board, candidates, sign, best_eval_for_us,
                   effective_max_eval_cost, config, gap_threshold):
    """Priority search at depth 2: play move X, opponent plays natural response, NOW is it hard?

    For each candidate move X:
        - Push X
        - Get opponent's best response (their top Stockfish move)
        - Push that response
        - Now measure: does the BOT have a follow-up Y that creates a Concept 1 position?
        - If gap after Y >= gap_threshold, we found a 2-move chain

    Returns list of (move_x, rank, eval_for_us, best_gap_at_depth2, chain_info) sorted by gap desc.
    """
    results = []
    for rank, (move_x, eval_white) in enumerate(candidates):
        eval_for_us = eval_white * sign
        eval_cost = best_eval_for_us - eval_for_us
        if eval_cost > effective_max_eval_cost:
            continue

        board.push(move_x)

        # Get opponent's most natural response (top Stockfish move)
        opp_responses = get_top_moves(engine, board, num_moves=1, depth=config.deep_depth)
        if not opp_responses:
            board.pop()
            continue

        opp_move = opp_responses[0][0]
        board.push(opp_move)

        # Now it's our turn again. Find OUR best follow-up Y that creates a hard position.
        our_followups = get_top_moves(engine, board, num_moves=config.num_candidates,
                                      depth=config.deep_depth)

        best_chain_gap = 0.0
        best_chain_info = None

        # Check our top follow-ups for one that creates a Concept 1 position
        our_best_eval = our_followups[0][1] * sign if our_followups else 0.0
        for y_move, y_eval_white in our_followups:
            y_eval_cost = our_best_eval - (y_eval_white * sign)
            if y_eval_cost > effective_max_eval_cost:
                continue

            board.push(y_move)
            gap, deep_best_opp = _measure_gap_after_position(
                engine, board, config.deep_depth)
            board.pop()

            if gap > best_chain_gap:
                best_chain_gap = gap
                best_chain_info = {
                    "opp_response": opp_move,
                    "our_followup": y_move,
                    "gap_after_followup": gap,
                }

        board.pop()  # undo opp_move
        board.pop()  # undo move_x

        if best_chain_gap >= gap_threshold:
            results.append((move_x, rank, eval_for_us, best_chain_gap, best_chain_info))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def _search_depth3(engine, board, candidates, sign, best_eval_for_us,
                   effective_max_eval_cost, config, gap_threshold):
    """Priority search at depth 3: X -> opp responds -> Y -> opp responds -> NOW is it hard?

    For each candidate X:
        - Push X, opponent plays best response
        - We play best follow-up Y, opponent plays best response
        - NOW measure: does the bot have a follow-up Z that creates Concept 1?

    Returns list of (move_x, rank, eval_for_us, best_gap_at_depth3, chain_info) sorted by gap desc.
    """
    results = []
    for rank, (move_x, eval_white) in enumerate(candidates):
        eval_for_us = eval_white * sign
        eval_cost = best_eval_for_us - eval_for_us
        if eval_cost > effective_max_eval_cost:
            continue

        board.push(move_x)

        # Opponent's best response to X
        opp_resp1 = get_top_moves(engine, board, num_moves=1, depth=config.deep_depth)
        if not opp_resp1:
            board.pop()
            continue
        opp_move1 = opp_resp1[0][0]
        board.push(opp_move1)

        # Our best follow-up Y (just take the top move to keep search manageable)
        our_y_moves = get_top_moves(engine, board, num_moves=3, depth=config.deep_depth)
        if not our_y_moves:
            board.pop()
            board.pop()
            continue

        best_chain_gap = 0.0
        best_chain_info = None

        # Try top 3 Y moves
        for y_move, _ in our_y_moves:
            board.push(y_move)

            # Opponent's best response to Y
            opp_resp2 = get_top_moves(engine, board, num_moves=1, depth=config.deep_depth)
            if not opp_resp2:
                board.pop()
                continue
            opp_move2 = opp_resp2[0][0]
            board.push(opp_move2)

            # Now it's our turn. Find a Z that creates Concept 1.
            our_z_moves = get_top_moves(engine, board, num_moves=config.num_candidates,
                                         depth=config.deep_depth)
            our_z_best_eval = our_z_moves[0][1] * sign if our_z_moves else 0.0

            for z_move, z_eval_white in our_z_moves:
                z_eval_cost = our_z_best_eval - (z_eval_white * sign)
                if z_eval_cost > effective_max_eval_cost:
                    continue

                board.push(z_move)
                gap, _ = _measure_gap_after_position(engine, board, config.deep_depth)
                board.pop()

                if gap > best_chain_gap:
                    best_chain_gap = gap
                    best_chain_info = {
                        "opp_response1": opp_move1,
                        "our_followup_y": y_move,
                        "opp_response2": opp_move2,
                        "our_followup_z": z_move,
                        "gap_after_z": gap,
                    }

            board.pop()  # undo opp_move2
            board.pop()  # undo y_move

        board.pop()  # undo opp_move1
        board.pop()  # undo move_x

        if best_chain_gap >= gap_threshold:
            results.append((move_x, rank, eval_for_us, best_chain_gap, best_chain_info))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def select_stonefish_move(engine, board, config, current_shallow_depth, move_number,
                          our_clock_seconds=None):
    """Progressive depth search: the core of Stonefish.

    Priority system:
        1. High threshold (1.0 pawn gap) at depth 1 — immediate Concept 1
        2. High threshold at depth 2 — 2-move chain to Concept 1
        3. High threshold at depth 3 — 3-move chain to Concept 1
        4. Low threshold (0.5 pawn gap) back through depths 1-3
        5. Fallback: random from top 3 Stockfish moves

    At each depth, we look for a candidate move from our top-N Stockfish moves
    that leads to a position where the opponent has only one good reply
    (measured by the gap between their best and second-best response).

    Emergency mode — When clock is below emergency_clock_seconds:
        - Skip deep search entirely, play at emergency_depth

    Returns a MoveResult with all data for logging/chat.
    """
    from .game_state import MoveResult
    import random as _random

    our_color = board.turn
    sign = 1.0 if our_color == chess.WHITE else -1.0

    # === EMERGENCY MODE ===
    emergency = (our_clock_seconds is not None
                 and our_clock_seconds < config.emergency_clock_seconds)

    if emergency:
        emg_candidates = get_top_moves(
            engine, board,
            num_moves=3,
            depth=config.emergency_depth,
        )
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
            shallow_eval=round(ev * sign, 3),
            candidate_evals=[(board.san(m), round(e, 3)) for m, e in emg_candidates],
            nettlesomeness_score=0.0,
            gap=0.0,
            disagreement=0,
            was_flagged_critical=False,
            current_shallow_depth=config.emergency_depth,
            deep_depth=0,
            move_number=move_number,
            emergency_mode=True,
            search_depth_found=0,
            search_threshold_used=0.0,
        )

    # === PASS 1: Generate candidates at shallow depth (sets playing strength) ===
    num_cands = config.num_candidates
    candidates = get_top_moves(
        engine, board,
        num_moves=num_cands,
        depth=current_shallow_depth,
    )

    if not candidates:
        fallback = _random.choice(list(board.legal_moves))
        return MoveResult(
            move=fallback,
            move_san=board.san(fallback),
            move_rank=0,
            top_engine_move=board.san(fallback),
            deep_eval=0.0,
            shallow_eval=0.0,
            candidate_evals=[],
            nettlesomeness_score=0.0,
            gap=0.0,
            disagreement=0,
            was_flagged_critical=False,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
            move_number=move_number,
            search_depth_found=0,
            search_threshold_used=0.0,
        )

    best_eval_for_us = candidates[0][1] * sign
    top_engine_move_san = board.san(candidates[0][0])
    shallow_eval = best_eval_for_us

    # Build candidate evals for logging
    candidate_evals_log = [(board.san(m), round(e, 3)) for m, e in candidates]

    # === OPENING BOOK: just play Stockfish's top move for first N moves ===
    if move_number <= config.opening_book_moves:
        move = candidates[0][0]
        return MoveResult(
            move=move,
            move_san=board.san(move),
            move_rank=0,
            top_engine_move=top_engine_move_san,
            deep_eval=round(best_eval_for_us, 3),
            shallow_eval=round(shallow_eval, 3),
            candidate_evals=candidate_evals_log,
            nettlesomeness_score=0.0,
            gap=0.0,
            disagreement=0,
            was_flagged_critical=False,
            current_shallow_depth=current_shallow_depth,
            deep_depth=config.deep_depth,
            move_number=move_number,
            search_depth_found=0,
            search_threshold_used=0.0,
        )

    # Determine effective max_eval_cost based on position eval
    effective_max_eval_cost = config.max_eval_cost
    if shallow_eval < config.desperate_eval_threshold:
        effective_max_eval_cost = config.desperate_max_eval_cost
    elif shallow_eval < config.losing_eval_threshold:
        effective_max_eval_cost = config.losing_max_eval_cost

    # === PROGRESSIVE DEPTH SEARCH ===
    # Try high threshold (1.0) at depths 1, 2, 3
    # Then low threshold (0.5) at depths 1, 2, 3
    # Then fallback to random top 3

    high_threshold = config.gap_threshold        # 1.0 pawn default
    low_threshold = config.gap_threshold_low      # 0.5 pawn default

    chosen = None       # Will be (move, rank, eval_for_us, gap, search_depth, threshold)

    # --- High threshold pass ---
    # Depth 1
    d1_results = _search_depth1(engine, board, candidates, sign, best_eval_for_us,
                                effective_max_eval_cost, config, high_threshold)
    if d1_results:
        m, r, ev, g, _ = d1_results[0]
        chosen = (m, r, ev, g, 1, high_threshold)

    # Depth 2
    if chosen is None and config.max_search_depth >= 2:
        d2_results = _search_depth2(engine, board, candidates, sign, best_eval_for_us,
                                    effective_max_eval_cost, config, high_threshold)
        if d2_results:
            m, r, ev, g, _ = d2_results[0]
            chosen = (m, r, ev, g, 2, high_threshold)

    # Depth 3
    if chosen is None and config.max_search_depth >= 3:
        d3_results = _search_depth3(engine, board, candidates, sign, best_eval_for_us,
                                    effective_max_eval_cost, config, high_threshold)
        if d3_results:
            m, r, ev, g, _ = d3_results[0]
            chosen = (m, r, ev, g, 3, high_threshold)

    # --- Low threshold pass ---
    if chosen is None:
        # Reuse depth 1 results if we already have them, just filter lower
        d1_low = _search_depth1(engine, board, candidates, sign, best_eval_for_us,
                                effective_max_eval_cost, config, low_threshold)
        if d1_low:
            m, r, ev, g, _ = d1_low[0]
            chosen = (m, r, ev, g, 1, low_threshold)

    if chosen is None and config.max_search_depth >= 2:
        d2_low = _search_depth2(engine, board, candidates, sign, best_eval_for_us,
                                effective_max_eval_cost, config, low_threshold)
        if d2_low:
            m, r, ev, g, _ = d2_low[0]
            chosen = (m, r, ev, g, 2, low_threshold)

    if chosen is None and config.max_search_depth >= 3:
        d3_low = _search_depth3(engine, board, candidates, sign, best_eval_for_us,
                                effective_max_eval_cost, config, low_threshold)
        if d3_low:
            m, r, ev, g, _ = d3_low[0]
            chosen = (m, r, ev, g, 3, low_threshold)

    # --- Fallback: random from top 3 ---
    if chosen is None:
        top3 = candidates[:min(3, len(candidates))]
        pick = _random.choice(top3)
        chosen_move = pick[0]
        chosen_rank = candidates.index(pick)
        chosen_eval = pick[1] * sign
        chosen_gap = 0.0
        search_depth_found = 0
        search_threshold_used = 0.0
    else:
        chosen_move, chosen_rank, chosen_eval, chosen_gap, search_depth_found, search_threshold_used = chosen

    # === DISAGREEMENT CHECK for criticality ===
    # Only do this for depth-1 hits (immediate positions) where we have the info
    disagreement = 0
    if search_depth_found == 1 and d1_results:
        _, _, _, _, deep_best_opp = d1_results[0]
        if deep_best_opp is not None and chosen_gap >= config.gap_threshold:
            disagreement = compute_disagreement(
                engine, board, chosen_move, deep_best_opp,
                shallow_depth=config.shallow_comparison_depth,
            )

    criticality = assess_criticality(chosen_gap, disagreement, config)

    return MoveResult(
        move=chosen_move,
        move_san=board.san(chosen_move),
        move_rank=chosen_rank,
        top_engine_move=top_engine_move_san,
        deep_eval=round(chosen_eval, 3),
        shallow_eval=round(shallow_eval, 3),
        candidate_evals=candidate_evals_log,
        nettlesomeness_score=round(chosen_gap, 3),
        gap=round(chosen_gap, 3),
        disagreement=disagreement,
        was_flagged_critical=criticality.is_critical,
        current_shallow_depth=current_shallow_depth,
        deep_depth=config.deep_depth,
        move_number=move_number,
        search_depth_found=search_depth_found,
        search_threshold_used=round(search_threshold_used, 3),
    )

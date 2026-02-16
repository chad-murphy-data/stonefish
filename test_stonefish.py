"""
Stonefish Test Suite
====================
Tests gap-based scoring, depth disagreement, adaptive depth, two-pass selection, and self-play.
Run: python test_stonefish.py
"""

import chess
import chess.engine
import time
import sys
from collections import Counter

from engine import STOCKFISH_PATH, get_top_moves, RandomTopNBot, PureStockfishBot
from stonefish.config import StonefishConfig
from stonefish.scoring import (
    compute_move_score, compute_disagreement, assess_criticality, rank_opponent_move,
)
from stonefish.adaptive import AdaptiveDepthTracker
from stonefish.bot import StonefishBot


def test_gap_and_disagreement(engine):
    """Test gap-based scoring + depth disagreement on opening moves."""
    print("=" * 60)
    print("TEST 1: Gap Scoring + Depth Disagreement (Opening Position)")
    print("=" * 60)

    board = chess.Board()
    config = StonefishConfig()

    # Get top 5 candidate moves
    candidates = get_top_moves(engine, board, num_moves=5, depth=14)

    print(f"\nScoring top {len(candidates)} opening moves "
          f"(gap_threshold={config.gap_threshold}, "
          f"disagree_threshold={config.disagreement_threshold}):\n")
    print(f"  {'Move':<8} {'Eval':>6} {'Gap':>8} {'Disagree':>9} {'Critical?':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*9} {'-'*10}")

    for rank, (move, eval_white) in enumerate(candidates):
        sign = 1.0
        score = compute_move_score(
            engine, board, move,
            candidate_eval=eval_white * sign,
            candidate_rank=rank,
            depth=14,
        )

        # Compute disagreement for this candidate
        if score.deep_best_opponent_move is not None:
            disagree = compute_disagreement(
                engine, board, move, score.deep_best_opponent_move,
                shallow_depth=config.shallow_comparison_depth,
            )
        else:
            disagree = 0

        crit = assess_criticality(score.gap, disagree, config)
        crit_str = "YES" if crit.is_critical else "no"

        print(f"  {score.move_san:<8} {score.own_eval:>+6.2f} {score.gap:>8.3f} "
              f"{disagree:>9} {crit_str:>10}")

    print("\n  OK - gap scoring + disagreement produces values for all candidates")
    return True


def test_disagreement_filter():
    """Test that assess_criticality requires BOTH gap AND disagreement."""
    print("\n" + "=" * 60)
    print("TEST 2: Disagreement Filter Logic")
    print("=" * 60)

    config = StonefishConfig(gap_threshold=1.0, disagreement_threshold=3)

    # High gap, low disagreement = NOT critical (obvious move)
    r1 = assess_criticality(gap=2.0, disagreement=0, config=config)
    assert not r1.is_critical, "Gap=2.0, disagree=0 should NOT be critical"
    print(f"  Gap=2.0, disagree=0: critical={r1.is_critical} OK (obvious recapture)")

    # High gap, borderline disagreement = NOT critical
    r2 = assess_criticality(gap=1.5, disagreement=2, config=config)
    assert not r2.is_critical, "Gap=1.5, disagree=2 should NOT be critical"
    print(f"  Gap=1.5, disagree=2: critical={r2.is_critical} OK (below threshold)")

    # High gap, meets disagreement = CRITICAL
    r3 = assess_criticality(gap=1.5, disagreement=3, config=config)
    assert r3.is_critical, "Gap=1.5, disagree=3 should be critical"
    print(f"  Gap=1.5, disagree=3: critical={r3.is_critical} OK (genuinely hard)")

    # High gap, high disagreement = CRITICAL
    r4 = assess_criticality(gap=2.0, disagreement=5, config=config)
    assert r4.is_critical, "Gap=2.0, disagree=5 should be critical"
    print(f"  Gap=2.0, disagree=5: critical={r4.is_critical} OK (very hard)")

    # Low gap, high disagreement = NOT critical (no consequences)
    r5 = assess_criticality(gap=0.5, disagreement=5, config=config)
    assert not r5.is_critical, "Gap=0.5, disagree=5 should NOT be critical"
    print(f"  Gap=0.5, disagree=5: critical={r5.is_critical} OK (no consequences)")

    print("\n  All disagreement filter tests passed!")
    return True


def test_adaptive_depth():
    """Test the adaptive depth state machine (boost only, target_band=1)."""
    print("\n" + "=" * 60)
    print("TEST 3: Adaptive Depth Tracker (Boost Only, Band=1)")
    print("=" * 60)

    tracker = AdaptiveDepthTracker(
        base_depth=6,
        boost_on_miss=5,
        boost_duration=5,
        target_band=1,
    )

    print(f"\n  Base depth: {tracker.base_depth}")
    assert tracker.current_depth == 6
    print(f"  Initial depth: {tracker.current_depth} OK")

    # Non-critical moves don't trigger
    tracker.record_opponent_move(move_rank=0, was_critical=False, move_number=5)
    assert tracker.current_depth == 6
    print(f"  After non-critical move: {tracker.current_depth} OK (unchanged)")

    # Rank 1 = miss with target_band=1
    tracker.record_opponent_move(move_rank=1, was_critical=True, move_number=10)
    assert tracker.current_depth == 11
    print(f"  After miss (rank 1, critical): {tracker.current_depth} OK (boosted)")

    # Tick down
    for i in range(5):
        tracker.tick()
    assert tracker.current_depth == 6
    print(f"  After 5 ticks (expired): {tracker.current_depth} OK (back to base)")

    # Rank 0 = found
    tracker.record_opponent_move(move_rank=0, was_critical=True, move_number=20)
    assert tracker.current_depth == 6
    assert tracker.active_adjustment is None
    print(f"  After find (rank 0): {tracker.current_depth} OK (back to baseline)")

    # Find cancels active boost
    tracker.record_opponent_move(move_rank=2, was_critical=True, move_number=25)
    assert tracker.current_depth == 11
    tracker.record_opponent_move(move_rank=0, was_critical=True, move_number=26)
    assert tracker.current_depth == 6
    print(f"  Find cancels boost: {tracker.current_depth} OK")

    print("\n  All adaptive depth tests passed!")
    return True


def test_two_pass_selection(engine):
    """Test that two-pass selection returns all expected fields including disagreement."""
    print("\n" + "=" * 60)
    print("TEST 4: Two-Pass Move Selection (Gap + Disagreement)")
    print("=" * 60)

    config = StonefishConfig(deep_depth=12, base_depth=6)
    bot = StonefishBot(engine, config)
    bot.start_game(chess.WHITE)

    board = chess.Board()

    start = time.time()
    result = bot.choose_move_full(board)
    elapsed = time.time() - start

    print(f"\n  Move: {result.move_san} (rank {result.move_rank})")
    print(f"  Top engine move: {result.top_engine_move}")
    print(f"  Deep eval: {result.deep_eval:+.3f}")
    print(f"  Shallow eval: {result.shallow_eval:+.3f}")
    print(f"  Nettlesomeness: {result.nettlesomeness_score:.3f}")
    print(f"  Gap (move1 vs move2): {result.gap:.3f}")
    print(f"  Disagreement: {result.disagreement}")
    print(f"  Critical: {result.was_flagged_critical}")
    print(f"  Emergency: {result.emergency_mode}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Candidates: {len(result.candidate_evals)}")

    assert result.move is not None
    assert result.deep_depth == 12
    assert result.current_shallow_depth == 6
    assert hasattr(result, 'gap')
    assert hasattr(result, 'disagreement')
    assert isinstance(result.disagreement, int)

    bot.end_game()
    print("\n  Two-pass selection OK")
    return True


def test_self_play(engine, num_games=4):
    """Play Stonefish vs PureStockfish and print stats."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Self-Play ({num_games} games, Stonefish vs PureStockfish)")
    print("=" * 60)

    config = StonefishConfig(deep_depth=12, base_depth=6)
    stonefish = StonefishBot(engine, config)
    stockfish = PureStockfishBot(engine, depth=14)

    all_ranks = []
    all_critical_counts = []
    all_critical_found = []
    all_disagreements = []
    all_moves = []
    results = {"W": 0, "D": 0, "L": 0}

    for game_num in range(num_games):
        sf_is_white = game_num % 2 == 0
        if sf_is_white:
            white_bot, black_bot = stonefish, stockfish
        else:
            white_bot, black_bot = stockfish, stonefish

        stonefish.start_game(chess.WHITE if sf_is_white else chess.BLACK)

        board = chess.Board()
        move_count = 0

        start = time.time()
        for _ in range(300):
            if board.is_game_over():
                break

            if board.turn == chess.WHITE:
                bot = white_bot
            else:
                bot = black_bot

            move = bot.choose_move(board)
            board.push(move)
            move_count += 1

        elapsed = time.time() - start

        state = stonefish.end_game(board)
        if state:
            for mr in state.move_results:
                all_ranks.append(mr.move_rank)
                all_disagreements.append(mr.disagreement)
            critical_count = sum(1 for o in state.opponent_results if o.was_critical)
            found_count = sum(1 for o in state.opponent_results
                              if o.was_critical and o.found_critical)
            all_critical_counts.append(critical_count)
            all_critical_found.append(found_count)

        all_moves.append(move_count)

        board_result = board.result()
        if board_result == "1-0":
            outcome = "W" if sf_is_white else "L"
        elif board_result == "0-1":
            outcome = "L" if sf_is_white else "W"
        else:
            outcome = "D"
        results[outcome] += 1

        color_str = "White" if sf_is_white else "Black"
        print(f"  Game {game_num + 1}: Stonefish ({color_str}) -"
              f"{board_result} in {move_count} moves ({elapsed:.0f}s) "
              f"[critical: {critical_count}, found: {found_count}]")

    # Summary
    print(f"\n  {'='*50}")
    print(f"  RESULTS: {results['W']}W / {results['D']}D / {results['L']}L")
    print(f"  Avg game length: {sum(all_moves) / len(all_moves):.0f} half-moves")

    rank_counter = Counter(all_ranks)
    print(f"\n  Stonefish move rank distribution:")
    for rank in sorted(rank_counter.keys()):
        count = rank_counter[rank]
        pct = count / len(all_ranks) * 100
        bar = "#" * int(pct / 2)
        print(f"    Rank {rank}: {count:>4} ({pct:>5.1f}%) {bar}")

    non_top = sum(1 for r in all_ranks if r > 0)
    non_top_pct = non_top / len(all_ranks) * 100 if all_ranks else 0
    print(f"\n  Non-#1 moves: {non_top}/{len(all_ranks)} ({non_top_pct:.1f}%)")

    # Critical moments
    if all_critical_counts:
        avg_critical = sum(all_critical_counts) / len(all_critical_counts)
        total_critical = sum(all_critical_counts)
        total_found = sum(all_critical_found)
        solve_rate = total_found / total_critical * 100 if total_critical > 0 else 0
        print(f"\n  Critical moments (gap>={config.gap_threshold} AND disagree>={config.disagreement_threshold}):")
        print(f"    Avg per game: {avg_critical:.1f}")
        print(f"    Total: {total_critical}")
        print(f"    Opponent found (rank 0): {total_found} ({solve_rate:.0f}%)")

    # Disagreement distribution
    if all_disagreements:
        disagree_counter = Counter(all_disagreements)
        print(f"\n  Disagreement distribution:")
        for d in sorted(disagree_counter.keys()):
            count = disagree_counter[d]
            pct = count / len(all_disagreements) * 100
            print(f"    Rank {d}: {count:>4} ({pct:>5.1f}%)")

    assert non_top_pct > 0, "Stonefish should play non-top moves sometimes!"
    print(f"\n  Self-play test passed OK")
    return True


def test_opponent_move_tracking(engine):
    """Test that opponent move inference works in play_game flow."""
    print("\n" + "=" * 60)
    print("TEST 6: Opponent Move Tracking")
    print("=" * 60)

    config = StonefishConfig(deep_depth=12, base_depth=6)
    stonefish = StonefishBot(engine, config)
    stockfish = PureStockfishBot(engine, depth=12)

    stonefish.start_game(chess.WHITE)
    board = chess.Board()

    for i in range(6):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            move = stonefish.choose_move(board)
        else:
            move = stockfish.choose_move(board)
        board.push(move)

    state = stonefish.end_game(board)
    print(f"\n  Our moves recorded: {len(state.move_results)}")
    print(f"  Opponent moves recorded: {len(state.opponent_results)}")

    assert len(state.move_results) == 3, f"Expected 3 our moves, got {len(state.move_results)}"
    assert len(state.opponent_results) == 3, f"Expected 3 opp moves, got {len(state.opponent_results)}"

    for i, opp in enumerate(state.opponent_results):
        print(f"    Opp move {i+1}: {opp.move_san} (rank {opp.move_rank}, "
              f"cost {opp.eval_cost:.2f}, critical={opp.was_critical})")

    print(f"\n  Opponent move tracking OK")
    return True


if __name__ == "__main__":
    print("Stonefish Test Suite")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    try:
        test_gap_and_disagreement(engine)
        test_disagreement_filter()
        test_adaptive_depth()
        test_two_pass_selection(engine)
        test_opponent_move_tracking(engine)

        num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        test_self_play(engine, num_games=num_games)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        engine.quit()

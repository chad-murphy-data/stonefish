"""
Stonefish Test Suite
====================
Tests the three-tier Maia puzzle detection system, adaptive depth,
move selection, presets, and self-play.

Run: python test_stonefish.py
"""

import chess
import chess.engine
import time
import sys
from collections import Counter

from engine import STOCKFISH_PATH, get_top_moves, PureStockfishBot
from stonefish.config import StonefishConfig
from stonefish.scoring import (
    detect_puzzle, detect_positive_puzzle, detect_mate_puzzle,
    rank_opponent_move, get_difficulty_label, get_eval_threshold,
    involves_material_difference, _score_puzzle_raw,
)
from stonefish.adaptive import AdaptiveDepthTracker
from stonefish.bot import StonefishBot
from stonefish.maia import MaiaEngine, MaiaPrediction
from stonefish.presets import ELO_PRESETS, get_nearest_preset, apply_preset


def test_presets():
    """Test that elo presets load and apply correctly."""
    print("=" * 60)
    print("TEST 1: Elo Presets")
    print("=" * 60)

    # Check all presets exist
    expected_elos = [500, 750, 1000, 1250, 1500, 1750, 2000]
    for elo in expected_elos:
        assert elo in ELO_PRESETS, f"Missing preset for {elo}"
        preset = ELO_PRESETS[elo]
        assert "floor_rating" in preset
        assert "stretch_rating" in preset
        assert "reach_rating" in preset
        assert "mate_retry_budget" in preset
        print(f"  {elo}: floor={preset['floor_rating']}, "
              f"stretch={preset['stretch_rating']}, "
              f"reach={preset['reach_rating']}")

    # Test nearest preset
    assert get_nearest_preset(600)["floor_rating"] == 500
    assert get_nearest_preset(1100)["floor_rating"] == 1000
    assert get_nearest_preset(1400)["floor_rating"] == 1500
    print(f"\n  Nearest preset lookup: OK")

    # Test apply_preset
    config = StonefishConfig()
    apply_preset(config, 1000)
    assert config.floor_rating == 1000
    assert config.stretch_rating == 1300
    assert config.reach_rating == 1700
    assert config.max_mate_depth == 3
    print(f"  Apply preset 1000: floor={config.floor_rating}, "
          f"stretch={config.stretch_rating}, reach={config.reach_rating}")

    # Test high-elo presets with None (Stockfish) tiers
    config2 = StonefishConfig()
    apply_preset(config2, 1750)
    assert config2.reach_rating is None, "1750 reach should be None (Stockfish)"
    print(f"  Apply preset 1750: reach={config2.reach_rating} (Stockfish)")

    config3 = StonefishConfig()
    apply_preset(config3, 2000)
    assert config3.stretch_rating is None, "2000 stretch should be None"
    assert config3.reach_rating is None, "2000 reach should be None"
    print(f"  Apply preset 2000: stretch={config3.stretch_rating}, "
          f"reach={config3.reach_rating} (both Stockfish)")

    print("\n  All preset tests passed!")
    return True


def test_maia_engine(engine):
    """Test that MaiaEngine produces predictions."""
    print("\n" + "=" * 60)
    print("TEST 2: Maia Engine (Simulated)")
    print("=" * 60)

    maia = MaiaEngine(stockfish_engine=engine)
    board = chess.Board()

    # Test prediction at various ratings
    for rating in [500, 1000, 1500, None]:
        pred = maia.predict(board, rating)
        label = str(rating) if rating else "Stockfish"
        print(f"\n  Rating {label}:")
        print(f"    Top move: {board.san(pred.top_move)}")
        print(f"    Distribution size: {len(pred.distribution)}")
        top3 = pred.top_n(3)
        for m in top3:
            prob = pred.prob_for(m)
            print(f"      {board.san(m)}: {prob:.1%}")
        assert pred.top_move in board.legal_moves

    # Test three-tier prediction
    tiers = maia.predict_three_tier(board, 1000, 1500, None)
    print(f"\n  Three-tier prediction:")
    print(f"    Floor (1000): {board.san(tiers.floor.top_move)}")
    print(f"    Stretch (1500): {board.san(tiers.stretch.top_move)}")
    print(f"    Reach (SF): {board.san(tiers.reach.top_move)}")

    print("\n  Maia engine tests passed!")
    return True


def test_difficulty_labels():
    """Test difficulty label assignment."""
    print("\n" + "=" * 60)
    print("TEST 3: Difficulty Labels + Scoring")
    print("=" * 60)

    assert get_difficulty_label(0.02) == "very hard"
    assert get_difficulty_label(0.10) == "hard"
    assert get_difficulty_label(0.25) == "moderate"
    assert get_difficulty_label(0.40) == "findable"
    print(f"  Difficulty labels: OK")

    # Test scoring
    score = _score_puzzle_raw(
        eval_gap=2.0, floor_prob=0.10, tiers_agreeing=2,
        is_positive=False, is_mate=False, generosity=0.5,
        eval_cost_to_create=0.3,
    )
    assert score > 0, f"Expected positive score, got {score}"
    print(f"  Puzzle scoring (negative, 2.0 gap, 10% floor): {score:.2f}")

    # Positive puzzle scaled by generosity
    score_pos = _score_puzzle_raw(
        eval_gap=2.0, floor_prob=0.10, tiers_agreeing=2,
        is_positive=True, is_mate=False, generosity=0.5,
        eval_cost_to_create=0.3,
    )
    assert score_pos < score, "Positive puzzle should score lower than negative"
    print(f"  Puzzle scoring (positive, generosity=0.5): {score_pos:.2f}")

    # Mate bonus
    score_mate = _score_puzzle_raw(
        eval_gap=2.0, floor_prob=0.10, tiers_agreeing=0,
        is_positive=False, is_mate=True, generosity=0.5,
        eval_cost_to_create=0.0,
    )
    assert score_mate > score, "Mate should score higher"
    print(f"  Puzzle scoring (mate): {score_mate:.2f}")

    print("\n  All scoring tests passed!")
    return True


def test_adaptive_depth():
    """Test the adaptive depth state machine (boost only, target_band=1)."""
    print("\n" + "=" * 60)
    print("TEST 4: Adaptive Depth Tracker")
    print("=" * 60)

    tracker = AdaptiveDepthTracker(
        base_depth=6, boost_on_miss=5,
        boost_duration=5, target_band=1,
    )

    assert tracker.current_depth == 6
    print(f"  Initial depth: {tracker.current_depth} OK")

    tracker.record_opponent_move(move_rank=0, was_critical=False, move_number=5)
    assert tracker.current_depth == 6
    print(f"  After non-critical: {tracker.current_depth} OK")

    tracker.record_opponent_move(move_rank=1, was_critical=True, move_number=10)
    assert tracker.current_depth == 11
    print(f"  After miss (rank 1): {tracker.current_depth} OK (boosted)")

    for i in range(5):
        tracker.tick()
    assert tracker.current_depth == 6
    print(f"  After 5 ticks: {tracker.current_depth} OK (expired)")

    tracker.record_opponent_move(move_rank=2, was_critical=True, move_number=25)
    assert tracker.current_depth == 11
    tracker.record_opponent_move(move_rank=0, was_critical=True, move_number=26)
    assert tracker.current_depth == 6
    print(f"  Find cancels boost: {tracker.current_depth} OK")

    print("\n  All adaptive depth tests passed!")
    return True


def test_move_selection(engine):
    """Test that move selection returns all expected fields."""
    print("\n" + "=" * 60)
    print("TEST 5: Move Selection (Three-Tier Maia)")
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
    print(f"  Puzzle found: {result.puzzle_found}")
    print(f"  Puzzle type: {result.puzzle_type}")
    print(f"  Puzzle floor prob: {result.puzzle_floor_prob:.0%}")
    print(f"  Puzzle disagreement: {result.puzzle_disagreement}")
    print(f"  Positions screened: {result.positions_screened}")
    print(f"  Emergency: {result.emergency_mode}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Candidates: {len(result.candidate_evals)}")

    assert result.move is not None
    assert result.deep_depth == 12
    assert result.current_shallow_depth == 6
    assert hasattr(result, 'puzzle_found')
    assert hasattr(result, 'puzzle_type')
    assert hasattr(result, 'puzzle_floor_prob')

    bot.end_game()
    print("\n  Move selection OK")
    return True


def test_opponent_tracking(engine):
    """Test that opponent move tracking works."""
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
              f"cost {opp.eval_cost:.2f}, puzzle={opp.was_puzzle})")

    print(f"\n  Opponent move tracking OK")
    return True


def test_self_play(engine, num_games=4):
    """Play Stonefish vs PureStockfish and print puzzle detection stats."""
    print("\n" + "=" * 60)
    print(f"TEST 7: Self-Play ({num_games} games, Stonefish vs PureStockfish)")
    print("=" * 60)

    config = StonefishConfig(deep_depth=12, base_depth=6)
    stonefish = StonefishBot(engine, config)
    stockfish_bot = PureStockfishBot(engine, depth=14)

    all_ranks = []
    all_puzzle_counts = []
    all_puzzle_solved = []
    all_puzzle_types = Counter()
    all_moves = []
    results = {"W": 0, "D": 0, "L": 0}

    for game_num in range(num_games):
        sf_is_white = game_num % 2 == 0
        if sf_is_white:
            white_bot, black_bot = stonefish, stockfish_bot
        else:
            white_bot, black_bot = stockfish_bot, stonefish

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
                if mr.puzzle_found:
                    all_puzzle_types[mr.puzzle_type] += 1

            puzzle_count = sum(1 for o in state.opponent_results if o.was_puzzle)
            solved_count = sum(1 for o in state.opponent_results
                               if o.was_puzzle and o.found_puzzle)
            all_puzzle_counts.append(puzzle_count)
            all_puzzle_solved.append(solved_count)

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
        print(f"  Game {game_num + 1}: Stonefish ({color_str}) - "
              f"{board_result} in {move_count} moves ({elapsed:.0f}s) "
              f"[puzzles: {puzzle_count}, solved: {solved_count}]")

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

    # Puzzle stats
    if all_puzzle_counts:
        avg_puzzles = sum(all_puzzle_counts) / len(all_puzzle_counts)
        total_puzzles = sum(all_puzzle_counts)
        total_solved = sum(all_puzzle_solved)
        solve_rate = total_solved / total_puzzles * 100 if total_puzzles > 0 else 0
        print(f"\n  Puzzles:")
        print(f"    Avg per game: {avg_puzzles:.1f}")
        print(f"    Total: {total_puzzles}")
        print(f"    Opponent solved: {total_solved} ({solve_rate:.0f}%)")
        if all_puzzle_types:
            print(f"    Types: {dict(all_puzzle_types)}")

    print(f"\n  Self-play test passed OK")
    return True


if __name__ == "__main__":
    print("Stonefish Test Suite (Three-Tier Maia Puzzle Detector)")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    try:
        test_presets()
        test_maia_engine(engine)
        test_difficulty_labels()
        test_adaptive_depth()
        test_move_selection(engine)
        test_opponent_tracking(engine)

        num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        test_self_play(engine, num_games=num_games)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        engine.quit()

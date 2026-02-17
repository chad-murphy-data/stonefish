"""
Stonefish Calibration Tool
==========================
Runs Stonefish against Lichess Maia bots with full logging, then analyzes
the results to produce a tuning report.

Usage:
    # Play 50 games against maia5
    python calibrate_maia.py --games 50 --opponents maia5

    # Play against multiple Maia bots
    python calibrate_maia.py --games 20 --opponents maia1 maia5 maia9

    # Analyze existing logs without playing
    python calibrate_maia.py --analyze game_logs/calibration

    # Custom Stonefish config
    python calibrate_maia.py --games 50 --opponents maia5 --base-depth 8 --band 3
"""

import os
import sys
import json
import glob
import time
import logging
import argparse
import threading
import webbrowser
import requests
from datetime import datetime
from collections import Counter, defaultdict

import chess
import chess.engine
import berserk

from engine import STOCKFISH_PATH
from stonefish.config import StonefishConfig
from stonefish.bot import StonefishBot
from stonefish.logger import StonefishLogger
from stonefish.chat import ChatEngine
from stonefish.presets import apply_preset, get_nearest_preset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate")

# Lichess game statuses that mean the game is over
TERMINAL_STATUSES = {"mate", "resign", "stalemate", "timeout",
                     "outoftime", "draw", "aborted", "noStart",
                     "cheat", "unknownFinish", "variantEnd"}


# ---------------------------------------------------------------------------
# Game playing (against Lichess bots)
# ---------------------------------------------------------------------------

def load_token() -> str:
    """Load Lichess API token."""
    token = os.environ.get("LICHESS_TOKEN", "").strip()
    if token:
        return token
    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lichess.token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    print("ERROR: No Lichess API token found. Save to lichess.token or set LICHESS_TOKEN.")
    sys.exit(1)


def play_calibration_game(client, bot_id, opponent, sf_engine, config, log_dir,
                          watch=False, token=None):
    """Challenge an opponent on Lichess and play one full game with logging.

    If watch=True, opens each game in the browser.
    Returns the game log filepath on success, None on failure.
    """
    # Challenge
    clock_limit = 600  # 10+5
    clock_increment = 5
    log.info(f"Challenging {opponent} (10+5)...")

    try:
        client.challenges.create(
            opponent, rated=False,
            clock_limit=clock_limit, clock_increment=clock_increment,
        )
    except berserk.exceptions.ResponseError as e:
        log.error(f"Failed to challenge {opponent}: {e}")
        return None

    # Wait for game to start using raw HTTP streaming with real timeout
    game_id = None
    log.info("Waiting for game to start (60s timeout)...")
    auth_header = {"Authorization": f"Bearer {token}"}

    try:
        # timeout=(connect_timeout, read_timeout) — read timeout means
        # if no data arrives for 60s, it raises a timeout exception
        resp = requests.get(
            "https://lichess.org/api/stream/event",
            headers=auth_header,
            stream=True,
            timeout=(10, 60),
        )
        resp.raise_for_status()

        deadline = time.time() + 60
        for line in resp.iter_lines():
            if time.time() > deadline:
                log.warning(f"Timed out waiting for {opponent} to accept (>60s)")
                resp.close()
                return None

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "gameStart":
                game_info = event.get("game", {})
                game_id = game_info.get("gameId", game_info.get("id", ""))
                # Verify this game is against the opponent we challenged
                game_opponent = game_info.get("opponent", {}).get("id", "")
                if game_opponent and game_opponent.lower() != opponent.lower():
                    log.info(f"Ignoring game {game_id} (vs {game_opponent}, expected {opponent})")
                    continue
                log.info(f"Game started: {game_id} (vs {game_opponent or opponent})")
                if watch:
                    url = f"https://lichess.org/{game_id}"
                    log.info(f"Watch at: {url}")
                    webbrowser.open(url)
                resp.close()
                break

            elif event_type == "challengeDeclined":
                log.warning(f"{opponent} declined the challenge")
                resp.close()
                return None

            elif event_type == "challenge":
                # Decline any incoming challenges while calibrating
                cid = event.get("challenge", {}).get("id", "")
                if cid:
                    try:
                        client.bots.decline_challenge(cid, reason="later")
                    except Exception:
                        pass

    except requests.exceptions.Timeout:
        log.warning(f"HTTP timeout waiting for {opponent} (>60s)")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"Stream error waiting for game: {e}")
        return None

    if not game_id:
        log.error("No game started")
        return None

    # Play the game
    bot = StonefishBot(sf_engine, config)
    logger = StonefishLogger(output_dir=log_dir)
    chat = ChatEngine(config)
    our_color = None
    board = chess.Board()
    last_processed = 0
    opponent_name = opponent

    game_status = "unknown"

    try:
        for event in client.bots.stream_game_state(game_id):
            event_type = event.get("type", "")
            log.debug(f"Event: {event_type} (keys: {list(event.keys())})")

            if event_type == "chatLine":
                # Handle chat — check for quiet/talk
                text = event.get("text", "")
                chat.handle_incoming_chat(text)
                continue

            elif event_type == "gameFull":
                white = event.get("white", {})
                white_id = white.get("id", white.get("name", ""))
                our_color = chess.WHITE if white_id == bot_id else chess.BLACK
                color_str = "white" if our_color == chess.WHITE else "black"
                log.info(f"Playing as {color_str} vs {opponent}")

                bot.start_game(our_color)
                logger.start_game(config, our_color, opponent_name)

                # Trigger 1: Onboarding message
                _send_chat(client, game_id, chat.on_game_start())

                state = event.get("state", {})
                game_status = state.get("status", "started")
                if game_status not in TERMINAL_STATUSES:
                    board, last_processed = _process_state(
                        client, game_id, state, our_color, bot, logger, chat,
                        sf_engine, config, board, last_processed, bot_id,
                    )
                else:
                    # Game already over (e.g., abort, etc.)
                    log.info(f"Game {game_id} already ended: {game_status}")

            elif event_type == "gameState":
                game_status = event.get("status", "started")
                moves_in_event = event.get("moves", "")
                num_moves_in_event = len(moves_in_event.split()) if moves_in_event else 0
                log.debug(f"gameState: status={game_status}, moves={num_moves_in_event}")

                if game_status not in TERMINAL_STATUSES:
                    # Game still in progress — process moves
                    board, last_processed = _process_state(
                        client, game_id, event, our_color, bot, logger, chat,
                        sf_engine, config, board, last_processed, bot_id,
                    )
                else:
                    # Game ended — rebuild final board state from moves
                    if moves_in_event:
                        board = chess.Board()
                        for uci_str in moves_in_event.split():
                            try:
                                board.push_uci(uci_str)
                            except ValueError:
                                break
                    log.info(f"Game {game_id} ended: {game_status} "
                             f"(after {num_moves_in_event} half-moves)")
                    break

    except Exception as e:
        import traceback
        log.error(f"Game {game_id}: error during play: {e}")
        log.error(f"Game {game_id}: traceback:\n{traceback.format_exc()}")

    # ALWAYS write logs, regardless of how the game ended
    log.info(f"Game {game_id}: finalizing (status={game_status}, "
             f"board_moves={len(board.move_stack)}, "
             f"game_over={board.is_game_over()})")
    # Determine result string from game status or board state
    if board.is_game_over():
        result_str = board.result()
    elif game_status in ("resign", "mate", "stalemate", "draw"):
        result_str = game_status
    elif game_status in ("outoftime", "timeout"):
        result_str = f"timeout"
    elif game_status == "aborted":
        result_str = "aborted"
    else:
        result_str = game_status if game_status != "started" else "*"

    # Trigger 5: Post-game summary (list of messages, sent with delay)
    game_state = bot.game_state
    if game_state:
        try:
            summary_messages = chat.generate_post_game_summary(game_state)
            for msg in summary_messages:
                _send_chat(client, game_id, msg)
                time.sleep(1)  # Avoid rate limiting / out-of-order
        except Exception as e:
            log.warning(f"Failed to send post-game summary: {e}")

    bot.end_game(board)
    filepath = logger.end_game(result_str, board)
    if filepath:
        log.info(f"Game {game_id}: {result_str} -- log saved to {filepath}")
    else:
        log.warning(f"Game {game_id}: {result_str} -- no log written (game may not have started)")
    return filepath


def _send_chat(client, game_id, msg):
    """Send a chat message, logging failures."""
    if not msg:
        return
    try:
        client.bots.post_message(game_id, msg)
        log.debug(f"Chat sent: {msg[:60]}...")
    except Exception as e:
        log.warning(f"Chat send failed: {e} (msg: {msg[:60]}...)")


def _process_state(client, game_id, state, our_color, bot, logger, chat,
                   sf_engine, config, board, last_processed, bot_id):
    """Process a game state, analyze opponent moves, play our move.

    Extracts clock times from the state event (wtime/btime in ms) and
    passes our remaining clock to bot.choose_move_full() for emergency mode.
    """
    moves_str = state.get("moves", "")
    board = chess.Board()
    move_list = []
    if moves_str:
        for uci_str in moves_str.split():
            try:
                move = board.push_uci(uci_str)
                move_list.append(move)
            except ValueError:
                log.error(f"Invalid move: {uci_str}")
                return board, last_processed

    # Don't process if board position is already terminal
    if board.is_game_over():
        return board, last_processed

    if our_color is None:
        return board, last_processed

    current_half = len(move_list)

    # Process new opponent moves
    for i in range(last_processed, current_half):
        move_color = chess.WHITE if (i % 2 == 0) else chess.BLACK
        if move_color != our_color:
            temp_board = chess.Board()
            for j in range(i):
                temp_board.push(move_list[j])
            opp_result = bot.notify_opponent_move(temp_board, move_list[i])
            if opp_result:
                logger.record_opponent_move(opp_result)
                _send_chat(client, game_id, chat.on_opponent_move(opp_result))

    last_processed = current_half

    # Our turn?
    if board.turn != our_color:
        return board, last_processed

    # Extract our clock time from the state event
    # berserk returns wtime/btime as datetime.timedelta objects (or sometimes ints in ms)
    our_clock_seconds = None
    raw_time = state.get("wtime") if our_color == chess.WHITE else state.get("btime")
    if raw_time is not None:
        try:
            # berserk parses these as timedelta objects
            our_clock_seconds = raw_time.total_seconds()
        except AttributeError:
            # Fallback: raw integer milliseconds (e.g., from raw API)
            our_clock_seconds = float(raw_time) / 1000.0

    move_num = board.fullmove_number
    clock_str = f" ({our_clock_seconds:.0f}s)" if our_clock_seconds is not None else ""
    log.info(f"Move {move_num}, thinking...{clock_str}")

    start = time.time()
    result = bot.choose_move_full(board, our_clock_seconds=our_clock_seconds)
    elapsed = time.time() - start

    logger.record_our_move(result)
    _send_chat(client, game_id, chat.on_our_move(result))

    emg_tag = " [EMERGENCY]" if result.emergency_mode else ""
    puzzle_flag = f" *** PUZZLE ({result.puzzle_type}) ***" if result.puzzle_found else ""
    log.info(f"Playing {result.move_san} (rank {result.move_rank}, "
             f"puzzle={result.puzzle_type}, "
             f"eval_gap={result.puzzle_eval_gap:.2f}, "
             f"floor_prob={result.puzzle_floor_prob:.0%}, "
             f"{elapsed:.1f}s)"
             f"{emg_tag}{puzzle_flag}")

    for attempt in range(3):
        try:
            client.bots.make_move(game_id, result.move.uci())
            break
        except Exception as e:
            log.warning(f"make_move attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1)
            else:
                log.error(f"Failed to make move after 3 attempts: {e}")

    last_processed += 1
    return board, last_processed


def run_calibration(opponents, num_games, config, log_dir, watch=False):
    """Run calibration games against Lichess bots."""
    token = load_token()
    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    account = client.account.get()
    bot_id = account.get("id", "")
    log.info(f"Logged in as: {bot_id}")

    # Start Stockfish
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf_engine.configure({
        "Threads": config.stockfish_threads,
        "Hash": config.stockfish_hash_mb,
    })

    os.makedirs(log_dir, exist_ok=True)
    results = []

    try:
        for opponent in opponents:
            log.info(f"\n{'='*60}")
            log.info(f"CALIBRATING vs {opponent} ({num_games} games)")
            log.info(f"{'='*60}")

            for game_num in range(num_games):
                log.info(f"\n--- {opponent} game {game_num + 1}/{num_games} ---")

                filepath = play_calibration_game(
                    client, bot_id, opponent, sf_engine, config, log_dir,
                    watch=watch, token=token,
                )

                if filepath:
                    results.append(filepath)
                    log.info(f"Progress: {len(results)} games logged")
                else:
                    log.warning(f"Game failed, retrying after 10s...")
                    time.sleep(10)
                    # Retry once
                    filepath = play_calibration_game(
                        client, bot_id, opponent, sf_engine, config, log_dir,
                        watch=watch, token=token,
                    )
                    if filepath:
                        results.append(filepath)

                # Delay between games
                if game_num < num_games - 1:
                    time.sleep(5)

    except KeyboardInterrupt:
        log.info("Calibration interrupted!")
    finally:
        sf_engine.quit()

    log.info(f"\nCalibration complete: {len(results)} games logged to {log_dir}")
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_calibration_logs(log_dir):
    """Read all game logs and produce a calibration report."""
    files = sorted(glob.glob(os.path.join(log_dir, "stonefish_game_*.json")))
    if not files:
        print(f"No game logs found in {log_dir}")
        return None

    all_games = []
    for f in files:
        with open(f) as fp:
            all_games.append(json.load(fp))

    print(f"\n{'='*70}")
    print(f"STONEFISH CALIBRATION REPORT")
    print(f"{'='*70}")
    print(f"Total games analyzed: {len(all_games)}")

    # Group by opponent
    by_opponent = defaultdict(list)
    for game in all_games:
        opp = game.get("metadata", {}).get("opponent", "unknown")
        by_opponent[opp].append(game)

    for opp, games in by_opponent.items():
        print(f"\n{'='*70}")
        print(f"  vs {opp} ({len(games)} games)")
        print(f"{'='*70}")
        _print_opponent_report(games)

    # Overall
    if len(by_opponent) > 1:
        print(f"\n{'='*70}")
        print(f"  OVERALL ({len(all_games)} games)")
        print(f"{'='*70}")
        _print_opponent_report(all_games)

    # Top 10 criticality positions
    print(f"\n{'='*70}")
    print(f"  TOP 10 HIGHEST-CRITICALITY POSITIONS")
    print(f"{'='*70}")
    _print_top_criticality(all_games)

    # Tuning recommendations
    print(f"\n{'='*70}")
    print(f"  TUNING RECOMMENDATIONS")
    print(f"{'='*70}")
    _print_recommendations(all_games)

    return all_games


def _print_opponent_report(games):
    """Print stats for a set of games."""
    total_puzzles = 0
    total_solved = 0
    total_eval_cost = 0.0
    all_ranks = []
    all_eval_gaps = []
    all_floor_probs = []
    puzzle_types = Counter()
    game_lengths = []
    wins = draws = losses = 0
    emergency_moves = 0
    total_sf_moves = 0
    mate_puzzles = 0

    for game in games:
        summary = game.get("summary", {})
        if not summary:
            continue

        puzzle_count = summary.get("puzzles_created", 0)
        solved_count = summary.get("puzzles_solved", 0)
        total_puzzles += puzzle_count
        total_solved += solved_count
        total_eval_cost += summary.get("total_eval_cost_of_misses", 0)
        game_lengths.append(summary.get("total_moves", 0))
        mate_puzzles += summary.get("mate_puzzles", 0)

        # Puzzle type breakdown
        for ptype, count in summary.get("puzzle_type_breakdown", {}).items():
            puzzle_types[ptype] += count

        result = summary.get("result", "*")
        our_color = game.get("metadata", {}).get("our_color", "white")
        if result == "1-0":
            if our_color == "white":
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if our_color == "black":
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

        # Rank distribution
        rank_dist = summary.get("stonefish_rank_distribution", {})
        for rank_str, count in rank_dist.items():
            for _ in range(count):
                all_ranks.append(int(rank_str))

        # Puzzle data from moves
        for move in game.get("moves", []):
            if move.get("side") == "stonefish":
                total_sf_moves += 1
                if move.get("puzzle_found"):
                    eg = move.get("puzzle_eval_gap", 0)
                    fp = move.get("puzzle_floor_prob", 0)
                    if eg:
                        all_eval_gaps.append(eg)
                    all_floor_probs.append(fp)
                if move.get("emergency_mode"):
                    emergency_moves += 1

    num_games = len(games)
    avg_puzzles = total_puzzles / num_games if num_games else 0
    solve_rate = total_solved / total_puzzles * 100 if total_puzzles else 0
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    print(f"\n  Results: {wins}W / {draws}D / {losses}L")
    print(f"  Win rate: {wins / num_games * 100:.0f}%")
    print(f"  Avg game length: {avg_length:.0f} moves")
    print(f"\n  Puzzles:")
    print(f"    Avg per game: {avg_puzzles:.1f} (target: 3-4)")
    print(f"    Total: {total_puzzles}")
    print(f"    Opponent solved: {total_solved} ({solve_rate:.0f}%)")
    print(f"    Total eval cost of misses: {total_eval_cost:.1f} pawns")
    print(f"    Mate puzzles: {mate_puzzles}")
    if puzzle_types:
        print(f"    Type breakdown: {dict(puzzle_types)}")

    # Emergency stats
    if total_sf_moves > 0:
        emg_pct = emergency_moves / total_sf_moves * 100
        print(f"\n  Performance:")
        print(f"    Emergency mode moves: {emergency_moves}/{total_sf_moves} ({emg_pct:.0f}%)")

    # Rank distribution
    if all_ranks:
        print(f"\n  Stonefish Move Rank Distribution:")
        rank_counter = Counter(all_ranks)
        for rank in sorted(rank_counter.keys()):
            count = rank_counter[rank]
            pct = count / len(all_ranks) * 100
            bar = "#" * int(pct / 2)
            print(f"    Rank {rank}: {count:>4} ({pct:>5.1f}%) {bar}")

        non_top = sum(1 for r in all_ranks if r > 0)
        print(f"    Non-#1 moves: {non_top}/{len(all_ranks)} ({non_top / len(all_ranks) * 100:.1f}%)")

    # Eval gap distribution for puzzles
    if all_eval_gaps:
        print(f"\n  Puzzle Eval Gap Distribution:")
        _print_histogram(all_eval_gaps, bucket_size=0.5, label="pawns")

    # Floor probability distribution
    if all_floor_probs:
        print(f"\n  Floor Probability Distribution (how often floor finds correct move):")
        _print_histogram(all_floor_probs, bucket_size=0.05, label="")



def _print_histogram(values, bucket_size=0.1, label=""):
    """Print a simple text histogram."""
    if not values:
        return
    min_val = min(values)
    max_val = max(values)
    buckets = defaultdict(int)
    for v in values:
        bucket = round(int(v / bucket_size) * bucket_size, 2)
        buckets[bucket] += 1

    max_count = max(buckets.values()) if buckets else 1
    for bucket in sorted(buckets.keys()):
        count = buckets[bucket]
        bar_len = int(count / max_count * 30)
        bar = "#" * bar_len
        print(f"    {bucket:>5.1f} {label}: {count:>4} {bar}")


def _print_top_criticality(all_games):
    """Find and print the top 10 highest-scoring puzzle positions."""
    positions = []
    for game in all_games:
        opponent = game.get("metadata", {}).get("opponent", "?")
        for move in game.get("moves", []):
            if move.get("side") == "stonefish" and move.get("puzzle_found"):
                positions.append({
                    "opponent": opponent,
                    "move_number": move.get("move_number", 0),
                    "move": move.get("stonefish_move", "?"),
                    "puzzle_type": move.get("puzzle_type", "?"),
                    "eval_gap": move.get("puzzle_eval_gap", 0),
                    "floor_prob": move.get("puzzle_floor_prob", 0),
                    "disagreement": move.get("puzzle_disagreement", "?"),
                    "score": move.get("puzzle_score", 0),
                })

    positions.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Found {len(positions)} puzzle positions total.\n")
    for i, p in enumerate(positions[:10]):
        print(f"  {i+1}. vs {p['opponent']} move {p['move_number']}: "
              f"{p['move']} ({p['puzzle_type']}, "
              f"eval_gap={p['eval_gap']:.2f}, "
              f"floor={p['floor_prob']:.0%}, "
              f"tiers={p['disagreement']}, "
              f"score={p['score']:.2f})")


def _print_recommendations(all_games):
    """Generate tuning recommendations based on calibration data."""
    total_puzzles = 0
    total_solved = 0
    num_games = len(all_games)

    for game in all_games:
        summary = game.get("summary", {})
        total_puzzles += summary.get("puzzles_created", 0)
        total_solved += summary.get("puzzles_solved", 0)

    avg_puzzles = total_puzzles / num_games if num_games else 0
    solve_rate = total_solved / total_puzzles * 100 if total_puzzles else 0

    print()

    # Puzzle frequency (target: 3-4 per game)
    if avg_puzzles < 2:
        print(f"  * Puzzles avg {avg_puzzles:.1f}/game (below target 3-4).")
        print(f"    -> Consider LOWERING puzzle_eval_threshold (currently 1.0)")
        print(f"       Try 0.7 to detect more puzzles.")
        print(f"    -> Or increase max_lookahead to search deeper.")
    elif avg_puzzles > 6:
        print(f"  * Puzzles avg {avg_puzzles:.1f}/game (above target 3-4).")
        print(f"    -> Consider RAISING puzzle_eval_threshold")
        print(f"       Try 1.2 to be more selective.")
    else:
        print(f"  * Puzzles avg {avg_puzzles:.1f}/game -- IN TARGET RANGE (3-4)")

    # Solve rate (target: 20-40%)
    if total_puzzles > 0:
        if solve_rate > 50:
            print(f"\n  * Solve rate {solve_rate:.0f}% (above target 20-40%).")
            print(f"    -> Puzzles may be too easy. Raise puzzle_eval_threshold or")
            print(f"       widen the gap between Floor and Stretch tiers.")
        elif solve_rate < 15:
            print(f"\n  * Solve rate {solve_rate:.0f}% (below target 20-40%).")
            print(f"    -> Puzzles may be too hard. Lower puzzle_eval_threshold or")
            print(f"       bring Stretch tier closer to Floor tier.")
        else:
            print(f"\n  * Solve rate {solve_rate:.0f}% -- IN TARGET RANGE (20-40%)")

    print(f"\n  NOTE: These are recommendations only. Review the data and decide.")
    print(f"  Use --elo-preset to auto-configure tiers for a target rating.")
    print(f"  All parameters are in stonefish/config.py (StonefishConfig).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stonefish Calibration Tool")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per opponent (default: 50)")
    parser.add_argument("--opponents", nargs="*", default=["maia5"],
                        help="Bot usernames to play against (default: maia5)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to log dir for analysis only (no games played)")
    parser.add_argument("--log-dir", type=str, default="game_logs/calibration",
                        help="Directory for game logs (default: game_logs/calibration)")
    parser.add_argument("--watch", action="store_true",
                        help="Open each game in the browser to watch live")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (verbose event tracing)")

    # Elo preset (configures everything)
    parser.add_argument("--elo-preset", type=int, default=None,
                        help="Elo preset (500/750/1000/1250/1500/1750/2000)")

    # Individual overrides
    parser.add_argument("--floor-rating", type=int, default=None)
    parser.add_argument("--stretch-rating", type=int, default=None)
    parser.add_argument("--reach-rating", type=int, default=None)
    parser.add_argument("--puzzle-eval-threshold", type=float, default=None)
    parser.add_argument("--soft-eval-threshold", type=float, default=None)
    parser.add_argument("--max-mate-depth", type=int, default=None)
    parser.add_argument("--generosity", type=float, default=None)
    parser.add_argument("--max-lookahead", type=int, default=None)
    parser.add_argument("--enable-positive", action="store_true", default=None)
    parser.add_argument("--no-positive", action="store_true", default=False)

    # Engine config
    parser.add_argument("--deep-depth", type=int, default=12)
    parser.add_argument("--base-depth", type=int, default=6)
    parser.add_argument("--emergency-clock", type=int, default=60)
    parser.add_argument("--opening-book-moves", type=int, default=5)
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("calibrate").setLevel(logging.DEBUG)

    if args.analyze:
        analyze_calibration_logs(args.analyze)
        return

    # Build config: start with defaults, apply preset, then individual overrides
    config = StonefishConfig(
        deep_depth=args.deep_depth,
        base_depth=args.base_depth,
        emergency_clock_seconds=args.emergency_clock,
        opening_book_moves=args.opening_book_moves,
    )

    if args.elo_preset is not None:
        apply_preset(config, args.elo_preset)

    # Individual overrides
    if args.floor_rating is not None:
        config.floor_rating = args.floor_rating
    if args.stretch_rating is not None:
        config.stretch_rating = args.stretch_rating
    if args.reach_rating is not None:
        config.reach_rating = args.reach_rating
    if args.puzzle_eval_threshold is not None:
        config.puzzle_eval_threshold = args.puzzle_eval_threshold
    if args.soft_eval_threshold is not None:
        config.soft_puzzle_eval_threshold = args.soft_eval_threshold
    if args.max_mate_depth is not None:
        config.max_mate_depth = args.max_mate_depth
    if args.generosity is not None:
        config.puzzle_generosity = args.generosity
    if args.max_lookahead is not None:
        config.max_lookahead = args.max_lookahead
    if args.enable_positive is not None:
        config.enable_positive_puzzles = True
    if args.no_positive:
        config.enable_positive_puzzles = False

    print(f"Stonefish Calibration (Three-Tier Maia Puzzle Detector)")
    print(f"  Opponents: {', '.join(args.opponents)}")
    print(f"  Games per opponent: {args.games}")
    if args.elo_preset:
        print(f"  Elo preset: {args.elo_preset}")
    print(f"  Tiers: floor={config.floor_rating}, "
          f"stretch={config.stretch_rating}, reach={config.reach_rating}")
    print(f"  Puzzle: eval_threshold={config.puzzle_eval_threshold}, "
          f"generosity={config.puzzle_generosity}, "
          f"mate_depth={config.max_mate_depth}, "
          f"lookahead={config.max_lookahead}")
    print(f"  Engine: deep={config.deep_depth}, base={config.base_depth}")
    print(f"  Clock: emergency<{config.emergency_clock_seconds}s")
    print(f"  Logs: {args.log_dir}")
    print()

    filepaths = run_calibration(args.opponents, args.games, config, args.log_dir,
                               watch=args.watch)

    if filepaths:
        print(f"\n{'='*70}")
        print(f"ANALYZING {len(filepaths)} GAMES")
        print(f"{'='*70}")
        analyze_calibration_logs(args.log_dir)


if __name__ == "__main__":
    main()

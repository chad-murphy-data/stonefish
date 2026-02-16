"""
Challenge a bot on Lichess and open the game in your browser to watch.

Usage:
    # Challenge maia1 (default)
    python challenge_and_watch.py

    # Challenge a specific bot
    python challenge_and_watch.py maia9

    # With custom time control (minutes+seconds)
    python challenge_and_watch.py maia5 --clock 5+3

    # Challenge multiple Maia bots in sequence
    python challenge_and_watch.py maia1 maia5 maia9
"""

import os
import sys
import time
import logging
import argparse
import webbrowser
import threading

import chess
import chess.engine
import berserk

from engine import NettlesomeBot, STOCKFISH_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nettlesome")


def load_token():
    token = os.environ.get("LICHESS_TOKEN", "").strip()
    if token:
        return token
    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lichess.token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    print("ERROR: No token found. Save it to lichess.token or set LICHESS_TOKEN.")
    sys.exit(1)


def play_game(client, game_id, bot_id, move_chooser):
    """Play a single game, return the result."""
    board = chess.Board()
    our_color = None
    result = None

    for event in client.bots.stream_game_state(game_id):
        event_type = event.get("type", "")

        if event_type == "gameFull":
            white = event.get("white", {})
            white_id = white.get("id", white.get("name", ""))
            our_color = chess.WHITE if white_id == bot_id else chess.BLACK
            color_str = "white" if our_color == chess.WHITE else "black"

            opponent = event.get("black" if our_color == chess.WHITE else "white", {})
            opp_name = opponent.get("id", opponent.get("name", "?"))
            log.info(f"Game {game_id}: playing as {color_str} vs {opp_name}")

            try:
                client.bots.post_message(game_id, "Nettlesome bot: I play the hardest move to answer. GL!")
            except Exception:
                pass

            state = event.get("state", {})
            board, result = process_state(client, game_id, state, our_color, move_chooser)
            if result:
                return result

        elif event_type == "gameState":
            board, result = process_state(client, game_id, event, our_color, move_chooser)
            if result:
                return result

    return result or "unknown"


def process_state(client, game_id, state, our_color, move_chooser):
    """Process a game state update. Returns (board, result_or_None)."""
    moves_str = state.get("moves", "")
    board = chess.Board()
    if moves_str:
        for uci_str in moves_str.split():
            board.push_uci(uci_str)

    status = state.get("status", "started")
    if status != "started":
        winner = state.get("winner", "")
        if status == "mate" or status == "resign":
            if winner == "white":
                result = "1-0"
            elif winner == "black":
                result = "0-1"
            else:
                result = status
        elif status == "draw" or status == "stalemate":
            result = "1/2-1/2"
        else:
            result = status
        log.info(f"Game {game_id}: ended — {result}")
        return board, result

    if board.is_game_over():
        result = board.result()
        log.info(f"Game {game_id}: game over — {result}")
        return board, result

    if our_color is None or board.turn != our_color:
        return board, None

    move_num = board.fullmove_number
    log.info(f"Game {game_id}: move {move_num}, thinking...")

    start = time.time()
    move = move_chooser.get_move(board)
    elapsed = time.time() - start

    san = board.san(move)
    uci = move.uci()
    log.info(f"Game {game_id}: playing {san} ({uci}) [{elapsed:.1f}s]")

    try:
        client.bots.make_move(game_id, uci)
    except berserk.exceptions.ResponseError as e:
        log.error(f"Game {game_id}: failed to make move: {e}")

    return board, None


def challenge_and_play(client, bot_id, opponent, move_chooser, clock_limit, clock_increment):
    """Send a challenge, wait for it to start, play the game."""
    log.info(f"Challenging {opponent} ({clock_limit//60}+{clock_increment})...")

    try:
        resp = client.challenges.create(
            opponent, rated=False,
            clock_limit=clock_limit, clock_increment=clock_increment,
        )
    except berserk.exceptions.ResponseError as e:
        log.error(f"Failed to challenge {opponent}: {e}")
        return None

    # Wait for game to start via event stream
    log.info("Waiting for game to start...")
    game_id = None

    for event in client.bots.stream_incoming_events():
        event_type = event.get("type", "")

        if event_type == "gameStart":
            game_info = event.get("game", {})
            game_id = game_info.get("gameId", game_info.get("id", ""))
            log.info(f"Game started: {game_id}")

            # Open in browser
            url = f"https://lichess.org/{game_id}"
            log.info(f"Watch at: {url}")
            webbrowser.open(url)

            # Play the game
            result = play_game(client, game_id, bot_id, move_chooser)
            return {"opponent": opponent, "game_id": game_id, "result": result}

        elif event_type == "challengeDeclined":
            log.warning(f"{opponent} declined the challenge")
            return None

        elif event_type == "gameFinish":
            # Might be a different game, keep listening
            pass

    return None


class MoveChooser:
    def __init__(self, depth):
        self.sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.sf_engine.configure({"Threads": 2, "Hash": 256})
        self.bot = NettlesomeBot(
            self.sf_engine, num_candidates=7, num_responses=3,
            depth=depth, max_eval_cost=1.0,
        )
        self._lock = threading.Lock()

    def get_move(self, board):
        with self._lock:
            return self.bot.choose_move(board)

    def stop(self):
        self.sf_engine.quit()


def main():
    parser = argparse.ArgumentParser(description="Challenge bots and watch StonefishBot play")
    parser.add_argument("opponents", nargs="*", default=["maia1"],
                        help="Bot username(s) to challenge (default: maia1)")
    parser.add_argument("--clock", type=str, default="10+5",
                        help="Time control (default: 10+5)")
    parser.add_argument("--depth", type=int, default=12,
                        help="Engine depth (default: 12)")
    parser.add_argument("--games", type=int, default=1,
                        help="Games per opponent (default: 1)")
    args = parser.parse_args()

    parts = args.clock.split("+")
    clock_limit = int(parts[0]) * 60
    clock_increment = int(parts[1]) if len(parts) > 1 else 0

    token = load_token()
    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    account = client.account.get()
    bot_id = account.get("id", "")
    log.info(f"Logged in as: {bot_id}")

    move_chooser = MoveChooser(args.depth)

    results = []
    try:
        for opponent in args.opponents:
            for game_num in range(args.games):
                if len(args.opponents) > 1 or args.games > 1:
                    log.info(f"\n--- {opponent} game {game_num + 1}/{args.games} ---")

                result = challenge_and_play(
                    client, bot_id, opponent, move_chooser,
                    clock_limit, clock_increment,
                )
                if result:
                    results.append(result)

                # Small delay between games
                if game_num < args.games - 1 or opponent != args.opponents[-1]:
                    time.sleep(3)

        # Summary
        if results:
            print(f"\n{'='*50}")
            print(f"RESULTS")
            print(f"{'='*50}")
            for r in results:
                print(f"  vs {r['opponent']}: {r['result']}  (lichess.org/{r['game_id']})")

    except KeyboardInterrupt:
        log.info("Interrupted!")
    finally:
        move_chooser.stop()


if __name__ == "__main__":
    main()

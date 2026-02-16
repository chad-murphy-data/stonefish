"""
Stonefish Lichess Bot
=====================
Connects the Stonefish engine to Lichess for live games with chat
and critical moment detection.

Usage:
    # Listen for challenges and play games
    python lichess_bot.py

    # Challenge a specific player
    python lichess_bot.py --challenge USERNAME

    # With custom config
    python lichess_bot.py --base-depth 8 --deep-depth 18 --band 3

Setup:
    1. Create a Lichess BOT account (or upgrade an existing one)
    2. Generate an API token at https://lichess.org/account/oauth/token
       - Enable scopes: bot:play, challenge:read, challenge:write
    3. Save the token to lichess.token in this directory
       OR set the LICHESS_TOKEN environment variable
"""

import os
import sys
import time
import logging
import argparse
import threading
from queue import Queue
from contextlib import contextmanager

import chess
import chess.engine
import berserk

from typing import Optional

from engine import STOCKFISH_PATH
from stonefish.config import StonefishConfig
from stonefish.bot import StonefishBot
from stonefish.logger import StonefishLogger
from stonefish.chat import ChatEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stonefish")


# ---------------------------------------------------------------------------
# Stockfish Pool -- manages N engine instances for concurrent games
# ---------------------------------------------------------------------------

class StockfishPool:
    """Thread-safe pool of Stockfish engine instances.

    Each game acquires an engine for the duration of a move computation,
    then returns it. This allows N concurrent games with N engines.
    """

    def __init__(self, path: str, num_engines: int = 3,
                 threads: int = 2, hash_mb: int = 256):
        self._engines: Queue = Queue()
        self._num_engines = num_engines
        for i in range(num_engines):
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": threads, "Hash": hash_mb})
            self._engines.put(engine)
            log.info(f"Stockfish engine {i+1}/{num_engines} ready")

    @contextmanager
    def acquire(self):
        """Acquire an engine for use. Blocks if all are busy."""
        engine = self._engines.get()
        try:
            yield engine
        finally:
            self._engines.put(engine)

    def shutdown(self):
        """Stop all engines."""
        while not self._engines.empty():
            try:
                engine = self._engines.get_nowait()
                engine.quit()
            except Exception:
                pass
        log.info("All Stockfish engines stopped")


# ---------------------------------------------------------------------------
# Game Handler -- manages a single Lichess game
# ---------------------------------------------------------------------------

class GameHandler:
    """Handles a single Lichess game in its own thread.

    Each game gets its own StonefishBot (with independent game state
    and adaptive depth), StonefishLogger, and ChatEngine.
    """

    def __init__(self, client: berserk.Client, game_id: str, bot_id: str,
                 pool: StockfishPool, config: StonefishConfig):
        self.client = client
        self.game_id = game_id
        self.bot_id = bot_id
        self.pool = pool
        self.config = config
        self.board = chess.Board()
        self.our_color = None
        self.opponent_name = "unknown"

        # Per-game instances
        self.bot = StonefishBot(engine=None, config=config)
        self.logger = StonefishLogger()
        self.chat = ChatEngine(config)

        # Track which moves we've already processed
        self._last_processed_moves = 0

    def run(self):
        """Main game loop -- stream events and respond."""
        log.info(f"Game {self.game_id}: streaming game state...")
        backoff = 1
        while True:
            try:
                for event in self.client.bots.stream_game_state(self.game_id):
                    backoff = 1
                    event_type = event.get("type", "")
                    if event_type == "gameFull":
                        self._handle_game_full(event)
                    elif event_type == "gameState":
                        self._apply_state(event)
                    elif event_type == "chatLine":
                        self._handle_chat(event)
                    elif event_type == "opponentGone":
                        log.info(f"Game {self.game_id}: opponent disconnected")
                break  # Stream ended cleanly
            except berserk.exceptions.ResponseError as e:
                log.error(f"Game {self.game_id}: API error: {e}")
                break
            except Exception as e:
                log.warning(f"Game {self.game_id}: stream error: {e}")
                log.info(f"Game {self.game_id}: reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        # Game over -- send post-game summary and write log
        self._on_game_end()
        log.info(f"Game {self.game_id}: handler finished")

    def _handle_game_full(self, event):
        """Process the initial gameFull event."""
        white = event.get("white", {})
        black = event.get("black", {})
        white_id = white.get("id", white.get("name", ""))
        black_id = black.get("id", black.get("name", ""))

        self.our_color = chess.WHITE if white_id == self.bot_id else chess.BLACK
        self.opponent_name = black_id if self.our_color == chess.WHITE else white_id
        color_str = "white" if self.our_color == chess.WHITE else "black"
        log.info(f"Game {self.game_id}: {white_id} (W) vs {black_id} (B) -- we are {color_str}")

        # Initialize per-game state
        self.bot.start_game(self.our_color)
        self.logger.start_game(self.config, self.our_color, self.opponent_name)

        # Send onboarding message
        onboarding = self.chat.on_game_start()
        self._send_chat(onboarding)

        # Apply initial state
        state = event.get("state", {})
        self._apply_state(state)

    def _apply_state(self, state):
        """Process a game state update -- rebuild board, detect new moves, play if our turn."""
        moves_str = state.get("moves", "")

        # Rebuild board from move list
        self.board = chess.Board()
        move_list = []
        if moves_str:
            for uci_str in moves_str.split():
                try:
                    move = self.board.push_uci(uci_str)
                    move_list.append(move)
                except ValueError:
                    log.error(f"Game {self.game_id}: invalid move: {uci_str}")
                    return

        # Check if game is over
        status = state.get("status", "started")
        if status != "started" or self.board.is_game_over():
            return

        if self.our_color is None:
            return

        current_half_moves = len(move_list)

        # Process any new opponent moves we haven't seen
        if current_half_moves > self._last_processed_moves:
            # Find opponent moves in the new portion
            for i in range(self._last_processed_moves, current_half_moves):
                move = move_list[i]
                # Determine whose move this was
                # Move at index 0 is white's first move, index 1 is black's, etc.
                move_color = chess.WHITE if (i % 2 == 0) else chess.BLACK

                if move_color != self.our_color:
                    # This is an opponent move -- analyze it
                    # Build board state BEFORE this move
                    temp_board = chess.Board()
                    for j in range(i):
                        temp_board.push(move_list[j])

                    with self.pool.acquire() as engine:
                        self.bot.engine = engine
                        opp_result = self.bot.notify_opponent_move(temp_board, move)
                        self.bot.engine = None

                    if opp_result:
                        self.logger.record_opponent_move(opp_result)
                        # Chat response to opponent's move
                        chat_msg = self.chat.on_opponent_move(opp_result)
                        if chat_msg:
                            self._send_chat(chat_msg)

            self._last_processed_moves = current_half_moves

        # Is it our turn?
        if self.board.turn != self.our_color:
            return

        # Our turn -- compute and play
        move_num = self.board.fullmove_number
        log.info(f"Game {self.game_id}: move {move_num}, thinking...")

        start = time.time()
        with self.pool.acquire() as engine:
            self.bot.engine = engine
            result = self.bot.choose_move_full(self.board)
            self.bot.engine = None
        elapsed = time.time() - start

        # Log our move
        self.logger.record_our_move(result)

        # Chat about criticality of the resulting position
        chat_msg = self.chat.on_our_move(result)
        if chat_msg:
            self._send_chat(chat_msg)

        san = result.move_san
        uci = result.move.uci()
        log.info(f"Game {self.game_id}: playing {san} ({uci}) "
                 f"[rank {result.move_rank}, nettl={result.nettlesomeness_score:.2f}, "
                 f"crit={result.criticality_score:.2f}, {elapsed:.1f}s]")

        try:
            self.client.bots.make_move(self.game_id, uci)
        except berserk.exceptions.ResponseError as e:
            log.error(f"Game {self.game_id}: failed to make move {uci}: {e}")

        self._last_processed_moves += 1  # We just played a move

    def _handle_chat(self, event):
        """Process incoming chat messages."""
        username = event.get("username", "")
        text = event.get("text", "")
        room = event.get("room", "player")

        if username == self.bot_id:
            return  # Ignore our own messages

        log.info(f"Game {self.game_id}: chat from {username}: {text}")
        self.chat.handle_incoming_chat(text)

    def _on_game_end(self):
        """Send post-game summary and write log file."""
        if self.bot.game_state is None:
            return

        # Generate post-game summary
        game_state = self.bot.game_state
        summary = self.chat.generate_post_game_summary(game_state)
        if summary:
            self._send_chat(summary)

        # Determine result
        result_str = self.board.result() if self.board.is_game_over() else "*"

        # End bot game state
        self.bot.end_game(self.board)

        # Write log
        filepath = self.logger.end_game(result_str, self.board)
        if filepath:
            log.info(f"Game {self.game_id}: log saved to {filepath}")

    def _send_chat(self, message: str):
        """Send a chat message to the player room."""
        if not message:
            return
        try:
            # Lichess chat has a character limit; split long messages
            for chunk in self._split_message(message, max_len=400):
                self.client.bots.post_message(self.game_id, chunk)
        except Exception as e:
            log.warning(f"Game {self.game_id}: chat failed: {e}")

    @staticmethod
    def _split_message(message: str, max_len: int = 400) -> list:
        """Split a message into chunks that fit Lichess chat limits."""
        if len(message) <= max_len:
            return [message]
        lines = message.split("\n")
        chunks = []
        current = ""
        for line in lines:
            if current and len(current) + len(line) + 1 > max_len:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        return chunks


# ---------------------------------------------------------------------------
# Main Bot -- listens for events, manages games
# ---------------------------------------------------------------------------

class StonefishLichessBot:
    """Main bot that listens for Lichess events and manages concurrent games."""

    SUPPORTED_VARIANTS = {"standard", "fromPosition"}

    def __init__(self, token: str, config: Optional[StonefishConfig] = None,
                 rated_only: bool = False, casual_only: bool = False):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session)
        self.config = config or StonefishConfig()
        self.rated_only = rated_only
        self.casual_only = casual_only
        self.pool = None
        self.active_games: dict = {}
        self.bot_id = None

    def start(self):
        """Start the bot: init engines, connect, listen for events."""
        account = self.client.account.get()
        self.bot_id = account.get("id", "")
        title = account.get("title", "")
        log.info(f"Logged in as: {self.bot_id} (title: {title})")

        if title != "BOT":
            log.warning("Account is NOT a BOT account! Upgrade at lichess.org or via API.")

        # Start engine pool
        self.pool = StockfishPool(
            STOCKFISH_PATH,
            num_engines=self.config.max_concurrent_games,
            threads=self.config.stockfish_threads,
            hash_mb=self.config.stockfish_hash_mb,
        )

        try:
            self._event_loop_with_reconnect()
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self.pool.shutdown()

    def _event_loop_with_reconnect(self):
        """Event loop with automatic reconnection."""
        backoff = 1
        while True:
            try:
                self._event_loop()
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.warning(f"Event stream disconnected: {e}")
                log.info(f"Reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                log.info("Reconnecting to event stream...")

    def _event_loop(self):
        """Main event loop -- process challenges, game starts, etc."""
        log.info("Listening for incoming events...")
        for event in self.client.bots.stream_incoming_events():
            event_type = event.get("type", "")
            if event_type == "challenge":
                self._handle_challenge(event.get("challenge", {}))
            elif event_type == "challengeCanceled":
                log.info("Challenge canceled")
            elif event_type == "gameStart":
                self._handle_game_start(event.get("game", {}))
            elif event_type == "gameFinish":
                self._handle_game_finish(event.get("game", {}))

    def _handle_challenge(self, challenge):
        """Accept or decline an incoming challenge."""
        challenge_id = challenge.get("id", "")
        challenger = challenge.get("challenger", {}).get("id", "?")
        variant = challenge.get("variant", {}).get("key", "standard")
        rated = challenge.get("rated", False)
        speed = challenge.get("speed", "?")
        time_control = challenge.get("timeControl", {})
        tc_limit = time_control.get("limit", 0)

        log.info(f"Challenge from {challenger}: {variant} {speed} rated={rated} limit={tc_limit}s")

        # Check variant
        if variant not in self.SUPPORTED_VARIANTS:
            log.info(f"Declining: unsupported variant {variant}")
            self._decline(challenge_id, "variant")
            return

        # Check time control -- minimum 10+0
        if tc_limit < self.config.min_time_control_seconds:
            log.info(f"Declining: time control too fast ({tc_limit}s < {self.config.min_time_control_seconds}s)")
            self._decline(challenge_id, "timeControl")
            return

        # Check rated/casual preference
        if self.rated_only and not rated:
            log.info("Declining: casual (rated-only mode)")
            self._decline(challenge_id, "casual")
            return

        if self.casual_only and rated:
            log.info("Declining: rated (casual-only mode)")
            self._decline(challenge_id, "rated")
            return

        # Check concurrent game limit
        active_count = sum(1 for t in self.active_games.values() if t.is_alive())
        if active_count >= self.config.max_concurrent_games:
            log.info(f"Declining: already playing {active_count} game(s)")
            self._decline(challenge_id, "later")
            return

        log.info(f"Accepting challenge from {challenger}")
        try:
            self.client.bots.accept_challenge(challenge_id)
        except berserk.exceptions.ResponseError as e:
            log.error(f"Failed to accept challenge: {e}")

    def _decline(self, challenge_id: str, reason: str):
        """Decline a challenge with a reason."""
        try:
            self.client.bots.decline_challenge(challenge_id, reason=reason)
        except Exception as e:
            log.warning(f"Failed to decline challenge: {e}")

    def _handle_game_start(self, game_info):
        """Spawn a game handler thread for a new game."""
        game_id = game_info.get("gameId", game_info.get("id", ""))
        if not game_id:
            return

        log.info(f"Game started: {game_id}")
        handler = GameHandler(
            self.client, game_id, self.bot_id,
            self.pool, self.config,
        )
        thread = threading.Thread(
            target=handler.run, name=f"game-{game_id}", daemon=True,
        )
        self.active_games[game_id] = thread
        thread.start()

    def _handle_game_finish(self, game_info):
        """Clean up after a game ends."""
        game_id = game_info.get("gameId", game_info.get("id", ""))
        log.info(f"Game finished: {game_id}")
        self.active_games.pop(game_id, None)

    def challenge_player(self, username: str, rated: bool = False,
                         clock_limit: int = 600, clock_increment: int = 5):
        """Send a challenge to a player."""
        log.info(f"Challenging {username} ({clock_limit}s+{clock_increment}s, rated={rated})")
        try:
            self.client.challenges.create(
                username, rated=rated,
                clock_limit=clock_limit, clock_increment=clock_increment,
            )
            log.info("Challenge sent!")
        except berserk.exceptions.ResponseError as e:
            log.error(f"Failed to challenge {username}: {e}")


# ---------------------------------------------------------------------------
# Token loading and CLI
# ---------------------------------------------------------------------------

def load_token() -> str:
    """Load Lichess API token from env or file."""
    token = os.environ.get("LICHESS_TOKEN", "").strip()
    if token:
        return token

    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lichess.token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            token = f.read().strip()
        if token:
            return token

    print("ERROR: No Lichess API token found!")
    print()
    print("Either:")
    print("  1. Set the LICHESS_TOKEN environment variable")
    print("  2. Save your token to lichess.token in this directory")
    print()
    print("Get a token at: https://lichess.org/account/oauth/token")
    print("Required scopes: bot:play, challenge:read, challenge:write")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Stonefish Lichess Bot")
    parser.add_argument("--challenge", type=str, default=None,
                        help="Challenge a specific player")
    parser.add_argument("--clock", type=str, default="10+5",
                        help="Time control for outgoing challenges (e.g. '15+10')")
    parser.add_argument("--rated", action="store_true",
                        help="Make outgoing challenges rated")
    parser.add_argument("--rated-only", action="store_true",
                        help="Only accept rated challenges")
    parser.add_argument("--casual-only", action="store_true",
                        help="Only accept casual challenges")

    # Stonefish config overrides
    parser.add_argument("--deep-depth", type=int, default=18,
                        help="Deep pass depth (default: 18)")
    parser.add_argument("--base-depth", type=int, default=6,
                        help="Base shallow depth (default: 6)")
    parser.add_argument("--band", type=int, default=5,
                        help="Target band size (default: 5)")
    parser.add_argument("--max-games", type=int, default=3,
                        help="Max concurrent games (default: 3)")
    args = parser.parse_args()

    config = StonefishConfig(
        deep_depth=args.deep_depth,
        base_depth=args.base_depth,
        target_band=args.band,
        max_concurrent_games=args.max_games,
    )

    token = load_token()
    bot = StonefishLichessBot(
        token=token, config=config,
        rated_only=args.rated_only, casual_only=args.casual_only,
    )

    if args.challenge:
        parts = args.clock.split("+")
        clock_limit = int(parts[0]) * 60
        clock_increment = int(parts[1]) if len(parts) > 1 else 0

        # For challenge mode, start the pool manually
        bot.pool = StockfishPool(
            STOCKFISH_PATH,
            num_engines=1,
            threads=config.stockfish_threads,
            hash_mb=config.stockfish_hash_mb,
        )
        account = bot.client.account.get()
        bot.bot_id = account.get("id", "")
        log.info(f"Logged in as: {bot.bot_id}")
        bot.challenge_player(
            args.challenge, rated=args.rated,
            clock_limit=clock_limit, clock_increment=clock_increment,
        )
        try:
            bot._event_loop()
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            bot.pool.shutdown()
    else:
        bot.start()


if __name__ == "__main__":
    main()

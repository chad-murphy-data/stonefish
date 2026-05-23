"""
Nettlesome Chess Engine
=======================
A chess bot that doesn't play the best move — it plays the move that's 
hardest for the opponent to respond to.

For each candidate move, we evaluate the opponent's top responses and 
score based on how much the opponent suffers if they miss the best reply.

Three bot types:
- NettlesomeBot: picks the move maximizing opponent difficulty
- RandomTopNBot: picks randomly from top N stockfish moves (simulates strong but imperfect player)
- PureStockfishBot: always plays the #1 stockfish move (control)
"""

import chess
import chess.engine
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import os
import shutil

def _find_stockfish():
    """Find the Stockfish binary, checking common locations."""
    # Environment variable override
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Common locations
    candidates = [
        r"C:\Users\chadm\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Last resort: check PATH
    found = shutil.which("stockfish")
    if found:
        return found
    # Fall back to the first candidate (will error on open)
    return candidates[0]

STOCKFISH_PATH = _find_stockfish()


@dataclass
class MoveScore:
    """Evaluation of a candidate move from the nettlesome perspective."""
    move: chess.Move
    own_eval: float          # How good this move is for us (cp, from our perspective)
    best_response_eval: float  # Opponent's best response eval (from our perspective, so lower = better for opponent)
    second_response_eval: float  # Opponent's 2nd best response
    third_response_eval: float   # Opponent's 3rd best response
    difficulty_score: float  # Our composite difficulty score (higher = harder for opponent)
    top_response_moves: List[chess.Move] = field(default_factory=list)  # opponent's top-N reply moves

    def __repr__(self):
        return (f"Move({self.move}: own={self.own_eval:+.2f}, "
                f"gap_1_2={self.second_response_eval - self.best_response_eval:.2f}, "
                f"gap_1_3={self.third_response_eval - self.best_response_eval:.2f}, "
                f"difficulty={self.difficulty_score:.2f})")


@dataclass
class NettlesomeMoment:
    """A position where the bot deliberately deviated from Stockfish #1 in
    favor of a higher-difficulty move. Captures what the opponent SHOULD
    have played and what they actually did."""
    move_num: int                       # halfmove number (1-indexed)
    our_move_san: str                   # SAN of the move we played
    sf_top_san: str                     # SAN of the SF #1 move we passed up
    eval_cost: float                    # pawns sacrificed vs SF #1
    gap_1_2: float                      # pawn gap between opponent's #1 and #2 replies
    top_response_ucis: List[str] = field(default_factory=list)  # opponent's top-N replies (UCI)
    opponent_reply_uci: Optional[str] = None  # what opponent actually played
    opponent_rank: Optional[int] = None       # 1=found #1, 2=#2, 3=#3, 4=outside top-3
    # Maia-EV scoring fields. Populated only when a maia_oracle is wired in.
    p_maia_top: Optional[float] = None         # P(Maia plays SF #1 reply)
    ev: Optional[float] = None                 # (1 - p_maia_top) * gap_1_2 - eval_cost


@dataclass
class GameStats:
    """Per-game diagnostics from a NettlesomeBot's perspective."""
    moments: List[NettlesomeMoment] = field(default_factory=list)
    result: Optional[float] = None      # 1.0=we won, 0.0=we lost, 0.5=draw
    total_moves: int = 0
    color: Optional[chess.Color] = None

    @property
    def n_moments(self) -> int:
        return len(self.moments)

    @property
    def n_opp_found_top(self) -> int:
        return sum(1 for m in self.moments if m.opponent_rank == 1)

    @property
    def n_opp_found_top3(self) -> int:
        return sum(1 for m in self.moments if m.opponent_rank is not None and m.opponent_rank <= 3)

    @property
    def avg_gap(self) -> float:
        if not self.moments: return 0.0
        return sum(m.gap_1_2 for m in self.moments) / len(self.moments)

    @property
    def avg_eval_cost(self) -> float:
        if not self.moments: return 0.0
        return sum(m.eval_cost for m in self.moments) / len(self.moments)

    @property
    def rank_distribution(self) -> dict:
        dist = {1: 0, 2: 0, 3: 0, "4+": 0}
        for m in self.moments:
            if m.opponent_rank is None: continue
            if m.opponent_rank >= 4: dist["4+"] += 1
            else: dist[m.opponent_rank] += 1
        return dist


def get_top_moves(engine, board, num_moves=5, depth=16, time_limit=None):
    """Get the top N moves from Stockfish with evaluations.
    
    Returns list of (move, score_in_cp_from_white_perspective) tuples.
    """
    kwargs = {"multipv": num_moves}
    if time_limit:
        limit = chess.engine.Limit(time=time_limit)
    else:
        limit = chess.engine.Limit(depth=depth)
    
    result = engine.analyse(board, limit, **kwargs)
    
    moves = []
    for info in result:
        if "pv" not in info or "score" not in info:
            continue
        move = info["pv"][0]
        score = info["score"].white()
        
        # Convert to centipawns, handling mate scores
        if score.is_mate():
            mate_in = score.mate()
            cp = 10000 if mate_in > 0 else -10000
        else:
            cp = score.score()
        
        moves.append((move, cp / 100.0))  # Convert to pawns
    
    return moves


def score_candidate_move(engine, board, candidate_move, num_responses=3, depth=16):
    """Score a candidate move by how difficult it is for the opponent to respond.
    
    Returns a MoveScore with the difficulty assessment.
    """
    # Get our eval for this move
    our_color = board.turn  # True = white
    
    # Make the candidate move
    board.push(candidate_move)
    
    # Get opponent's top responses
    opponent_moves = get_top_moves(engine, board, num_moves=num_responses, depth=depth)
    
    board.pop()
    
    if len(opponent_moves) < 2:
        # Position has very few legal moves, not useful for difficulty scoring
        return MoveScore(
            move=candidate_move,
            own_eval=0.0,
            best_response_eval=0.0,
            second_response_eval=0.0,
            third_response_eval=0.0,
            difficulty_score=0.0
        )
    
    # Evals are from white's perspective
    # We want to measure from the perspective of the side that just moved
    # "Higher eval = better for white"
    # If we're white, we WANT high eval (opponent's best response should be low for them = high for us)
    # If we're black, we WANT low eval
    
    sign = 1.0 if our_color == chess.WHITE else -1.0

    # Mate scores come out of get_top_moves as ±100 pawns (cp ±10000). Leaving
    # those uncapped poisons gap_1_2 averages: a single mate-trap registers a
    # 100-200p gap and drags every aggregate. Clamp at ±10p -- still huge
    # enough that mate-traps dominate selection, but no longer outliers.
    def _clamp(v, cap=10.0):
        return max(-cap, min(cap, v))

    # Opponent's best response (from our perspective: worst for us)
    best_resp = _clamp(opponent_moves[0][1] * sign)
    second_resp = _clamp(opponent_moves[1][1] * sign) if len(opponent_moves) > 1 else best_resp
    third_resp = _clamp(opponent_moves[2][1] * sign) if len(opponent_moves) > 2 else second_resp
    
    # The opponent WANTS to minimize our eval (make best_resp as negative as possible for us)
    # If they miss the best response, the eval stays higher for us
    # Gap = how much we gain if opponent plays 2nd best instead of best
    # From our perspective: second_resp - best_resp (should be positive if 2nd is worse for opponent)
    
    # Actually let me think about this more carefully:
    # opponent_moves are sorted by what's best for the OPPONENT (best first)
    # So from our perspective, opponent_moves[0] is the WORST for us
    # and opponent_moves[1] is slightly less bad for us
    # The gap (second - best) from our perspective means: 
    #   how much better for US if opponent plays 2nd best instead of best
    
    gap_1_2 = second_resp - best_resp  # How much we gain if they miss #1
    gap_1_3 = third_resp - best_resp   # How much we gain if they miss #1 and #2
    
    # Difficulty score: weighted combination
    # Higher = harder for opponent (bigger penalty for missing the best move)
    difficulty = 0.6 * gap_1_2 + 0.4 * gap_1_3
    
    return MoveScore(
        move=candidate_move,
        own_eval=0.0,  # Will be filled in by the caller
        best_response_eval=best_resp,
        second_response_eval=second_resp,
        third_response_eval=third_resp,
        difficulty_score=difficulty,
        top_response_moves=[m for m, _ in opponent_moves],
    )


class NettlesomeBot:
    """Plays the move that maximizes opponent difficulty.

    Tracks "nettlesome moments" -- positions where we deliberately passed up
    Stockfish's top move for one that's harder for the opponent to answer --
    and the opponent's reply rank, so we can measure whether the strategy
    actually works.
    """

    # Minimum eval_cost to count a move as a nettlesome moment.
    # Anything below this is effectively the SF #1 move (or a tied alternative)
    # and isn't a deliberate sacrifice.
    NETTLESOME_MIN_COST = 0.05

    def __init__(self, engine, num_candidates=10, num_responses=3,
                 depth=16, max_eval_cost=1.0, label="Nettlesome",
                 maia_oracle=None, baseline_bot=None,
                 # Puzzle-mode filter: when enabled, instead of picking the
                 # highest-EV trap, we filter candidates to those that meet
                 # all the conditions below and pick the one with policy
                 # probability closest to 0.5 (maximum game-deciding
                 # uncertainty). Disabled when puzzle_mode=False (default).
                 puzzle_mode=False,
                 min_eval_cost=0.0, min_gap=0.0, min_gap_ratio=0.0,
                 p_maia_min=0.0, p_maia_max=1.0):
        self.engine = engine
        self.num_candidates = num_candidates
        self.num_responses = num_responses
        self.depth = depth
        self.max_eval_cost = max_eval_cost  # Max pawns we'll sacrifice for difficulty
        self.label = label
        # Optional MaiaPolicyEngine. When set, choose_move switches to
        # expected-value scoring: EV = (1 - P_maia_top) * gap_1_2 - eval_cost,
        # and only deviates from SF #1 when some candidate has EV > 0.
        self.maia_oracle = maia_oracle
        # Optional fallback bot for non-trap positions. When set, the bot
        # plays baseline_bot.choose_move(board) on any move where no
        # positive-EV trap is found (instead of SF #1). Useful for
        # weakening overall play to a target rating while keeping the
        # sharp traps the EV logic finds.
        self.baseline_bot = baseline_bot
        self.puzzle_mode = puzzle_mode
        self.min_eval_cost = min_eval_cost
        self.min_gap = min_gap
        self.min_gap_ratio = min_gap_ratio
        self.p_maia_min = p_maia_min
        self.p_maia_max = p_maia_max
        self.stats: GameStats = GameStats()
        self._move_counter = 0
        self._pending_moment: Optional[NettlesomeMoment] = None

    def reset_stats(self, color: chess.Color):
        """Reset per-game stats. Call before a new game."""
        self.stats = GameStats(color=color)
        self._move_counter = 0
        self._pending_moment = None

    def note_opponent_reply(self, opponent_move: chess.Move):
        """Record that the opponent just played `opponent_move`. If we set a
        nettlesome trap on our previous move, classify the opponent's rank."""
        if self._pending_moment is None:
            return
        reply_uci = opponent_move.uci()
        self._pending_moment.opponent_reply_uci = reply_uci
        try:
            idx = self._pending_moment.top_response_ucis.index(reply_uci)
            self._pending_moment.opponent_rank = idx + 1
        except ValueError:
            self._pending_moment.opponent_rank = 4  # outside the top-N we tracked
        self._pending_moment = None

    def choose_move(self, board):
        """Choose the most nettlesome move.

        Two scoring paths depending on whether a Maia oracle is wired in:

        - With oracle: expected-value scoring. For each candidate trap we
          ask Maia for its policy on board-after-our-move, look up the
          probability Maia plays SF's #1 reply, and compute
              EV = (1 - p_maia_top) * gap_1_2 - eval_cost
          Play the highest-EV candidate; if none has EV > 0, play SF #1.

        - Without oracle: legacy difficulty scoring with an eval-cost
          penalty. Picks the highest-difficulty candidate among those that
          fit the eval_cost budget.
        """
        self._move_counter += 1

        candidates = get_top_moves(self.engine, board,
                                   num_moves=self.num_candidates,
                                   depth=self.depth)
        if not candidates:
            return random.choice(list(board.legal_moves))

        sf_top_move, sf_top_eval = candidates[0]
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_eval_for_us = sf_top_eval * sign

        # Score each candidate (response evals + raw gaps)
        scored = []  # list of (MoveScore, eval_cost)
        for move, eval_cp in candidates:
            eval_for_us = eval_cp * sign
            eval_cost = best_eval_for_us - eval_for_us
            if eval_cost > self.max_eval_cost:
                continue
            ms = score_candidate_move(self.engine, board, move,
                                      self.num_responses, self.depth)
            ms.own_eval = eval_for_us
            scored.append((ms, eval_cost))

        if not scored:
            return sf_top_move

        if self.maia_oracle is not None:
            # EV path: ask Maia for the probability of finding SF #1 reply
            # after we play each candidate, then pick the highest-EV trap.
            ev_scored = []  # (ms, eval_cost, p_maia_top, ev)
            for ms, eval_cost in scored:
                gap = ms.second_response_eval - ms.best_response_eval
                if ms.move == sf_top_move:
                    # The control: deviation gain is 0 by definition.
                    ev_scored.append((ms, eval_cost, None, 0.0))
                    continue
                # Probe Maia at the position we'd reach after our move
                next_board = board.copy()
                next_board.push(ms.move)
                if next_board.is_game_over():
                    # No reply means no trap value -- EV is just -eval_cost
                    ev_scored.append((ms, eval_cost, None, -eval_cost))
                    continue
                policy = self.maia_oracle.predict_policy(next_board)
                sf_top_reply = ms.top_response_moves[0] if ms.top_response_moves else None
                p_top = policy.get(sf_top_reply, 0.0) if sf_top_reply else 0.0
                ev = (1.0 - p_top) * gap - eval_cost
                ev_scored.append((ms, eval_cost, p_top, ev))

            if self.puzzle_mode:
                # Strict trap filter: keep only candidates meeting every
                # condition. Pick the one whose P_maia_top is closest to
                # 0.5 -- the moment where the opponent's choice most
                # genuinely determines the game.
                filtered = []
                for ms, cost, p, ev in ev_scored:
                    if ms.move == sf_top_move:
                        continue
                    if p is None:
                        continue
                    if cost < self.min_eval_cost:
                        continue
                    gap = ms.second_response_eval - ms.best_response_eval
                    if gap < self.min_gap:
                        continue
                    if cost > 0 and (gap / cost) < self.min_gap_ratio:
                        continue
                    if not (self.p_maia_min <= p <= self.p_maia_max):
                        continue
                    filtered.append((ms, cost, p, ev))
                if not filtered:
                    if self.baseline_bot is not None:
                        return self.baseline_bot.choose_move(board)
                    return sf_top_move
                filtered.sort(key=lambda x: abs(x[2] - 0.5))
                chosen_ms, chosen_cost, chosen_p, chosen_ev = filtered[0]
                chosen_move = chosen_ms.move
            else:
                # Sort by EV descending. Ties broken by lower cost.
                ev_scored.sort(key=lambda x: (-x[3], x[1]))
                chosen_ms, chosen_cost, chosen_p, chosen_ev = ev_scored[0]
                chosen_move = chosen_ms.move

                # If the best move is SF #1 (or no candidate has positive
                # EV), fall back to the baseline_bot or SF #1.
                if chosen_move == sf_top_move or chosen_ev <= 0.0:
                    if self.baseline_bot is not None:
                        return self.baseline_bot.choose_move(board)
                    return sf_top_move
        else:
            # Legacy difficulty-score path (with hand-tuned cost penalty)
            for ms, eval_cost in scored:
                ms.difficulty_score -= eval_cost * 0.5
            scored.sort(key=lambda x: x[0].difficulty_score, reverse=True)
            chosen_ms, chosen_cost = scored[0]
            chosen_move = chosen_ms.move
            chosen_p, chosen_ev = None, None

        # Did we set a nettlesome trap? (passed up SF #1 for a costlier move)
        if chosen_move != sf_top_move and chosen_cost >= self.NETTLESOME_MIN_COST:
            try:
                our_san = board.san(chosen_move)
            except Exception:
                our_san = chosen_move.uci()
            try:
                sf_san = board.san(sf_top_move)
            except Exception:
                sf_san = sf_top_move.uci()
            gap = chosen_ms.second_response_eval - chosen_ms.best_response_eval
            moment = NettlesomeMoment(
                move_num=self._move_counter,
                our_move_san=our_san,
                sf_top_san=sf_san,
                eval_cost=round(chosen_cost, 3),
                gap_1_2=round(gap, 3),
                top_response_ucis=[m.uci() for m in chosen_ms.top_response_moves],
                p_maia_top=round(chosen_p, 4) if chosen_p is not None else None,
                ev=round(chosen_ev, 3) if chosen_ev is not None else None,
            )
            self.stats.moments.append(moment)
            self._pending_moment = moment

        return chosen_move

    @property
    def name(self):
        mode_tag = "+Puzzle" if self.puzzle_mode else ("+EV" if self.maia_oracle is not None else "")
        tight_tag = "Tight" if self.max_eval_cost <= 0.5 else ""
        base_tag = ""
        if self.baseline_bot is not None:
            base_tag = f"/{getattr(self.baseline_bot, 'name', 'baseline')}"
        base = self.label if self.label != "Nettlesome" else "Nettlesome"
        return (f"{base}{tight_tag}{mode_tag}{base_tag}"
                f"(d={self.depth},c={self.num_candidates},ec={self.max_eval_cost})")


class RandomTopNBot:
    """Picks randomly from the top N stockfish moves."""
    
    def __init__(self, engine, top_n=5, depth=16, weights=None):
        self.engine = engine
        self.top_n = top_n
        self.depth = depth
        # Default weights favor top moves but allow mistakes
        self.weights = weights or self._default_weights()
    
    def _default_weights(self):
        """Generate decreasing weights for top N moves."""
        # Top move gets most weight, declining from there
        w = [1.0 / (i + 1) for i in range(self.top_n)]
        total = sum(w)
        return [x / total for x in w]
    
    def choose_move(self, board):
        """Pick randomly from top N moves."""
        candidates = get_top_moves(self.engine, board, 
                                   num_moves=self.top_n, 
                                   depth=self.depth)
        
        if not candidates:
            return random.choice(list(board.legal_moves))
        
        moves = [m for m, _ in candidates]
        weights = self.weights[:len(moves)]
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return random.choices(moves, weights=weights, k=1)[0]
    
    @property
    def name(self):
        return f"RandomTop{self.top_n}(d={self.depth})"


class PureStockfishBot:
    """Always plays the #1 Stockfish move."""
    
    def __init__(self, engine, depth=16):
        self.engine = engine
        self.depth = depth
    
    def choose_move(self, board):
        """Play the best move."""
        result = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        if "pv" in result:
            return result["pv"][0]
        return random.choice(list(board.legal_moves))
    
    @property
    def name(self):
        return f"PureStockfish(d={self.depth})"


class MaiaBot:
    """Wraps Maia (lc0 + Maia weights) as a stochastic playing partner.

    Maia is trained to predict the move a player at a given Elo would make.
    Its output is a policy distribution over legal moves. To model real
    human variance, we *sample* from that distribution rather than picking
    the argmax -- the way lc0 with nodes=1 would.

    temperature controls how peaked the sampling is:
        1.0 (default) -- sample directly from the trained policy
        < 1.0         -- bias toward the argmax (more deterministic)
        <= 0          -- always pick the argmax (legacy behavior)
    """

    def __init__(self, weights_path, rating=1900, lc0_path="lc0",
                 backend="eigen", threads=1, temperature=1.0, seed=None):
        from maia_policy import MaiaPolicyEngine
        import random as _random
        self.weights_path = weights_path
        self.rating = rating
        self.temperature = temperature
        self._rng = _random.Random(seed) if seed is not None else _random
        self._policy_engine = MaiaPolicyEngine(
            weights_path, lc0_path=lc0_path, backend=backend,
            threads=threads, rating=rating,
        )

    def choose_move(self, board):
        if self.temperature <= 0:
            return self._policy_engine.predict_top(board)
        return self._policy_engine.sample_move(board, self.temperature, rng=self._rng)

    def quit(self):
        try:
            self._policy_engine.quit()
        except Exception:
            pass

    @property
    def name(self):
        if self.temperature <= 0:
            return f"Maia{self.rating}(argmax)"
        if self.temperature == 1.0:
            return f"Maia{self.rating}"
        return f"Maia{self.rating}(T={self.temperature})"


def play_game(white_bot, black_bot, max_moves=200, verbose=False, live=False,
              game_label=""):
    """Play a single game between two bots. Returns result from white's perspective.

    Returns: (result, num_moves, move_list)
        result: 1.0 = white wins, 0.0 = black wins, 0.5 = draw

    If a bot is a NettlesomeBot, its per-game stats are reset at the start
    and populated during the game (visible afterwards via `bot.stats`).

    `live=True` prints a compact one-line status per move, with [N!] marking
    moves where the nettlesome bot deliberately deviated from SF #1.
    """
    board = chess.Board()
    moves = []

    # Reset stats on any nettlesome bots
    if isinstance(white_bot, NettlesomeBot):
        white_bot.reset_stats(chess.WHITE)
    if isinstance(black_bot, NettlesomeBot):
        black_bot.reset_stats(chess.BLACK)

    if live and game_label:
        print(f"    [{game_label}] starting", flush=True)

    last_status_ply = 0

    for ply in range(max_moves):
        if board.is_game_over():
            break

        active = white_bot if board.turn == chess.WHITE else black_bot
        passive = black_bot if board.turn == chess.WHITE else white_bot
        side_char = "W" if board.turn == chess.WHITE else "B"

        # Pre-compute SAN before pushing (board.san() needs the move to be legal in the current position)
        move = active.choose_move(board)
        try:
            move_san = board.san(move)
        except Exception:
            move_san = move.uci()

        # Tell the passive bot what just landed -- it may want to classify
        if isinstance(passive, NettlesomeBot):
            passive.note_opponent_reply(move)

        moves.append(move)
        board.push(move)

        if verbose:
            print(f"  {ply + 1}. {side_char}: {move_san}")

        if live:
            # Was this a nettlesome moment?
            tag = ""
            if isinstance(active, NettlesomeBot) and active.stats.moments:
                last_m = active.stats.moments[-1]
                if last_m.move_num == active._move_counter:
                    tag = f" [N! cost={last_m.eval_cost:.2f} gap={last_m.gap_1_2:.2f}]"
            # Print every 10 plies, every nettlesome moment, or at end
            should_print = bool(tag) or (ply - last_status_ply) >= 10
            if should_print:
                prefix = f"    [{game_label}] " if game_label else "    "
                print(f"{prefix}ply {ply + 1:3d} {side_char}: {move_san}{tag}", flush=True)
                last_status_ply = ply

    result_str = board.result()
    if result_str == "1-0":
        result = 1.0
    elif result_str == "0-1":
        result = 0.0
    else:
        result = 0.5

    # Populate per-bot result + move count from each side's perspective
    if isinstance(white_bot, NettlesomeBot):
        white_bot.stats.result = result
        white_bot.stats.total_moves = len(moves)
    if isinstance(black_bot, NettlesomeBot):
        black_bot.stats.result = 1.0 - result
        black_bot.stats.total_moves = len(moves)

    return result, len(moves), moves


def run_simulation(bot_a, bot_b, num_games=20, verbose=False, live=False):
    """Run a simulation between two bots, alternating colors.

    Returns dict with results plus aggregated nettlesome stats.
    """
    results = {
        "bot_a_name": bot_a.name,
        "bot_b_name": bot_b.name,
        "bot_a_wins": 0,
        "bot_b_wins": 0,
        "draws": 0,
        "total_games": 0,
        "total_moves": 0,
        "game_results": [],
        "nettlesome_games": [],  # list of GameStats from nettlesome bots
    }

    for game_num in range(num_games):
        if game_num % 2 == 0:
            white, black = bot_a, bot_b
            a_is_white = True
        else:
            white, black = bot_b, bot_a
            a_is_white = False

        game_label = f"G{game_num + 1}/{num_games}"
        if live:
            print(f"  [{game_label}] W={white.name} vs B={black.name}", flush=True)
        elif verbose:
            print(f"\nGame {game_num + 1}/{num_games}: "
                  f"W={white.name} vs B={black.name}")

        start = time.time()
        score, num_moves, _ = play_game(white, black, verbose=False, live=live,
                                        game_label=game_label)
        elapsed = time.time() - start

        a_score = score if a_is_white else 1.0 - score

        if a_score == 1.0:
            results["bot_a_wins"] += 1
            outcome = f"{bot_a.name} wins"
        elif a_score == 0.0:
            results["bot_b_wins"] += 1
            outcome = f"{bot_b.name} wins"
        else:
            results["draws"] += 1
            outcome = "Draw"

        results["total_games"] += 1
        results["total_moves"] += num_moves
        results["game_results"].append(a_score)

        # Capture nettlesome stats from whichever side is the NettlesomeBot
        for bot in (white, black):
            if isinstance(bot, NettlesomeBot) and bot is bot_a:
                results["nettlesome_games"].append(bot.stats)

        # Per-game one-liner with nettlesome stats inline
        extras = ""
        if isinstance(bot_a, NettlesomeBot):
            s = bot_a.stats
            top_rate = (s.n_opp_found_top / s.n_moments) if s.n_moments else 0.0
            extras = (f"  | nettlesome: {s.n_moments} moments, "
                      f"opp-found-top={s.n_opp_found_top}/{s.n_moments} ({top_rate:.0%}), "
                      f"avg-cost={s.avg_eval_cost:.2f}p, avg-gap={s.avg_gap:.2f}p")
        print(f"  Game {game_num + 1}: {outcome} in {num_moves} moves ({elapsed:.1f}s){extras}",
              flush=True)

    total = results["total_games"]
    a_score = (results["bot_a_wins"] + 0.5 * results["draws"]) / total
    avg_moves = results["total_moves"] / total

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {bot_a.name} vs {bot_b.name}")
    print(f"{'='*60}")
    print(f"  {bot_a.name}: {results['bot_a_wins']}W / {results['draws']}D / {results['bot_b_wins']}L")
    print(f"  Score: {a_score:.1%}")
    print(f"  Avg game length: {avg_moves:.0f} moves")

    # Aggregate nettlesome diagnostics across games
    nett_games = results["nettlesome_games"]
    if nett_games:
        total_moments = sum(s.n_moments for s in nett_games)
        total_found_top = sum(s.n_opp_found_top for s in nett_games)
        total_found_top3 = sum(s.n_opp_found_top3 for s in nett_games)
        weighted_cost = sum(m.eval_cost for s in nett_games for m in s.moments)
        weighted_gap = sum(m.gap_1_2 for s in nett_games for m in s.moments)
        agg_dist = {1: 0, 2: 0, 3: 0, "4+": 0}
        for s in nett_games:
            d = s.rank_distribution
            for k in agg_dist:
                agg_dist[k] += d[k]
        top_rate = (total_found_top / total_moments) if total_moments else 0.0
        top3_rate = (total_found_top3 / total_moments) if total_moments else 0.0
        print(f"  ---")
        print(f"  Nettlesome diagnostics across {len(nett_games)} games:")
        print(f"    moments played:       {total_moments} ({total_moments/len(nett_games):.1f}/game)")
        print(f"    opp found #1 reply:   {total_found_top}/{total_moments} ({top_rate:.0%})")
        print(f"    opp in top-3 reply:   {total_found_top3}/{total_moments} ({top3_rate:.0%})")
        print(f"    avg eval cost paid:   {weighted_cost/total_moments:.2f} pawns" if total_moments else "    avg eval cost paid:   n/a")
        print(f"    avg gap 1-2 created:  {weighted_gap/total_moments:.2f} pawns" if total_moments else "    avg gap 1-2 created:  n/a")
        print(f"    rank distribution:    #1={agg_dist[1]}, #2={agg_dist[2]}, #3={agg_dist[3]}, miss={agg_dist['4+']}")
    print(f"{'='*60}", flush=True)

    return results


if __name__ == "__main__":
    import sys
    import os

    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    maia_weights = os.environ.get("MAIA_WEIGHTS", "/home/user/.maia/maia-1900.pb.gz")
    maia_rating = int(os.environ.get("MAIA_RATING", "1900"))

    print(f"Nettlesome Chess Engine Simulation")
    print(f"Games per matchup: {num_games}, Depth: {depth}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Maia weights: {maia_weights} (rating {maia_rating})")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    # MaiaPolicyEngine drives lc0 directly to expose per-move policy
    # probabilities. The +EV bots use it during move selection.
    from maia_policy import MaiaPolicyEngine
    maia_oracle = MaiaPolicyEngine(maia_weights, rating=maia_rating)

    # Baseline = stochastic Maia 1900 (T=1) -- the calibration sweet spot
    # from the previous tournament where Stonefish becomes beatable.
    baseline_t1 = MaiaBot(maia_weights, rating=maia_rating, temperature=1.0, seed=3)

    # Strict puzzle-filter Stonefish. Plays a trap only when the position
    # meets every filter condition simultaneously; otherwise plays its
    # Maia baseline. Picks by P closest to 0.5 (maximum opponent
    # uncertainty -- the real "puzzle moment of the game").
    stonefish_puzzle = NettlesomeBot(
        engine, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.5,           # allow costlier traps
        maia_oracle=maia_oracle, baseline_bot=baseline_t1,
        puzzle_mode=True,
        min_eval_cost=0.5,                        # each find gives opp ~0.5p
        min_gap=1.5,                              # each miss costs opp ~1.0p (net)
        min_gap_ratio=2.0,                        # gap >= 2 * cost (asymmetric)
        p_maia_min=0.30, p_maia_max=0.70,         # findable for 1900, not trivial
    )

    # Loose-EV Stonefish at T=1 baseline for direct comparison
    stonefish_ev_t1 = NettlesomeBot(
        engine, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.0,
        maia_oracle=maia_oracle, baseline_bot=baseline_t1,
    )

    stockfish = PureStockfishBot(engine, depth=depth)
    maia = MaiaBot(maia_weights, rating=maia_rating, temperature=1.0, seed=99)

    all_results = {}

    matchups = [
        ("Stonefish+Puzzle (T=1) vs Maia", stonefish_puzzle, maia),
        ("Stonefish+EV     (T=1) vs Maia", stonefish_ev_t1,  maia),
        ("PureStockfish          vs Maia", stockfish,        maia),
    ]

    for label, bot_a, bot_b in matchups:
        print("=" * 60, flush=True)
        print(f"MATCHUP: {label}")
        print("=" * 60, flush=True)
        r = run_simulation(bot_a, bot_b, num_games=num_games, live=True)
        all_results[label] = r
        print(flush=True)

    # Final comparison: divergence bots vs control, against each opponent
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Matchup':<38} {'Score':>7} {'W-D-L':>10} {'AvgLen':>7}")
    print("-" * 70)
    for label, r in all_results.items():
        total = r["total_games"]
        score = (r["bot_a_wins"] + 0.5 * r["draws"]) / total
        avg_len = r["total_moves"] / total
        wdl = f"{r['bot_a_wins']}-{r['draws']}-{r['bot_b_wins']}"
        print(f"{label:<38} {score:>6.1%} {wdl:>10} {avg_len:>6.0f}")

    print()
    s = all_results.get("PureStockfish          vs Maia")
    if s:
        s_score = (s["bot_a_wins"] + 0.5 * s["draws"]) / s["total_games"]
        for label in ("Stonefish+Puzzle (T=1) vs Maia",
                      "Stonefish+EV     (T=1) vs Maia"):
            r = all_results.get(label)
            if not r:
                continue
            r_score = (r["bot_a_wins"] + 0.5 * r["draws"]) / r["total_games"]
            print(f"  {label:<34} {r_score:>6.1%}  (edge over SF: {r_score - s_score:+.1%})")
        print(f"  {'PureStockfish          vs Maia':<34} {s_score:>6.1%}  (control)")
        print()

    for bot in (maia, baseline_t1):
        try: bot.quit()
        except Exception: pass
    maia_oracle.quit()
    engine.quit()

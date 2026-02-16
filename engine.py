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
from dataclasses import dataclass

STOCKFISH_PATH = r"C:\Users\chadm\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe"


@dataclass
class MoveScore:
    """Evaluation of a candidate move from the nettlesome perspective."""
    move: chess.Move
    own_eval: float          # How good this move is for us (cp, from our perspective)
    best_response_eval: float  # Opponent's best response eval (from our perspective, so lower = better for opponent)
    second_response_eval: float  # Opponent's 2nd best response
    third_response_eval: float   # Opponent's 3rd best response
    difficulty_score: float  # Our composite difficulty score (higher = harder for opponent)
    
    def __repr__(self):
        return (f"Move({self.move}: own={self.own_eval:+.2f}, "
                f"gap_1_2={self.second_response_eval - self.best_response_eval:.2f}, "
                f"gap_1_3={self.third_response_eval - self.best_response_eval:.2f}, "
                f"difficulty={self.difficulty_score:.2f})")


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
    
    # Opponent's best response (from our perspective: worst for us)
    best_resp = opponent_moves[0][1] * sign  # From our perspective
    second_resp = opponent_moves[1][1] * sign if len(opponent_moves) > 1 else best_resp
    third_resp = opponent_moves[2][1] * sign if len(opponent_moves) > 2 else second_resp
    
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
        difficulty_score=difficulty
    )


class NettlesomeBot:
    """Plays the move that maximizes opponent difficulty."""
    
    def __init__(self, engine, num_candidates=10, num_responses=3, 
                 depth=16, max_eval_cost=1.0):
        self.engine = engine
        self.num_candidates = num_candidates
        self.num_responses = num_responses
        self.depth = depth
        self.max_eval_cost = max_eval_cost  # Max pawns we'll sacrifice for difficulty
    
    def choose_move(self, board):
        """Choose the most nettlesome move."""
        # Get our top candidate moves
        candidates = get_top_moves(self.engine, board, 
                                   num_moves=self.num_candidates, 
                                   depth=self.depth)
        
        if not candidates:
            # Fallback: random legal move
            return random.choice(list(board.legal_moves))
        
        best_eval = candidates[0][1]  # Best available eval
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_eval_for_us = best_eval * sign
        
        # Score each candidate by opponent difficulty
        scored = []
        for move, eval_cp in candidates:
            eval_for_us = eval_cp * sign
            
            # Skip moves that cost too much eval
            eval_cost = best_eval_for_us - eval_for_us
            if eval_cost > self.max_eval_cost:
                continue
            
            ms = score_candidate_move(self.engine, board, move, 
                                      self.num_responses, self.depth)
            ms.own_eval = eval_for_us
            
            # Adjust difficulty score by eval cost
            # Penalize moves that cost us eval, but not too harshly
            ms.difficulty_score -= eval_cost * 0.5
            
            scored.append(ms)
        
        if not scored:
            return candidates[0][0]  # Fallback to best move
        
        # Pick the move with highest difficulty score
        scored.sort(key=lambda x: x.difficulty_score, reverse=True)
        return scored[0].move
    
    @property
    def name(self):
        return f"Nettlesome(d={self.depth},c={self.num_candidates})"


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


def play_game(white_bot, black_bot, max_moves=200, verbose=False):
    """Play a single game between two bots. Returns result from white's perspective.
    
    Returns: (result, num_moves, move_list)
        result: 1.0 = white wins, 0.0 = black wins, 0.5 = draw
    """
    board = chess.Board()
    moves = []
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        if board.turn == chess.WHITE:
            move = white_bot.choose_move(board)
        else:
            move = black_bot.choose_move(board)
        
        if verbose:
            side = "W" if board.turn == chess.WHITE else "B"
            print(f"  {move_num + 1}. {side}: {board.san(move)}")
        
        moves.append(move)
        board.push(move)
    
    # Determine result
    result = board.result()
    if result == "1-0":
        return 1.0, len(moves), moves
    elif result == "0-1":
        return 0.0, len(moves), moves
    else:
        return 0.5, len(moves), moves


def run_simulation(bot_a, bot_b, num_games=20, verbose=False):
    """Run a simulation between two bots, alternating colors.
    
    Returns dict with results.
    """
    results = {
        "bot_a_name": bot_a.name,
        "bot_b_name": bot_b.name,
        "bot_a_wins": 0,
        "bot_b_wins": 0,
        "draws": 0,
        "total_games": 0,
        "total_moves": 0,
        "game_results": []
    }
    
    for game_num in range(num_games):
        # Alternate colors
        if game_num % 2 == 0:
            white, black = bot_a, bot_b
            a_is_white = True
        else:
            white, black = bot_b, bot_a
            a_is_white = False
        
        if verbose:
            print(f"\nGame {game_num + 1}/{num_games}: "
                  f"W={white.name} vs B={black.name}")
        
        start = time.time()
        score, num_moves, _ = play_game(white, black, verbose=False)
        elapsed = time.time() - start
        
        # Convert to bot_a's perspective
        if a_is_white:
            a_score = score
        else:
            a_score = 1.0 - score
        
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
        
        print(f"  Game {game_num + 1}: {outcome} in {num_moves} moves ({elapsed:.1f}s)")
    
    # Summary
    total = results["total_games"]
    a_score = (results["bot_a_wins"] + 0.5 * results["draws"]) / total
    avg_moves = results["total_moves"] / total
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {bot_a.name} vs {bot_b.name}")
    print(f"{'='*60}")
    print(f"  {bot_a.name}: {results['bot_a_wins']}W / {results['draws']}D / {results['bot_b_wins']}L")
    print(f"  Score: {a_score:.1%}")
    print(f"  Avg game length: {avg_moves:.0f} moves")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    
    print(f"Nettlesome Chess Engine Simulation")
    print(f"Games: {num_games}, Depth: {depth}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print()
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    # Configure engine for speed
    engine.configure({"Threads": 2, "Hash": 256})
    
    # Create bots
    nettlesome = NettlesomeBot(engine, num_candidates=7, num_responses=3, 
                                depth=depth, max_eval_cost=1.0)
    random_top5 = RandomTopNBot(engine, top_n=5, depth=depth)
    stockfish = PureStockfishBot(engine, depth=depth)
    
    # Run matchups
    print("=" * 60)
    print("MATCHUP 1: Nettlesome vs RandomTop5")
    print("=" * 60)
    r1 = run_simulation(nettlesome, random_top5, num_games=num_games)
    
    print()
    print("=" * 60)
    print("MATCHUP 2: PureStockfish vs RandomTop5")
    print("=" * 60)
    r2 = run_simulation(stockfish, random_top5, num_games=num_games)
    
    print()
    print("=" * 60)
    print("MATCHUP 3: Nettlesome vs PureStockfish")
    print("=" * 60)
    r3 = run_simulation(nettlesome, stockfish, num_games=num_games)
    
    # Final comparison
    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    n_vs_r = (r1["bot_a_wins"] + 0.5 * r1["draws"]) / r1["total_games"]
    s_vs_r = (r2["bot_a_wins"] + 0.5 * r2["draws"]) / r2["total_games"]
    n_vs_s = (r3["bot_a_wins"] + 0.5 * r3["draws"]) / r3["total_games"]
    
    print(f"  Nettlesome vs RandomTop5:    {n_vs_r:.1%}")
    print(f"  PureStockfish vs RandomTop5: {s_vs_r:.1%}")
    print(f"  Nettlesome vs PureStockfish: {n_vs_s:.1%}")
    print()
    
    if n_vs_r > s_vs_r:
        print(f"  >>> Nettlesome beats RandomTop5 MORE than PureStockfish does!")
        print(f"  >>> Edge: {n_vs_r - s_vs_r:+.1%}")
    else:
        print(f"  >>> PureStockfish beats RandomTop5 more than Nettlesome")
        print(f"  >>> Difference: {s_vs_r - n_vs_r:+.1%}")
    
    engine.quit()

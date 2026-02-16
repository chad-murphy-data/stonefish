"""
Extended simulation with more opponents and statistics.
"""

import chess
import chess.engine
import time
import sys
from engine import (NettlesomeBot, RandomTopNBot, PureStockfishBot, 
                     play_game, run_simulation, STOCKFISH_PATH)

def run_full_tournament(num_games=10, depth=12):
    """Run full tournament between all bot types."""
    
    print(f"NETTLESOME CHESS ENGINE - FULL TOURNAMENT")
    print(f"Games per matchup: {num_games}, Depth: {depth}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print()
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})
    
    # Create bots
    nettlesome = NettlesomeBot(engine, num_candidates=7, num_responses=3, 
                                depth=depth, max_eval_cost=1.0)
    
    nettlesome_tight = NettlesomeBot(engine, num_candidates=7, num_responses=3, 
                                      depth=depth, max_eval_cost=0.3)
    
    random_top5 = RandomTopNBot(engine, top_n=5, depth=depth)
    random_top3 = RandomTopNBot(engine, top_n=3, depth=depth)
    
    # Custom weights: 70/20/10 (your original suggestion)
    random_weighted = RandomTopNBot(engine, top_n=3, depth=depth, 
                                     weights=[0.7, 0.2, 0.1])
    
    stockfish = PureStockfishBot(engine, depth=depth)
    
    all_results = {}
    
    matchups = [
        ("Nettlesome vs RandomTop5", nettlesome, random_top5),
        ("Nettlesome vs RandomTop3", nettlesome, random_top3),
        ("Nettlesome vs Weighted70/20/10", nettlesome, random_weighted),
        ("PureStockfish vs RandomTop5", stockfish, random_top5),
        ("PureStockfish vs RandomTop3", stockfish, random_top3),
        ("PureStockfish vs Weighted70/20/10", stockfish, random_weighted),
        ("NettlesomeTight vs RandomTop5", nettlesome_tight, random_top5),
        ("Nettlesome vs PureStockfish", nettlesome, stockfish),
    ]
    
    for label, bot_a, bot_b in matchups:
        print(f"\n{'='*60}")
        print(f"MATCHUP: {label}")
        print(f"{'='*60}")
        result = run_simulation(bot_a, bot_b, num_games=num_games)
        
        total = result["total_games"]
        score = (result["bot_a_wins"] + 0.5 * result["draws"]) / total
        avg_len = result["total_moves"] / total
        
        all_results[label] = {
            "score": score,
            "wins": result["bot_a_wins"],
            "draws": result["draws"],
            "losses": result["bot_b_wins"],
            "avg_length": avg_len,
        }
    
    # Print comparison table
    print(f"\n\n{'='*70}")
    print(f"TOURNAMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Matchup':<40} {'Score':>6} {'W-D-L':>10} {'AvgLen':>7}")
    print(f"{'-'*70}")
    
    for label, r in all_results.items():
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        print(f"{label:<40} {r['score']:>5.1%} {wdl:>10} {r['avg_length']:>6.0f}")
    
    # Key comparison
    print(f"\n{'='*70}")
    print(f"KEY FINDING: Does Nettlesome beat imperfect opponents better than pure Stockfish?")
    print(f"{'='*70}")
    
    comparisons = [
        ("RandomTop5", "Nettlesome vs RandomTop5", "PureStockfish vs RandomTop5"),
        ("RandomTop3", "Nettlesome vs RandomTop3", "PureStockfish vs RandomTop3"),
        ("Weighted70/20/10", "Nettlesome vs Weighted70/20/10", "PureStockfish vs Weighted70/20/10"),
    ]
    
    for opp_name, nett_key, sf_key in comparisons:
        n_score = all_results[nett_key]["score"]
        s_score = all_results[sf_key]["score"]
        n_len = all_results[nett_key]["avg_length"]
        s_len = all_results[sf_key]["avg_length"]
        
        print(f"\n  vs {opp_name}:")
        print(f"    Nettlesome:    {n_score:.1%} (avg {n_len:.0f} moves)")
        print(f"    PureStockfish: {s_score:.1%} (avg {s_len:.0f} moves)")
        
        if n_score > s_score:
            print(f"    >>> Nettlesome wins more! Edge: {n_score - s_score:+.1%}")
        elif n_score == s_score and n_len < s_len:
            print(f"    >>> Same win rate, but Nettlesome wins {s_len - n_len:.0f} moves FASTER!")
        elif n_score == s_score:
            print(f"    >>> Tied on win rate and speed")
        else:
            print(f"    >>> PureStockfish wins more by {s_score - n_score:+.1%}")
            if n_len < s_len:
                print(f"    >>> But Nettlesome wins are {s_len - n_len:.0f} moves faster when it does win")
    
    engine.quit()
    return all_results


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    
    run_full_tournament(num_games=num_games, depth=depth)

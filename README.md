# Nettlesome Chess Engine

A chess bot that doesn't play the best move — it plays the move that's hardest for the opponent to respond to.

## The Idea

For each candidate move, the bot evaluates the opponent's top responses. It picks the move where the gap between the opponent's best response and their 2nd/3rd best response is largest. Every move says "find the one right answer or suffer."

## Setup

```bash
# Install stockfish
# Mac: brew install stockfish
# Ubuntu: sudo apt install stockfish
# Or download from https://stockfishchess.org/download/

# Install python-chess
pip install python-chess

# Update STOCKFISH_PATH in engine.py to match your installation
# Mac (homebrew): /opt/homebrew/bin/stockfish
# Ubuntu (apt): /usr/bin/stockfish
```

## Quick Test

```bash
python test_quick.py
```

This shows the nettlesome scoring for opening moves. You should see that some "worse" moves (like e3) have higher difficulty scores than "better" moves (like Nf3).

## Run Tournament

```bash
# 10 games per matchup, depth 12 (takes ~15-30 min)
python tournament.py 10 12

# Quick version: 4 games, depth 10 (~2 min)
python tournament.py 4 10
```

## Three Bots

- **NettlesomeBot**: Picks the move maximizing opponent difficulty
- **RandomTopNBot**: Picks randomly from top N stockfish moves (simulates imperfect player)
- **PureStockfishBot**: Always plays the #1 move (control group)

## Key Parameters

- `depth`: Stockfish search depth (10=fast, 16=strong, 20+=slow but accurate)
- `num_candidates`: How many of our moves to evaluate (7 is good balance)
- `num_responses`: How many opponent responses to check (3 is enough)
- `max_eval_cost`: Max pawns we'll sacrifice for difficulty (1.0 = aggressive, 0.3 = conservative)

## Early Results (depth 10, 4 games)

Both Nettlesome and PureStockfish beat RandomTop5 100%, but Nettlesome won in 62 moves avg vs PureStockfish in 84 moves. Nettlesome forces errors faster.

## Next Steps

- Run larger tournament (50+ games) for statistical significance
- Wrap as Lichess bot using `berserk` library
- Test against human players
- Explore anti-cheat applications (honeypot positions where shallow engine gives wrong answer)

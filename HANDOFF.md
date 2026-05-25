# Stonefish — Handoff (May 2026)

You're picking up a chess bot project that's calibrated, tested against a synthetic
opponent, and ready for real-opponent validation. This doc tells you everything you
need to keep going.

## What Stonefish is

A chess bot designed for the following user experience: the bot sets ~3-5 "puzzle moments"
per game (positions where finding the right reply is hard but rewarding); the *opponent's*
ability to solve those puzzles determines the game's outcome. Find ~50%+ of them: opp wins.
Find fewer: opp loses.

It's built on Stockfish 16 for engine smarts + lc0 with Maia 1900 weights for human-tier
move probabilities.

## Repo layout

```
engine.py                    # Core: NettlesomeBot, MaiaBot, WeakenedStockfishBot,
                             # EquilibriumBaselineBot, CoinFlipTesterBot,
                             # PureStockfishBot, RandomTopNBot, play_game, run_simulation
                             # Also: __main__ runs a small head-to-head tournament

maia_policy.py               # MaiaPolicyEngine: queries lc0 directly to extract per-move
                             # policy probabilities (P(Maia plays move X))

qa_test.py                   # QA harness against stochastic Maia 1900 opponent
qa_tester.py                 # QA harness against CoinFlipTesterBot (controllable opp)
trace_tester.py              # Single-game move-by-move trace vs CoinFlipTesterBot
trace_game.py                # Single-game move-by-move trace vs MaiaBot

results/*.log                # Saved QA tournament outputs

.maia/maia-1100.pb.gz, maia-1500.pb.gz, maia-1900.pb.gz   # NOT in repo; live at
                             # /home/user/.maia/ — download from
                             # https://github.com/CSSLab/maia-chess/raw/master/maia_weights/
```

## Current bot behavior (NettlesomeBot v1)

Each move, the bot picks one of these modes in priority order:

1. **Post-trap give-back** (if `_post_trap_remaining > 0`): play `post_trap_baseline.choose_move()`
   For the 5 own moves after opp solved a trap (rank == 1), use a *weaker* baseline so opp can convert.

2. **Endgame precise mode** (if `endgame_mode=True` and in endgame and opp solve rate < 50%):
   Play SF #1 — convert any earned advantage decisively. "Endgame" = move >= 30 OR pieces <= 14.

3. **Trap-setting (`puzzle_mode=True`)**: 
   - Get SF top-7 candidates at depth 10 (max sacrifice 1.5p)
   - Query Maia oracle for `P(Maia plays SF top reply)` at each candidate
   - Filter: `eval_cost >= 0.20`, `gap >= 0.50`, `gap/cost >= 1.5`, `0.20 <= P_maia <= 0.80`
   - Pick the candidate with `P_maia_top` closest to 0.5 (max uncertainty)
   - If no candidate passes: fall through to baseline

4. **Baseline**: `baseline_bot.choose_move()` — between traps, outside endgame.

## Current Stonefish v1 configuration (see `engine.py` `__main__`)

```python
baseline_eq = WeakenedStockfishBot(STOCKFISH_PATH, target_elo=1900, move_time=0.3)
give_back_baseline = WeakenedStockfishBot(STOCKFISH_PATH, target_elo=1500, move_time=0.3)

stonefish_v1 = NettlesomeBot(
    engine,                               # full-strength SF for trap design
    num_candidates=7, num_responses=3,
    depth=10, max_eval_cost=1.5,
    maia_oracle=maia_oracle,              # MaiaPolicyEngine, lc0+Maia1900
    baseline_bot=baseline_eq,             # SF Elo 1900 between traps
    puzzle_mode=True,
    min_eval_cost=0.20,
    min_gap=0.50,
    min_gap_ratio=1.5,
    p_maia_min=0.20, p_maia_max=0.80,
    post_trap_baseline=give_back_baseline, # SF Elo 1500 for 5 moves after found trap
    post_trap_duration=5,
    endgame_mode=True,
    endgame_min_move=30, endgame_min_pieces=14,
)
```

`conversion_mode` (probabilistic "reverse Stonefish" — bot itself misses SF #1
sometimes) is implemented but commented out, reserved for an ELO-aware future.

## QA results to date

Against `CoinFlipTesterBot` (Stockfish baseline + weighted coin for trap responses).
15 games per probability, depth 10. Two key sweeps:

**With EquilibriumBaselineBot baseline (engineered drift, the previous design):**
```
find_prob   W-D-L    Stonefish    actual find
0.30        12-0-3   80%          24%
0.50        6-0-9    40%          53%
0.70        4-0-11   27%          73%
0.90        0-0-15   0%           94%
```
Clean monotonic curve. Decisive at extremes. Some noise in middle.

**With WeakenedStockfishBot baseline (genuine 1900, the current design):**
Started but NOT FINISHED — was at p=0.30 G4 when handed off. First 4 games:
- G1: WIN, 100% found (1 moment only)
- G2: WIN, 25% found
- G3: DRAW, 83% found (200 moves)
- G4: WIN, 25% found

Early signs: the weakened-SF baseline might be too forgiving — Stonefish won G1
despite 100% find rate. May need to retune with the new baseline. **Suggest the new
agent's first action be: run the full QA sweep with the WeakenedSF baseline to get
the new curve.**

Command to run:
```bash
python3 -u qa_tester.py 15 10 0.3 0.5 0.7 0.9 2>&1 | tee /tmp/qa_sf1900_full.log
```
Estimated runtime: ~30-45 min (4 probs x 15 games, weakened SF adds 0.3s/move on top of regular Stockfish).

## QA spec (the four rules)

These are the calibration targets:
1. **~4-6 traps per game** on average (mean in [4, 6], >=80% of games have >=3 moments)
2. **Find > 60% → opp wins** (Stonefish loses or draws >= 70% of those games)
3. **Find < 60% → opp loses** (Stonefish wins or draws >= 70% of those games)
4. **find_rate predicts outcome** (Pearson r >= 0.5)

User accepts that humans convert worse than the synthetic tester, so a tester
break-even around 50% find rate corresponds to a "humans find > 60%" win threshold
in actual play.

## Roadmap (per user)

1. **DONE**: Build the trap mechanic
2. **DONE**: Validate find→outcome property against CoinFlipTester
3. **IN PROGRESS**: Nerf bot to actual 1900 strength (WeakenedStockfishBot baseline)
   - Re-run QA with new baseline
   - Possibly retune give-back / endgame mode params if curve degrades
4. **NEXT**: Test against real Maia 1900 (stochastic Maia at T=1)
   - Use `qa_test.py` (which uses MaiaBot opponent, not CoinFlipTesterBot)
   - Expect higher variance — Maia 1900 at T=1 has ~2-4p single-move blunders
5. **AFTER THAT**: Port to Lichess
   - `lichess_bot.py` was MOVED to /home/user/puzzle-bot/ in earlier split — needs to
     be copied back and re-wired for Stonefish v1 (NettlesomeBot + MaiaPolicyEngine
     + WeakenedStockfishBot). Existing `GameHandler` + `StockfishPool` + event loop
     are reusable; per-game bot construction needs rewriting.
   - Test against `maia9` (Maia 1900 on Lichess) first, then real humans

## Environment

- Stockfish at `/usr/games/stockfish` (Stockfish 16, supports `UCI_LimitStrength` +
  `UCI_Elo` in range [1320, 3190])
- lc0 at `/usr/local/bin/lc0` (v0.32.1, built from source with Eigen backend)
- Maia weights at `/home/user/.maia/maia-{1100,1500,1900}.pb.gz`
- Python 3.11; `chess>=1.0` (note: PyPI `chess` package fails to build without
  `--use-pep517`)
- All on Ubuntu in a sandboxed cloud env. GitHub access restricted to
  `chad-murphy-data/stonefish`. The `/home/user/puzzle-bot/` directory is the
  staged-but-unpushed sister repo containing the Maia three-tier beginner bot.

## Important context

- **The original Stonefish was a "divergence" bot** (picks moves with the largest
  gap between opponent's #1 and #2 best replies). We split out the beginner
  puzzle-bot variant to `/home/user/puzzle-bot/` (which the user needs to push
  to a separate GitHub repo `chad-murphy-data/puzzle-bot` themselves, since
  this container's git proxy is scoped only to `chad-murphy-data/stonefish`).
- **The current "Stonefish v1" is a hybrid**: keeps the divergence concept
  (the puzzle filter targets P_maia close to 0.5 = max divergence) but adds
  Maia-informed EV scoring, give-back, and endgame-by-solve-rate mechanics.
- **The mechanic the user keeps coming back to**: chess.com-style "!" moments
  (one move good, others lose). The puzzle filter naturally produces those
  via the gap_ratio constraint.

## Open questions / things to think about

1. **Is the WeakenedSF baseline appropriate strength?** Maia 1900 plays
   differently from SF-throttled-to-1900. Real 1900 humans different again.
   Will likely need iteration once we see real Maia/human games.

2. **Does the give-back feel right?** Conceptually it's "the bot blew it
   after you found the puzzle." In real play this might feel cheesy if it's
   too obvious. Watch for this in Lichess tests.

3. **Hardness-scaled give-back** is unbuilt — user mentioned wanting harder
   puzzles (lower P_maia) to give more reward when found. Easy to add later
   (scale `post_trap_duration` inversely with `P_maia_top`).

4. **Reverse-Stonefish** (probabilistic SF#1 miss at the bot's own clear-best
   moments) is reserved for an ELO-tunable weakened mode. Currently commented
   out, defaulting to SF#1. Add when needed.

5. **No tests for the actual game-playing classes beyond QA.** `test_quick.py`
   is the only pytest-able test, and it just exercises the old divergence
   scoring. Worth adding unit tests for NettlesomeBot's mode-selection logic.

## How to run things

```bash
# QA against synthetic tester (controllable find rate, deterministic coin)
python3 qa_tester.py 15 10 0.3 0.5 0.7 0.9

# QA against real Maia 1900 (stochastic baseline, noisier)
python3 qa_test.py 30 10

# Single-game trace vs CoinFlipTester at find_prob=0.6 (Stonefish=white)
python3 trace_tester.py w 0.6 10 500

# Single-game trace vs Maia 1900 (Stonefish=white)
python3 trace_game.py w 10

# Head-to-head tournament (whatever __main__ in engine.py is configured for)
python3 engine.py 10 10
```

Most scripts take `(num_games, depth)` as args. Default depth 10 is fast enough.

## Commit conventions

The git log has structured messages explaining each change. Recent commits worth
reading to understand the design:

- `5199027` WeakenedStockfishBot: nerf Stonefish baseline to actual ~1900 Elo
- `ba5fb8e` CoinFlipTesterBot: deterministic per-position coin (hash-based)
- `258fde4` Endgame mode: tie precision to opponent's puzzle solve rate
- `67b678f` NettlesomeBot: add endgame conversion + reverse-Stonefish
- `6aa505c` Stonefish v1: bump give-back target_delta to 0.7
- `ef578bf` Stonefish v1: switch baseline to Equilibrium SF
- `1244d34` Puzzle filter: soften thresholds + add post-trap weakness lever

Read those messages to see *why* each piece is the way it is.

## To kick off

1. **Verify the env works**: `python3 -c "import chess; from engine import *; from maia_policy import *; print('ok')"`
2. **Run the full QA sweep** with the new WeakenedSF baseline (command above)
3. **Compare results** to the EquilibriumBaselineBot sweep above
4. **If the curve still looks good**: move on to `qa_test.py 30 10` (real Maia)
5. **If the curve degrades**: the give-back at Elo 1500 might be too weak;
   try `target_elo=1700` or revert give-back to EquilibriumBaselineBot

Good luck. The system works — most of the remaining work is real-opponent
validation and (eventually) Lichess deployment.

"""
Stonefish baseline-noise diagnostic
====================================
At target_elo=1900, Stockfish's own randomized weakening can introduce
move-to-move eval swings that have nothing to do with the puzzle mechanic.
This script consumes the per-move JSON written by qa_tester.py and
quantifies: (a) how much eval Stonefish "leaks" on non-trap moves
(baseline noise), (b) the signed eval impact of each trap moment, and
(c) whether per-game find_rate predicts per-game outcome.

Usage:
    python3 analyze_noise.py results/qa_tester_n30_d10.json

The headline numbers:
    Pearson r(find_rate, outcome)  -- if near 0, baseline noise drowns
                                       out the trap signal at p=0.5.
    SNR = median |trap_impact| / median |non-trap drift per move|
                                    -- ratio of per-event signal to
                                       per-move noise. >1 means traps
                                       move the eval more than baseline
                                       drift does on average.
"""

import json
import math
import sys
from collections import Counter
from statistics import mean, median, pstdev


def trap_impact(moment):
    """Signed eval impact of a single trap moment, in pawns from
    Stonefish's perspective.

    - Opp found SF #1 reply (rank=1): Stonefish paid `eval_cost` and got
      nothing back -> impact = -eval_cost
    - Opp missed (rank>=2): they played a worse-by-at-least-`gap` move,
      net for Stonefish = gap - eval_cost. (Slightly approximate because
      rank=3 may give back even more than gap_1_2, but gap_1_2 is a
      conservative floor.)
    - Rank unknown / unset: impact unknown, treat as 0.
    """
    rank = moment.get("opponent_rank")
    cost = moment.get("eval_cost", 0.0)
    gap = moment.get("gap_1_2", 0.0)
    if rank is None:
        return 0.0
    if rank == 1:
        return -cost
    return gap - cost


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def per_game_stats(g):
    moves_log = g.get("moves_log", []) or []
    moments_data = g.get("moments_data", []) or []

    by_mode = Counter(m["mode"] for m in moves_log)
    non_trap_costs = [m["eval_cost"] for m in moves_log if m["mode"] != "trap"]
    baseline_costs = [m["eval_cost"] for m in moves_log if m["mode"] == "baseline"]
    give_back_costs = [m["eval_cost"] for m in moves_log if m["mode"] == "give_back"]

    # For EquilibriumMaintainer baselines: "drift from target" is the
    # real noise metric. eval_cost is misleading because the maintainer
    # INTENTIONALLY plays sub-SF moves to hit target. What we actually
    # care about is whether played_eval lands near target_eval -- if so,
    # the baseline is silent and only traps move the score.
    baseline_drift_from_target = []
    for m in moves_log:
        if m["mode"] != "baseline":
            continue
        t = m.get("target_eval")
        if t is None:
            continue
        baseline_drift_from_target.append(m["played_eval"] - t)

    impacts = [trap_impact(m) for m in moments_data]
    signed_total = sum(impacts)
    abs_total = sum(abs(i) for i in impacts)

    return {
        "g": g["g"],
        "outcome": g["bot_score"],
        "find_rate": g.get("find_rate"),
        "moments": g["moments"],
        "found": g["found"],
        "moves": g["moves"],
        "n_baseline": by_mode.get("baseline", 0),
        "n_give_back": by_mode.get("give_back", 0),
        "n_endgame_precise": by_mode.get("endgame_precise", 0),
        "non_trap_drift_total": sum(non_trap_costs),
        "non_trap_drift_per_move": (mean(non_trap_costs) if non_trap_costs else 0.0),
        "non_trap_drift_abs_sum": sum(abs(c) for c in non_trap_costs),
        "baseline_drift_total": sum(baseline_costs),
        "baseline_drift_per_move": (mean(baseline_costs) if baseline_costs else 0.0),
        "give_back_drift_total": sum(give_back_costs),
        # EquilibriumMaintainer-specific: signed and absolute drift from target
        "baseline_target_drift_signed": sum(baseline_drift_from_target),
        "baseline_target_drift_abs_sum": sum(abs(d) for d in baseline_drift_from_target),
        "baseline_target_drift_per_move": (mean(baseline_drift_from_target)
                                            if baseline_drift_from_target else 0.0),
        "n_baseline_with_target": len(baseline_drift_from_target),
        "trap_signed_total": signed_total,
        "trap_abs_total": abs_total,
        "trap_impacts": impacts,
    }


def analyze_bucket(prob, bucket):
    games = bucket["games"]
    rows = [per_game_stats(g) for g in games]
    n = len(rows)

    wins = sum(1 for r in rows if r["outcome"] == 1.0)
    losses = sum(1 for r in rows if r["outcome"] == 0.0)
    draws = n - wins - losses

    # Pearson r between find_rate and outcome (excluding games with no moments)
    paired = [(r["find_rate"], r["outcome"]) for r in rows
              if r["find_rate"] is not None]
    r_find_outcome = (pearson([x for x, _ in paired], [y for _, y in paired])
                      if len(paired) >= 2 else float("nan"))

    # Pearson r between trap_signed_total and outcome
    paired_t = [(r["trap_signed_total"], r["outcome"]) for r in rows]
    r_trap_outcome = (pearson([x for x, _ in paired_t], [y for _, y in paired_t])
                      if len(paired_t) >= 2 else float("nan"))

    # Pearson r between baseline_drift_total and outcome
    paired_b = [(r["baseline_drift_total"], r["outcome"]) for r in rows]
    r_drift_outcome = (pearson([x for x, _ in paired_b], [y for _, y in paired_b])
                       if len(paired_b) >= 2 else float("nan"))

    # SNR: per-game median absolute trap impact vs per-move median non-trap drift
    per_game_trap_abs = [r["trap_abs_total"] for r in rows if r["moments"] > 0]
    per_move_drift_abs = []
    for g in games:
        for m in g.get("moves_log", []):
            if m["mode"] != "trap":
                per_move_drift_abs.append(abs(m["eval_cost"]))

    med_trap_per_game = median(per_game_trap_abs) if per_game_trap_abs else 0.0
    med_drift_per_move = median(per_move_drift_abs) if per_move_drift_abs else 0.0

    # Per-event normalised SNR: median |impact per trap| vs median |cost per non-trap move|
    per_trap_abs = []
    for r in rows:
        per_trap_abs.extend(abs(i) for i in r["trap_impacts"])
    med_trap_per_event = median(per_trap_abs) if per_trap_abs else 0.0

    snr_per_event = (med_trap_per_event / med_drift_per_move
                     if med_drift_per_move > 0 else float("inf"))

    print()
    print("=" * 72)
    print(f"BUCKET: find_prob = {prob}   (n={n} games)")
    print("=" * 72)
    print(f"Outcome:               W-D-L = {wins}-{draws}-{losses}   "
          f"Stonefish score = {(wins + 0.5*draws)/n:.1%}")
    print()
    print("Correlation (Pearson r):")
    print(f"  find_rate     ~ outcome   : r = {r_find_outcome:+.3f}   "
          f"(want strongly NEGATIVE for the trap mechanic to work)")
    print(f"  trap_signed   ~ outcome   : r = {r_trap_outcome:+.3f}   "
          f"(want POSITIVE; bigger trap-net = win)")
    print(f"  baseline_drift~ outcome   : r = {r_drift_outcome:+.3f}   "
          f"(want NEAR ZERO; if strong, drift is driving outcomes)")
    print()
    print("Signal vs noise (pawns):")
    print(f"  median |trap impact| per event:    {med_trap_per_event:.3f}")
    print(f"  median |non-trap drift| per move:  {med_drift_per_move:.3f}")
    print(f"  SNR (per-event / per-move):        {snr_per_event:.2f}x")
    print(f"  median |trap impact| per game:     {med_trap_per_game:.3f}")

    # EquilibriumMaintainer view: drift from target. Only meaningful if
    # target_eval was logged (i.e., baseline supports set_equilibrium).
    target_drift_abs = []
    for g in games:
        for m in g.get("moves_log", []):
            if m["mode"] != "baseline":
                continue
            t = m.get("target_eval")
            if t is None:
                continue
            target_drift_abs.append(abs(m["played_eval"] - t))
    if target_drift_abs:
        med_target_drift = median(target_drift_abs)
        mean_target_drift = mean(target_drift_abs)
        snr_vs_target = (med_trap_per_event / med_target_drift
                         if med_target_drift > 0 else float("inf"))
        med_per_game_target = median([r["baseline_target_drift_abs_sum"]
                                       for r in rows
                                       if r["n_baseline_with_target"] > 0]
                                      or [0.0])
        print()
        print("EquilibriumMaintainer view (drift from target_eval):")
        print(f"  median |drift from target| per move:  {med_target_drift:.3f}")
        print(f"  mean   |drift from target| per move:  {mean_target_drift:.3f}")
        print(f"  SNR (|trap event| / |target drift|):  {snr_vs_target:.2f}x")
        print(f"  median |target drift| sum per game:   {med_per_game_target:.3f}")
    print()
    print("Per-game breakdown:")
    print(f"  {'G':>2}  {'outc':>4}  {'find':>5}  {'mom':>3}  {'fnd':>3}  "
          f"{'mvs':>3}  {'base':>4}  {'gvbk':>4}  {'egPr':>4}  "
          f"{'base_drift':>10}  {'trap_net':>9}")
    for r in rows:
        fr = (f"{r['find_rate']*100:>4.0f}%" if r['find_rate'] is not None else "  n/a")
        outc = ("W" if r["outcome"] == 1.0 else
                "L" if r["outcome"] == 0.0 else "D")
        print(f"  {r['g']:>2}  {outc:>4}  {fr:>5}  {r['moments']:>3}  "
              f"{r['found']:>3}  {r['moves']:>3}  "
              f"{r['n_baseline']:>4}  {r['n_give_back']:>4}  "
              f"{r['n_endgame_precise']:>4}  "
              f"{r['baseline_drift_total']:>+10.2f}  "
              f"{r['trap_signed_total']:>+9.2f}")

    return {
        "prob": prob, "n": n, "wins": wins, "draws": draws, "losses": losses,
        "r_find_outcome": r_find_outcome,
        "r_trap_outcome": r_trap_outcome,
        "r_drift_outcome": r_drift_outcome,
        "med_trap_per_event": med_trap_per_event,
        "med_drift_per_move": med_drift_per_move,
        "snr_per_event": snr_per_event,
    }


def main(path):
    data = json.load(open(path))
    print(f"Analyzing: {path}")
    print(f"  games/bucket = {data['num_games_per_prob']}   "
          f"depth = {data['depth']}   probs = {data['probs']}")

    summaries = []
    for prob_str, bucket in data["results"].items():
        prob = float(prob_str)
        summary = analyze_bucket(prob, bucket)
        summaries.append(summary)

    # Aggregate summary
    if len(summaries) > 1:
        print()
        print("=" * 72)
        print("AGGREGATE")
        print("=" * 72)
        print(f"{'prob':>5} {'W-D-L':>9} {'r(find,out)':>12} {'r(trap,out)':>12} "
              f"{'r(drift,out)':>13} {'SNR/event':>10}")
        for s in summaries:
            wdl = f"{s['wins']}-{s['draws']}-{s['losses']}"
            print(f"{s['prob']:>5.2f} {wdl:>9} {s['r_find_outcome']:>+12.3f} "
                  f"{s['r_trap_outcome']:>+12.3f} {s['r_drift_outcome']:>+13.3f} "
                  f"{s['snr_per_event']:>9.2f}x")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/qa_tester_n30_d10.json"
    main(path)

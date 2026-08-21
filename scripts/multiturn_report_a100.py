#!/usr/bin/env python3
"""Per-turn ever-correct + FF/FT/TF/TT transition report for the A100
follower-mode multiturn run, requested by PI (2026-08-21) to match the
primary sm_120 server's own report format.

Pure read-only analysis over results/eval/multiturn_state_a100.json's
per-chain "history" list (already recorded by scripts/multiturn.py's
cmd_evaluate, unmodified) -- this script does not touch the orchestrator,
protocol, or chain state at all, same category as scripts/analyze.py.

Definitions:
- ever_correct@turn N: a chain counts if ANY turn 1..N in its history has
  correctness=True (cumulative, not just the current turn's verdict --
  matches "회복률" framing in CLAUDE.md's 지표 list: once a chain has ever
  produced a correct kernel, that's a recovery, even if a later optimize-
  phase turn regresses it back to incorrect).
- FF/FT/TF/TT@turn N: transition of consecutive turns' correctness,
  (turn N-1 -> turn N): False->False, False->True, True->False, True->True.
  Chains with no turn N-1..N pair yet (not that far along) are excluded from
  that turn's transition counts, not silently zero-filled.

Usage:
    python scripts/multiturn_report_a100.py --state results/eval/multiturn_state_a100.json
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(REPO_ROOT / "results" / "eval" / "multiturn_state_a100.json"))
    args = ap.parse_args()

    state = json.loads(Path(args.state).read_text())
    chains = list(state["chains"].values())
    max_turn = max(c["turn"] for c in chains)

    print(f"total chains: {len(chains)}, max turn reached: {max_turn}\n")

    print(f"{'turn':4s} {'ever_correct':>12s} {'FF':>6s} {'FT':>6s} {'TF':>6s} {'TT':>6s} "
          f"{'terminated_by_this_turn':>24s}")
    for n in range(1, max_turn + 1):
        ever_correct = 0
        ff = ft = tf = tt = 0
        terminated = 0
        for c in chains:
            hist = {h["turn"]: h for h in c["history"]}
            if n not in hist:
                continue
            # ever-correct through turn n
            if any(hist[t]["correctness"] for t in hist if t <= n):
                ever_correct += 1
            # transition n-1 -> n
            if (n - 1) in hist:
                prev, cur = hist[n - 1]["correctness"], hist[n]["correctness"]
                if not prev and not cur:
                    ff += 1
                elif not prev and cur:
                    ft += 1
                elif prev and not cur:
                    tf += 1
                else:
                    tt += 1
            if c["terminated"] and c["turn"] == n:
                terminated += 1
        print(f"{n:4d} {ever_correct:12d} {ff:6d} {ft:6d} {tf:6d} {tt:6d} {terminated:24d}")

    print("\nby language|model, ever_correct at max turn reached:")
    import collections
    by = collections.defaultdict(lambda: {"n": 0, "ever_correct": 0})
    for c in chains:
        k = f"{c['language']}|{c['model'].split('/')[-1]}"
        by[k]["n"] += 1
        hist = c["history"]
        if any(h["correctness"] for h in hist):
            by[k]["ever_correct"] += 1
    for k in sorted(by):
        d = by[k]
        print(f"  {k:45s} {d['ever_correct']:4d}/{d['n']:4d} ({100*d['ever_correct']/d['n']:.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""deep-turn-probe final deliverables (EXPLORATORY_PROTOCOL.md #7), computed
directly from results/exploratory/state_arm{A,B}.json's history[] arrays --
not from a report script's printed aggregates (established discipline: recompute
from source). Read-only w.r.t. the state files; multiturn.py untouched.

Writes:
  results/exploratory/fig_armA_turn_correct.csv   -- per-chain turn x correct
  results/exploratory/fig_armB_speedup_trajectory.csv -- per-chain best-speedup
      running trajectory + gap vs eager (torch.compile gap not available --
      no torch.compile baseline was ever measured in this experiment or the
      main experiment, so that column is left empty rather than fabricated).

Usage:
    python scripts/exploratory_export_figs.py
"""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
STATE_A = REPO_ROOT / "results" / "exploratory" / "state_armA.json"
STATE_B = REPO_ROOT / "results" / "exploratory" / "state_armB.json"
OUT_A = REPO_ROOT / "results" / "exploratory" / "fig_armA_turn_correct.csv"
OUT_B = REPO_ROOT / "results" / "exploratory" / "fig_armB_speedup_trajectory.csv"


def export_armA():
    state = json.loads(STATE_A.read_text())
    rows = []
    for cid, c in sorted(state["chains"].items()):
        for h in c["history"]:
            rows.append({
                "chain_id": cid,
                "task": c["task"],
                "model": c["model"],
                "sample_index": c["sample_index"],
                "turn": h["turn"],
                "compiled": int(bool(h.get("compiled"))),
                "correct": int(bool(h.get("correctness"))),
                "gen_status": h.get("gen_status"),
            })
    with open(OUT_A, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chain_id", "task", "model", "sample_index",
                                           "turn", "compiled", "correct", "gen_status"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_A} ({len(rows)} rows, {len(state['chains'])} chains)")


def export_armB():
    state = json.loads(STATE_B.read_text())
    rows = []
    for cid, c in sorted(state["chains"].items()):
        seeded = c["sample_index"] >= 900
        best_so_far = None
        for h in c["history"]:
            turn_speedup = h.get("speedup")
            # turn 1 history entries (both fresh and seeded) don't carry a
            # "speedup" key the way turn>=2 evaluate-appended entries do;
            # seeded chains' turn-1 speedup lives in the chain's own
            # best_speedup (set at init from armB_seed_timing.json), and
            # fresh chains are never correct at turn 1 in this dataset.
            if h["turn"] == 1 and h.get("correctness") and turn_speedup is None:
                turn_speedup = c["best_speedup"] if c["best_turn"] == 1 else None
            if h.get("correctness") and turn_speedup is not None:
                if best_so_far is None or turn_speedup > best_so_far:
                    best_so_far = turn_speedup
            rows.append({
                "chain_id": cid,
                "task": c["task"],
                "model": c["model"],
                "sample_index": c["sample_index"],
                "seeded_from_main_experiment": int(seeded),
                "turn": h["turn"],
                "correct": int(bool(h.get("correctness"))),
                "turn_speedup": turn_speedup if turn_speedup is not None else "",
                "best_speedup_so_far": best_so_far if best_so_far is not None else "",
                "gap_vs_eager_x": (best_so_far - 1.0) if best_so_far is not None else "",
                "gap_vs_torch_compile_x": "",  # not measured in this or the main experiment
            })
    with open(OUT_B, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chain_id", "task", "model", "sample_index",
                                           "seeded_from_main_experiment", "turn", "correct",
                                           "turn_speedup", "best_speedup_so_far",
                                           "gap_vs_eager_x", "gap_vs_torch_compile_x"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_B} ({len(rows)} rows, {len(state['chains'])} chains)")


if __name__ == "__main__":
    export_armA()
    export_armB()

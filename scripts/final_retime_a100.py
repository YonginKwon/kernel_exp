#!/usr/bin/env python3
"""A100-server final re-timing pass over every best-so-far correct kernel in
the Phase 2 multiturn run, requested by PI (2026-08-22) once turn 10 (k_max)
closed out the run and the GPU sat idle.

Follower-mode note (~/kernel-lang-2x2/CLAUDE.md Phase 2): does not modify
evaluate.py/multiturn.py -- reuses evaluate.py's own time_one_isolated()
(unmodified) with the identical retry pattern evaluate.py's run_timing()
already uses (TIMING_MAX_ATTEMPTS=3, fresh spawn subprocess per attempt),
GPU-exclusive (assert_gpu_exclusive(), same check evaluate.py enforces at
every entry point).

Source-kernel selection per chain (from results/eval/multiturn_state_a100.json):
  - best_code / best_turn, when set -- the chain has an established best-so-
    far speedup already (multiturn.py's own tracking, cmd_evaluate lines
    ~355-357/408-412: best_code/best_turn/best_speedup are only ever set
    together, atomically, when a timing attempt succeeds).
  - otherwise, when correctness is True at the chain's current turn and
    best_code is still None -- this is the "correct but never successfully
    timed" case (reproducible torch.cuda.synchronize() segfault; see this
    server's 2026-08-22 turn-10 report, 6 such chains). Use the chain's
    current `code` field, since these chains are permanently stuck at the
    turn where correctness first held (cmd_generate's optimize-phase +
    last_timing-is-None skip condition never lets them advance).

Writes results/eval/final_retime_a100.json, keyed by chain_id, incrementally
(checkpoint after every chain) so a crash/interrupt only loses in-flight
work, same resumability pattern as evaluate.py's run_timing(). Does NOT
mutate multiturn_state_a100.json -- that file stays exactly what
scripts/multiturn.py (unmodified) produced; this is a separate, additive
measurement pass.

Usage:
    source .venv/bin/activate  # + this server's own CUDA_HOME/PATH/CXX exports
    python scripts/final_retime_a100.py \
        --state results/eval/multiturn_state_a100.json \
        --out results/eval/final_retime_a100.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "KernelBench" / "src"))

import evaluate as ev  # noqa: E402

TIMING_MAX_ATTEMPTS = ev.TIMING_MAX_ATTEMPTS  # 3, same constant evaluate.py uses


def select_targets(chains: dict) -> list:
    targets = []
    for cid, c in chains.items():
        if c.get("best_code") is not None:
            targets.append({
                "chain_id": cid, "task": c["task"], "language": c["language"],
                "condition": c["condition"], "model": c["model"],
                "sample_index": c["sample_index"], "code": c["best_code"],
                "source_turn": c["best_turn"], "source": "best_code",
                "prior_best_speedup": c["best_speedup"],
            })
        elif c["correctness"] and c.get("code"):
            targets.append({
                "chain_id": cid, "task": c["task"], "language": c["language"],
                "condition": c["condition"], "model": c["model"],
                "sample_index": c["sample_index"], "code": c["code"],
                "source_turn": c["turn"], "source": "current_code_never_timed",
                "prior_best_speedup": None,
            })
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(REPO_ROOT / "results" / "eval" / "multiturn_state_a100.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "eval" / "final_retime_a100.json"))
    args = ap.parse_args()

    ev.assert_gpu_exclusive()

    state = json.loads(Path(args.state).read_text())
    chains = state["chains"]
    targets = select_targets(chains)

    out_path = Path(args.out)
    records = {}
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        records = prior.get("records", {})
        print(f"[retime] resuming, {len(records)} chain(s) already in checkpoint")

    todo = [t for t in targets if t["chain_id"] not in records]
    print(f"[retime] {len(targets)} chain(s) with a correct kernel total, "
          f"{len(records)} already done, {len(todo)} to time")

    for i, t in enumerate(todo, 1):
        rec = {"task": t["task"], "language": t["language"]}
        result = None
        for attempt in range(1, TIMING_MAX_ATTEMPTS + 1):
            result = ev.time_one_isolated(rec, t["code"])
            if "timing_exception" not in result:
                if attempt > 1:
                    result["recovered_after_attempts"] = attempt
                break
            print(f"[retime] {i}/{len(todo)} {t['chain_id']} attempt {attempt}/{TIMING_MAX_ATTEMPTS} "
                  f"-> {result['timing_exception'][:100]}"
                  + (" -- retrying" if attempt < TIMING_MAX_ATTEMPTS else ""))

        entry = {**{k: v for k, v in t.items() if k != "code"}, **result}
        records[t["chain_id"]] = entry

        if "timing_exception" in result:
            print(f"[retime] {i}/{len(todo)} {t['chain_id']} -> STILL FAILING after "
                  f"{TIMING_MAX_ATTEMPTS} attempts: {result['timing_exception'][:100]}")
        else:
            print(f"[retime] {i}/{len(todo)} {t['chain_id']} -> "
                  f"kernel={result['kernel_ms']['median']:.4g}ms "
                  f"baseline={result['baseline_ms']['median']:.4g}ms "
                  f"speedup={result['speedup']:.3g}x")

        out_path.write_text(json.dumps({"records": records, "status": "in_progress"}, indent=2, default=str))

    n_ok = sum(1 for r in records.values() if "timing_exception" not in r)
    n_fail = len(records) - n_ok
    out_path.write_text(json.dumps({"records": records, "status": "done"}, indent=2, default=str))
    print(f"[retime] done. {n_ok}/{len(records)} chains successfully timed, {n_fail} still failing.")


if __name__ == "__main__":
    main()

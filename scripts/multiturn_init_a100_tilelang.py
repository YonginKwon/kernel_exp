#!/usr/bin/env python3
"""A100-server chain-state bootstrap for the Phase 2 SCOPE EXPANSION's
TileLang track (PI instruction 2026-08-25: bring this server's coverage up
to parity with PRO 6000's full 4-language 2x2 design; TileLang was
explicitly excluded from this server's original Phase 1/2 scope).

Same non-invasive-reuse discipline as scripts/multiturn_init_a100.py (see
that file's own header for the full rationale): duplicates ONLY cmd_init's
chain-building logic against this server's own new turn-1 source
(results/eval/eval_a100_tilelang_0shot.json, produced by
scripts/phase2_expand_evaluate_turn1.sh), re-importing chain_id/K_MAX/
clean_32_tasks from multiturn.py/analyze.py directly. Writes to a SEPARATE
state file (PI decision 2026-08-25) so the already-completed 960-chain
multiturn_state_a100.json is never touched.

Usage:
    source .venv/bin/activate  # + this server's CUDA_HOME/PATH/CXX exports
    python scripts/multiturn_init_a100_tilelang.py \\
        --state results/eval/multiturn_state_a100_tilelang.json

Chain count check: 32 clean tasks x 1 language (tilelang) x 5 samples x
2 models = 320.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from multiturn import chain_id, K_MAX  # noqa: E402
from analyze import clean_32_tasks  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
EVAL_DIR = REPO_ROOT / "results" / "eval"
TURN1_SOURCE = EVAL_DIR / "eval_a100_tilelang_0shot.json"
TIMING_SOURCE = EVAL_DIR / "timing_a100_tilelang.json"
STATE_DEFAULT = EVAL_DIR / "multiturn_state_a100_tilelang.json"


def _load_turn1_records():
    data = json.loads(TURN1_SOURCE.read_text())["records"]
    clean32 = set(clean_32_tasks())
    return [r for r in data if r["condition"] == "0shot" and r["language"] == "tilelang"
            and r["task"] in clean32]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=str(STATE_DEFAULT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    state_path = Path(args.state)
    if state_path.exists() and not args.force:
        print(f"[refuse] {state_path} already exists -- pass --force to reinitialize "
              f"(this discards any turn>1 progress).", file=sys.stderr)
        return 1

    records = _load_turn1_records()
    print(f"[init-a100-tilelang] {len(records)} turn-1 chains (32-task 0shot, tilelang only)")

    chains = {}
    for r in records:
        raw = json.loads((RAW_DIR / r["path"]).read_text())
        cid = chain_id(r["language"], r["condition"], r["task"], r["model"], r["sample_index"])
        chains[cid] = {
            "chain_id": cid,
            "language": r["language"], "condition": r["condition"], "task": r["task"],
            "model": r["model"], "sample_index": r["sample_index"],
            "original_prompt": raw["prompt"],
            "turn": 1,
            "phase": "optimize" if r.get("correctness") else "repair",
            "code": raw.get("parsed_code"),
            "gen_status": r["gen_status"],
            "compiled": bool(r.get("compiled")), "correctness": bool(r.get("correctness")),
            "metadata": r.get("metadata") or {},
            "last_timing": None,
            "best_speedup": None, "best_code": None, "best_turn": None,
            "no_improve_streak": 0,
            "terminated": False, "termination_reason": None,
            "history": [{"turn": 1, "compiled": bool(r.get("compiled")),
                         "correctness": bool(r.get("correctness")), "gen_status": r["gen_status"]}],
        }
        if chains[cid]["correctness"]:
            chains[cid]["best_code"] = raw.get("parsed_code")
            chains[cid]["best_turn"] = 1

    backfilled, missing = 0, []
    if TIMING_SOURCE.exists():
        timing_by_path = {r["path"]: r for r in json.loads(TIMING_SOURCE.read_text())["records"]
                           if "timing_exception" not in r}
        for r in records:
            if not r.get("correctness"):
                continue
            cid = chain_id(r["language"], r["condition"], r["task"], r["model"], r["sample_index"])
            t = timing_by_path.get(r["path"])
            if t:
                chains[cid]["last_timing"] = {"kernel_ms": t["kernel_ms"]["median"],
                                               "baseline_ms": t["baseline_ms"]["median"],
                                               "speedup": t["speedup"]}
                chains[cid]["best_speedup"] = t["speedup"]
                backfilled += 1
            else:
                missing.append(cid)
    else:
        print(f"[init-a100-tilelang] {TIMING_SOURCE} not found -- no turn-1 timing backfill; "
              f"multiturn.py cmd_evaluate's catch-up pass will measure lazily.", file=sys.stderr)
        missing = [c["chain_id"] for c in chains.values() if c["phase"] == "optimize"]

    n_optimize = sum(1 for c in chains.values() if c["phase"] == "optimize")
    print(f"[init-a100-tilelang] backfilled last_timing for {backfilled}/{n_optimize} "
          f"turn-1-correct chains")
    if missing:
        print(f"[init-a100-tilelang] {len(missing)} turn-1-correct chain(s) have no timing yet "
              f"(lazy catch-up): {missing}")

    state = {"chains": chains, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k_max": K_MAX}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"[init-a100-tilelang] wrote {state_path} ({len(chains)} chains, all at turn 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

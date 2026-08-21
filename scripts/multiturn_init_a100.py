#!/usr/bin/env python3
"""A100-server entry point for scripts/multiturn.py's chain-state bootstrap.

Follower-mode note (~/kernel-lang-2x2/CLAUDE.md Phase 2): this server does
not modify multiturn.py itself (the §3.4 orchestrator/harness) -- its
cmd_init() hardcodes the PRO 6000 box's own turn-1 source files
(results/eval/full_run_20260819.json + docinject_run_20260820T072056.json,
conditions "0shot"/"docinject"). This server's turn-1 data is a different,
already-complete file (results/eval/eval_a100_full.json, Phase 1: 3
languages -- cuda/ptx/triton, no tilelang -- 0shot only, no docinject,
37 tasks) that cmd_init has no way to know about. Rather than edit the
shared orchestrator, this script duplicates ONLY cmd_init's chain-building
logic (structurally identical, verified against multiturn.py 7fe6d43) against
this server's own turn-1 file, and re-imports everything else (chain_id,
constants, the 32-task clean-task filter) directly from multiturn.py /
analyze.py so there is exactly one definition of the actual protocol logic.

Once this writes its state file, scripts/multiturn.py's own generate/
evaluate/report subcommands run completely unmodified against it (they take
--state as a plain path, no PRO-6000-specific assumptions) -- only cmd_init
needed this workaround.

Usage:
    source .venv/bin/activate  # + this server's own CUDA_HOME/PATH/CXX exports
    python scripts/multiturn_init_a100.py --state results/eval/multiturn_state_a100.json

Chain count check: 32 clean tasks x 3 languages x 5 samples x 2 models = 960
(no docinject, no tilelang on this server -- PI-confirmed scope, CLAUDE.md
Phase 2 "PI 결정 2026-08-20" item 3).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from multiturn import chain_id, K_MAX  # noqa: E402
from analyze import FLAWED_TASKS, clean_32_tasks  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
EVAL_DIR = REPO_ROOT / "results" / "eval"
TURN1_SOURCE = EVAL_DIR / "eval_a100_full.json"
TIMING_SOURCE = EVAL_DIR / "timing_a100.json"
STATE_DEFAULT = EVAL_DIR / "multiturn_state_a100.json"


def _load_turn1_records():
    data = json.loads(TURN1_SOURCE.read_text())["records"]
    clean32 = set(clean_32_tasks())
    return [r for r in data if r["condition"] == "0shot" and r["task"] in clean32]


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
    print(f"[init-a100] {len(records)} turn-1 chains (32-task 0shot, "
          f"{len(FLAWED_TASKS)} flawed tasks excluded, no docinject/tilelang on this server)")

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
        print(f"[init-a100] WARNING: {TIMING_SOURCE} not found -- no turn-1 timing backfill. "
              f"Run: python scripts/evaluate.py --timing --timing-sources {TURN1_SOURCE} "
              f"--out {TIMING_SOURCE}", file=sys.stderr)
        missing = [c["chain_id"] for c in chains.values() if c["phase"] == "optimize"]

    n_optimize = sum(1 for c in chains.values() if c["phase"] == "optimize")
    print(f"[init-a100] backfilled last_timing for {backfilled}/{n_optimize} turn-1-correct chains "
          f"from {TIMING_SOURCE.name}")
    if missing:
        print(f"[init-a100] {len(missing)} turn-1-correct chain(s) have NO timing yet (will be "
              f"measured lazily before their first optimize-phase turn, see multiturn.py "
              f"cmd_evaluate's catch-up pass): {missing}")

    state = {"chains": chains, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k_max": K_MAX}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"[init-a100] wrote {state_path} ({len(chains)} chains, all at turn 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

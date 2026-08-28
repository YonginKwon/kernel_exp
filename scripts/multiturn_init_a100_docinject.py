#!/usr/bin/env python3
"""A100-server chain-state bootstrap for the Phase 2 SCOPE EXPANSION's
docinject ablation track (PI instruction 2026-08-25: replicate PRO 6000's
full experiment, including the document-injection ablation, which had never
been run on this server for ANY language before this expansion).

Same non-invasive-reuse discipline as scripts/multiturn_init_a100.py: reads
this server's own new turn-1 sources (results/eval/eval_a100_docinject_
{cuda,ptx,triton,tilelang}.json, produced by
scripts/phase2_expand_evaluate_turn1.sh), filters each to the 17
audit-clean docinject tasks via analyze.py's docinject_clean_tasks() (single
source of truth, same list PRO 6000's own cmd_init uses), and re-imports
chain_id/K_MAX from multiturn.py directly. Writes to a SEPARATE state file
(PI decision 2026-08-25) so multiturn_state_a100.json (0shot, 3 languages,
already complete) and multiturn_state_a100_tilelang.json (0shot, tilelang)
are never touched.

Usage:
    source .venv/bin/activate  # + this server's CUDA_HOME/PATH/CXX exports
    python scripts/multiturn_init_a100_docinject.py \\
        --state results/eval/multiturn_state_a100_docinject.json

Chain count check: 17 clean tasks x 4 languages x 5 samples x 2 models = 680.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from multiturn import chain_id, K_MAX  # noqa: E402
from analyze import docinject_clean_tasks  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
EVAL_DIR = REPO_ROOT / "results" / "eval"
LANGUAGES = ["cuda", "ptx", "triton", "tilelang"]
TIMING_SOURCE = EVAL_DIR / "timing_a100_docinject.json"
STATE_DEFAULT = EVAL_DIR / "multiturn_state_a100_docinject.json"


def _load_turn1_records():
    clean17 = set(docinject_clean_tasks())
    out = []
    for lang in LANGUAGES:
        src = EVAL_DIR / f"eval_a100_docinject_{lang}.json"
        data = json.loads(src.read_text())["records"]
        out += [r for r in data if r["condition"] == "docinject" and r["language"] == lang
                and r["task"] in clean17]
    return out


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
    print(f"[init-a100-docinject] {len(records)} turn-1 chains (17-task docinject, "
          f"{LANGUAGES})")

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
        print(f"[init-a100-docinject] {TIMING_SOURCE} not found -- no turn-1 timing backfill; "
              f"multiturn.py cmd_evaluate's catch-up pass will measure lazily.", file=sys.stderr)
        missing = [c["chain_id"] for c in chains.values() if c["phase"] == "optimize"]

    n_optimize = sum(1 for c in chains.values() if c["phase"] == "optimize")
    print(f"[init-a100-docinject] backfilled last_timing for {backfilled}/{n_optimize} "
          f"turn-1-correct chains")
    if missing:
        print(f"[init-a100-docinject] {len(missing)} turn-1-correct chain(s) have no timing yet "
              f"(lazy catch-up): {missing}")

    state = {"chains": chains, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k_max": K_MAX}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"[init-a100-docinject] wrote {state_path} ({len(chains)} chains, all at turn 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""deep-turn-probe exploratory-experiment entry point for scripts/multiturn.py's
chain-state bootstrap (EXPLORATORY_PROTOCOL.md). Builds state_armA.json (PTX
repair-endurance, 20 fresh chains) and state_armB.json (Triton optimization-
endurance, 3 chains seeded from the concluded main experiment's best_code +
5 fresh chains).

Follower-mode note: does not modify multiturn.py -- duplicates only
cmd_init's chain-building shape (verified against multiturn.py) against this
experiment's own isolated turn-1 sources
(results/exploratory/eval_turn1.json, results/exploratory/armB_seed_timing.json),
and re-imports chain_id from multiturn.py directly so there is exactly one
definition of chain-id formatting. k_max=100 is baked into both state files
(#3.4's cmd_evaluate already reads state.get("k_max", K_MAX), so no code
change is needed to honor EXPLORATORY_PROTOCOL.md deviation (1)).

Usage:
    python scripts/exploratory_init.py \
        --eval-turn1 results/exploratory/eval_turn1.json \
        --armb-seed-timing results/exploratory/armB_seed_timing.json \
        --raw-dir results/exploratory/raw \
        --state-armA results/exploratory/state_armA.json \
        --state-armB results/exploratory/state_armB.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from multiturn import chain_id  # noqa: E402

K_MAX_EXPLORATORY = 100

ARM_A_TASKS = {"19_ReLU", "1_Square_matrix_multiplication_"}
ARM_A_LANGUAGE = "ptx"
ARM_B_FRESH_TASK = "82_conv_depthwise_2D_square_input_square_kernel"
ARM_B_FRESH_LANGUAGE = "triton"


def _chain_from_turn1_record(r, raw_dir: Path):
    raw_path = raw_dir / r["path"] if not str(r["path"]).startswith(str(raw_dir)) else Path(r["path"])
    raw = json.loads(raw_path.read_text())
    cid = chain_id(r["language"], r["condition"], r["task"], r["model"], r["sample_index"])
    chain = {
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
    if chain["correctness"]:
        chain["best_code"] = raw.get("parsed_code")
        chain["best_turn"] = 1
    return chain


def _seed_chain(seed_key, seed, exploratory_sample_index):
    """Builds an Arm B seed chain directly at turn 1 / optimize phase from a
    main-experiment best_code, using this experiment's own fresh timing
    (armB_seed_timing.json) as its turn-1 last_timing/best_speedup anchor.
    sample_index is offset to 900+ to make provenance unambiguous in
    chain_id (never collides with this experiment's own fresh 0-4 samples)."""
    t = seed["timing"]
    cid = chain_id(seed["language"], seed["condition"], seed["task"], seed["model"], exploratory_sample_index)
    if "timing_exception" in t:
        raise RuntimeError(f"seed {seed_key} has no valid timing: {t['timing_exception']}")
    last_timing = {"kernel_ms": t["kernel_ms"]["median"], "baseline_ms": t["baseline_ms"]["median"],
                   "speedup": t["speedup"]}
    return {
        "chain_id": cid,
        "language": seed["language"], "condition": seed["condition"], "task": seed["task"],
        "model": seed["model"], "sample_index": exploratory_sample_index,
        "original_prompt": seed["original_prompt"],
        "turn": 1,
        "phase": "optimize",
        "code": None,
        "gen_status": "seeded_from_main_experiment",
        "compiled": True, "correctness": True,
        "metadata": {
            "seeded_from_main_experiment": True,
            "main_experiment_chain_id": seed["chain_id"],
            "main_experiment_sample_index": seed["sample_index"],
            "main_experiment_best_speedup": seed["main_best_speedup"],
            "main_experiment_best_turn": seed["main_best_turn"],
        },
        "last_timing": last_timing,
        "best_speedup": t["speedup"], "best_code": None, "best_turn": 1,
        "no_improve_streak": 0,
        "terminated": False, "termination_reason": None,
        "history": [{"turn": 1, "compiled": True, "correctness": True,
                     "gen_status": "seeded_from_main_experiment"}],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-turn1", required=True)
    ap.add_argument("--armb-seed-timing", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--state-armA", required=True)
    ap.add_argument("--state-armB", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    records = json.loads(Path(args.eval_turn1).read_text())["records"]

    # best_code fixups: for the codepath's best_code to be set on init we need
    # the *code that was actually compiled* -- eval_turn1.json's records don't
    # embed it, so re-read parsed_code from the raw sample file per record
    # (handled inside _chain_from_turn1_record).

    armA_records = [r for r in records if r["language"] == ARM_A_LANGUAGE and r["task"] in ARM_A_TASKS]
    armB_fresh_records = [r for r in records
                           if r["language"] == ARM_B_FRESH_LANGUAGE and r["task"] == ARM_B_FRESH_TASK]

    print(f"[init-exploratory] Arm A: {len(armA_records)} turn-1 PTX records "
          f"({sorted(ARM_A_TASKS)}, expect 20)")
    print(f"[init-exploratory] Arm B fresh: {len(armB_fresh_records)} turn-1 Triton records "
          f"({ARM_B_FRESH_TASK} x Qwen, expect 5)")

    armA_chains = {}
    for r in armA_records:
        c = _chain_from_turn1_record(r, raw_dir)
        armA_chains[c["chain_id"]] = c

    armB_chains = {}
    for r in armB_fresh_records:
        c = _chain_from_turn1_record(r, raw_dir)
        armB_chains[c["chain_id"]] = c

    seeds = json.loads(Path(args.armb_seed_timing).read_text())
    for i, (key, seed) in enumerate(sorted(seeds.items())):
        c = _seed_chain(key, seed, exploratory_sample_index=900 + i)
        armB_chains[c["chain_id"]] = c
        print(f"[init-exploratory] Arm B seed: {key} -> {c['chain_id']} "
              f"(fresh best_speedup={c['best_speedup']:.4f})")

    assert len(armA_chains) == 20, f"expected 20 Arm A chains, got {len(armA_chains)}"
    assert len(armB_chains) == 8, f"expected 8 Arm B chains, got {len(armB_chains)}"

    for label, chains, out_path in [("A", armA_chains, args.state_armA), ("B", armB_chains, args.state_armB)]:
        out_path = Path(out_path)
        if out_path.exists() and not args.force:
            print(f"[refuse] {out_path} already exists -- pass --force to reinitialize.", file=sys.stderr)
            return 1
        state = {"chains": chains, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "k_max": K_MAX_EXPLORATORY, "experiment": "deep-turn-probe", "arm": label}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=2, default=str))
        print(f"[init-exploratory] wrote {out_path} ({len(chains)} chains, k_max={K_MAX_EXPLORATORY})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

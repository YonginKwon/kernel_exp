#!/usr/bin/env python3
"""One-off: fresh GPU-exclusive timing for deep-turn-probe Arm B's 3 seed
chains (best_code copied read-only from the concluded main experiment's
results-a100:results/eval/multiturn_state_a100.json -- see
EXPLORATORY_PROTOCOL.md #5). Per protocol, this experiment re-measures under
its own GPU-exclusive conditions rather than reusing the main experiment's
recorded speedup.

Reuses scripts/evaluate.py's time_one_isolated()/assert_gpu_exclusive()
completely unmodified -- no harness edit.

Usage:
    python scripts/exploratory_seed_armB_timing.py \
        --seeds /path/to/armB_seeds.json --out results/exploratory/armB_seed_timing.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate as ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev.assert_gpu_exclusive()

    seeds = json.loads(Path(args.seeds).read_text())
    results = {}
    for key, s in seeds.items():
        record = {"language": s["language"], "task": s["task"]}
        print(f"[timing] {key} ({s['task']} / {s['model']}) ...", flush=True)
        t = ev.time_one_isolated(record, s["best_code"], timeout=180)
        results[key] = {**{k: v for k, v in s.items() if k != "best_code"}, "timing": t}
        if "timing_exception" in t:
            print(f"[timing] {key} FAILED: {t['timing_exception']}", flush=True)
        else:
            print(f"[timing] {key} speedup={t['speedup']:.4f} "
                  f"(main-exp was {s['main_best_speedup']:.4f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())

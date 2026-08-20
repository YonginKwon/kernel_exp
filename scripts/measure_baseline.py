#!/usr/bin/env python3
"""PyTorch-eager fp16 baseline latency for tasks/level1_subset.json's 37
tasks, on this evaluation machine. Protocol (kernel-lang-2x2 CLAUDE.md
"타이밍 프로토콜" / this server's Phase 2 prep step 2): warmup 25, trials
100, median, torch.cuda.synchronize per trial, GPU exclusive, re-measured on
every machine rather than reusing literature/other-hardware numbers.

Reuses third_party/KernelBench's own kernelbench.timing.measure_ref_program_time
(cuda_event method) for the actual timed forward passes -- same code path
evaluate.py's correctness checks already trust for model/inputs construction.
That function's own get_timing_stats() reports mean/std/min/max but not
median; monkeypatched at runtime here (never edits the vendored file) to add
it, the same pattern evaluate.py already uses for its -std= stripping fix.

Never writes to results/raw or results/eval/eval_*.json (CLAUDE.md rule:
those are read-only once written) -- this writes a new, separate file,
results/eval/baseline_<tag>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "KernelBench" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate import load_tasks, load_reference_code  # noqa: E402
from evaluate import assert_gpu_exclusive  # noqa: E402

import kernelbench.timing as kb_timing  # noqa: E402

_orig_get_timing_stats = kb_timing.get_timing_stats


def _get_timing_stats_with_median(elapsed_times, device=None):
    stats = _orig_get_timing_stats(elapsed_times, device=device)
    stats["median"] = float(f"{np.median(elapsed_times):.4g}")
    return stats


kb_timing.get_timing_stats = _get_timing_stats_with_median

from kernelbench.timing import measure_ref_program_time  # noqa: E402


def env_fingerprint() -> dict:
    info = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import subprocess
        info["driver_version"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "eval" / "baseline_a100_fp16.json"))
    ap.add_argument("--hardware-tag", default="A100-80GB, sm_80",
                     help="value for the output's env.hardware_tag field")
    args = ap.parse_args()

    assert_gpu_exclusive()

    tasks = load_tasks()
    print(f"[baseline] {len(tasks)} tasks, warmup={args.warmup} trials={args.trials} "
          f"precision={args.precision} device=cuda:0", file=sys.stderr)

    results = {}
    failed = []
    for i, task in enumerate(tasks):
        name = task["name"]
        src = load_reference_code(name)
        print(f"[baseline] ({i + 1}/{len(tasks)}) {name}", file=sys.stderr)
        stats = measure_ref_program_time(
            ref_arch_name=name,
            ref_arch_src=src,
            num_warmup=args.warmup,
            num_trials=args.trials,
            discard_first=1,
            timing_method="cuda_event",
            use_torch_compile=False,
            device=torch.device("cuda:0"),
            verbose=False,
            precision=args.precision,
        )
        if stats is None:
            failed.append(name)
            print(f"[baseline]   FAILED", file=sys.stderr)
        else:
            print(f"[baseline]   median={stats['median']}ms mean={stats['mean']}ms", file=sys.stderr)
        results[name] = stats

    env = env_fingerprint()
    env["hardware_tag"] = args.hardware_tag
    out = {
        "protocol": {
            "warmup": args.warmup, "trials": args.trials, "stat": "median",
            "precision": args.precision, "timing_method": "cuda_event",
            "discard_first": 1,
        },
        "env": env,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": len(tasks),
        "failed_count": len(failed),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[baseline] wrote {out_path}", file=sys.stderr)

    if failed:
        print(f"[baseline] {len(failed)}/{len(tasks)} tasks FAILED: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tolerance audit (PI request, 2026-08-20, pre-multi-turn): for each of the
37 selected tasks, compute how large the reference model's output elements
typically are and compare that to the fixed fp16 correctness tolerance
(atol=rtol=1e-2, tasks/SELECTION.md #4.1, third_party/KernelBench's
get_tolerance_for_precision). torch.allclose checks |a-b| <= atol + rtol*|b|;
for a task whose real outputs are much smaller than atol, the rtol term is
negligible and the check degenerates to "anything near zero passes" --
exactly the failure mode found in 23_Softmax (paper/RESULTS_REPORT_20260820.md
#7-1): a kernel that computes ~0.3% of the required output still passed
correctness because expected values were ~2.5e-6, far below atol=1e-2.

CPU-only by design (PI instruction) -- this doesn't need the GPU or its
exclusivity gate; it only runs the *reference* PyTorch model once per task to
read off output magnitudes, no custom kernel involved.

Usage:
    source scripts/env.sh && source .venv/bin/activate
    python scripts/audit_tolerance.py [--out results/eval/tolerance_audit.json]
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "KernelBench" / "src"))

import torch  # noqa: E402
from kernelbench import eval as kb_eval  # noqa: E402

LEVEL1_DIR = REPO_ROOT / "third_party" / "KernelBench" / "KernelBench" / "level1"
TASKS_PATH = REPO_ROOT / "tasks" / "level1_subset.json"
PRECISION = torch.float16  # tasks/SELECTION.md #4.1
ATOL = kb_eval.get_tolerance_for_precision(PRECISION)  # 1e-2, read from the harness, not hardcoded
FLAG_RATIO = 10.0  # PI instruction: flag tasks where atol exceeds output magnitude by >10x


def audit_one(task: str, device: torch.device) -> dict:
    src = (LEVEL1_DIR / f"{task}.py").read_text()
    context = {}
    Model, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(src, context)

    def cast_input(x):
        # NOTE (2026-08-20, found while running this audit): the real harness's
        # own kernelbench.eval._process_input_tensor() casts EVERY get_inputs()
        # tensor to PRECISION unconditionally, including integer class-index
        # targets (e.g. 95_CrossEntropyLoss's torch.randint(...) label tensor,
        # which nn.CrossEntropyLoss requires as Long). That would break the
        # REFERENCE model's own forward pass the moment any custom kernel for
        # that task actually compiles (0/79 real samples have so far, all
        # failing earlier for unrelated reasons -- so this hasn't manifested
        # yet, but it will under the multi-turn repair protocol's extra
        # attempts). This audit reproduces the harness's precision-casting
        # faithfully for floating tensors but skips it for integer/bool
        # tensors so the audit itself doesn't inherit the same bug --
        # flagged separately in the result as `label_cast_bug_risk`, not
        # silently patched into the harness (that's a PI call, out of scope
        # for "measure output magnitude").
        if not torch.is_tensor(x):
            return x
        if x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool):
            return x.to(device=device)
        return x.to(dtype=PRECISION).to(device=device)

    kb_eval.set_seed(42)
    init_inputs = get_init_inputs()
    init_inputs = [cast_input(x) for x in init_inputs]
    kb_eval.set_seed(42)
    model = Model(*init_inputs).to(device=device, dtype=PRECISION)
    kb_eval.set_seed(42)
    inputs = get_inputs()
    has_int_input = any(torch.is_tensor(x) and x.dtype in
                         (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool)
                         for x in inputs)
    inputs = [cast_input(x) for x in inputs]

    with torch.no_grad():
        output = model(*inputs)

    abs_out = output.abs().float()  # fp32 for the stats themselves -- just measurement, not the check
    n_inf = torch.isinf(output).sum().item()
    n_nan = torch.isnan(output).sum().item()
    finite = abs_out[torch.isfinite(abs_out)]
    median_abs = finite.median().item() if finite.numel() else float("nan")
    mean_abs = finite.mean().item() if finite.numel() else float("nan")
    max_abs = finite.max().item() if finite.numel() else float("nan")
    min_abs_nonzero = finite[finite > 0].min().item() if (finite > 0).any() else 0.0

    # ratio uses median (robust to a few large outlier elements, e.g. in
    # reductions) as "typical" output magnitude -- mean_abs kept alongside
    # for transparency since some tasks' distributions are heavily skewed.
    ratio = ATOL / median_abs if median_abs > 0 else float("inf")

    return {
        "task": task,
        "output_shape": list(output.shape),
        "output_numel": output.numel(),
        "median_abs": median_abs,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "min_abs_nonzero": min_abs_nonzero,
        "n_inf": n_inf,
        "n_nan": n_nan,
        "atol": ATOL,
        "atol_over_median_ratio": ratio,
        "flagged": ratio > FLAG_RATIO,
        "reference_overflow_flagged": n_inf > 0 or n_nan > 0,  # separate category -- ratio metric can't see this (ATOL/inf == 0, looks "fine")
        "label_cast_bug_risk": has_int_input,  # see cast_input()'s note above
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "eval" / "tolerance_audit.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # PI instruction: CPU work
    tasks_data = json.loads(TASKS_PATH.read_text())
    all_tasks = sorted(t for fam in tasks_data["families"].values() for t in fam)
    print(f"[audit] {len(all_tasks)} tasks, device={device}, precision={PRECISION}, atol=rtol={ATOL}")

    results = []
    for i, task in enumerate(all_tasks, 1):
        try:
            r = audit_one(task, device)
        except Exception as e:
            r = {"task": task, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "error" in r:
            print(f"[audit] {i:2d}/{len(all_tasks)} {task:45s} ERROR: {r['error'][:100]}")
        else:
            flags = []
            if r["flagged"]:
                flags.append("TOO-TOLERANT")
            if r["reference_overflow_flagged"]:
                flags.append("REFERENCE-OVERFLOW")
            flagtxt = f" *** {'/'.join(flags)} ***" if flags else ""
            print(f"[audit] {i:2d}/{len(all_tasks)} {task:45s} "
                  f"median|out|={r['median_abs']:.3g} atol/median={r['atol_over_median_ratio']:.3g} "
                  f"n_inf={r['n_inf']} n_nan={r['n_nan']}{flagtxt}")

    flagged = [r for r in results if r.get("flagged")]
    overflow = [r for r in results if r.get("reference_overflow_flagged")]
    errored = [r for r in results if "error" in r]
    print(f"\n=== {len(flagged)}/{len(all_tasks)} task(s) TOO-TOLERANT (atol=rtol={ATOL} exceeds "
          f"median|output| by >{FLAG_RATIO}x) ===")
    for r in sorted(flagged, key=lambda r: -r["atol_over_median_ratio"]):
        print(f"  {r['task']:45s} median|out|={r['median_abs']:.3g} ratio={r['atol_over_median_ratio']:.3g}x")
    print(f"\n=== {len(overflow)}/{len(all_tasks)} task(s) REFERENCE-OVERFLOW "
          f"(reference model itself produces inf/nan under fp16) ===")
    for r in overflow:
        print(f"  {r['task']:45s} n_inf={r['n_inf']} n_nan={r['n_nan']} / {r['output_numel']} elements")
    print(f"\n{len(errored)} error(s):")
    for r in errored:
        print(f"  {r['task']:45s} {r['error'][:120]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"atol": ATOL, "flag_ratio_threshold": FLAG_RATIO, "results": results}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

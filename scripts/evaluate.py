#!/usr/bin/env python3
"""Compile + correctness evaluation for generated kernels (CLAUDE.md's
directory structure names this scripts/evaluate.py: "컴파일 + 정확성 +
타이밍" -- timing is not implemented yet, out of scope for the pilot go/no-go
check this was first built for; compile+correctness is).

Reads results/raw/ (written by scripts/generate.py) -- NEVER writes there and
NEVER modifies a generated sample (CLAUDE.md rule 1: results/ is read-only
LLM output). Writes only to results/eval/.

Backend dispatch:
- cuda / triton / tilelang: KernelBench's own eval_kernel_against_ref()
  (third_party/KernelBench), backend=<language>, precision=fp16 (the fixed
  protocol decision, tasks/SELECTION.md #4.1). This is the same harness
  scripts/smoke_kernelbench_harness.py verified during the 8/10 milestone.
- ptx: KernelBench doesn't have a "ptx" backend (it isn't one of vLLM's...
  err, one of eval.py's recognized `backend` strings), because there's no
  such thing as a generic "PTX execution backend" -- our harness
  (harness/ptx/ptx_harness.py) exposes ptx_load/ptx_launch as free functions
  a generated ModelNew calls directly (PROMPT_SPEC.md §2's PTX block). To
  make `exec()`-ing that generated code resolve those names, this script
  pre-seeds the exec namespace with them (mirrors what KernelBench's own
  load_custom_model() does with a plain context dict) and then reuses
  KernelBench's set_seed/load_original_model_and_inputs/
  run_and_check_correctness helpers directly for everything else, so the
  correctness methodology (tolerance, seeding, trial count) is identical to
  the other 3 languages -- no separate "PTX-only" judgment logic anywhere.

GPU exclusivity (CLAUDE.md rule 6, 2026-08-19, hard rule -- NO bypass flag):
this script refuses to run at all if nvidia-smi shows any other process on
the GPU (most likely a vLLM generation server left running). Generation and
evaluation sharing a GPU would contaminate timing measurements; the rule is
enforced even though this pilot build doesn't measure timing yet, so nobody
has to remember to add the check later once timing does land.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "KernelBench" / "src"))
sys.path.insert(0, str(REPO_ROOT / "harness" / "ptx"))

import torch  # noqa: E402
from kernelbench import eval as kb_eval  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
EVAL_DIR = REPO_ROOT / "results" / "eval"
LEVEL1_DIR = REPO_ROOT / "third_party" / "KernelBench" / "KernelBench" / "level1"

PRECISION = torch.float16  # tasks/SELECTION.md #4.1: all 4 languages unified to fp16


def find_samples(language=None, condition=None, model_dir=None, task=None, raw_dir=RAW_DIR):
    """Yield (path, record) for every results/raw/.../sample_*.json matching
    the given filters (each optional)."""
    for p in sorted(raw_dir.glob("*/*/*/*/*sample_*.json")):
        # path shape: raw_dir/<language>/<condition>/<task>/<model_dir>/sample_N.json
        lang, cond, task_name, model_d = p.parts[-5], p.parts[-4], p.parts[-3], p.parts[-2]
        if language and lang != language:
            continue
        if condition and cond != condition:
            continue
        if task and task_name != task:
            continue
        if model_dir and model_d != model_dir:
            continue
        yield p, json.loads(p.read_text())


def assert_gpu_exclusive():
    """CLAUDE.md rule 6: hard-refuse to run if the GPU has any other process
    on it (typically a vLLM server left running from generation). No bypass
    flag exists here on purpose -- do not add one; fix the actual conflict
    (stop the vLLM server) instead."""
    try:
        mem_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        used_str, util_str = [x.strip() for x in mem_out.split(",")]
        used, util = int(used_str), int(util_str)
    except Exception as e:
        print(f"[refuse] could not query nvidia-smi to verify GPU exclusivity: "
              f"{type(e).__name__}: {e}. Refusing to run rather than assume the "
              f"GPU is free.", file=sys.stderr)
        sys.exit(1)

    apps_out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()

    if used > 500 or util > 5 or apps_out:
        print(f"[refuse] GPU is not exclusive to this evaluation run "
              f"({used}MiB used, {util}% util). CLAUDE.md rule 6: evaluate.py "
              f"must not run while generation (vLLM) or anything else is on the "
              f"GPU -- timing contamination. This is a hard rule with no bypass "
              f"flag; stop the other process first (e.g. "
              f"'scripts/serve_local.sh --stop').", file=sys.stderr)
        if apps_out:
            print(apps_out, file=sys.stderr)
        sys.exit(1)


def load_reference(task_name: str) -> str:
    path = LEVEL1_DIR / f"{task_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"reference source not found: {path}")
    return path.read_text()


def eval_ptx(code: str, ref_src: str, device):
    """Correctness check for the PTX track, sharing KernelBench's own
    seeding/tolerance/trial logic (kb_eval.run_and_check_correctness) so the
    judgment methodology is identical to cuda/triton/tilelang -- only the
    *loading* step (exec with ptx_load/ptx_launch pre-seeded) is PTX-specific."""
    from ptx_harness import ptx_load, ptx_launch, PTXCompileError

    context = {"ptx_load": ptx_load, "ptx_launch": ptx_launch, "__builtins__": __builtins__}
    try:
        compile(code, "<ptx_generated>", "exec")
        exec(code, context)
    except SyntaxError as e:
        return kb_eval.KernelExecResult(compiled=False, metadata={"compilation_error_name": "SyntaxError", "compilation_error": str(e)})
    except PTXCompileError as e:
        return kb_eval.KernelExecResult(compiled=False, metadata={"compilation_error_name": "PTXCompileError", "compilation_error": e.stderr})
    except Exception as e:
        return kb_eval.KernelExecResult(compiled=False, metadata={"compilation_error_name": type(e).__name__, "compilation_error": str(e)})

    ModelNew = context.get("ModelNew")
    if ModelNew is None:
        return kb_eval.KernelExecResult(compiled=False, metadata={"compilation_error_name": "NoModelNew", "compilation_error": "ModelNew not defined in generated code"})

    ref_context = {}
    Model, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(ref_src, ref_context)
    kb_eval.set_seed(42)
    init_inputs = get_init_inputs()

    try:
        with torch.no_grad():
            kb_eval.set_seed(42)
            original_model = Model(*init_inputs).to(device=device, dtype=PRECISION)
            kb_eval.set_seed(42)
            new_model = ModelNew(*init_inputs).to(device=device, dtype=PRECISION)
    except Exception as e:
        return kb_eval.KernelExecResult(compiled=True, correctness=False,
                                         metadata={"runtime_error_name": type(e).__name__, "runtime_error": str(e)})

    return kb_eval.run_and_check_correctness(
        original_model, new_model, get_inputs, metadata={}, num_correct_trials=1,
        verbose=False, seed=42, device=device, backend="ptx", precision=PRECISION,
    )


def eval_one(record: dict, device) -> dict:
    if record["status"] != "generated":
        return {"compiled": False, "correctness": False, "eval_skipped_reason": record["status"]}

    code = record["parsed_code"]
    ref_src = load_reference(record["task"])
    language = record["language"]

    try:
        if language in ("cuda", "triton", "tilelang"):
            result = kb_eval.eval_kernel_against_ref(
                original_model_src=ref_src, custom_model_src=code, backend=language,
                precision=PRECISION, measure_performance=False, verbose=False, device=device,
            )
            if result is None:
                return {"compiled": False, "correctness": False, "eval_skipped_reason": "lock_retry"}
        elif language == "ptx":
            result = eval_ptx(code, ref_src, device)
        else:
            raise ValueError(f"unknown language {language!r}")
    except Exception as e:
        return {"compiled": False, "correctness": False,
                "eval_exception": f"{type(e).__name__}: {e}"}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "metadata": {k: str(v)[:2000] for k, v in result.metadata.items()},
    }


def run_eval(language=None, condition=None, model_dir=None, task=None, raw_dir=RAW_DIR):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = []
    for path, record in find_samples(language, condition, model_dir, task, raw_dir):
        result = eval_one(record, device)
        entry = {
            "path": str(path.relative_to(raw_dir)),
            "task": record["task"], "task_family": record.get("task_family"),
            "language": record["language"], "condition": record["condition"],
            "model": record["model"], "sample_index": record["sample_index"],
            "gen_status": record["status"],
            **result,
        }
        records.append(entry)
        status_str = "SKIP" if not result.get("compiled") and "eval_skipped_reason" in result else \
            ("PASS" if result.get("correctness") else ("COMPILED_WRONG" if result.get("compiled") else "COMPILE_FAIL"))
        print(f"[eval] {record['language']:8s} {record['task']:35s} sample={record['sample_index']} "
              f"gen={record['status']:15s} -> {status_str}")
    return records


def summarize(records: list) -> dict:
    by_lang = {}
    for r in records:
        lang = r["language"]
        d = by_lang.setdefault(lang, {"n": 0, "generated": 0, "compiled": 0, "correct": 0})
        d["n"] += 1
        if r["gen_status"] == "generated":
            d["generated"] += 1
        if r.get("compiled"):
            d["compiled"] += 1
        if r.get("correctness"):
            d["correct"] += 1
    return by_lang


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", choices=["cuda", "triton", "ptx", "tilelang"], default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--model-dir", default=None, help="the model's sanitized dir name under results/raw/.../<task>/<model_dir>/")
    ap.add_argument("--task", default=None)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out", default=None, help="default: results/eval/eval_<timestamp>.json")
    args = ap.parse_args()

    assert_gpu_exclusive()

    records = run_eval(args.language, args.condition, args.model_dir, args.task, Path(args.raw_dir))
    summary = summarize(records)

    print("\n=== summary (compiled / correct out of n, generated=parsed-ok count) ===")
    for lang, d in sorted(summary.items()):
        print(f"  {lang:8s} n={d['n']:3d} generated={d['generated']:3d} "
              f"compiled={d['compiled']:3d} correct={d['correct']:3d}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else EVAL_DIR / f"eval_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

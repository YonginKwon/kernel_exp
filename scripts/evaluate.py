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
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "KernelBench" / "src"))
sys.path.insert(0, str(REPO_ROOT / "harness" / "ptx"))

# NOTE (2026-08-20, PI-approved -- see prompts/PROMPT_SPEC.md §7 change
# history): was 2000/4000. The 2026-08-19 full-run eval showed this cap
# routinely cut a CUDA sample's nvcc invocation off before the "error:"
# line -- 197/705 compile-failure records lost their real diagnostic this
# way, and the loss was asymmetric across models (172/181 for gpt-oss-120b
# vs. 25/124 for Qwen3-Coder-30B-A3B-Instruct), which would have distorted
# the paper's error-classification table by model. This is a pure logging
# cap: it does not change compiled/correctness verdicts or re-run any
# evaluation, only how much of the diagnostic text gets kept. Raised well
# above any observed real diagnostic length (max seen so far: 4000, at the
# old cap -- i.e. we don't yet know the true max, hence the generous margin).
COMPILE_ERROR_LOG_CAP = 20000

# NOTE (2026-08-19): tried monkeypatching subprocess.run globally here to
# force-capture nvcc/ninja/g++ output for every compile failure (torch's
# build-failure path sometimes raises a bare
# `RuntimeError("Error building extension 'name'")` with no compiler output
# attached -- confirmed the real diagnostic exists, it just doesn't always
# reach str(exception)). Reverted: across ~40 sequential compiles in one
# process it produced spurious "Ninja is required to load C++ extensions"
# failures on EVERY sample -- is_ninja_available()'s `except Exception:
# return False` was swallowing something (most likely descriptor/resource
# pressure from forcing capture_output=True on every subprocess.run call in
# the process, including internal ones this script doesn't care about) that
# only manifests at that scale, not in an isolated single-sample repro. Not
# worth the risk for the full ~1,480-sample run. When a compile failure's
# `metadata.compilation_error` is short and uninformative, reproduce that
# one sample in a fresh, isolated process instead (see the CUDA probe
# classification notes, tasks/ or CLAUDE.md) rather than capturing globally.
_captured_subprocess_output: list[str] = []

import torch  # noqa: E402
import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

# NOTE (2026-08-19, PI-approved -- see prompts/PROMPT_SPEC.md §7 change
# history): 27.6% (51/185) of Qwen3-Coder-30B-A3B-Instruct's generated CUDA
# samples call load_inline(..., extra_cuda_cflags=[..., "-std=c++14"]) --
# boilerplate the model writes itself, unrelated to kernel correctness. This
# server's PyTorch/ATen requires C++17, and torch's cpp_extension only
# appends its own "-std=c++17" default when no "-std=" flag is already
# present in cuda_flags -- so the model's stale flag silently wins and every
# such sample fails to compile for a reason that has nothing to do with the
# kernel logic being tested (gpt-oss-120b: 0/185, doesn't do this). Same
# principle as the (not-triggered) architecture-flag branch from the CUDA
# probe: the harness owns toolchain-level flags, not the model. Strip any
# model-supplied -std= flag here so torch's own default (C++17) always
# applies; PROMPT_SPEC's CUDA block also now tells the model not to pass one.
def _strip_std_flags(flags):
    if not flags:
        return flags
    return [f for f in flags if not str(f).startswith("-std=")]


def _make_std_flag_stripping_wrapper(original_fn):
    def _wrapped(*args, **kwargs):
        if "extra_cflags" in kwargs:
            kwargs["extra_cflags"] = _strip_std_flags(kwargs["extra_cflags"])
        if "extra_cuda_cflags" in kwargs:
            kwargs["extra_cuda_cflags"] = _strip_std_flags(kwargs["extra_cuda_cflags"])
        return original_fn(*args, **kwargs)
    return _wrapped


_cpp_extension.load_inline = _make_std_flag_stripping_wrapper(_cpp_extension.load_inline)
_cpp_extension.load = _make_std_flag_stripping_wrapper(_cpp_extension.load)

from kernelbench import eval as kb_eval  # noqa: E402
from kernelbench import timing as kb_timing  # noqa: E402

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


def _attach_captured_output(metadata: dict):
    """Merge whatever subprocess.run captured (see the module-level monkeypatch
    above) into a compile-failure metadata dict in place -- torch's own
    exception message is sometimes just "Error building extension 'name'"
    with the real nvcc/g++ diagnostic never making it past the subprocess's
    stdout, so this is the only reliable way to keep it in results/eval/."""
    if not _captured_subprocess_output:
        return
    full = "\n".join(_captured_subprocess_output)[-COMPILE_ERROR_LOG_CAP:]
    existing = metadata.get("compilation_error", "")
    if len(full) > len(existing):
        metadata["compilation_error_full_stdout"] = full


def _recover_real_compile_error(code: str, language: str, build_dir: str | None = None) -> dict:
    """KernelBench's eval_kernel_against_ref() returns None (not a
    KernelExecResult) when it thinks compilation hit transient lock
    contention -- its own detection is `"lock" in str(exception)`, a loose
    substring match. Confirmed empirically (2026-08-19 CUDA probe, 7/40
    samples) that this also fires on genuine compile errors whose text
    happens to contain the word "lock" incidentally, silently discarding a
    real compile failure as "please retry" forever. This redoes the load
    directly (mirroring eval_kernel_against_ref's own backend dispatch) to
    capture the actual exception text instead of losing it."""
    context = {}
    try:
        if language in ("triton", "tilelang"):
            ModelNew, tf = kb_eval.load_custom_model_with_tempfile(code, entry_point="ModelNew")
            if tf:
                tf.close()
                os_remove_quiet(tf.name)
        else:
            kb_eval.load_custom_model(code, context, build_directory=build_dir)
    except Exception as e:
        return {"compilation_error_name": type(e).__name__,
                "compilation_error": str(e)[:COMPILE_ERROR_LOG_CAP],
                "recovered_from": "kernelbench_false_positive_lock_retry"}
    # Loading succeeded on retry with no exception -- genuinely transient.
    return None


def os_remove_quiet(path):
    try:
        os.remove(path)
    except OSError:
        pass


def eval_one(record: dict, device, precompiled_build_dir: str | None = None) -> dict:
    if record["status"] != "generated":
        return {"compiled": False, "correctness": False, "eval_skipped_reason": record["status"]}

    code = record["parsed_code"]
    ref_src = load_reference(record["task"])
    language = record["language"]

    # CUDA backend only: give every sample its own build directory instead of
    # PyTorch's shared default (~/.cache/torch_extensions/<...>/<name>/).
    # Different generated samples routinely pick the same generic extension
    # name (e.g. "relu_cuda", "matmul_ext") -- with a shared cache dir this
    # causes real cross-sample corruption (one sample's failed/partial build
    # leaves stale artifacts that a later, unrelated sample with the same
    # name then fails to load: "... .so: cannot open shared object file"),
    # not because that later sample's own code is broken. Confirmed
    # empirically during the 2026-08-19 CUDA probe. Triton/TileLang don't
    # need this -- they load via tempfile, not the named extension cache.
    #
    # IMPORTANT: each *attempt* below (initial try, retry, recovery) needs
    # its OWN fresh build_dir, not a shared one -- reusing one directory
    # across attempts leaves ninja's incremental-build state (.ninja_log)
    # from the first (failed) attempt in place, and a later attempt's ninja
    # invocation then treats stale/partial artifacts as "already built" and
    # no-ops instead of recompiling, producing a *different* failure
    # ("... .so: cannot open shared object file", because the .so was never
    # actually linked) that masks the real compiler error. Confirmed
    # empirically -- this is exactly what happened the first time this
    # retry logic was added, 2026-08-19.
    build_dirs = []
    _first_build_dir_call = [True]

    def fresh_build_dir():
        # P0-b (2026-08-20): if a precompiled build dir was handed in (see
        # precompile_cuda_batch()), the FIRST call reuses it -- ninja sees
        # identical source/flags already built and no-ops instead of
        # recompiling, so the nvcc cost happened earlier in parallel, not
        # here in the serialized GPU-touching pass. Every subsequent call
        # (this function's own internal retry/recovery attempts below)
        # still gets a genuinely fresh dir, exactly as before -- reusing a
        # dir across a *failed* attempt is the documented staleness bug
        # this project already hit once (see the retry comment above).
        if precompiled_build_dir is not None and _first_build_dir_call[0]:
            _first_build_dir_call[0] = False
            build_dirs.append(precompiled_build_dir)
            return precompiled_build_dir
        d = tempfile.mkdtemp(prefix="k2x2_eval_")
        build_dirs.append(d)
        return d

    _captured_subprocess_output.clear()
    try:
        if language in ("cuda", "triton", "tilelang"):
            result = kb_eval.eval_kernel_against_ref(
                original_model_src=ref_src, custom_model_src=code, backend=language,
                precision=PRECISION, measure_performance=False, verbose=False, device=device,
                build_dir=fresh_build_dir() if language == "cuda" else None,
            )
            if result is None:
                # Retry once with a FRESH build dir (genuine transient
                # contention would clear by now); fall back to capturing the
                # real error, also with a fresh dir, if it's still None.
                result = kb_eval.eval_kernel_against_ref(
                    original_model_src=ref_src, custom_model_src=code, backend=language,
                    precision=PRECISION, measure_performance=False, verbose=False, device=device,
                    build_dir=fresh_build_dir() if language == "cuda" else None,
                )
            if result is None:
                recovered = _recover_real_compile_error(
                    code, language, build_dir=fresh_build_dir() if language == "cuda" else None)
                if recovered is not None:
                    _attach_captured_output(recovered)
                    return {"compiled": False, "correctness": False, "metadata": recovered}
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
        for d in build_dirs:
            shutil.rmtree(d, ignore_errors=True)

    metadata = {k: str(v)[:COMPILE_ERROR_LOG_CAP] for k, v in result.metadata.items()}
    if not result.compiled:
        _attach_captured_output(metadata)
    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "metadata": metadata,
    }


def _eval_worker_entry(record, result_conn, precompiled_build_dir=None):
    """Runs in an isolated (spawn) subprocess -- see eval_one_isolated()."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        result = eval_one(record, device, precompiled_build_dir=precompiled_build_dir)
    except Exception as e:
        result = {"compiled": False, "correctness": False,
                   "eval_exception": f"{type(e).__name__}: {e}"}
    try:
        result_conn.send(result)
    finally:
        result_conn.close()


def eval_one_isolated(record: dict, timeout: int = 300, precompiled_build_dir: str | None = None) -> dict:
    """Runs eval_one() in a fresh subprocess (multiprocessing 'spawn').

    NOTE (2026-08-20): a genuinely buggy generated kernel can trigger a CUDA
    "illegal memory access" during the correctness check. That corrupts the
    CUDA context for the rest of the process -- confirmed empirically: the
    2026-08-19 full-run attempt hit exactly this on a Triton sample, and the
    NEXT sample's cleanup (`torch.cuda.empty_cache()` in eval_one's `finally`)
    raised an uncaught torch.AcceleratorError that killed the whole
    evaluate.py process. Since results were only written to disk once, at
    the very end, that lost all progress on the ~1,480-sample run. Isolating
    each sample in its own subprocess means a corrupted context (or any other
    crash/hang) only kills that one sample's subprocess -- the parent process
    (and the incremental checkpoint in run_eval) is unaffected. A subprocess
    that dies or exceeds `timeout` without sending a result is recorded as an
    eval_exception, not silently dropped."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_eval_worker_entry, args=(record, child_conn, precompiled_build_dir))
    proc.start()
    child_conn.close()

    result = None
    if parent_conn.poll(timeout):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    parent_conn.close()

    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if result is not None:
        return result
    return {"compiled": False, "correctness": False,
            "eval_exception": f"eval subprocess produced no result "
                               f"(exitcode={proc.exitcode}) -- likely a CUDA "
                               f"context crash (e.g. illegal memory access) "
                               f"or a hang killed after a {timeout}s timeout"}


PRECOMPILE_BUILD_ROOT = REPO_ROOT / "results" / ".eval_build_cache"
# PI instruction 2026-08-20 (P0-b) set nvcc workers >= 16 for throughput.
# PI instruction 2026-08-21 (post 14:25 power-loss crash, GPU power cap
# rejected -- 600W stays): halved to 8 to cut the CPU compile-phase power
# peak instead. Throughput/wall-clock trade-off accepted deliberately.
PRECOMPILE_WORKERS = 8


def deterministic_build_dir(path: str) -> Path:
    """Stable per-sample build dir (unlike eval_one's own tempfile.mkdtemp,
    which is deliberately fresh every call -- see its docstring). Needs to be
    stable ONLY so a parallel precompile pass and the later serialized real
    pass agree on the same directory for the SAME sample's SAME code; each
    is still a dedicated directory, no cross-sample sharing."""
    h = hashlib.sha1(path.encode()).hexdigest()[:16]
    d = PRECOMPILE_BUILD_ROOT / h
    d.mkdir(parents=True, exist_ok=True)
    return d


def _precompile_worker_entry(record, build_dir, result_conn):
    """Runs in an isolated (spawn) subprocess. CUDA only: compiles + dlopens
    the generated extension (kb_eval.load_custom_model) WITHOUT ever
    instantiating a model or touching a CUDA device -- torch.utils.
    cpp_extension's build+import step is pure host-side compiler/linker work;
    a CUDA context is only created lazily by the first actual device op
    (e.g. `.to(device="cuda")`), which this deliberately never calls. That is
    what makes it safe to run many of these concurrently on one GPU-having
    machine: nothing here touches the GPU, so P0-b's requirement (decouple
    compile from the GPU stage, nvcc workers >= 16) can run genuinely in
    parallel while the real eval loop stays serialized for GPU safety.

    On failure, captures the real compiler error here (mirroring
    _recover_real_compile_error's exact error-formatting shape) instead of
    just a bool -- a confirmed compile failure needs no GPU-touching work at
    all, so the caller can build the sample's final eval record directly
    from this and skip the serialized pass entirely for it (measured: without
    this, failing samples were compiled TWICE -- once here, once again in
    the serial pass -- which made small failure-heavy batches net SLOWER
    with precompile on than off; this fixes that)."""
    try:
        context = {}
        kb_eval.load_custom_model(record["parsed_code"], context, build_directory=str(build_dir))
        if context.get("ModelNew") is None:
            result_conn.send({"ok": False, "metadata": {
                "compilation_error_name": "NoModelNew",
                "compilation_error": "ModelNew not defined in generated code"}})
        else:
            result_conn.send({"ok": True, "metadata": None})
    except Exception as e:
        result_conn.send({"ok": False, "metadata": {
            "compilation_error_name": type(e).__name__,
            "compilation_error": str(e)[:COMPILE_ERROR_LOG_CAP]}})
    finally:
        result_conn.close()


def _precompile_one(record, build_dir, timeout=300):
    """Returns {"ok": True} (compiled, build_dir is reusable), {"ok": False,
    "metadata": {...}} (confirmed compile failure, real error captured), or
    None (subprocess crashed/timed out -- caller falls back to treating this
    sample as not-precompiled, i.e. normal fresh-tempdir behavior in the
    serial pass, same as if precompile had never run for it)."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_precompile_worker_entry, args=(record, build_dir, child_conn))
    proc.start()
    child_conn.close()
    result = None
    if parent_conn.poll(timeout):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    parent_conn.close()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return result


def precompile_cuda_batch(records_with_paths: list, max_workers: int = PRECOMPILE_WORKERS):
    """Precompiles every CUDA sample's extension in parallel (up to
    max_workers concurrent nvcc/ninja invocations, each in its own isolated
    subprocess + build dir -- pure CPU work, no GPU touch, see
    _precompile_worker_entry).

    Returns (build_dir_map, failure_entries):
    - build_dir_map: {path: build_dir} for samples that compiled -- the
      serial pass reuses this dir (ninja no-ops on the unchanged build, so
      the nvcc cost already happened here, in parallel) and proceeds
      straight to the GPU-touching correctness check.
    - failure_entries: {path: metadata_dict} for samples with a CONFIRMED
      compile failure -- the caller builds the final eval record directly
      from this and skips the serial pass for it entirely (no second
      compile attempt, no subprocess).
    A sample missing from BOTH dicts means its precompile subprocess crashed
    or timed out (rare) -- the serial pass treats it exactly as if
    precompile had never run (fresh tempdir, full retry logic intact).

    records_with_paths: list of (path_str, record) for CUDA, gen_status=="generated" only."""
    if not records_with_paths:
        return {}, {}
    print(f"[precompile] {len(records_with_paths)} CUDA sample(s), "
          f"{max_workers} parallel nvcc worker(s)...")
    build_dir_map = {}
    failure_entries = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for path, record in records_with_paths:
            build_dir = deterministic_build_dir(path)
            futures[pool.submit(_precompile_one, record, build_dir)] = (path, build_dir)
        done = 0
        for fut in as_completed(futures):
            path, build_dir = futures[fut]
            done += 1
            try:
                result = fut.result()
            except Exception:
                result = None
            if result is None:
                shutil.rmtree(build_dir, ignore_errors=True)  # crashed/timed out -- fall back
            elif result["ok"]:
                build_dir_map[path] = str(build_dir)
            else:
                failure_entries[path] = result["metadata"]
                shutil.rmtree(build_dir, ignore_errors=True)  # confirmed failure, dir no longer needed
            if done % 20 == 0 or done == len(futures):
                print(f"[precompile] {done}/{len(futures)} done "
                      f"({len(build_dir_map)} compiled, {len(failure_entries)} confirmed-failed)")
    skipped_gpu_pass = len(failure_entries)
    print(f"[precompile] {len(build_dir_map)}/{len(records_with_paths)} compiled successfully "
          f"(reused in serial pass), {skipped_gpu_pass} confirmed-failed (serial pass SKIPPED "
          f"for these -- no second compile attempt)")
    return build_dir_map, failure_entries


NUM_WARMUP = 25   # PI protocol (CLAUDE.md / tasks/SELECTION.md #4.1): warmup 25, measure 100, median
NUM_TRIALS = 100
EXCESSIVE_SPEEDUP_THRESHOLD = 10.0  # PI instruction 2026-08-20: flag for manual review, verdict unchanged


def _timing_stats(elapsed_times: list) -> dict:
    """mean/median/std/min/max/num_trials from a raw elapsed-times list (ms).
    kernelbench.timing.get_timing_stats() gives mean/std/min/max but not
    median, which the project's timing protocol requires -- computed here
    instead of patching third_party/KernelBench."""
    arr = np.array(elapsed_times, dtype=float)
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)),
        "num_trials": len(elapsed_times),
    }


def _load_model_new_for_timing(language: str, code: str, build_dir: str | None = None):
    """Loads ModelNew the same way eval_one()/_recover_real_compile_error()
    do per language. Returns (ModelNew, cleanup) -- cleanup() must be called
    when done (removes the triton/tilelang tempfile; no-op for cuda/ptx)."""
    if language in ("triton", "tilelang"):
        ModelNew, tf = kb_eval.load_custom_model_with_tempfile(code, entry_point="ModelNew")

        def cleanup():
            tf.close()
            os_remove_quiet(tf.name)

        return ModelNew, cleanup
    elif language == "cuda":
        context = {}
        kb_eval.load_custom_model(code, context, build_directory=build_dir)
        ModelNew = context.get("ModelNew")
        if ModelNew is None:
            raise RuntimeError("ModelNew not defined after load_custom_model")
        return ModelNew, lambda: None
    elif language == "ptx":
        from ptx_harness import ptx_load, ptx_launch  # noqa: F401 -- pre-seeded into exec namespace below

        context = {"ptx_load": ptx_load, "ptx_launch": ptx_launch, "__builtins__": __builtins__}
        exec(code, context)
        ModelNew = context.get("ModelNew")
        if ModelNew is None:
            raise RuntimeError("ModelNew not defined in generated PTX code")
        return ModelNew, lambda: None
    else:
        raise ValueError(f"unknown language {language!r}")


def _time_forward(model, get_inputs_fn, device, language: str) -> dict:
    """One warmup(25)+measure(100) timing pass over model(*inputs), same
    protocol for both the generated kernel and the PyTorch-eager baseline so
    the two numbers are directly comparable. Reuses KernelBench's own
    cuda-event timer (kernelbench.timing.time_execution_with_cuda_event) --
    it already does torch.cuda.synchronize before/after every trial -- just
    called here with num_warmup/num_trials/discard_first matching the
    project's protocol instead of KernelBench's own (lower) defaults, and
    kept outside eval_kernel_against_ref() (which hardcodes num_warmup=3 and
    doesn't expose it to the caller)."""
    kb_eval.set_seed(42)
    inputs = get_inputs_fn()
    inputs = [kb_eval._process_input_tensor(x, device, language, PRECISION) for x in inputs]
    elapsed = kb_timing.time_execution_with_cuda_event(
        model, inputs, num_warmup=NUM_WARMUP, num_trials=NUM_TRIALS, discard_first=0,
        verbose=False, device=device,
    )
    return _timing_stats(elapsed)


def time_one(record: dict, code: str, device) -> dict:
    """Times an already-verified-correct sample: kernel latency and a
    freshly-remeasured PyTorch-eager fp16 baseline on THIS GPU (CLAUDE.md /
    tasks/SELECTION.md #4.1 condition (1): never reuse literature/cached
    baseline numbers). The baseline is remeasured per sample rather than
    cached per task -- redundant across samples of the same task, but avoids
    any cross-process caching complexity/staleness risk for what is, at
    n=228 correct samples total, a cheap redundancy to accept."""
    language = record["language"]
    ref_src = load_reference(record["task"])
    ref_context = {}
    Model, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(ref_src, ref_context)

    build_dir = tempfile.mkdtemp(prefix="k2x2_time_") if language == "cuda" else None
    cleanup_model = lambda: None
    try:
        ModelNew, cleanup_model = _load_model_new_for_timing(language, code, build_dir=build_dir)

        kb_eval.set_seed(42)
        init_inputs = get_init_inputs()
        init_inputs = [kb_eval._process_input_tensor(x, device, language, PRECISION) for x in init_inputs]

        kb_eval.set_seed(42)
        with torch.no_grad():
            new_model = ModelNew(*init_inputs).to(device=device, dtype=PRECISION)
            kernel_stats = _time_forward(new_model, get_inputs, device, language)

            kb_eval.set_seed(42)
            original_model = Model(*init_inputs).to(device=device, dtype=PRECISION)
            baseline_stats = _time_forward(original_model, get_inputs, device, language)

        speedup = baseline_stats["median"] / kernel_stats["median"]
        return {
            "kernel_ms": kernel_stats, "baseline_ms": baseline_stats,
            "speedup": speedup,
            "excessive_speedup_flag": speedup > EXCESSIVE_SPEEDUP_THRESHOLD,
            "num_warmup": NUM_WARMUP, "num_trials": NUM_TRIALS,
        }
    except Exception as e:
        return {"timing_exception": f"{type(e).__name__}: {e}"}
    finally:
        cleanup_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if build_dir:
            shutil.rmtree(build_dir, ignore_errors=True)


def _timing_worker_entry(record, code, result_conn):
    """Runs in an isolated (spawn) subprocess -- see time_one_isolated()."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        result = time_one(record, code, device)
    except Exception as e:
        result = {"timing_exception": f"{type(e).__name__}: {e}"}
    try:
        result_conn.send(result)
    finally:
        result_conn.close()


def time_one_isolated(record: dict, code: str, timeout: int = 180) -> dict:
    """Same isolation rationale as eval_one_isolated(): a single sample
    crashing (or an unexpectedly slow timing loop) only costs that one
    sample's subprocess, never the whole timing run."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_timing_worker_entry, args=(record, code, child_conn))
    proc.start()
    child_conn.close()

    result = None
    if parent_conn.poll(timeout):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    parent_conn.close()

    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if result is not None:
        return result
    return {"timing_exception": f"timing subprocess produced no result "
                                 f"(exitcode={proc.exitcode}) -- crash or hang "
                                 f"killed after a {timeout}s timeout"}


def collect_correct_records(source_paths: list[Path]) -> list[dict]:
    """Pulls every compiled+correct record out of one or more eval JSON
    files (e.g. the 0-shot full run and the docinject ablation) and attaches
    the generated code from results/raw/. Only correct samples are timed --
    timing an incorrect kernel's latency is meaningless."""
    out = []
    for p in source_paths:
        data = json.loads(p.read_text())
        for r in data["records"]:
            if r.get("compiled") and r.get("correctness"):
                raw = json.loads((RAW_DIR / r["path"]).read_text())
                out.append({**r, "_code": raw["parsed_code"], "_source": str(p)})
    return out


TIMING_MAX_ATTEMPTS = 3  # 1 try + 2 retries -- see run_timing()'s docstring note


def run_timing(source_paths: list[Path], checkpoint_path: Path, prior_records=None) -> list:
    """NOTE (2026-08-20): timing a correct CUDA sample intermittently segfaults
    inside torch.cuda.synchronize() (confirmed via PYTHONFAULTHANDLER=1 -- crash
    is in kernelbench.timing.time_execution_with_cuda_event's synchronize call,
    not in this project's code). Directly reproduced as FLAKY, not deterministic:
    the identical sample crashed on one invocation and then succeeded twice in a
    row on immediate retries with no code change. This is the same class of
    intermittent segfault already documented multiple times on this GPU/driver/
    torch combination (CLAUDE.md: vLLM CUDA-graph capture, FlashInfer MoE
    autotuner) -- not something a code fix here can address. Each sample's
    subprocess isolation (time_one_isolated) already contains the blast radius
    to that one sample; retrying a crashed/timed-out sample up to
    TIMING_MAX_ATTEMPTS times (fresh subprocess each attempt) is the same
    mitigation pattern eval_one() already uses for its own transient-failure
    case (lock-contention retry)."""
    records = list(prior_records or [])
    already_done = {r["path"] for r in records}
    to_time = [r for r in collect_correct_records(source_paths) if r["path"] not in already_done]
    print(f"[timing] {len(to_time)} correct sample(s) to time "
          f"({len(already_done)} already in checkpoint, skipped)")

    for i, r in enumerate(to_time, 1):
        result = None
        for attempt in range(1, TIMING_MAX_ATTEMPTS + 1):
            result = time_one_isolated(r, r["_code"])
            if "timing_exception" not in result:
                if attempt > 1:
                    result["recovered_after_attempts"] = attempt
                break
            print(f"[timing] {i}/{len(to_time)} {r['language']:8s} {r['task']:35s} "
                  f"sample={r['sample_index']} attempt {attempt}/{TIMING_MAX_ATTEMPTS} "
                  f"-> {result['timing_exception'][:100]}"
                  + (" -- retrying (fresh subprocess)" if attempt < TIMING_MAX_ATTEMPTS else ""))
        entry = {
            "path": r["path"], "task": r["task"], "task_family": r.get("task_family"),
            "language": r["language"], "condition": r["condition"],
            "model": r["model"], "sample_index": r["sample_index"], "source": r["_source"],
            **result,
        }
        records.append(entry)
        if "timing_exception" in result:
            print(f"[timing] {i}/{len(to_time)} {r['language']:8s} {r['task']:35s} "
                  f"sample={r['sample_index']} -> ERROR: {result['timing_exception'][:100]}")
        else:
            flag = " *** EXCESSIVE SPEEDUP ***" if result["excessive_speedup_flag"] else ""
            print(f"[timing] {i}/{len(to_time)} {r['language']:8s} {r['task']:35s} "
                  f"sample={r['sample_index']} -> kernel={result['kernel_ms']['median']:.4g}ms "
                  f"baseline={result['baseline_ms']['median']:.4g}ms "
                  f"speedup={result['speedup']:.3g}x{flag}")
        checkpoint_path.write_text(json.dumps(
            {"records": records, "status": "in_progress"}, indent=2, default=str))
    return records


def summarize_timing(records: list) -> dict:
    by = {}
    for r in records:
        if "timing_exception" in r:
            continue
        k = f"{r['language']}|{r['model'].split('/')[-1]}|{r['condition']}"
        by.setdefault(k, {"n": 0, "speedups": [], "fast_1": 0, "excessive_flagged": 0})
        d = by[k]
        d["n"] += 1
        d["speedups"].append(r["speedup"])
        if r["speedup"] > 1.0:
            d["fast_1"] += 1
        if r["excessive_speedup_flag"]:
            d["excessive_flagged"] += 1
    out = {}
    for k, d in by.items():
        geomean = float(np.exp(np.mean(np.log(np.array(d["speedups"])))))
        out[k] = {
            "n": d["n"], "fast_1": d["fast_1"], "fast_1_frac": d["fast_1"] / d["n"],
            "speedup_geomean": geomean, "excessive_flagged": d["excessive_flagged"],
        }
    return out


def load_checkpoint_records(path: Path) -> list:
    """Reads the records already evaluated in a previous (possibly interrupted) run.

    NOTE (2026-08-20): the 2026-08-19 full run was killed at 1,087/1,480 when the
    shell session it was attached to went away. The per-sample checkpoint had all
    1,087 results on disk, but there was no way to continue from them, so the only
    options were re-evaluating everything or splitting the run by --language and
    merging by hand. --resume closes that gap: samples already present in the
    checkpoint are never re-evaluated (results/ is read-only measurement data --
    re-running a sample could silently change a recorded verdict), and the new
    results are appended to the same file so the finished artifact is one complete
    1,480-record run rather than a pile of per-language shards."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("records", [])


def run_eval(language=None, condition=None, model_dir=None, task=None, raw_dir=RAW_DIR,
             checkpoint_path=None, prior_records=None, precompile_workers=PRECOMPILE_WORKERS):
    records = list(prior_records or [])
    already_done = {r["path"] for r in records}
    skipped = 0

    to_eval = []
    for path, record in find_samples(language, condition, model_dir, task, raw_dir):
        rel = str(path.relative_to(raw_dir))
        if rel in already_done:
            skipped += 1
            continue
        to_eval.append((rel, record))

    # P0-b (2026-08-20): parallel nvcc precompile for CUDA, decoupled from
    # the serialized GPU-touching pass below -- see precompile_cuda_batch().
    build_dir_map, precompile_failures = {}, {}
    if precompile_workers > 1:
        cuda_to_precompile = [(p, r) for p, r in to_eval
                               if r["language"] == "cuda" and r["status"] == "generated"]
        build_dir_map, precompile_failures = precompile_cuda_batch(
            cuda_to_precompile, max_workers=precompile_workers)

    for rel, record in to_eval:
        if rel in precompile_failures:
            # Confirmed compile failure already captured with the real
            # compiler error during the parallel precompile pass -- no GPU
            # touch needed for a failed compile, so skip the serial
            # subprocess entirely (see precompile_cuda_batch's docstring).
            result = {"compiled": False, "correctness": False, "metadata": precompile_failures[rel]}
        else:
            result = eval_one_isolated(record, precompiled_build_dir=build_dir_map.get(rel))
        entry = {
            "path": rel,
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
        # Checkpoint after every sample (2026-08-20, see eval_one_isolated's
        # docstring) -- a crash mid-run now loses at most one sample's worth
        # of work instead of the entire run.
        if checkpoint_path is not None:
            checkpoint_path.write_text(json.dumps(
                {"summary": summarize(records), "records": records, "status": "in_progress"},
                indent=2, default=str))
    if skipped:
        print(f"[eval] --resume: skipped {skipped} sample(s) already present in the checkpoint")
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
    ap.add_argument("--resume", default=None,
                    help="continue an interrupted run from its checkpoint JSON: samples already "
                         "recorded there are skipped (never re-evaluated) and new results are "
                         "appended to the same file. Implies --out <RESUME> unless --out is given.")
    ap.add_argument("--timing", action="store_true",
                    help="switch modes entirely: instead of compile+correctness, measure "
                         "latency (warmup 25 / measure 100 / median, kernel + freshly "
                         "remeasured PyTorch-eager fp16 baseline) for every compiled+correct "
                         "sample pulled from --timing-sources. Ignores --language/--condition/"
                         "--model-dir/--task/--raw-dir.")
    ap.add_argument("--timing-sources", nargs="+", default=None,
                    help="one or more eval JSON files (e.g. results/eval/full_run_20260819.json "
                         "results/eval/docinject_run_....json) to pull compiled+correct records "
                         "from. Required with --timing -- no default, to avoid silently timing "
                         "the wrong run.")
    ap.add_argument("--precompile-workers", type=int, default=PRECOMPILE_WORKERS,
                    help="parallel nvcc/ninja workers for CUDA precompilation before the "
                         "serialized GPU-touching eval pass (P0-b, 2026-08-20). Pure CPU work, "
                         "no GPU touch -- see precompile_cuda_batch(). Set to 0 or 1 to disable "
                         "and compile inline as before. Ignored with --timing.")
    args = ap.parse_args()

    assert_gpu_exclusive()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.timing:
        if not args.timing_sources:
            print("[refuse] --timing requires --timing-sources <eval.json> [<eval.json> ...]",
                  file=sys.stderr)
            return 1
        source_paths = [Path(p) for p in args.timing_sources]
        for p in source_paths:
            if not p.exists():
                print(f"[refuse] --timing-sources path does not exist: {p}", file=sys.stderr)
                return 1
        if args.out:
            out_path = Path(args.out)
        elif args.resume:
            out_path = Path(args.resume)
        else:
            out_path = EVAL_DIR / f"timing_{time.strftime('%Y%m%dT%H%M%S')}.json"

        prior_records = []
        if args.resume:
            resume_path = Path(args.resume)
            prior_records = load_checkpoint_records(resume_path)
            print(f"[timing] --resume {resume_path}: {len(prior_records)} sample(s) already timed")

        records = run_timing(source_paths, checkpoint_path=out_path, prior_records=prior_records)
        summary = summarize_timing(records)

        print("\n=== timing summary (lang|model|condition: n, fast_1, speedup geomean) ===")
        for k, d in sorted(summary.items()):
            print(f"  {k:55s} n={d['n']:3d} fast_1={d['fast_1']:3d} ({100*d['fast_1_frac']:5.1f}%) "
                  f"speedup_geomean={d['speedup_geomean']:.3g}x excessive_flagged={d['excessive_flagged']}")
        flagged = [r for r in records if r.get("excessive_speedup_flag")]
        if flagged:
            print(f"\n=== {len(flagged)} sample(s) flagged for manual review (speedup > "
                  f"{EXCESSIVE_SPEEDUP_THRESHOLD}x, verdict unchanged) ===")
            for r in flagged:
                print(f"  {r['path']}  speedup={r['speedup']:.3g}x")

        out_path.write_text(json.dumps(
            {"summary": summary, "records": records, "status": "complete"}, indent=2, default=str))
        print(f"\nwrote {out_path}")
        return 0

    if args.out:
        out_path = Path(args.out)
    elif args.resume:
        out_path = Path(args.resume)
    else:
        out_path = EVAL_DIR / f"eval_{time.strftime('%Y%m%dT%H%M%S')}.json"

    prior_records = []
    if args.resume:
        resume_path = Path(args.resume)
        prior_records = load_checkpoint_records(resume_path)
        print(f"[eval] --resume {resume_path}: {len(prior_records)} sample(s) already evaluated")

    records = run_eval(args.language, args.condition, args.model_dir, args.task, Path(args.raw_dir),
                        checkpoint_path=out_path, prior_records=prior_records,
                        precompile_workers=args.precompile_workers)
    summary = summarize(records)

    print("\n=== summary (compiled / correct out of n, generated=parsed-ok count) ===")
    for lang, d in sorted(summary.items()):
        print(f"  {lang:8s} n={d['n']:3d} generated={d['generated']:3d} "
              f"compiled={d['compiled']:3d} correct={d['correct']:3d}")

    out_path.write_text(json.dumps({"summary": summary, "records": records, "status": "complete"},
                                   indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

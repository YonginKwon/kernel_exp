"""Smoke test: run KernelBench's OWN eval_kernel_against_ref() path for the
cuda/triton/tilelang backends, using KernelBench's stock illustrative examples
(src/kernelbench/prompts/model_ex_add.py + model_new_ex_add*.py) -- not a task
from tasks/level1_subset.json, not results data, not authored by this session.

This is the actual code path scripts/evaluate.py will call once generate.py
starts producing real kernels, so it's a closer harness check than the
raw scripts/smoke_{cuda,triton,tilelang}.py scripts (which bypass KernelBench's
eval.py entirely). PTX has no such upstream path -- see scripts/smoke_ptx.py /
harness/ptx/ instead.
"""
import sys

sys.path.insert(0, "third_party/KernelBench/src")
from kernelbench import eval as kb_eval

ORIGINAL_SRC = open("third_party/KernelBench/src/kernelbench/prompts/model_ex_add.py").read()

CASES = [
    ("cuda", "third_party/KernelBench/src/kernelbench/prompts/model_new_ex_add.py", "float32"),
    ("triton", "third_party/KernelBench/src/kernelbench/prompts/model_new_ex_add_triton.py", "float32"),
    ("tilelang", "third_party/KernelBench/src/kernelbench/prompts/model_new_ex_add_tilelang.py", "float16"),
]

import torch

PRECISION = {"float32": torch.float32, "float16": torch.float16}


def main():
    rc = 0
    for backend, path, prec_name in CASES:
        custom_src = open(path).read()
        precision = PRECISION[prec_name]
        result = kb_eval.eval_kernel_against_ref(
            original_model_src=ORIGINAL_SRC,
            custom_model_src=custom_src,
            backend=backend,
            precision=precision,
            measure_performance=False,
            verbose=False,
        )
        if result is None:
            print(f"[result] kernelbench-harness backend={backend}: RETRY (lock file transient)")
            rc = 1
            continue
        status = "PASS" if (result.compiled and result.correctness) else "FAIL"
        print(f"[result] kernelbench-harness backend={backend} precision={prec_name}: {status} "
              f"compiled={result.compiled} correctness={result.correctness} "
              f"meta_keys={list(result.metadata.keys())}")
        if status != "PASS":
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

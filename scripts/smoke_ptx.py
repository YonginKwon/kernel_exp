"""PTX go/no-go smoke test (CLAUDE.md 8/10 milestone).

Verifies harness/ptx's LLM-facing API (ptx_load / ptx_launch, exactly as
promised by prompts/PROMPT_SPEC.md §2's PTX language block) end-to-end on
this machine's actual GPU: ptxas assembly -> cuModuleLoadData ->
cuLaunchKernel -> correct result, sharing PyTorch's CUDA context so
tensor.data_ptr() is a valid device pointer.

Fixture PTX (vecadd.ptx) was produced once via `nvcc -arch=sm_120a -ptx` from a
throwaway .cu file, purely to get a real, driver-valid PTX text targeting this
GPU's compute capability -- this validates the *harness*, not an LLM's PTX
authoring ability. Not experimental data.

Fixture path is parametrized (--fixture) so this same smoke test runs on other
GPU architectures (e.g. the kernel-lang-2x2 A100 probe, sm_80) without editing
this file -- default stays sm_120a to keep the primary machine's behavior
unchanged. [sync-needed] mirror this parametrization back to the main repo.
"""
import argparse
import sys
import torch

sys.path.insert(0, "harness/ptx")
from ptx_harness import ptx_load, ptx_launch, PTXCompileError

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", default="scripts/fixtures/vecadd.ptx")
ARGS, _ = parser.parse_known_args()

PTX_SRC = open(ARGS.fixture).read()


def main():
    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    print(f"[env] gpu={torch.cuda.get_device_name(dev)} cc={cc}")

    torch.cuda.init()  # ensure torch's primary context exists before ptx_load shares it

    try:
        module = ptx_load(PTX_SRC)
    except PTXCompileError as e:
        print(f"[result] ptx_load (ptxas assemble): FAIL\n{e.stderr}")
        return 1
    print("[result] ptx_load (ptxas assemble + cuModuleLoadData): PASS")

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)

    block = 256
    grid = ((n + block - 1) // block, 1, 1)
    ptx_launch(module, "vecadd", grid, (block, 1, 1),
               [x.data_ptr(), y.data_ptr(), out.data_ptr(), n])
    torch.cuda.synchronize(dev)

    ok = torch.allclose(out, x + y, atol=1e-4, rtol=1e-4)
    print(f"[result] ptx_launch + cuLaunchKernel: {'PASS' if ok else 'FAIL'} "
          f"max_diff={(out - (x + y)).abs().max().item():.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

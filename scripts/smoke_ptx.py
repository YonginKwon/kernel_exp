"""PTX go/no-go smoke test (CLAUDE.md 8/10 milestone).

Verifies the harness/ptx cuModuleLoad wrapper end-to-end on this machine's actual
GPU: ptxas assembly -> cuModuleLoadData -> cuLaunchKernel -> correct result,
sharing PyTorch's CUDA context so tensor.data_ptr() is a valid device pointer.

Fixture PTX (vecadd.ptx) was produced once via `nvcc -arch=sm_120a -ptx` from a
throwaway .cu file, purely to get a real, driver-valid PTX text targeting this
GPU's compute capability -- this validates the *harness*, not an LLM's PTX
authoring ability. Not experimental data.
"""
import sys
import torch

sys.path.insert(0, "harness/ptx")
from ptx_harness import assemble_ptx, CuModuleRunner, PTXCompileError

PTX_SRC = open("scripts/fixtures/vecadd.ptx").read()


def main():
    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    arch = f"sm_{cc[0]}{cc[1]}a"  # 'a' suffix: Blackwell family-specific ISA, matches CLAUDE.md's sm_120
    print(f"[env] gpu={torch.cuda.get_device_name(dev)} cc={cc} arch={arch}")

    try:
        cubin = assemble_ptx(PTX_SRC, arch=arch)
    except PTXCompileError as e:
        print(f"[result] ptxas assemble: FAIL\n{e.stderr}")
        return 1
    print(f"[result] ptxas assemble: PASS ({len(cubin)} bytes cubin)")

    # touch cuda so torch's primary context exists before we retain/share it
    torch.cuda.init()

    runner = CuModuleRunner(device_index=0)
    fn = runner.load(cubin, "vecadd")

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)

    block = 256
    grid = ((n + block - 1) // block, 1, 1)
    runner.launch(fn, grid=grid, block=(block, 1, 1),
                  args=[x.data_ptr(), y.data_ptr(), out.data_ptr(), n])
    runner.synchronize()

    ok = torch.allclose(out, x + y, atol=1e-4, rtol=1e-4)
    print(f"[result] cuModuleLoad + cuLaunchKernel: {'PASS' if ok else 'FAIL'} "
          f"max_diff={(out - (x + y)).abs().max().item():.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

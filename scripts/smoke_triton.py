"""Smoke test: does the installed Triton compile and run on this GPU (sm_120)?

Not an experiment. Verifies toolchain only. Fixture kernel is a trivial vector add,
written by hand as harness scaffolding -- never used as experimental data.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def main():
    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    print(f"[env] gpu={torch.cuda.get_device_name(dev)} cc={cc} sm_{cc[0]}{cc[1]}")
    print(f"[env] torch={torch.__version__} triton={triton.__version__}")

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    BLOCK = 1024
    _add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize(dev)

    ok = torch.allclose(out, x + y, atol=1e-4, rtol=1e-4)
    print(f"[result] triton compile+run: {'PASS' if ok else 'FAIL'} "
          f"max_diff={(out - (x + y)).abs().max().item():.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

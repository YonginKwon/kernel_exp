"""Smoke test: does a hand-compiled CUDA C++ extension build and run on sm_120?

Not an experiment. Verifies toolchain only (torch.utils.cpp_extension path, which is
what KernelBench's CUDA backend / eval.py load_custom_model uses under the hood).
Fixture kernel is a trivial vector add, hand-written harness scaffolding.
"""
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

CPP_SRC = "torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);"


def main():
    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    print(f"[env] gpu={torch.cuda.get_device_name(dev)} cc={cc} sm_{cc[0]}{cc[1]}")
    print(f"[env] torch={torch.__version__}")

    mod = load_inline(
        name="smoke_cuda_add",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["add_cuda"],
        verbose=True,
    )

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = mod.add_cuda(x, y)
    torch.cuda.synchronize(dev)

    ok = torch.allclose(out, x + y, atol=1e-4, rtol=1e-4)
    print(f"[result] cuda-c++ compile+run: {'PASS' if ok else 'FAIL'} "
          f"max_diff={(out - (x + y)).abs().max().item():.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

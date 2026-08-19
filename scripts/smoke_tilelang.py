"""Smoke test: does the installed TileLang compile and run on this GPU (sm_120)?

Not an experiment. Verifies toolchain only. Fixture kernels are hand-written harness
scaffolding -- never used as experimental data.

Two cases on purpose:
  1. elementwise add in fp32  -> probes whether TileLang itself is fp32-capable
     (KernelBench's eval.py asserts tilelang == fp16/bf16; we need to know if that
      assert reflects a real limitation or is just policy -- it decides whether the
      2x2 comparison can run at a single uniform precision)
  2. tiled matmul in fp16     -> probes the tensor-core path on Blackwell
"""
import torch
import tilelang
import tilelang.language as T


def build_add(n, block, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            tx = T.get_thread_binding(0)
            i = bx * block + tx
            if i < n:
                C[i] = A[i] + B[i]
    return main


def build_matmul(M, N, K, bM, bN, bK, dtype="float16", accum="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bK, bN), dtype)
            C_l = T.alloc_fragment((bM, bN), accum)
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=3):
                T.copy(A[by * bM, ko * bK], A_s)
                T.copy(B[ko * bK, bx * bN], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * bM, bx * bN])
    return main


def case_add():
    n, block = 1 << 20, 256
    kern = tilelang.compile(build_add(n, block), out_idx=[2])
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = kern(a, b)
    torch.cuda.synchronize()
    ok = torch.allclose(c, a + b, atol=1e-4, rtol=1e-4)
    return ok, (c - (a + b)).abs().max().item()


def case_matmul():
    M = N = K = 1024
    kern = tilelang.compile(build_matmul(M, N, K, 128, 128, 32), out_idx=[2])
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kern(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).half()
    ok = torch.allclose(c, ref, atol=1e-2, rtol=1e-2)
    return ok, (c.float() - ref.float()).abs().max().item()


def main():
    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    print(f"[env] gpu={torch.cuda.get_device_name(dev)} cc={cc} sm_{cc[0]}{cc[1]}")
    print(f"[env] torch={torch.__version__} tilelang={tilelang.__version__}")

    rc = 0
    for name, fn in (("elementwise_add fp32", case_add), ("tiled_matmul fp16", case_matmul)):
        try:
            ok, diff = fn()
            print(f"[result] tilelang {name}: {'PASS' if ok else 'FAIL'} max_diff={diff:.3e}")
            rc |= 0 if ok else 1
        except Exception as e:
            print(f"[result] tilelang {name}: ERROR {type(e).__name__}: {str(e)[:600]}")
            rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

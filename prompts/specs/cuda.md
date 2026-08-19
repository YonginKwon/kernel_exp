<!-- INTERNAL PROVENANCE (not injected into the prompt -- spec_loader.py strips
everything above the "---" marker before use).
Source: NVIDIA CUDA C++ Programming Guide, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
Retrieved: 2026-08-19. Sections referenced: "Writing SIMT Kernels" (kernel
qualifiers, launch syntax, thread hierarchy), "CUDA C++ Memory Model" (memory
spaces, __shared__, __syncthreads()). Compiled by hand from stable, versioned
CUDA C++ language semantics (kernel syntax and memory-space qualifiers have
been unchanged since CUDA 2.0); the live doc site did not return full section
text to automated fetch (JS-rendered SPA), so wording below is written from
the language semantics directly rather than quoted verbatim, except for the
canonical VecAdd example pattern which mirrors the guide's own worked example.
Target arch line reflects CLAUDE.md's measured GPU (sm_120a, Blackwell).
-->

---
# CUDA C++ Language Reference (excerpt for GPU kernel authoring)

## 1. Syntax overview

### 1.1 Function qualifiers

- `__global__`: a kernel entry point. Runs on the GPU, called from host code
  with the `<<<...>>>` launch syntax. Must return `void`.
- `__device__`: a function that runs on the GPU and is callable only from
  other `__device__` or `__global__` code (not from the host).
- `__host__`: runs on the CPU (default for unqualified functions). A function
  can be marked `__host__ __device__` to compile for both.

```cpp
__global__ void my_kernel(const float* a, const float* b, float* out, int n);
```

### 1.2 Launch configuration

A kernel is launched with an execution configuration between `<<<` and `>>>`:

```cpp
my_kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

- `gridDim`: number of thread blocks, as `dim3` (1D/2D/3D). A plain integer
  is treated as `dim3(n, 1, 1)`.
- `blockDim`: number of threads per block, same `dim3` semantics.
- `sharedMemBytes` (optional): bytes of dynamic shared memory to reserve.
- `stream` (optional): CUDA stream for async ordering; omit for the default
  stream.

Block size is capped by hardware (1024 threads/block on all currently
supported architectures, including sm_120a). Total threads launched =
`gridDim.x*gridDim.y*gridDim.z * blockDim.x*blockDim.y*blockDim.z`.

### 1.3 Thread/block indexing (built-in variables, valid only inside
`__global__`/`__device__` code)

- `threadIdx` (`dim3`): this thread's index within its block.
- `blockIdx` (`dim3`): this block's index within the grid.
- `blockDim` (`dim3`): block dimensions, as configured at launch.
- `gridDim` (`dim3`): grid dimensions, as configured at launch.

The idiomatic global element index for a 1D launch:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) { /* ... */ }   // bounds check: n is rarely a multiple of blockDim.x
```

For 2D/3D data, apply the same pattern per axis (`.x`, `.y`, `.z`).

### 1.4 Synchronization

- `__syncthreads()`: barrier across all threads in a block. Required before
  reading `__shared__` memory that other threads in the block just wrote.
  All threads in the block must reach the same `__syncthreads()` call
  (never place it inside divergent `if` branches that not all threads take).
- Atomics (`atomicAdd`, `atomicMax`, `atomicCAS`, ...): needed when multiple
  threads write the same address (e.g. accumulating into a shared or global
  reduction output).

### 1.5 Common built-ins

Math intrinsics (`__expf`, `__logf`, `sqrtf`, `fmaxf`, `fminf`, `rsqrtf`,
`tanhf`, `erff`, `exp2f`, `__fdividef`, ...) operate on `float`/`double`
per-thread scalars, exactly like standard C math functions -- there is no
block-level vector API in raw CUDA C++ (unlike Triton/TileLang, where a
kernel body operates on a whole tile at once). Fast/approximate variants
(the `__`-prefixed intrinsics, e.g. `__expf` vs `expf`) trade a small amount
of accuracy for throughput; either is acceptable unless the reference
implementation's numerics require full precision.

For activation functions (ReLU, Sigmoid, Tanh, GELU, Softplus, ...), express
them as ordinary per-thread scalar arithmetic using these intrinsics -- there
is no built-in "activation op", you write the formula directly, e.g.
`float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }`.

### 1.6 Grid-stride loops

When the problem size `n` exceeds `gridDim.x * blockDim.x` (or when the
launch configuration is chosen independently of `n`, e.g. a fixed number of
blocks for good occupancy regardless of input size), each thread processes
multiple elements with a strided loop instead of one:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; i < n; i += stride) {
    out[i] = f(in[i]);
}
```

This pattern also applies to reductions and to any per-row work over a
non-contiguous dimension (e.g. Softmax/LayerNorm/normalization kernels
launched with one block per row): the block-local loop walks the row with
`threadIdx.x`/`blockDim.x` as the stride instead of the whole-grid stride
shown above.

### 1.7 Warp execution model

Threads execute in groups of 32 called *warps*, in lockstep (SIMT: Single
Instruction, Multiple Threads). All threads in a warp execute the same
instruction at the same time; when threads in a warp take different branches
of an `if`/`else` (*warp divergence*), the warp executes both paths
serially with the inactive threads masked off, so divergent branches cost
extra cycles even though correctness is unaffected. Prefer branch conditions
that are uniform across a warp (e.g. boundary checks aligned to 32-element
tiles) when performance-sensitive, though correctness never requires this.

Warp-level primitives (`__shfl_sync`, `__shfl_down_sync`, `__ballot_sync`,
...) exchange values directly between threads in the same warp without
going through shared memory -- useful for small reductions (e.g. summing 32
values) but not required; a `__shared__`-memory tree reduction (§2) is
always a valid, simpler alternative.

### 1.8 Restrict and const-correctness

Mark read-only kernel pointer arguments `const T* __restrict__` when no two
pointer arguments alias the same memory -- this is a hint the compiler can
use for more aggressive load scheduling/caching. It's an optimization, not a
correctness requirement; a kernel is correct without it.

### 1.9 fp16 (half precision)

This study's protocol fixes all tensor inputs/outputs to `float16`
(`torch::kFloat16`, which maps to `__half`/`at::Half` on the C++ side).
`#include <cuda_fp16.h>` for the `__half` type and its intrinsics:

```cpp
#include <cuda_fp16.h>

__global__ void vecadd_half_kernel(const __half* a, const __half* b, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);          // native half add
        // equivalently: out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
    }
}
```

`data_ptr<at::Half>()` on the host side gives a pointer whose bytes are
bit-compatible with `__half*`; a `reinterpret_cast<__half*>` (or using
`at::Half*` directly, since it has the same layout) is the usual bridge.
As stated in the common prompt template, **accumulation precision is your
choice**: computing in `float` internally (via `__half2float` /
`__float2half`) and only storing the final result as `__half` is a common
and acceptable strategy for reductions/matmul where fp16 accumulation would
lose too much precision -- there is no requirement to keep every
intermediate in `__half`.

### 1.10 Where errors surface

A malformed kernel definition (bad syntax, wrong argument types against the
`torch::Tensor`/`data_ptr<T>()` calls in the host wrapper) is caught by
`nvcc` at compile time -- this is what this study logs as a "compile
failure" and, per the protocol, may earn one repair turn with the raw
compiler message. A kernel that compiles but reads/writes out of bounds
typically does *not* fail immediately (device memory reads/writes are not
bounds-checked by hardware); it either silently corrupts memory, produces
wrong values that fail the correctness check, or raises an
"illegal memory access" error at the next `cudaDeviceSynchronize()` /
kernel launch that touches the corrupted state. Always guard the global
index against the tensor's actual element count, as in §1.3.

## 2. Memory model

CUDA exposes the GPU's memory hierarchy explicitly; the programmer chooses
where each value lives.

| Space | Qualifier / access | Scope | Speed | Notes |
|---|---|---|---|---|
| Registers | plain local variables | per-thread | fastest | compiler-allocated; spills to local memory if the kernel uses too many |
| Local memory | plain local variables (compiler decides) | per-thread | slow (physically in DRAM, cached) | used for large per-thread arrays or register spills; not a separate qualifier, it's a compiler fallback |
| Shared memory | `__shared__` | per-block | fast (on-chip SRAM) | visible to all threads in the same block; must `__syncthreads()` after writing, before another thread reads |
| Global memory | plain pointer arguments (device pointers) | whole grid + host | slow (off-chip DRAM, cached by L1/L2) | what `tensor.data_ptr<T>()` gives you; coalesced access (consecutive threads touch consecutive addresses) is critical for bandwidth |
| Constant memory | `__constant__` | whole grid, read-only during kernel | fast when broadcast (all threads read the same address) | rarely needed for the kernels in this study; global memory is the default choice |

Static shared memory: `__shared__ float tile[128];` -- size fixed at compile
time. Dynamic shared memory: declare `extern __shared__ float tile[];` in the
kernel and pass the byte size as the 3rd launch-config argument.

A minimal shared-memory pattern (block-local staging before a reduction or a
tiled matmul):

```cpp
__shared__ float tile[256];
int tid = threadIdx.x;
tile[tid] = input[blockIdx.x * 256 + tid];
__syncthreads();               // wait for the whole block to finish writing
// ... now every thread in the block may read any tile[j] ...
```

Memory-space rule of thumb for this study's kernels: read inputs from global
memory into registers or shared memory, compute, write the result back to
global memory. There is no implicit caching of *values* across kernel
launches -- each kernel invocation starts from global memory state.

### 2.1 Coalescing

Global memory is served in contiguous cache-line-sized transactions. When
consecutive threads in a warp (§1.7) read/write consecutive addresses (e.g.
`data[blockIdx.x*blockDim.x + threadIdx.x]`), the hardware coalesces those
32 accesses into a small number of wide transactions -- this is the single
biggest factor in memory-bound kernel throughput (elementwise ops,
reductions, normalization). Strided access patterns (e.g. `data[threadIdx.x
* stride]` with `stride` large, as can happen when iterating a
non-innermost tensor dimension) serialize into many small transactions and
are far slower for the same amount of data moved. For row-major tensors,
indexing so that `threadIdx.x` varies the fastest-varying (innermost,
contiguous) dimension gives coalesced access.

### 2.2 Shared memory bank conflicts

Shared memory is physically organized into 32 banks; each bank services one
address per cycle. If multiple threads in a warp read/write different
addresses in the *same* bank in the same instruction, those accesses
serialize (a bank conflict) -- this doesn't affect correctness, only
throughput. Accessing shared memory with a stride that is a multiple of 32
`float`s is the classic conflict pattern (common when a tile's row length is
a power of two equal to or exceeding 32); padding the shared-memory tile's
row length by one element (e.g. `__shared__ float tile[32][33]` instead of
`[32][32]`) is the standard fix, but is a performance detail, not a
correctness requirement for this study.

### 2.3 Cache hierarchy and occupancy

Global memory reads are cached in L2 (shared across the whole GPU) and, on
most architectures, L1 (per-SM, often unified with the shared-memory
hardware). Using more `__shared__` memory or more registers per thread block
reduces how many blocks can be resident on a streaming multiprocessor (SM)
at once (*occupancy*) -- fewer resident warps means less latency-hiding.
This is again a performance/tuning concern; for a functionally-correct
kernel meeting this study's correctness bar, occupancy tuning is optional.

### 2.4 Read-only data cache

`__ldg(ptr)` (or, equivalently, marking a pointer `const T* __restrict__` as
in §1.8, which lets the compiler emit the same path automatically) routes a
global-memory read through the read-only data cache, a separate cache path
from the general L1/L2 hierarchy that can help when the same read-only input
address is reused across many threads (e.g. a convolution kernel's filter
weights, or a broadcast operand in matmul-with-scalar-style tasks). This is
a throughput optimization, not required for correctness.

## 3. Minimal complete example

Element-wise vector addition, written exactly as this project's harness
expects (`torch.utils.cpp_extension.load_inline`, per `PROMPT_SPEC.md` §2's
CUDA language block): a CUDA source string with the kernel, a host wrapper
function that computes the launch configuration and returns a `torch::Tensor`,
and a C++ declaration string for the extension binding.

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vecadd_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor vecadd_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    vecadd_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor vecadd_cuda(torch::Tensor a, torch::Tensor b);"

vecadd = load_inline(
    name="vecadd_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vecadd_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.vecadd = vecadd

    def forward(self, a, b):
        return self.vecadd.vecadd_cuda(a, b)
```

This example generalizes directly to the kernels required by this study:
replace the body of `vecadd_kernel` with the target operator's per-element
(or per-tile, using `__shared__` staging as in §2) computation, and adjust the
launch configuration (`grid`, `block`) to the input tensor's shape.

Guidance by operator family (all are patterns, not requirements -- any
correct implementation is acceptable regardless of how it's structured):

- **Elementwise / activation** (ReLU, Sigmoid, GELU, ...): identical
  structure to the example above -- one thread per output element, no
  shared memory needed.
- **Reduction across a dimension** (Sum/Mean/Max/Argmax reduction, Softmax,
  Norm layers): launch one block per row (or per reduced-over slice), have
  each thread load and combine several elements into a register, stage
  partial results into `__shared__` memory, `__syncthreads()`, then do a
  tree reduction over the shared array (e.g. `for (int s = blockDim.x/2; s >
  0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads(); }`)
  before one thread writes the final value to global memory. Softmax/LayerNorm
  need two such reductions in sequence (e.g. max then sum-of-exp).
- **Pooling** (Max/Avg Pooling 1D/2D/3D): one thread per output element,
  each thread loops over its receptive-field window in the input and
  combines (max or running average) -- no shared memory required unless
  reusing input tiles across neighboring output elements for bandwidth.
- **Convolution** (standard/transposed/depthwise, 1D/2D/3D): the direct
  (non-im2col) formulation is a nested loop per output element over the
  kernel's receptive field and input channels, exactly like pooling but
  with a multiply-accumulate instead of a max/average; a `__shared__`-memory
  tile of the input patch (with the halo region for the kernel's spatial
  extent) avoids redundant global loads when neighboring output pixels
  share input pixels.
- **Matmul-family** (including batched, transposed, structured variants):
  the standard pattern tiles both input matrices into `__shared__` memory
  block by block along the shared (K) dimension, `__syncthreads()` after
  each tile load, accumulates the partial dot product in a register, and
  repeats until K is exhausted; write the accumulated result once at the
  end. This is the same tiling idea as TileLang's `T.gemm`/`T.Pipelined`
  (see `tilelang.md`), expressed by hand instead of through a library call.
- **Cumulative/scan** (cumsum, cumprod): sequential dependency along the
  scan axis prevents a naive one-thread-per-element mapping; a block-level
  scan (e.g. Hillis-Steele or Blelloch) over `__shared__` memory, or a
  simpler one-thread-per-row sequential loop when the scanned dimension is
  the only work per row, are both acceptable for correctness.
- **Loss functions** (MSE, CrossEntropy, Hinge, ...): typically a per-element
  computation followed by a full reduction to a scalar (or a per-row
  reduction, depending on the reference's reduction argument) -- combine the
  elementwise and reduction patterns above; a common approach is one kernel
  for the per-element/per-row loss and a second, tiny kernel (or an atomic
  add into a single output element) for the final reduction across blocks.
- **Attention** (scaled dot-product attention): composed of matmul (Q·K^T),
  a row-wise Softmax (the reduction pattern above), and a second matmul
  (·V) -- most straightforwardly implemented as three kernel launches
  chained by the host wrapper function (`forward` may call `load_inline`
  compiled functions more than once), though a fused single-kernel
  implementation is also acceptable if it computes the same result.

### 3.1 `load_inline` build notes

`load_inline` compiles and caches the extension by content hash of its
source strings under `~/.cache/torch_extensions/`; a rebuild is only
triggered when the CUDA/C++ source text changes, not when unrelated Python
code around it changes. Extra compiler flags (rarely needed) can be passed
via `extra_cuda_cflags=[...]`/`extra_cflags=[...]` keyword arguments to
`load_inline` if a specific kernel needs them (e.g. `--use_fast_math`);
omitting them uses the toolchain's defaults, which are sufficient for every
kernel in this study's scope. `verbose=True` on `load_inline` prints the
exact `nvcc`/host-compiler invocation and any compiler diagnostics, which is
useful when a kernel fails to compile and the raw error text needs to be
inspected.

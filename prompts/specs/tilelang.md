<!-- INTERNAL PROVENANCE (not injected -- spec_loader.py strips everything above
the "---" marker before use).
Sources:
- TileLang GitHub README, https://github.com/tile-ai/tilelang (retrieved
  2026-08-19; the matmul_relu example referenced in §3.1 is quoted from this
  page's live fetch -- T.Kernel/T.alloc_shared/T.alloc_fragment/T.copy/
  T.gemm/T.Pipelined/T.Parallel/T.clear usage patterns).
- TileLang "Language Basics" programming guide,
  https://tilelang.com/programming_guides/language_basics.html (URL
  confirmed via web search 2026-08-19: "T.alloc_shared allocates shared
  memory across the entire thread block... T.alloc_fragment allocates
  register space for local accumulation... corresponds to register files on
  NVIDIA GPUs"; live fetch blocked by the site's bot protection (HTTP 403)
  for direct retrieval, so this document draws on the fetched GitHub README
  plus this project's own installed tilelang==0.1.13, verified in
  scripts/smoke_tilelang.py -- both the fp32 elementwise-add and the fp16
  tiled-matmul (tensor-core) smoke cases PASS on this project's GPU, see
  CLAUDE.md).
- Minimal example in §3 is KernelBench's own stock fixture,
  third_party/KernelBench/src/kernelbench/prompts/model_new_ex_add_tilelang.py
  (not authored for this document; confirmed compiled+correct through
  KernelBench's actual eval_kernel_against_ref() harness via
  scripts/smoke_kernelbench_harness.py during the 8/10 milestone).
-->

---
# TileLang Language Reference (excerpt for GPU kernel authoring)

TileLang is a Python-embedded DSL for writing tile-level GPU kernels,
JIT-compiled (via `tilelang.compile` / `@tilelang.jit`) down to CUDA C++ and
then through the same `nvcc` toolchain as the CUDA C++ track. Like Triton, it
operates at the level of *tiles* (blocks of data) rather than individual
threads -- but unlike Triton, TileLang requires the programmer to explicitly
name each tile's home in the memory hierarchy (shared memory vs. register
fragment, mirroring CUDA C++'s explicit `__shared__` model, §2) rather than
leaving that entirely to the compiler.

## 1. Syntax overview

### 1.1 Kernel definition (`T.prim_func`)

```python
import tilelang
import tilelang.language as T

@T.prim_func
def my_kernel(A: T.Tensor((M, N), "float16"), B: T.Tensor((M, N), "float16"),
              C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        ...

kernel = tilelang.compile(my_kernel, out_idx=[2])   # out_idx: which arg(s) are outputs
result = kernel(a_tensor, b_tensor)                  # returns the output tensor(s)
```

- Kernel parameters are declared with `T.Tensor((shape...), dtype)` type
  annotations -- these describe a *global-memory* view over whatever tensor
  is passed in at call time (the TileLang analogue of a raw device pointer
  argument in CUDA C++/PTX, or an untyped `x_ptr` in Triton, but shape- and
  dtype-checked).
- `T.Kernel(grid_x, grid_y, ..., threads=N)` opens the kernel's launch
  configuration as a `with` block: the grid dimensions (like CUDA C++'s
  `gridDim`, §"cuda.md" §1.2) and `threads=` (the block size, like CUDA
  C++'s `blockDim`). The `with ... as (bx, by)` binds the block-index
  variables for each grid axis -- the TileLang analogue of `blockIdx.x`/
  `blockIdx.y`.
- `tilelang.compile(func, out_idx=[...])` JIT-compiles the `T.prim_func` and
  returns a callable; `out_idx` tells TileLang which parameter position(s)
  hold outputs (so it can allocate and return them rather than requiring
  the caller to pre-allocate).

### 1.2 Memory allocation primitives

Inside a `T.Kernel` block, tile-sized working memory is allocated
explicitly by which hardware resource it should live in:

- `T.alloc_shared((shape...), dtype)`: allocates a tile in on-chip shared
  memory, shared by every thread in the block (`cuda.md` §2's
  `__shared__`). Used to stage a tile of input data loaded from global
  memory before compute.
- `T.alloc_fragment((shape...), dtype)`: allocates a tile in **fragment
  memory**, which maps to the GPU's register file -- used for
  per-thread-group accumulation (e.g. a matmul's running output tile)
  before it's written back to global memory. This is the closest TileLang
  concept to a CUDA C++ kernel's plain local/register variables (`cuda.md`
  §2), but tile-shaped rather than scalar.
- `T.alloc_local((shape...), dtype)`: per-thread local scratch (smaller,
  simpler cousin of `T.alloc_fragment`, used for non-tensor-core
  elementwise accumulation).

### 1.3 Data movement and initialization

- `T.copy(src, dst)`: copies a tile between any two of {global memory
  (indexed slice of a `T.Tensor` parameter), shared memory, fragment
  memory} -- e.g. `T.copy(A[by*block_M, ko*block_K], A_shared)` copies a
  `block_M x block_K` tile starting at that offset from the global tensor
  `A` into the shared-memory buffer `A_shared`. The destination's shape
  determines how much is copied; slicing a `T.Tensor` with a starting
  index (as above) rather than a full range is TileLang's tile-indexing
  idiom.
- `T.clear(tile)`: zero-initializes a fragment/local tile (typically an
  accumulator, before a reduction loop starts adding into it).
- `T.fill(tile, value)`: initializes every element of a tile to `value`.

### 1.4 Compute primitives

- `T.gemm(A_shared, B_shared, C_local)`: maps a tile-level matrix multiply
  onto the hardware's matmul path (tensor cores where available) --
  `C_local += A_shared @ B_shared`, accumulating into the fragment tile
  `C_local`. This is TileLang's equivalent of Triton's `tl.dot`
  (`triton.md` §1.5) or a hand-written `.shared`-tiled accumulation loop in
  CUDA C++/PTX (`cuda.md` §3, `ptx.md` §3.2).
- `T.Parallel(d0, d1, ...)`: a `for`-loop-like construct for elementwise
  work over a tile's index space, and TileLang's mechanism for layout
  inference (the compiler figures out how to map the logical iteration
  space onto physical threads):

  ```python
  for i, j in T.Parallel(block_M, block_N):
      C_local[i, j] = A_shared[i, j] + B_shared[i, j]
  ```

  This is the TileLang analogue of Triton's implicit whole-tile elementwise
  arithmetic (`triton.md` §1.5) -- but written as an explicit loop over
  tile coordinates rather than an operator applied to a whole tile value.
- `T.Pipelined(num_iterations, num_stages=N)`: a `for`-loop over an outer
  dimension (typically the K-dimension of a matmul, or any
  reduction/accumulation axis split into chunks) that the compiler
  software-pipelines across `num_stages` -- overlapping the next
  iteration's global-to-shared `T.copy` with the current iteration's
  `T.gemm`/compute, hiding memory latency. Functionally equivalent to a
  plain Python `for` loop (`num_stages=1` degrades to that); the staging
  count is a performance parameter, not a correctness one.
- `T.reduce_max`/`T.reduce_sum` (and similar): tile-level reductions along
  a given axis, the TileLang analogue of Triton's `tl.max`/`tl.sum`
  (`triton.md` §1.5) or a manual shared-memory tree reduction in CUDA
  C++/PTX.
- `T.max(a, b)`/`T.min(a, b)`/ordinary Python arithmetic operators: apply
  to individual tile elements inside a `T.Parallel` loop body (as in the
  ReLU epilogue `T.max(C_local[i, j], 0)` in §3.1's GEMM example) or to
  whole tiles where shapes broadcast-align.

### 1.4.1 Vectorized/broadcast tile arithmetic

Inside a `T.Parallel` loop body, ordinary Python arithmetic on indexed tile
elements (`A_shared[i, j] + B_shared[i, j]`, as in §3) is scalar-per-element,
resolved by the compiler across the whole loop's iteration space -- there is
no separate "vector width" the programmer manages, unlike PTX's explicit
`.v4` vectorized loads (`ptx.md` §2.3). Whole-tile operations (without an
explicit `T.Parallel` loop) are also valid where shapes align, e.g.
`C_local[:, :] = A_shared[:, :] + B_shared[:, :]` for a full-tile add, which
TileLang lowers the same way as the explicit loop form -- both are provided
for readability, not as functionally distinct choices.

### 1.5 Grid/block index helpers

- `T.ceildiv(a, b)`: ceiling division, used to size the grid from a
  tensor's shape and the chosen block size (`cuda.md` §1.3's
  `(n + block - 1) / block` idiom, spelled as a named helper).
- `T.get_thread_binding(axis)`: returns the current thread's index within
  its block along the given axis (the TileLang analogue of CUDA C++'s
  `threadIdx.x`, needed when a kernel does need per-thread rather than
  purely tile-level indexing).

### 1.6 fp16

This study's protocol fixes all tensor inputs/outputs to `float16` --
declare kernel parameters as `T.Tensor((shape...), "float16")` (dtype names
are plain strings: `"float16"`, `"bfloat16"`, `"float32"`, matching
`torch`'s dtype names without the `torch.` prefix). As stated in the common
prompt template, accumulation precision is your choice: `T.alloc_fragment`
accumulators are commonly declared `"float32"` even when the surrounding
tensors are `"float16"` (as in §3.1's `C_local = T.alloc_fragment((block_M,
block_N), "float32")` for an `fp16` GEMM) -- `T.gemm`/`T.copy` handle the
narrowing cast to the `fp16` output tensor automatically when the shapes and
the copy's destination dtype require it.

### 1.7 Compilation entry points

Two equivalent ways to compile a `T.prim_func`:

```python
kernel = tilelang.compile(my_kernel, out_idx=[2], target="cuda")
```

or, as a decorator directly on the function (used in TileLang's own README
example, §3.1's GEMM pattern):

```python
@tilelang.jit
def my_kernel(A, B, ...):
    ...
```

`target="cuda"` selects the NVIDIA backend (the only one relevant on this
project's GPU); it can be omitted and TileLang will auto-detect it from the
available device. `out_idx` (only needed with the `tilelang.compile(...)`
call form, not `@tilelang.jit`) names which parameter position(s) are
outputs, letting the compiled kernel allocate and return them rather than
requiring the caller to pre-allocate an output tensor -- either call form
produces the same compiled kernel; §3's example uses the explicit
`tilelang.compile(..., out_idx=[2])` form because its `T.prim_func` is built
inside a helper function parameterized by shape (`build_elementwise_add_kernel`),
which the plain `@tilelang.jit` decorator form doesn't support directly.

### 1.8 Where errors surface

A malformed `T.prim_func` (wrong tile shapes passed to `T.copy`/`T.gemm`,
an out-of-range `T.Tensor` slice, a Python-level error in the function body)
is typically caught either at `tilelang.compile()` time (TVM's lowering
passes) or by the downstream `nvcc` compile step it eventually invokes --
either way, this is what this study logs as a compile failure, with the
raised exception's text (which may itself embed the underlying `nvcc`
diagnostic, per this project's own toolchain notes in CLAUDE.md) available
for the repair turn. A logic error that compiles fine (e.g. a wrong offset
in a `T.copy` slice, or an accumulator not `T.clear`-ed before a
`T.Pipelined` loop) produces wrong output values, caught by the correctness
check -- the same failure mode as every other language in this study.

## 2. Memory model

TileLang's memory model is explicit, like CUDA C++/PTX, but expressed at
tile granularity rather than per-thread/per-address:

| TileLang | Physical resource | Scope | CUDA C++ equivalent |
|---|---|---|---|
| `T.Tensor` kernel parameter | global memory (DRAM) | whole grid + host | plain device pointer |
| `T.alloc_shared` | on-chip shared memory | per-block (`T.Kernel` instance) | `__shared__` |
| `T.alloc_fragment` | register file (tensor-core-aware layout) | per-block, tile-shaped | plain local/register variables, but tile-shaped |
| `T.alloc_local` | registers (simple per-thread scratch) | per-thread | plain local/register variables |

A `T.Tensor` parameter's indexing (`A[by * block_M, ko * block_K]`, as used
in `T.copy` calls) addresses global memory directly -- there is no separate
"convert to global address space" step the way PTX needs `cvta.to.global`
(`ptx.md` §1.5, §2.1); TileLang's compiler resolves this from the
parameter's declared type.

### 2.1 Why explicit shared/fragment placement matters

Because `T.gemm` targets the hardware's tensor-core path specifically, its
operands (`A_shared`/`B_shared` in §3.1's example) generally need to be in
shared memory (matching the tensor-core instruction's expected input
staging) while its accumulator (`C_local`) needs to be in fragment memory
(matching the tensor-core instruction's output register layout) -- this is
why TileLang exposes the shared-vs-fragment distinction explicitly rather
than hiding it the way Triton's `tl.dot` does. For purely elementwise work
(`T.Parallel` without `T.gemm`), the shared-vs-fragment choice matters less;
either can hold the working tile, and `T.alloc_shared` is the more common
default when the tile is read more than once per block (reused across
neighboring output elements, e.g. in pooling/convolution).

### 2.2 Synchronization

Unlike CUDA C++, there is no manual `T.sync()`/`bar.sync` call needed
between a `T.copy` into shared memory and a subsequent read of that shared
tile within the same `T.Kernel` block -- TileLang's compiler inserts the
necessary barriers automatically based on the data dependencies between
`T.copy`/`T.gemm`/`T.Parallel` statements it sees. `T.Pipelined`'s
staggered overlap (§1.4) is likewise handled by the compiler; you write the
loop body once and it software-pipelines it under the hood.

## 3. Minimal complete example

Element-wise vector addition (2D, contiguous), written exactly as this
project's harness expects it (`ModelNew` calling a `tilelang.compile`-d
kernel, per `PROMPT_SPEC.md` §2's TileLang language block) -- this is
KernelBench's own stock illustrative fixture for the TileLang backend,
confirmed to compile and run correctly through the actual evaluation
harness (`eval_kernel_against_ref`, backend="tilelang") during this
project's 8/10 milestone:

```python
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_elementwise_add_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256,
                                  threads: int = 128, dtype: str = "float16"):

    @T.prim_func
    def elementwise_add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                C[y, x] = A[y, x] + B[y, x]

    return tilelang.compile(elementwise_add_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, dtype: str):
        key = (M, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_elementwise_add_kernel(M, N, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_c, B_c = A.contiguous(), B.contiguous()
        original_shape = A_c.shape
        A_c = A_c.view(-1, A_c.size(-1))
        B_c = B_c.view(-1, B_c.size(-1))
        M, N = A_c.shape
        kernel = self._get_kernel(M, N, "float16")
        C = kernel(A_c, B_c)
        return C.view(original_shape)
```

Reading it against §1: `T.Tensor((M, N), dtype)` declares the three global
tensor parameters (§2); `T.Kernel(...) as (bx, by)` opens the grid with one
block per `(block_M, block_N)` output tile (§1.1); `T.Parallel(block_M,
block_N)` iterates every element of that tile (§1.4), with `A[y, x]` /
`B[y, x]` / `C[y, x]` indexing directly into global memory (§2); the kernel
is compiled once per distinct `(M, N, dtype)` and cached, since
`tilelang.compile` specializes on the shapes baked into the `T.prim_func`
closure -- the same reason KernelBench's own fixture wraps construction in
a small cache keyed by shape.

### 3.1 Operator family guidance

The same patterns as `cuda.md` §3, `ptx.md` §3.2, and `triton.md` §3.1,
expressed with TileLang's explicit shared/fragment primitives (all are
patterns, not requirements):

- **Elementwise / activation**: identical structure to §3 -- `T.Parallel`
  over the output tile, no `T.alloc_shared`/`T.alloc_fragment` needed
  beyond what a `T.Kernel` block already provides.
- **Reduction / Softmax / Norm layers**: `T.alloc_shared` a tile of the
  row/reduced-over slice, `T.copy` it in, then `T.reduce_max`/`T.reduce_sum`
  (§1.4) -- no manual barrier needed (§2.2).
- **Pooling / Convolution**: `T.alloc_shared` a tile of the input patch
  including the halo needed for the kernel's receptive field, `T.copy` it
  in once per block, then `T.Parallel` over output elements with a nested
  loop (plain Python `for`, since the kernel-window extent is a compile-time
  constant in every task in this study) accumulating max/multiply-add from
  the shared tile.
- **Matmul-family**: `T.alloc_shared` for the two operand tiles,
  `T.alloc_fragment` for the accumulator, `T.Pipelined` over the K
  dimension with `T.copy` (global-to-shared) + `T.gemm` (shared-to-fragment
  accumulate) each iteration, `T.copy` the fragment back to the global
  output tile at the end -- exactly the GEMM pattern TileLang's own README
  shows (`matmul_relu`: same structure, with a `T.Parallel` ReLU epilogue
  applied to `C_local` before the final `T.copy`).
- **Cumulative/scan**: no built-in scan primitive; a per-row sequential
  Python `for` loop over the scan axis inside the `T.Kernel` block (reading
  and writing through a `T.alloc_shared` or `T.alloc_local` running value)
  is the simplest correct implementation.
- **Loss functions**: elementwise/per-row computation (`T.Parallel`)
  followed by a reduction (`T.reduce_sum`) to a scalar or per-row value.
- **Attention**: compose `T.gemm` (Q·K^T) + row Softmax (`T.reduce_max`/
  `T.reduce_sum` inside a `T.Parallel`/loop) + `T.gemm` (·V) -- either as
  separate `T.prim_func` kernels called in sequence from `forward`, or
  fused into one `T.Kernel` block with `T.Pipelined` over K/V tiles
  (the FlashAttention-style pattern, structurally close to the GEMM
  pattern above with an added online-softmax rescale step); either is
  acceptable if correct.

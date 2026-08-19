<!-- INTERNAL PROVENANCE (not injected -- spec_loader.py strips everything above
the "---" marker before use).
Sources:
- Triton "Vector Addition" tutorial, https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
  (retrieved 2026-08-19; the add_kernel/add() code in §3 below is quoted
  verbatim from this page's live fetch).
- Triton "Introduction" programming guide, https://triton-lang.org/main/programming-guide/chapter-1/introduction.html
  (retrieved 2026-08-19; "Blocked Program, Scalar Threads" framing and the
  block-level compiler-optimization list in §1.1/§2 are drawn from this
  page's live fetch).
- Triton language API reference, https://triton-lang.org/main/python-api/triton.language.html
  (function signatures for tl.load/tl.store/tl.arange/tl.dot/reduction ops,
  cross-checked against this project's installed triton==3.4.0 at
  scripts/smoke_triton.py's PASS run, see CLAUDE.md).
-->

---
# Triton Language Reference (excerpt for GPU kernel authoring)

Triton is a Python-embedded DSL, JIT-compiled to PTX/GPU machine code via
`triton.jit`. Its execution model is **"Blocked Program, Scalar Threads"** --
the inverse of CUDA C++'s "Scalar Program, Blocked Threads": in Triton, the
programmer writes one program instance that operates on a *block* (tile) of
data at once, and the compiler is responsible for mapping that block-level
program down to individual hardware threads, memory coalescing, shared-memory
staging, and instruction scheduling. There is no `threadIdx`, no manual
`__shared__` declaration, and no manual `__syncthreads()` in ordinary Triton
kernels -- the compiler inserts these automatically from the block-level
operations you write.

## 1. Syntax overview

### 1.1 Kernel definition and launch

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
my_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
```

- `@triton.jit` marks a function as a Triton kernel; it is compiled the
  first time it's called with a given set of argument types/shapes and
  cached thereafter.
- `tl.constexpr` parameters (e.g. `BLOCK_SIZE`) are compile-time constants --
  they must be Python ints/known-at-trace-time values, not runtime tensor
  values, because the compiler uses them to specialize the generated code
  (e.g. to allocate registers/shared memory of a fixed size).
- The launch uses `kernel[grid](args...)` syntax, where `grid` is either a
  tuple of ints or (as above) a function of the `meta` dict of `constexpr`
  arguments -- letting the grid size depend on `BLOCK_SIZE` without
  hardcoding it twice.
- `triton.cdiv(a, b)` computes ceiling division (`(a + b - 1) // b`), the
  standard way to compute "number of blocks needed to cover `n` elements".

### 1.1.1 Launch tuning parameters

`kernel[grid](args..., BLOCK_SIZE=1024, num_warps=4, num_stages=2)` accepts
two additional keyword arguments beyond the kernel's own `constexpr`
parameters:

- `num_warps`: how many warps (§ "warp" has the same 32-thread meaning as in
  CUDA C++, `cuda.md` §1.7) the compiler uses to execute one program
  instance. More warps means more parallelism within a block at the cost of
  register/shared-memory pressure per warp.
- `num_stages`: how many pipeline stages the compiler uses for software
  pipelining of global-memory loads against compute (relevant for
  matmul-style kernels with a loop over a K dimension, analogous to
  TileLang's explicit `T.Pipelined`, `tilelang.md` §1).

Both have working defaults; tuning them is a performance concern, not a
correctness requirement for this study.

### 1.2 Program identity and index computation

Each kernel instance ("program") identifies its position in the launch grid
with `tl.program_id(axis)` (axis `0`/`1`/`2` for a 1D/2D/3D grid) -- the
Triton analogue of CUDA C++'s `blockIdx`, except there is no per-thread
`threadIdx` to also read: a single `tl.program_id` call gives you the whole
block's position, and `tl.arange` generates the within-block offsets for
every element the block will touch:

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)   # a whole block of indices, not one
```

`tl.arange(0, BLOCK_SIZE)` produces a 1D tile of indices `[0, 1, ...,
BLOCK_SIZE-1]`; arithmetic on it (as above) is elementwise over the whole
tile at once. For 2D/multi-dimensional data, combine two `tl.arange` calls
via broadcasting (e.g. `row_offsets[:, None] * stride + col_offsets[None,
:]`) to build a 2D tile of addresses.

### 1.3 Masks (bounds checking)

Because a block processes `BLOCK_SIZE` elements at once and `n_elements` is
rarely an exact multiple of `BLOCK_SIZE`, out-of-bounds lanes within the
last block must be masked rather than branched around (Triton has no
per-lane early-exit the way CUDA C++'s `if (i < n) return;` does at the
thread level):

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # OOB lanes read 0.0 instead of faulting
tl.store(out_ptr + offsets, result, mask=mask)         # OOB lanes are simply not written
```

`other=` (default `0`) is the value substituted for masked-out lanes on a
load; it never affects `store`, which just skips masked lanes entirely.
Every `tl.load`/`tl.store` that might run past a tensor's real extent needs
a `mask=`, or it can read/write out of bounds.

### 1.4 Control flow

Ordinary Python `if`/`for`/`while` inside a `@triton.jit` function are
evaluated by the compiler at trace time if their condition depends only on
`tl.constexpr` values (compile-time specialization, e.g. looping a fixed
number of unrolled stages) -- they are not GPU-runtime branches in that
case. For a runtime-data-dependent loop bound (e.g. iterating over a
reduction axis whose length is a regular kernel argument, not a
`constexpr`), a plain Python `for` loop over a `range()` built from that
argument still works; Triton compiles it to an actual loop in the generated
code, executed uniformly by the whole block (there is no per-lane
divergence to reason about, since the program operates block-wise).

### 1.5 Common operations

- **Elementwise math**: ordinary Python operators (`+`, `*`, ...) and
  `tl.math`/`tl` functions (`tl.exp`, `tl.sqrt`, `tl.maximum`, `tl.minimum`,
  `tl.where(cond, a, b)`) apply elementwise across a whole tile.
- **Reductions**: `tl.sum(x, axis=...)`, `tl.max(x, axis=...)`,
  `tl.argmax(x, axis=...)` reduce a tile along one axis, producing a
  smaller tile (or a scalar if reducing the only axis) -- this replaces the
  manual shared-memory tree reduction CUDA C++/PTX need by hand (`cuda.md`
  §3, `ptx.md` §3.2): a block-wise Softmax is `x - tl.max(x, axis=0)` then
  `tl.exp(...)` then divide by `tl.sum(...)`, no explicit synchronization.
- **Matmul**: `tl.dot(a, b)` computes a block-level matrix multiply
  (mapped to tensor-core instructions where available), the Triton
  analogue of manually accumulating over `.shared`-memory tiles in CUDA
  C++/PTX.
- **Pointer arithmetic**: pointers are plain integers/tensors of integers;
  `x_ptr + offsets` (elementwise pointer + tile-of-ints) is how a tile of
  addresses is built, mirroring `cvta.to.global` + `mul.wide` +
  `add` in PTX (`ptx.md` §1.5) but expressed as one line of tile arithmetic.

### 1.6 fp16

This study's protocol fixes all tensor inputs/outputs to `float16`. Triton
infers element dtype from the input tensor's dtype at launch -- a kernel
written generically (no explicit `.to(tl.float32)`/`.to(tl.float16)` casts)
operates on `fp16` values directly when called with `fp16` tensors. As
stated in the common prompt template, accumulation precision is your
choice: `tl.dot(a, b)` and reduction ops accept an `input_precision`/
implicit accumulation in `float32` (the common, numerically safer default
for matmul and large reductions) even when the inputs and final output are
`fp16` -- cast a loaded tile up with `x.to(tl.float32)` before accumulating
and back down with `.to(tl.float16)` before the final `tl.store` if the
default behavior of a given op isn't already doing this. Concretely, a
reduction over `fp16` inputs with wide accumulation looks like:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # fp16 tile
acc = tl.sum(x.to(tl.float32), axis=0)                # accumulate in fp32
result = acc.to(tl.float16)                            # narrow at the end
tl.store(out_ptr + row_idx, result)
```

`tl.dot(a, b)` similarly accepts `fp16` (or `bf16`) operand tiles and by
default accumulates in `fp32` internally, exposing the result as whatever
dtype you cast it to before storing -- this is the standard mixed-precision
matmul pattern and requires no special flags beyond the `.to(...)` casts
shown above.

### 1.7 Block pointers (alternative addressing API)

Instead of building an explicit tile of addresses with `tl.arange` +
pointer arithmetic (§1.2-§1.3), `tl.make_block_ptr` describes a tile's
location declaratively -- useful for multi-dimensional tiles (e.g. matmul,
convolution) where manual stride arithmetic gets verbose:

```python
block_ptr = tl.make_block_ptr(
    base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
)
tile = tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
```

`boundary_check`/`padding_option` replace the explicit `mask=`/`other=` of
§1.3 for this API. Both addressing styles (explicit `tl.arange` arithmetic,
or `tl.make_block_ptr`) are equally valid; use whichever is clearer for a
given kernel's indexing pattern.

### 1.8 Where errors surface

A Python-level error in the kernel body (wrong argument count, an
unsupported operation on a `tl.constexpr`, a type error the Triton compiler
can detect) raises at call time, when the kernel is first JIT-compiled --
this is what this study logs as a compile failure, with the raised
exception's text available for the repair turn. A logic error that compiles
fine (e.g. a missing `mask=` on a load that touches unmapped memory, or a
wrong `offsets` computation) either produces wrong output values (caught by
the correctness check) or, for a sufficiently out-of-bounds access, an
illegal-memory-access error at kernel launch -- the same failure mode as
CUDA C++/PTX (`cuda.md` §1.10, `ptx.md` §1.9), just triggered by a masking
bug instead of a missing bounds-check branch.

### 1.9 Debugging aids

`tl.device_print("label", value)` prints a tile's values from within a
running kernel (routed through the same mechanism as a CUDA C++ `printf`
inside a `__global__` function) -- useful for inspecting intermediate
values while developing, though not needed for a kernel to be correct.
Setting the environment variable `TRITON_INTERPRET=1` runs a kernel through
a pure-Python interpreter instead of compiling it to GPU code, which turns
any Python exception (including ones from bugs a normal compiled run would
only show as silently wrong output) into an immediately visible traceback --
useful during development, irrelevant to how this study's harness evaluates
the final generated kernel (which always runs compiled, on-GPU).

## 2. Memory model

Triton has no manual memory-space qualifiers the way CUDA C++/PTX do
(`__shared__`/`.shared`, `__constant__`/`.const`) -- every `tl.load`/
`tl.store` operates on global memory (an ordinary device pointer plus a
tile of offsets, §1.3), and the compiler decides internally when and how to
stage data through shared memory / registers to make that efficient. This
is the central practical difference from CUDA C++ and PTX: you write what
tile of global memory to read/write and what to compute over it; you do
not write where intermediate values live in the memory hierarchy.

### 2.1 Contiguity and strides

For multi-dimensional tensors, a tile's addresses are computed from
per-axis strides, exactly mirroring how the same tensor's `.stride()`
would be used in raw pointer arithmetic:

```python
row_idx = tl.program_id(0)
col_offsets = tl.arange(0, BLOCK_N)
ptrs = base_ptr + row_idx * row_stride + col_offsets * col_stride
mask = col_offsets < n_cols
row = tl.load(ptrs, mask=mask, other=0.0)
```

As in CUDA C++'s coalescing discussion (`cuda.md` §2.1), a tile whose
innermost `tl.arange` axis matches the tensor's contiguous (stride-1)
dimension gets efficient, coalesced access; the compiler cannot fix a
fundamentally strided access pattern, only schedule around it.

### 2.2 Shared memory (implicit)

Operations like `tl.dot` (matmul) automatically stage operand tiles through
on-chip shared memory / tensor-core input registers as needed -- this
happens transparently. If a kernel needs to explicitly reuse a loaded tile
across multiple later reads within the same program instance (e.g. loading
an input tile once and reducing over it twice), simply keep it bound to a
Python variable in the kernel body -- the compiler manages its lifetime and
placement; there is no separate "allocate shared memory" step to write.

### 2.2.1 Autotuning (optional)

`@triton.autotune(configs=[...], key=[...])`, stacked above `@triton.jit`,
lets the compiler pick the best of several `(BLOCK_SIZE, num_warps,
num_stages)` combinations by benchmarking each at first call for a given
input shape (the `key` list names which kernel arguments trigger
re-tuning when they change, e.g. `key=["n_elements"]`). This is entirely
optional -- a single hardcoded configuration, as in the §3 example, is a
complete and correct kernel; autotuning only affects speed, and re-tuning
at first call adds one-time overhead that this study's timing protocol
(CLAUDE.md: warmup 25 / measure 100) already accounts for via its warmup
phase.

### 2.3 Numerical accumulation

Because reductions and `tl.dot` operate over a whole tile per call, the
accumulator's dtype (chosen via `.to(tl.float32)` casts, §1.6, or an op's
`acc`/accumulator-dtype argument where available) matters more than in a
scalar per-thread accumulation loop (CUDA C++/PTX) -- getting this right
for `fp16` inputs (accumulate wide, narrow at the end) is the main memory-
model-adjacent decision a Triton kernel in this study needs to make.

## 3. Minimal complete example

The element-wise vector-add kernel from Triton's own "Vector Addition"
tutorial (quoted verbatim), adapted to the `ModelNew` convention this
study's harness expects (`PROMPT_SPEC.md` §2's Triton language block):

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    output_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return triton_add(a, b)
```

Reading it against §1: `tl.program_id`/`tl.arange` build a tile of indices
for this program instance (§1.2); `mask` guards the last, partial block
(§1.3); `tl.load`/`tl.store` move a whole tile between global memory and
registers in one call (§2); the arithmetic (`x + y`) is ordinary Python,
applied elementwise across the tile.

### 3.1 Operator family guidance

The same patterns as `cuda.md` §3 and `ptx.md` §3.2, expressed with Triton's
block-level primitives (all are patterns, not requirements):

- **Elementwise / activation**: identical structure to §3 -- replace `x + y`
  with the target formula, using `tl` math functions (§1.5) as needed.
- **Reduction / Softmax / Norm layers**: launch one program per row
  (`tl.program_id(0)` indexes the row), load the whole row (or loop over
  chunks of it with a Python `for` if it exceeds one block, §1.4),
  `tl.sum`/`tl.max` (§1.5) replace the manual shared-memory tree reduction
  needed in CUDA C++/PTX.
- **Pooling / Convolution**: load the receptive-field window as a tile
  (extending `offsets` with the window's extent, masked at the input's
  edges for padding), reduce with `tl.max`/`tl.sum` or accumulate a
  multiply-add across a small unrolled or `constexpr`-bounded loop over the
  kernel window.
- **Matmul-family**: `tl.dot` (§1.5) over `BLOCK_M x BLOCK_K` /
  `BLOCK_K x BLOCK_N` tiles, looped over the K dimension in chunks of
  `BLOCK_K`, accumulating into a `float32` tile (§1.6, §2.3) that is cast
  down and stored at the end -- the standard Triton matmul tutorial pattern.
- **Cumulative/scan**: no built-in block-wide scan primitive; a per-row
  sequential Python `for` loop (§1.4) over the scan axis, or tiling the
  axis and combining a `tl.cumsum`-style local scan (available in recent
  Triton as `tl.cumsum`) with a running carry across tiles, are both
  acceptable.
- **Loss functions**: elementwise/per-row computation followed by a
  reduction (`tl.sum`) to a scalar or per-row value, composing §1.5's
  reduction ops directly.
- **Attention**: `tl.dot` (Q·K^T) + a numerically-stable row Softmax
  (§1.5, subtract the row max before `tl.exp`) + `tl.dot` (·V) --
  either as three separate kernel launches from `forward`, or fused into
  one kernel using an online-softmax accumulation loop over K/V blocks
  (the FlashAttention-style pattern); either is acceptable if correct. In
either case, use `tl.dot`'s default `fp32` accumulation (§2.3) for both
matmuls -- attention's numerics are sensitive to accumulation precision
because the Softmax normalization compounds any rounding error from the
first matmul.

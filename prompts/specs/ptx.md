<!-- INTERNAL PROVENANCE (not injected -- spec_loader.py strips everything above
the "---" marker before use).
Source: NVIDIA "Parallel Thread Execution ISA", https://docs.nvidia.com/cuda/parallel-thread-execution/
(PTX ISA version 8.7, matching the toolchain pinned in CLAUDE.md: nvcc 12.8.93).
Retrieved: 2026-08-19; directive/instruction semantics cross-checked against
the live fetch of that page (module structure, .param/.reg declarations,
special registers %tid/%ctaid/%ntid, ld/st/mov/mad/setp/bra/cvta.to.global
instruction set) plus this project's own toolchain verification. The minimal
example in §3 is NOT hand-written for this document -- it is
scripts/fixtures/vecadd.ptx, produced by `nvcc -arch=sm_120a -ptx` from a
throwaway .cu file during the 8/10 milestone and confirmed to assemble
(ptxas) and run (cuModuleLoad/cuLaunchKernel) correctly on this project's
actual GPU via scripts/smoke_ptx.py -- i.e. it is real, driver-validated PTX
for this exact target, not a paraphrase.
-->

---
# PTX ISA Reference (excerpt for GPU kernel authoring)

PTX (Parallel Thread Execution) is NVIDIA's low-level virtual ISA: an
assembly-like, statically-typed, register-based language that `ptxas`
assembles into a device-specific `.cubin`, or that the CUDA driver JIT-compiles
at load time. Unlike CUDA C++, there is no host/device split in the source
file itself, no C-family control-flow syntax, and no library calls -- every
value movement and arithmetic operation is an explicit instruction.

## 1. Syntax overview

### 1.1 Module structure

Every PTX module begins with three directives:

```ptx
.version 8.7
.target sm_120a
.address_size 64
```

- `.version`: the PTX ISA version this module is written against. `ptxas`
  rejects a module whose `.version` it doesn't support; use `8.7` to match
  this project's pinned toolchain (nvcc/ptxas 12.8.93, see CLAUDE.md).
- `.target`: the GPU architecture to assemble for. **This project's measured
  GPU requires `sm_120a`** (the `a` suffix selects Blackwell-family-specific
  instructions; plain `sm_120` is also valid but may not expose every
  hardware feature) -- do not use `sm_86` or any other value.
- `.address_size`: `64` for all currently supported architectures (32-bit
  addressing is legacy and unused here).

### 1.2 Kernel entry points

```ptx
.visible .entry kernel_name(
    .param .u64 param_0,      // e.g. a pointer argument
    .param .u32 param_1        // e.g. a scalar int argument
)
{
    // register declarations, then instructions
    ret;
}
```

- `.visible .entry`: marks a callable kernel (analogous to CUDA C++'s
  `__global__`). The name after `.entry` is what a host launcher (or this
  project's `harness/ptx/ptx_harness.py`, via `cuModuleGetFunction`) looks
  up by string.
- `.param` declarations list the kernel's arguments in order, each with an
  explicit bit-width type (`.u64` for a 64-bit pointer/unsigned int, `.u32`
  for a 32-bit unsigned int, `.f32` for a 32-bit float, `.f16` for a 16-bit
  float, etc.) -- there is no implicit argument marshaling; the type here
  must match what the launcher packs into `cuLaunchKernel`'s argument array
  (see `harness/ptx/ptx_harness.py::CuModuleRunner.launch`, which packs each
  Python int as a `ctypes.c_uint64` -- the driver reads only the number of
  bytes the `.param` declaration specifies, so this is safe for both pointer
  and 32-bit scalar arguments).
- A non-kernel callable function uses `.func` instead of `.entry` (rarely
  needed for this study; every task is a single kernel entry point).

### 1.3 Registers

Registers are declared per-kernel with a type and a count, then referenced
by index:

```ptx
.reg .pred   %p<2>;    // predicates (booleans), for setp/bra
.reg .b32    %r<6>;    // 32-bit untyped bits (also used for s32/u32 int math)
.reg .b64    %rd<11>;  // 64-bit untyped bits (addresses, s64/u64 math)
.reg .f32    %f<4>;    // 32-bit float
.reg .f16    %h<4>;    // 16-bit float
.reg .f64    %fd<4>;   // 64-bit float (double)
```

`%p<2>` declares registers `%p0`, `%p1` (2 total); reference them as
`%p0`/`%p1` in instructions. Registers are per-thread (there is no
block-shared register space -- shared *memory*, §2, is the mechanism for
that). PTX registers are virtual: `ptxas` performs its own allocation onto
physical hardware registers, so declaring "too many" is not a correctness
concern, only a potential performance/occupancy one.

### 1.4 Special registers (thread/block identity)

Read-only registers, available without declaration, mirroring CUDA C++'s
built-in variables:

| PTX | CUDA C++ equivalent | Meaning |
|---|---|---|
| `%tid.x/y/z` | `threadIdx.x/y/z` | thread index within its CTA (block) |
| `%ntid.x/y/z` | `blockDim.x/y/z` | CTA (block) dimensions |
| `%ctaid.x/y/z` | `blockIdx.x/y/z` | CTA (block) index within the grid |
| `%nctaid.x/y/z` | `gridDim.x/y/z` | grid dimensions |

Read into a general register with `mov`, e.g. `mov.u32 %r3, %ctaid.x;`.

### 1.5 Core instructions

Instructions follow `opcode.type destination, sources;`. Common ones:

- **Data movement**: `mov.u32 %r1, %r2;` (register-to-register or special
  register), `mov.u64 %rd1, param;`.
- **Memory**: `ld.param.u64 %rd1, [param_0];` (read a kernel parameter into a
  register), `ld.global.f32 %f1, [%rd8];` (read from global memory through a
  converted address), `st.global.f32 [%rd10], %f3;` (write to global
  memory). The `.param`/`.global` qualifier selects the address space
  (§2.1) -- it must match where the pointer actually points.
- **Arithmetic**: `add.s32 %r1, %r2, %r3;`, `mul.wide.s32 %rd1, %r2, %r3;`
  (32x32-bit multiply producing a 64-bit result -- the standard idiom for
  computing a byte offset from an element index and element size), `mad.lo.s32
  %r1, %r2, %r3, %r4;` (fused multiply-add: `%r1 = %r2*%r3 + %r4`, the
  idiomatic way to compute a flattened global thread index from
  `%ctaid`/`%ntid`/`%tid` in one instruction).
- **Comparison / control flow**: `setp.ge.s32 %p1, %r1, %r2;` (set predicate
  `%p1 = (%r1 >= %r2)`), `@%p1 bra LABEL;` (branch to `LABEL` if `%p1` is
  true; `@!%p1` branches if false). There is no `if`/`for` syntax -- every
  branch is an explicit predicated `bra` to a label, and labels are plain
  identifiers followed by `:` (e.g. `$L__BB0_2:`). A bounds check that skips
  out-of-range threads (the PTX analogue of CUDA C++'s `if (i < n) { ... }`)
  is: compute the index, `setp.ge` against `n`, `@%p1 bra SKIP;`, do the
  work, `SKIP: ret;`.
- **Address space conversion**: `cvta.to.global.u64 %rd4, %rd1;` converts a
  generic address (as received from a `.param` pointer) into an explicit
  global-space address before it can be used with `ld.global`/`st.global`.
  This step is easy to forget but required -- omitting it is a common
  source of `ptxas` errors or, if it assembles anyway, illegal-memory-access
  failures at runtime.

### 1.6 fp16

Half-precision values use `.f16` registers and `.b16` for raw bit
manipulation. `ptxas` on sm_120a supports native `.f16` arithmetic
instructions (`add.f16`, `mul.f16`, `fma.rn.f16`, ...) as well as the packed
two-at-a-time `.f16x2` forms; converting to `.f32` for the actual arithmetic
and back is also correct and often simpler to write by hand:

```ptx
.reg .f16 %h1, %h2, %h3;
.reg .f32 %f1, %f2, %f3;

ld.global.f16 %h1, [%rd8];
ld.global.f16 %h2, [%rd9];
cvt.f32.f16   %f1, %h1;
cvt.f32.f16   %f2, %h2;
add.f32       %f3, %f1, %f2;
cvt.rn.f16.f32 %h3, %f3;
st.global.f16 [%rd10], %h3;
```

As stated in the common prompt template, accumulation precision is your
choice -- computing in `.f32` and converting only at load/store boundaries
(as above) is a common, acceptable strategy, especially for reductions and
matmul-style accumulation where native `.f16` accumulation would lose
precision.

### 1.7 Loops

There is no `for`/`while` syntax; a loop is a label plus a conditional
branch back to it, exactly like the bounds-check pattern in §1.5 but
looping instead of skipping:

```ptx
	mov.u32 	%r_i, 0;              // loop counter i = 0
LOOP_HEAD:
	setp.ge.s32 	%p1, %r_i, %r_n;   // i >= n ?
	@%p1 bra 	LOOP_END;
	// ... loop body using %r_i ...
	add.s32 	%r_i, %r_i, 1;         // i++
	bra 	LOOP_HEAD;
LOOP_END:
```

This is the PTX-level building block for any kernel that accumulates over a
receptive field or a K-dimension (matmul, convolution, pooling, reductions,
cumulative scans -- see the operator-family guidance below), by hand instead
of through a compiler-generated loop as in CUDA C++/Triton/TileLang.

### 1.8 Type conversion (`cvt`)

`cvt.dtype_out.dtype_in dst, src;` converts between numeric types, e.g.
`cvt.f32.s32 %f1, %r1;` (int to float), `cvt.rn.f16.f32 %h1, %f1;`
(float to half, round-to-nearest -- the `.rn` rounding-mode qualifier is
required for narrowing float conversions), `cvt.s32.f32 %r1, %f1;`
(float to int, truncating unless a rounding-mode qualifier is given).

### 1.9 Where errors surface

`ptxas` rejects a module that fails to parse, references an undeclared
register, has a type mismatch between an instruction and its operands, or
targets an unsupported `.version`/`.target` pair -- this is what this
study's harness (`harness/ptx/ptx_harness.py::assemble_ptx`) reports as a
compile failure, with `ptxas`'s stderr text available verbatim for the
repair turn. A module that assembles but has an incorrect address-space
conversion, a wrong `mul.wide` operand order, or an out-of-bounds index
typically does not fail at assembly -- it produces wrong values or an
illegal-memory-access error at `cuLaunchKernel`/`cuCtxSynchronize`, exactly
as in CUDA C++ (`cuda.md` §1.10).

## 2. Memory model

### 2.1 Address spaces

PTX exposes the same physical hierarchy as CUDA C++, addressed explicitly by
instruction qualifier rather than by C-style pointer-type qualifiers:

| PTX space | Qualifier on `ld`/`st` | Scope | CUDA C++ equivalent |
|---|---|---|---|
| Global | `.global` | whole grid + host | plain device pointer |
| Shared | `.shared` | per-CTA (block) | `__shared__` |
| Local | `.local` | per-thread | compiler-managed local/spill memory |
| Parameter | `.param` | read-only, per-thread (kernel args) | function argument |
| Constant | `.const` | whole grid, read-only | `__constant__` |

A kernel parameter (`.param`) that is itself a pointer must be converted to
the `.global` address space with `cvta.to.global` (§1.5) before the pointee
can be read/written with `ld.global`/`st.global` -- the raw parameter value
is a *generic* address, not yet tagged with a specific space.

### 2.2 Shared memory declaration

Statically-sized shared memory is declared at module or kernel scope with
`.shared` and an explicit byte size/alignment:

```ptx
.shared .align 4 .b8 tile[1024];   // 1024 bytes, 4-byte aligned
```

Access it with `ld.shared`/`st.shared`, using an address computed relative
to the `tile` symbol (via `mov.u32 %r, tile;` plus offset arithmetic, or the
generic-to-shared conversion `cvta.shared`). As in CUDA C++ (`cuda.md`
§1.4), a barrier is required before one thread reads shared memory another
thread just wrote: `bar.sync 0;` (the PTX equivalent of `__syncthreads()`;
the `0` names the barrier resource, `0` is the default used by essentially
every kernel).

### 2.3 Alignment and vectorized access

`ld`/`st` instructions can move more than one element per instruction when
addresses are suitably aligned, e.g. `ld.global.v4.f32 {%f1,%f2,%f3,%f4},
[%rd1];` loads four consecutive `f32` values in one instruction if `%rd1` is
16-byte aligned. This is a bandwidth optimization (fewer, wider memory
transactions -- the PTX-level analogue of CUDA C++'s coalescing discussion,
`cuda.md` §2.1); a scalar `ld.global.f32` per element, as in the minimal
example below, is correct and sufficient for this study's correctness bar.

## 3. Minimal complete example

A complete, driver-validated element-wise vector-add kernel targeting this
project's GPU (`sm_120a`) -- produced by `nvcc -arch=sm_120a -ptx` and
confirmed via `ptxas` assembly + `cuModuleLoad`/`cuLaunchKernel` execution
against real PyTorch tensors during this project's toolchain verification
(exact numerical match against `a + b`):

```ptx
.version 8.7
.target sm_120a
.address_size 64

.visible .entry vecadd(
	.param .u64 vecadd_param_0,
	.param .u64 vecadd_param_1,
	.param .u64 vecadd_param_2,
	.param .u32 vecadd_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<6>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<11>;

	ld.param.u64 	%rd1, [vecadd_param_0];
	ld.param.u64 	%rd2, [vecadd_param_1];
	ld.param.u64 	%rd3, [vecadd_param_2];
	ld.param.u32 	%r2, [vecadd_param_3];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd4, %rd7;
	ld.global.f32 	%f1, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.f32 	%f2, [%rd9];
	add.f32 	%f3, %f1, %f2;
	add.s64 	%rd10, %rd6, %rd7;
	st.global.f32 	[%rd10], %f3;
$L__BB0_2:
	ret;
}
```

Reading it against §1: three `.u64` pointer params (`a`, `b`, `out`) and one
`.u32` scalar param (`n`, the element count) are loaded from `.param` space
(§1.5, §2.1); the flattened global thread index is computed with
`mad.lo.s32` from the special registers (§1.4); the bounds check is a
`setp.ge` + predicated `bra` past the body (§1.5); each pointer is converted
to `.global` space with `cvta.to.global` (§1.5, §2.1) before use;
`mul.wide.s32 ..., 4` turns the element index into a byte offset for
4-byte `f32` elements (use `2` instead of `4` for `f16` data, per §1.6); and
the result is written back with `st.global.f32`.

### 3.1 Harness wiring

Per `PROMPT_SPEC.md` §2's PTX language block, wrap the PTX module as a
Python string constant and drive it through the project's harness API
(`harness/ptx/ptx_harness.py`, already imported by the evaluation
pipeline -- do not redefine `ptx_load`/`ptx_launch`, only call them):

```python
import torch
import torch.nn as nn

PTX_SOURCE = r"""
.version 8.7
.target sm_120a
.address_size 64
... (the module text, as above) ...
"""


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = ptx_load(PTX_SOURCE)

    def forward(self, a, b):
        out = torch.empty_like(a)
        n = a.numel()
        block = 256
        grid = ((n + block - 1) // block, 1, 1)
        ptx_launch(self.module, "vecadd", grid, (block, 1, 1),
                   [a.data_ptr(), b.data_ptr(), out.data_ptr(), n])
        return out
```

`ptx_load`/`ptx_launch` handle `ptxas` assembly, `cuModuleLoad`, and
`cuLaunchKernel` -- write only the PTX module and the launch-parameter
computation (grid/block sizes, argument order matching the `.param` list).

### 3.2 Operator family guidance

The same patterns as `cuda.md` §3, expressed with the primitives above
(all are patterns, not requirements):

- **Elementwise / activation**: identical structure to §3's example -- one
  thread per output element (§1.5's bounds-check idiom), no shared memory.
- **Reduction / Softmax / Norm layers**: one CTA per row; each thread
  accumulates a strided subset with the loop pattern in §1.7, stores its
  partial value to `.shared` memory (§2.2), `bar.sync 0;`, then a tree
  reduction over the shared array (successive halvings of the active thread
  range, each followed by `bar.sync 0;`) before one thread's `st.global`.
- **Pooling / Convolution**: nested §1.7-style loops over the receptive
  field/kernel window per output element, accumulating into a register
  (`max` via `max.f32`/`max.f16`, or a running multiply-add via `fma.rn.f32`
  for convolution); a `.shared`-memory input tile with the halo region
  avoids redundant `ld.global` traffic across neighboring output elements.
- **Matmul-family**: tile both operands into `.shared` memory per K-chunk
  (§2.2), `bar.sync 0;` after each tile load, accumulate the dot product
  in a register across an outer loop (§1.7) over K-chunks, `bar.sync 0;`
  again before the next chunk's shared-memory tile is overwritten, then one
  `st.global` per output element at the end.
- **Cumulative/scan**: a strictly sequential per-row loop (§1.7) over the
  scan axis is the simplest correct implementation by hand; a parallel
  block-level scan is possible but not required for correctness.
- **Loss functions**: elementwise/per-row computation (as above) followed
  by a reduction to a scalar (as above) -- the two patterns compose exactly
  as in `cuda.md`.
- **Attention**: compose matmul + row-wise Softmax + matmul as in `cuda.md`
  §3 -- most practically as three separate `.entry` kernels in the same PTX
  module, called in sequence from `ModelNew.forward` via three `ptx_launch`
  calls (a single fused kernel is also acceptable if correct, but
  substantially longer to author directly in PTX).

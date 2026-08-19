"""cuModuleLoad wrapper for the PTX track (LRPL / low-abstraction cell of the 2x2).

Why this exists: PTX is not a language torch.utils.cpp_extension or the Triton/
TileLang JIT paths can load. An LLM-generated ``.ptx`` file has to be:
  1. assembled with ptxas -> .cubin  (this step gives us the "compile success /
     compile error message" signal the protocol needs for the repair turn)
  2. loaded with the CUDA driver API (cuModuleLoad) -> cuLaunchKernel

This module is harness infrastructure, not experimental code — it must stay
generic (compile-and-run *any* single-entry-point PTX kernel against caller-
supplied device pointers), so it can execute whatever the LLM produced. It must
never fix up or rewrite the PTX text it is given.

Kept dependency-free (ctypes only, no ``cuda-python`` package) since generation
machines may not have that installed.
"""
import ctypes
import os
import subprocess
import tempfile
from dataclasses import dataclass


def _cuda():
    lib = ctypes.CDLL("libcuda.so.1")
    return lib


class PTXCompileError(RuntimeError):
    """Raised when ptxas fails. .stderr holds the raw compiler message for the
    repair-turn protocol (CLAUDE.md: '컴파일 에러 메시지만 주는 수리 1턴')."""

    def __init__(self, stderr: str, cmd: list[str]):
        self.stderr = stderr
        self.cmd = cmd
        super().__init__(stderr)


@dataclass
class PTXKernel:
    module: int  # CUmodule
    function: int  # CUfunction


def assemble_ptx(ptx_src: str, arch: str, ptxas_path: str = "ptxas", extra_args: list[str] | None = None) -> bytes:
    """Run ptxas on PTX source text. Returns cubin bytes.

    arch: e.g. "sm_120a" -- must match CLAUDE.md's recorded compute capability.
    Raises PTXCompileError with the raw ptxas stderr on failure.
    """
    with tempfile.TemporaryDirectory() as td:
        ptx_path = os.path.join(td, "kernel.ptx")
        cubin_path = os.path.join(td, "kernel.cubin")
        with open(ptx_path, "w") as f:
            f.write(ptx_src)
        cmd = [ptxas_path, f"-arch={arch}", "-o", cubin_path, ptx_path]
        if extra_args:
            cmd[1:1] = extra_args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PTXCompileError(proc.stderr, cmd)
        with open(cubin_path, "rb") as f:
            return f.read()


def _check(lib, status, what):
    if status != 0:
        buf = ctypes.c_char_p()
        try:
            lib.cuGetErrorString(status, ctypes.byref(buf))
            msg = buf.value.decode() if buf.value else "?"
        except Exception:
            msg = "?"
        raise RuntimeError(f"CUDA driver call failed ({what}): status={status} msg={msg}")


class CuModuleRunner:
    """Loads a cubin via the driver API and launches one kernel by name.

    Usage:
        runner = CuModuleRunner(device_index=0)
        cubin = assemble_ptx(ptx_src, arch="sm_120a")
        fn = runner.load(cubin, "vecadd")
        runner.launch(fn, grid=(g,1,1), block=(b,1,1), args=[x_ptr, y_ptr, out_ptr, n])
    """

    def __init__(self, device_index: int = 0):
        self.lib = _cuda()
        self.lib.cuInit(0)
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        _check(self.lib, self.lib.cuDeviceGet(ctypes.byref(self.device), device_index), "cuDeviceGet")
        # Reuse the primary context (same context PyTorch's CUDA runtime holds),
        # so torch tensor device pointers are valid without a separate context.
        _check(self.lib, self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device), "cuDevicePrimaryCtxRetain")
        _check(self.lib, self.lib.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")

    def load(self, cubin: bytes, entry_point: str) -> int:
        module = ctypes.c_void_p()
        _check(self.lib, self.lib.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        function = ctypes.c_void_p()
        _check(self.lib, self.lib.cuModuleGetFunction(ctypes.byref(function), module, entry_point.encode()), "cuModuleGetFunction")
        return function

    def launch(self, function, grid: tuple[int, int, int], block: tuple[int, int, int], args: list, shared_mem: int = 0):
        """args: list of ctypes-compatible scalars/pointers (as produced by tensor.data_ptr())."""
        c_args = []
        arg_ptrs = (ctypes.c_void_p * len(args))()
        holders = []  # keep references alive
        for i, a in enumerate(args):
            if isinstance(a, int):
                # Could be a device pointer (from tensor.data_ptr()) or an int32 scalar.
                # Caller is responsible for wrapping scalars distinctly if needed;
                # for this harness we treat all ints as 64-bit (pointer-sized) by default
                # unless wrapped in ctypes already.
                c = ctypes.c_uint64(a)
            else:
                c = a
            holders.append(c)
            arg_ptrs[i] = ctypes.cast(ctypes.byref(c), ctypes.c_void_p)

        status = self.lib.cuLaunchKernel(
            function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            None,  # default stream
            arg_ptrs,
            None,
        )
        _check(self.lib, status, "cuLaunchKernel")

    def synchronize(self):
        _check(self.lib, self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


# ---------------------------------------------------------------------------
# ptx_load / ptx_launch: the API surface promised to the LLM by
# prompts/PROMPT_SPEC.md §2's PTX language block. Generated ModelNew code
# calls exactly these two names -- keep the signatures in lockstep with the
# prompt text (a signature change here without a matching PROMPT_SPEC.md
# change breaks every already-issued PTX prompt for the current run).
#
#   module = ptx_load(PTX_SOURCE)
#   ptx_launch(module, "kernel_name", grid, block, args)
#
# Thin wrappers over assemble_ptx + CuModuleRunner above: one lazily-created
# process-wide runner (so ModelNew doesn't need to construct one), one arch
# auto-detected from the actual device (so generated code never has to embed
# it), and a per-(module, name) function-handle cache (cuModuleGetFunction is
# a real driver call; no need to repeat it on every forward()).
# ---------------------------------------------------------------------------

_runner: "CuModuleRunner | None" = None
_module_cache: dict[int, ctypes.c_void_p] = {}
_function_cache: dict[tuple[int, str], ctypes.c_void_p] = {}


def _get_runner() -> "CuModuleRunner":
    global _runner
    if _runner is None:
        _runner = CuModuleRunner(device_index=0)
    return _runner


def _detect_arch() -> str:
    # [sync-needed] the family-specific "a" suffix (sm_90a, sm_120a, ...) is
    # only valid for architectures that define family-specific PTX features --
    # it hard-errors on ptxas for others (e.g. Ampere sm_80: "Value 'sm_80a'
    # is not defined for option 'gpu-name'"). Detected on the kernel-lang-2x2
    # A100 probe; only sm_120a was ever exercised on the primary sm_120 box so
    # this never surfaced there.
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    suffix = "a" if cc in (90, 100, 101, 120) else ""
    return f"sm_{major}{minor}{suffix}"


def ptx_load(ptx_source: str, arch: str | None = None) -> int:
    """Assemble a PTX module (ptxas) and load it via the CUDA driver.

    Returns an opaque module handle to pass to ptx_launch. Raises
    PTXCompileError (with .stderr = the raw ptxas message) on assembly
    failure -- this is what the repair-turn protocol (PROMPT_SPEC.md §3.3)
    surfaces back to the model verbatim.
    """
    runner = _get_runner()
    cubin = assemble_ptx(ptx_source, arch=arch or _detect_arch())
    module = ctypes.c_void_p()
    _check(runner.lib, runner.lib.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
    _module_cache[module.value] = module
    return module.value


def ptx_launch(module: int, kernel_name: str, grid: tuple[int, int, int],
                block: tuple[int, int, int], args: list, shared_mem: int = 0) -> None:
    """Launch `kernel_name` from a module returned by ptx_load.

    args: device pointers (tensor.data_ptr()) and/or plain Python ints for
    scalar kernel parameters, in the exact order of the kernel's .param list.
    """
    runner = _get_runner()
    key = (module, kernel_name)
    if key not in _function_cache:
        mod_handle = _module_cache[module]
        function = ctypes.c_void_p()
        _check(runner.lib, runner.lib.cuModuleGetFunction(ctypes.byref(function), mod_handle, kernel_name.encode()),
               "cuModuleGetFunction")
        _function_cache[key] = function
    runner.launch(_function_cache[key], grid=grid, block=block, args=args, shared_mem=shared_mem)

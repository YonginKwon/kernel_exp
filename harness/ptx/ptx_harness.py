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

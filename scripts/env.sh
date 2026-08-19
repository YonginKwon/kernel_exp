#!/usr/bin/env bash
# Toolchain environment for kernel-lang-2x2 on this machine (RTX PRO 6000 Blackwell, sm_120).
# `source` this before compiling/running any of the 4 backends (CUDA C++, Triton,
# TileLang, PTX) or their smoke tests. See CLAUDE.md "실측 환경" for why each var
# is needed -- short version: system nvcc (12.0) can't target sm_120a, system g++
# (13.3) is too new for nvcc 12.8, and TileLang's nvcc invocation doesn't pass its
# own -I for CUDA_HOME so the *system* include search order has to be fixed via
# CPATH or it silently falls back to /usr/include's stale cuda_fp8.h.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_HOME="$REPO_ROOT/third_party/cuda-sm120-toolchain"
export PATH="$REPO_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++-12
export CPATH="$CUDA_HOME/include"
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

# sanity echo (comment out if this gets noisy in generate.py logs)
if [ "${KERNEL2X2_ENV_QUIET:-0}" != "1" ]; then
    echo "[env] CUDA_HOME=$CUDA_HOME"
    echo "[env] nvcc: $(command -v nvcc) ($(nvcc --version | tail -1))"
    echo "[env] CXX=$CXX ($($CXX --version | head -1))"
fi

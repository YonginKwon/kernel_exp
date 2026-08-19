#!/usr/bin/env bash
# ============================================================================
# scripts/serve_local.sh -- DEFAULT generation path (2026-08-19 re-plan): vLLM
# serving on THIS server (PRO 6000, single GPU) for one of the two open-weight
# models at a time. H100 (scripts/serve_h100.sh) is now an opportunistic
# accelerator, not the default -- it isn't reliably available.
#
# Self-contained: conda-or-venv env creation, GPU occupancy preflight
# (nvidia-smi, aborts if busy -- no sudo needed anywhere), health-checked
# launch, and a JSON manifest (HF revision, vLLM version, dtype, GPU model)
# for CLAUDE.md's "실행 후 기록" fields, same shape as serve_h100.sh's.
#
# Usage:
#   scripts/serve_local.sh gptoss              # openai/gpt-oss-120b, port 8000
#   scripts/serve_local.sh qwen                 # Qwen3-Coder-30B-A3B-Instruct, port 8001 (CONFIRMED model, PI 2026-08-19)
#   scripts/serve_local.sh qwen --try-next-fp8  # opt-in: retry the original 80B-A3B-FP8 ladder (see below)
#   scripts/serve_local.sh --stop
#   scripts/serve_local.sh --status
#
# CONFIRMED (CLAUDE.md, PI decision 2026-08-19): this experiment's "Qwen" model
# is Qwen/Qwen3-Coder-30B-A3B-Instruct. The original target, Qwen3-Coder-Next-
# FP8 (80B-A3B), segfaults on this GPU's hybrid Gated-DeltaNet kernel path
# regardless of context size (not an OOM -- confirmed by direct investigation,
# see CLAUDE.md) -- so `qwen` serves the confirmed model directly, no ladder,
# no wasted multi-minute failed attempts on every serve.
#
# --try-next-fp8 re-enables the original 3-step downgrade ladder for future
# retesting (e.g. after a vLLM/flashinfer update that might fix the hybrid-
# attention kernel) -- NOT part of the default path:
#   1. Qwen3-Coder-Next-FP8, full context
#   2. same checkpoint, --max-model-len reduced to 65536
#   3. fall back to Qwen3-Coder-30B-A3B-Instruct (== today's default anyway)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/vllm"
ENV_DIR="$REPO_ROOT/.venv-vllm"
mkdir -p "$LOG_DIR"

GPTOSS_REPO="openai/gpt-oss-120b"
GPTOSS_PORT=8000
QWEN_REPO="Qwen/Qwen3-Coder-Next-FP8"
QWEN_FALLBACK_REPO="Qwen/Qwen3-Coder-30B-A3B-Instruct"
QWEN_PORT=8001
GPU_INDEX=0

ACTION="serve"
TARGET=""
TRY_NEXT_FP8=0

for arg in "$@"; do
    case "$arg" in
        gptoss|qwen) TARGET="$arg" ;;
        --try-next-fp8) TRY_NEXT_FP8=1 ;;
        --stop) ACTION="stop" ;;
        --status) ACTION="status" ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[serve_local] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

log() { echo "[serve_local] $*"; }
pidfile() { echo "$LOG_DIR/local_$1.pid"; }

# ----------------------------------------------------------------------------
# --status / --stop
# ----------------------------------------------------------------------------
if [ "$ACTION" = "status" ]; then
    for name in gptoss qwen; do
        pf="$(pidfile "$name")"
        if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
            log "$name: RUNNING (pid $(cat "$pf"))"
        else
            log "$name: not running"
        fi
    done
    exit 0
fi

if [ "$ACTION" = "stop" ]; then
    for name in gptoss qwen; do
        pf="$(pidfile "$name")"
        if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
            log "stopping $name (pid $(cat "$pf"))"
            kill "$(cat "$pf")"
            rm -f "$pf"
        else
            log "$name: not running, nothing to stop"
        fi
    done
    exit 0
fi

if [ -z "$TARGET" ]; then
    log "FATAL: specify a model: 'gptoss' or 'qwen' (or --stop / --status)."
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. GPU pre-flight.
# ----------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found."
    exit 1
fi

log "GPU pre-flight check:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "$LOG_DIR/preflight_nvidia_smi.txt"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_INDEX")"

read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX" | tr ',' ' ')
if [ "$USED" -gt 500 ] || [ "$UTIL" -gt 5 ]; then
    log "FATAL: GPU $GPU_INDEX appears occupied (${USED}MiB used, ${UTIL}% util). Refusing to start."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG_DIR/preflight_nvidia_smi.txt"
    log "This script will not kill processes it didn't start. Clear it yourself first if stale."
    exit 1
fi
log "GPU $GPU_INDEX ($GPU_NAME) idle -- proceeding."

# ----------------------------------------------------------------------------
# 2. Python environment (no sudo).
# ----------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    log "conda found -- using conda env 'kernel2x2-vllm'"
    if ! conda env list | grep -q "^kernel2x2-vllm "; then
        conda create -y -n kernel2x2-vllm python=3.12
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate kernel2x2-vllm
else
    log "conda not found -- using venv at $ENV_DIR"
    if [ ! -d "$ENV_DIR" ]; then
        python3 -m venv "$ENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
fi
PY=python

log "installing/upgrading vllm (>=0.15.0) + huggingface_hub"
pip install -q --upgrade pip
pip install -q --upgrade "vllm>=0.15.0" huggingface_hub

VLLM_VERSION="$($PY -c 'import vllm; print(vllm.__version__)')"
log "vllm version: $VLLM_VERSION"

# ----------------------------------------------------------------------------
# 2.1 CUDA toolchain fix, discovered running this exact script (2026-08-19):
# vLLM's flashinfer dependency JIT-compiles a sampling kernel and, on this
# GPU (sm_120a), needs an nvcc reporting CUDA >= 12.9 (flashinfer's own
# device-capability check requires it for SM 12.x) -- otherwise it silently
# falls back to a misleading "FlashInfer requires GPUs with sm75 or higher"
# error. `pip install vllm` here pulls torch's own bundled CUDA (a "cu13"
# nvidia/cu13/ tree under site-packages) which HAS an nvcc >= 12.9 -- but its
# sibling `nvidia-cuda-runtime` package version can land on a DIFFERENT
# major.minor than the nvcc package resolved (both are independently
# versioned pip packages), and flashinfer hard-asserts they match
# (CUDART_VERSION vs __CUDACC_VER__). Force them to match explicitly rather
# than trust whatever pip's resolver happened to pick.
CU_DIR="$($PY -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
if [ -d "$CU_DIR" ]; then
    NVCC_VER="$("$CU_DIR/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    log "pinning nvidia-cuda-runtime to match bundled nvcc (release $NVCC_VER)"
    if ! pip install -q "nvidia-cuda-runtime==${NVCC_VER}.*"; then
        log "WARNING: could not pin an exact nvidia-cuda-runtime==${NVCC_VER}.* match -- if"
        log "the qwen/gptoss launch below fails with a flashinfer 'CUDA compiler and CUDA"
        log "toolkit headers are incompatible' error, this is why; pin it by hand."
    fi
    rm -rf "${HOME}/.cache/flashinfer"  # clear any cache built against the old mismatch
    export CUDA_HOME="$CU_DIR"
    export PATH="$CU_DIR/bin:$PATH"
    export CPATH="$CU_DIR/include"
    log "CUDA_HOME=$CUDA_HOME (nvcc $("$CU_DIR/bin/nvcc" --version | tail -1))"
else
    log "WARNING: no bundled nvidia/cu13 tree found under this venv's torch install --"
    log "flashinfer's sm_120a JIT compile may fail. See CLAUDE.md's toolchain notes."
fi

# ----------------------------------------------------------------------------
# 3. Helpers.
# ----------------------------------------------------------------------------
resolve_hf_revision() {
    local repo="$1"
    # HF cache dirnames use "--" between org and name (e.g. models--openai--gpt-oss-120b),
    # not a single "-" -- confirmed against a real download, 2026-08-19.
    local cache_name="models--$(echo "$repo" | sed 's|/|--|')"
    local hub_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/$cache_name/snapshots"
    if [ -d "$hub_dir" ]; then
        ls "$hub_dir" | head -1
    else
        echo "UNKNOWN"
    fi
}

# Launch one attempt; returns 0=healthy, 1=process died early, 2=timed out.
# Sets ATTEMPT_PID as a side effect.
try_launch() {
    local repo="$1" port="$2" logfile="$3"; shift 3
    local extra_args=("$@")
    log "attempt: model=$repo port=$port extra_args='${extra_args[*]:-<none>}'"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" nohup "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$repo" --port "$port" --tensor-parallel-size 1 --trust-remote-code \
        "${extra_args[@]}" > "$logfile" 2>&1 &
    ATTEMPT_PID=$!

    local waited=0 max_wait=1800
    while [ "$waited" -lt "$max_wait" ]; do
        if ! kill -0 "$ATTEMPT_PID" 2>/dev/null; then
            log "attempt died early (see $logfile)"
            return 1
        fi
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            log "attempt healthy."
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    log "attempt timed out after ${max_wait}s without dying or becoming healthy -- killing it."
    kill "$ATTEMPT_PID" 2>/dev/null || true
    return 2
}

looks_like_oom() {
    # \bOOM\b matters: vLLM/torch's CUDACachingAllocator logs the abbreviated
    # "memory allocation failed with OOM" (no literal "out of memory"
    # substring) when a real OOM happens during warmup/profiling, which the
    # other alternatives alone would miss (confirmed against a real gpt-oss-
    # 120b OOM->SIGSEGV during warmup, 2026-08-19).
    grep -qiE "out of memory|CUDA error: out of memory|torch\.(cuda\.)?OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|\bOOM\b" "$1" 2>/dev/null
}

write_manifest() {
    local name="$1" repo="$2" port="$3" revision="$4" downgraded="$5" reason="$6" max_len="$7"
    cat > "$LOG_DIR/${name}_manifest.json" <<JSON
{
  "name": "$name",
  "hf_repo": "$repo",
  "hf_revision": "$revision",
  "vllm_version": "$VLLM_VERSION",
  "port": $port,
  "tensor_parallel_size": 1,
  "gpu_model": "$GPU_NAME",
  "cuda_visible_devices": "$GPU_INDEX",
  "downgraded": $downgraded,
  "downgrade_reason": "$reason",
  "max_model_len_override": "$max_len",
  "base_url": "http://localhost:${port}/v1",
  "started_at": "$(date -Iseconds)"
}
JSON
    log "wrote $LOG_DIR/${name}_manifest.json -- copy into CLAUDE.md's '실행 후 기록' fields."
}

# ----------------------------------------------------------------------------
# 4. Launch.
# ----------------------------------------------------------------------------
if [ "$TARGET" = "gptoss" ]; then
    # --max-model-len 32768 + --gpu-memory-utilization 0.85: gpt-oss-120b's
    # default (much longer) context, combined with vLLM's default 0.9 memory
    # utilization, over-commits this GPU during the profiling/warmup pass
    # (66GiB weights leaves too little slack) and crashes with a CUDA OOM ->
    # SIGSEGV during warmup. --enforce-eager: even with the above fix, CUDA
    # graph capture itself segfaults on this GPU/vLLM/torch combination
    # (cuCtxSynchronize crash during "Profiling CUDA graph memory") --
    # eager mode skips graph capture entirely, trading some throughput for
    # actual stability. All three found running this exact script,
    # 2026-08-19. 32768 is generous headroom over this project's actual
    # protocol (prompt + 8192 max output tokens, PROMPT_SPEC.md #4).
    if try_launch "$GPTOSS_REPO" "$GPTOSS_PORT" "$LOG_DIR/gptoss.log" \
        --max-model-len 32768 --gpu-memory-utilization 0.85 --enforce-eager; then
        echo "$ATTEMPT_PID" > "$(pidfile gptoss)"
        write_manifest gptoss "$GPTOSS_REPO" "$GPTOSS_PORT" "$(resolve_hf_revision "$GPTOSS_REPO")" false "" "32768"
    else
        log "FATAL: gpt-oss-120b failed to come up. See $LOG_DIR/gptoss.log"
        exit 1
    fi

elif [ "$TARGET" = "qwen" ] && [ "$TRY_NEXT_FP8" -eq 0 ]; then
    # Confirmed model (PI decision 2026-08-19, CLAUDE.md) -- serve directly,
    # no ladder. --enforce-eager: CUDA graph capture segfaults on this
    # GPU/vLLM/torch combination regardless of model (confirmed with
    # gpt-oss-120b, 2026-08-19).
    log "serving CONFIRMED qwen model: ${QWEN_FALLBACK_REPO} (pass --try-next-fp8 to instead"
    log "retry the original 80B-A3B-FP8 ladder)"
    if try_launch "$QWEN_FALLBACK_REPO" "$QWEN_PORT" "$LOG_DIR/qwen.log" --enforce-eager; then
        echo "$ATTEMPT_PID" > "$(pidfile qwen)"
        write_manifest qwen "$QWEN_FALLBACK_REPO" "$QWEN_PORT" "$(resolve_hf_revision "$QWEN_FALLBACK_REPO")" true \
            "Qwen3-Coder-Next-FP8 segfaults on this GPU's hybrid Gated-DeltaNet kernel path (not OOM); 30B-A3B-Instruct confirmed as the experiment's Qwen model (PI decision 2026-08-19, CLAUDE.md)" ""
        log "qwen up: ${QWEN_FALLBACK_REPO}."
        exit 0
    else
        log "FATAL: ${QWEN_FALLBACK_REPO} failed to come up. See $LOG_DIR/qwen.log"
        exit 1
    fi

elif [ "$TARGET" = "qwen" ] && [ "$TRY_NEXT_FP8" -eq 1 ]; then
    log "=== --try-next-fp8: attempt 1/3, full context, ${QWEN_REPO} ==="
    if try_launch "$QWEN_REPO" "$QWEN_PORT" "$LOG_DIR/qwen_attempt1.log" --enforce-eager; then
        echo "$ATTEMPT_PID" > "$(pidfile qwen)"
        write_manifest qwen "$QWEN_REPO" "$QWEN_PORT" "$(resolve_hf_revision "$QWEN_REPO")" false "" "default"
        log "qwen up: $QWEN_REPO, full context, no downgrade."
        exit 0
    fi
    if ! looks_like_oom "$LOG_DIR/qwen_attempt1.log"; then
        log "FATAL: attempt 1 failed for a non-OOM reason -- not retrying blindly."
        log "See $LOG_DIR/qwen_attempt1.log"
        exit 1
    fi
    log "attempt 1 looks like an OOM. === attempt 2/3: reduced max-model-len=65536 ==="

    if try_launch "$QWEN_REPO" "$QWEN_PORT" "$LOG_DIR/qwen_attempt2.log" --max-model-len 65536 --enforce-eager; then
        echo "$ATTEMPT_PID" > "$(pidfile qwen)"
        write_manifest qwen "$QWEN_REPO" "$QWEN_PORT" "$(resolve_hf_revision "$QWEN_REPO")" true \
            "OOM at full context on 1x${GPU_NAME}; reduced max-model-len to 65536" "65536"
        log "REPORTED: qwen downgraded to reduced context (65536), same checkpoint. See manifest."
        exit 0
    fi
    if ! looks_like_oom "$LOG_DIR/qwen_attempt2.log"; then
        log "FATAL: attempt 2 failed for a non-OOM reason -- not retrying blindly."
        log "See $LOG_DIR/qwen_attempt2.log"
        exit 1
    fi
    log "attempt 2 still OOM. === attempt 3/3: fallback to ${QWEN_FALLBACK_REPO} ==="

    if try_launch "$QWEN_FALLBACK_REPO" "$QWEN_PORT" "$LOG_DIR/qwen_attempt3.log" --enforce-eager; then
        echo "$ATTEMPT_PID" > "$(pidfile qwen)"
        write_manifest qwen "$QWEN_FALLBACK_REPO" "$QWEN_PORT" "$(resolve_hf_revision "$QWEN_FALLBACK_REPO")" true \
            "Qwen3-Coder-Next-FP8 OOM'd even at max-model-len=65536 on 1x${GPU_NAME}; downgraded model to Qwen3-Coder-30B-A3B-Instruct" ""
        log "REPORTED: qwen DOWNGRADED to ${QWEN_FALLBACK_REPO}. This changes the experiment's model"
        log "identity -- CLAUDE.md must be updated with this fact before any real run uses it."
        exit 0
    fi
    log "FATAL: all 3 attempts failed. See $LOG_DIR/qwen_attempt{1,2,3}.log"
    exit 1
fi

log "done. '$0 --status' to check, '$0 --stop' to shut down."

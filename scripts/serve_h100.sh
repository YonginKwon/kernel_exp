#!/usr/bin/env bash
# ============================================================================
# scripts/serve_h100.sh -- vLLM serving for kernel-lang-2x2's two open-weight
# generation models. Run this ON THE H100x2 DEPARTMENT SERVER, not the
# evaluation machine. Self-contained: creates its own Python env, installs
# vLLM, checks GPU availability, launches the server(s), health-checks them,
# and writes a manifest of exactly what got served (HF revision, vLLM
# version, dtype/quantization) for CLAUDE.md's "실행 후 기록 예정" fields.
# No sudo anywhere -- conda (if present) or `python3 -m venv` otherwise, both
# user-space.
#
# ----------------------------------------------------------------------------
# IMPORTANT: this deviates from the original "1 GPU per model" plan.
# ----------------------------------------------------------------------------
# Qwen3-Coder-Next-FP8's own model card and vLLM's official recipe
# (https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-Next.html)
# require tensor-parallel-size >= 2 -- every documented deployment example
# uses TP=2 or TP=4; no single-GPU (TP=1) deployment is documented, because
# of the model's hybrid Gated-Attention/Gated-DeltaNet architecture (custom
# kernels), not just raw VRAM. gpt-oss-120b, by contrast, is designed to fit
# one 80GB GPU (MXFP4 quantized out of the box; official recipe is plain
# `vllm serve openai/gpt-oss-120b` on a single GPU).
#
# With exactly 2 H100s total, Qwen alone needs BOTH -- there is no way to run
# both models concurrently with one GPU each. This script serves them
# SEQUENTIALLY by default: Qwen first (both GPUs), stop it, then gpt-oss
# (one GPU). Total GPU-time is unaffected; wall-clock for running both is
# just additive instead of overlapped, which is a non-issue for a pilot and
# fine for the full run too. `--concurrent-risky-tp1` instead forces Qwen
# onto a single GPU (tensor-parallel-size 1) so both can run at once --
# UNSUPPORTED by Qwen's own docs, may OOM or fail on unimplemented kernels
# for the hybrid architecture at TP=1. Only pass it if you've separately
# confirmed TP=1 actually works on this box; the default is the safe path.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/vllm"
ENV_DIR="$REPO_ROOT/.venv-vllm"
mkdir -p "$LOG_DIR"

QWEN_REPO="Qwen/Qwen3-Coder-Next-FP8"
QWEN_PORT=8000
GPTOSS_REPO="openai/gpt-oss-120b"
GPTOSS_PORT=8001

MODE="sequential"          # sequential | concurrent-risky-tp1
TARGET="both"               # both | qwen | gptoss
ACTION="serve"              # serve | stop | status

for arg in "$@"; do
    case "$arg" in
        --concurrent-risky-tp1) MODE="concurrent-risky-tp1" ;;
        --sequential) MODE="sequential" ;;
        --model=qwen) TARGET="qwen" ;;
        --model=gptoss) TARGET="gptoss" ;;
        --model=both) TARGET="both" ;;
        --stop) ACTION="stop" ;;
        --status) ACTION="status" ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            echo
            echo "Usage: $0 [--model=both|qwen|gptoss] [--sequential|--concurrent-risky-tp1] [--stop|--status]"
            exit 0
            ;;
        *) echo "[serve_h100] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

log() { echo "[serve_h100] $*"; }

# ----------------------------------------------------------------------------
# --status / --stop
# ----------------------------------------------------------------------------
pidfile() { echo "$LOG_DIR/$1.pid"; }

if [ "$ACTION" = "status" ]; then
    for name in qwen gptoss; do
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
    for name in qwen gptoss; do
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

# ----------------------------------------------------------------------------
# 1. GPU pre-flight: abort if any target GPU is already occupied.
# ----------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found. Is this the H100 server?"
    exit 1
fi

log "GPU pre-flight check:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "$LOG_DIR/preflight_nvidia_smi.txt"

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -lt 2 ]; then
    log "FATAL: expected 2 GPUs (H100x2), found $NUM_GPUS. Refusing to proceed --"
    log "GPU indices/assignments below assume exactly 2."
    exit 1
fi

OCCUPIED=0
while IFS=',' read -r idx used util; do
    used=$(echo "$used" | tr -d ' ')
    util=$(echo "$util" | tr -d ' ')
    if [ "$used" -gt 500 ] || [ "$util" -gt 5 ]; then
        log "GPU $idx appears occupied: ${used}MiB used, ${util}% util"
        OCCUPIED=1
    fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

if [ "$OCCUPIED" -eq 1 ]; then
    log "FATAL: at least one GPU is already in use. Refusing to start (would either"
    log "OOM or silently share a GPU with someone else's job). Processes using GPUs:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG_DIR/preflight_nvidia_smi.txt"
    log "If this is stale (e.g. a zombie process), clear it yourself first -- this"
    log "script will not kill processes it didn't start."
    exit 1
fi
log "GPUs 0,1 idle -- proceeding."

# ----------------------------------------------------------------------------
# 2. Python environment (no sudo): prefer conda if present, else venv.
# ----------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    log "conda found -- using conda env 'kernel2x2-vllm'"
    if ! conda env list | grep -q "^kernel2x2-vllm "; then
        conda create -y -n kernel2x2-vllm python=3.12
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate kernel2x2-vllm
    PY=python
else
    log "conda not found -- using venv at $ENV_DIR"
    if [ ! -d "$ENV_DIR" ]; then
        python3 -m venv "$ENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    PY=python
fi

log "installing/upgrading vllm (>=0.15.0, required for Qwen3-Coder-Next-FP8) + huggingface_hub"
pip install -q --upgrade pip
pip install -q --upgrade "vllm>=0.15.0" huggingface_hub

VLLM_VERSION="$($PY -c 'import vllm; print(vllm.__version__)')"
log "vllm version: $VLLM_VERSION"

# ----------------------------------------------------------------------------
# 3. Helpers: launch one vllm serve, wait for health, resolve HF revision.
# ----------------------------------------------------------------------------
wait_for_health() {
    local port="$1" name="$2" max_wait=1800 waited=0
    log "waiting for $name (port $port) to become healthy (up to ${max_wait}s -- first"
    log "launch downloads the checkpoint from HF, which is large)..."
    while [ "$waited" -lt "$max_wait" ]; do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            log "$name is up."
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    log "FATAL: $name did not become healthy within ${max_wait}s. See $LOG_DIR/${name}.log"
    return 1
}

resolve_hf_revision() {
    # HF cache layout: $HF_HOME/hub/models--ORG--NAME/snapshots/<sha>/ -- the
    # snapshot dirname IS the resolved commit hash after download.
    local repo="$1"
    local cache_name="models--$(echo "$repo" | sed 's|/|--|')"
    local hub_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/$cache_name/snapshots"
    if [ -d "$hub_dir" ]; then
        ls "$hub_dir" | head -1
    else
        echo "UNKNOWN (snapshot dir not found: $hub_dir)"
    fi
}

write_manifest() {
    local name="$1" repo="$2" port="$3" tp="$4" gpus="$5"
    local revision; revision="$(resolve_hf_revision "$repo")"
    cat > "$LOG_DIR/${name}_manifest.json" <<JSON
{
  "name": "$name",
  "hf_repo": "$repo",
  "hf_revision": "$revision",
  "vllm_version": "$VLLM_VERSION",
  "port": $port,
  "tensor_parallel_size": $tp,
  "cuda_visible_devices": "$gpus",
  "base_url": "http://localhost:${port}/v1",
  "started_at": "$(date -Iseconds)"
}
JSON
    log "wrote $LOG_DIR/${name}_manifest.json -- copy its contents into CLAUDE.md's"
    log "'실행 후 기록 예정' placeholders."
}

# ----------------------------------------------------------------------------
# 4. Launch.
# ----------------------------------------------------------------------------
launch_qwen() {
    local tp gpus
    if [ "$MODE" = "concurrent-risky-tp1" ]; then
        tp=1; gpus="0"
        log "WARNING: launching Qwen3-Coder-Next-FP8 with --tensor-parallel-size 1."
        log "This is NOT what Qwen's own docs recommend (TP>=2) -- it may OOM or hit"
        log "an unimplemented-kernel error for the hybrid Gated-DeltaNet architecture."
    else
        tp=2; gpus="0,1"
    fi
    log "launching Qwen3-Coder-Next-FP8 (TP=$tp, GPUs $gpus, port $QWEN_PORT)"
    CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$QWEN_REPO" \
        --port "$QWEN_PORT" \
        --tensor-parallel-size "$tp" \
        --trust-remote-code \
        > "$LOG_DIR/qwen.log" 2>&1 &
    echo $! > "$(pidfile qwen)"
    wait_for_health "$QWEN_PORT" qwen
    write_manifest qwen "$QWEN_REPO" "$QWEN_PORT" "$tp" "$gpus"
}

launch_gptoss() {
    local gpus="1"
    if [ "$MODE" = "sequential" ] || [ "$TARGET" != "both" ]; then
        gpus="0"   # Qwen already stopped (sequential) or not launched at all
    fi
    log "launching gpt-oss-120b (TP=1, GPU $gpus, port $GPTOSS_PORT)"
    CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$GPTOSS_REPO" \
        --port "$GPTOSS_PORT" \
        --tensor-parallel-size 1 \
        > "$LOG_DIR/gptoss.log" 2>&1 &
    echo $! > "$(pidfile gptoss)"
    wait_for_health "$GPTOSS_PORT" gptoss
    write_manifest gptoss "$GPTOSS_REPO" "$GPTOSS_PORT" 1 "$gpus"
}

if [ "$MODE" = "concurrent-risky-tp1" ] && [ "$TARGET" = "both" ]; then
    launch_qwen
    launch_gptoss
elif [ "$TARGET" = "qwen" ]; then
    launch_qwen
elif [ "$TARGET" = "gptoss" ]; then
    launch_gptoss
else
    # sequential + both: Qwen needs both GPUs, so it must run and finish
    # before gpt-oss starts. This call does NOT stop Qwen automatically --
    # run generate.py against Qwen's endpoint now, then re-run this script
    # with --stop, then --model=gptoss for the second model.
    log "MODE=sequential, TARGET=both: launching Qwen only (uses both GPUs)."
    log "Run scripts/generate.py against http://localhost:${QWEN_PORT}/v1 now."
    log "When done: '$0 --stop', then '$0 --model=gptoss' for gpt-oss-120b."
    launch_qwen
fi

log "done. '$0 --status' to check, '$0 --stop' to shut down."

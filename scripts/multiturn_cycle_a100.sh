#!/usr/bin/env bash
# ============================================================================
# scripts/multiturn_cycle_a100.sh -- ONE turn-cycle of the #3.4 multiturn
# orchestrator (scripts/multiturn.py) on this A100 server, per
# ~/kernel-lang-2x2/CLAUDE.md Phase 2's "턴 사이클 실행" section:
#   1. serve gptoss -> generate -> stop
#   2. serve qwen -> generate -> stop
#   3. evaluate (GPU exclusive) -> report
#
# Follower-mode note: this is operational plumbing (env-var serving recipe +
# command sequencing), not a protocol/harness change -- multiturn.py itself
# (the actual #3.4 orchestrator logic: feedback templates, termination
# conditions, chain structure) is invoked completely unmodified. Mirrors
# serve_local.sh's own category (per-server operational script, not the
# experiment protocol).
#
# Usage:
#   scripts/multiturn_cycle_a100.sh
#
# Exits non-zero and leaves logs in logs/multiturn/ on any stage failure --
# does NOT retry automatically (CLAUDE.md Phase 2: "이상 발생 시 진행을
# 멈추고 보고만 한다").
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE="$REPO_ROOT/results/eval/multiturn_state_a100.json"
LOG_DIR="$REPO_ROOT/logs/multiturn"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"

log() { echo "[cycle $TS] $*"; }

source "$REPO_ROOT/.venv/bin/activate"
export CUDA_HOME="$REPO_ROOT/third_party/cuda-a100-toolchain"
export PATH="$REPO_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.0"

# --- stage 1: gptoss ---------------------------------------------------
log "starting gptoss serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_GPTOSS_GPU_MEM_UTIL="0.92" \
bash scripts/serve_local.sh gptoss > "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
log "gptoss healthy, generating"
python scripts/multiturn.py generate --state "$STATE" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_gptoss.log" 2>&1
log "gptoss generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
sleep 5

# --- stage 2: qwen -------------------------------------------------------
log "starting qwen serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_QWEN_MAX_MODEL_LEN="32768" \
bash scripts/serve_local.sh qwen > "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
log "qwen healthy, generating"
python scripts/multiturn.py generate --state "$STATE" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_qwen.log" 2>&1
log "qwen generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
sleep 5

# --- stage 3: evaluate (GPU exclusive) + report -------------------------
log "GPU exclusivity check + evaluate"
python scripts/multiturn.py evaluate --state "$STATE" \
    > "$LOG_DIR/${TS}_evaluate.log" 2>&1
log "evaluate done, report:"
python scripts/multiturn.py report --state "$STATE" | tee "$LOG_DIR/${TS}_report.log"
log "cycle complete"

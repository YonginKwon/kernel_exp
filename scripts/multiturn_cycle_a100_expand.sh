#!/usr/bin/env bash
# ============================================================================
# scripts/multiturn_cycle_a100_expand.sh -- ONE turn-cycle of the #3.4
# multiturn orchestrator for the Phase 2 SCOPE EXPANSION (TileLang track +
# docinject ablation track, PI instruction 2026-08-25), covering BOTH new
# state files in a single serve/generate/evaluate pass:
#   1. serve gptoss -> generate (tilelang state, docinject state) -> stop
#   2. serve qwen -> generate (tilelang state, docinject state) -> stop
#   3. evaluate both states (GPU exclusive) -> report both
#
# Mirrors scripts/multiturn_cycle_a100.sh's own operational shape exactly
# (same env exports, same serve/generate/evaluate/report sequencing) --
# multiturn.py itself is invoked completely unmodified, only against two
# NEW state files instead of the original multiturn_state_a100.json (which
# this script never touches).
#
# Usage: bash scripts/multiturn_cycle_a100_expand.sh
# Exits non-zero and leaves logs in logs/phase2_expand/ on any stage
# failure -- does NOT retry automatically.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE_TILELANG="$REPO_ROOT/results/eval/multiturn_state_a100_tilelang.json"
STATE_DOCINJECT="$REPO_ROOT/results/eval/multiturn_state_a100_docinject.json"
LOG_DIR="$REPO_ROOT/logs/phase2_expand"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"

log() { echo "[expand-cycle $TS] $*"; }

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
log "gptoss healthy, generating (tilelang then docinject)"
python scripts/multiturn.py generate --state "$STATE_TILELANG" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_tilelang_gptoss.log" 2>&1
python scripts/multiturn.py generate --state "$STATE_DOCINJECT" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_docinject_gptoss.log" 2>&1
log "gptoss generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
sleep 5

# --- stage 2: qwen -------------------------------------------------------
log "starting qwen serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_QWEN_MAX_MODEL_LEN="32768" \
bash scripts/serve_local.sh qwen > "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
log "qwen healthy, generating (tilelang then docinject)"
python scripts/multiturn.py generate --state "$STATE_TILELANG" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_tilelang_qwen.log" 2>&1
python scripts/multiturn.py generate --state "$STATE_DOCINJECT" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency 16 --confirm-run \
    > "$LOG_DIR/${TS}_gen_docinject_qwen.log" 2>&1
log "qwen generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
sleep 5

# --- stage 3: evaluate (GPU exclusive) + report, both states -------------
log "GPU exclusivity check + evaluate tilelang"
python scripts/multiturn.py evaluate --state "$STATE_TILELANG" \
    > "$LOG_DIR/${TS}_evaluate_tilelang.log" 2>&1
log "evaluate docinject"
python scripts/multiturn.py evaluate --state "$STATE_DOCINJECT" \
    > "$LOG_DIR/${TS}_evaluate_docinject.log" 2>&1
log "report tilelang:"
python scripts/multiturn.py report --state "$STATE_TILELANG" | tee "$LOG_DIR/${TS}_report_tilelang.log"
log "report docinject:"
python scripts/multiturn.py report --state "$STATE_DOCINJECT" | tee "$LOG_DIR/${TS}_report_docinject.log"
log "cycle complete"

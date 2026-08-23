#!/usr/bin/env bash
# ============================================================================
# scripts/exploratory_cycle.sh -- ONE turn-cycle of the deep-turn-probe
# exploratory experiment (EXPLORATORY_PROTOCOL.md), covering BOTH arms
# (state_armA.json, state_armB.json) in a single serve/generate/evaluate
# pass, mirroring scripts/multiturn_cycle_a100.sh's operational shape
# (serve gptoss -> generate -> stop; serve qwen -> generate -> stop;
# evaluate both state files GPU-exclusive -> report both).
#
# Follower-mode / isolation note: calls scripts/exploratory_multiturn.py
# (thin monkeypatch wrapper, multiturn.py itself unmodified) against
# results/exploratory/state_arm{A,B}.json ONLY -- never touches
# results/eval/multiturn_state_a100.json or the results-a100 branch.
#
# Usage: scripts/exploratory_cycle.sh
# Exits non-zero and leaves logs in logs/exploratory/ on any stage failure --
# no auto-retry (EXPLORATORY_PROTOCOL.md #6: 이상 발생 시 정지·기록·보고).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE_A="$REPO_ROOT/results/exploratory/state_armA.json"
STATE_B="$REPO_ROOT/results/exploratory/state_armB.json"
LOG_DIR="$REPO_ROOT/logs/exploratory"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"

log() { echo "[expl-cycle $TS] $*"; }

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
log "gptoss healthy, generating (arm A then arm B)"
python scripts/exploratory_multiturn.py generate --state "$STATE_A" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency 8 --confirm-run \
    > "$LOG_DIR/${TS}_gen_armA_gptoss.log" 2>&1
python scripts/exploratory_multiturn.py generate --state "$STATE_B" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency 8 --confirm-run \
    > "$LOG_DIR/${TS}_gen_armB_gptoss.log" 2>&1
log "gptoss generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
sleep 5

# --- stage 2: qwen -------------------------------------------------------
log "starting qwen serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_QWEN_MAX_MODEL_LEN="32768" \
bash scripts/serve_local.sh qwen > "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
log "qwen healthy, generating (arm A then arm B)"
python scripts/exploratory_multiturn.py generate --state "$STATE_A" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency 8 --confirm-run \
    > "$LOG_DIR/${TS}_gen_armA_qwen.log" 2>&1
python scripts/exploratory_multiturn.py generate --state "$STATE_B" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency 8 --confirm-run \
    > "$LOG_DIR/${TS}_gen_armB_qwen.log" 2>&1
log "qwen generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
sleep 5

# --- stage 3: evaluate (GPU exclusive) + report, both arms ---------------
log "GPU exclusivity check + evaluate arm A"
python scripts/exploratory_multiturn.py evaluate --state "$STATE_A" \
    > "$LOG_DIR/${TS}_evaluate_armA.log" 2>&1
log "evaluate arm B"
python scripts/exploratory_multiturn.py evaluate --state "$STATE_B" \
    > "$LOG_DIR/${TS}_evaluate_armB.log" 2>&1
log "report arm A:"
python scripts/exploratory_multiturn.py report --state "$STATE_A" | tee "$LOG_DIR/${TS}_report_armA.log"
log "report arm B:"
python scripts/exploratory_multiturn.py report --state "$STATE_B" | tee "$LOG_DIR/${TS}_report_armB.log"
log "cycle complete"

#!/usr/bin/env bash
# ============================================================================
# scripts/phase2_expand_generate_turn1.sh -- turn-1 generation for the Phase 2
# SCOPE EXPANSION (PI instruction, 2026-08-25: "A100에서도 PRO 6000에서 했던
# 실험을 모두 수행하자"): brings this A100 server's coverage up to parity
# with PRO 6000's full 2x2 design, which the original A100 Phase 1 scope
# note explicitly excluded ("언어 3개: CUDA, PTX, Triton. TileLang 제외").
#
# Two additions, both entirely NEW turn-1 data (never generated on this
# server before):
#   (a) TileLang, condition=0shot, all 37 tasks (37x5=185 calls/model) --
#       matches how the original 3-language 0shot turn-1 data was generated
#       (all 37 tasks; the 32-clean-task filter is applied later, at
#       multiturn-init time, exactly like the existing 3 languages).
#   (b) docinject ablation, all 4 languages (cuda/ptx/triton/tilelang),
#       condition=docinject, the PI-approved 20-task subset
#       (tasks/level1_subset.json's doc_ablation_subset_of_20, 20x5=100
#       calls/model/language) -- never run on this server for ANY language
#       before. Filtered to the 17 clean tasks at multiturn-init time,
#       mirroring multiturn.py's own docinject_clean_tasks() treatment.
#
# Follower-mode / non-invasive-reuse: calls scripts/generate.py (unmodified)
# once per (language, condition) pair. No protocol/harness file touched.
#
# Usage: bash scripts/phase2_expand_generate_turn1.sh
# Exits non-zero on any single generate.py call's failure -- no auto-retry.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/phase2_expand"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"

log() { echo "[phase2-expand-gen $TS] $*" | tee -a "$LOG_DIR/${TS}_driver.log"; }

source "$REPO_ROOT/.venv/bin/activate"
export CUDA_HOME="$REPO_ROOT/third_party/cuda-a100-toolchain"
export PATH="$REPO_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.0"

# fixed, documented base seeds (fresh for this batch -- no reuse of turn-1
# seeds from the original 3-language 0shot run, which this doesn't touch)
SEED_TILELANG_0SHOT_GPTOSS=42000
SEED_TILELANG_0SHOT_QWEN=43000
SEED_DOCINJECT_GPTOSS_BASE=44000   # + 100*lang_index
SEED_DOCINJECT_QWEN_BASE=45000

gen() {
    local lang="$1" cond="$2" model="$3" url="$4" manifest="$5" seed="$6" tag="$7"
    log "generate: language=$lang condition=$cond model=$model seed=$seed"
    python scripts/generate.py --language "$lang" --condition "$cond" \
        --base-url "$url" --model "$model" --manifest "$manifest" \
        --seed "$seed" --concurrency 16 --confirm-run \
        > "$LOG_DIR/${TS}_gen_${tag}.log" 2>&1
}

# --- stage 1: gptoss ---------------------------------------------------
log "starting gptoss serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_GPTOSS_GPU_MEM_UTIL="0.92" \
bash scripts/serve_local.sh gptoss > "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
log "gptoss healthy, generating (tilelang 0shot + docinject x4 langs)"

gen tilelang 0shot openai/gpt-oss-120b http://localhost:8000/v1 \
    logs/vllm/gptoss_manifest.json "$SEED_TILELANG_0SHOT_GPTOSS" tilelang_0shot_gptoss

i=0
for lang in cuda ptx triton tilelang; do
    gen "$lang" docinject openai/gpt-oss-120b http://localhost:8000/v1 \
        logs/vllm/gptoss_manifest.json "$((SEED_DOCINJECT_GPTOSS_BASE + i * 100))" "docinject_${lang}_gptoss"
    i=$((i + 1))
done

log "gptoss generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_gptoss.log" 2>&1
sleep 5

# --- stage 2: qwen -------------------------------------------------------
log "starting qwen serve"
KERNEL2X2_VLLM_VERSION_SPEC="vllm==0.10.1" \
KERNEL2X2_CUDA_HOME_OVERRIDE="$CUDA_HOME" \
KERNEL2X2_QWEN_MAX_MODEL_LEN="32768" \
bash scripts/serve_local.sh qwen > "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
log "qwen healthy, generating (tilelang 0shot + docinject x4 langs)"

gen tilelang 0shot Qwen/Qwen3-Coder-30B-A3B-Instruct http://localhost:8001/v1 \
    logs/vllm/qwen_manifest.json "$SEED_TILELANG_0SHOT_QWEN" tilelang_0shot_qwen

i=0
for lang in cuda ptx triton tilelang; do
    gen "$lang" docinject Qwen/Qwen3-Coder-30B-A3B-Instruct http://localhost:8001/v1 \
        logs/vllm/qwen_manifest.json "$((SEED_DOCINJECT_QWEN_BASE + i * 100))" "docinject_${lang}_qwen"
    i=$((i + 1))
done

log "qwen generate done, stopping serve"
bash scripts/serve_local.sh --stop >> "$LOG_DIR/${TS}_serve_qwen.log" 2>&1
sleep 5

log "=== turn-1 generation complete for all new batches ==="

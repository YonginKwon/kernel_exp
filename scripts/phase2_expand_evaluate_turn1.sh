#!/usr/bin/env bash
# ============================================================================
# scripts/phase2_expand_evaluate_turn1.sh -- turn-1 compile+correctness eval
# for the Phase 2 scope-expansion batches (see
# phase2_expand_generate_turn1.sh's header for what/why). Run AFTER that
# script and AFTER both vLLM servers are confirmed stopped (GPU exclusivity,
# evaluate.py's own assert_gpu_exclusive() enforces this per call anyway).
#
# Writes 5 scoped eval files (one per language+condition pair) rather than
# one combined file, so results/eval/multiturn_init_a100_{tilelang,
# docinject}.py can each read exactly the sources they need -- mirrors
# multiturn.py's own cmd_init pattern of reading distinct 0shot/docinject
# source files.
#
# Usage: bash scripts/phase2_expand_evaluate_turn1.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/phase2_expand"
mkdir -p "$LOG_DIR"
EVAL_DIR="$REPO_ROOT/results/eval"
TS="$(date +%Y%m%dT%H%M%S)"

log() { echo "[phase2-expand-eval $TS] $*" | tee -a "$LOG_DIR/${TS}_driver.log"; }

source "$REPO_ROOT/.venv/bin/activate"
export CUDA_HOME="$REPO_ROOT/third_party/cuda-a100-toolchain"
export PATH="$REPO_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.0"

log "evaluate: tilelang 0shot"
python scripts/evaluate.py --language tilelang --condition 0shot \
    --out "$EVAL_DIR/eval_a100_tilelang_0shot.json" \
    > "$LOG_DIR/${TS}_eval_tilelang_0shot.log" 2>&1

for lang in cuda ptx triton tilelang; do
    log "evaluate: $lang docinject"
    python scripts/evaluate.py --language "$lang" --condition docinject \
        --out "$EVAL_DIR/eval_a100_docinject_${lang}.json" \
        > "$LOG_DIR/${TS}_eval_docinject_${lang}.log" 2>&1
done

log "=== turn-1 evaluation complete for all new batches ==="

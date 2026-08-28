#!/usr/bin/env bash
# ============================================================================
# scripts/multiturn_run_a100_expand.sh -- drives
# scripts/multiturn_cycle_a100_expand.sh turn after turn (k=2..10) until
# BOTH the tilelang and docinject state files report every chain terminated
# (k_max_reached or no_improvement_3_turns -- standard #3.4 termination,
# UNCHANGED from the main protocol; this is not an exploratory deviation)
# or k_max=10 is hit. PI instruction 2026-08-25: bring this A100 server's
# coverage up to parity with PRO 6000's full experiment.
#
# Every cycle, commits + pushes results/eval/{multiturn_state_a100_tilelang,
# multiturn_state_a100_docinject}.json + results/raw + logs to a NEW,
# isolated branch, results-a100-phase2-expand (PI decision 2026-08-25:
# explicitly requested a separate branch rather than pushing onto the
# already-finalized results-a100 branch). Stops immediately (does NOT loop
# past the failure) if a cycle exits non-zero.
#
# git-topology note: local `master` and `results-a100` have DIVERGED
# (results-a100's tip bcd6d4f, the final Phase 2 results, is NOT an
# ancestor of master -- master separately merged origin/master's P0-a/b
# harness work and continued with exploratory-track commits). Pushing this
# expansion's NEW commits (built on master, which has all the harness code
# this needs) to a brand-new branch name sidesteps that divergence entirely
# -- no merge, no non-fast-forward risk, results-a100 itself is never
# touched by this script.
#
# Meant to run under `nohup ... &` / a background task -- a full run can
# take multiple hours (up to k=10 x (320 + 680) chains).
#
# Usage: bash scripts/multiturn_run_a100_expand.sh [--max-turns N]
# ============================================================================
set -uo pipefail  # NOT -e: a single cycle failure should stop the LOOP
                   # cleanly (with a message), not kill the script via trap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE_TILELANG="$REPO_ROOT/results/eval/multiturn_state_a100_tilelang.json"
STATE_DOCINJECT="$REPO_ROOT/results/eval/multiturn_state_a100_docinject.json"
LOG_DIR="$REPO_ROOT/logs/phase2_expand"
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/driver.log"

MAX_TURNS=10
if [ "${1:-}" = "--max-turns" ]; then MAX_TURNS="$2"; fi

dlog() { echo "[$(date +%Y%m%dT%H%M%S)] $*" | tee -a "$DRIVER_LOG"; }

all_terminated() {
    source "$REPO_ROOT/.venv/bin/activate"
    python3 - "$STATE_TILELANG" "$STATE_DOCINJECT" <<'PYEOF'
import json, sys
paths = sys.argv[1:]
all_done = True
for p in paths:
    state = json.loads(open(p).read())
    chains = state["chains"].values()
    n = len(chains)
    n_term = sum(1 for c in chains if c["terminated"])
    if n_term < n:
        all_done = False
    print(f"{p}: {n_term}/{n} terminated")
sys.exit(0 if all_done else 1)
PYEOF
}

TARGET_BRANCH="results-a100-phase2-expand"

push_checkpoint() {
    local turn_label="$1"
    dlog "pushing checkpoint (turn ~$turn_label) to $TARGET_BRANCH"
    git add -f results/eval/multiturn_state_a100_tilelang.json \
        results/eval/multiturn_state_a100_docinject.json \
        results/eval/eval_a100_tilelang_0shot.json \
        results/eval/eval_a100_docinject_*.json \
        results/raw/tilelang results/raw/cuda/docinject results/raw/ptx/docinject \
        results/raw/triton/docinject \
        scripts/phase2_expand_*.sh \
        scripts/multiturn_init_a100_tilelang.py scripts/multiturn_init_a100_docinject.py \
        scripts/multiturn_cycle_a100_expand.sh scripts/multiturn_run_a100_expand.sh \
        2>&1 | tee -a "$DRIVER_LOG"
    if git diff --cached --quiet; then
        dlog "nothing new to commit at this checkpoint"
    else
        git commit -m "phase2-expand: checkpoint at turn ~$turn_label (tilelang + docinject tracks)" 2>&1 | tee -a "$DRIVER_LOG"
    fi
    git push origin HEAD:refs/heads/"$TARGET_BRANCH" 2>&1 | tee -a "$DRIVER_LOG"
}

dlog "=== multiturn_run_a100_expand.sh starting, max_turns=$MAX_TURNS ==="

turn_count=0
while [ "$turn_count" -lt "$MAX_TURNS" ]; do
    turn_count=$((turn_count + 1))
    dlog "--- driving cycle $turn_count ---"
    if ! bash scripts/multiturn_cycle_a100_expand.sh >> "$DRIVER_LOG" 2>&1; then
        dlog "!!! multiturn_cycle_a100_expand.sh FAILED at driver iteration $turn_count -- stopping loop, NOT retrying (follower-mode: stop/record/report). See $LOG_DIR for the failing stage's log."
        exit 1
    fi

    push_checkpoint "$turn_count"

    if all_terminated | tee -a "$DRIVER_LOG"; then
        dlog "all chains terminated in both tracks -- stopping loop"
        break
    fi
done

dlog "=== multiturn_run_a100_expand.sh finished after $turn_count driver iteration(s) ==="

#!/usr/bin/env bash
# ============================================================================
# scripts/exploratory_run.sh -- drives scripts/exploratory_cycle.sh turn
# after turn (k=2..100) until both Arm A and Arm B state files report every
# chain terminated (k_max_reached, since NO_IMPROVE_LIMIT is disabled --
# EXPLORATORY_PROTOCOL.md #2) or k_max=100 is hit. Every 20 turns, commits +
# pushes results/exploratory/ to the results-a100-exploratory branch
# (EXPLORATORY_PROTOCOL.md #6). Stops immediately (does NOT loop past the
# failure) if a cycle exits non-zero -- follower-mode "정지·기록·보고".
#
# Meant to run under `nohup ... &` / a background task, NOT interactively --
# a full run can take many hours (up to k=100 x 28 chains).
#
# Usage: bash scripts/exploratory_run.sh [--max-turns N]
# ============================================================================
set -uo pipefail  # NOT -e: a single cycle failure should stop the LOOP
                   # cleanly (with a message), not kill the script via trap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE_A="$REPO_ROOT/results/exploratory/state_armA.json"
STATE_B="$REPO_ROOT/results/exploratory/state_armB.json"
LOG_DIR="$REPO_ROOT/logs/exploratory"
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/driver.log"

MAX_TURNS=100
if [ "${1:-}" = "--max-turns" ]; then MAX_TURNS="$2"; fi

dlog() { echo "[$(date +%Y%m%dT%H%M%S)] $*" | tee -a "$DRIVER_LOG"; }

all_terminated() {
    source "$REPO_ROOT/.venv/bin/activate"
    python3 - "$STATE_A" "$STATE_B" <<'PYEOF'
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

push_checkpoint() {
    local turn_label="$1"
    dlog "pushing interim checkpoint (turn ~$turn_label) to results-a100-exploratory"
    git add -A results/exploratory EXPLORATORY_PROTOCOL.md scripts/exploratory_*.py scripts/exploratory_*.sh 2>&1 | tee -a "$DRIVER_LOG"
    if git diff --cached --quiet; then
        dlog "nothing new to commit at this checkpoint"
    else
        git commit -m "deep-turn-probe: interim checkpoint at turn ~$turn_label" 2>&1 | tee -a "$DRIVER_LOG"
    fi
    if git rev-parse --verify results-a100-exploratory >/dev/null 2>&1; then
        git push origin HEAD:results-a100-exploratory 2>&1 | tee -a "$DRIVER_LOG"
    else
        git push origin HEAD:refs/heads/results-a100-exploratory 2>&1 | tee -a "$DRIVER_LOG"
    fi
}

dlog "=== exploratory_run.sh starting, max_turns=$MAX_TURNS ==="

turn_count=0
while [ "$turn_count" -lt "$MAX_TURNS" ]; do
    turn_count=$((turn_count + 1))
    dlog "--- driving cycle $turn_count ---"
    if ! bash scripts/exploratory_cycle.sh >> "$DRIVER_LOG" 2>&1; then
        dlog "!!! exploratory_cycle.sh FAILED at driver iteration $turn_count -- stopping loop, NOT retrying (follower-mode: stop/record/report). See $LOG_DIR for the failing stage's log."
        exit 1
    fi

    if [ $((turn_count % 20)) -eq 0 ]; then
        push_checkpoint "$turn_count"
    fi

    if all_terminated | tee -a "$DRIVER_LOG"; then
        dlog "all chains terminated in both arms -- stopping loop"
        push_checkpoint "final-$turn_count"
        break
    fi
done

dlog "=== exploratory_run.sh finished after $turn_count driver iteration(s) ==="

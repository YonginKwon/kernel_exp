#!/usr/bin/env bash
set -uo pipefail
cd /home/crojjang/kernel-lang-2x2
source scripts/env.sh
source .venv/bin/activate

STATE=results/eval/multiturn_state.json
CONCURRENCY=16
CUTOFF_EPOCH=$(date -d "2026-08-25 06:00 KST" +%s)
REPORT_DIR=logs/multiturn/turn_reports
MAX_SERVE_ATTEMPTS=5

log(){ echo "[multiturn-loop $(date -Iseconds)] $*"; }

wait_gpu_clear(){
  for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$USED" -le 500 ]; then log "GPU clear ($USED MiB)"; return 0; fi
    log "waiting for GPU to clear ($USED MiB)..."; sleep 10
  done
  log "FATAL: GPU never cleared"; return 1
}

# Retries serve_local.sh up to MAX_SERVE_ATTEMPTS times (this GPU/vLLM stack
# has a documented flaky FlashInfer MoE-autotuner segfault on qwen, ~2026-08-20
# session: failed twice, succeeded on 3rd/4th plain retry with no config change).
serve_with_retry(){
  local target="$1"
  for attempt in $(seq 1 "$MAX_SERVE_ATTEMPTS"); do
    log "serve $target attempt $attempt/$MAX_SERVE_ATTEMPTS"
    if scripts/serve_local.sh "$target"; then
      return 0
    fi
    log "serve $target attempt $attempt failed -- cleaning up before retry"
    scripts/serve_local.sh --stop || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
    wait_gpu_clear || true
  done
  log "FATAL: serve $target failed after $MAX_SERVE_ATTEMPTS attempts"
  return 1
}

any_active(){
  python3 -c "
import json
d = json.load(open('$STATE'))
print(sum(1 for c in d['chains'].values() if not c['terminated']))
"
}

turn_number(){
  python3 -c "
import json, collections
d = json.load(open('$STATE'))
turns = [c['turn'] for c in d['chains'].values() if not c['terminated']]
print(max(turns) if turns else 0)
"
}

LOOP_EXIT_REASON="fatal"  # overwritten to "complete" only on the two clean-exit paths below
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$CUTOFF_EPOCH" ]; then
    log "=== CUTOFF REACHED (2026-08-25 06:00 KST) -- stopping loop, no partial turn started ==="
    LOOP_EXIT_REASON="complete"
    break
  fi
  ACTIVE=$(any_active)
  if [ "$ACTIVE" -eq 0 ]; then
    log "=== all chains terminated -- multi-turn protocol complete ==="
    LOOP_EXIT_REASON="complete"
    break
  fi
  NEXT_TURN=$(( $(turn_number) + 1 ))
  log "=== starting turn advance to turn $NEXT_TURN ($ACTIVE active chain(s)) ==="

  if ! serve_with_retry gptoss; then break; fi
  python scripts/multiturn.py generate --state "$STATE" \
    --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \
    --manifest logs/vllm/gptoss_manifest.json --concurrency "$CONCURRENCY" --confirm-run
  GEN_GPTOSS_RC=$?
  scripts/serve_local.sh --stop; sleep 5; wait_gpu_clear || true
  if [ "$GEN_GPTOSS_RC" -ne 0 ]; then log "FATAL: gpt-oss generate failed (rc=$GEN_GPTOSS_RC)"; break; fi

  if ! serve_with_retry qwen; then break; fi
  python scripts/multiturn.py generate --state "$STATE" \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --base-url http://localhost:8001/v1 \
    --manifest logs/vllm/qwen_manifest.json --concurrency "$CONCURRENCY" --confirm-run
  GEN_QWEN_RC=$?
  scripts/serve_local.sh --stop; sleep 5; wait_gpu_clear || true
  if [ "$GEN_QWEN_RC" -ne 0 ]; then log "FATAL: qwen generate failed (rc=$GEN_QWEN_RC)"; break; fi

  # nvcc workers 16->8 2026-08-21: power-peak mitigation after the 14:25
  # power-loss crash (GPU power cap rejected by PI -- 600W stays).
  python scripts/multiturn.py evaluate --state "$STATE" --precompile-workers 8
  EVAL_RC=$?
  if [ "$EVAL_RC" -ne 0 ]; then log "FATAL: evaluate failed (rc=$EVAL_RC)"; break; fi

  REPORT_FILE="$REPORT_DIR/turn_${NEXT_TURN}_report.txt"
  python scripts/multiturn.py report --state "$STATE" | tee "$REPORT_FILE"
  log "=== turn $NEXT_TURN complete, report written to $REPORT_FILE ==="

  # One-time final_retiming.py smoke test (PI instruction 2026-08-21): a
  # cheap 3-kernel dry run of the checkpoint/resume + power-drift-check
  # paths, in the GPU-idle gap between this turn's evaluate finishing and
  # the next turn's vLLM serve starting -- so the FIRST time this script
  # ever runs for real isn't turn 10 unattended. Marker file makes it
  # fire exactly once across the whole multi-turn run, not every turn.
  SMOKE_MARKER=logs/multiturn/.final_retiming_smoke_done
  if [ ! -f "$SMOKE_MARKER" ]; then
    log "=== final_retiming.py smoke test (3 chains, one-time) ==="
    wait_gpu_clear || true
    SMOKE_OUT="results/eval/final_retiming_smoketest.json"
    python scripts/final_retiming.py --state "$STATE" --out "$SMOKE_OUT" --limit 3 \
      --power-note "SMOKE TEST -- not real results" \
      2>&1 | tee "logs/multiturn/final_retiming_smoketest_$(date +%Y%m%dT%H%M%S).log"
    SMOKE_RC=$?
    if [ "$SMOKE_RC" -eq 0 ]; then
      date -Iseconds > "$SMOKE_MARKER"
      log "=== final_retiming.py smoke test OK -> $SMOKE_OUT (marker written, won't repeat) ==="
    else
      log "=== final_retiming.py smoke test FAILED (rc=$SMOKE_RC) -- no marker written, will retry next turn boundary ==="
    fi
    wait_gpu_clear || true
  fi
done

log "=== loop exited (reason: $LOOP_EXIT_REASON) ==="

# Final re-timing pass (CLAUDE.md item 4, PI instruction 2026-08-21): only
# on a CLEAN exit (turn 10 completed / all chains terminated, or the
# 2026-08-25 cutoff) -- never after a FATAL break (serve/generate/evaluate
# failure), since that leaves the GPU state and/or multiturn_state.json
# mid-turn and re-timing then would just measure a broken run. A crash
# mid-loop instead re-enters this whole script fresh via start_loop.sh's
# @reboot hook (see its docstring) -- the retiming pass only ever runs once
# the loop has genuinely finished.
# One-time remote backup of multiturn_state.json (PI instruction 2026-08-22,
# post turn-8 report): results/eval/ is normally kept out of git entirely
# (CLAUDE.md: results/ is read-only DATA, not source -- see .gitignore) but
# the crash frequency this run has shown (4 interruptions in turn 8 alone)
# means the re-timing pass itself could die mid-pass with no off-machine
# copy of the state it was timing against. `git add -f` is a one-time,
# explicit exception for this one file -- NOT a change to the .gitignore
# policy, so nothing else under results/ starts getting tracked. Best-effort:
# logged loudly on failure (network/auth), but does not block the re-timing
# pass itself -- the measurement is the priority once the loop has cleanly
# finished.
backup_state_before_retiming(){
  log "=== backing up $STATE to origin before final re-timing pass ==="
  if ! timeout 120 git add -f "$STATE"; then
    log "WARNING: git add -f on $STATE failed -- proceeding WITHOUT remote backup"; return 1
  fi
  if git diff --cached --quiet -- "$STATE"; then
    log "state file unchanged since last backup commit -- nothing to push"; return 0
  fi
  if ! git commit -m "backup: multiturn_state.json snapshot before final re-timing pass ($(date -Iseconds))"; then
    log "WARNING: git commit of state backup failed -- proceeding WITHOUT remote backup"; return 1
  fi
  if ! timeout 120 git push origin master; then
    log "WARNING: git push of state backup failed (commit kept locally) -- proceeding WITHOUT confirmed remote backup"; return 1
  fi
  log "=== state backup pushed to origin/master ==="
}

if [ "$LOOP_EXIT_REASON" = "complete" ]; then
  wait_gpu_clear || true
  backup_state_before_retiming || true
  RETIMING_OUT="results/eval/final_timing_$(date +%Y%m%dT%H%M%S).json"
  POWER_NOTE="600W cap (no artificial throttle, PI decision 2026-08-21), turbo default, GPU-exclusive -- see logs/power_monitor.log for the contemporaneous curve"
  log "=== starting final re-timing pass -> $RETIMING_OUT ==="
  python scripts/final_retiming.py --state "$STATE" --out "$RETIMING_OUT" --power-note "$POWER_NOTE" \
    2>&1 | tee "logs/multiturn/final_retiming_$(date +%Y%m%dT%H%M%S).log"
  RETIME_RC=$?
  if [ "$RETIME_RC" -ne 0 ]; then
    log "FATAL: final re-timing pass failed (rc=$RETIME_RC) -- see the log above"
  else
    log "=== final re-timing pass complete -> $RETIMING_OUT ==="
  fi
else
  log "=== skipping final re-timing pass (exit reason was fatal, not a clean completion) ==="
fi

#!/usr/bin/env bash
# High-resolution power/thermal monitor, 30s sampling.
# Purpose: capture the curve immediately preceding any future crash (the
# 14:25 crash was diagnosed as a power-loss sudden-death, so post-mortem
# needs the last ~2 minutes of GPU power/temp + CPU load before the cut).
# Idempotent via flock, safe for repeated @reboot invocation (same pattern
# as logs/multiturn/start_loop.sh).
set -uo pipefail
cd /home/crojjang/kernel-lang-2x2 || exit 1

LOCK=logs/power_monitor.lock
LOG=logs/power_monitor.log
INTERVAL=30

exec 201>"$LOCK"
if ! flock -n 201; then
  echo "[power_monitor $(date -Iseconds)] lock held by a live process -- not starting a second monitor" >&2
  exit 0
fi

mkdir -p logs
if [ ! -f "$LOG" ]; then
  echo "timestamp,gpu_power_w,gpu_temp_c,gpu_util_pct,gpu_mem_used_mib,loadavg_1,loadavg_5,loadavg_15,cpu_busy_pct" > "$LOG"
fi

# CPU busy % since last sample, from /proc/stat deltas (no root needed;
# RAPL package-power (energy_uj) is root-only on this box -- see CLAUDE.md/
# report note. Not logged here.)
prev_idle=0
prev_total=0
read_cpu_busy_pct(){
  read -r _ u n s i iw irq sirq st g gn < /proc/stat
  idle=$((i + iw))
  total=$((u + n + s + i + iw + irq + sirq + st + g + gn))
  if [ "$prev_total" -eq 0 ]; then
    pct=""
  else
    d_idle=$((idle - prev_idle))
    d_total=$((total - prev_total))
    if [ "$d_total" -gt 0 ]; then
      pct=$(awk -v di="$d_idle" -v dt="$d_total" 'BEGIN{printf "%.1f", 100*(1-di/dt)}')
    else
      pct=""
    fi
  fi
  prev_idle=$idle
  prev_total=$total
  echo "$pct"
}

echo "[power_monitor $(date -Iseconds)] started, sampling every ${INTERVAL}s -> $LOG" >&2

while true; do
  TS=$(date -Iseconds)
  GPU_LINE=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null | head -1)
  GPU_LINE=${GPU_LINE:-",,,"}
  LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg)
  L1=$(echo "$LOADAVG" | cut -d' ' -f1)
  L5=$(echo "$LOADAVG" | cut -d' ' -f2)
  L15=$(echo "$LOADAVG" | cut -d' ' -f3)
  CPU_PCT=$(read_cpu_busy_pct)
  # normalize GPU_LINE "P, T, U, M" -> "P,T,U,M"
  GPU_CSV=$(echo "$GPU_LINE" | tr -d ' ')
  echo "${TS},${GPU_CSV},${L1},${L5},${L15},${CPU_PCT}" >> "$LOG"
  sleep "$INTERVAL"
done

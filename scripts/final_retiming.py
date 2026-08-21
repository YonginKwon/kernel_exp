#!/usr/bin/env python3
"""Final re-timing pass -- CLAUDE.md item 4 (PI instruction, 2026-08-21,
post-14:25-crash runbook).

Re-measures, in ONE batch under a single fixed power/turbo/GPU-exclusive
condition, every multi-turn chain's best-so-far CORRECT kernel plus a
canonical per-task eager-fp16 PyTorch baseline. This is the ONLY timing
source the paper's speedup / fast_1 figures may cite. Turn-loop timing
(`last_timing` / `best_speedup` inside multiturn_state.json, and the
turn-1-only results/eval/timing_20260820.json) was measured under varying
concurrent-load conditions across the whole multi-turn run (vLLM serving
vs. idle, worker-count changes, etc.) and is demoted to appendix-only
reference -- see CLAUDE.md item 4. Report this file's --power-note (record
the actual turbo/power state at run time) alongside every number so a
reader can tell this pass apart from the turn-loop numbers.

Read-only w.r.t. multiturn_state.json and results/raw/ -- only reads
`best_code` (never `code`/`_pending_code`), never writes state back
(CLAUDE.md rule 1: generated kernel code is read-only data).

Run ONCE, after the multi-turn loop exits (turn 10 completion or the
2026-08-25 06:00 KST cutoff) -- see run_all_turns.sh's tail call.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from analyze import clean_32_tasks, FLAWED_TASKS  # noqa: E402

NO_TURBO_PATH = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")


def _power_state():
    """(no_turbo, gpu_power_limit_w) -- read fresh, no caching. Used both to
    stamp the output and to verify the environment didn't drift mid-pass
    (PI requirement: fixed power/turbo state for the whole pass, not just
    'whatever it happened to be')."""
    no_turbo = NO_TURBO_PATH.read_text().strip() if NO_TURBO_PATH.exists() else "unknown"
    try:
        limit = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception as e:
        limit = f"unknown ({e})"
    return {"cpu_no_turbo": no_turbo, "gpu_power_limit_w": limit}


def _build_record(chain):
    return {
        "task": chain["task"], "task_family": None, "language": chain["language"],
        "condition": chain["condition"], "model": chain["model"],
        "sample_index": chain["sample_index"], "status": "generated",
        "parsed_code": chain["best_code"], "path": chain["chain_id"],
    }


def _write(out_path, payload, partial):
    payload = {**payload, "partial": partial}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(out_path)  # atomic on the same filesystem -- no torn reads


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="multiturn_state.json path")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--power-note", default=None,
                     help="optional free-text override/addendum; the actual turbo/power-limit "
                          "readings are captured automatically at start and end regardless")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                     help="write partial progress to --out every N chains, so a crash mid-pass "
                          "(this box has crashed 4x in one day -- 2026-08-21) loses at most N "
                          "chains of redone work instead of the whole pass")
    args = ap.parse_args()

    ev.assert_gpu_exclusive()

    out_path = Path(args.out)
    start_power = _power_state()
    print(f"[final_retiming] power/turbo state at start: {start_power}")

    state = json.loads(Path(args.state).read_text())
    chains = state["chains"]
    primary_32 = set(clean_32_tasks())

    candidates = [c for c in chains.values() if c.get("best_code")]
    n_primary = sum(1 for c in candidates if c["task"] in primary_32)
    print(f"[final_retiming] {len(candidates)} chain(s) with a best-so-far correct kernel "
          f"({n_primary} on the 32-task primary set, {len(candidates) - n_primary} appendix-only)")
    if not candidates:
        print("[final_retiming] nothing to retime -- refusing to write an empty report")
        return 1

    # Resume support: a prior --out (from a pass this same box's crash
    # interrupted) already has some chain_ids timed -- skip those instead of
    # redoing the whole batch. Only trusted if its power/turbo stamp matches
    # this run's start state, otherwise the two halves wouldn't be
    # comparable and we start clean.
    results = []
    done_ids = set()
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            if prior.get("partial") and prior.get("power_state_start") == start_power:
                results = prior.get("results", [])
                done_ids = {r["chain_id"] for r in results}
                print(f"[final_retiming] resuming from partial {out_path} -- "
                      f"{len(done_ids)} chain(s) already timed this pass")
            elif prior.get("partial"):
                print(f"[final_retiming] found partial {out_path} but power/turbo state differs "
                      f"from now -- discarding it and starting this pass clean")
        except Exception as e:
            print(f"[final_retiming] couldn't read existing {out_path} ({e}) -- starting clean")

    todo = [c for c in candidates if c["chain_id"] not in done_ids]

    def _snapshot(partial, extra=None):
        payload = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "power_note": args.power_note,
            "power_state_start": start_power,
            "power_state_end": None if partial else _power_state(),
            "protocol": ("warmup 25 / measure 100 / median, torch.cuda.synchronize, GPU-exclusive "
                         "(evaluate.py time_one_isolated) -- CLAUDE.md timing protocol"),
            "note": ("Authoritative timing source for the paper. Turn-loop timing in "
                     "multiturn_state.json / results/eval/timing_20260820.json is stimulus-only "
                     "(used for optimize-phase feedback during generation), NOT for reported "
                     "speedup/fast_1 figures -- see CLAUDE.md item 4 / paper Setup."),
            "n_chains_total": len(candidates), "n_chains_retimed": len(results),
            "n_primary_32_chains": n_primary,
            "flawed_tasks_excluded_from_primary": list(FLAWED_TASKS),
            "results": results,
            **(extra or {}),
        }
        _write(out_path, payload, partial=partial)

    for i, c in enumerate(todo, 1):
        rec = _build_record(c)
        timing = ev.time_one_isolated(rec, c["best_code"])
        results.append({
            "chain_id": c["chain_id"], "task": c["task"], "language": c["language"],
            "condition": c["condition"], "model": c["model"], "sample_index": c["sample_index"],
            "is_primary_32": c["task"] in primary_32,
            "turn_loop_best_turn": c.get("best_turn"),
            "turn_loop_best_speedup": c.get("best_speedup"),
            **timing,
        })
        if i % args.checkpoint_every == 0 or i == len(todo):
            print(f"[final_retiming] {len(results)}/{len(candidates)} done")
            _snapshot(partial=True)

    end_power = _power_state()
    if end_power != start_power:
        print(f"[final_retiming] FATAL: power/turbo state drifted during the pass -- "
              f"start={start_power} end={end_power}. Results kept as partial at {out_path}; "
              f"fix the environment and re-run to resume (matching chains will be skipped).")
        return 1

    n_exceptions = sum(1 for r in results if "timing_exception" in r)

    # Canonical per-task baseline: median of THIS pass's freshly-measured
    # baseline_ms across every chain sharing that task. All fp16 eager,
    # seed 42, warmup 25 / measure 100 / median (evaluate.py time_one) --
    # independent of which chain/kernel it rode along with, so pooling
    # across chains for the same task is valid and cuts baseline noise
    # (n=228-ish redundant single-chain baselines -> one canonical value).
    by_task = {}
    for r in results:
        if "timing_exception" in r:
            continue
        by_task.setdefault(r["task"], []).append(r["baseline_ms"]["median"])
    baseline_by_task = {
        t: {"median_baseline_ms": statistics.median(v), "n_chains": len(v),
            "is_primary_32": t in primary_32}
        for t, v in by_task.items()
    }
    missing_primary_baseline = sorted(primary_32 - set(baseline_by_task))
    if missing_primary_baseline:
        print(f"[final_retiming] WARNING: {len(missing_primary_baseline)} primary-32 task(s) have "
              f"NO correct chain to retime, so no baseline either: {missing_primary_baseline}")

    _snapshot(partial=False, extra={
        "n_timing_exceptions": n_exceptions,
        "missing_primary_32_baseline_tasks": missing_primary_baseline,
        "baseline_by_task": baseline_by_task,
    })
    print(f"[final_retiming] wrote {out_path} -- {len(results)} kernel(s) retimed "
          f"({n_exceptions} timing exception(s)), {len(baseline_by_task)} distinct task baseline(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

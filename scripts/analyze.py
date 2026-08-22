#!/usr/bin/env python3
"""Aggregation for kernel-lang-2x2 (CLAUDE.md's directory structure: "집계,
표/그림 생성"). CPU-only -- reads existing results/eval/*.json, no GPU work.

Primary analysis basis (PI, 2026-08-20): 32 of the 37 selected tasks --
scripts/audit_tolerance.py (paper/RESULTS_REPORT_20260820.md #7-2) found 5
tasks where fp16 atol=rtol=1e-2 makes the correctness check unreliable
(reference output far below tolerance, or the reference itself overflows to
inf under fp16). Every table below is computed BOTH ways: CLEAN (32 tasks,
the paper's primary numbers) and ALL (37 tasks, kept as an appendix) --
never silently drop the 37-task view, always print both.

Usage:
    source scripts/env.sh && source .venv/bin/activate
    python scripts/analyze.py [--out results/eval/analysis_<timestamp>.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "results" / "eval"
TASKS_PATH = REPO_ROOT / "tasks" / "level1_subset.json"
FIGURES_DIR_DEFAULT = REPO_ROOT / "paper" / "figures_data"

# The 5 tasks flagged by scripts/audit_tolerance.py, 2026-08-20 (see
# paper/RESULTS_REPORT_20260820.md #7-2 and prompts/PROMPT_SPEC.md #3.4 for
# the full rationale per task). TOO-TOLERANT: atol >> typical output
# magnitude. REFERENCE-OVERFLOW: the reference model itself is inf under
# fp16. Both make correctness verdicts for that task untrustworthy.
FLAWED_TASKS = {
    "23_Softmax": "too-tolerant (atol/median ratio 4090x)",
    "90_cumprod": "too-tolerant (median output underflows to 0)",
    "4_Matrix_vector_multiplication_": "reference-overflow (all 2048 output elements inf)",
    "6_Matmul_with_large_K_dimension_": "reference-overflow (all 65536 output elements inf)",
    "95_CrossEntropyLoss": "reference-overflow (loss itself is inf)",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def all_37_tasks():
    d = load_json(TASKS_PATH)
    return sorted(t for fam in d["families"].values() for t in fam)


def clean_32_tasks():
    return sorted(set(all_37_tasks()) - set(FLAWED_TASKS))


def docinject_20_tasks():
    return sorted(load_json(TASKS_PATH)["doc_ablation_subset_of_20"]["tasks"])


def docinject_clean_tasks():
    return sorted(set(docinject_20_tasks()) - set(FLAWED_TASKS))


def agg_compiled_correct(records, task_filter=None):
    """{"language|model": {n, compiled, correct}} -- optionally restricted
    to task_filter (a set of task names); records outside it are skipped.
    Keys are "|"-joined strings (not tuples) so results are JSON-serializable
    as-is."""
    by = {}
    for r in records:
        if task_filter is not None and r["task"] not in task_filter:
            continue
        k = f"{r['language']}|{r['model'].split('/')[-1]}"
        d = by.setdefault(k, {"n": 0, "compiled": 0, "correct": 0})
        d["n"] += 1
        if r.get("compiled"):
            d["compiled"] += 1
        if r.get("correctness"):
            d["correct"] += 1
    return by


def print_table(title, all_by, clean_by):
    print(f"\n=== {title} ===")
    print(f"{'lang':9s} {'model':28s} | {'n(37)':>6s} {'compiled':>9s} {'correct':>8s} "
          f"| {'n(32)':>6s} {'compiled':>9s} {'correct':>8s}")
    for k in sorted(set(all_by) | set(clean_by)):
        lang, model = k.split("|", 1)
        a = all_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        c = clean_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        print(f"{lang:9s} {model:28s} | {a['n']:6d} {a['compiled']:9d} {a['correct']:8d} "
              f"| {c['n']:6d} {c['compiled']:9d} {c['correct']:8d}")


def table1_main_run(full_run_path):
    """Table 1: 2x2 main result, 0-shot, 37 tasks (all) vs 32 tasks (clean)."""
    recs = load_json(full_run_path)["records"]
    recs = [r for r in recs if r["condition"] == "0shot"]
    all_by = agg_compiled_correct(recs, task_filter=set(all_37_tasks()))
    clean_by = agg_compiled_correct(recs, task_filter=set(clean_32_tasks()))
    print_table("Table 1: 0-shot main run (n(37)=185/cell, n(32)~160/cell)", all_by, clean_by)
    return {"all_37": all_by, "clean_32": clean_by}


def table2_docinject(full_run_path, docinject_path):
    """Table 2: docinject before/after, 20-task subset (all) vs 17-task
    (clean -- the 3 flawed tasks that happen to be in the 20-task subset
    removed: 23_Softmax, 6_Matmul_with_large_K_dimension_, 95_CrossEntropyLoss)."""
    zero = load_json(full_run_path)["records"]
    doc = load_json(docinject_path)["records"]
    subset20 = set(docinject_20_tasks())
    subset17 = set(docinject_clean_tasks())
    overlap = subset20 - subset17
    print(f"\n=== Table 2: docinject 20-task subset -- {len(overlap)}/20 tasks flawed "
          f"(excluded for the clean 17-task basis): {sorted(overlap)} ===")

    zero_sub_all = [r for r in zero if r["condition"] == "0shot" and r["task"] in subset20]
    zero_sub_clean = [r for r in zero if r["condition"] == "0shot" and r["task"] in subset17]
    doc_all = [r for r in doc if r["condition"] == "docinject" and r["task"] in subset20]
    doc_clean = [r for r in doc if r["condition"] == "docinject" and r["task"] in subset17]

    zero_all_by = agg_compiled_correct(zero_sub_all)
    zero_clean_by = agg_compiled_correct(zero_sub_clean)
    doc_all_by = agg_compiled_correct(doc_all)
    doc_clean_by = agg_compiled_correct(doc_clean)

    print(f"{'lang':9s} {'model':28s} | {'0shot(20)':>17s} {'doc(20)':>15s} "
          f"| {'0shot(17)':>17s} {'doc(17)':>15s}")
    for k in sorted(set(zero_all_by) | set(doc_all_by)):
        lang, model = k.split("|", 1)
        za = zero_all_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        da = doc_all_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        zc = zero_clean_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        dc = doc_clean_by.get(k, {"n": 0, "compiled": 0, "correct": 0})
        print(f"{lang:9s} {model:28s} | "
              f"{za['compiled']:3d}/{za['n']:3d},{za['correct']:3d}c   "
              f"{da['compiled']:3d}/{da['n']:3d},{da['correct']:3d}c | "
              f"{zc['compiled']:3d}/{zc['n']:3d},{zc['correct']:3d}c   "
              f"{dc['compiled']:3d}/{dc['n']:3d},{dc['correct']:3d}c")
    return {
        "flawed_in_subset20": sorted(overlap),
        "subset20": {"0shot": zero_all_by, "docinject": doc_all_by},
        "subset17_clean": {"0shot": zero_clean_by, "docinject": doc_clean_by},
    }


def speedup_table(timing_path):
    """Speedup geomean/fast_1, raw (all tasks present) vs clean (flawed
    tasks excluded task-wide, not just flagged samples -- see #7-1)."""
    recs = [r for r in load_json(timing_path)["records"] if "timing_exception" not in r]

    def agg(records):
        by = {}
        for r in records:
            k = (r["language"], r["model"].split("/")[-1], r["condition"])
            by.setdefault(k, []).append(r["speedup"])
        out = {}
        for k, sp in by.items():
            arr = np.array(sp)
            out[k] = {"n": len(arr), "fast_1": int((arr > 1).sum()),
                       "geomean": float(np.exp(np.mean(np.log(arr))))}
        return out

    raw = agg(recs)
    clean = agg([r for r in recs if r["task"] not in FLAWED_TASKS])
    print(f"\n=== Speedup (raw vs clean -- {len(FLAWED_TASKS)} flawed tasks excluded task-wide) ===")
    print(f"{'lang':9s} {'model':25s} {'cond':10s} | {'n(raw)':>6s} {'fast1':>5s} {'geo':>7s} "
          f"| {'n(clean)':>8s} {'fast1':>5s} {'geo':>7s}")
    for k in sorted(set(raw) | set(clean)):
        r = raw.get(k, {"n": 0, "fast_1": 0, "geomean": float("nan")})
        c = clean.get(k, {"n": 0, "fast_1": 0, "geomean": float("nan")})
        print(f"{k[0]:9s} {k[1]:25s} {k[2]:10s} | {r['n']:6d} {r['fast_1']:5d} {r['geomean']:7.3g} "
              f"| {c['n']:8d} {c['fast_1']:5d} {c['geomean']:7.3g}")
    return {"raw": {"|".join(k): v for k, v in raw.items()},
            "clean": {"|".join(k): v for k, v in clean.items()}}


def a100_cross_comparison(pro6000_path, a100_path):
    """PRO 6000 (Blackwell) vs A100 (Ampere), 0-shot, cuda/ptx/triton only
    (A100 probe has no tilelang), 37-task and 32-task clean basis."""
    pro = [r for r in load_json(pro6000_path)["records"]
           if r["condition"] == "0shot" and r["language"] in ("cuda", "ptx", "triton")]
    if not Path(a100_path).exists():
        print(f"\n=== A100 cross-comparison SKIPPED -- {a100_path} not found "
              f"(git show origin/results-a100:results/eval/eval_a100_full.json > {a100_path}) ===")
        return None
    a100 = load_json(a100_path)["records"]

    print("\n=== PRO 6000 (Blackwell, sm_120a) vs A100 (Ampere, sm_80), 0-shot, cuda/ptx/triton ===")
    out = {}
    for label, recs in (("PRO6000", pro), ("A100", a100)):
        all_by = agg_compiled_correct(recs, task_filter=set(all_37_tasks()))
        clean_by = agg_compiled_correct(recs, task_filter=set(clean_32_tasks()))
        print_table(f"{label} (n(37)=185/cell, n(32)~160/cell)", all_by, clean_by)
        out[label] = {"all_37": all_by, "clean_32": clean_by}
    return out


def write_csv(rows, out_path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[analyze] wrote {out_path} ({len(rows)} rows)")


def fig1_ever_correct_rows(state, hardware="sm_120"):
    """paper/figures_data/fig1_ever_correct.csv rows -- see PI spec in
    paper/FINAL_DELIVERABLES.md #1. One row per (turn, lang, model,
    condition), turn 1..max_turn. ever_correct is recomputed directly from
    history[].correctness (NOT chain['best_turn'] -- best_turn is only set
    when a correct turn's timing ALSO succeeded, so it silently undercounts
    a chain that was correct but hit a timing exception; see
    multiturn.py cmd_report's PRIMARY(a) comment for the same fix applied
    there). A terminated chain stays in its group at every turn -- its
    contribution to `ever` is already monotonic by construction (it only
    ever gets counted once it was actually correct by that turn), so
    terminated chains need no special-casing to "hold" their value."""
    chains = list(state["chains"].values())
    max_turn = max((h["turn"] for c in chains for h in c["history"]), default=1)
    keys = sorted({(c["language"], c["model"].split("/")[-1]) for c in chains})
    rows = []
    for t in range(1, max_turn + 1):
        for lang, model in keys:
            for cond in ("0shot", "docinject"):
                group = [c for c in chains if c["language"] == lang
                         and c["model"].split("/")[-1] == model and c["condition"] == cond]
                if not group:
                    continue
                ever = sum(1 for c in group
                           if any(h["correctness"] for h in c["history"] if h["turn"] <= t))
                rows.append({"hardware": hardware, "turn": t, "lang": lang, "model": model,
                             "condition": cond, "ever_correct": ever, "n": len(group)})
    return rows


def fig2_speedup_rows(state, hardware="sm_120"):
    """paper/figures_data/fig2_speedup.csv rows -- see PI spec in
    paper/FINAL_DELIVERABLES.md #1. Turn-loop (in-chain) best-so-far
    speedup trajectory, same basis as the turn report's PRIMARY(b) table --
    NOT the final re-timing pass (that's final_retiming_summary_rows below,
    the sole speedup source for the paper's actual numbers)."""
    chains = list(state["chains"].values())
    max_turn = max((h["turn"] for c in chains for h in c["history"]), default=1)
    keys = sorted({(c["language"], c["model"].split("/")[-1]) for c in chains})
    rows = []
    for t in range(1, max_turn + 1):
        for lang, model in keys:
            group = [c for c in chains if c["language"] == lang and c["model"].split("/")[-1] == model]
            best_so_far = []
            for c in group:
                vals = [h["speedup"] for h in c["history"]
                        if h["turn"] <= t and h["correctness"] and h.get("speedup") is not None]
                if vals:
                    best_so_far.append(max(vals))
            if best_so_far:
                arr = np.array(best_so_far, dtype=float)
                rows.append({"hardware": hardware, "turn": t, "lang": lang, "model": model,
                             "n_timed": len(best_so_far),
                             "geomean_best_speedup": float(np.exp(np.mean(np.log(arr))))})
    return rows


def final_retiming_summary_rows(retiming_json_path):
    """paper/figures_data/final_retiming_summary.csv rows -- see PI spec in
    paper/FINAL_DELIVERABLES.md #1. Reads a scripts/final_retiming.py output
    JSON (results/eval/final_timing_*.json). This CSV -- not fig2_speedup.csv
    above, and not any turn-loop number in multiturn_state.json -- is the
    ONLY source the paper's reported speedup/fast_1 figures may cite
    (CLAUDE.md item 4 / final_retiming.py's own docstring)."""
    data = load_json(retiming_json_path)
    results = [r for r in data["results"] if "timing_exception" not in r]
    by = {}
    for r in results:
        k = (r["language"], r["model"].split("/")[-1], r["condition"])
        by.setdefault(k, []).append(r["speedup"])
    rows = []
    for (lang, model, cond), sp in sorted(by.items()):
        arr = np.array(sp, dtype=float)
        rows.append({"lang": lang, "model": model, "condition": cond, "n": len(arr),
                     "geomean": float(np.exp(np.mean(np.log(arr)))),
                     "fast_1": int((arr > 1).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-run", default=str(EVAL_DIR / "full_run_20260819.json"))
    ap.add_argument("--docinject", default=str(EVAL_DIR / "docinject_run_20260820T072056.json"))
    ap.add_argument("--timing", default=str(EVAL_DIR / "timing_20260820.json"))
    ap.add_argument("--a100", default=str(EVAL_DIR / "eval_a100_full.json"),
                    help="from `git show origin/results-a100:results/eval/eval_a100_full.json`")
    ap.add_argument("--out", default=None, help="default: results/eval/analysis_<timestamp>.json")
    ap.add_argument("--multiturn-state", default=str(EVAL_DIR / "multiturn_state.json"),
                    help="drives fig1_ever_correct.csv / fig2_speedup.csv (paper/FINAL_DELIVERABLES.md #1)")
    ap.add_argument("--figures-dir", default=str(FIGURES_DIR_DEFAULT))
    ap.add_argument("--hardware", default="sm_120", help="stamped into fig1_ever_correct.csv/fig2_speedup.csv")
    ap.add_argument("--final-retiming", default=None,
                    help="path to a results/eval/final_timing_*.json (scripts/final_retiming.py "
                         "output) -- only exists after the multi-turn loop completes turn 10 / "
                         "hits the cutoff. Produces final_retiming_summary.csv, the SOLE source "
                         "the paper's speedup/fast_1 numbers may cite (not fig2_speedup.csv).")
    args = ap.parse_args()

    print(f"[analyze] primary basis: {len(clean_32_tasks())} clean tasks "
          f"(37 - {len(FLAWED_TASKS)} flawed: {sorted(FLAWED_TASKS)})")

    results = {"flawed_tasks": FLAWED_TASKS}
    if Path(args.full_run).exists():
        results["table1"] = table1_main_run(args.full_run)
    if Path(args.full_run).exists() and Path(args.docinject).exists():
        results["table2"] = table2_docinject(args.full_run, args.docinject)
    if Path(args.timing).exists():
        results["speedup"] = speedup_table(args.timing)
    results["a100_cross"] = a100_cross_comparison(args.full_run, args.a100)

    figures_dir = Path(args.figures_dir)
    if Path(args.multiturn_state).exists():
        state = json.loads(Path(args.multiturn_state).read_text())
        write_csv(fig1_ever_correct_rows(state, hardware=args.hardware),
                  figures_dir / "fig1_ever_correct.csv",
                  ["hardware", "turn", "lang", "model", "condition", "ever_correct", "n"])
        write_csv(fig2_speedup_rows(state, hardware=args.hardware),
                  figures_dir / "fig2_speedup.csv",
                  ["hardware", "turn", "lang", "model", "n_timed", "geomean_best_speedup"])
    else:
        print(f"[analyze] SKIPPED fig1/fig2 CSVs -- {args.multiturn_state} not found")
    if args.final_retiming:
        if Path(args.final_retiming).exists():
            write_csv(final_retiming_summary_rows(args.final_retiming),
                      figures_dir / "final_retiming_summary.csv",
                      ["lang", "model", "condition", "n", "geomean", "fast_1"])
        else:
            print(f"[analyze] SKIPPED final_retiming_summary.csv -- {args.final_retiming} not found")

    import time
    out_path = Path(args.out) if args.out else EVAL_DIR / f"analysis_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Checks an eval JSON covers every generated sample exactly once.

Run after evaluate.py finishes (in particular after a --resume continuation,
where records come from more than one process):

    python scripts/verify_eval_completeness.py results/eval/full_run_20260819.json

Three things can go wrong when a run is stitched together from a checkpoint,
and all three are silent in the summary counts, so they get asserted here:
  1. MISSING  -- a raw sample under results/raw/ that no record covers.
  2. DUPLICATE-- the same sample path recorded twice (would double-count it in
                 every rate the paper reports).
  3. EXTRA    -- a record whose raw sample no longer exists on disk.
Exit status is 0 only if all three sets are empty and the record count matches
the number of raw samples.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--expect", type=int, default=None,
                    help="expected sample count (e.g. 1480); default: count raw files")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_paths = {str(p.relative_to(raw_dir)) for p in raw_dir.rglob("sample_*.json")}
    data = json.loads(Path(args.eval_json).read_text())
    records = data.get("records", [])
    counts = collections.Counter(r["path"] for r in records)

    missing = sorted(raw_paths - set(counts))
    extra = sorted(set(counts) - raw_paths)
    dupes = sorted(p for p, n in counts.items() if n > 1)
    expected = args.expect if args.expect is not None else len(raw_paths)

    print(f"eval file      : {args.eval_json}")
    print(f"run status     : {data.get('status')}")
    print(f"raw samples    : {len(raw_paths)}")
    print(f"eval records   : {len(records)}  (unique paths: {len(counts)})")
    print(f"expected       : {expected}")
    print(f"missing        : {len(missing)}")
    print(f"duplicates     : {len(dupes)}")
    print(f"extra          : {len(extra)}")
    for label, items in (("MISSING", missing), ("DUPLICATE", dupes), ("EXTRA", extra)):
        for p in items[:20]:
            print(f"  {label}: {p}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more {label}")

    ok = (not missing and not dupes and not extra
          and len(records) == expected and data.get("status") == "complete")
    print("\nVERIFY: " + ("PASS" if ok else "FAIL"))
    if not ok and data.get("status") != "complete":
        print("  (run status is not 'complete' -- evaluate.py has not finished)")

    if ok:
        by = collections.defaultdict(lambda: collections.Counter())
        for r in records:
            k = (r["language"], r["model"].split("/")[-1])
            by[k]["n"] += 1
            by[k]["gen"] += r["gen_status"] == "generated"
            by[k]["trunc"] += r["gen_status"] == "truncated"
            by[k]["fmt"] += r["gen_status"] == "format_failure"
            by[k]["compiled"] += bool(r.get("compiled"))
            by[k]["correct"] += bool(r.get("correctness"))
        print(f"\n{'lang':9s} {'model':30s} {'n':>4s} {'gen':>4s} {'trunc':>5s} "
              f"{'fmt':>4s} {'comp':>5s} {'corr':>5s}")
        for k in sorted(by):
            c = by[k]
            print(f"{k[0]:9s} {k[1]:30s} {c['n']:4d} {c['gen']:4d} {c['trunc']:5d} "
                  f"{c['fmt']:4d} {c['compiled']:5d} {c['correct']:5d}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

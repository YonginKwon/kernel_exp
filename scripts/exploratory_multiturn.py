#!/usr/bin/env python3
"""deep-turn-probe wrapper around scripts/multiturn.py's generate/evaluate/
report subcommands (EXPLORATORY_PROTOCOL.md). Delegates to multiturn.py's
own cmd_generate/cmd_evaluate/cmd_report COMPLETELY UNMODIFIED -- this file
only monkeypatches two bare module-level constants before each call, per
this project's established non-invasive-reuse convention (see
scripts/multiturn_init_a100.py, scripts/measure_baseline.py precedent):

  - multiturn.RAW_DIR       -> results/exploratory/raw (isolation: this
                                experiment's generations never touch
                                results/raw/, the main experiment's tree)
  - multiturn.NO_IMPROVE_LIMIT -> a very large number (EXPLORATORY_PROTOCOL.md
                                deviation (2): no-improvement-3-turn early
                                termination disabled for long-horizon
                                observation)

k_max=100 (deviation (1)) needs NO monkeypatch -- cmd_evaluate already reads
state.get("k_max", K_MAX), and state_armA.json/state_armB.json were written
with k_max=100 baked in by scripts/exploratory_init.py.

Usage: identical subcommands/flags to `python scripts/multiturn.py`, e.g.:
    python scripts/exploratory_multiturn.py generate \\
        --state results/exploratory/state_armA.json \\
        --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \\
        --manifest logs/vllm/gptoss_manifest.json --concurrency 8 --confirm-run
    python scripts/exploratory_multiturn.py evaluate --state results/exploratory/state_armA.json
    python scripts/exploratory_multiturn.py report --state results/exploratory/state_armA.json
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import multiturn as mt  # noqa: E402
import evaluate as ev  # noqa: E402

EXPLORATORY_RAW_DIR = REPO_ROOT / "results" / "exploratory" / "raw"
NO_IMPROVE_LIMIT_DISABLED = 10 ** 9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--state", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--confirm-run", action="store_true")
    p.set_defaults(func=mt.cmd_generate)

    p = sub.add_parser("evaluate")
    p.add_argument("--state", required=True)
    p.add_argument("--precompile-workers", type=int, default=ev.PRECOMPILE_WORKERS)
    p.set_defaults(func=mt.cmd_evaluate)

    p = sub.add_parser("report")
    p.add_argument("--state", required=True)
    p.set_defaults(func=mt.cmd_report)

    args = ap.parse_args()

    mt.RAW_DIR = EXPLORATORY_RAW_DIR
    mt.NO_IMPROVE_LIMIT = NO_IMPROVE_LIMIT_DISABLED
    print(f"[exploratory] monkeypatched multiturn.RAW_DIR={mt.RAW_DIR}, "
          f"multiturn.NO_IMPROVE_LIMIT={mt.NO_IMPROVE_LIMIT} (deviations per "
          f"EXPLORATORY_PROTOCOL.md; multiturn.py itself unmodified)", file=sys.stderr)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

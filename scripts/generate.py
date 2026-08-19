#!/usr/bin/env python3
"""The ONLY kernel-generation entry point for this project (CLAUDE.md rule 2:
"커널 생성은 오직 scripts/generate.py의 API 호출로만 수행한다"). Never write or
edit a generated kernel by hand, and never generate one in a Claude Code
session directly -- this script is the sole path to results/.

Prompt construction goes entirely through prompts/spec_loader.py, which
parses prompts/PROMPT_SPEC.md -- PROMPT_SPEC.md is the single source of
truth for prompt text; nothing here hardcodes template strings.

API calls go through litellm (already a KernelBench dependency, see
requirements.txt), so one code path handles both providers: pass a
litellm-style model string (e.g. "gpt-5.1" or "claude-opus-5-20260101") via
--model. Reads OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment only
(CLAUDE.md rule: never in code or logs).

Every call is logged in full (CLAUDE.md rule 4): model string, temperature,
seed, prompt (hash + verbatim text), timestamp, torch/CUDA/driver version,
and the raw response -- to results/raw/, which is read-only data once
written (CLAUDE.md rule 1: never hand-edit anything under results/).

Usage:
    # Always estimate cost first (no API calls, no network):
    python scripts/generate.py --language cuda --condition 0shot \\
        --provider-model gpt-5.1 --dry-run

    # Then run for real:
    python scripts/generate.py --language cuda --condition 0shot \\
        --provider-model gpt-5.1 --confirm-cost
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "prompts"))
from spec_loader import get_spec, load_language_spec, LANGUAGE_DISPLAY  # noqa: E402

TASKS_PATH = REPO_ROOT / "tasks" / "level1_subset.json"
KERNELBENCH_LEVEL1 = REPO_ROOT / "third_party" / "KernelBench" / "KernelBench" / "level1"
RESULTS_RAW = REPO_ROOT / "results" / "raw"

# Rough $/1M-token estimates for cost projection ONLY (--dry-run). Not used
# for anything billed. Update if the pinned models (still undecided -- see
# CLAUDE.md "모델: ... 버전 문자열 고정") change; unknown models fall back to
# a conservative placeholder and print a warning rather than silently
# under-estimating.
_PRICE_PER_1M_FALLBACK = {"input": 5.0, "output": 15.0}
_KNOWN_PRICES_PER_1M = {
    # populate once the PI pins exact model version strings (CLAUDE.md rule:
    # "모델 버전 문자열, ... 로그로 남긴다"); left empty on purpose so the
    # fallback's conservative estimate + warning is what shows up until then.
}


def load_tasks(family_filter=None):
    manifest = json.loads(TASKS_PATH.read_text())
    tasks = []
    for family, names in manifest["families"].items():
        if family_filter and family not in family_filter:
            continue
        for name in names:
            tasks.append({"family": family, "name": name})
    return tasks


def load_reference_code(task_name: str) -> str:
    path = KERNELBENCH_LEVEL1 / f"{task_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"task source not found: {path}")
    return path.read_text()


def extract_first_code_block(text: str) -> str | None:
    """PROMPT_SPEC.md §5: first complete fenced code block, any/no language
    tag. None (-> 'format_failure' at the caller) if there isn't one."""
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def env_fingerprint() -> dict:
    """torch/CUDA/driver versions for the log (CLAUDE.md rule 4)."""
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["driver_version"] = out.stdout.strip()
    except Exception as e:
        info["driver_version_error"] = str(e)
    return info


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # crude fallback, clearly not exact


def build_all_prompts(language: str, tasks: list, condition: str) -> list:
    spec = get_spec()
    doc_text = load_language_spec(language) if condition == "docinject" else None
    prompts = []
    for task in tasks:
        ref = load_reference_code(task["name"])
        prompt = spec.build_prompt(language, ref, doc_spec_text=doc_text)
        prompts.append({"task": task, "prompt": prompt})
    return prompts


def dry_run_report(language: str, tasks: list, condition: str, num_samples: int, model: str):
    prompts = build_all_prompts(language, tasks, condition)
    total_input_tokens = 0
    for p in prompts:
        total_input_tokens += estimate_tokens(p["prompt"])
    spec = get_spec()
    max_out = spec.generation_params.get("max_output_tokens", 8192)
    total_calls = len(tasks) * num_samples
    total_output_tokens_upper_bound = total_calls * max_out  # worst case, most won't hit the cap

    price = _KNOWN_PRICES_PER_1M.get(model, _PRICE_PER_1M_FALLBACK)
    if model not in _KNOWN_PRICES_PER_1M:
        print(f"[warn] no pinned price for model={model!r} -- using a conservative "
              f"fallback (${_PRICE_PER_1M_FALLBACK['input']}/{_PRICE_PER_1M_FALLBACK['output']} "
              f"per 1M in/out tokens). Real cost may differ; update _KNOWN_PRICES_PER_1M "
              f"once the PI pins model prices.")

    input_cost = (total_input_tokens * num_samples / 1_000_000) * price["input"]
    output_cost_upper = (total_output_tokens_upper_bound / 1_000_000) * price["output"]

    print(f"[dry-run] language={language} condition={condition} model={model}")
    print(f"[dry-run] tasks={len(tasks)} samples/task={num_samples} total_calls={total_calls}")
    print(f"[dry-run] avg prompt tokens/task={total_input_tokens // max(len(tasks),1)} "
          f"(cl100k_base estimate, not exact for every provider's tokenizer)")
    print(f"[dry-run] estimated input cost: ${input_cost:.2f}")
    print(f"[dry-run] estimated output cost (UPPER BOUND, assumes every "
          f"response hits max_output_tokens={max_out}): ${output_cost_upper:.2f}")
    print(f"[dry-run] estimated total cost: ${input_cost:.2f} - ${input_cost + output_cost_upper:.2f}")
    return prompts


def call_model(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None):
    import litellm
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    response = litellm.completion(**kwargs)
    return response


def run_generation(language: str, tasks: list, condition: str, model: str,
                    num_samples: int, temperature: float, max_tokens: int,
                    seed_base: int | None, out_dir: Path):
    spec = get_spec()
    doc_text = load_language_spec(language) if condition == "docinject" else None
    env_info = env_fingerprint()

    for task in tasks:
        ref = load_reference_code(task["name"])
        prompt = spec.build_prompt(language, ref, doc_spec_text=doc_text)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        for i in range(num_samples):
            seed = (seed_base + i) if seed_base is not None else None
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

            record = {
                "task": task["name"],
                "task_family": task["family"],
                "language": language,
                "condition": condition,
                "sample_index": i,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "timestamp": timestamp,
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
                "env": env_info,
            }

            try:
                response = call_model(model, prompt, temperature, max_tokens, seed)
                raw_text = response.choices[0].message.content
                record["response_raw"] = raw_text
                record["response_finish_reason"] = response.choices[0].finish_reason
                record["usage"] = dict(response.usage) if response.usage else None
                parsed = extract_first_code_block(raw_text)
                record["parsed_code"] = parsed
                record["status"] = "generated" if parsed is not None else "format_failure"
                if record["response_finish_reason"] == "length":
                    record["status"] = "truncated"
            except Exception as e:
                record["status"] = "api_error"
                record["error"] = f"{type(e).__name__}: {e}"

            task_dir = out_dir / language / condition / task["name"] / model.replace("/", "_")
            task_dir.mkdir(parents=True, exist_ok=True)
            out_path = task_dir / f"sample_{i}.json"
            out_path.write_text(json.dumps(record, indent=2, default=str))
            print(f"[gen] {task['name']} sample={i} status={record['status']} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=list(LANGUAGE_DISPLAY))
    ap.add_argument("--condition", default="0shot", choices=["0shot", "docinject"])
    ap.add_argument("--family", action="append", default=None,
                     help="restrict to one or more tasks/level1_subset.json families "
                          "(repeatable). Default: all families.")
    ap.add_argument("--provider-model", dest="model", required=True,
                     help="litellm-style model string, e.g. 'gpt-5.1' or "
                          "'claude-opus-5-20260101'. No default -- CLAUDE.md requires "
                          "the exact pinned version string be explicit and logged, "
                          "not guessed by this script.")
    ap.add_argument("--samples", type=int, default=None,
                     help="default: PROMPT_SPEC.md §4's value (5)")
    ap.add_argument("--temperature", type=float, default=None,
                     help="default: PROMPT_SPEC.md §4's value (0.8)")
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="default: PROMPT_SPEC.md §4's value (8192)")
    ap.add_argument("--seed", type=int, default=None,
                     help="base seed; sample i uses seed+i if given. Only honored by "
                          "providers/models that support it.")
    ap.add_argument("--out-dir", default=str(RESULTS_RAW))
    ap.add_argument("--dry-run", action="store_true",
                     help="build every prompt and estimate cost; NO API calls, NO network.")
    ap.add_argument("--confirm-cost", action="store_true",
                     help="required (in addition to omitting --dry-run) to actually spend "
                          "money -- CLAUDE.md: '실행 전 예상 API 비용을 추산해 보고할 것'.")
    args = ap.parse_args()

    if args.condition == "docinject":
        approved = json.loads(TASKS_PATH.read_text())["doc_ablation_subset_of_20"]["status"]
        if approved.startswith("PROPOSED"):
            print("[refuse] tasks/level1_subset.json's doc_ablation_subset_of_20 is still "
                  "PROPOSED (PI has not approved it, see tasks/SELECTION.md #4.2). "
                  "Refusing to run --condition docinject until that's approved.", file=sys.stderr)
            return 1

    tasks = load_tasks(family_filter=set(args.family) if args.family else None)
    if args.condition == "docinject":
        approved_names = set(json.loads(TASKS_PATH.read_text())["doc_ablation_subset_of_20"]["tasks"])
        tasks = [t for t in tasks if t["name"] in approved_names]

    spec = get_spec()
    num_samples = args.samples or spec.generation_params.get("num_samples", 5)
    temperature = args.temperature if args.temperature is not None else spec.generation_params.get("temperature", 0.8)
    max_tokens = args.max_tokens or spec.generation_params.get("max_output_tokens", 8192)

    if args.dry_run:
        dry_run_report(args.language, tasks, args.condition, num_samples, args.model)
        return 0

    if not args.confirm_cost:
        print("[refuse] this would make real, billed API calls. Run with --dry-run first "
              "to see the cost estimate, then re-run with --confirm-cost to proceed.",
              file=sys.stderr)
        return 1

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[refuse] neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set in the "
              "environment. CLAUDE.md: API keys only via env vars, never in code/logs.",
              file=sys.stderr)
        return 1

    dry_run_report(args.language, tasks, args.condition, num_samples, args.model)
    print(f"\n[confirmed] proceeding with {len(tasks) * num_samples} real API calls...")
    run_generation(args.language, tasks, args.condition, args.model, num_samples,
                    temperature, max_tokens, args.seed, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

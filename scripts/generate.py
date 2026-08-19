#!/usr/bin/env python3
"""The ONLY kernel-generation entry point for this project (CLAUDE.md rule 2:
"커널 생성은 오직 scripts/generate.py의 API 호출로만 수행한다"). Never write or
edit a generated kernel by hand, and never generate one in a Claude Code
session directly -- this script is the sole path to results/.

Prompt construction goes entirely through prompts/spec_loader.py, which
parses prompts/PROMPT_SPEC.md -- PROMPT_SPEC.md is the single source of
truth for prompt text; nothing here hardcodes template strings.

2026-08-19: API-provider path (litellm/OpenAI-Anthropic) dropped per PI
directive. Generation now targets a locally-served, OpenAI-compatible vLLM
endpoint (see scripts/serve_h100.sh, run on the department's H100x2 server)
via the `openai` SDK pointed at that endpoint's --base-url. There is no API
key in any real sense -- vLLM does not authenticate -- so this script never
reads OPENAI_API_KEY/ANTHROPIC_API_KEY and never checks for them; the only
required connection info is --base-url (no default, so a typo fails loudly
instead of silently hitting the wrong server) and --model (the exact name
vLLM serves the checkpoint under -- normally the HF repo id, e.g.
"Qwen/Qwen3-Coder-Next-FP8" or "openai/gpt-oss-120b").

Every call is logged in full (CLAUDE.md rule 4): base_url + model, HF
checkpoint revision + vLLM version + dtype (read from
logs/vllm/<name>_manifest.json, written by serve_h100.sh -- pass its path
via --manifest so this data isn't hand-typed), temperature, seed, prompt
(hash + verbatim text), timestamp, torch/CUDA/driver version (this
-- evaluation -- machine's), and the raw response -- to results/raw/, which
is read-only data once written (CLAUDE.md rule 1: never hand-edit anything
under results/).

Usage:
    # Always dry-run first (builds every prompt, no network calls):
    python scripts/generate.py --language cuda --condition 0shot \\
        --base-url http://h100-host:8000/v1 --model Qwen/Qwen3-Coder-Next-FP8 \\
        --manifest logs/vllm/qwen_manifest.json --dry-run

    # Then run for real:
    python scripts/generate.py --language cuda --condition 0shot \\
        --base-url http://h100-host:8000/v1 --model Qwen/Qwen3-Coder-Next-FP8 \\
        --manifest logs/vllm/qwen_manifest.json --confirm-run
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "prompts"))
from spec_loader import get_spec, load_language_spec, LANGUAGE_DISPLAY  # noqa: E402

TASKS_PATH = REPO_ROOT / "tasks" / "level1_subset.json"
KERNELBENCH_LEVEL1 = REPO_ROOT / "third_party" / "KernelBench" / "KernelBench" / "level1"
RESULTS_RAW = REPO_ROOT / "results" / "raw"


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
    """torch/CUDA/driver versions for the log (CLAUDE.md rule 4) -- this is
    the EVALUATION machine's environment (where this script runs), not the
    H100 generation server's. The generation server's facts (HF revision,
    vLLM version, dtype) come from --manifest instead, see load_manifest()."""
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


def load_manifest(path: str | None) -> dict:
    """serve_h100.sh writes logs/vllm/<name>_manifest.json with the HF
    revision / vLLM version / dtype facts CLAUDE.md rule 4 requires. Passing
    it is optional but strongly recommended -- without it those fields are
    just recorded as null and have to be filled in by hand later."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] --manifest {path} does not exist -- HF revision/vLLM version/"
              f"dtype will be logged as null. Copy it from the H100 server first.",
              file=sys.stderr)
        return {}
    return json.loads(p.read_text())


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


def dry_run_report(language: str, tasks: list, condition: str, num_samples: int,
                    model: str, base_url: str) -> list:
    prompts = build_all_prompts(language, tasks, condition)
    total_input_tokens = sum(estimate_tokens(p["prompt"]) for p in prompts)
    total_calls = len(tasks) * num_samples

    print(f"[dry-run] language={language} condition={condition} model={model} base_url={base_url}")
    print(f"[dry-run] tasks={len(tasks)} samples/task={num_samples} total_calls={total_calls}")
    print(f"[dry-run] avg prompt tokens/task={total_input_tokens // max(len(tasks), 1)} "
          f"(cl100k_base estimate, not exact for every tokenizer)")
    print("[dry-run] no cost estimate (local inference, no per-token billing) -- "
          "the thing to check before a big run is wall-clock: time one sample by hand "
          "against the endpoint first if this is a new model/box.")
    return prompts


def check_endpoint_reachable(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
        print(f"[refuse] endpoint not reachable: GET {url} -> {type(e).__name__}: {e}", file=sys.stderr)
        return False


def call_model(client, model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    return client.chat.completions.create(**kwargs)


_print_lock = threading.Lock()


def _generate_one(client, language, condition, model, base_url, manifest, spec, doc_text,
                   env_info, temperature, max_tokens, seed_base, out_dir, task, i):
    ref = load_reference_code(task["name"])
    prompt = spec.build_prompt(language, ref, doc_spec_text=doc_text)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    seed = (seed_base + i) if seed_base is not None else None
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    record = {
        "task": task["name"],
        "task_family": task["family"],
        "language": language,
        "condition": condition,
        "sample_index": i,
        "model": model,
        "base_url": base_url,
        "hf_revision": manifest.get("hf_revision"),
        "vllm_version": manifest.get("vllm_version"),
        "tensor_parallel_size": manifest.get("tensor_parallel_size"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "timestamp": timestamp,
        "prompt_sha256": prompt_hash,
        "prompt": prompt,
        "env": env_info,
    }

    try:
        response = call_model(client, model, prompt, temperature, max_tokens, seed)
        raw_text = response.choices[0].message.content
        record["response_raw"] = raw_text
        record["response_finish_reason"] = response.choices[0].finish_reason
        record["usage"] = response.usage.model_dump() if response.usage else None
        parsed = extract_first_code_block(raw_text or "")
        record["parsed_code"] = parsed
        record["status"] = "generated" if parsed is not None else "format_failure"
        if record["response_finish_reason"] == "length":
            record["status"] = "truncated"
    except Exception as e:
        record["status"] = "request_error"
        record["error"] = f"{type(e).__name__}: {e}"

    task_dir = out_dir / language / condition / task["name"] / model.replace("/", "_")
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / f"sample_{i}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    with _print_lock:
        print(f"[gen] {task['name']} sample={i} status={record['status']} -> {out_path}")
    return record["status"]


def run_generation(language: str, tasks: list, condition: str, model: str, base_url: str,
                    manifest: dict, num_samples: int, temperature: float, max_tokens: int,
                    seed_base: int | None, out_dir: Path, concurrency: int = 1):
    from openai import OpenAI
    # vLLM handles concurrent requests fine (continuous batching); each
    # in-flight request needs its own client-side connection, but the openai
    # SDK's client is safe to share across threads for this. File writes are
    # one-per-sample-path so there's no cross-thread contention there either.
    client = OpenAI(base_url=base_url, api_key="not-needed-for-vllm")

    spec = get_spec()
    doc_text = load_language_spec(language) if condition == "docinject" else None
    env_info = env_fingerprint()

    jobs = [(task, i) for task in tasks for i in range(num_samples)]

    if concurrency <= 1:
        for task, i in jobs:
            _generate_one(client, language, condition, model, base_url, manifest, spec, doc_text,
                          env_info, temperature, max_tokens, seed_base, out_dir, task, i)
        return

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_generate_one, client, language, condition, model, base_url, manifest,
                        spec, doc_text, env_info, temperature, max_tokens, seed_base, out_dir, task, i)
            for task, i in jobs
        ]
        for f in as_completed(futures):
            f.result()  # re-raise any exception that escaped _generate_one's own try/except


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=list(LANGUAGE_DISPLAY))
    ap.add_argument("--condition", default="0shot", choices=["0shot", "docinject"])
    ap.add_argument("--family", action="append", default=None,
                     help="restrict to one or more tasks/level1_subset.json families "
                          "(repeatable). Default: all families.")
    ap.add_argument("--task", action="append", default=None,
                     help="restrict to specific task name(s) (repeatable), e.g. for a "
                          "pilot run. Combines with --family if both given.")
    ap.add_argument("--base-url", required=True,
                     help="OpenAI-compatible endpoint, e.g. http://h100-host:8000/v1 "
                          "(scripts/serve_h100.sh's Qwen port) or :8001 (gpt-oss). "
                          "No default -- a wrong/missing URL must fail loudly, not "
                          "silently hit some other server.")
    ap.add_argument("--model", required=True,
                     help="the model name vLLM serves the checkpoint under (normally "
                          "the HF repo id, e.g. 'Qwen/Qwen3-Coder-Next-FP8' or "
                          "'openai/gpt-oss-120b').")
    ap.add_argument("--manifest", default=None,
                     help="path to serve_h100.sh's <name>_manifest.json, for HF "
                          "revision / vLLM version / tensor_parallel_size in the log. "
                          "Optional but recommended.")
    ap.add_argument("--samples", type=int, default=None,
                     help="default: PROMPT_SPEC.md §4's value (5)")
    ap.add_argument("--temperature", type=float, default=None,
                     help="default: PROMPT_SPEC.md §4's value (0.8)")
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="default: PROMPT_SPEC.md §4's value (8192)")
    ap.add_argument("--seed", type=int, default=None,
                     help="base seed; sample i uses seed+i. PROMPT_SPEC.md §4 requires "
                          "this be set for every real (non-dry-run) call.")
    ap.add_argument("--out-dir", default=str(RESULTS_RAW))
    ap.add_argument("--concurrency", type=int, default=1,
                     help="parallel in-flight requests to the endpoint (vLLM batches "
                          "concurrent requests fine). Default 1 (sequential); use e.g. "
                          "8-16 for large runs. Each sample still gets its own logged "
                          "record regardless of concurrency.")
    ap.add_argument("--dry-run", action="store_true",
                     help="build every prompt and report counts/token estimate; NO "
                          "network calls.")
    ap.add_argument("--confirm-run", action="store_true",
                     help="required (in addition to omitting --dry-run) to actually "
                          "call the endpoint -- guards against a fat-fingered command "
                          "kicking off hundreds of generations against a shared server.")
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
    if args.task:
        wanted = set(args.task)
        tasks = [t for t in tasks if t["name"] in wanted]
        missing = wanted - {t["name"] for t in tasks}
        if missing:
            print(f"[refuse] unknown --task name(s) not in tasks/level1_subset.json: {missing}", file=sys.stderr)
            return 1

    spec = get_spec()
    num_samples = args.samples or spec.generation_params.get("num_samples", 5)
    temperature = args.temperature if args.temperature is not None else spec.generation_params.get("temperature", 0.8)
    max_tokens = args.max_tokens or spec.generation_params.get("max_output_tokens", 8192)

    if args.dry_run:
        dry_run_report(args.language, tasks, args.condition, num_samples, args.model, args.base_url)
        return 0

    if not args.confirm_run:
        print("[refuse] this would make real requests against the endpoint. Run with "
              "--dry-run first, then re-run with --confirm-run to proceed.", file=sys.stderr)
        return 1

    if args.seed is None:
        print("[refuse] PROMPT_SPEC.md §4 requires a fixed seed per call for a real "
              "(non-dry-run) generation run. Pass --seed.", file=sys.stderr)
        return 1

    if not check_endpoint_reachable(args.base_url):
        return 1

    manifest = load_manifest(args.manifest)
    dry_run_report(args.language, tasks, args.condition, num_samples, args.model, args.base_url)
    print(f"\n[confirmed] proceeding with {len(tasks) * num_samples} real requests to {args.base_url} ...")
    run_generation(args.language, tasks, args.condition, args.model, args.base_url, manifest,
                    num_samples, temperature, max_tokens, args.seed, Path(args.out_dir),
                    concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Multi-turn completion protocol (PROMPT_SPEC.md #3.4, PI-confirmed
2026-08-20). Advances chains one turn at a time through a repair phase
(incorrect) <-> optimization phase (correct, does not terminate) until a
chain hits one of #3.4's termination conditions or k=10.

Chains: turn 1 is the ALREADY-GENERATED 0-shot/docinject data (results/raw/,
results/eval/full_run_20260819.json + docinject_run_20260820T072056.json),
restricted to the "clean" task basis (32 tasks for 0-shot, 17 for docinject
-- the 5 tolerance-flawed tasks from scripts/audit_tolerance.py are excluded
from multi-turn ENTIRELY, see PROMPT_SPEC.md #3.4's "완전 제외" section --
imported from scripts/analyze.py so there's one source of truth for the list).

Every turn's prompt is assembled FRESH and stateless (PROMPT_SPEC #3.4):
    {original turn-1 task prompt} + "Your previous solution:" + {previous
    turn's code} + {this turn's feedback -- one of 5 fixed templates from
    prompts/spec_loader.py, chosen by the previous turn's outcome}
No chat history, no accumulation across turns beyond that.

Usage (see logs/multiturn/run_pipeline.sh for the full turn-cycle wrapper):
    source scripts/env.sh && source .venv/bin/activate

    # One-time: bootstrap chain state from the existing turn-1 data.
    python scripts/multiturn.py init --state results/eval/multiturn_state.json

    # Generation stage (vLLM must be up at --base-url for --model). Network
    # calls only -- writes raw records, does NOT evaluate.
    python scripts/multiturn.py generate --state results/eval/multiturn_state.json \\
        --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 \\
        --manifest logs/vllm/gptoss_manifest.json --concurrency 16 --confirm-run

    # Evaluation stage (vLLM must be DOWN -- GPU exclusivity). Compiles
    # (CUDA precompiled in parallel, see evaluate.py), checks correctness,
    # times newly-correct samples, updates chain state + termination.
    python scripts/multiturn.py evaluate --state results/eval/multiturn_state.json

    # Summary: correct@turn / best-speedup@turn / termination breakdown.
    python scripts/multiturn.py report --state results/eval/multiturn_state.json
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

import generate as gen  # noqa: E402
import evaluate as ev  # noqa: E402
from analyze import FLAWED_TASKS, clean_32_tasks, docinject_clean_tasks  # noqa: E402
from spec_loader import get_spec  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
EVAL_DIR = REPO_ROOT / "results" / "eval"
STATE_DEFAULT = EVAL_DIR / "multiturn_state.json"
K_MAX = 10
NO_IMPROVE_LIMIT = 3  # PROMPT_SPEC #3.4 chain termination condition (2)


def chain_id(language, condition, task, model, sample_index):
    return f"{language}|{condition}|{task}|{model}|{sample_index}"


def _feedback_for(chain, spec):
    """One of #3.4's 5 fixed templates, chosen by the PREVIOUS turn's
    outcome. Only the aggregate stats go in for correctness failures (no
    reference tensor values, PROMPT_SPEC #3.4)."""
    gs = chain["gen_status"]
    if gs in ("truncated", "format_failure"):
        return spec.build_repair_parse_failure_feedback(), "repair"
    if not chain["compiled"]:
        md = chain.get("metadata") or {}
        err = md.get("compilation_error") or md.get("runtime_error") or json.dumps(md)[:2000]
        # PTX/CUDA/Triton/TileLang all funnel compile-time failures through
        # "compilation_error" in this project's metadata shape (evaluate.py) --
        # a distinct RUNTIME failure (compiled=True) is the case below.
        return spec.build_repair_compile_feedback(err), "repair"
    if not chain["correctness"]:
        md = chain.get("metadata") or {}
        if "runtime_error" in md:
            return spec.build_repair_runtime_feedback(md["runtime_error"]), "repair"
        max_diff = md.get("max_difference")
        max_diff = max_diff[-1] if isinstance(max_diff, list) and max_diff else (max_diff or "unknown")
        # KernelBench's own metadata doesn't give a mismatch FRACTION (only
        # max/avg absolute difference) -- avg_difference is the closest
        # available aggregate proxy; documented here rather than silently
        # treated as the literal "fraction of mismatched elements" text.
        avg_diff = md.get("avg_difference")
        avg_diff = avg_diff[-1] if isinstance(avg_diff, list) and avg_diff else (avg_diff or "unknown")
        return spec.build_repair_correctness_feedback(max_diff, avg_diff), "repair"
    # correctness True -> optimization phase, needs this turn's timing.
    t = chain.get("last_timing")
    if not t:
        raise ValueError(f"chain {chain['chain_id']}: correct but no timing recorded for feedback")
    return spec.build_optimization_feedback(t["kernel_ms"], t["baseline_ms"], t["speedup"]), "optimize"


def build_prompt(chain, spec):
    feedback, phase = _feedback_for(chain, spec)
    prompt = (chain["original_prompt"].rstrip("\n") + "\n\nYour previous solution:\n```\n"
              + (chain["code"] or "") + "\n```\n\n" + feedback)
    return prompt, phase


# --------------------------------------------------------------------------
# init: bootstrap chain state from the existing turn-1 (0-shot/docinject) data
# --------------------------------------------------------------------------

def _load_turn1_records():
    zero = json.loads((EVAL_DIR / "full_run_20260819.json").read_text())["records"]
    doc = json.loads((EVAL_DIR / "docinject_run_20260820T072056.json").read_text())["records"]
    clean32, clean17 = set(clean_32_tasks()), set(docinject_clean_tasks())
    out = [r for r in zero if r["condition"] == "0shot" and r["task"] in clean32]
    out += [r for r in doc if r["condition"] == "docinject" and r["task"] in clean17]
    return out


def cmd_init(args):
    state_path = Path(args.state)
    if state_path.exists() and not args.force:
        print(f"[refuse] {state_path} already exists -- pass --force to reinitialize "
              f"(this discards any turn>1 progress).", file=sys.stderr)
        return 1

    records = _load_turn1_records()
    print(f"[init] {len(records)} turn-1 chains (32-task 0-shot + 17-task docinject, "
          f"{len(FLAWED_TASKS)} flawed tasks excluded)")

    chains = {}
    for r in records:
        raw = json.loads((RAW_DIR / r["path"]).read_text())
        cid = chain_id(r["language"], r["condition"], r["task"], r["model"], r["sample_index"])
        chains[cid] = {
            "chain_id": cid,
            "language": r["language"], "condition": r["condition"], "task": r["task"],
            "model": r["model"], "sample_index": r["sample_index"],
            "original_prompt": raw["prompt"],
            "turn": 1,
            "phase": "optimize" if r.get("correctness") else "repair",
            "code": raw.get("parsed_code"),
            "gen_status": r["gen_status"],
            "compiled": bool(r.get("compiled")), "correctness": bool(r.get("correctness")),
            "metadata": r.get("metadata") or {},
            "last_timing": None,
            "best_speedup": None, "best_code": None, "best_turn": None,
            "no_improve_streak": 0,
            "terminated": False, "termination_reason": None,
            "history": [{"turn": 1, "compiled": bool(r.get("compiled")),
                         "correctness": bool(r.get("correctness")), "gen_status": r["gen_status"]}],
        }
        if chains[cid]["correctness"]:
            # best-so-far tracking starts immediately if turn 1 is already correct.
            chains[cid]["best_code"] = raw.get("parsed_code")
            chains[cid]["best_turn"] = 1

    # Turn-1-correct chains start in the optimization phase, which needs
    # last_timing to build the next turn's feedback (#3.4's fixed format).
    # Reuse the P0-a timing run (results/eval/timing_20260820.json) instead
    # of re-measuring -- it already covers every compiled+correct sample
    # from these same two eval files, keyed by the same "path" field.
    timing_path = EVAL_DIR / "timing_20260820.json"
    backfilled, missing = 0, []
    if timing_path.exists():
        timing_by_path = {r["path"]: r for r in json.loads(timing_path.read_text())["records"]
                           if "timing_exception" not in r}
        for r in records:
            if not r.get("correctness"):
                continue
            cid = chain_id(r["language"], r["condition"], r["task"], r["model"], r["sample_index"])
            t = timing_by_path.get(r["path"])
            if t:
                chains[cid]["last_timing"] = {"kernel_ms": t["kernel_ms"]["median"],
                                               "baseline_ms": t["baseline_ms"]["median"],
                                               "speedup": t["speedup"]}
                chains[cid]["best_speedup"] = t["speedup"]
                backfilled += 1
            else:
                missing.append(cid)
    n_optimize = sum(1 for c in chains.values() if c["phase"] == "optimize")
    print(f"[init] backfilled last_timing for {backfilled}/{n_optimize} turn-1-correct chains "
          f"from {timing_path.name}")
    if missing:
        print(f"[init] {len(missing)} turn-1-correct chain(s) have NO timing yet (will be "
              f"measured lazily before their first optimize-phase turn, see cmd_generate): "
              f"{missing}")

    state = {"chains": chains, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k_max": K_MAX}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"[init] wrote {state_path} ({len(chains)} chains, all at turn 1)")
    return 0


# --------------------------------------------------------------------------
# generate: advance active chains for ONE model by one turn (network only)
# --------------------------------------------------------------------------

def _raw_path_for_turn(chain, turn):
    model_dir = chain["model"].replace("/", "_")
    base = RAW_DIR / chain["language"] / chain["condition"] / chain["task"] / model_dir
    if turn == 1:
        return base / f"sample_{chain['sample_index']}.json"
    return base / f"sample_{chain['sample_index']}_turns" / f"turn_{turn}.json"


def _generate_one_turn(chain, client, model, base_url, manifest, spec, env_info,
                        temperature, max_tokens, out_dir):
    next_turn = chain["turn"] + 1
    prompt, phase = build_prompt(chain, spec)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    # Deterministic-but-turn-distinct seed: turn 1's own seed (already
    # logged) isn't reconstructable from chain state alone (not stored), so
    # derive a fresh, reproducible seed from the chain identity + turn
    # number instead -- logged in full below either way (CLAUDE.md rule 4).
    seed = int(hashlib.sha1(f"{chain['chain_id']}|{next_turn}".encode()).hexdigest()[:8], 16) % (2**31)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    record = {
        "chain_id": chain["chain_id"], "turn": next_turn, "phase": phase,
        "task": chain["task"], "language": chain["language"], "condition": chain["condition"],
        "sample_index": chain["sample_index"], "model": model, "base_url": base_url,
        "hf_revision": manifest.get("hf_revision"), "vllm_version": manifest.get("vllm_version"),
        "tensor_parallel_size": manifest.get("tensor_parallel_size"),
        "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
        "timestamp": timestamp, "prompt_sha256": prompt_hash, "prompt": prompt, "env": env_info,
    }
    try:
        response = gen.call_model(client, model, prompt, temperature, max_tokens, seed)
        raw_text = response.choices[0].message.content
        record["response_raw"] = raw_text
        record["response_finish_reason"] = response.choices[0].finish_reason
        record["usage"] = response.usage.model_dump() if response.usage else None
        parsed = gen.extract_first_code_block(raw_text or "")
        record["parsed_code"] = parsed
        record["status"] = "generated" if parsed is not None else "format_failure"
        if record["response_finish_reason"] == "length":
            record["status"] = "truncated"
    except Exception as e:
        record["status"] = "request_error"
        record["error"] = f"{type(e).__name__}: {e}"

    out_path = _raw_path_for_turn(chain, next_turn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str))
    return record


def cmd_generate(args):
    if not args.confirm_run:
        print("[refuse] this makes real requests. Pass --confirm-run.", file=sys.stderr)
        return 1
    state_path = Path(args.state)
    state = json.loads(state_path.read_text())
    chains = state["chains"]

    active = [c for c in chains.values()
              if c["model"] == args.model and not c["terminated"] and "_pending_turn" not in c]
    # Optimize-phase chains need last_timing to build #3.4's fixed feedback
    # format -- if it's still missing (e.g. init's backfill from
    # timing_20260820.json didn't cover this sample, or a prior timing
    # attempt hit the #7-1 reproducible-segfault case), skip this chain for
    # THIS generation round rather than crash the whole batch. GPU access
    # (needed to retry timing) isn't available here -- vLLM is serving on
    # this same GPU during the generate stage -- so the retry happens in
    # cmd_evaluate's catch-up pass instead, before the NEXT generate call.
    blocked = [c for c in active if c["phase"] == "optimize" and c["last_timing"] is None]
    active = [c for c in active if not (c["phase"] == "optimize" and c["last_timing"] is None)]
    if blocked:
        print(f"[generate] {len(blocked)} optimize-phase chain(s) still missing last_timing -- "
              f"skipped this round, run `evaluate` (its catch-up pass) before retrying: "
              f"{[c['chain_id'] for c in blocked][:10]}{' ...' if len(blocked) > 10 else ''}")

    print(f"[generate] model={args.model} turn target={active[0]['turn']+1 if active else '-'} "
          f"active chains={len(active)}")
    if not active:
        print("[generate] nothing to do for this model (all terminated or already pending eval)")
        return 0

    if not gen.check_endpoint_reachable(args.base_url):
        return 1
    manifest = gen.load_manifest(args.manifest)
    env_info = gen.env_fingerprint()
    spec = get_spec()
    temperature = args.temperature if args.temperature is not None else spec.generation_params.get("temperature", 0.8)
    max_tokens = args.max_tokens or spec.generation_params.get("max_output_tokens", 8192)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="not-needed-for-vllm")

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_generate_one_turn, c, client, args.model, args.base_url, manifest,
                                spec, env_info, temperature, max_tokens, RAW_DIR): c for c in active}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                record = fut.result()
                c["_pending_turn"] = record["turn"]
                c["_pending_status"] = record["status"]
                c["_pending_code"] = record.get("parsed_code")
            except Exception as e:
                c["_pending_turn"] = c["turn"] + 1
                c["_pending_status"] = "request_error"
                c["_pending_code"] = None
                c["_pending_error"] = f"{type(e).__name__}: {e}"
            done += 1
            if done % 50 == 0 or done == len(active):
                print(f"[generate] {done}/{len(active)} done")

    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"[generate] wrote {state_path} -- {done} chain(s) now pending evaluation")
    return 0


# --------------------------------------------------------------------------
# evaluate: compile + correctness (+ timing for newly-correct) -- GPU-serial
# --------------------------------------------------------------------------

def _build_eval_record(chain):
    """A record dict shaped like evaluate.py's find_samples()/eval_one()
    expect, from a chain pending evaluation."""
    return {
        "task": chain["task"], "task_family": None, "language": chain["language"],
        "condition": chain["condition"], "model": chain["model"],
        "sample_index": chain["sample_index"], "status": chain["_pending_status"],
        "parsed_code": chain.get("_pending_code"),
    }


def cmd_evaluate(args):
    ev.assert_gpu_exclusive()
    state_path = Path(args.state)
    state = json.loads(state_path.read_text())
    chains = state["chains"]

    # Catch-up pass: any correct chain still missing last_timing (init's
    # backfill gap, or a prior #7-1-style reproducible-timing-failure) gets
    # one more attempt here, GPU-exclusive as required. cmd_generate skips
    # these chains for a round until this succeeds.
    untimed = [c for c in chains.values() if c["correctness"] and c["last_timing"] is None]
    if untimed:
        print(f"[evaluate] catch-up timing for {len(untimed)} correct-but-untimed chain(s)...")
        recovered = 0
        for c in untimed:
            rec = _build_eval_record({**c, "_pending_status": "generated", "_pending_code": c["code"]})
            timing = ev.time_one_isolated(rec, c["code"]) if c["code"] else {"timing_exception": "no code"}
            if "timing_exception" not in timing:
                c["last_timing"] = {"kernel_ms": timing["kernel_ms"]["median"],
                                     "baseline_ms": timing["baseline_ms"]["median"],
                                     "speedup": timing["speedup"]}
                if c["best_speedup"] is None or timing["speedup"] > c["best_speedup"]:
                    c["best_speedup"] = timing["speedup"]
                    c["best_code"] = c["code"]
                recovered += 1
        print(f"[evaluate] catch-up timing: {recovered}/{len(untimed)} recovered "
              f"({len(untimed) - recovered} still unmeasurable, will retry next evaluate call)")
        state_path.write_text(json.dumps(state, indent=2, default=str))

    pending = [c for c in chains.values() if "_pending_turn" in c]
    print(f"[evaluate] {len(pending)} chain(s) pending evaluation")
    if not pending:
        return 0

    cuda_to_precompile = [
        (c["chain_id"], _build_eval_record(c)) for c in pending
        if c["language"] == "cuda" and c["_pending_status"] == "generated"
    ]
    build_dir_map, precompile_failures = {}, {}
    if args.precompile_workers > 1:
        build_dir_map, precompile_failures = ev.precompile_cuda_batch(
            cuda_to_precompile, max_workers=args.precompile_workers)

    done = 0
    for c in pending:
        rec = _build_eval_record(c)
        if c["chain_id"] in precompile_failures:
            result = {"compiled": False, "correctness": False,
                      "metadata": precompile_failures[c["chain_id"]]}
        elif rec["status"] == "generated":
            result = ev.eval_one_isolated(rec, precompiled_build_dir=build_dir_map.get(c["chain_id"]))
        else:
            result = {"compiled": False, "correctness": False, "eval_skipped_reason": rec["status"]}

        turn = c.pop("_pending_turn")
        status = c.pop("_pending_status")
        code = c.pop("_pending_code")
        c.pop("_pending_error", None)

        c["turn"] = turn
        c["gen_status"] = status
        c["code"] = code
        c["compiled"] = bool(result.get("compiled"))
        c["correctness"] = bool(result.get("correctness"))
        c["metadata"] = result.get("metadata") or {}
        c["last_timing"] = None

        if c["correctness"]:
            timing = ev.time_one_isolated(rec, code) if code else {"timing_exception": "no code"}
            if "timing_exception" not in timing:
                c["last_timing"] = {"kernel_ms": timing["kernel_ms"]["median"],
                                     "baseline_ms": timing["baseline_ms"]["median"],
                                     "speedup": timing["speedup"]}
                speedup = timing["speedup"]
                improved = c["best_speedup"] is None or speedup > c["best_speedup"]
                if improved:
                    c["best_speedup"] = speedup
                    c["best_code"] = code
                    c["best_turn"] = turn
                    c["no_improve_streak"] = 0
                else:
                    c["no_improve_streak"] += 1
            else:
                # correct but couldn't time it (rare, e.g. reproducible
                # segfault like #7-1's 2 samples) -- doesn't count as an
                # "improvement" or a "no-improvement" turn either; treated
                # as a neutral turn so the 3-strike counter isn't gamed by
                # a chain that just keeps failing to time out.
                pass
            c["phase"] = "optimize"
        else:
            c["phase"] = "repair"
            c["no_improve_streak"] = 0  # only meaningful within the optimize phase

        c["history"].append({"turn": turn, "compiled": c["compiled"], "correctness": c["correctness"],
                              "gen_status": status,
                              "speedup": c["last_timing"]["speedup"] if c["last_timing"] else None})

        # PROMPT_SPEC #3.4 termination conditions.
        if turn >= state.get("k_max", K_MAX):
            c["terminated"], c["termination_reason"] = True, "k_max_reached"
        elif c["phase"] == "optimize" and c["no_improve_streak"] >= NO_IMPROVE_LIMIT:
            c["terminated"], c["termination_reason"] = True, "no_improvement_3_turns"
        # condition (1) "정답 없이 k 소진" is subsumed by k_max_reached above
        # (a chain that never went correct just keeps re-entering repair
        # until k_max fires) -- no separate check needed.

        done += 1
        if done % 50 == 0 or done == len(pending):
            print(f"[evaluate] {done}/{len(pending)} done")

    state_path.write_text(json.dumps(state, indent=2, default=str))
    n_correct = sum(1 for c in chains.values() if c["correctness"])
    n_terminated = sum(1 for c in chains.values() if c["terminated"])
    print(f"[evaluate] wrote {state_path} -- {n_correct}/{len(chains)} correct, "
          f"{n_terminated}/{len(chains)} terminated")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(args):
    state = json.loads(Path(args.state).read_text())
    chains = list(state["chains"].values())
    print(f"total chains: {len(chains)}")

    import collections
    by_turn_lang = collections.defaultdict(lambda: {"n": 0, "correct": 0, "terminated": 0})
    for c in chains:
        k = (c["turn"], c["language"], c["model"].split("/")[-1])
        d = by_turn_lang[k]
        d["n"] += 1
        d["correct"] += int(c["correctness"])
        d["terminated"] += int(c["terminated"])
    print(f"\n{'turn':4s} {'lang':9s} {'model':28s} {'n':>5s} {'correct':>7s} {'terminated':>10s}")
    for k in sorted(by_turn_lang):
        d = by_turn_lang[k]
        print(f"{k[0]:4d} {k[1]:9s} {k[2]:28s} {d['n']:5d} {d['correct']:7d} {d['terminated']:10d}")

    term_reasons = collections.Counter(c["termination_reason"] for c in chains if c["terminated"])
    print(f"\ntermination reasons: {dict(term_reasons)}")

    speedups = [c["best_speedup"] for c in chains if c["best_speedup"] is not None]
    if speedups:
        import numpy as np
        print(f"\nchains with a best-so-far correct kernel: {len(speedups)}/{len(chains)}, "
              f"best_speedup geomean: {float(np.exp(np.mean(np.log(np.array(speedups))))):.3g}x")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--state", default=str(STATE_DEFAULT))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("generate")
    p.add_argument("--state", default=str(STATE_DEFAULT))
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--confirm-run", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate")
    p.add_argument("--state", default=str(STATE_DEFAULT))
    p.add_argument("--precompile-workers", type=int, default=ev.PRECOMPILE_WORKERS)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report")
    p.add_argument("--state", default=str(STATE_DEFAULT))
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

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

# PI instruction 2026-08-21 (item 4): don't leave these blocked indefinitely,
# retried every single evaluate call's catch-up pass -- terminate them now
# with the documented reason. Both are the SAME two samples #7-1 already
# found reproducibly (not flakily) segfault during timing: 5 attempts across
# P0-a, timeout raised 180->400s, identical crash location (cuCtxSynchronize)
# every time. Excluded from speedup aggregation (last_timing stays None);
# their correctness verdict (True) is untouched.
KNOWN_TIMING_UNMEASURABLE = {
    "cuda|docinject|57_conv_transposed_2D__square_input__square_kernel|openai/gpt-oss-120b|1":
        "reproducible segfault in torch.cuda.synchronize() during timing (paper/"
        "RESULTS_REPORT_20260820.md #7-1): 5 attempts, timeout raised 180->400s, "
        "identical crash location every time -- not flaky, confirmed unmeasurable.",
    "tilelang|docinject|97_ScaledDotProductAttention|openai/gpt-oss-120b|1":
        "reproducible segfault in torch.cuda.synchronize() during timing (paper/"
        "RESULTS_REPORT_20260820.md #7-1): 5 attempts, timeout raised 180->400s, "
        "identical crash location every time -- not flaky, confirmed unmeasurable.",
    # PI decision 2026-08-23 (turn-10 wrap-up, via Claude Code monitoring
    # session): these 14 correctness=True chains never once produced a
    # last_timing across the entire run -- generate() skips untimed chains
    # every round, so they were frozen at whatever turn they last generated
    # at (turn 2-9, see below) while the catch-up-timing pass in cmd_evaluate
    # retried all of them every single round with a monotonically-worsening
    # recovery rate (4/19 -> 3/16 -> 1/14 -> 2/13 -> 0/13 -> 0/13 across
    # 2026-08-23 09:37-11:38 KST) and NONE of these specific 14 ever recovered
    # in any round. Unlike the two entries above, no single reproducible crash
    # signature was captured per-chain here (would require pulling each out
    # for isolated manual reproduction, which was judged not worth delaying
    # the run for) -- termination is on persistent-zero-recovery empirical
    # grounds only. Without this, run_all_turns.sh would keep re-serving both
    # models every ~15-25 min for these stragglers alone until the 2026-08-25
    # 06:00 KST cutoff. Excluded from speedup aggregation (last_timing stays
    # None); correctness verdict (True) is untouched.
    "tilelang|0shot|36_RMSNorm_|openai/gpt-oss-120b|4":
        "never recovered timing in any catch-up round; frozen at turn 3. See block comment above.",
    "triton|0shot|35_GroupNorm_|openai/gpt-oss-120b|2":
        "never recovered timing in any catch-up round; frozen at turn 5. See block comment above.",
    "triton|0shot|41_Max_Pooling_1D|Qwen/Qwen3-Coder-30B-A3B-Instruct|1":
        "never recovered timing in any catch-up round; frozen at turn 9. See block comment above.",
    # CORRECTION (2026-08-23, post-final_retiming.py cross-check, paper/FINAL_REPORT.md
    # #3): this chain's *latest-turn* catch-up attempts never recovered (matching the
    # block comment above), but its earlier turn's best_code/best_speedup HAD already
    # been banked from a prior successful timing -- final_retiming.py retimed that
    # banked best_code successfully (speedup 0.0197). So "never recovered" was accurate
    # for the turn this chain was stuck on, but imprecise as a claim about the chain's
    # entire history: no data was lost by terminating it (best_code was already the
    # historical best and is unaffected by termination), but call it "re-verify latest
    # turn only, historical best already banked" rather than "never measurable ever".
    # Same correction applies to the 3 entries below marked the same way.
    "triton|0shot|41_Max_Pooling_1D|Qwen/Qwen3-Coder-30B-A3B-Instruct|2":
        "latest-turn catch-up never recovered (frozen at turn 6), but an earlier turn's "
        "best_code was already banked and retimed successfully in final_retiming.py "
        "(speedup 0.0197) -- no data lost. See 2026-08-23 CORRECTION comment above.",
    "triton|0shot|57_conv_transposed_2D__square_input__square_kernel|openai/gpt-oss-120b|1":
        "never recovered timing in any catch-up round; frozen at turn 6. See block comment above.",
    "cuda|docinject|57_conv_transposed_2D__square_input__square_kernel|openai/gpt-oss-120b|2":
        "never recovered timing in any catch-up round; frozen at turn 7. See block comment above.",
    "cuda|docinject|57_conv_transposed_2D__square_input__square_kernel|openai/gpt-oss-120b|3":
        "never recovered timing in any catch-up round; frozen at turn 9. See block comment above.",
    "tilelang|docinject|35_GroupNorm_|openai/gpt-oss-120b|0":
        "never recovered timing in any catch-up round; frozen at turn 2. See block comment above.",
    "tilelang|docinject|35_GroupNorm_|openai/gpt-oss-120b|3":
        "never recovered timing in any catch-up round; frozen at turn 3. See block comment above.",
    "tilelang|docinject|40_LayerNorm|openai/gpt-oss-120b|0":
        "never recovered timing in any catch-up round; frozen at turn 3. See block comment above.",
    "tilelang|docinject|42_Max_Pooling_2D|openai/gpt-oss-120b|3":
        "never recovered timing in any catch-up round; frozen at turn 8. See block comment above.",
    "tilelang|docinject|47_Sum_reduction_over_a_dimension|openai/gpt-oss-120b|3":
        "latest-turn catch-up never recovered (frozen at turn 6), but an earlier turn's "
        "best_code was already banked and retimed successfully in final_retiming.py "
        "(speedup 0.1172) -- no data lost. See 2026-08-23 CORRECTION comment above.",
    "triton|docinject|42_Max_Pooling_2D|openai/gpt-oss-120b|1":
        "latest-turn catch-up never recovered (frozen at turn 8), but an earlier turn's "
        "best_code was already banked and retimed successfully in final_retiming.py "
        "(speedup 2.0943) -- no data lost. See 2026-08-23 CORRECTION comment above.",
    "triton|docinject|42_Max_Pooling_2D|openai/gpt-oss-120b|2":
        "latest-turn catch-up never recovered (frozen at turn 7), but an earlier turn's "
        "best_code was already banked and retimed successfully in final_retiming.py "
        "(speedup 2.3716) -- no data lost. See 2026-08-23 CORRECTION comment above.",
}


def chain_id(language, condition, task, model, sample_index):
    return f"{language}|{condition}|{task}|{model}|{sample_index}"


def _feedback_for(chain, spec):
    """One of #3.4's 5 fixed templates, chosen by the PREVIOUS turn's
    outcome. Only the aggregate stats go in for correctness failures (no
    reference tensor values, PROMPT_SPEC #3.4)."""
    gs = chain["gen_status"]
    # "request_error" (generate()'s vLLM call itself raised -- network/API
    # exception, e.g. a crash mid-request) means chain["code"] is None, same
    # as truncated/format_failure: no previous solution exists at all. Found
    # 2026-08-21: routed through the generic not-compiled branch below
    # instead, which built a fake "failed to compile with error: {}" from
    # the empty metadata dict -- misrepresenting a dead generation request
    # as a compile failure. All 164 of the turn-4 reboot-storm chains hit
    # this (see 301bc04); this fix only prevents recurrence in later turns.
    if gs in ("truncated", "format_failure", "request_error"):
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

    # PI item 4 (2026-08-21): terminate the 2 known reproducibly-unmeasurable
    # chains outright instead of retrying every turn's catch-up pass forever.
    for cid, reason in KNOWN_TIMING_UNMEASURABLE.items():
        c = chains.get(cid)
        if c and not c["terminated"]:
            c["terminated"], c["termination_reason"] = True, "timing_unmeasurable"
            c["termination_detail"] = reason
            print(f"[evaluate] terminating known-unmeasurable chain {cid}: {reason[:80]}...")

    # Catch-up pass: any ACTIVE correct chain still missing last_timing
    # (init's backfill gap) gets one more attempt here, GPU-exclusive as
    # required. cmd_generate skips these chains for a round until this
    # succeeds. Terminated chains (incl. the ones just above) are excluded
    # -- no more retries for them.
    untimed = [c for c in chains.values()
               if c["correctness"] and c["last_timing"] is None and not c["terminated"]]
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
        # result's diagnostic text isn't always under "metadata" -- e.g.
        # eval_one_isolated's no-result fallback (subprocess crash/timeout,
        # evaluate.py's _eval_worker_entry docstring) puts it under
        # "eval_exception" instead. `or {}` used to silently discard that
        # (found 2026-08-21: 164 turn-4 chains recorded compiled=False with
        # completely empty metadata, losing exactly the info needed to tell
        # "genuinely buggy kernel" apart from "eval subprocess crashed for
        # an environment reason" -- can't recover the already-lost ones, but
        # keep it for every turn from here on).
        c["metadata"] = result.get("metadata") or {
            k: v for k, v in result.items() if k not in ("compiled", "correctness")
        }
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

def _hist_turn(c, t):
    for h in c["history"]:
        if h["turn"] == t:
            return h
    return None


def _geomean(values):
    import numpy as np
    arr = np.array(values, dtype=float)
    return float(np.exp(np.mean(np.log(arr))))


def _turn_speedup(chain, h):
    """A history entry's speedup for a turn, if it was correct and timed.
    cmd_init's turn-1 history entries never got a "speedup" key at all
    (unlike cmd_evaluate's turn>=2 entries, which always include it, even
    as None) -- turn 1's own timing, when it succeeded, only lives on the
    chain's best_speedup/best_turn fields (backfilled from
    results/eval/timing_20260820.json). best_turn==1 uniquely identifies
    "turn 1's own timing succeeded and became the (necessarily first-ever)
    best", so best_speedup IS turn 1's speedup in that case. Same fix as
    scripts/analyze.py's fig2_speedup_rows() (2026-08-24) -- keep both in
    sync since PRIMARY(b) below and fig2_speedup.csv are meant to agree."""
    if h["turn"] == 1 and "speedup" not in h:
        return chain["best_speedup"] if chain.get("best_turn") == 1 else None
    return h.get("speedup")


def cmd_report(args):
    import collections
    import numpy as np

    state = json.loads(Path(args.state).read_text())
    chains = list(state["chains"].values())
    max_turn = max((h["turn"] for c in chains for h in c["history"]), default=1)
    print(f"total chains: {len(chains)}  |  turns present: 1..{max_turn}")

    # ------------------------------------------------------------------
    # PI item 1 (2026-08-21): primary curves are ever-correct@turn and
    # best-speedup@turn (both monotonic by construction) -- NOT the
    # point-in-time correctness table, which the optimization phase can
    # push down (see the turn1->2 regression finding). Point-in-time stays
    # as an auxiliary table below.
    # ------------------------------------------------------------------
    print("\n=== PRIMARY (a) ever-correct@turn -- cumulative, monotonic ===")
    print(f"{'turn':4s} {'lang':9s} {'model':28s} {'ever_correct':>12s} {'/n':>6s}")
    keys = sorted({(c["language"], c["model"].split("/")[-1]) for c in chains})
    for t in range(1, max_turn + 1):
        for lang, model in keys:
            group = [c for c in chains if c["language"] == lang and c["model"].split("/")[-1] == model]
            if not group:
                continue
            # NOT c["best_turn"] -- that's only set when a correct turn's
            # timing ALSO succeeded (see cmd_evaluate), so a chain that was
            # correct but hit a timing exception (the #7-1-style reproducible
            # segfault cases) would silently never count as ever-correct.
            # Recompute directly from history's correctness bools instead.
            ever = sum(1 for c in group
                       if any(h["correctness"] for h in c["history"] if h["turn"] <= t))
            print(f"{t:4d} {lang:9s} {model:28s} {ever:12d} {'/' + str(len(group)):>6s}")

    # ------------------------------------------------------------------
    # PI request 2026-08-22 (post turn-8 report): ever-correct@turn split
    # by workstream (0shot vs docinject) rather than pooled across both --
    # this is the evidence base for Figure 1's condition-split lines and
    # the revised §5 sentence, so print counts AND the ratio against each
    # condition's own denominator (0shot n=160, docinject n=85 per
    # lang x model -- NOT 245; pooling them under /245 would understate
    # docinject's rate since it's the smaller subset).
    # ------------------------------------------------------------------
    print("\n=== PRIMARY (a'): ever-correct@turn split by workstream (0shot vs docinject) ===")
    print(f"{'turn':4s} {'lang':9s} {'model':28s} {'cond':10s} {'ever_correct':>12s} {'/n':>6s} {'ratio':>7s}")
    for t in range(1, max_turn + 1):
        for lang, model in keys:
            for cond in ("0shot", "docinject"):
                group = [c for c in chains if c["language"] == lang
                         and c["model"].split("/")[-1] == model and c["condition"] == cond]
                if not group:
                    continue
                ever = sum(1 for c in group
                           if any(h["correctness"] for h in c["history"] if h["turn"] <= t))
                ratio = ever / len(group)
                print(f"{t:4d} {lang:9s} {model:28s} {cond:10s} {ever:12d} "
                      f"{'/' + str(len(group)):>6s} {ratio:7.1%}")

    print("\n=== PRIMARY (b) best-speedup@turn -- geomean of best-so-far, monotonic ===")
    print(f"{'turn':4s} {'lang':9s} {'model':28s} {'n(timed)':>9s} {'geomean':>9s}")
    for t in range(1, max_turn + 1):
        for lang, model in keys:
            group = [c for c in chains if c["language"] == lang and c["model"].split("/")[-1] == model]
            best_so_far = []
            for c in group:
                vals = [_turn_speedup(c, h) for h in c["history"] if h["turn"] <= t and h["correctness"]]
                vals = [v for v in vals if v is not None]
                if vals:
                    best_so_far.append(max(vals))
            if best_so_far:
                print(f"{t:4d} {lang:9s} {model:28s} {len(best_so_far):9d} {_geomean(best_so_far):9.3g}")

    # ------------------------------------------------------------------
    # Auxiliary: point-in-time correctness (can fall due to optimize-phase
    # regressions -- do not use as the headline number, see PRIMARY above).
    # ------------------------------------------------------------------
    print("\n=== AUXILIARY: point-in-time correctness@turn (NOT monotonic -- see PRIMARY) ===")
    by_turn_lang = collections.defaultdict(lambda: {"n": 0, "correct": 0, "terminated": 0})
    for c in chains:
        k = (c["turn"], c["language"], c["model"].split("/")[-1])
        d = by_turn_lang[k]
        d["n"] += 1
        d["correct"] += int(c["correctness"])
        d["terminated"] += int(c["terminated"])
    print(f"{'turn':4s} {'lang':9s} {'model':28s} {'n':>5s} {'correct':>7s} {'terminated':>10s}")
    for k in sorted(by_turn_lang):
        d = by_turn_lang[k]
        print(f"{k[0]:4d} {k[1]:9s} {k[2]:28s} {d['n']:5d} {d['correct']:7d} {d['terminated']:10d}")

    # ------------------------------------------------------------------
    # Denominator caveat: any compile/correctness rate above is computed
    # over ALL chains at that turn, including ones whose generation was
    # truncated (max_tokens cut) or unparseable (format_failure) -- those
    # never had a chance to compile at all. PTX is the language where this
    # denominator inflation is largest (see #3.4 pilot notes), so call it
    # out explicitly per turn/lang rather than let a reader silently divide
    # a compile rate by a denominator that includes unparseable samples.
    # ------------------------------------------------------------------
    print("\n=== denominator caveat: truncated/format_failure generations by turn x lang "
          "(these count in the 'n' above but never had a chance to compile) ===")
    by_turn_lang_bad = collections.defaultdict(int)
    for c in chains:
        for h in c["history"]:
            if h.get("gen_status") in ("truncated", "format_failure"):
                by_turn_lang_bad[(h["turn"], c["language"])] += 1
    for k in sorted(by_turn_lang_bad):
        t, lang = k
        n_lang_turn = sum(1 for c in chains if c["language"] == lang
                           and any(h["turn"] == t for h in c["history"]))
        print(f"turn {t:2d} {lang:9s}: {by_turn_lang_bad[k]:4d}/{n_lang_turn} "
              f"unparseable (truncated or format_failure)")

    # ------------------------------------------------------------------
    # PI item 1: per-turn transition 4-class + running cumulative totals.
    # ------------------------------------------------------------------
    print("\n=== transition 4-class per turn-step (FF/FT/TF/TT) + cumulative ===")
    cum = collections.Counter()
    for t in range(2, max_turn + 1):
        step = collections.Counter()
        for c in chains:
            h1, h2 = _hist_turn(c, t - 1), _hist_turn(c, t)
            if h1 is None or h2 is None:
                continue
            step[(h1["correctness"], h2["correctness"])] += 1
        cum.update(step)
        print(f"turn {t-1}->{t}: FF={step[(False,False)]:4d} FT={step[(False,True)]:4d} "
              f"TF={step[(True,False)]:4d} TT={step[(True,True)]:4d}  |  "
              f"cumulative: FT={cum[(False,True)]:4d} TF={cum[(True,False)]:4d} "
              f"net={cum[(False,True)]-cum[(True,False)]:+d}")

    # ------------------------------------------------------------------
    # PI item 2: 0-shot vs docinject workstream split, latest transition +
    # turn-1 baseline (answers: is a given language/model's gain workstream-
    # specific -- feedback-alone vs doc-injection x feedback interaction?).
    # ------------------------------------------------------------------
    if max_turn >= 2:
        print(f"\n=== workstream split (0shot vs docinject), turn 1 baseline and turn {max_turn-1}->{max_turn} transition ===")
        print(f"{'lang':9s} {'model':28s} {'cond':10s} {'t1_correct':>10s} {'FT(new)':>8s} {'TF(regr)':>9s}")
        for lang, model in keys:
            for cond in ("0shot", "docinject"):
                group = [c for c in chains if c["language"] == lang
                         and c["model"].split("/")[-1] == model and c["condition"] == cond]
                if not group:
                    continue
                t1_correct = sum(1 for c in group if (_hist_turn(c, 1) or {}).get("correctness"))
                ft = tf = 0
                for c in group:
                    h1, h2 = _hist_turn(c, max_turn - 1), _hist_turn(c, max_turn)
                    if h1 is None or h2 is None:
                        continue
                    if not h1["correctness"] and h2["correctness"]:
                        ft += 1
                    elif h1["correctness"] and not h2["correctness"]:
                        tf += 1
                print(f"{lang:9s} {model:28s} {cond:10s} {t1_correct:10d} {ft:8d} {tf:9d}")

        # PI request 2026-08-21 (post turn-5 report): the workstream split's
        # decisive finding was in the CUMULATIVE picture (e.g. TileLang
        # recovery concentrated almost entirely in docinject across the
        # whole run), not any single turn-step -- add it explicitly rather
        # than making a reader sum turn-steps by hand.
        print(f"\n=== workstream split (0shot vs docinject), CUMULATIVE FT/TF summed over all "
              f"turn-steps 1->2 .. {max_turn-1}->{max_turn} ===")
        print(f"{'lang':9s} {'model':28s} {'cond':10s} {'FT(cum)':>8s} {'TF(cum)':>8s} {'net':>6s}")
        for lang, model in keys:
            for cond in ("0shot", "docinject"):
                group = [c for c in chains if c["language"] == lang
                         and c["model"].split("/")[-1] == model and c["condition"] == cond]
                if not group:
                    continue
                ft_cum = tf_cum = 0
                for c in group:
                    for t in range(2, max_turn + 1):
                        h1, h2 = _hist_turn(c, t - 1), _hist_turn(c, t)
                        if h1 is None or h2 is None:
                            continue
                        if not h1["correctness"] and h2["correctness"]:
                            ft_cum += 1
                        elif h1["correctness"] and not h2["correctness"]:
                            tf_cum += 1
                print(f"{lang:9s} {model:28s} {cond:10s} {ft_cum:8d} {tf_cum:8d} {ft_cum-tf_cum:+6d}")

    # ------------------------------------------------------------------
    # PI item 3: oscillation tracking (correctness sign-flips across the
    # full history so far). A chain needs >=3 turns of history to show a
    # completed oscillation (correct->incorrect->correct or the reverse);
    # with max_turn < 3 this will read all zero, which is expected, not a bug.
    # ------------------------------------------------------------------
    print(f"\n=== oscillation tracking (correctness sign-flips in history, informational only -- no intervention) ===")
    osc_counts = []
    for c in chains:
        seq = [h["correctness"] for h in sorted(c["history"], key=lambda h: h["turn"])]
        flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        osc_counts.append(flips)
    osc_hist = collections.Counter(osc_counts)
    print(f"flip-count distribution across all {len(chains)} chains: {dict(sorted(osc_hist.items()))}")
    oscillating = sum(1 for f in osc_counts if f >= 2)  # >=2 flips = at least one full back-and-forth
    print(f"chains with >=2 sign-flips (at least one completed oscillation): {oscillating}")

    # ------------------------------------------------------------------
    term_reasons = collections.Counter(c["termination_reason"] for c in chains if c["terminated"])
    print(f"\ntermination reasons: {dict(term_reasons)}")

    speedups = [c["best_speedup"] for c in chains if c["best_speedup"] is not None]
    if speedups:
        print(f"\nchains with a best-so-far correct kernel: {len(speedups)}/{len(chains)}, "
              f"best_speedup geomean: {_geomean(speedups):.3g}x")
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

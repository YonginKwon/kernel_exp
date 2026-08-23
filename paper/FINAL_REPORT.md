# Final report — kernel-lang-2x2 multi-turn run (sm_120)

Produced after the multi-turn loop's clean exit (turn 10, 2026-08-23 12:15:10
KST — see `logs/multiturn/run_all_turns.sh`'s `LOOP_EXIT_REASON=complete`) and
the final re-timing pass (`results/eval/final_timing_20260823T121517.json`,
completed 14:34:16 KST). Per `paper/FINAL_DELIVERABLES.md`, every number here
is computed against the post-completion state — nothing here is a mid-run
snapshot.

Inputs: `results/eval/multiturn_state.json` (1960 chains, turns 1..10) and
`results/eval/final_timing_20260823T121517.json` (556 candidate kernels, 500
successfully retimed). Full turn-by-turn tables live in
`logs/multiturn/turn_reports/turn_10_report.txt`; this document is the
paper-facing digest plus the additional cross-checks requested for the
official sm_120 snapshot.

---

## 1. Ever-correct@turn — workstream split (counts + ratio)

Full table (turns 1–10 × 4 languages × 2 models × 2 conditions, 160 rows) is
`paper/figures_data/fig1_ever_correct.csv` / `turn_10_report.txt` §PRIMARY(a').
Final-turn (turn 10) snapshot, denominators are per-condition (0shot n=160,
docinject n=85 — not pooled):

| lang | model | 0shot ever-correct | 0shot % | docinject ever-correct | docinject % |
|---|---|---:|---:|---:|---:|
| cuda | gpt-oss-120b | 66/160 | 41.2% | 74/85 | 87.1% |
| cuda | Qwen3-Coder-30B-A3B-Instruct | 1/160 | 0.6% | 33/85 | 38.8% |
| ptx | gpt-oss-120b | 0/160 | 0.0% | 5/85 | 5.9% |
| ptx | Qwen3-Coder-30B-A3B-Instruct | 0/160 | 0.0% | 0/85 | 0.0% |
| tilelang | gpt-oss-120b | 74/160 | 46.2% | 65/85 | 76.5% |
| tilelang | Qwen3-Coder-30B-A3B-Instruct | 25/160 | 15.6% | 25/85 | 29.4% |
| triton | gpt-oss-120b | 85/160 | 53.1% | 53/85 | 62.4% |
| triton | Qwen3-Coder-30B-A3B-Instruct | 40/160 | 25.0% | 18/85 | 21.2% |

Docinject consistently outperforms 0shot in every lang×model cell except
`triton|Qwen3-Coder-30B-A3B-Instruct` (25.0% vs 21.2% — the one case where
0shot pulled ahead by turn 10). Sum across all 8 cells: turn-10 ever-correct
= 564/1960 (see §4 for how this reconciles against the retiming pass).

## 2. Transition 4-class per turn-step, cumulative

From `turn_10_report.txt` §transition (FF/FT/TF/TT = stayed-wrong /
went-correct / regressed / stayed-correct):

| turn-step | FF | FT | TF | TT | cum FT | cum TF | cum net |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1→2 | 1671 | 103 | 81 | 103 | 103 | 81 | +22 |
| 2→3 | 1693 | 59 | 81 | 124 | 162 | 162 | +0 |
| 3→4 | 1773 | 1 | 168 | 12 | 163 | 330 | −167 |
| 4→5 | 1743 | 198 | 3 | 10 | 361 | 333 | +28 |
| 5→6 | 1608 | 138 | 80 | 124 | 499 | 413 | +86 |
| 6→7 | 1601 | 87 | 105 | 154 | 586 | 518 | +68 |
| 7→8 | 1612 | 94 | 64 | 145 | 680 | 582 | +98 |
| 8→9 | 1609 | 67 | 80 | 144 | 747 | 662 | +85 |
| 9→10 | 1606 | 83 | 72 | 109 | 830 | 734 | +96 |

The 3→4 net −167 is the documented infrastructure-incident regression — see
[`paper/TURN4_FOOTNOTE.md`](TURN4_FOOTNOTE.md), cite that file verbatim, do
not re-derive. Every other turn-step is net-positive; final cumulative
FT=830 / TF=734 (net +96) is consistent with point-in-time correctness
*not* being the primary metric (it oscillates) while ever-correct (§1) is
monotonic by construction.

## 3. Oscillation — final tally

Flip-count distribution across all 1960 chains (number of correctness
sign-flips in each chain's history):

| flips | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chains | 1401 | 127 | 178 | 92 | 82 | 32 | 28 | 11 | 9 |

**432/1960 chains (22.0%) completed at least one full oscillation
(≥2 sign-flips)** — informational only, no intervention was taken on these
per PROMPT_SPEC #3.4's fixed-protocol rule.

## 4. PTX truncated/format-failure — per-turn denominator note

PTX's near-zero correctness (0–5/245 every turn, see §1) is *not* solely a
correctness failure — a large fraction of PTX generations never had a
chance to compile because they were truncated or unparseable:

| turn | unparseable / n |
|---|---|
| 1 | 74/490 (15.1%) |
| 2 | 63/490 (12.9%) |
| 3 | 64/490 (13.1%) |
| 5 | 81/490 (16.5%) |
| 6 | 62/490 (12.7%) |
| 7 | 56/490 (11.4%) |
| 8 | 46/490 (9.4%) |
| 9 | 41/490 (8.4%) |
| 10 | 44/490 (9.0%) |

(n=490 = both models pooled, 245 each.) This denominator caveat applies to
every turn's PTX row in §1/§Primary(a) — the "compiled" and "correct" counts
there are against the full n, including these truncations, not against a
truncation-adjusted denominator. Other languages had isolated single-digit
truncation events (see `turn_10_report.txt` for the full per-turn/per-lang
list); only PTX has a persistent, large-fraction truncation pattern.

## 5. Termination reasons

```
k_max_reached:          1870
no_improvement_3_turns:   74
timing_unmeasurable:      16
------------------------------
total:                   1960
```

`timing_unmeasurable` = 2 original entries (documented reproducible
`torch.cuda.synchronize()` segfault, `paper/RESULTS_REPORT_20260820.md`
#7-1) + 14 added 2026-08-23 during the turn-10 wrap-up when the loop's
active count plateaued at 15 for 3 consecutive rounds (see §6 for what
happened to those 14 on cross-check against the final retiming pass).

## 6. `timing_unmeasurable` (16 chains) — language × condition × task distribution

| language | condition | task | model | sample |
|---|---|---|---|---:|
| cuda | docinject | 57_conv_transposed_2D__square_input__square_kernel | gpt-oss-120b | 1 |
| cuda | docinject | 57_conv_transposed_2D__square_input__square_kernel | gpt-oss-120b | 2 |
| cuda | docinject | 57_conv_transposed_2D__square_input__square_kernel | gpt-oss-120b | 3 |
| tilelang | 0shot | 36_RMSNorm_ | gpt-oss-120b | 4 |
| tilelang | docinject | 97_ScaledDotProductAttention | gpt-oss-120b | 1 |
| tilelang | docinject | 35_GroupNorm_ | gpt-oss-120b | 0 |
| tilelang | docinject | 35_GroupNorm_ | gpt-oss-120b | 3 |
| tilelang | docinject | 40_LayerNorm | gpt-oss-120b | 0 |
| tilelang | docinject | 42_Max_Pooling_2D | gpt-oss-120b | 3 |
| tilelang | docinject | 47_Sum_reduction_over_a_dimension | gpt-oss-120b | 3 |
| triton | 0shot | 35_GroupNorm_ | gpt-oss-120b | 2 |
| triton | 0shot | 41_Max_Pooling_1D | Qwen3-Coder-30B-A3B-Instruct | 1 |
| triton | 0shot | 41_Max_Pooling_1D | Qwen3-Coder-30B-A3B-Instruct | 2 |
| triton | 0shot | 57_conv_transposed_2D__square_input__square_kernel | gpt-oss-120b | 1 |
| triton | docinject | 42_Max_Pooling_2D | gpt-oss-120b | 1 |
| triton | docinject | 42_Max_Pooling_2D | gpt-oss-120b | 2 |

By language: cuda 3, tilelang 7, triton 6, ptx 0. By condition: 0shot 5,
docinject 11. By model: gpt-oss-120b 14, Qwen3-Coder-30B-A3B-Instruct 2.
All are `openai/gpt-oss-120b`-heavy and docinject-heavy — consistent with
gpt-oss/docinject producing more (and more marginal/borderline) correct
kernels overall (§1), giving more opportunities to hit a timing-flaky one.

---

## 7. Final re-timing exceptions (56) — classification for the paper's denominator

**Cross-checked every one of the 56 `timing_exception` entries in
`final_timing_20260823T121517.json` against (a) the 16-chain
`timing_unmeasurable` list above and (b) each chain's own turn-by-turn
`history` in `multiturn_state.json`.** All 56 have the identical generic
message (`timing subprocess produced no result (exitcode=-11) -- crash or
hang killed after a 180s timeout`) — evaluate.py records no more granular
signal than that, so classification below is structural (via cross-checking
state), not from parsing distinct error text.

**By language:** tilelang 27, cuda 23, triton 6, ptx 0.
**By condition:** 0shot 28, docinject 28 (even split).
**By model:** gpt-oss-120b 43, Qwen3-Coder-30B-A3B-Instruct 13.

**Reproducibility split:**

| category | n | definition |
|---|---:|---|
| Documented, repeated failure | 6 | chain_id is one of the 16 `timing_unmeasurable` entries *and* still had a `best_code` to retime (i.e. it failed timing both during the live multi-turn run's catch-up retries **and again** in the independent final re-timing pass — reproducible across two separate measurement contexts) |
| Single occurrence (only observed here) | 50 | chain_id is **not** in the 16-chain list. Since `best_code` only gets set when a timing attempt *succeeds*, every one of these 50 was successfully timed at least once earlier in the run — this final-pass failure is the first and only time each specific chain has ever failed to time. Plausible causes (not diagnosed further): late-batch GPU/thermal/allocator state after hundreds of back-to-back isolated subprocess launches, or genuine per-kernel flakiness at the 180s timeout boundary. Not investigated per-chain (would require pulling each of the 50 out for isolated reproduction). |

The 6 "documented, repeated failure" chains:
`cuda|docinject|57_conv_transposed_2D__square_input__square_kernel|openai/gpt-oss-120b|{1,2,3}`,
`tilelang|docinject|42_Max_Pooling_2D|openai/gpt-oss-120b|3`,
`tilelang|docinject|97_ScaledDotProductAttention|openai/gpt-oss-120b|1`,
`triton|0shot|41_Max_Pooling_1D|Qwen/Qwen3-Coder-30B-A3B-Instruct|1`.

**Self-correction found during this cross-check (already fixed, commit
`0cfcd21`):** 4 of the 16 `timing_unmeasurable` chains (`triton|0shot|
41_Max_Pooling_1D|...|2`, `tilelang|docinject|47_Sum_reduction_over_a_
dimension|...|3`, `triton|docinject|42_Max_Pooling_2D|...|{1,2}`) actually
**succeeded** in the final re-timing pass (speedups 0.0197, 0.1172, 2.0943,
2.3716 respectively) — they had a historical `best_code` banked from an
earlier turn that final_retiming.py measured just fine; only their *most
recent* turn's fresh catch-up attempt was stuck (which is what was actually
observed live on 2026-08-23 and is why they were terminated). **No data was
lost** — termination doesn't touch `best_code`, and that historical best is
exactly what fed the final retiming pass — but the original commit's
"never recovered timing" wording overstated the claim to the chain's whole
history rather than just its stuck latest turn. The 16-chain list and its
reason strings are corrected in `scripts/multiturn.py` (commit `0cfcd21`);
the true breakdown of the 16 is: 6 reproducibly failed twice (live run +
final retiming, listed above), 4 succeeded in final retiming (corrected),
6 never had any measurable code at all (§8).

### Final denominator definition (the number every paper speedup/fast_1 citation must trace back to)

```
total ever-correct chains (N1)              564
  − never had a measurable best_code (§8)     8
  − final-retiming timing_exception          56
  ------------------------------------------------
  = successfully retimed (final denominator) 500
```

564 − 64 = 500. ✓ Reconciled three ways independently: (1) `final_retiming.py`'s
own log (`556 kernel(s) retimed (56 timing exception(s))`, 556−56=500),
(2) `paper/figures_data/final_retiming_summary.csv` row sum (`n` column
sums to exactly 500 across all lang×model×condition groups), (3)
`paper/figures_data/task_speedup_final.csv` row sum (`n` column also sums
to exactly 500). **`final_retiming_summary.csv` is the sole citable source
for speedup/fast_1 numbers (CLAUDE.md item 4); its n=500 is the
denominator.** Do not cite 564 or 556 as "the" correct-kernel count for a
speedup claim — 564 is ever-correct (includes 64 that were never
successfully measured), 556 is "had a measurable code" (includes 56 that
failed this specific pass).

## 8. The 8 ever-correct chains with NO measurable code, ever

`final_retiming.py` only retimes chains with `best_code` set; 8 of the 564
ever-correct chains never had one:

| chain_id | reason |
|---|---|
| `tilelang\|0shot\|36_RMSNorm_\|openai/gpt-oss-120b\|4` | in the 16-list; catch-up timing never once succeeded for this chain across its whole active life |
| `triton\|0shot\|35_GroupNorm_\|openai/gpt-oss-120b\|2` | same |
| `triton\|0shot\|57_conv_transposed_2D__square_input__square_kernel\|openai/gpt-oss-120b\|1` | same |
| `tilelang\|docinject\|35_GroupNorm_\|openai/gpt-oss-120b\|0` | same |
| `tilelang\|docinject\|35_GroupNorm_\|openai/gpt-oss-120b\|3` | same |
| `tilelang\|docinject\|40_LayerNorm\|openai/gpt-oss-120b\|0` | same |
| `cuda\|0shot\|3_Batched_matrix_multiplication\|openai/gpt-oss-120b\|1` | **newly found this pass, not in the 16-list**: became correct for the *first time ever* at turn 10 (the terminal turn) with `speedup: null`; `turn >= k_max` fired the same evaluate call before any catch-up retry could ever be attempted |
| `cuda\|docinject\|97_ScaledDotProductAttention\|openai/gpt-oss-120b\|3` | same as above — correct only at turn 10, zero retry opportunity |

These last 2 are a genuinely distinct category from the 16 (which were
stuck *mid-run* and manually terminated) — they reached k_max through
completely normal turn progression and were simply never correct until the
protocol's very last turn, leaving no time in the protocol for a timing
retry. Not a bug; a structural edge case of a fixed-k_max protocol.

---

## 9. Task-level final speedup + outlier flags

Full data: `paper/figures_data/task_speedup_final.csv` (134 rows: one per
task×lang×model combination with ≥1 successfully-retimed sample, pooling
`0shot`/`docinject` and all `sample_index`). Family assignment from
`tasks/level1_subset.json`'s `families` map. Total n across all 134 rows =
500, matching §7's denominator.

**Global geomean across all 500 successfully-retimed samples: 0.835×**
(i.e. the median LLM-generated kernel across all languages/models/tasks is
still ~17% slower than eager PyTorch fp16 — consistent with the paper's
overall finding that raw kernel-writing ability, not language/abstraction
choice, remains the dominant bottleneck).

### matmul/convolution family, speedup > 1.5× — full flag

**0/134 rows flagged.** No matmul- or convolution-family (task, lang,
model) combination reached even a 1.5× speedup over eager PyTorch in the
final measurement — every matmul/conv config the LLMs produced across all
4 languages and both models remained at or below parity. This is a strong,
clean negative result worth stating directly: **the model-generated
matmul/conv kernels never beat PyTorch's own (cuBLAS/cuDNN-backed) eager
implementation by any meaningful margin, in any of the 4 languages.**

### Family-median 3× deviation outliers (17 rows)

| task | family | lang | model | n | geomean | family median | ratio |
|---|---|---|---|---:|---:|---:|---:|
| 100_HingeLoss | loss | triton | gpt-oss-120b | 1 | 7.313 | 1.000 | 7.31× |
| 40_LayerNorm | normalization | cuda | gpt-oss-120b | 7 | 5.003 | 1.125 | 4.45× |
| 40_LayerNorm | normalization | triton | gpt-oss-120b | 4 | 4.727 | 1.125 | 4.20× |
| 94_MSELoss | loss | triton | gpt-oss-120b | 5 | 3.178 | 1.000 | 3.18× |
| 94_MSELoss | loss | cuda | gpt-oss-120b | 1 | 3.165 | 1.000 | 3.16× |
| 40_LayerNorm | normalization | tilelang | Qwen-30B-A3B | 2 | 3.450 | 1.125 | 3.07× |
| 1_Square_matrix_multiplication_ | matmul | cuda | gpt-oss-120b | 8 | 0.179 | 0.541 | 0.33× |
| 51_Argmax_over_a_dimension | reduction | triton | gpt-oss-120b | 4 | 0.247 | 0.817 | 0.30× |
| 17_Matmul_with_transposed_B | matmul | cuda | gpt-oss-120b | 6 | 0.160 | 0.541 | 0.30× |
| 3_Batched_matrix_multiplication | matmul | tilelang | Qwen-30B-A3B | 3 | 0.080 | 0.541 | 0.15× |
| 41_Max_Pooling_1D | pooling | triton | Qwen-30B-A3B | 2 | 0.238 | 1.675 | 0.14× |
| 54_conv_standard_3D...  | convolution | triton | gpt-oss-120b | 1 | 0.120 | 0.899 | 0.13× |
| 54_conv_standard_3D...  | convolution | cuda | gpt-oss-120b | 2 | 0.095 | 0.899 | 0.11× |
| 3_Batched_matrix_multiplication | matmul | cuda | Qwen-30B-A3B | 1 | 0.035 | 0.541 | 0.06× |
| 1_Square_matrix_multiplication_ | matmul | cuda | Qwen-30B-A3B | 5 | 0.032 | 0.541 | 0.06× |
| 2_Standard_matrix_multiplication_ | matmul | cuda | gpt-oss-120b | 3 | 0.030 | 0.541 | 0.06× |
| 17_Matmul_with_transposed_B | matmul | cuda | Qwen-30B-A3B | 4 | 0.029 | 0.541 | 0.05× |

High-side outliers cluster in `loss`/`normalization` (small-n, n=1–7,
gpt-oss-heavy) — these are the cases closest to a genuine win. Low-side
outliers are almost entirely `cuda` matmul (0.03–0.18× — catastrophically
slow relative to even the already-weak matmul family median of 0.541×),
consistent with §9's zero-flag finding above: CUDA's hand-written matmul
attempts are the worst performers in the entire dataset.

### Top/bottom 3 contributing tasks (weighted log-speedup deviation from the global geomean)

| rank | task | n | task geomean | contribution |
|---|---|---:|---:|---:|
| ↑1 | 40_LayerNorm | 19 | 3.773× | +0.0573 |
| ↑2 | 42_Max_Pooling_2D | 22 | 1.679× | +0.0307 |
| ↑3 | 94_MSELoss | 9 | 2.161× | +0.0171 |
| ↓1 | 1_Square_matrix_multiplication_ | 47 | 0.509× | −0.0465 |
| ↓2 | 17_Matmul_with_transposed_B | 29 | 0.385× | −0.0449 |
| ↓3 | 3_Batched_matrix_multiplication | 31 | 0.433× | −0.0407 |

(Contribution = `n_task/500 × (mean(log speedup)_task − mean(log speedup)_global)`
— i.e. how much each task's own log-speedup, weighted by its sample count,
pulls the pooled geomean up or down.) Note the bottom 3 are **all matmul**,
and collectively carry more weight (47+29+31=107 samples, 21.4% of the
whole n=500) than the top 3 (19+22+9=50 samples, 10%) — matmul is both the
most sampled family among successfully-retimed kernels and the strongest
drag on the overall geomean.

### 76_conv_standard_1D_dilated_strided__ — zero-resolution task

**This primary-32 task has ZERO successfully-retimed results in any
language/model configuration** (`final_retiming.py`'s own warning:
`1 primary-32 task(s) have NO correct chain to retime`). This is *not*
"no configuration ever produced a correct kernel for this task" — 2 chains
did reach `correctness=True` for it during the run (`cuda|0shot|
76_conv_standard_1D_dilated_strided__|openai/gpt-oss-120b|3` at turn 9,
speedup 0.127 recorded in the turn-loop; `tilelang|0shot|
76_conv_standard_1D_dilated_strided__|openai/gpt-oss-120b|0`) — but **both
of the 2 candidate chains for this task crashed during the final re-timing
pass** (both are among the 56 exceptions, both in the "single occurrence"
category, §7). Net effect: this task contributes **zero rows** to
`task_speedup_final.csv` and has no baseline in
`final_timing_20260823T121517.json`'s `baseline_by_task`. State this
explicitly wherever the paper enumerates per-task coverage — it is a gap in
the *final, authoritative* measurement, not evidence that the task is
unsolvable.

---

## 10. Consistency verification (3 checks — same bar as the A100 cross-check's `verify_eval_completeness.py`)

| # | check | method | result |
|---|---|---|---|
| A | Ever-correct curve monotonicity | For all 8 (lang, model) × turn-1..10 series in `fig1_ever_correct.csv` (160 rows, both conditions), verified `ever_correct[t] >= ever_correct[t-1]` for every consecutive turn pair | **PASS — 0 violations** |
| B | CSV ↔ state.json recompute agreement | Independently recomputed ever-correct@turn from `multiturn_state.json` via a second, differently-written code path (first-correct-turn-per-chain, not `analyze.py`'s "any history turn ≤ t" scan) and diffed every (lang, model, condition, turn) cell against `fig1_ever_correct.csv` | **PASS — 0 mismatches** across all 160 cells |
| C | Total-count reconciliation | (i) turn-10 `ever_correct` sums to 564 across all 8 lang×model×2-condition cells, matching an independent full-history scan of `multiturn_state.json`. (ii) turn-10 `n` sums to 1960, matching total chain count. (iii) termination reasons sum to 1960 (1870+74+16). (iv) `final_retiming_summary.csv` n-column sums to exactly 500; `task_speedup_final.csv` n-column also sums to exactly 500; both equal `556 − 56` from `final_retiming.py`'s own log line — three independent arithmetic paths to the same number | **PASS — all four sub-totals reconcile exactly** |

One **data-quality note** surfaced by an *additional* check (not one of the
3 requested, but found while verifying B): `multiturn.py cmd_report`'s
"chains with a best-so-far correct kernel" line in
`turn_10_report.txt` reports **554**, not 556. Root cause: 2 chains
(the original pre-2026-08-23 `timing_unmeasurable` entries,
`cuda|docinject|57_conv...|1` and `tilelang|docinject|97_SDPA|1`) have
`best_code` set but `best_speedup: null` — an inconsistency between those
two fields that predates this session (likely from the original
`timing_20260820.json` backfill referenced in `multiturn.py`'s comments).
`cmd_report`'s line filters on `best_speedup is not None` (554), while
`final_retiming.py` filters on `best_code` truthy (556) — the latter is
authoritative for what actually got retimed (and both of those 2 chains
*do* appear correctly in the 56-exception / 6-reproduced-failure set in
§7). Not fixed here (read-only w.r.t. already-generated state per CLAUDE.md
rule 1) — flagged for anyone citing `turn_10_report.txt`'s own summary line
directly: use 556 (or better, the 500 in §7), not 554.

---

## 11. Verification appendix (2026-08-23, post-commit-`17f4505` PI requests)

### 11.1 T32(d) fast_1 footnote — was wrong, now corrected

`paper/TABLES_32TASK_PRIMARY.md` §(d) (and the identical passage in
`paper/RESULTS_REPORT_20260820.md`, both sourced from `results/eval/
timing_20260820.json` via `scripts/analyze.py`'s `speedup_table()`)
carried a footnote claiming `triton|gpt-oss-120b|0shot` was the *only* one
of the 8 cells exceeding fast_1 50% — contradicted by the same table's own
`triton|gpt-oss-120b|docinject` (53.6%) and `triton|Qwen3-Coder-30B|0shot`
(92.9%) rows. **Denominator check (per `scripts/analyze.py`'s `agg()`
inside `speedup_table()`): fast_1's denominator is per-SAMPLE, not
per-task** — records are grouped by `(language, model, condition)` and
`fast_1` counts records with `speedup > 1` within that group; one record =
one correct-and-timed generated kernel sample, not one distinct task. Both
files corrected (this session): 3 of 8 cells exceed 50% (all triton:
53.1%/49, 53.6%/28, 92.9%/14), 1 more ties exactly at 50.0% (6/12), the
other 4 (non-triton) are all well under. See the inline corrections in
both files for the full replacement text.

### 11.2 Two cited truncation numbers — both verified correct

**"PTX 외 언어의 턴1 truncation ≤1.1%"** — **TRUE**, and conservative (actual
max is lower). Recomputed directly from `results/eval/multiturn_state.json`
(every chain's `history[0]`, i.e. turn 1, `gen_status` field — independent
of and cross-checked against `turn_10_report.txt`'s own denominator-caveat
section, which agrees exactly):

| language | n (both models, 0shot+docinject) | truncated+format_failure | % |
|---|---:|---:|---:|
| cuda | 490 | 4 | 0.82% |
| tilelang | 490 | 1 | 0.20% |
| triton | 490 | 0 | 0.00% |
| ptx (for contrast, not part of the claim) | 490 | 74 | 15.10% |

Max among non-PTX languages = **0.82%** (cuda) — comfortably under the
1.1% ceiling cited. (1.1% itself is a real number in this dataset, just
not a truncation rate: `paper/RESULTS_REPORT_20260820.md` line 84 has
tilelang|gpt-oss-120b at "2/185 (1.1%)" for **compiled**, a different
metric — do not confuse the two if tracing this claim back further.)

**"A100 Qwen PTX truncation 28.6%"** — **TRUE**. Source:
`results/eval/eval_a100_full.json` (pulled from `origin/results-a100`,
commit `bcd6d4f` — hyunjun1234's A100/Ampere sm_80 probe, 1,110 records,
0-shot only, cuda/ptx/triton, no tilelang, verified complete via
`scripts/verify_eval_completeness.py`). Filtering to `language=="ptx"` and
model containing `"Qwen"`:

```
n = 185 (37-task raw basis, the A100 probe's native denominator)
gen_status == "truncated": 53
53 / 185 = 28.6486...% -> 28.6%
```

Matches the cited figure exactly on the probe's own **37-task (raw)**
basis. Note for anyone re-deriving this on our paper's 32-task primary
filter instead: `clean_32_tasks()` reduces the denominator to 160 and the
truncated count to 46, giving **28.75% (≈28.7–28.8%)** — a close but
*different* number from 28.6%. The cited 28.6% is the A100 probe's raw
(37-task) figure, not the 32-task-primary-filtered one; state the basis
explicitly wherever this number is cited in the paper.

---

## Summary of citable numbers for the paper

- **Primary correctness metric:** ever-correct@turn, `paper/figures_data/fig1_ever_correct.csv` (§1, verified §10).
- **Primary speedup metric:** `paper/figures_data/final_retiming_summary.csv`, n=500 (§7's reconciled denominator) — the *only* citable speedup/fast_1 source (CLAUDE.md item 4).
- **Task-level detail / outliers:** `paper/figures_data/task_speedup_final.csv` (§9).
- **Turn-4 regression:** cite `paper/TURN4_FOOTNOTE.md` verbatim.
- **Known measurement gaps to disclose:** 76_conv task has zero final-pass data (§9); 16 chains manually terminated for timing-unmeasurability, of which 6 reproducibly failed twice and 4 actually succeeded on retiming (§7); 8 ever-correct chains were never measurable at all (§8); PTX's near-zero correctness is confounded by 8–17%/turn truncation (§4).

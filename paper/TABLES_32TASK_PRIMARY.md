# 32과제 기준 주 분석표 (PI 확정 기준, 2026-08-20)

37과제 중 §7-2 허용오차 감사에서 결함으로 확인된 5과제(`23_Softmax`,
`4_Matrix_vector_multiplication_`, `6_Matmul_with_large_K_dimension_`,
`90_cumprod`, `95_CrossEntropyLoss`)를 제외한 **32과제**가 논문의 주 분석
기준이다. docinject는 자체 20과제 층화 부분집합 중 위 5과제와 겹치는 3개
(`23_Softmax`, `6_Matmul_with_large_K_dimension_`, `95_CrossEntropyLoss`)만
제외한 **17과제** 기준(20과제 설계 자체는 확장하지 않음,
`tasks/SELECTION.md` §4.3 유지).

정확성 판정(compiled/correctness) 자체는 원본과 동일 — 이 문서는 **집계
필터만 다르게 적용**한 결과다(`scripts/analyze.py`, 재실행 재현 가능).
37과제/20과제 원판정 전체는 `paper/TABLES_37TASK_APPENDIX.md` 참고.

## (a) 표1 — 본 실행 0-shot, 32과제 기준 (2×2 메인 결과)

| 언어 | 모델 | n | compiled | correct |
|---|---|---:|---:|---:|
| cuda | gpt-oss-120b | 160 | 1 (0.6%) | 0 |
| cuda | Qwen3-Coder-30B | 160 | 45 (28.1%) | 0 |
| ptx | gpt-oss-120b | 160 | 130 (81.3%) | 0 |
| ptx | Qwen3-Coder-30B | 160 | 84 (52.5%) | 0 |
| tilelang | gpt-oss-120b | 160 | 1 (0.6%) | 0 |
| tilelang | Qwen3-Coder-30B | 160 | 0 (0%) | 0 |
| **triton** | **gpt-oss-120b** | 160 | 154 (96.3%) | **49 (30.6%)** |
| **triton** | **Qwen3-Coder-30B** | 160 | 139 (86.9%) | **14 (8.75%)** |

전체 정확률(4언어 합산, n=1,280): 63/1,280 = 4.9%, 전부 Triton.

## (b) 표2 — docinject 전후, 17과제 기준 (0-shot 동일 과제 대조군 포함)

| 언어 | 모델 | 0-shot(17) compiled/correct | docinject(17) compiled/correct | Δcompiled | Δcorrect |
|---|---|---:|---:|---:|---:|
| cuda | gpt-oss-120b | 0/85, 0 | 56/85, **42** | +56 | **+42** |
| cuda | Qwen3-Coder-30B | 25/85, 0 | 0/85, 0 | −25 | 0 |
| ptx | gpt-oss-120b | 68/85, 0 | 74/85, 1 | +6 | +1 |
| ptx | Qwen3-Coder-30B | 43/85, 0 | 0/85, 0 | −43 | 0 |
| tilelang | gpt-oss-120b | 1/85, 0 | 79/85, **25** | +78 | **+25** |
| tilelang | Qwen3-Coder-30B | 0/85, 0 | 70/85, **15** | +70 | **+15** |
| triton | gpt-oss-120b | 81/85, 17 | 83/85, 28 | +2 | +11 |
| triton | Qwen3-Coder-30B | 70/85, 11 | 74/85, 12 | +4 | +1 |

n=85/셀(17과제×5샘플). 참고: docinject 20과제 원판정(n=100/셀) 대비
17과제 기준의 방향·크기는 거의 동일 — 정확성 표에는 결함 과제 제외가
speedup 표만큼 큰 영향을 주지 않았다(§7-1 참고).

## (c) A100 교차검증표, 32과제 기준

PRO 6000(Blackwell, sm_120a) vs A100(Ampere, sm_80), 0-shot, cuda/ptx/triton
(A100 프로브에 tilelang 없음). 출처: `origin/results-a100`
(hyunjun1234, 1,110건, 전수검증 완료 — HF 체크포인트 동일, vLLM
0.10.1/torch 2.5.1+cu121/driver 535.309.01로 PRO6000과 스택 다름).

| 언어 | 모델 | PRO6000 n(32) compiled/correct | A100 n(32) compiled/correct |
|---|---|---:|---:|
| cuda | gpt-oss-120b | 1/160, 0 | 3/160, **1** |
| cuda | Qwen3-Coder-30B | 45/160, 0 | 36/160, 0 |
| ptx | gpt-oss-120b | 130/160, 0 | 133/160, 0 |
| ptx | Qwen3-Coder-30B | 84/160, 0 | 98/160, 0 |
| triton | gpt-oss-120b | 154/160, 49 | 151/160, **55** |
| triton | Qwen3-Coder-30B | 139/160, 14 | 140/160, **21** |

정성적 패턴(Triton 지배, CUDA/PTX 붕괴)이 두 아키텍처에서 재현됨 —
Triton 정답 수는 A100이 양쪽 모델 다 더 높음(원인 미규명, 상관만 보고,
인과 서술 금지).

## (d) speedup 표, 32과제 기준 정제 (fast_1 포함)

`results/eval/timing_20260820.json`(warmup 25/측정 100/중앙값, 정답 커널당
PyTorch eager fp16 베이스라인 재측정) 중 결함 5과제를 과제 단위로 대칭
제외한 값. 원값(37/20과제 포함)은 부록 참고.

| 언어 | 모델 | 조건 | n | fast_1 | speedup geomean |
|---|---|---|---:|---:|---:|
| cuda | gpt-oss-120b | docinject | 41 | 8 (19.5%) | 0.247x |
| ptx | gpt-oss-120b | docinject | 1 | 1 (100%) | 1.00x |
| tilelang | gpt-oss-120b | docinject | 24 | 8 (33.3%) | 0.713x |
| tilelang | Qwen3-Coder-30B | docinject | 15 | 3 (20.0%) | 0.170x |
| triton | gpt-oss-120b | 0shot | 49 | 26 (53.1%) | 1.14x |
| triton | gpt-oss-120b | docinject | 28 | 15 (53.6%) | 0.797x |
| triton | Qwen3-Coder-30B | 0shot | 14 | 13 (92.9%) | 1.12x |
| triton | Qwen3-Coder-30B | docinject | 12 | 6 (50.0%) | 0.849x |

**정정 (2026-08-23, 표 자체와 모순되는 각주 발견)**: "8셀 중 `triton|
gpt-oss-120b|0shot`이 유일하게 50%를 넘는다"는 틀렸다 — 위 표에서
`triton|gpt-oss-120b|docinject`도 53.6%(15/28), `triton|Qwen3-Coder-30B|0shot`도
92.9%(13/14)로 50%를 넘고, `triton|Qwen3-Coder-30B|docinject`은 정확히
50.0%(6/12)로 동률이다. 정확한 서술: **8셀 중 3셀이 fast_1 50%를 넘고(모두
triton), 1셀은 동률, non-triton 4셀은 전부 미만.** fast_1의 분모는 과제 단위가
아니라 **정답 판정 + 타이밍 측정 둘 다 성공한 샘플 단위**다(`scripts/analyze.py`
`speedup_table()`의 `agg()`: (language, model, condition)별로 묶은 레코드 중
`speedup > 1`인 레코드 수 — 레코드 1개 = 샘플 1개, 과제가 아님). 226/228
측정 완료, 2건은 재현 가능한 세그폴트로 측정 불가(§7-1) — speedup 집계에서
자연 제외됨. 상세 근거는 `paper/RESULTS_REPORT_20260820.md`의 동일 정정
참고.

## 재현 방법

```
source scripts/env.sh && source .venv/bin/activate
python scripts/analyze.py --out results/eval/analysis_<timestamp>.json
```

## 재검증 이력

- 2026-08-21 11:14 KST: 08-21 07:19 재부팅 복구 후 재실행, (a)~(d) 4표 전부
  본 문서 수치와 완전 일치 확인(byte-for-byte). 입력 4개(`full_run_20260819.json`,
  `docinject_run_20260820T072056.json`, `timing_20260820.json`,
  `eval_a100_full.json`)는 턴1 스냅샷이라 진행 중인 멀티턴 루프(턴 4+, 별도
  파일에 기록)의 영향을 받지 않음 — 갱신 불필요, 재현성만 재확인.
- 2026-08-21 11:40 KST: 11:20 재부팅(2차) 복구 후 독립 재실행(`analysis_verify_
  20260821T114029.json`)으로 재확인 — (a)~(d) 4표 수치 전부 다시 일치. 두 번째
  재부팅도 입력 4개 파일을 건드리지 않았음(mtime 전부 2026-08-20, 두 크래시
  모두 이전).

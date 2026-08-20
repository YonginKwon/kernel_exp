# 37과제 원판정 부록 (논문 부록용)

`paper/TABLES_32TASK_PRIMARY.md`가 논문 본문의 주 분석 기준(32/17과제)이다.
이 문서는 결함 5과제를 제외하지 않은 **원판정 전체**를 부록 자료로 보존한다.
판정 자체는 두 문서 사이에 차이가 없다 — 집계 대상 과제 범위만 다르다.

## 표1 — 본 실행 0-shot, 37과제 (n=185/셀)

| 언어 | 모델 | compiled | correct |
|---|---|---:|---:|
| cuda | gpt-oss-120b | 1 (0.5%) | 0 |
| cuda | Qwen3-Coder-30B | 51 (27.6%) | 0 |
| ptx | gpt-oss-120b | 149 (80.5%) | 0 |
| ptx | Qwen3-Coder-30B | 106 (57.3%) | 0 |
| tilelang | gpt-oss-120b | 2 (1.1%) | 0 |
| tilelang | Qwen3-Coder-30B | 0 (0%) | 0 |
| triton | gpt-oss-120b | 179 (96.8%) | 59 (31.9%) |
| triton | Qwen3-Coder-30B | 163 (88.1%) | 24 (13.0%) |

전체 정확률(n=1,480): 83/1,480 = 5.6%.

## 표2 — docinject 전후, 20과제 원부분집합 (n=100/셀)

| 언어 | 모델 | 0-shot(20) compiled/correct | docinject(20) compiled/correct |
|---|---|---:|---:|
| cuda | gpt-oss-120b | 0/100, 0 | 66/100, 48 |
| cuda | Qwen3-Coder-30B | 30/100, 0 | 0/100, 0 |
| ptx | gpt-oss-120b | 79/100, 0 | 88/100, 1 |
| ptx | Qwen3-Coder-30B | 55/100, 0 | 0/100, 0 |
| tilelang | gpt-oss-120b | 2/100, 0 | 92/100, 29 |
| tilelang | Qwen3-Coder-30B | 0/100, 0 | 82/100, 16 |
| triton | gpt-oss-120b | 96/100, 25 | 98/100, 32 |
| triton | Qwen3-Coder-30B | 85/100, 19 | 89/100, 19 |

## A100 교차검증표, 37과제 원판정

| 언어 | 모델 | PRO6000 n(37) compiled/correct | A100 n(37) compiled/correct |
|---|---|---:|---:|
| cuda | gpt-oss-120b | 1/185, 0 | 3/185, 1 |
| cuda | Qwen3-Coder-30B | 51/185, 0 | 38/185, 0 |
| ptx | gpt-oss-120b | 149/185, 0 | 155/185, 0 |
| ptx | Qwen3-Coder-30B | 106/185, 0 | 116/185, 0 |
| triton | gpt-oss-120b | 179/185, 59 | 176/185, 63 |
| triton | Qwen3-Coder-30B | 163/185, 24 | 165/185, 33 |

## speedup 표, 원값(결함 5과제 포함, 9건 이상치 미제거)

| 언어 | 모델 | 조건 | n | fast_1 | speedup geomean |
|---|---|---|---:|---:|---:|
| cuda | gpt-oss-120b | docinject | 47 | 8 (17.0%) | 0.249x |
| ptx | gpt-oss-120b | docinject | 1 | 1 (100%) | 1.00x |
| tilelang | Qwen3-Coder-30B | docinject | 16 | 4 (25.0%) | 0.214x |
| tilelang | gpt-oss-120b | docinject | 28 | 8 (28.6%) | 0.487x |
| triton | Qwen3-Coder-30B | 0shot | 24 | 18 (75.0%) | **2.13x** ⚠ |
| triton | Qwen3-Coder-30B | docinject | 19 | 10 (52.6%) | **2.38x** ⚠ |
| triton | gpt-oss-120b | 0shot | 59 | 26 (44.1%) | 0.797x |
| triton | gpt-oss-120b | docinject | 32 | 15 (46.9%) | 0.576x |

⚠ 표시 2행은 `23_Softmax` 9건의 허위 speedup(405~1,350x, 참조 fp16
atol=rtol=1e-2가 이 과제 출력 크기보다 4,090배 커서 부분 계산이 통과)이
포함된 오염 값 — **논문에 인용 금지**, 반드시
`paper/TABLES_32TASK_PRIMARY.md` (d)의 정제값(1.12x/0.849x)을 쓸 것.

## 재현 방법

```
source scripts/env.sh && source .venv/bin/activate
python scripts/analyze.py --out results/eval/analysis_<timestamp>.json
```

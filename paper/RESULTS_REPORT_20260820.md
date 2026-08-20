# kernel-lang-2x2 — 결과 보고서 (2026-08-20 기준)

논문작성 에이전트/PI 공유용. 마감 2026-08-29 (AoE), NeurIPS 워크숍 4쪽.
연구 질문·가설·관련 연구는 `RESEARCH_CONTEXT.md`, 실험 설계·방법론 규칙은
`CLAUDE.md`가 원본이다 — 이 문서는 **지금까지 확보된 실측 데이터와 그 해석**만
담는다. 숫자가 바뀌면(재분류·추가 실행) 이 문서를 갱신할 것.

## 0. 한 줄 요약

37과제×4언어×5샘플×2모델 0-shot 본 실행(1,480건)과 20과제 층화 부분집합 문서
주입(docinject) ablation(800건) 모두 완료·전수검증 통과. **Triton만 유의미한
정답률을 낸다(0-shot 22.4%)**; CUDA/PTX/TileLang은 0-shot 정답 0%로 API/문법
실패가 지배적. 문서 주입은 **gpt-oss엔 4개 언어 전부 강하게 도움**이 되지만
**Qwen엔 TileLang만 돕고 CUDA·PTX는 오히려 0%로 붕괴**시킨다 — 모델 의존적
효과로, H3("문서 주입이 LRPL 격차를 줄인다")를 단순 지지하지 않는다. **타이밍
(speedup) 측정 완료(§7-1, 226/228)** — 정답 커널은 대체로 PyTorch eager보다
느리다(geomean 0.2~0.8x, Qwen Triton만 예외적으로 빠른데 이 중 9건은 정확성
판정 결함으로 인한 착시로 확인돼 제외 후 다시 계산하면 역시 0.5x대). **수리
1턴(멀티턴) 프로토콜은 아직 실행되지 않았다** — 별도 착수 예정.

---

## 1. 실험 설계 (변경 없음, CLAUDE.md 원본)

2×2: 자원 수준(HRPL/LRPL) × 추상화 수준(저/고).

|         | HRPL (고자원) | LRPL (저자원) |
|---------|---------------|----------------|
| 저추상  | CUDA C++      | PTX            |
| 고추상  | Triton        | TileLang       |

- 모델: `openai/gpt-oss-120b` (MXFP4), `Qwen/Qwen3-Coder-30B-A3B-Instruct` (bf16) —
  둘 다 로컬 vLLM 서빙(0.27.1), 이 서버(RTX PRO 6000 Blackwell, sm_120a) 단독.
  원안이던 Qwen3-Coder-Next-FP8(80B-A3B)은 hybrid Gated-DeltaNet 커널 세그폴트로
  강등(PI 승인 2026-08-19, CLAUDE.md에 상세).
- 정밀도: 4개 언어 전부 fp16 통일 (PI 승인 2026-08-19, atol=rtol=1e-2,
  KernelBench 하드코딩 기본값 그대로).
- 프로토콜: temperature 0.8, max_tokens 8192, seed 명시 고정(sample i = base_seed+i).
- 정확성 판정: KernelBench 표준 하니스(`eval_kernel_against_ref`) 그대로 사용,
  자체 검증기 없음.

## 2. 데이터 인벤토리

| 실행 | 조건 | 규모 | 결과 파일 | 상태 |
|---|---|---|---|---|
| 본 실행 | 0-shot, 37과제 | 4언어×37과제×5샘플×2모델 = 1,480 | `results/eval/full_run_20260819.json` | **완료, 전수검증 PASS** (`scripts/verify_eval_completeness.py`) |
| Ablation | docinject, 20과제 층화 부분집합 | 4언어×20과제×5샘플×2모델 = 800 | `results/eval/docinject_run_20260820T072056.json` | **완료, 전수검증 PASS** |
| CUDA 프로브 (본 실행 전 사전 조사) | 0-shot, 4과제 | 4과제×5샘플×2모델 = 40 | `results/eval/cuda_probe_final_20260819.json` | 완료 (하니스 판정용, 정식 결과 아님) |

생성 원본은 `results/raw/<lang>/<condition>/<task>/<model_dir>/sample_N.json` —
전문(프롬프트+응답 원문+생성 파라미터+HF 리비전+vLLM 버전+GPU 기종) 보존, git
추적 안 함(`.gitignore`, 읽기 전용 데이터 취급).

**재현성 필드 확인**: 모든 레코드에 `hf_revision`, `vllm_version`, `dtype`,
`temperature`, `seed`, `prompt_sha256`, `gpu_name`(RTX PRO 6000 Blackwell,
sm_120), 타임스탬프 존재. CLAUDE.md 규칙 4 충족.

---

## 3. 본 실행 결과 (0-shot, 37과제, n=185/셀)

| 언어 | 모델 | compiled | correct | 비고 |
|---|---|---:|---:|---|
| cuda | gpt-oss-120b | 1/185 (0.5%) | 0/185 | API/문법 실패 지배 (§5) |
| cuda | Qwen3-Coder-30B | 51/185 (27.6%) | 0/185 | `-std=c++14` 하니스 우회 적용 후 수치(§6) |
| ptx | gpt-oss-120b | 149/185 (80.5%) | 0/185 | truncated 2/185 |
| ptx | Qwen3-Coder-30B | 106/185 (57.3%) | 0/185 | **truncated 57/185(30.8%)** — 분모 주의(§5) |
| tilelang | gpt-oss-120b | 2/185 (1.1%) | 0/185 | API 환각 지배 (§5) |
| tilelang | Qwen3-Coder-30B | 0/185 (0%) | 0/185 | API 환각 지배 (§5) |
| **triton** | gpt-oss-120b | 179/185 (96.8%) | **59/185 (31.9%)** | 4언어 중 유일하게 유의미한 정답 |
| **triton** | Qwen3-Coder-30B | 163/185 (88.1%) | **24/185 (13.0%)** | |

**전체 정확률(4언어 합산, n=1480): 83/1480 = 5.6%, 전부 Triton.**
CUDA/PTX/TileLang 3개 언어 정답 0/1110.

### H1/H2 예비 판정 (본 실행만으로는 결론 유보 권장)

- H1(자원 우세: CUDA·Triton ≫ PTX·TileLang)도, H2(추상화 우세: Triton·TileLang >
  CUDA·PTX)도 **깔끔하게 지지되지 않는다.** Triton(고추상+고자원)만 압도적으로
  성공하고, 나머지 세 셀은 컴파일 단계에서 전멸에 가깝다.
- CUDA(HRPL)가 PTX(LRPL)보다 낮은 컴파일률(특히 gpt-oss 0.5% vs 80.5%)을 보이는
  역전 현상의 원인은 자원 수준이 아니라 **하니스 결합 방식의 차이**로 보인다:
  CUDA는 `load_inline`의 cpp/cuda 이중 소스 + pybind 바인딩을 모델이 직접
  맞춰야 하는 구조적 부담이 있고(§5.1 기타 참고), PTX 하니스는 그런 이중
  바인딩 요구가 없다. **인과 서술 금지(PI, 2026-08-20) — 상관만 보고할 것.**
  이 해석은 방증 하나를 확보했다: §4의 docinject 결과에서 gpt-oss CUDA
  컴파일률이 0.5%→66%로 회복됐는데, 이는 정확히 "관용구(pybind 바인딩 등)를
  스펙 형태로 명시하면 부담이 줄어든다"는 인터페이스 관용구 부담 가설과
  **consistent**하다. 다만 이것도 어디까지나 방증이지 증명이 아니다 — CUDA
  실패가 정말 "자원과 무관한 구조적 문제"인지, 낮은 자원 수준이 이 관용구
  습득을 어렵게 만든 결과인지는 이 데이터만으론 구분 불가하며, 논문에서는
  "~와 일치한다(consistent with)" 이상의 인과적 표현을 쓰지 않는다.
- Triton의 실패 프로파일도 다른 3언어와 질적으로 다르다: compile_fail
  28/370(7.6%)뿐이고 나머지 실패(259/370, 70%)는 "컴파일은 되지만 오답" —
  즉 API/문법이 아니라 **커널 로직 자체의 수치 오류**가 지배적. 이는 Triton의
  Python 임베디드 문법 + 성숙한 컴파일러가 구조적 오류를 조기에 걸러준다는
  뜻일 수 있다 (가설, 미검증).

## 4. 문서 주입(docinject) Ablation 결과 (20과제 층화 부분집합, n=100/셀)

과제 선정: `tasks/level1_subset.json`의 `doc_ablation_subset_of_20`, 37과제
계열 분포와 전 계열 5%p 이내로 비례하는 층화 표본(승인 근거:
`tasks/SELECTION.md` §4.3). 주입 문서: `prompts/specs/{cuda,ptx,triton,tilelang}.md`
(~5k 토큰/언어, 4언어 동일 조건 적용).

| 언어 | 모델 | 0-shot (동일 20과제, n=100) | docinject (n=100) | Δcompiled | Δcorrect |
|---|---|---:|---:|---:|---:|
| cuda | gpt-oss-120b | 0/100, 0/100 | 66/100, **48/100** | +66 | **+48** |
| cuda | Qwen3-Coder-30B | 30/100, 0/100 | **0/100**, 0/100 | **−30** | 0 |
| ptx | gpt-oss-120b | 79/100, 0/100 | 88/100, 1/100 | +9 | +1 |
| ptx | Qwen3-Coder-30B | 55/100, 0/100 | **0/100**, 0/100 | **−55** | 0 |
| tilelang | gpt-oss-120b | 2/100, 0/100 | 92/100, **29/100** | +90 | **+29** |
| tilelang | Qwen3-Coder-30B | 0/100, 0/100 | 82/100, **16/100** | +82 | **+16** |
| triton | gpt-oss-120b | 96/100, 25/100 | 98/100, 32/100 | +2 | +7 |
| triton | Qwen3-Coder-30B | 85/100, 19/100 | 89/100, 19/100 | +4 | 0 |

### H3 판정: 지지되지 않음, 모델 의존적 (논문에서 반드시 이렇게 프레이밍할 것)

- **gpt-oss**: 4개 언어 전부 개선, 특히 CUDA(0→48%)·TileLang(0→29%)에서 극적.
  LRPL(PTX·TileLang)뿐 아니라 HRPL(CUDA)도 크게 개선됐다는 점에서, "문서 주입이
  LRPL 격차를 좁힌다"는 H3의 좁은 버전(LRPL에만 효과)은 성립하지 않는다 —
  gpt-oss에게는 **자원 수준과 무관하게 구조화된 스펙 자체가 도움**이 된 것으로
  보인다.
- **Qwen**: TileLang만 개선(0→16%), CUDA·PTX는 **0-shot보다 더 나빠져 0%로
  붕괴**. 두 붕괴 모두 근본 원인을 직접 확인함(§5.2):
  - CUDA: `c10::Half*`→`__half*` 타입 불일치가 지배적 실패(81건 중 59건 동일
    패턴). **주입 문서가 정확히 이 문제의 해법을 명시하는데도** 적용하지 못함.
  - PTX: 84/84 "generated" 샘플 전부 `import torch`/`import torch.nn as nn`
    없이 코드를 시작 — 원본 응답(`response_raw`)에서 직접 확인, 파서 결함
    아님. 주입 문서 자체는 두 import를 포함한 완전한 예시를 담고 있음.
  - 두 경우 다 **하니스·프롬프트 버그 아님**(각각 컴파일러 커맨드라인·원본
    응답 텍스트로 직접 확인) — Qwen이 긴 in-context 스펙을 만나면 보일러플레이트
    서두(임포트)를 생략하거나 스펙이 제시한 해법을 적용하지 못하는 실제 행동
    변화로 보인다.
- **H3 프레이밍 확정 (PI, 2026-08-20): "문서 주입 효과의 부호는 모델 의존적".**
  H3를 "문서 주입은 보편적으로 LRPL을 돕는다"가 아니라 이 형태로 확정 보고한다.
  이 자체가 흥미로운 결과다 — 단일 모델·단일 언어로 일반화된 기존 문헌(§4.1
  MultiPL-T 등)의 암묵적 가정에 대한 반례.
  **Qwen docinject 붕괴(CUDA·PTX 0%)는 하니스 무죄가 직접 검증된 확정
  데이터다 — 재실행 금지.** CUDA는 컴파일러 커맨드라인에서 `-std=` 재발
  없음을, PTX는 `response_raw`(모델 원본 응답 텍스트, 파서 개입 이전)에서
  import 누락을 직접 확인했다(둘 다 §5). 재현이 안 되거나 이상해 보여도 이
  결론을 재검증한다는 명목으로 재실행하지 말 것 — 이미 원인까지 특정된
  확정 결과다.

## 5. 오류 분류 (RESEARCH_CONTEXT.md §7 표3 재료)

### 5.1 CUDA — gpt-oss-120b (0-shot, 184/185건 컴파일 실패 전수 분류)

| 범주 | 건수 | 비율 |
|---|---:|---:|
| cpp/cuda 함수명·시그니처 불일치(pybind 바인딩 ≠ 정의) | 81 | 44.0% |
| 존재하지 않는 매크로/식별자 (`AT_DISPATCH_HALF_TYPES` 등 환각) | 46 | 25.0% |
| 컴파일러 진단 미포착(2000자 캡에 잘림, **2026-08-20 캡 상향 후 미재수집**) | 23 | 12.5% |
| 존재하지 않는 CUDA/ATen API 환각 (`at::cuda::getCurrentCUDAStream` 등) | 9 | 4.9% |
| 디바이스 전용 내장 변수(`blockIdx` 등)를 호스트 코드에 사용 | 6 | 3.3% |
| Python 자체 문법 오류(`'''` 개수 불일치 등, 파서 무결 확인) | 6 | 3.3% |
| `load_inline()` 잘못된 kwarg | 3 | 1.6% |
| 파싱 실패(포맷) | 3 | 1.6% |
| 기타 | 7 | 3.8% |

### 5.1b CUDA — Qwen3-Coder-30B (0-shot) — **미완, 재수집 필요**

`-std=c++14` 하니스 우회 적용 후 재평가된 133건의 컴파일 실패 중 **114건(86%)이
2000자 캡에 잘려 있어 아직 분류 불가**(2026-08-20 캡을 20000자로 올렸지만
기존 레코드는 재생성 안 됨). 분류 가능했던 19건: fn-name-mismatch 8, 디바이스
내장 오용 5, Python 문법 4, 기타 2. **논문에 "CUDA 오류 분류"를 언어 간
비교표로 쓰려면 이 114건을 격리 재컴파일로 재수집해야 한다(§7 TODO 참고,
방법은 CUDA 프로브 때와 동일 — 판정은 불변, 진단 텍스트만 재수집).**

docinject 조건에서는 Qwen CUDA 81건 전부 온전한 진단 확보(캡 상향 이후 실행)
— 59건이 `c10::Half*`/`__half*` 타입 불일치, 19건이 관련 링크 단계 undefined
symbol. §4 참고.

### 5.2 TileLang (0-shot, 346/347건 컴파일 실패 전수 분류, 3축)

| 축 | 건수 | 비율 |
|---|---:|---:|
| (a) 존재하지 않는 API 환각 (`tilelang.prim_func`/`Buffer`/`Ptr` 등 149건,
  `tilelang.jit`을 서브모듈로 오인 51건, `from tilelang import T` 44건,
  **TVM/TileLang 프레임워크 통째 혼동 80건** — 아래 참고) | 333 | 96.2% |
| (b) import 순서·환경 경로 문제 (하니스 후보) | **0 확정** | 0% |
| (c) 기타 (Python import 누락 7, 순수 문법 오류 6) | 13 | 3.8% |

(b)는 실재하는 환경 특이점(이 tilelang 0.1.13 설치는 `tvm`을 독립 패키지로
갖지 않고 `import tilelang` 시 자신의 번들 경로를 sys.path에 얹음 — 하니스는
모델 코드 실행 전에 tilelang을 미리 import하지 않으므로 이론상 순서 의존적
실패가 가능함)를 직접 검증했으나, "TVM/TileLang 혼동" 80건 전부(94%가
`tvm.build`/`tvm.runtime` 사용) 순서를 고쳐도 몇 줄 뒤 다시 깨지는 것을
확인 — **0건이 순서 수정만으로 회생 가능**. 즉 (a)로 귀속. 하니스 변경 없음.

### 5.3 PTX (0-shot, 두 모델 합산 56/370건 컴파일 실패)

| 범주 | 건수 |
|---|---:|
| `import torch` 누락 (`NameError: name 'torch' is not defined`) | 53 |
| `import torch.nn as nn` 누락 | 1 |
| ModelNew 클래스 없음 | 2 |

모델별: gpt-oss 34건, Qwen 22건 — 0-shot에선 소수 실패 모드였으나(§4에서
docinject 조건에 Qwen만 84/84로 폭증한 것과 대비). **truncated 59건(주로
Qwen, 57/185=30.8%)은 별도 실패 범주 — max_tokens=8192 도달, 재생성하지 않고
분모 병기하기로 결정(PI 지시).**

### 5.4 Triton (0-shot, 370건 중 compiled+wrong 259건, compile_fail 28건)

Triton은 다른 3언어와 실패 프로파일이 질적으로 다르다 — compile_fail은
28/370(7.6%)뿐, 나머지 실패는 **"컴파일은 통과했지만 결과가 틀림"**(259/370,
70%). 세부 수치 오류 원인 분류는 아직 안 함 — §7 TODO.

---

## 6. 하니스 변경 이력 (전부 git 커밋, `prompts/PROMPT_SPEC.md` §7이 원본)

| 날짜 | 변경 | 근거 | 판정 영향 |
|---|---|---|---|
| 08-19 | CUDA `-std=` 모델 지정 플래그 스트리핑(C++17 강제) | Qwen CUDA 51/185(27.6%)가 `-std=c++14` 직접 지정, 하니스 기본값과 충돌 | Qwen CUDA만, 재평가함 |
| 08-19 | eval 샘플별 서브프로세스 격리 + 샘플 단위 체크포인트 | CUDA illegal-memory-access가 프로세스 전체를 오염시켜 1,480건 실행 유실한 사고 | 없음(안정성) |
| 08-20 | `evaluate.py --resume` 추가 | 체크포인트 이어받기, 재평가 금지 | 없음 |
| 08-20 | 컴파일 진단 로깅 캡 2000/4000→20000자 | 704건 중 197건이 진단 절단(모델 간 비대칭: gpt-oss CUDA 172/181 vs Qwen 25/124) | **없음(순수 로깅)** — 기존 절단분 재수집은 미완(§7 TODO) |
| 08-20 | 닫히지 않은 코드펜스 허용 — **검토 후 기각** | 1.6%(3/185) 구제 대비 사전 등록 프로토콜 변경 비용이 큼 | 없음(변경 안 함) |
| 08-20 | `doc_ablation_subset_of_20` 승인 | PI 조건부 사전 승인(Triton 정답 확인 + gpt-oss CUDA 판정 종결) 충족, 층화 재검증(전 계열 5%p 이내) | 없음(과제 선정만) |

## 7-1. 타이밍/speedup 결과 (P0-a, 2026-08-20 완료)

`scripts/evaluate.py --timing` 신설(warmup 25/측정 100/중앙값,
`torch.cuda.synchronize` 내장 — `kernelbench.timing.time_execution_with_cuda_event`
직접 호출, KernelBench 자체 `eval_kernel_against_ref`는 warmup을 3으로 하드코딩해
노출하지 않아 별도 경로로 구현). 베이스라인은 PyTorch eager fp16, 샘플마다 이
GPU에서 새로 재측정(캐시/문헌 수치 재사용 없음). 결과 파일: `results/eval/timing_20260820.json`.

**대상**: 본 실행 + docinject의 compiled ∧ correct 전수 228건. **226/228(99.1%)
측정 완료.** 나머지 2건(`cuda/docinject/57_conv_transposed_2D.../sample_1`,
`tilelang/docinject/97_ScaledDotProductAttention/.../sample_1`)은 재현 가능한
세그폴트(`torch.cuda.synchronize()` → `cuCtxSynchronize`, 5회 시도·타임아웃
180→400s로 늘려도 매번 동일 지점에서 재현)로 측정 불가 — **flaky 아님**, 이
두 샘플에 한정된 문제로 보임(둘 다 상대적으로 무거운 커널: conv-transpose,
attention). 측정치 없이 데이터에서 결측으로 남김, 판정(compiled/correctness)은
불변.

| lang | model | condition | n | fast_1 | speedup geomean | excessive(>10x) flagged |
|---|---|---|---:|---:|---:|---:|
| cuda | gpt-oss-120b | docinject | 47 | 8 (17.0%) | 0.249x | 0 |
| ptx | gpt-oss-120b | docinject | 1 | 1 (100%) | 1.00x | 0 |
| tilelang | gpt-oss-120b | docinject | 28 | 8 (28.6%) | 0.487x | 0 |
| tilelang | Qwen3-Coder-30B | docinject | 16 | 4 (25.0%) | 0.214x | 0 |
| triton | gpt-oss-120b | 0shot | 59 | 26 (44.1%) | 0.797x | 0 |
| triton | gpt-oss-120b | docinject | 32 | 15 (46.9%) | 0.576x | 0 |
| **triton** | **Qwen3-Coder-30B** | **0shot** | 24 | 18 (75.0%) | **2.13x** | **5** |
| **triton** | **Qwen3-Coder-30B** | **docinject** | 19 | 10 (52.6%) | **2.38x** | **4** |

### 이상치 — Qwen Triton geomean은 오염돼 있음, 정제된 값을 쓸 것

**9건 전부 `23_Softmax` 한 과제에서만 나왔다**(0-shot 5건, docinject 4건, 전부
Qwen). speedup 405~1,350x. 원인을 코드까지 직접 확인: 이 과제의 `get_inputs()`가
`(4096, 393216)` — **행 하나가 393,216열**인 극단적으로 넓은 텐서를 만드는데,
Qwen이 생성한 Triton 커널은 `BLOCK_SIZE=1024`를 하드코딩하고 열 방향으로
루프/그리드 분할을 하지 않아 **각 행의 앞 1024열만 계산하고 나머지 39만여
열은 `torch.empty_like`의 미초기화 메모리로 남긴다**(샘플 하나는 그리드
launch 자체도 `cdiv(batch_size, BLOCK_SIZE)`로 잘못 설정해 4096행 중 4행만
처리). 그런데도 **correctness 판정을 통과했다** — 원인은 fp16 atol=rtol=1e-2
허용오차가, 이 과제의 정상 출력값 크기(softmax 원소 기댓값 ≈ 1/393216 ≈
2.5e-6)보다 4,000배 가까이 크기 때문으로 보인다: 사실상 아무 값이나 0 근처면
통과한다. **이건 하니스 버그가 아니라 이 특정 과제의 정확성 판정 기준이
당초 설계(§SELECTION.md #4.1)가 상정하지 않은 극단적 텐서 크기와 만나 무너진
사례** — speedup>10x 자동 플래그가 정확히 설계된 목적대로 작동해 잡아낸 것이다.
**판정은 변경하지 않았다**(PI 지시대로), 다만 이 9건을 뺀 정제된 geomean은
다음과 같이 근본적으로 다르다:

| lang | model | condition | n(제외 후) | fast_1 | geomean(제외 후) |
|---|---|---|---:|---:|---:|
| triton | Qwen3-Coder-30B | 0shot | 19 | 13 (68.4%) | **0.532x** (2.13x 아님) |
| triton | Qwen3-Coder-30B | docinject | 15 | 6 (40.0%) | **0.559x** (2.38x 아님) |

정제 후 Qwen Triton도 gpt-oss와 마찬가지로 **eager보다 느림**(0.53~0.56x) —
9건을 포함한 표면적 geomean 2.13x/2.38x는 단일 과제의 정확성 판정 결함이
만든 착시다. **논문·분석에는 정제된 값(제외 후)을 쓸 것.** 표3(오류 분류)에
`23_Softmax`류(정확성 판정이 텐서 크기에 취약한 과제) 방법론 캐비어트를
추가할 필요가 있다 — 향후 유사 과제 재검토 대상.

## 7. 미완료 / TODO (논문 작성 전 확인 필요)

1. ~~타이밍/스피드업 측정 미구현~~ — **완료, §7-1 참고 (2026-08-20).**
2. **수리 1턴(컴파일 에러 메시지만 제공) 프로토콜 미구현.** CLAUDE.md 프로토콜의
   핵심 축 하나 — 회복률 지표가 없음. `scripts/generate.py`에 관련 로직 없음.
3. **Qwen CUDA 0-shot 실패 133건 중 114건(86%) 재분류 필요** — 캡 상향 전
   기록이라 여전히 절단됨. 격리 재컴파일로 재수집(§6 참고, 판정 불변).
4. **Triton 259건 "compiled but wrong"의 수치 오류 세부 분류 미완.**
5. **CUDA 0-shot의 "PTX보다 낮은 컴파일률" 원인(§3, 하니스 결합 방식 가설)이
   검증되지 않음** — 논문에 인과적으로 서술하지 말 것, 상관만 보고하거나
   추가 조사 필요.
6. **`prompts/specs/*.md`가 docinject 조건에서 모델 행동을 바꾸는 메커니즘**
   (Qwen의 import 생략, 해법 미적용) — 왜 그런지는 미상, 결과만 확인됨.
7. 토큰 수/솔루션(CLAUDE.md 지표) 집계 미실행 — `results/raw/`의 `usage.completion_tokens`
   에서 바로 뽑을 수 있음, `scripts/analyze.py` 아직 이 집계 없음.

## 8. 하드웨어·환경 (논문 Setup 섹션용, CLAUDE.md 원본과 동일)

RTX PRO 6000 Blackwell Workstation Edition, sm_120(PTX `.target sm_120a`),
드라이버 595.84, CUDA 13.2(드라이버)/12.8.93(사용 nvcc), torch 2.8.0+cu128,
triton 3.4.0, tilelang 0.1.13, vLLM 0.27.1. **문헌의 A6000 수치와 비교
금지** — 원 문서가 상정한 하드웨어와 다름(2026-08-19 세션에서 정정).

## 9. 논문 작성 에이전트를 위한 빠른 참조

- 표1(2×2 메인 결과) 소스: 위 §3 표, 원본 `results/eval/full_run_20260819.json`
- 표2(문서 주입 전후) 소스: 위 §4 표, 원본 `results/eval/docinject_run_20260820T072056.json`
  + 비교 대조군은 같은 파일이 아니라 `full_run_20260819.json`에서
  `condition=="0shot" and task in doc_ablation_subset_of_20`으로 필터링해야 함
  (동일 20과제 기준 비교 — 전체 37과제 기준과 섞지 말 것).
- 표3(오류 분류) 소스: 위 §5 — CUDA(gpt-oss 완결/Qwen 미완), TileLang(완결),
  PTX·Triton(개략만).
- 그림1(speedup) 소스: **없음, §7-1 참고**.
- 과제 선정·제외 근거: `tasks/SELECTION.md`
- 정확성 판정 기준값: fp16, atol=rtol=1e-2 (`tasks/SELECTION.md` §4.1)
- 재현성 5요소(체크포인트/vLLM/dtype/샘플링/GPU) 로그 위치: `results/raw/*/*/*/*/sample_*.json`
  각 레코드 자체 필드, `logs/vllm/{gptoss,qwen}_manifest.json`

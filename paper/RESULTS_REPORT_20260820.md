# kernel-lang-2x2 — 결과 보고서 (2026-08-20 기준)

논문작성 에이전트/PI 공유용. 마감 2026-08-29 (AoE), NeurIPS 워크숍 4쪽.
연구 질문·가설·관련 연구는 `RESEARCH_CONTEXT.md`, 실험 설계·방법론 규칙은
`CLAUDE.md`가 원본이다 — 이 문서는 **지금까지 확보된 실측 데이터와 그 해석**만
담는다. 숫자가 바뀌면(재분류·추가 실행) 이 문서를 갱신할 것.

## 0. 한 줄 요약

37과제×4언어×5샘플×2모델 0-shot 본 실행(1,480건)과 20과제 층화 부분집합 문서
주입(docinject) ablation(800건) 모두 완료·전수검증 통과. **주 분석 기준은
2026-08-20 오후부로 32과제(37 − 결함 5과제)로 전환됨 — 이 문서의 표1/표2/
speedup은 전부 32과제(또는 docinject는 17과제) 기준을 주 열로, 37/20과제
원판정을 부록으로 병기한다.** **Triton만 유의미한 정답률**(32과제 기준
0-shot: gpt-oss 30.6%, Qwen 8.75%); CUDA/PTX/TileLang은 정답 0%로 API/문법
실패가 지배적. 문서 주입은 **gpt-oss엔 4개 언어 전부 강하게 도움**이 되지만
**Qwen엔 TileLang만 돕고 CUDA·PTX는 오히려 0%로 붕괴**시킨다 — 모델 의존적
효과로, H3("문서 주입이 LRPL 격차를 줄인다")를 단순 지지하지 않는다. **타이밍
(speedup) 측정 완료(§7-1, 226/228)** — 정답 커널은 대체로 PyTorch eager보다
느리다. **허용오차 감사 완료(§7-2): 37과제 중 5개(13.5%)가 fp16
atol=rtol=1e-2 아래서 정확성 판정을 신뢰할 수 없다** — `23_Softmax`(출력이
허용오차보다 4,000배 작음), `4_Matrix_vector_multiplication_`·
`6_Matmul_with_large_K_dimension_`(참조 자체가 fp16 오버플로로 전 원소
`inf`), `90_cumprod`(출력 다수가 0으로 언더플로), `95_CrossEntropyLoss`(참조
loss가 fp16 오버플로, 관련 라벨-dtype 하니스 버그는 **수정하지 않기로 결정**
— 이 과제 제외로 발현 경로 소멸, known issue로만 기록). **A100(Ampere)
교차검증 완료(§3-A)**: 정성적 패턴(Triton 지배, CUDA/PTX 붕괴)이 하드웨어
간 재현됨. **멀티턴 프로토콜(§3.4) 확정, 결함 5과제는 수리 국면 포함
전체에서 제외**(체인 1,960개: 본 실행 32과제×4×5×2=1,280 + docinject
17과제×4×5×2=680) — **아직 실행 전, P0-b(생성 동시성·컴파일 병렬화) 구현 후 착수 예정.**

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

## 3. 본 실행 결과 (0-shot)

**주 분석 기준 전환 (PI, 2026-08-20): 32과제(37 − §7-2 결함 5과제)를 논문의
주 분석 기준으로 확정.** 이하 표는 32과제 기준을 주 열로, 37과제 원판정을
부록 열로 병기한다. `scripts/analyze.py`로 재산출(원본 판정 파일 자체는
변경하지 않음 — 집계 시 과제 필터만 적용).

| 언어 | 모델 | compiled(32) | correct(32) | compiled(37,부록) | correct(37,부록) | 비고 |
|---|---|---:|---:|---:|---:|---|
| cuda | gpt-oss-120b | 1/160 (0.6%) | 0/160 | 1/185 (0.5%) | 0/185 | API/문법 실패 지배 (§5) |
| cuda | Qwen3-Coder-30B | 45/160 (28.1%) | 0/160 | 51/185 (27.6%) | 0/185 | `-std=c++14` 하니스 우회 적용 후 수치(§6) |
| ptx | gpt-oss-120b | 130/160 (81.3%) | 0/160 | 149/185 (80.5%) | 0/185 | truncated (32과제 기준 집계에 포함) |
| ptx | Qwen3-Coder-30B | 84/160 (52.5%) | 0/160 | 106/185 (57.3%) | 0/185 | **truncated 다수** — 분모 주의(§5) |
| tilelang | gpt-oss-120b | 1/160 (0.6%) | 0/160 | 2/185 (1.1%) | 0/185 | API 환각 지배 (§5) |
| tilelang | Qwen3-Coder-30B | 0/160 (0%) | 0/160 | 0/185 (0%) | 0/185 | API 환각 지배 (§5) |
| **triton** | gpt-oss-120b | 154/160 (96.3%) | **49/160 (30.6%)** | 179/185 (96.8%) | 59/185 (31.9%) | 4언어 중 유일하게 유의미한 정답 |
| **triton** | Qwen3-Coder-30B | 139/160 (86.9%) | **14/160 (8.75%)** | 163/185 (88.1%) | 24/185 (13.0%) | |

**32과제 기준 전체 정확률(4언어 합산, n=1280): 63/1280 = 4.9%, 전부 Triton.**
(37과제 기준: 83/1480 = 5.6%.) 두 모델의 Triton 정답 중 상당수가 결함
5과제(특히 `23_Softmax`, `6_Matmul_with_large_K_dimension_`)에 몰려 있었음을
반영 — gpt-oss 59→49(−17%), Qwen 24→14(−42%). **Qwen 쪽 하락폭이 더 커서,
32과제 기준으로 보면 gpt-oss와 Qwen의 Triton 정답률 격차가 37과제 기준보다
더 벌어진다**(37과제: 31.9%/13.0%, 격차 2.45배 → 32과제: 30.6%/8.75%, 격차
3.5배).

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
  컴파일률이 0.6%→65.9%로 회복됐는데(32/17과제 정제 기준, §4), 이는 정확히
  "관용구(pybind 바인딩 등)를
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

### 3-A. 하드웨어 교차검증 (A100, PI ②, 2026-08-20)

협업자(hyunjun1234)가 `results-a100` 브랜치에 별도 서버(NVIDIA A100 80GB
PCIe, Ampere sm_80)에서 0-shot 3언어(cuda/ptx/triton — **tilelang 없음**,
이 프로브 범위 밖)×37과제×5샘플×2모델 = 1,110건을 생성·평가해 push했다.
`scripts/analyze.py`로 32과제 기준 재집계(전수검증: 1,110/1,110, 중복 0,
HF 체크포인트 리비전은 PRO 6000과 동일). **스택 차이는 명시**: vLLM
0.10.1(PRO6000은 0.27.1), torch 2.5.1+cu121(PRO6000은 2.8.0+cu128), 드라이버
535.309.01. base_seed도 다름(4000 vs 200) — 개별 표본 단위 재현이 아니라
집계 비율 비교 목적.

| | | PRO 6000 (Blackwell) 32과제 | A100 (Ampere) 32과제 |
|---|---|---:|---:|
| cuda | gpt-oss-120b | 1/160 (0.6%), 0 correct | 3/160 (1.9%), **1 correct** |
| cuda | Qwen3-Coder-30B | 45/160 (28.1%), 0 correct | 36/160 (22.5%), 0 correct |
| ptx | gpt-oss-120b | 130/160 (81.3%), 0 correct | 133/160 (83.1%), 0 correct |
| ptx | Qwen3-Coder-30B | 84/160 (52.5%), 0 correct | 98/160 (61.3%), 0 correct |
| triton | gpt-oss-120b | 154/160 (96.3%), 49 correct | 151/160 (94.4%), 55 correct |
| triton | Qwen3-Coder-30B | 139/160 (86.9%), 14 correct | 140/160 (87.5%), 21 correct |

**정성적 패턴이 하드웨어 간에 재현된다** — Triton만 유의미한 정답,
CUDA/PTX는 사실상 0%, 컴파일률 크기도 대체로 비슷한 범위. gpt-oss CUDA가
A100에서 1건 정답(PRO6000은 0건)이 유일한 질적 차이인데 표본 1건이라
결론에 영향 없음. Triton 정답 수는 A100이 양쪽 모델 다 더 높다(49→55,
14→21) — 다른 vLLM/torch 스택·아키텍처 차이가 원인일 수 있으나 원인
규명은 범위 밖, **상관만 보고**(§3의 인과 서술 금지 원칙과 동일하게 적용).
결론: **본 실행의 핵심 발견(Triton 지배, CUDA/PTX/TileLang 붕괴)이 단일
GPU 아키텍처의 우연이 아니라는 방증**으로 논문 부록에 넣을 것을 제안.

## 4. 문서 주입(docinject) Ablation 결과

과제 선정: `tasks/level1_subset.json`의 `doc_ablation_subset_of_20`, 37과제
계열 분포와 전 계열 5%p 이내로 비례하는 층화 표본(승인 근거:
`tasks/SELECTION.md` §4.3). 주입 문서: `prompts/specs/{cuda,ptx,triton,tilelang}.md`
(~5k 토큰/언어, 4언어 동일 조건 적용).

**주 분석 기준 전환**: 이 20과제 부분집합 중 **3개가 §7-2 결함 5과제와
겹친다**(`23_Softmax`, `6_Matmul_with_large_K_dimension_`,
`95_CrossEntropyLoss` — `4_Matrix_vector_multiplication_`·`90_cumprod`은
애초에 20과제 부분집합에 없음). 그래서 docinject의 **감사 통과 기준은
17과제**다. 이하 표는 17과제(n=85/셀)를 주 열로, 20과제(n=100/셀) 원판정을
부록 열로 병기한다.

| 언어 | 모델 | 0-shot(17) | docinject(17) | Δcompiled(17) | Δcorrect(17) | *0-shot(20,부록)* | *docinject(20,부록)* |
|---|---|---:|---:|---:|---:|---:|---:|
| cuda | gpt-oss-120b | 0/85, 0/85 | 56/85, **42/85** | +56 | **+42** | *0/100, 0/100* | *66/100, 48/100* |
| cuda | Qwen3-Coder-30B | 25/85, 0/85 | **0/85**, 0/85 | **−25** | 0 | *30/100, 0/100* | *0/100, 0/100* |
| ptx | gpt-oss-120b | 68/85, 0/85 | 74/85, 1/85 | +6 | +1 | *79/100, 0/100* | *88/100, 1/100* |
| ptx | Qwen3-Coder-30B | 43/85, 0/85 | **0/85**, 0/85 | **−43** | 0 | *55/100, 0/100* | *0/100, 0/100* |
| tilelang | gpt-oss-120b | 1/85, 0/85 | 79/85, **25/85** | +78 | **+25** | *2/100, 0/100* | *92/100, 29/100* |
| tilelang | Qwen3-Coder-30B | 0/85, 0/85 | 70/85, **15/85** | +70 | **+15** | *0/100, 0/100* | *82/100, 16/100* |
| triton | gpt-oss-120b | 81/85, 17/85 | 83/85, 28/85 | +2 | +11 | *96/100, 25/100* | *98/100, 32/100* |
| triton | Qwen3-Coder-30B | 70/85, 11/85 | 74/85, 12/85 | +4 | +1 | *85/100, 19/100* | *89/100, 19/100* |

17과제 기준으로도 20과제 원판정과 결론의 방향·크기는 거의 동일하다 —
`23_Softmax` 등 결함 과제가 이 특정 표(compiled/correct 카운트, speedup
아님)에는 큰 왜곡을 만들지 않았다(왜곡은 주로 §7-1 speedup 집계에서
발생). 다만 절대 수치를 인용할 땐 17과제 기준을 쓸 것.

### H3 판정: 지지되지 않음, 모델 의존적 (논문에서 반드시 이렇게 프레이밍할 것)

- **gpt-oss**: 4개 언어 전부 개선, 특히 CUDA(0→49.4%)·TileLang(0→29.4%)에서
  극적(17과제 기준). LRPL(PTX·TileLang)뿐 아니라 HRPL(CUDA)도 크게
  개선됐다는 점에서, "문서 주입이 LRPL 격차를 좁힌다"는 H3의 좁은 버전
  (LRPL에만 효과)은 성립하지 않는다 — gpt-oss에게는 **자원 수준과 무관하게
  구조화된 스펙 자체가 도움**이 된 것으로 보인다.
- **Qwen**: TileLang만 개선(0→17.6%), CUDA·PTX는 **0-shot보다 더 나빠져 0%로
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
비교표로 쓰려면 이 114건을 격리 재컴파일로 재수집해야 한다(§8 TODO 참고,
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
70%). 세부 수치 오류 원인 분류는 아직 안 함 — §8 TODO.

---

## 6. 하니스 변경 이력 (전부 git 커밋, `prompts/PROMPT_SPEC.md` §7이 원본)

| 날짜 | 변경 | 근거 | 판정 영향 |
|---|---|---|---|
| 08-19 | CUDA `-std=` 모델 지정 플래그 스트리핑(C++17 강제) | Qwen CUDA 51/185(27.6%)가 `-std=c++14` 직접 지정, 하니스 기본값과 충돌 | Qwen CUDA만, 재평가함 |
| 08-19 | eval 샘플별 서브프로세스 격리 + 샘플 단위 체크포인트 | CUDA illegal-memory-access가 프로세스 전체를 오염시켜 1,480건 실행 유실한 사고 | 없음(안정성) |
| 08-20 | `evaluate.py --resume` 추가 | 체크포인트 이어받기, 재평가 금지 | 없음 |
| 08-20 | 컴파일 진단 로깅 캡 2000/4000→20000자 | 704건 중 197건이 진단 절단(모델 간 비대칭: gpt-oss CUDA 172/181 vs Qwen 25/124) | **없음(순수 로깅)** — 기존 절단분 재수집은 미완(§8 TODO) |
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

**이상치 처리 방식 (PI, 2026-08-20 수정): 샘플 단위가 아니라 과제 단위 대칭
제외.** 애초 9개 플래그 샘플만 빼는 안은 기각됐다 — 근거는 §7-2 허용오차
감사에서 확정됐다: `23_Softmax`뿐 아니라 **`4_Matrix_vector_multiplication_`,
`6_Matmul_with_large_K_dimension_`도 correct 판정이 신뢰 불가**임이 드러나,
이 timing 데이터셋(228건)에 실제로 등장하는 **3개 과제 전부**를 speedup
집계(geomean·fast_1 둘 다)에서 **전 셀 대칭 제외**했다(§7-2에서 걸린 나머지
2개, `90_cumprod`/`95_CrossEntropyLoss`는 이 데이터셋에 correct 표본이 아예
없어 이 표엔 영향 없음 — §7-2·§7-3 멀티턴 조치는 5개 전부 대상). 원값/정제값을
병기한다. **정확성 표(§3·§4)의 판정 자체는 변경하지 않는다** — compiled/
correctness는 그대로, speedup 집계에서만 이 3개 과제를 뺀다.

| lang | model | condition | n(원) | fast_1(원) | geomean(원) | n(정제) | fast_1(정제) | geomean(정제) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cuda | gpt-oss-120b | docinject | 47 | 8 (17.0%) | 0.249x | 41 | 8 (19.5%) | 0.247x |
| ptx | gpt-oss-120b | docinject | 1 | 1 (100%) | 1.00x | 1 | 1 (100%) | 1.00x |
| tilelang | gpt-oss-120b | docinject | 28 | 8 (28.6%) | 0.487x | 24 | 8 (33.3%) | 0.713x |
| tilelang | Qwen3-Coder-30B | docinject | 16 | 4 (25.0%) | 0.214x | 15 | 3 (20.0%) | 0.170x |
| triton | gpt-oss-120b | 0shot | 59 | 26 (44.1%) | 0.797x | 49 | 26 (53.1%) | 1.14x |
| triton | gpt-oss-120b | docinject | 32 | 15 (46.9%) | 0.576x | 28 | 15 (53.6%) | 0.797x |
| **triton** | **Qwen3-Coder-30B** | **0shot** | 24 | 18 (75.0%) | 2.13x | 14 | 13 (92.9%) | **1.12x** |
| **triton** | **Qwen3-Coder-30B** | **docinject** | 19 | 10 (52.6%) | 2.38x | 12 | 6 (50.0%) | **0.849x** |

**논문·분석에는 정제값(오른쪽 3열)을 쓸 것.** 방향이 셀마다 다르다는 점에
주의: gpt-oss 계열은 오히려 **위로** 이동한다(0.797x→1.14x, 0.576x→0.797x,
0.487x→0.713x) — 이 3개 과제에서 gpt-oss 커널의 speedup이 유독 낮았다는
뜻이므로 제외하면 평균이 올라간다. triton|Qwen 두 셀은 **크게 낮아진다**
(2.13x/2.38x → 1.12x/0.849x) — `23_Softmax`의 405~1,350x 허위 speedup이
빠지는 효과가 지배적이다. **정정 (2026-08-23, 표 자체와 모순되는 각주 발견)**:
"8셀 중 유일하게 triton|gpt-oss-120b|0shot이 50%를 넘는다"는 틀렸다 — 바로 이
표에서 triton|gpt-oss-120b|docinject도 53.6%(15/28), triton|Qwen3-Coder-30B|0shot도
92.9%(13/14)로 50%를 넘고, triton|Qwen3-Coder-30B|docinject은 정확히 50.0%(6/12)로
동률이다. 정제 후 **8셀 중 3셀이 fast_1 50%를 넘고(모두 triton), 1셀은 동률** —
"유일하게"가 아니라 "triton 4셀 중 3셀 초과·1셀 동률, non-triton 4셀은 전부 미만"이
정확한 서술이다. fast_1의 분모는 과제 단위가 아니라 **정답 판정 + 타이밍 측정
둘 다 성공한 샘플 단위**다(`scripts/analyze.py`의 `speedup_table()`이 호출하는
`agg()`: records를 (language, model, condition)별로 묶어 그 안에서
`speedup > 1`인 레코드 개수를 세는 방식 — 레코드 1개 = 샘플 1개, 과제가 아님).
표3 서술 시 이 정정된 문장을 참고.

### 근거: 왜 과제 전체를 신뢰할 수 없나

**`23_Softmax`**: 코드를 직접 확인 — `get_inputs()`가 `(4096, 393216)`(행
하나가 393,216열)인데 Qwen 생성 Triton 커널 다수가 `BLOCK_SIZE=1024`를
하드코딩하고 열 방향 루프/그리드 분할을 하지 않아 각 행의 앞 1024열만
계산하고 나머지는 `torch.empty_like`의 미초기화 메모리로 남긴다. fp16
atol=rtol=1e-2가 이 과제의 정상 출력 크기(≈2.5e-6)보다 4,000배 커서 통과했다.

**`4_Matrix_vector_multiplication_`, `6_Matmul_with_large_K_dimension_`**:
§7-2에서 새로 확인 — 이 두 과제는 **참조(PyTorch eager) 모델의 출력 자체가
fp16에서 오버플로해 전 원소가 `inf`**다(K=1,048,576 / 524,288의 큰 축적
차원, 입력이 `torch.rand`(U[0,1))라 기댓값상 출력이 수십만 단위로 fp16
최댓값 65504를 가볍게 넘긴다). `torch.allclose`는 부호가 같은 무한대끼리는
같다고 처리하므로, **생성된 커널이 내부적으로 fp32 누산 등을 써서 우연히
같은 방향으로 오버플로하기만 하면** 실제 계산 내용과 무관하게 통과할 수
있다. 이건 하니스 버그가 아니라 **§4.1의 fp16 통일 결정이 이 두 과제(원래
fp32/fp64 기준으로 선정됐을 축적 차원)와 만나 상정 못한 조합**을 만든
경우다.

두 부류 다 **하니스 버그가 아니라 특정 과제 × fp16 통일 정밀도의 조합이
정확성 판정을 무력화한 사례**이며, gpt-oss의 해당 과제 샘플들(speedup이
정상 범위로 보인 것들 포함)도 **같은 취약점 아래서 나온 결과라 마찬가지로
신뢰할 근거가 없다** — 이게 표본 단위가 아니라 과제 단위로 대칭 제외한
이유다. 정확성 판정 자체(§3·§4의 compiled/correctness)는 바꾸지 않는다 —
캐비어트로만 남긴다.

## 7-2. 허용오차 감사 (P0-b 착수 전 필수, 2026-08-20 완료)

37과제 전부에 대해 참조(PyTorch eager, fp16) 모델을 1회 forward해 출력
원소 크기(중앙값 |output|)를 재고, fp16 correctness 허용오차(atol=rtol=1e-2,
`kernelbench.eval.get_tolerance_for_precision`에서 직접 읽음)와 비교했다.
CPU 전용(`scripts/audit_tolerance.py`, GPU·배타성 게이트 불필요 — 참조
모델만 1회 순전파, 커스텀 커널 없음). 결과 파일: `results/eval/tolerance_audit.json`.

**판정 기준 2가지, 둘 다 "atol/median" 비율 하나로는 못 잡는 게 있어서 별도
플래그로 분리**:
- **과다-관용**: `atol / median|output| > 10` — 정상 출력값이 허용오차보다
  훨씬 작아서, 거의 아무 값이나 허용오차 안에 들어와 버리는 경우.
- **참조-오버플로**: 참조 모델 출력 자체에 `inf`/`nan`이 있는 경우 —
  `atol/median` 비율 계산으로는 오히려 "문제없음"처럼 보이는 사각지대라
  (유한수/무한대=0) 별도로 직접 `torch.isinf`/`isnan`을 검사했다.

**5/37 과제(13.5%) 플래그** — 이 중 3개(`23_Softmax`,
`4_Matrix_vector_multiplication_`, `6_Matmul_with_large_K_dimension_`)는
§7-1의 228건 correct 표본 중 **42건(18.4%)**에 실제로 걸려 있었다(§7-1에
반영 완료). 나머지 2개(`90_cumprod`, `95_CrossEntropyLoss`)는 이번 실행
데이터엔 correct 표본이 없어(`95_CrossEntropyLoss`는 0/79 항상 컴파일
단계에서 실패) 이번 speedup 표엔 영향 없지만, **멀티턴에서는 새로 correct에
도달할 수 있으므로** ③ 조치 대상에 포함한다(아래).

| task | 참조 출력 median\|abs\| | atol/median | inf 원소 수 | 플래그 |
|---|---:|---:|---:|---|
| 4_Matrix_vector_multiplication_ | nan | inf | 2048 | 과다-관용/참조-오버플로 |
| 6_Matmul_with_large_K_dimension_ | nan | inf | 65536 | 과다-관용/참조-오버플로 |
| 90_cumprod | 0 | inf | 0 | 과다-관용 |
| 95_CrossEntropyLoss | nan | inf | 1 | 과다-관용/참조-오버플로 |
| 23_Softmax | 2.44e-06 | 4.09e+03 | 0 | 과다-관용 |
| 39_L2Norm_ | 0.00338 | 2.96 | 0 | - |
| 94_MSELoss | 0.152 | 0.066 | 0 | - |
| 67_conv_standard_1D | 0.207 | 0.0482 | 0 | - |
| 76_conv_standard_1D_dilated_strided__ | 0.207 | 0.0482 | 0 | - |
| 50_conv_standard_2D__square_input__square_kernel | 0.223 | 0.0449 | 0 | - |
| 57_conv_transposed_2D__square_input__square_kernel | 0.226 | 0.0442 | 0 | - |
| 82_conv_depthwise_2D_square_input_square_kernel | 0.257 | 0.0389 | 0 | - |
| 54_conv_standard_3D__square_input__square_kernel | 0.259 | 0.0387 | 0 | - |
| 25_Swish | 0.311 | 0.0321 | 0 | - |
| 26_GELU_ | 0.346 | 0.0289 | 0 | - |
| 22_Tanh | 0.462 | 0.0216 | 0 | - |
| 19_ReLU | 0.5 | 0.02 | 0 | - |
| 44_Average_Pooling_1D | 0.5 | 0.02 | 0 | - |
| 48_Mean_reduction_over_a_dimension | 0.5 | 0.02 | 0 | - |
| 97_ScaledDotProductAttention | 0.5 | 0.02 | 0 | - |
| 21_Sigmoid | 0.623 | 0.0161 | 0 | - |
| 35_GroupNorm_ | 0.866 | 0.0115 | 0 | - |
| 40_LayerNorm | 0.866 | 0.0115 | 0 | - |
| 33_BatchNorm | 0.867 | 0.0115 | 0 | - |
| 36_RMSNorm_ | 0.869 | 0.0115 | 0 | - |
| 41_Max_Pooling_1D | 0.917 | 0.0109 | 0 | - |
| 42_Max_Pooling_2D | 0.958 | 0.0104 | 0 | - |
| 100_HingeLoss | 0.996 | 0.01 | 0 | - |
| 49_Max_reduction_over_a_dimension | 1 | 0.01 | 0 | - |
| 3_Batched_matrix_multiplication | 256 | 3.91e-05 | 0 | - |
| 8_Matmul_with_irregular_shapes_ | 737 | 1.36e-05 | 0 | - |
| 1_Square_matrix_multiplication_ | 1.02e+03 | 9.77e-06 | 0 | - |
| 51_Argmax_over_a_dimension | 1.4e+03 | 7.15e-06 | 0 | - |
| 17_Matmul_with_transposed_B | 2.05e+03 | 4.88e-06 | 0 | - |
| 2_Standard_matrix_multiplication_ | 2.05e+03 | 4.88e-06 | 0 | - |
| 47_Sum_reduction_over_a_dimension | 2.05e+03 | 4.88e-06 | 0 | - |
| 89_cumsum | 8.19e+03 | 1.22e-06 | 0 | - |

**37과제 전체 결과 순위표는 파일 하단 부록에 이어짐 — 전체 데이터는
`results/eval/tolerance_audit.json` 참고.** 참고로 다음으로 비율이 높은
비플래그 과제는 `39_L2Norm_`(2.96x, 10x 미만이라 플래그 안 됨) — 경계에
가까운 편이니 표3 해석 시 참고.

### 추가 발견 — `_process_input_tensor` 하니스 함수의 label-dtype 버그 — **PI 결정: 수정하지 않음 (2026-08-20)**

감사 스크립트를 만들다가 발견: KernelBench의 `_process_input_tensor`(CUDA/
Triton/TileLang 정확성 경로와 `evaluate.py`의 PTX 경로 전부가 공유)가
`get_inputs()`가 반환하는 **모든** 텐서를 무조건 `precision`(fp16)으로
캐스팅한다 — `95_CrossEntropyLoss`의 정수 클래스 인덱스 타겟(`torch.randint`,
Long이어야 함)도 예외 없이 캐스팅되어 `nn.CrossEntropyLoss`가
`RuntimeError: expected scalar type Long but found Half`로 죽는다.
**지금까지는 이 버그가 한 번도 발현되지 않았다** — 이 과제의 실제 샘플
79/79 전부가 컴파일 단계 등 더 이른 지점에서 이미 실패해서다(우연).
하지만 **멀티턴 수리 국면에서 모델이 여러 번 시도하다 컴파일을 통과시키는
순간 이 버그가 발현**되고, 그러면 커널 로직과 무관하게 항상
`correctness=False`(runtime_error)가 나온다 — 수리 피드백이 "니 커널이
틀렸다"고 말하지만 실은 하니스 자체의 타입 캐스팅 버그다. **이건 이번 ②
감사가 요청받은 atol 비율 문제와는 다른 종류의 버그이고, 고칠지 말지는
범위 밖이라 PI 판단이 필요하다** — 옵션: (a) `_process_input_tensor` 호출
전에 정수/불리언 타입 텐서는 캐스팅에서 제외하도록 하니스 수정(harness 레벨
변경, 전 언어 동시 적용 필요), (b) `95_CrossEntropyLoss`를 아예 과제
목록에서 제외, (c) 그대로 두고 "이 과제는 구조적으로 항상 실패한다"로
캐비어트만 남김. 이 감사 스크립트(`scripts/audit_tolerance.py`)는 정수
텐서를 캐스팅에서 제외해 이 버그를 **재현하지 않고** 우회했다(측정
목적으로만 — 실제 하니스는 그대로 둠).

**PI 결정 (2026-08-20): 옵션 (c) 채택 — 하니스는 수정하지 않는다.**
근거: `95_CrossEntropyLoss`가 §7-3에 따라 **멀티턴 전체(수리 국면 포함)에서
과제 목록 자체에서 빠지므로**, 이 버그가 발현될 수 있는 유일한 경로(모델이
컴파일을 통과시키는 순간)가 애초에 사라진다. 하니스 수정(옵션 a, 전 언어
동시 적용 필요한 프로토콜 변경) 없이 known issue로만 기록하고 넘어간다 —
이 문서와 `scripts/audit_tolerance.py`의 주석이 재현 방법과 함께 그 기록이다.

### 7-3. 멀티턴 프로토콜 수정 — 결함 5과제 완전 제외 (PI, 2026-08-20 확장)

**최적화 국면만 제외에서 멀티턴 전체 제외로 확장.** PROMPT_SPEC §3.4에
기록 완료. 사유가 과제 부류별로 다르다: **참조-오버플로 3과제**
(`4_Matrix_vector_multiplication_`, `6_Matmul_with_large_K_dimension_`,
`95_CrossEntropyLoss`)는 참조 자체가 `inf`라 **수리 국면의 정확성 실패
피드백(최대절대오차·불일치 비율)조차 무의미**(inf 기준 오차는 모델에게
줄 수 있는 유용한 신호가 아님) — 수리 국면 유지가 의미 없으므로 통째로
제외. **과다-관용 2과제**(`23_Softmax`, `90_cumprod`)는 정확성 판정
자체가 신뢰 불가라 correct 도달이 진짜 정답인지 판별 불가 — 이 과제로
체인을 진행해봐야 측정할 신호가 없으므로 역시 통째로 제외.

**멀티턴 체인 구성 (PI 확인, 2026-08-20): 워크스트림별로 기존 과제 범위를
유지한 채 결함 5과제만 제외** — docinject를 32과제로 확장하지 않는다(원래
승인된 20과제 층화 ablation 설계, `tasks/SELECTION.md` §4.3를 그대로 유지).

| 워크스트림 | 과제 범위 | 체인 수 (언어4×샘플5×모델2) |
|---|---:|---:|
| 본 실행 | 32과제(37−5) | 32×4×5×2 = **1,280** |
| docinject | 17과제(20−3, §4 참고) | 17×4×5×2 = **680** |
| **합계** | | **1,960** (원래 혼합 기준 2,280에서 재계산) |

이 5개 과제는 **정확성 판정이 성능(speedup) 피드백 아래서 부분 계산·
오버플로 일치 같은 "해킹"을 걸러낼 수 없어 수리 국면조차 신호가 없다** —
그래서 최적화 국면뿐 아니라 **체인 자체를 구성하지 않는다**(위 표에 이미
반영). 사유는 PROMPT_SPEC §3.4·§7 변경 이력에 기록.

## 8. 미완료 / TODO (논문 작성 전 확인 필요)

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

## 9. 하드웨어·환경 (논문 Setup 섹션용, CLAUDE.md 원본과 동일)

RTX PRO 6000 Blackwell Workstation Edition, sm_120(PTX `.target sm_120a`),
드라이버 595.84, CUDA 13.2(드라이버)/12.8.93(사용 nvcc), torch 2.8.0+cu128,
triton 3.4.0, tilelang 0.1.13, vLLM 0.27.1. **문헌의 A6000 수치와 비교
금지** — 원 문서가 상정한 하드웨어와 다름(2026-08-19 세션에서 정정).

## 10. 논문 작성 에이전트를 위한 빠른 참조

- 표1(2×2 메인 결과) 소스: 위 §3 표, 원본 `results/eval/full_run_20260819.json`
- 표2(문서 주입 전후) 소스: 위 §4 표, 원본 `results/eval/docinject_run_20260820T072056.json`
  + 비교 대조군은 같은 파일이 아니라 `full_run_20260819.json`에서
  `condition=="0shot" and task in doc_ablation_subset_of_20`으로 필터링해야 함
  (동일 20과제 기준 비교 — 전체 37과제 기준과 섞지 말 것).
- 표3(오류 분류) 소스: 위 §5 — CUDA(gpt-oss 완결/Qwen 미완), TileLang(완결),
  PTX·Triton(개략만).
- 그림1(speedup) 소스: §7-1, 정제값(3개 과제 대칭 제외 후) 표 사용 — 원값 아님.
- 표4(허용오차 감사) 소스: §7-2 전체, 원본 `results/eval/tolerance_audit.json`.
- 과제 선정·제외 근거: `tasks/SELECTION.md`
- 정확성 판정 기준값: fp16, atol=rtol=1e-2 (`tasks/SELECTION.md` §4.1) — 단
  5개 과제(§7-2)는 이 기준값 자체가 해당 과제 출력 크기와 안 맞아 사실상
  무의미, 캐비어트 필수.
- 재현성 5요소(체크포인트/vLLM/dtype/샘플링/GPU) 로그 위치: `results/raw/*/*/*/*/sample_*.json`
  각 레코드 자체 필드, `logs/vllm/{gptoss,qwen}_manifest.json`

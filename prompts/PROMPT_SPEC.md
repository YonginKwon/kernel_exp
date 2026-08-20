# PROMPT_SPEC.md — 프롬프트 템플릿 명세 (초안 v0.1, PI 검토용)

설계 원칙: 언어 간 차이는 §2의 "언어 블록"에만 존재한다. 나머지는 전 언어 공통.
이 파일의 확정본이 prompts/의 유일한 진실이 되며, 변경은 git commit + 전 언어 동시 적용.

## 1. 공통 템플릿 (모든 언어·모든 조건 동일)

```
You are given a PyTorch reference implementation of a neural network operator.
Write a functionally equivalent GPU kernel in {LANGUAGE}.

Requirements:
- All tensor inputs and outputs are float16 (fp16). Accumulation precision is
  your choice unless the reference dictates otherwise.
- Target GPU: {TARGET_GPU_LINE}.
- Your kernel must be a drop-in replacement: same input shapes, same output
  shapes and dtype as the reference.
- Output ONLY the complete {LANGUAGE} solution inside a single code block.
  No explanation outside the code block.

{LANGUAGE_BLOCK}

Reference implementation:
```python
{REFERENCE_CODE}
```
```

주: KernelBench 표준 입출력 규약(Model/ModelNew 클래스 형식)은 하니스가 요구하는
형태에 맞춰 {LANGUAGE_BLOCK} 안의 "출력 형식" 항목으로 통일 지정한다.

## 2. 언어 블록 (언어당 차이가 허용되는 유일한 부분)

각 블록은 다음 3요소만 포함한다 — ① 출력 형식(하니스가 요구하는 래핑),
② 컴파일 대상 명시, ③ 문법 미주입 선언. 예시 커널·최적화 힌트·성능 조언 금지.

### CUDA
```
Write CUDA C++ using inline PyTorch extension conventions
(torch.utils.cpp_extension.load_inline). Provide the full Python file
defining ModelNew that compiles and launches your CUDA kernel.
Do not pass a -std= (C++ standard) flag in extra_cflags or
extra_cuda_cflags; the harness supplies the required C++ standard
uniformly and overrides any you provide.
```

### Triton
```
Write a Triton kernel (triton.jit) with a Python wrapper. Provide the full
Python file defining ModelNew that calls your Triton kernel.
```

### PTX
```
Write the complete kernel in raw PTX ISA (.target {PTX_TARGET}). Provide:
(1) the PTX module as a Python string constant, and
(2) a ModelNew class that loads it via the provided ptx_harness API:
    module = ptx_load(PTX_SOURCE); ptx_launch(module, "kernel_name", grid, block, args).
The harness handles module loading; you write only the PTX and launch parameters.
```

### TileLang
```
Write a TileLang kernel (tilelang.jit / T.prim_func). Provide the full
Python file defining ModelNew that calls your TileLang kernel.
```

## 3. 조건별 변형

### 3.1 0-shot (기본)
위 템플릿 그대로.

### 3.2 문서 주입 (ablation, 과제 20개 부분집합)
공통 템플릿의 {LANGUAGE_BLOCK} 뒤에 삽입:

```
Language reference (for your use):
{LANGUAGE_SPEC_5K}
```

- {LANGUAGE_SPEC_5K}: 언어별 공식 문서에서 발췌한 ~5k 토큰 명세.
  4개 언어 모두 동일 예산(5k±10% 토큰), 동일 구성(문법 개요 → 메모리 모델 →
  최소 완전 예제 1개)으로 작성. HRPL에도 동일하게 주입한다 (LRPL에만 주입하면
  조건 간 비교가 무너짐).
- 발췌 출처를 prompts/specs/SOURCES.md에 기록.

### 3.3 수리 1턴 — **폐기, §3.4로 대체 (PI, 2026-08-20)**

아래 초안(컴파일 실패에만 수리 1회, 정확성 실패는 즉시 오답 확정)은 본 실행
(0-shot 1,480건)·docinject ablation(800건)의 턴 1을 만드는 데만 쓰였고,
멀티턴 완주 프로토콜(§3.4)로 대체됐다 — 역사 기록으로 남긴다.

```
Your previous solution failed to compile with the following error:

{COMPILER_ERROR_VERBATIM}

Provide the corrected complete solution in a single code block.
```

### 3.4 멀티턴 완주 프로토콜 (PI 확정, 2026-08-20)

**목적**: k=10턴까지(또는 조기 종료 조건까지) 밀어붙여 pass@5 단일 시도의
한계를 넘어선 회복률·최적화 궤적을 측정한다. **턴 1은 이미 만들어진
데이터**(본 실행/docinject 각 조건의 0-shot 생성 결과)를 그대로 재사용 —
docinject 체인은 문서 주입 조건을 전 턴 유지, 그 외는 완전히 동일한
프로토콜.

**체인 구성 (PI 확정, 2026-08-20): §7-2 허용오차 감사에서 걸린 5개 과제는
멀티턴 전체(수리 국면 포함)에서 제외 — 워크스트림별 기존 과제 범위는
유지**(docinject를 32과제로 확장하지 않음, `tasks/SELECTION.md` §4.3의
20과제 층화 ablation 설계 그대로):

| 워크스트림 | 과제 수 | 체인 수(언어4×샘플5×모델2) |
|---|---:|---:|
| 본 실행 | 32(37−5) | 1,280 |
| docinject | 17(20−3, 겹치는 결함 과제만 제외) | 680 |
| **합계** | | **1,960** |

**히스토리 무상태**: 매 턴 프롬프트 = {원 과제 프롬프트} + {직전 턴 코드} +
{직전 턴 피드백}만. 대화 누적 금지 — 턴 N의 프롬프트에 턴 1..N-2의 내용이
들어가면 안 됨.

**구조**: 수리 국면(incorrect) ↔ 최적화 국면(correct, 종료 아님) 2단계.

#### 수리 국면 피드백 (incorrect인 동안, 4언어 동일, 3계층 고정)

에러 원문·집계치 외 요약·해석·힌트 추가 금지(에러 메시지의 "LLM 가독성"
측정 목적, §3.3의 원칙 계승). 모든 실패 유형이 동일하게 턴 1개를 소비 —
턴 내부 재시도 루프 없음. 4개 유형 각각 고정 템플릿(`prompts/spec_loader.py`
파싱 대상, 아래 소제목 이름 변경 금지):

##### 컴파일 실패
```
Your previous solution failed to compile with the following error:

{COMPILER_ERROR_VERBATIM}

Provide the corrected complete solution in a single code block.
```
(`COMPILER_ERROR_VERBATIM`은 20,000자 한도 내 원문 — `scripts/evaluate.py`의
`COMPILE_ERROR_LOG_CAP`과 동일 값)

##### 런타임 실패
(컴파일은 됐으나 실행 중 예외)
```
Your previous solution compiled but raised an error at runtime:

{RUNTIME_ERROR_VERBATIM}

Provide the corrected complete solution in a single code block.
```

##### 정확성 실패
(컴파일·실행은 됐으나 출력 불일치 — **집계치만, 참조 텐서 값 절대 포함 금지**)
```
Your previous solution ran but produced incorrect output:

Max absolute error: {MAX_ABS_ERROR}
Fraction of mismatched elements: {MISMATCH_FRACTION}

Provide the corrected complete solution in a single code block.
```

##### 파싱 실패
(truncated / 코드 블록 없음, §5 참고 — 치환 없는 고정 한 줄)
```
Your previous response was truncated / contained no code block. Provide the complete solution in a single code block.
```

#### 최적화 국면 피드백 (correct 도달 시 — 종료 아니라 전환, 4언어 동일 고정 형식)

```
Your kernel: {X} ms (median of 100 runs). PyTorch eager baseline: {Y} ms. Speedup: {Z}x. Improve the kernel's latency while preserving correctness.
```
- 이 형식 외 추가 정보(프로파일러 출력 — ncu/nsight 등 — 절대 포함 금지).
- 매 최적화 턴마다 정확성 재검사 — 틀려지면 그 턴은 **정확성 실패 피드백**을
  받고 수리 국면으로 복귀(위 3계층 중 정확성 항목 적용).
- 체인의 최종 성적은 **best-so-far correct 커널** 기준(가장 마지막이 아님 —
  최적화 도중 퇴화할 수 있으므로).

#### 멀티턴 완전 제외 과제 5개 (§7-2 허용오차 감사, 2026-08-20 확정 — 최적화 국면만이 아니라 체인 자체를 구성하지 않음)

`23_Softmax`, `4_Matrix_vector_multiplication_`,
`6_Matmul_with_large_K_dimension_`, `90_cumprod`, `95_CrossEntropyLoss` —
이 5개 과제는 위 "체인 구성" 표에서 이미 빠져 있다(32/17과제 범위에
포함 안 됨). 근거는 과제 부류별로 다르다(`paper/RESULTS_REPORT_20260820.md`
§7-2·§7-3):
- **참조-오버플로 3과제**(`4_Matrix_vector_multiplication_`,
  `6_Matmul_with_large_K_dimension_`, `95_CrossEntropyLoss`): 참조 모델
  출력 자체가 fp16에서 `inf`라, 수리 국면의 정확성 실패 피드백(최대절대
  오차·불일치 비율)조차 무의미한 신호다 — `inf` 기준 오차를 모델에게
  줘봐야 유용한 정보가 안 됨.
- **과다-관용 2과제**(`23_Softmax`, `90_cumprod`): fp16 atol=rtol=1e-2가
  이 과제들의 출력 크기와 근본적으로 안 맞아 정확성 판정 자체가 신뢰
  불가 — correct 도달이 진짜 정답인지 판별할 수 없으므로 체인을 진행해도
  측정할 신호가 없다.

`95_CrossEntropyLoss`는 부수적으로 §7-2에서 발견된 하니스 버그
(`_process_input_tensor`가 정수 라벨을 fp16으로 캐스팅)의 유일한 발현
경로였는데, 이 과제 제외로 그 경로 자체가 사라진다 — **하니스는 별도로
수정하지 않는다**(PI 결정, 2026-08-20, known issue로만 기록).

이전(2026-08-20 오전) 결정은 "최적화 국면만 제외, 수리 국면은 정상 포함"
이었으나, 같은 날 오후 위 사유로 **수리 국면을 포함한 완전 제외**로
확장됐다 — 역사 기록으로 남긴다.

#### 체인 종료 조건 (셋 중 하나)

① 정답 없이 k 소진, ② 최적화 국면에서 3턴 연속 speedup 무개선, ③ k=10 도달
(위 5개 과제는 correct 도달이 곧 ③에 준하는 즉시 종료).

#### 턴 사이클 실행 (PRO 6000 단독 — A100은 독립 병행 실험, 서빙 오프로드 없음)

1. vLLM 서빙(모델 A) → 생존 체인의 다음 턴 생성 → 서빙 중지 → 모델 B 서빙
   → 생성 → 서빙 중지
2. GPU 클리어 확인(`scripts/evaluate.py`의 `assert_gpu_exclusive()`, 우회
   없음) → 컴파일(CPU 병렬, nvcc 워커 16 이상)·정확성·타이밍 평가 — correct
   체인은 매 턴 타이밍 포함(warmup 25/측정 100/중앙값, §4 CLAUDE.md 프로토콜
   동일)
3. `correct@turn`, `best-speedup@turn`(언어별 geomean) 누적표 로그 기록 →
   다음 턴

**성능 선행 조건 2개 (완성 전 멀티턴 생성 시작 금지)**:
- 생성 동시화: vLLM 동시 요청 16~32개(샘플 독립, 요청별 로깅 유지).
- 컴파일 병렬화: 컴파일(CPU)을 GPU 단계에서 분리, nvcc 워커 16 이상.

**speedup > 10x 커널은 자동 플래그해 수동 검수 목록에 추가(판정은 유지)** —
§7-1에서 이 메커니즘이 `23_Softmax` 취약점을 실제로 잡아낸 전례 있음.

**로깅 규칙**(모델·seed·프롬프트/응답 원문·`prompt_sha256`·환경 버전·
`hardware` 필드) 전 턴 동일 — CLAUDE.md 규칙 4 그대로.

**k 절단 규칙**: k=10 목표, 단 2026-08-25(화) 06:00 KST 시점에 완료된 턴에서
**균일 절단** — 본 실행 체인과 docinject 체인, 4개 언어 전부 같은 턴 수로
끝나야 한다. 도달 턴 수를 보고하고, 진행 중이던 미완 턴의 데이터는 폐기
(부분 턴 사용 금지).

## 4. 생성 파라미터 (전 조건 고정, 2026-08-19 확정)

- temperature: **0.8** (확정), top_p: 기본값, 샘플 수: 5 (독립 호출)
- **seed: 매 호출 명시적으로 고정해 지정하고 로그에 기록** (재현성 확보 목적).
  같은 과제·언어·조건의 샘플 `i`는 `base_seed + i`. vLLM의 OpenAI 호환
  엔드포인트는 `seed` 파라미터를 항상 받으므로 예외 없이 채운다.
- max output tokens: 8192 (PTX가 장문이 되므로 여유 확보; 잘림 발생 시 기록하고
  해당 샘플은 "truncated"로 분류 — 실패의 한 종류로 집계)
- 모델: **오픈웨이트 2개, 로컬 vLLM 서빙** — gpt-oss-120b + Qwen3-Coder-Next-80B-A3B
  (공식 FP8 체크포인트; 96GB 카드 1장에 안 들어가면 강등 사다리 적용, CLAUDE.md
  실험 설계 절 참고) (2026-08-19 API 경로 폐기 → 2026-08-19 PRO 6000 단독
  완결로 재조정, CLAUDE.md 참고).
  기록 항목(API 시절의 "모델 버전 문자열" 한 줄을 대체): HF 체크포인트
  리비전(commit hash) + vLLM 버전 + dtype + **생성에 사용한 GPU 기종** + 위
  sampling 파라미터 전부 + 타임스탬프 + 전체 프롬프트/응답 (CLAUDE.md 규칙 4).

## 5. 응답 파싱 규약

- 첫 번째 완결 코드 블록만 추출. 코드 블록이 없으면 "format_failure"로 분류.
- 코드 블록 외 텍스트는 저장하되 사용하지 않음.
- 파싱 실패도 데이터다 — 수동 구제 금지.

## 6. 결정 사항 (PI 승인, 2026-08-19)

- **PTX 하니스 API 노출 수준: 초안(§2) 유지.** `ptx_load`/`ptx_launch` 시그니처만
  노출. 근거: 더 노출하면 PTX가 유리해지고, 덜 노출하면 로딩 보일러플레이트
  실패가 PTX 표현력 측정과 무관하게 결과를 오염시킴 — 현 수준이 그 중간점.
- **accumulation 정밀도: 자유(§1 원안대로).** fp32 강제하지 않음. 근거:
  정밀도 전략 선택도 언어 표현력의 일부로 취급.

## 7. 변경 이력

- **2026-08-19: CUDA 블록에 "-std= 플래그 금지" 문구 추가 (PI 승인).**
  근거: 본 실행(37과제×4언어×5샘플×2모델) 평가 중 Qwen3-Coder-30B-A3B-Instruct가
  생성한 CUDA 샘플의 27.6%(51/185)가 `load_inline`의 `extra_cuda_cflags`에
  `-std=c++14`를 직접 지정, 이 서버의 PyTorch/ATen(C++17 요구)과 충돌해
  커널 로직과 무관하게 컴파일 실패했다 (gpt-oss-120b는 0/185, 이 패턴 없음).
  CUDA 프로브(②)가 다룬 아키텍처 플래그 지배성 판정과 별개의 문제이며, 프로브가
  사용한 4과제 표본에서는 우연히 나타나지 않아 사전에 포착되지 못했다. 아키텍처
  플래그 사례와 동일한 원칙(하니스가 툴체인 종속 플래그를 일괄 공급하고 모델
  제공 값을 무시)을 적용하기로 PI 승인 (2026-08-19). 하니스 변경은
  `scripts/evaluate.py`의 CUDA 빌드 경로(모델이 지정한 `-std=` 플래그를
  `extra_cflags`/`extra_cuda_cflags`에서 제거 후 C++17 고정); 아키텍처 플래그는
  프로브 판정대로 변경하지 않음. 영향받은 51개 Qwen CUDA 샘플은 하니스 수정 후
  재평가(재생성 아님 — 하니스 레벨 수정이므로).

- **2026-08-20: 컴파일 실패 진단 텍스트 로깅 캡 2000/4000자 → 20000자 상향 (PI 승인).**
  근거: 본 실행 평가 중 `scripts/evaluate.py`가 컴파일 실패 시 기록하는
  `metadata.compilation_error`가 2000자(일부 경로는 4000자)에서 잘려, nvcc
  명령줄이 긴 CUDA 샘플은 실제 `error:` 진단 줄에 도달하기도 전에 텍스트가
  끊기는 경우가 있었다. 영향은 모델 간 비대칭이었다(gpt-oss-120b CUDA
  172/181건 vs. Qwen3-Coder-30B-A3B-Instruct CUDA 25/124건 — 704건의 컴파일
  실패 기록 중 197건이 이 캡에 걸림, PTX·TileLang은 0건). 그대로 두면 논문의
  오류 분류표가 모델별로 왜곡된 채 보고될 위험이 있어 승인. **판정(compiled/
  correctness)에는 영향 없음 — 순수 로깅 필드만 확장**하는 변경이므로 이미
  기록된 판정은 재평가하지 않는다. 캡에 걸려 잘린 197건의 원문 진단은 별도로
  격리 재컴파일(CUDA 프로브 때와 동일 절차 — 판정 불변, 진단 텍스트만 재수집)
  로 보강할 예정.
- **2026-08-20: 닫히지 않은 코드펜스(unterminated fence) 허용 완화 — 검토 후 기각 (PI 결정).**
  제안 배경: gpt-oss-120b CUDA 3건(1.6%, `finish_reason="stop"`이지만 닫는
  ` ``` `이 누락)이 §5의 "완결된 fenced code block 없으면 format_failure"
  규칙에 걸려 파싱 실패로 분류됨. `extract_first_code_block()`을 완화해
  닫는 펜스가 없어도 EOF까지를 취하면 이 3건을 회수할 수 있음을 확인했음.
  **기각 사유**: §5는 사전 등록된 파싱 규약이고("코드 블록이 없으면
  format_failure, 수동 구제 금지"), 출력 형식 준수 자체가 이 실험이 측정하는
  모델 능력의 일부다. 1.6%(3/185)를 구하자고 실행 중간에 파서를 바꾸면 결과에
  실질적 영향 없이 프로토콜 변경 이력만 늘어난다. 하니스는 변경하지 않음 —
  이 3건은 format_failure로 유지.
- **2026-08-20: §3.3(수리 1턴, 컴파일 실패에만 1회) → §3.4(멀티턴 완주
  프로토콜, k=10) 전면 대체 (PI 확정).** 근거: pass@5 단일 시도로는 회복
  궤적을 볼 수 없음 — 정확성 실패에도 집계치 피드백(참조 텐서 값은 제외)을
  주는 수리 국면 + correct 도달 후 종료 대신 latency 피드백으로 계속
  개선시키는 최적화 국면의 2단계 체인으로 확장. 4언어·전 조건(0-shot/
  docinject) 동시 적용, 히스토리 무상태(매 턴 원 과제+직전 코드+직전
  피드백만) 원칙 유지. 상세는 §3.4.
  **결함 5개 과제**(`23_Softmax`, `4_Matrix_vector_multiplication_`,
  `6_Matmul_with_large_K_dimension_`, `90_cumprod`, `95_CrossEntropyLoss`)
  — `paper/RESULTS_REPORT_20260820.md` §7-2 허용오차 감사(CPU 전용, 37과제
  전수)에서 fp16 atol=rtol=1e-2가 해당 과제 출력 크기와 근본적으로 안
  맞음을 확인(출력이 허용오차보다 훨씬 작거나 — `23_Softmax` 4,090배,
  `90_cumprod` 언더플로 — 참조 자체가 fp16 오버플로로 `inf` —
  `4_Matrix_vector_multiplication_`, `6_Matmul_with_large_K_dimension_`,
  `95_CrossEntropyLoss`). §7-2에서 이미 실제 correct 표본 42건(228건 중
  18.4%)에 이 취약점이 걸려 있었음을 확인 — 본 실행/docinject 결과보고서의
  speedup 집계에서도 해당 과제를 과제 단위로 대칭 제외함(재실행 아님, 집계
  방식만 변경). **최초 조치(당일 오전)는 correct 도달 시 최적화 국면만
  제외**였으나, **같은 날 오후 §3.4 "멀티턴 완전 제외 과제 5개"로 확장**
  — 참조-오버플로 3과제는 수리 국면 피드백조차 무의미하고, 과다-관용
  2과제는 correct 판정 자체를 못 믿어 체인 진행의 의미가 없다는 판단.
  아래 별도 항목 참고.
- **2026-08-20 (오후): 논문 주 분석 기준을 37과제 → 32과제로 전환 (PI
  확정).** 위 결함 5과제를 제외한 32과제가 표1(2×2 메인 결과)·표2
  (docinject 전후)·speedup·향후 correct@turn을 포함한 **모든 집계의 주
  기준**이 된다. 37과제 원판정은 부록으로 병기 유지(재평가 아님 — 집계
  시 과제 필터만 다르게 적용, `scripts/analyze.py`). docinject의 20과제
  층화 부분집합 중 3개(`23_Softmax`, `6_Matmul_with_large_K_dimension_`,
  `95_CrossEntropyLoss`)가 결함 5과제와 겹쳐 **docinject의 감사 통과
  기준은 17과제** — `tasks/SELECTION.md` §4.3의 20과제 층화 설계 자체는
  바꾸지 않는다(과제를 추가/축소하지 않고, 집계 시 3개만 걸러냄).
- **2026-08-20 (오후): 멀티턴 완전 제외 과제 5개 — 최적화 국면 제외에서
  체인 전체(수리 국면 포함) 제외로 확장 (PI 확정).** 상세 사유·체인 수
  재계산(본 실행 32×4×5×2=1,280, docinject 17×4×5×2=680, 합계 1,960)은
  §3.4 "멀티턴 완전 제외 과제 5개" 절 참고. docinject는 32과제로 확장하지
  않고 기존 20과제(결함 3개 제외 17과제) 설계를 유지한다.
- **2026-08-20 (오후): `_process_input_tensor`의 정수 라벨 fp16 캐스팅
  버그 — 하니스 미수정 확정 (PI 결정).** `95_CrossEntropyLoss`가 멀티턴
  전체에서 빠지면서 이 버그의 유일한 발현 경로(모델이 이 과제 커널을
  컴파일에 성공시키는 순간)가 사라짐 — 하니스 수정 없이 known issue로만
  기록. 상세: `paper/RESULTS_REPORT_20260820.md` §7-2 "추가 발견".

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
- Target GPU: NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0).
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
```

### Triton
```
Write a Triton kernel (triton.jit) with a Python wrapper. Provide the full
Python file defining ModelNew that calls your Triton kernel.
```

### PTX
```
Write the complete kernel in raw PTX ISA (.target sm_120). Provide:
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

### 3.3 수리 1턴 (컴파일 실패 시에만, 조건 불문)
```
Your previous solution failed to compile with the following error:

{COMPILER_ERROR_VERBATIM}

Provide the corrected complete solution in a single code block.
```
- 에러 원문만. 요약·해석·힌트 추가 금지 (에러 메시지의 "LLM 가독성" 측정 목적).
- 정확성 실패(컴파일은 됐으나 출력 불일치)에는 수리 기회 없음 — 오답 확정.

## 4. 생성 파라미터 (전 조건 고정, 2026-08-19 확정)

- temperature: **0.8** (확정), top_p: 기본값, 샘플 수: 5 (독립 호출)
- **seed: 매 호출 명시적으로 고정해 지정하고 로그에 기록** (재현성 확보 목적).
  같은 과제·언어·조건의 샘플 `i`는 `base_seed + i`. vLLM의 OpenAI 호환
  엔드포인트는 `seed` 파라미터를 항상 받으므로 예외 없이 채운다.
- max output tokens: 8192 (PTX가 장문이 되므로 여유 확보; 잘림 발생 시 기록하고
  해당 샘플은 "truncated"로 분류 — 실패의 한 종류로 집계)
- 모델: **오픈웨이트 2개, 로컬 vLLM 서빙** — Qwen3-Coder-Next-80B-A3B(공식 FP8
  체크포인트) + gpt-oss-120b (2026-08-19 API 경로 폐기, CLAUDE.md 참고).
  기록 항목(API 시절의 "모델 버전 문자열" 한 줄을 대체): HF 체크포인트
  리비전(commit hash) + vLLM 버전 + dtype + 위 sampling 파라미터 전부 +
  타임스탬프 + 전체 프롬프트/응답 (CLAUDE.md 규칙 4).

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

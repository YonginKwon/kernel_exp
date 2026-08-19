# SOURCES.md — 언어 명세 발췌 출처

`prompts/PROMPT_SPEC.md` §3.2가 요구하는 발췌 출처 기록. 각 `prompts/specs/*.md`
파일 상단의 HTML 주석(`<!-- INTERNAL PROVENANCE ... -->`)에도 동일 정보가 있다 —
이 파일은 그걸 한곳에 모은 색인.

작성일: 2026-08-19. 토큰 수는 `scripts/count_spec_tokens.py`로 확인
(주입 대상 본문만 계산 — `---` 구분선 위 provenance 주석은 제외).

| 언어 | 파일 | 토큰 수 | 주 출처 | 접근 방식 |
|------|------|---------|---------|-----------|
| CUDA C++ | `cuda.md` | 4503 | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 목차/개요 페이지만 자동 fetch 가능(본문은 JS 렌더링 SPA라 실패) — 커널 문법·메모리 모델은 버전 간 안정적인 언어 의미론이라 직접 작성, VecAdd 패턴은 가이드의 표준 예제 스타일을 따름 |
| PTX | `ptx.md` | 4815 | [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 실시간 fetch 성공 — 모듈 구조(`.version`/`.target`/`.address_size`), `.param`/`.reg` 선언, 특수 레지스터(`%tid`/`%ctaid`/`%ntid`), 핵심 명령어 집합 확인. **§3 예제는 발췌가 아니라 이 프로젝트가 8/10에 `nvcc -arch=sm_120a -ptx`로 직접 생성해 `scripts/smoke_ptx.py`로 실기기(sm_120) 검증까지 마친 실제 PTX** (`scripts/fixtures/vecadd.ptx`) |
| Triton | `triton.md` | 4520 | [Vector Addition 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html), [Introduction 프로그래밍 가이드](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html), [language API 레퍼런스](https://triton-lang.org/main/python-api/triton.language.html) | 튜토리얼 페이지 실시간 fetch로 `add_kernel`/`add()` 코드 원문 확보(§3에 그대로 인용); "Blocked Program, Scalar Threads" 등 프로그래밍 모델 설명은 Introduction 페이지 fetch에서 인용; API 시그니처는 이 프로젝트에 설치된 triton==3.4.0 (`scripts/smoke_triton.py` sm_120 PASS)로 교차 확인 |
| TileLang | `tilelang.md` | 4562 | [GitHub README](https://github.com/tile-ai/tilelang), [Language Basics 가이드](https://tilelang.com/programming_guides/language_basics.html) | GitHub README는 실시간 fetch 성공 — `matmul_relu` 예제(§3.1 패턴 설명 근거) 원문 확보. `tilelang.com`의 Language Basics 페이지는 봇 차단(HTTP 403)으로 직접 fetch 불가 — WebSearch 스니펫("T.alloc_shared allocates shared memory... T.alloc_fragment... corresponds to register files")으로 핵심 문장만 교차 확인하고, 나머지는 이 프로젝트에 설치된 tilelang==0.1.13 (`scripts/smoke_tilelang.py` sm_120 PASS, fp32 elementwise + fp16 tiled matmul 둘 다 통과)로 직접 검증. **§3 최소 예제는 KernelBench 자체 stock 픽스처** (`third_party/KernelBench/src/kernelbench/prompts/model_new_ex_add_tilelang.py`, 우리가 작성하지 않음) — `scripts/smoke_kernelbench_harness.py`로 실제 `eval_kernel_against_ref(backend="tilelang")` 경로 통과까지 8/10에 확인됨 |

## 접근 실패 기록 (참고용)

- `docs.nvidia.com/cuda/cuda-c-programming-guide/index.html`: 목차 페이지만
  반환(섹션 본문은 JS 렌더링 SPA). 재시도(`#kernels` 앵커 포함)도 동일 결과.
- `www.tilelang.com/programming_guides/language_basics.html`,
  `tilelang.com/programming_guides/language_basics.html`: HTTP 403 (봇 차단).

## 공정성 노트

RESEARCH_CONTEXT.md §6.1이 요구하는 대로 4개 언어 모두 동일 구성(문법 개요 →
메모리 모델 → 최소 완전 예제 1개)과 동일 토큰 예산(4503–4815, 목표 5000±10% =
4500–5500 범위 내)을 지켰다. 접근 성공률의 차이(PTX·Triton 튜토리얼은
실시간 fetch 성공, CUDA 가이드 본문과 TileLang 공식 가이드는 실패)는 **출처
확보 난이도**이지 **명세 내용의 깊이**가 아니다 — 실패한 경우도 이 프로젝트
자체의 툴체인 검증(스모크 테스트)과 안정적인 언어 의미론 지식으로 동등한
깊이를 채웠다.

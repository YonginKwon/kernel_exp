# 과제 선정 — KernelBench Level 1 부분집합

CLAUDE.md 8/10 마일스톤 산출물. 선정 목록은 `level1_subset.json` (기계 판독용).
이 문서는 선정 기준·제외 근거(RESEARCH_CONTEXT.md §6.2, 부록 재료)와 미해결 결정
사항을 기록한다.

## 1. 선정 결과

Level 1 전체 100개 중 **37개** 선정 (목표 30–40개 충족).

| 계열 | 개수 | 비고 |
|------|------|------|
| Matmul | 7 | 기본형 + 구조적 변형 1개(전치) 대표만 |
| Elementwise/활성화 | 6 | Softmax 포함 (reduction 다리 역할) |
| 정규화 | 5 | BatchNorm/GroupNorm/RMSNorm/LayerNorm/L2Norm |
| Pooling | 3 | |
| Reduction | 4 | Sum/Mean/Max/Argmax |
| Convolution | 6 | **아래 §2 참고 — 원본은 34개, 대폭 축소** |
| 누적/스캔 | 2 | cumsum, cumprod |
| Loss | 3 | |
| Attention | 1 | ScaledDotProductAttention |

문서 injection ablation용 20개 부분집합은 `level1_subset.json`의
`doc_ablation_subset_of_20`에 초안으로 넣어뒀다 (**PI 승인 전 상태로 표시**).

## 2. 선정 기준

**포함 기준**: 4개 언어(CUDA C++/PTX/Triton/TileLang) 모두로 단일 커스텀 커널
표현이 원리적으로 가능한 연산. Level 1의 100개는 전부 이 기준을 만족한다
(라이브러리 호출 없이 스레드/타일 단위로 재구현 가능) — 그래서 포함 기준은
사실상 필터링 역할을 하지 못했고, 실질적 선별은 아래 배제 기준으로 이뤄졌다.

**배제 기준** (연산이 "표현 불가능"해서가 아니라 표본 설계상 제외):

1. **근접 중복 제거 (matmul 구조 변형)**: 12–18번은 대각행렬/대칭/상삼각/하삼각/
   양변전치 matmul로 전부 "matmul + 구조적 성질" 패턴이다. 언어 간 비교에
   새 신호를 주지 않는 반복이라 판단해 전치(B) 하나만 대표로 남기고 나머지
   5개(12,13,14,15,16,18)는 뺐다.
2. **conv 조합폭발 억제**: Level 1의 conv류는 34개(50, 54–87)로 전체의 34%를
   차지한다 — 1D/2D/3D × 표준/전치/depthwise × 대칭/비대칭 입력·커널 ×
   stride/padding/dilation/grouped 조합이다. 이걸 비례 표집하면 "conv 파라미터
   조합"이 과제 다양성의 대부분을 차지하게 되어 4언어 비교의 통계적 검정력이
   conv 하나에 쏠린다. 대신 **패턴 공간**을 6개로 샘플링:
   표준 2D, 표준 1D, 표준 3D, depthwise 2D, 전치 2D, (dilated+strided 결합) 1D.
   나머지 28개 conv 변형은 제외.
3. **극단적 파라미터 조합 배제**: 72–81번대(grouped+strided+padded+dilated
   동시 결합 3D 전치 conv 등)는 §2의 6개 대표 안에 포함하지 않음 — 위 축소와
   같은 이유.
4. **중복 activation 축소**: LeakyReLU/SELU/HardSigmoid/HardTanh/ELU/Softplus/
   Softsign/LogSoftmax/MinGPTNewGelu(20,27–32,24,88)는 ReLU/Sigmoid/Tanh/Swish/
   GELU/Softmax 6개로 대표. 전부 "elementwise (+ 선택적 reduction)" 패턴이라
   나머지는 언어 비교에 한계 정보를 거의 더하지 않는다고 판단.
5. **정규화 축소**: FrobeniusNorm/L1Norm(37,38)은 L2Norm과 동일 패턴(전역 reduction
   후 나눗셈)이라 L2Norm 하나만 남김.
6. **cumulative 변형 축소**: cumsum_reverse/cumsum_exclusive/masked_cumsum(91–93)은
   cumsum과 동일 스캔 패턴이라 cumsum·cumprod 2개로 대표.
7. **Loss 축소**: HuberLoss/KLDivLoss/TripletMarginLoss(96,98,99)는 MSELoss/
   CrossEntropyLoss/HingeLoss 3개로 대표 — reduction+elementwise 패턴이 겹침.

이 배제는 **모든 조건(4언어 × 2모델)에 동일 적용**되므로 특정 언어에 유리한
편향은 아니다. 다만 "표본 설계자(나)의 판단"이라는 선택 편향 가능성은 여전히
있고, 이게 RESEARCH_CONTEXT.md §6.2가 우려하는 지점이다 — 그래서 이 문서를
남긴다. **PI가 배제 목록을 검토해 이견이 있으면 조정 가능.**

## 3. 아직 검증하지 않은 것 (다음 단계)

이 선정은 "연산이 4언어로 원리적으로 표현 가능한가"의 **정적 판단**이다.
아직 하지 않은 것:
- 37개 각각을 실제로 4개 하니스에 태워 컴파일+실행되는지 확인 (하니스
  자체의 스모크 테스트는 8/10에 통과했지만, 그건 손으로 만든 벡터덧셈/matmul
  fixture였지 이 37개 task 자체는 아님 — CLAUDE.md 규칙 2 때문에 이 37개의
  정답 커널을 내가 작성해 검증할 수는 없다. 실제 검증은 `scripts/generate.py`가
  가동된 뒤 pass@5 결과 자체로 이뤄진다).
- KernelBench 표준 하니스가 이 37개의 `get_inputs()`를 그대로 4개 backend
  경로(cuda/triton/tilelang/PTX-custom-wrapper)에 태울 때 dtype 캐스팅이
  기대대로 동작하는지 (§4의 정밀도 이슈와 직결).

## 4. 결정 사항

### 4.1 정밀도: 4개 언어 전부 fp16 통일 (PI 승인, 2026-08-19)

**결정: 선택지 (a) 채택 — CUDA C++·PTX·Triton·TileLang 전부 fp16으로 통일한다.**

**근거**:
1. **교란(confound) 제거.** 2×2 설계는 "언어만 바꾸고 나머지는 통제"가 전제다.
   TileLang 셀만 하니스 제약으로 fp16이 강제되는 상태에서 나머지 3언어를 fp32로
   두면, 정확률 차이가 표현/자원 수준 때문인지 정밀도 때문인지 분리할 수 없다.
   전부 fp16으로 맞추면 이 교란이 원천 제거된다.
2. **배포 관행과의 정합.** 실제 GPU 커널 배포(추론 서빙 등)에서 fp16/bf16은
   예외가 아니라 표준에 가깝다 — "저정밀도에서의 언어 비교"로 프레이밍을
   바꾸는 것이 인위적 제약이 아니라 오히려 현실적인 조건 설정이다.
3. RESEARCH_CONTEXT.md §5의 "값을 정하면 전 조건 고정" 원칙과 정합하고, 언어별
   분기 구현이 없어 공정성 위협(§6.3 컴파일러 혼입)도 추가로 만들지 않는다.

**승인 시 부여된 4개 조건과 이행 방법**:

| # | 조건 | 이행 |
|---|------|------|
| ① | 베이스라인 eager도 fp16으로 측정 | `evaluate.py`가 참조(PyTorch eager) 실행 시에도 입력·모델을 fp16으로 캐스팅한 뒤 타이밍 — fp32 기준값과 섞지 않는다. CLAUDE.md 타이밍 프로토콜(warmup 25/측정 100/중앙값)은 그대로, dtype만 fp16. |
| ② | 하니스 fp16 판정 기준(atol/rtol) 불변경, 실값 기록 | KernelBench `eval.py:get_tolerance_for_precision`의 fp16 값 **atol=rtol=1e-2**를 그대로 사용 (하드코딩된 값, 수정하지 않음). 아래 표로 기록. |
| ③ | dtype 명시는 공통 템플릿에만, 언어 블록에 half 힌트 금지 | `prompts/PROMPT_SPEC.md` §1 공통 템플릿에 "All tensor inputs and outputs are float16 (fp16)"로 이미 위치, §2의 4개 언어 블록에는 정밀도·half 처리 언급 없음 — 조건 충족 확인함 (변경 불필요). |
| ④ | PI 결정과 근거를 SELECTION.md에 기록 | 이 절. |

**적용되는 정확성 판정 기준값** (조건 ②, 변경 금지 — 논문 Setup에 이 값 그대로 명시):

| precision | atol | rtol | 출처 |
|-----------|------|------|------|
| fp16 (torch.float16) | 1e-2 | 1e-2 | `third_party/KernelBench/src/kernelbench/eval.py::get_tolerance_for_precision` |

**TileLang 정밀도 제약의 출처** (참고, 결정에 영향 없음 — 이미 fp16으로 통일하므로
무관해졌으나 기록 보존): `eval.py`의 `eval_kernel_against_ref`에 아래 assert가
있고, 이는 **KernelBench 하니스의 정책이지 TileLang 자체의 한계가 아니다** —
`scripts/smoke_tilelang.py`에서 KernelBench 하니스를 거치지 않고 TileLang을 직접
호출했을 때는 fp32 elementwise add가 정상 컴파일·실행됨(max_diff=0.000e+00)을
2026-08-19에 직접 확인했다.

### 4.2 doc_ablation_subset_of_20의 상태 (2026-08-19 시점, 이후 §4.3에서 승인됨)

아직 초안, PI 미승인. §1 표 참고 — 이번 결정 범위(§4.1, PROMPT_SPEC.md §6)에
포함되지 않았다.

### 4.3 doc_ablation_subset_of_20 승인 (PI, 2026-08-20)

**조건부 사전 승인**: 본 실행(37과제×4언어×5샘플×2모델) 완료·전수검증
(`scripts/verify_eval_completeness.py`, PASS) 후, PI가 "Triton 평가에서 정답이
확인되고 gpt-oss CUDA 판정이 종결되면 문서 ablation 생성→평가를 추가 승인 없이
진행하라"고 조건부 승인 (2026-08-20). 같은 날 두 조건 모두 충족:
- Triton 정답 확인: 342/370 컴파일, **83/370 정답**(gpt-oss 59, Qwen 24) — 4개
  언어 중 유일하게 유의미한 정답 신호.
- gpt-oss CUDA 판정 종결: 184/185 컴파일 실패 분류 완료(cpp/cuda 함수명 불일치
  44.0%, 존재하지 않는 `AT_DISPATCH_HALF_TYPES` 매크로발 연쇄 미정의 25.0% 등,
  하니스 원인 0건 확정) + 컴파일 실패 로그 2000/4000자 캡을 20000자로 상향
  (PI 승인, 판정 불변 — PROMPT_SPEC.md §7).

**서브셋 승인 전 재확인 (이 세션에서 수행)**: `doc_ablation_subset_of_20`의 20개
과제가 §1의 37과제 계열 분포를 비례적으로 반영하는지 검증 — 전 9개 계열에서
37과제 기준 비율과 20과제 기준 비율의 차이가 5%p 이내(예: matmul 18.9%→20.0%,
convolution 16.2%→20.0%, pooling 8.1%→5.0%), 20개 과제명 전부 `families`의
기존 과제와 일치(불일치 0건). 층화 표집으로서 유효하다고 판단해 **승인**.
`tasks/level1_subset.json`의 `doc_ablation_subset_of_20.status`를 "APPROVED"로
갱신 — `scripts/generate.py`의 `--condition docinject` 거부 게이트가 이 필드를
확인하므로, 이 갱신 자체가 실행을 여는 조건이다.

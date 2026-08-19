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

## 4. 미해결 결정 사항 — PI 확인 필요

### 4.1 TileLang 정밀도 제약

`third_party/KernelBench/src/kernelbench/eval.py`의 `eval_kernel_against_ref`:

```python
if backend_lower == "tilelang":
    assert precision == torch.float16 or precision == torch.bfloat16, \
        "TileLang only supports fp16 or bfloat16"
```

KernelBench 표준 하니스에 있는 하드코딩된 제약이다 (우리가 만든 게 아니고,
"자체 검증기를 만들지 말 것" 규칙상 우회하지 않았다). 반면 CUDA C++는
fp32 기본, Triton/PTX는 임의 정밀도가 가능하다.

**이게 왜 문제인가**: 2×2 설계가 "언어만 바꾸고 나머지는 통제"를 전제하는데,
TileLang 셀만 강제로 fp16/bf16이면 정밀도가 언어와 교란(confound)된다.
정확률 차이가 "표현/자원 수준" 때문인지 "정밀도" 때문인지 분리가 안 된다.

**선택지** (하나를 정해서 CLAUDE.md에 박아야 함 — 실험 설계 변경이라 PI 승인 필요):
- (a) **4개 언어 전부 fp16으로 통일.** 가장 깨끗한 통제. 단, CUDA/Triton은
  fp32도 잘 하므로 "저정밀도에서의 언어 비교"로 프레이밍이 바뀜 — 논문
  Setup에 명시 필요.
  - **(a) 권장.** RESEARCH_CONTEXT.md §5가 이미 온도·seed 등 "값을 정하면
  전 조건 고정"을 원칙으로 삼고 있어 이 원칙과 가장 잘 맞고, 별도 구현
  분기가 없어 공정성 위협(§6.3 컴파일러 혼입)도 덜 만든다.
- (b) 정밀도를 별도 통제 축으로 명시하고 TileLang만 fp16, 나머지 3언어는
  fp32로 실행 — "정밀도 차이"를 결과 해석 시 한계로 명시. 구현은 더 간단하지만
  표 1(2×2 메인 결과)의 해석에 각주가 필요해짐.
- (c) TileLang을 이 실험에서 제외 — RESEARCH_CONTEXT.md 자체가 배제하는
  선택지(3언어 후퇴는 PTX 지연시에만, PI 결정 사항)라 채택 안 함.

이 문서에서는 결정하지 않았다 — **다음 세션에서 PI 확인 후 CLAUDE.md에 기록하고
`scripts/generate.py`에 반영해야 진행 가능.**

### 4.2 doc_ablation_subset_of_20의 상태

초안일 뿐 미승인. §1 표 참고.

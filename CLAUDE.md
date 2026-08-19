

Claude · MD
# kernel-lang-2x2 — LLM 커널 표현 비교 실험
 
ML for Systems @ NeurIPS 2026 워크숍 논문 실험.
연구 질문: LLM 커널 생성에서 표현(언어)의 추상화 수준과 자원 수준(HRPL/LRPL)의 효과를 분리한다.
마감: 2026-08-29 (AoE). 논문 4쪽, NeurIPS 포맷.
 
## 실험 설계 (변경 금지 — 변경은 PI 승인 후에만)
 
2×2 통제 비교, 단일 GPU(RTX A6000):
 
|                | 고자원 (HRPL) | 저자원 (LRPL) |
|----------------|---------------|---------------|
| 낮은 추상화    | CUDA C++      | PTX           |
| 높은 추상화    | Triton        | TileLang      |
 
- 과제: KernelBench Level 1에서 4개 언어 모두 표현 가능한 30–40개
- 모델: OpenAI GPT 최신 1개 + Anthropic Claude 최신 1개 (버전 문자열 고정, 로그에 기록)
- 프로토콜: 언어당 pass@5 one-shot + 컴파일 에러 메시지만 주는 수리 1턴
- Ablation: 문서 주입 (언어 명세 ~5k 토큰 in-context), 과제 20개 부분집합
- 지표: 컴파일률 / 정확률 / fast_1 / speedup geomean / 솔루션당 토큰 수 / 오류 분류 / 수리 후 회복률
## ⚠️ 방법론적 경계 — 절대 규칙
 
1. **생성된 커널 코드를 절대 직접 수정하지 말 것.**
   results/ 아래의 모든 LLM 출력물은 읽기 전용 데이터다.
   커널이 틀렸어도 고치지 않는다. 틀린 것 자체가 측정 결과다.
2. **커널 생성은 오직 scripts/generate.py의 API 호출로만 수행한다.**
   Claude Code 세션에서 직접 커널을 작성·생성·개선하지 않는다.
   (이 규칙을 어기면 실험이 통제 비교가 아니게 되어 논문이 무효가 된다.)
3. **프롬프트 템플릿 변경은 전 언어에 동시 적용**하고 git commit으로 남긴다.
   특정 언어에만 유리한 힌트 추가 금지.
4. 모든 실험 실행은 다음을 로그로 남긴다:
   모델 버전 문자열, temperature, seed, 프롬프트 전문(해시 + 원문), 타임스탬프,
   torch/CUDA/드라이버 버전, 응답 원문 전체.
## 환경
 
- GPU: ~~RTX A6000 / sm_86~~ — **실측 결과 문서와 불일치. 2026-08-19 첫 세션에서 정정.**
- Python: conda 미설치 → venv `.venv` 사용 (conda env `kernel2x2` 대체).

### 실측 환경 (2026-08-19 기록, 논문 Setup에 이 값 사용)

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB, 600W) |
| Compute capability | **12.0 → `sm_120`** (문서 기재 sm_86 아님) |
| PTX `.target` | `sm_120` — PTX 프롬프트에 이 값 반영 |
| NVIDIA 드라이버 | 595.84 |
| CUDA (드라이버 지원) | 13.2 |
| 시스템 nvcc | 12.0.140 — **sm_120 미지원, 사용 불가** |
| 시스템 g++ | 13.3.0 — **nvcc 12.8은 13.x를 미지원(host_config.h 하드 에러), g++-12 사용** |
| 사용 nvcc | `~/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/nvcc` (12.8.93) |
| 사용 ptxas | 위 nvcc와 동일 배포판 (12.8.93) — `triton` pip 패키지 번들 ptxas도 동일 버전, 상호 대체 가능 |
| 사용 g++ | `/usr/bin/g++-12` (12.4.0) — apt로 이미 설치돼 있었음 |
| PTX ISA (nvcc 12.8.93 산출) | `.version 8.7`, `.target sm_120a` — **`.target`은 `sm_120`이 아니라 `sm_120a`** (Blackwell 패밀리 전용 명령 접미사 `a`; 프롬프트 명세에 반영할 것) |
| Python | 3.12.3 |
| torch | 2.8.0+cu128 (cuda 12.8, arch_list에 sm_120 포함) |
| triton | **3.4.0 — sm_120 스모크 PASS** (`scripts/smoke_triton.py`, 벡터 덧셈, exact match) |
| tilelang | **0.1.13 — sm_120 스모크 PASS** (`scripts/smoke_tilelang.py`, fp32 elementwise + fp16 tiled matmul 텐서코어 경로 모두 통과) |
| ninja | 1.13.0 (.venv에 pip 설치, CUDA C++ JIT 확장 빌드에 필요) |

### 툴체인 우회가 필요했던 이유 (재현 시 참고)

1. **시스템 nvcc(12.0)는 `sm_120a`를 모른다** → nvcc 12.8.93(위 경로)을 PATH 앞단에 둠.
2. **시스템 g++(13.3)는 nvcc 12.8이 거부한다** (`#error unsupported GNU version`) →
   `CXX=/usr/bin/g++-12`로 고정.
3. **TileLang 0.1.13의 nvcc 호출부(`jit/adapter/libgen.py`)가 `-I$CUDA_HOME/include`를
   붙이지 않는다** → PATH만 바꾸면 nvcc가 `/usr/include/cuda_fp8.h`(시스템 CUDA 12.0,
   Blackwell의 E8M0 FP8 변환 intrinsic 없음)를 집어 컴파일이 깨진다.
   `CPATH=$CUDA_HOME/include`로 우회(모든 컴파일 경로가 CPATH는 존중함).
4. `$CUDA_HOME`은 실제 설치본이 아니라 심볼릭 링크로 조립한 것:
   `third_party/cuda-sm120-toolchain/{bin→nvcc 12.8.93, include/lib64→triton 번들
   cudart 12.8.57}` — 둘 다 pip 배포 wheel 부산물이라 완전한 CTK가 아님. `nvcc`
   배포판엔 헤더가, `cudart` 배포판엔 헤더+런타임 라이브러리가 나뉘어 있어서 합쳤음.

패키지 고정 목록: `requirements.txt` (`pip install -r requirements.txt`, venv `.venv`
안에서). torch/triton은 이미 설치돼 있던 버전(2.8.0+cu128 / 3.4.0)을 그대로 씀 —
KernelBench 상류 pin(torch 2.9.*, triton 3.5.*)보다 낮지만 8/10 스모크를 전부
통과했으므로 업그레이드는 선택 사항으로 미룸.

**결론: 모든 실험 실행 전에 반드시 `source scripts/env.sh`.** (`generate.py`/
`evaluate.py` 등 모든 진입 스크립트 최상단 주석에도 명시할 것.) 이 우회가 없으면
CUDA C++·TileLang 두 트랙은 컴파일 자체가 실패하므로, results/ 안 컴파일 실패가
"모델의 한계"가 아니라 "환경 설정 누락"으로 오염될 위험이 있다 — 공정성 위협 문서
(RESEARCH_CONTEXT.md §6) 준하는 통제 항목으로 취급.

> ⚠️ **하드웨어 변경의 함의**: Blackwell은 문서가 상정한 Ampere(A6000)와 성능 특성이
> 다르다. speedup 분포 해석과 논문 Setup의 하드웨어 기술을 모두 이 값 기준으로 쓸 것.
> 문헌의 A6000 수치와 비교하지 말 것 (원래 규칙: 베이스라인 매 실행 재측정).

### PTX go/no-go 판정: **GO** (2026-08-19)

`harness/ptx/ptx_harness.py` (ctypes 기반 cuModuleLoad 래퍼, 의존성 없음) +
`scripts/smoke_ptx.py`로 검증:
ptxas 어셈블(PTX→cubin) → `cuModuleLoadData` → `cuLaunchKernel` → PyTorch 텐서와
결과 대조, 전 구간 통과 (exact match). 검증에 쓴 PTX는 `nvcc -arch=sm_120a -ptx`로
1회 생성한 벡터 덧셈 텍스트(`scripts/fixtures/vecadd.ptx`) — **인프라(래퍼) 검증용
픽스처이며 LLM PTX 작성 능력의 증거가 아님.** 컴파일 에러 메시지 반환 경로
(`PTXCompileError.stderr`)도 확보되어 있어 수리 1턴 프로토콜에 바로 연결 가능.
3언어 후퇴 결정은 필요 없음 — 4언어 전부 진행.

### 4개 하니스 스모크 테스트: **통과** (2026-08-19)

- CUDA / Triton / TileLang 3개는 KernelBench 자체 하니스
  (`eval.py:eval_kernel_against_ref`, `backend="cuda"/"triton"/"tilelang"`)를
  그대로 재사용 — CLAUDE.md 디렉터리 구조가 예고한 대로 신규 코드 불필요.
  KernelBench 자체 예시 픽스처(`prompts/model_ex_add.py` +
  `model_new_ex_add{,_triton,_tilelang}.py`, 우리가 작성한 게 아님)로
  `scripts/smoke_kernelbench_harness.py`가 3개 백엔드 전부 compiled=True,
  correctness=True 확인.
- PTX는 상류에 backend 자체가 없어 `harness/ptx/ptx_harness.py` 신규 작성 —
  위 PTX go/no-go 절 참고.
- 과제 선정: `tasks/level1_subset.json` + `tasks/SELECTION.md` — Level 1 100개 중
  37개, 선정 기준·제외 근거 문서화.
- **정밀도 결정 (PI 승인, 2026-08-19): 4개 언어 전부 fp16 통일.** 근거·4개 이행
  조건·판정 기준값(atol=rtol=1e-2)은 `tasks/SELECTION.md` §4.1 참고. 프롬프트
  반영은 `prompts/PROMPT_SPEC.md` §1 공통 템플릿.
- **프롬프트: `prompts/PROMPT_SPEC.md`가 유일한 진실.** 변경은 git commit +
  전 언어 동시 적용 (파일 자체 원칙, 1행). §6 두 항목(PTX 하니스 API 노출 수준,
  accumulation 정밀도 자유)도 PI 승인 완료 — 결정 사항으로 §6에 기록됨.
- 타이밍 프로토콜: warmup 25회, 측정 100회, 중앙값. torch.cuda.synchronize 필수.
  베이스라인은 PyTorch eager, 같은 GPU에서 매 실행 재측정 (문헌 수치 사용 금지).
## 디렉터리 구조
 
```
kernel-lang-2x2/
├── CLAUDE.md              # 이 파일
├── tasks/                 # 과제 정의 (KernelBench L1 부분집합 + 선정/제외 기준표)
├── prompts/               # 언어별 프롬프트 템플릿 + 문서 주입용 명세
├── scripts/
│   ├── generate.py        # API 호출 생성기 (유일한 생성 경로)
│   ├── evaluate.py        # 컴파일 + 정확성 + 타이밍
│   └── analyze.py         # 집계, 표/그림 생성
├── harness/
│   ├── cuda/              # KernelBench 하니스 재사용
│   ├── triton/            # KernelBench 하니스 재사용
│   ├── tilelang/          # 신규 (Triton 하니스 참조하여 작성)
│   └── ptx/               # 신규 (cuModuleLoad 래퍼) — 8/10 go/no-go
├── results/               # 생성물 + 평가 결과 (읽기 전용 취급, git-lfs 또는 .gitignore)
└── paper/                 # 이후 LaTeX
```
 
## 마일스톤
 
- 8/10: 과제 선정 완료, 4개 하니스 스모크 테스트 통과, **PTX go/no-go 판정**
- 8/17: 본 실험(~1,600 생성) + 문서 ablation(~800) 실행 완료
- 8/24: 분석 완료, 초고
- 8/29: 제출
## 작업 방식
 
- 모든 변경은 git commit. 커밋 메시지는 영어, 한 줄 요약.
- 장시간 실행은 tmux 세션 `exp` 안에서. 실행 전 예상 API 비용을 추산해 보고할 것.
- API 키는 환경변수(OPENAI_API_KEY, ANTHROPIC_API_KEY)로만. 코드·로그에 절대 기록 금지.
- 막히면 임의로 우회하지 말고 선택지를 정리해 보고할 것. 특히 PTX 래퍼가
  이틀 이상 지연되면 즉시 보고 (3언어 후퇴 결정은 PI가 한다).
 






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
- 모델: **오픈웨이트 2개, 로컬 vLLM 서빙** (2026-08-19 API 경로 폐기, PI 지시).
  gpt-oss-120b + Qwen3-Coder-Next-80B-A3B(공식 FP8 체크포인트, `Qwen/Qwen3-Coder-Next-FP8`).
  Qwen 쪽 강등 사다리(96GB 카드 1장 기준, 2026-08-19 계획 확정):
  ① 풀 컨텍스트로 시도 → ② 안 들어가면 max-model-len 축소 → ③ 그래도 안 되면
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`(표준 MoE, hybrid 아님, bf16 61GB)로 강등.
  강등이 실제로 일어나면 이 표를 실행 결과로 갱신하고 보고할 것 — 조용히 넘어가지 않는다.
  기록 항목: HF 체크포인트 리비전(commit hash) + vLLM 버전 + dtype + sampling
  파라미터 + **생성에 사용한 GPU 기종** — "버전 문자열"이 아니라 이 5가지 전부를
  로그에 남긴다.
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
   HF 체크포인트 리비전(commit hash) + vLLM 버전 + dtype + **생성에 사용한 GPU 기종**,
   temperature, seed, 프롬프트 전문(해시 + 원문), 타임스탬프,
   torch/CUDA/드라이버 버전(평가 서버 기준), 응답 원문 전체.
5. **기본 계획: 이 서버(PRO 6000, sm_120) 단독으로 생성+평가 완결.**
   (2026-08-19 재조정 — H100은 상시 가용이 아니라 기본 경로에서 제외.)
   `scripts/serve_local.sh`로 이 기계에서 모델 하나씩 vLLM 서빙 → 생성 →
   종료 → 다음 모델, 두 모델 다 끝난 뒤 평가. **한 모델의 생성은 반드시 한
   기계에서 완결한다** — 도중에 기계를 바꾸지 않는다 (HF 리비전·vLLM 버전·GPU
   기종 로그가 배치 단위로 일관돼야 하므로).
   `scripts/serve_h100.sh`(학과 H100×2, 별도 기계)는 **가속 옵션으로 격하** —
   H100이 비어 있을 때 모델 하나를 통째로 그쪽에 넘기는 용도로만 남겨둠, 기본
   경로 아님. API 키 개념 자체가 사라졌으므로 관련 차단 로직은 `--base-url`
   필수 인자로 대체됐다 (`scripts/generate.py` 참고).
6. **`scripts/evaluate.py`는 시작 시 GPU 배타성을 강제한다.** `nvidia-smi`로
   확인해 GPU에 타 프로세스(특히 vLLM 서버)가 남아 있으면 평가를 거부한다 —
   생성(vLLM)과 평가(타이밍 측정)가 같은 GPU를 동시에 쓰면 타이밍이 오염되기
   때문. **이 검사에는 우회 플래그를 두지 않는다** (PI 지시, 절대 규칙).
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

### 생성 아키텍처 — 2026-08-19 재조정: PRO 6000 단독 완결이 기본

API 경로를 폐기하고 로컬 vLLM 서빙으로 전환 (최초 결정). 처음엔 학과 H100×2를
생성 전용으로 쓰는 안이었으나, **H100이 상시 가용이 아니라는 게 드러나** 기본
계획을 이 서버(PRO 6000, sm_120, GPU 1장, 97,887 MiB) 단독 완결로 재조정했다.
H100은 비어 있을 때 모델 하나를 통째로 넘기는 가속 옵션으로 남긴다.

| 항목 | 값 |
|------|-----|
| 기본 경로 | `scripts/serve_local.sh <model>` — 이 서버(PRO 6000)에서 모델 1개씩 순차 서빙 |
| 가속 옵션 | `scripts/serve_h100.sh` — 학과 H100×2, 비어 있을 때만, 모델 1개를 통째로 |
| 모델 A | gpt-oss-120b (`openai/gpt-oss-120b`), MXFP4 기본 양자화, 포트 8000 |
| 모델 B | Qwen3-Coder-Next-80B-A3B (`Qwen/Qwen3-Coder-Next-FP8`), 포트 8001 — 96GB 카드 1장에 빠듯할 수 있음, 강등 사다리는 위 실험 설계 절 참고 |
| Qwen 실제 사용 체크포인트 | **실행 후 기록** — 강등 발생 시 여기 표시 (기본: Next-FP8, 강등 시: 30B-A3B-Instruct) |
| HF 체크포인트 리비전 | **실행 후 기록** (`serve_local.sh`가 `logs/vllm/<name>_manifest.json`에 자동 기록) |
| vLLM 버전 | **실행 후 기록** (manifest) |
| dtype | 모델 A: MXFP4(배포 기본값) / 모델 B: FP8(체크포인트 자체) 또는 강등 시 bf16 |
| GPU 기종 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (이 서버 실측값, 위 "실측 환경" 표) |

모든 서버는 OpenAI 호환 엔드포인트(`/v1/chat/completions`)로 뜬다.
`scripts/generate.py`는 `--base-url`(필수, 기본값 없음)로 이 엔드포인트를 받는다.
API 키는 없음 — vLLM은 인증하지 않으므로 더미 문자열을 OpenAI SDK에 채워 넣는다.

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
│   ├── generate.py        # 로컬 vLLM 호출 생성기 (유일한 생성 경로)
│   ├── serve_local.sh     # 기본 경로 — 이 서버(PRO 6000)에서 모델 1개씩 vLLM 서빙
│   ├── serve_h100.sh      # 가속 옵션 — 학과 H100×2, 비어 있을 때만 (사용자가 직접 실행)
│   ├── evaluate.py        # 컴파일 + 정확성 (타이밍은 아직 — 파일럿엔 불필요해 보류)
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
 
- 8/10: 과제 선정 완료, 4개 하니스 스모크 테스트 통과, **PTX go/no-go 판정** — 완료
- 8/17: 본 실험(~1,600 생성) + 문서 ablation(~800) 실행 완료 — **도구는 준비됨,
  실행은 아직.** `scripts/generate.py`(유일한 생성 경로, 규칙 2 준수) +
  `prompts/spec_loader.py`(PROMPT_SPEC.md를 파싱하는 유일한 진입점, 템플릿 중복
  없음) + `prompts/specs/{cuda,ptx,triton,tilelang}.md`(문서 주입용, 각
  4500–4815 토큰, 출처는 `prompts/specs/SOURCES.md`) 작성 완료.
  `--dry-run`으로 4언어 전부 프롬프트 생성·비용 추산 검증, `litellm mock_response`로
  로깅 파이프라인(모델 버전·temperature·seed·프롬프트 해시+원문·타임스탬프·
  torch/CUDA/드라이버 버전·응답 원문 — 규칙 4 전 항목)과 응답 파싱(`generated`/
  `format_failure`/`truncated` 세 상태 분기) 전부 확인함. **실제 실행 전 남은 것**:
  ① OPENAI_API_KEY/ANTHROPIC_API_KEY 미설정 — 아직 실제 API 호출 없음,
  ② 모델 버전 문자열 미확정 — "OpenAI GPT 최신 1개 + Anthropic Claude 최신 1개"를
  구체적 버전 문자열로 PI가 고정해야 `generate.py --provider-model`에 넣고 실행
  가능 (`scripts/generate.py`가 기본값을 두지 않고 필수 인자로 강제해둠 — 임의
  추측 방지), ③ 문서 ablation 20개 부분집합은 여전히 미승인(`tasks/SELECTION.md`
  §4.2) — `--condition docinject`는 코드 레벨에서 승인 전까지 자동 거부하도록
  만들어둠.
- 8/24: 분석 완료, 초고
- 8/29: 제출
## 작업 방식
 
- 모든 변경은 git commit. 커밋 메시지는 영어, 한 줄 요약.
- 장시간 실행은 tmux 세션 `exp` 안에서.
- **API 키 대신 로컬 엔드포인트.** `scripts/generate.py`는 `--base-url`을 필수
  인자로 요구하고 기본값을 두지 않는다 (오타로 엉뚱한 서버를 치는 사고 방지) —
  과거 "OPENAI_API_KEY/ANTHROPIC_API_KEY 미설정이면 거부" 로직은 "base_url
  미지정이면 거부"로 대체됐다. 비용 개념이 없어졌으므로 "실행 전 예상 API
  비용 추산" 규칙은 폐기 — 대신 실행 전 파일럿(2과제×4언어×2샘플×2모델)으로
  파싱/컴파일/정확성이 살아있는지 확인하고 보고한다.
- 막히면 임의로 우회하지 말고 선택지를 정리해 보고할 것. 특히 PTX 래퍼가
  이틀 이상 지연되면 즉시 보고 (3언어 후퇴 결정은 PI가 한다).
 




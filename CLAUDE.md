

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
| 사용 nvcc | `~/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/nvcc` (12.8.93) |
| 사용 ptxas | `~/.local/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas` (12.8.93) |
| Python | 3.12.3 |
| torch | 2.8.0+cu128 (cuda 12.8, arch_list에 sm_120 포함) |
| triton | 3.4.0 |
| tilelang | (설치 예정 — 확정 후 기록) |

> ⚠️ **하드웨어 변경의 함의**: Blackwell은 문서가 상정한 Ampere(A6000)와 성능 특성이
> 다르다. speedup 분포 해석과 논문 Setup의 하드웨어 기술을 모두 이 값 기준으로 쓸 것.
> 문헌의 A6000 수치와 비교하지 말 것 (원래 규칙: 베이스라인 매 실행 재측정).
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
 




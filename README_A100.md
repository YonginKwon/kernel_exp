# A100 서버 환경 오버레이 — 논문 재현성 기록

이 문서는 PI 요청(2026-08-21)에 따른 **"기준 커밋 + A100 환경 오버레이 커밋
목록"** 명세다. 목적: 이 서버(A100×2, Ampere sm_80)에서 실행된 Phase 2
멀티턴 결과가 본 실험(RTX PRO 6000, sm_120)과 **판정 로직·피드백 템플릿·
프롬프트 본문이 완전히 동일한 프로토콜** 위에서 나온 것임을 논문 부록/
재현성 섹션에서 인용할 수 있는 형태로 남긴다.

## 기준 커밋

`7fe6d43` (`origin/master`, PI가 2026-08-21 제공: "Push 완료 —
ee46e9b..7fe6d43 (P0-a/P0-b 전체: 타이밍, 허용오차 감사, 32과제 전환,
멀티턴 오케스트레이터)"). 이 서버는 이 커밋을 `git merge origin/master`로
그대로 받았다 — cherry-pick이나 부분 반영이 아니라 전체 히스토리 병합
(커밋 `1d46b67`).

## 프로토콜 무결성 확인 (PI 요청, 2026-08-21)

```
git diff 7fe6d43 HEAD -- scripts/evaluate.py scripts/generate.py scripts/multiturn.py
```
→ **0 lines** (세 파일 모두 `7fe6d43`와 바이트 단위로 동일). 판정 로직
(compiled/correctness 계산, timing 프로토콜, precompile 파이프라인),
`#3.4` 오케스트레이터의 체인 구조·종료 조건·피드백 조립 로직 전부 수정
없이 그대로 실행 중이다.

## 오버레이 커밋 (설정값 수준만, `7fe6d43` 위에 로컬 추가)

| 커밋 | 파일 | 변경 내용 | 판정/프로토콜 영향 |
|---|---|---|---|
| `1ebb2fe` | `harness/ptx/ptx_harness.py` | PTX arch-suffix "a"를 sm_{90,100,101,120}에만 붙이도록 수정 (Ampere sm_80엔 접미사 없음 — ptxas 하드에러 회피) | 없음 — sm_120에서 한 번도 발현 안 한 버그, A100 전용 아키텍처 문자열 처리 |
| `1ebb2fe` | `scripts/smoke_ptx.py` | `--fixture` 인자 추가 (기본값은 기존 sm_120a 픽스처 그대로) | 없음 — 스모크 테스트 전용, 판정 경로 아님 |
| `4b09f64` | `prompts/PROMPT_SPEC.md` | "Target GPU:"·PTX `.target` 두 줄을 `{TARGET_GPU_LINE}`/`{PTX_TARGET}` 플레이스홀더로 치환 | **본문 텍스트 값은 불변** — env var 미설정 시 원문과 바이트 동일 (검증됨) |
| `4b09f64` | `prompts/spec_loader.py` | 위 플레이스홀더를 `KERNEL2X2_TARGET_GPU_LINE`/`KERNEL2X2_PTX_TARGET` env var로 채움 (기본값 = sm_120 원문) | 없음 — 판정/피드백 템플릿 함수는 전혀 건드리지 않음(§3.4 5개 템플릿 로직 무변경) |
| `88acec7`/`9cf411f`/`4f33f9e` | `scripts/serve_local.sh` | vLLM 버전 pin, CUDA_HOME override, gptoss/qwen 메모리·컨텍스트 길이를 env var로 파라미터화 (기본값 = sm_120 원문 그대로) | 없음 — 생성 인프라(vLLM 서빙)만, 프롬프트·판정 로직 무관 |
| `c6dc6e6` | `scripts/measure_baseline.py` (신규) | Phase 2 게이트 전 준비 작업 ②용 신규 스크립트 (PyTorch eager fp16 베이스라인) | 없음 — 신규 파일, 기존 판정 경로 미수정 |

**공통 원칙**: 모든 오버레이는 env var 미설정 시 sm_120 서버와 바이트 단위
동일 동작(기본값 = 원문 하드코딩 값) — 검증 방법은 각 커밋 메시지에 기록.

## 신규 A100 전용 오케스트레이션 스크립트 (하니스 미수정, 2026-08-21)

`scripts/multiturn.py`(`#3.4` 오케스트레이터) 자체는 수정하지 않는다
(follower 모드 원칙, `~/kernel-lang-2x2/CLAUDE.md`). 이 서버 turn-1
데이터(`results/eval/eval_a100_full.json`, 3언어·0-shot만·docinject
없음)를 `multiturn.py`의 `cmd_init`이 알 수 없으므로, 대신 신규 진입점
스크립트를 추가했다 — 전부 `multiturn.py`의 기존 함수(`chain_id`, `K_MAX`)
와 `analyze.py`의 `clean_32_tasks()`/`FLAWED_TASKS`를 그대로 import해서
쓰고, 판정/프로토콜 로직은 재구현하지 않는다:

- `scripts/multiturn_init_a100.py` — 체인 상태 부트스트랩(`cmd_init`과
  구조적으로 동일 로직, 소스 파일만 A100 것으로). 이후
  `multiturn.py generate/evaluate/report`는 `--state` 경로만 받으므로
  완전히 무수정 그대로 실행.
- `scripts/multiturn_cycle_a100.sh` — 턴 사이클(gptoss 서빙→생성→중지→
  qwen 서빙→생성→중지→평가→리포트) 반복 실행용 셸 래퍼. 실험 인프라
  시퀀싱일 뿐 프로토콜 코드 아님(serve_local.sh와 동일 범주).
- `scripts/multiturn_report_a100.py` — 턴별 ever-correct + FF/FT/TF/TT
  전이 집계(PI 요청, 2026-08-21). `state["chains"][*]["history"]`(이미
  `multiturn.py`가 기록)를 읽기만 하는 순수 분석, `analyze.py`와 동일
  범주(판정 재계산 없음).

## 체인 구성 (이 서버, PI 확정 2026-08-20/21)

32과제(37 − 결함 5과제) × 3언어(cuda/ptx/triton, tilelang 없음) × 5샘플 ×
2모델 = **960 체인**. docinject 없음(A100 Phase 1이 0-shot 3언어만
수행했으므로). turn 1 = 이 서버 자체 Phase 1 데이터(1,110건 중 32과제
필터링분).

## k=10 상한 확인 (PI 요청, 2026-08-21)

`scripts/multiturn.py`(무수정, `7fe6d43`) `cmd_evaluate`:
```python
if turn >= state.get("k_max", K_MAX):      # K_MAX = 10
    c["terminated"], c["termination_reason"] = True, "k_max_reached"
```
매 `evaluate` 호출마다 전 체인에 적용되며, `cmd_generate`는
`not c["terminated"]`인 체인만 다음 턴 생성 대상으로 삼는다 — 즉 턴 10에
도달한 체인은 `terminated=True`로 표시되고 이후 `generate` 라운드에서
자동으로 제외된다. 이 서버에서도 sm_120과 동일 코드로 강제됨을 확인.

## 재현 방법

```
git log --oneline 7fe6d43..HEAD          # 오버레이 커밋 목록
git diff 7fe6d43 HEAD -- scripts/evaluate.py scripts/generate.py scripts/multiturn.py  # 0 lines 확인
```

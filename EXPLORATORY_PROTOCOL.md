# `deep-turn-probe` — 탐색적 장기 호라이즌 실험 프로토콜

- 상태: **탐색적(exploratory) 실험, 논문 본 실험(Phase 2, `results-a100`)과 완전 분리**
- 우선순위: **최하위** — 본 논문 관련 요청이 들어오면 즉시 양보/중단
- 지시자: 사용자, 2026-08-23
- 목적: k=100 장기 호라이즌에서 (1) PTX 최약 표현의 수리(repair) 실패가 호라이즌 문제인지 하드월(hard wall)인지, (2) Triton 최강 표현의 최적화(optimize) 피드백이 벤더 라이브러리 월(wall)에 얼마나 근접하는지 관찰

## 1. 본 실험과의 관계 — 완전 격리

이 실험은 논문 본 실험(Phase 2, `results-a100` 브랜치, 2026-08-22 턴10 종결,
최종 커밋 `bcd6d4fcb0d4db2dabd0735e245d074766e72e66`)과 **완전히 분리된 부산물
탐색**이다. 아래 규칙은 절대 규칙이다:

1. 본 실험이 이미 종결된 파일(`results/eval/multiturn_state_a100.json`,
   `results/eval/final_retime_a100.json`, `results/eval/fig_*.csv`)이나
   `results-a100` 브랜치를 **일체 변경하지 않는다.**
   유일한 예외: Arm B 시작점 결정을 위해 `multiturn_state_a100.json`을
   **읽기 전용**으로 조회하는 것(이미 완료, 아래 §3 참고) — 쓰기·수정 없음.
2. 이 실험의 산출물은 전부 `results/exploratory/` 아래에만 쓴다.
3. 이 실험의 브랜치는 신규 `results-a100-exploratory`뿐이다. `results-a100`에는
   푸시하지 않는다.
4. 이 실험의 수치는 논문 본문 표/그림에 섞이지 않는다(부록/별도 관찰 노트로만
   사용 가능, PI 승인 별도 필요).

## 2. 프로토콜 이탈 2건 (명시적, 사전 기록)

본 실험(`#3.4` 멀티턴 오케스트레이터, `scripts/multiturn.py`)과 동일한 판정
로직·프롬프트·피드백 템플릿을 그대로 재사용하되, 장기 호라이즌 관찰이라는
목적을 위해 다음 2개 파라미터만 이탈한다. **하니스 코드(`multiturn.py`/
`generate.py`/`evaluate.py`) 자체는 이 이탈을 위해 수정하지 않는다** — 기존
관례(비침투 재사용: 상태 파일 오버라이드, 호출 시점 monkeypatch)로 구현한다.

| # | 이탈 항목 | 본 실험 값 | 이 실험 값 | 사유 | 구현 방식 |
|---|---|---|---|---|---|
| ① | `k_max` | 10 | **100** | 장기 호라이즌에서 PTX 수리가 호라이즌 문제인지 하드월인지, Triton 최적화가 벤더월에 얼마나 근접하는지는 턴10 절단으로는 관찰 불가 | 상태 JSON에 `"k_max": 100` 직접 기록 (`multiturn.py`의 `cmd_evaluate`가 이미 `state.get("k_max", K_MAX)`로 상태 파일값을 우선하므로 코드 변경 불필요) |
| ② | `NO_IMPROVE_LIMIT`(3턴 무개선 조기 종료) | 3 | **사실상 비활성** | 이 실험의 관심사가 정확히 "수렴 이후에도 계속 관찰했을 때 무슨 일이 일어나는가"이므로, 3턴 무개선에서 조기 종료하면 관찰 대상 자체가 사라짐 | 호출 스크립트(`scripts/exploratory_multiturn.py`)에서 `multiturn.NO_IMPROVE_LIMIT`를 매우 큰 값(예: `10**9`)으로 monkeypatch 후 `multiturn.cmd_evaluate` 호출 — `multiturn.py` 파일 자체는 무수정 |

이 2건 외 **모든 규칙은 본 실험과 완전히 동일**:
- 생성된 커널 코드 직접 수정 금지 (CLAUDE.md 규칙 1).
- 프롬프트/피드백 템플릿 고정, 전 언어·양팔 동일 (`_feedback_for`/`build_prompt`
  무수정 재사용).
- 전 실행 로그 남김 (CLAUDE.md 규칙 4: HF 리비전, vLLM 버전, dtype, GPU 기종,
  temperature, seed, 프롬프트 해시, 타임스탬프, torch/CUDA/드라이버 버전).
- GPU 배타성 강제 (`evaluate.py`의 `assert_gpu_exclusive()` 무수정 재사용,
  우회 플래그 없음).

## 3. Arm A — PTX 수리 지구력(repair endurance)

- 언어: PTX (본 실험에서 두 모델 모두 거의 회복하지 못한 최약 표현, LRPL)
- 과제: `19_ReLU`, `1_Square_matrix_multiplication_`
- 모델: 본 실험과 동일 리비전 (gpt-oss-120b `b5c939de...`, Qwen3-Coder-30B-A3B-Instruct
  `b2cff646...`)
- 샘플: 과제×모델 조합당 **신규 5샘플** (본 실험 turn-1 샘플 재사용 아님 — 시드
  분리, `--seed`로 명시 기록)
- 체인 수: 2과제 × 2모델 × 5샘플 = **20 체인**
- k_max = 100 (§2-①)
- 피드백: repair-phase, `#3.4`와 동일 — 원문 컴파일/런타임 에러 텍스트만,
  correctness는 aggregate-only
- 출력: 체인별 turn×correct CSV (관찰 축: 몇 턴째에 최초 정답이 나오는지,
  나온 뒤에도 유지되는지, 100턴까지 한 번도 정답이 없는 체인은 몇 개인지)

## 4. Arm B — Triton 최적화 지구력(optimization endurance)

- 언어: Triton (본 실험에서 상대적으로 강세인 고추상화 표현, 최적화 피드백
  루프의 수렴 한계 관찰 대상)
- 과제:
  - `1_Square_matrix_multiplication_` — **하드월**: cuBLAS가 뒷받침하는 연산,
    이론적으로도 벤더 라이브러리를 능가하기 매우 어려움
  - `82_conv_depthwise_2D_square_input_square_kernel` — **소프트월**: 벤더
    지원이 약한 연산, 여유 있는 최적화 공간 예상
- 모델: Arm A와 동일 2개
- 시작점 (체인별 기록, §5 참고):
  - 본 실험에 해당 (과제,모델) 조합의 정답 체인이 있으면 → 그 best_code에서
    **최적화 단계만** 시작 (repair 재실행 없음)
  - 없으면 → 신규 5샘플로 처음부터(repair→optimize) 새 체인 시작
- k_max = 100 (§2-①)
- 피드백: optimize-phase, `#3.4`와 동일 — 측정치 + 개선 지시 1건, 프로파일러
  출력 없음
- 매 턴 correctness 재확인, best-so-far 유지
- 출력: 체인별 best-speedup@turn 궤적 CSV + eager 대비 격차, 가능하면
  `torch.compile` 대비 격차도("wall까지의 거리"가 이 실험의 y축)

## 5. Arm B 시작점 확정 (본 실험 상태 읽기 전용 조회, 2026-08-23 완료)

`results/eval/multiturn_state_a100.json`(본 실험, 쓰기 없이 조회만)에서 확인:

| task | model | 시작 방식 | 근거 |
|---|---|---|---|
| `1_Square_matrix_multiplication_` | Qwen3-Coder-30B-A3B-Instruct | 기존 best_code seed | sample4, speedup=0.9955 @turn3 |
| `1_Square_matrix_multiplication_` | gpt-oss-120b | 기존 best_code seed | sample3, speedup=2.270 @turn7 |
| `82_conv_depthwise_2D_square_input_square_kernel` | gpt-oss-120b | 기존 best_code seed | sample2, speedup=0.187 @turn5 (해당 조합 유일 정답 샘플) |
| `82_conv_depthwise_2D_square_input_square_kernel` | Qwen3-Coder-30B-A3B-Instruct | **신규 5샘플 풀체인** | 본 실험에서 0/5 ever_correct — 시드할 정답 없음 |

→ Arm B 체인 수: seed 3 + 신규 5 = **8 체인**
→ 총 체인 수(Arm A 20 + Arm B 8) = **28 체인**, k=100 기준 예상 최대 생성/평가
  ≈ 28 × 100턴 ≈ 2,800건 (조기 수렴 시 이보다 적음), 지시된 상한 ~4,000 이내

Seed 체인 3건은 본 실험 best_code를 그대로 turn-1 최적화 단계 입력으로 쓰되,
**이 실험의 GPU 배타 조건에서 신선하게 재측정**한 타이밍을 이 실험 turn-1의
`last_timing`으로 사용한다(본 실험 기록값을 그대로 갖다 쓰지 않음 — 이 실험
자체의 GPU 배타성·측정 일관성 원칙 준수).

## 6. 운영 규칙

- GPU 배타성: 서빙(generate)과 평가(evaluate) 단계 분리, 평가 중 타 프로세스 없음
  — 본 실험과 동일한 `serve→generate→stop→evaluate` 사이클.
- 매 20턴마다 중간 요약(체인 생존 현황, Arm A 정답 발생 여부, Arm B 궤적 요약)을
  `results-a100-exploratory` 브랜치에 push.
- 이상 발생 시(하니스 에러, 크래시, GPU 배타성 위반 등) **고치려 하지 말고 정지·
  기록·보고** — follower-mode 원칙과 동일.
- 최하위 우선순위 — 본 논문 관련 요청이 오면 이 실험은 즉시 중단하고 양보.
- 예상 최대 규모 ~4,000 생성/평가, 조기 수렴 시 더 일찍 종료 가능.

## 7. 산출물

- `results/exploratory/raw/` — 생성 원본 (generate.py `--out-dir` 격리)
- `results/exploratory/eval_*.json` — 평가 결과 (evaluate.py `--raw-dir`/`--out` 격리)
- `results/exploratory/state_armA.json`, `state_armB.json` — 체인 상태 (k_max=100)
- `results/exploratory/fig_armA_turn_correct.csv` — Arm A 체인별 turn×correct
- `results/exploratory/fig_armB_speedup_trajectory.csv` — Arm B 체인별
  best-speedup@turn + eager/(가능시 torch.compile) 격차

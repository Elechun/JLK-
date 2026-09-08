# DWI 기반 급성 뇌경색 병변 분할 + 뇌졸중 병인 분류 — 에이전트 협업 개인 프로젝트

> 제이엘케이(JLK) 전문연구요원 지원을 위한 **개인 포트폴리오 프로젝트**입니다. 공개 데이터(OpenNeuro SOOP, CC0)로
> JLK 의 JBS-01K 와 **같은 과제 구조**(DWI → 병변 분할·부피 → 뇌졸중 유형 분류)를 처음부터 끝까지 직접 구현·검증하는 것이 목표입니다.
> 회사 제품과의 성능 비교나 우위 주장은 하지 않으며, 임상 사용을 위한 것이 아닙니다. JLK 제품은 "문제 구조의 참조점"으로만 언급합니다.

## 무엇을 하는가
| 단계 | 내용 | 산출물 |
|---|---|---|
| 1. 데이터 | OpenNeuro **ds004889 (SOOP)**: 급성 뇌졸중 1,715명 DWI(TRACE b1000)+ADC, 급성 병변 마스크 1,451명, 병인 라벨(LAA/CE/SVO/기타/원인불명), NIHSS·mRS | `data/index.csv`, `data/splits.json` (환자 단위, seed 2026) |
| 2. 분할 | 2D U-Net(2채널, 128²) → 환자 단위 3D Dice, 환자 단위 검출(비어 있지 않은 출력률 **과** 병변 접촉률을 나란히; 음성 환자가 0명이라 특이도는 계산 불가), 병변 **부피 ICC(+CI)·Bland-Altman**, 소병변 Dice+검출률+%오차 | `runs/seg_unet2d/`, `results/` |
| 3. 분류 | 병변 특징(부피·개수·편측성·다영역 proxy) + 임상변수 → 병인 4클래스. 부피 단독 기준선·`full`·`scale_invariant` 등 6개 세트를 나란히 보고(모든 AUC 에 부트스트랩 CI). 예측 마스크 기반 dev CV 는 **분류 단계 CV** 이며 전체 파이프라인 OOF 가 아님(A5b) | `results/cls_*.json` |
| 4. 검증 | 에이전트 A1~A6 + Astra 가 주제·데이터·전처리·모델·방법론·진단을 교차검증한 보고서 | `docs/agents/*.md`, `docs/findings_log.md` |

## 에이전트 구성 (참조 구성의 이름·모델 그대로, 주제만 교체)
| 에이전트 | 모델 | 담당 | 상태 / 주요 성과 |
|---|---|---|---|
| **A1** 주제·포지셔닝 | Opus | JLK 공고·제품군 확인, README 평가 | ✅ JBS-01K 와 1:1 과제 대응 확인. 갭 P0 3건(DICOM, 2.5D/3D, 민감도·특이도 보고) 지적 → 민감도는 `metrics.py` 에 반영, 특이도는 음성 환자 0명이라 **계산 불가**(Astra 반박, NaN 으로 출력) |
| **Astra** 방향성 교차검증 | Fable | 주제 탐색·대안 데이터셋·범위 판정, A1 결론 반박/동의 | ✅ **현 방향 유지+범위 축소** 판정. A1 P0 중 DICOM·특이도 반박(음성군 없음), 2.5D 는 P1. SVO 가 부피만으로 분리됨(중앙값 1.13 mL) → 부피 단독 기준선 의무화, Cryptogenic 제외 |
| **A2** 데이터 적합성 | Opus | SOOP 특성, 결측, confound | ✅ **부피 특징 1개로 4클래스 macro AUC 0.650** 측정 → 헌장의 "크기 제외 AUC > 0.60" 기준이 학습 없이 통과됨을 폭로(A6 로 이관). 미해결이던 마스크 값 2/3 (10명) 을 **강도 코딩 아티팩트**로 종결(`>0` 이진화 타당). 임상변수 결측 399명이 단일 블록·병변 절반 크기 → MCAR 아님. GE·Siemens 39명 전원 라벨 없음 → 분류 코호트 100 % Philips |
| **A3** 전처리·통계 | Opus | RNG, split, 표시변환 | ✅ **ADC 정규화 버그 발견**: MAD=0 미방어로 **44/1233명(3.6 %)의 ADC 채널 전체가 ±6 포화**(최악 66 % 복셀). sub-235 는 ADC 격자가 최대 27 mm 어긋난 채 스택. 증강 RNG 가 워커로 fork 되어 `num_workers>0` 재현 불가(잠복). split 3회·전처리 1,233명 재실행 해시 대조. 테스트 15 → 35 |
| **A4** 모델·최적화 | Opus | 학습 설정, 속도 | ✅ **클립 ±6 이 병변 신호의 절반을 잘라내고 있었다**(병변 TRACE z 중앙값 +6.24, 병변 복셀 50.7 %가 z>6). clip 10 으로 **val Dice 0.5950 → 0.6826**(시드 잡음의 12배). `size` 키를 **어떤 코드도 읽지 않던** 버그 수정. bf16+channels_last 로 1.96배 가속하되 `cudnn.benchmark` 는 재현성 때문에 거부. **Dice +0.010 을 포기하고 ICC 여유를 택해 epochs 40 확정**. 테스트 35 → 42 |
| **A5a** 방법론 교차검증 | **Codex (OpenAI GPT-6)** | 수치·알고리즘 | ✅ **독립 판정 [보류]**. `metrics.py` 전 함수를 NumPy 로 독립 구현해 1e-10 일치(지표 계층 무결). 그러나 [높음] 4건: **RAS 좌우축이 특징·증강에서 뒤바뀜**, **"test 를 읽지 않았다"는 문자 그대로 거짓**(인덱스 생성·A2 부피 재계산), 헌장 v2 는 dev 결과를 본 뒤 개정, 예측마스크 CV 는 전체 파이프라인 OOF 아님(447명 중 367명이 seg-train) |
| **A5b** 방법론 교차검증 | Fable | 수치·알고리즘 (독립), A5a 반박 | ✅ A5a 17건 판정(`docs/agents/A5b_report.md`). 확정: 캐시 좌우축(H)·특징 수정, 증강 축 `flip_axis` 명시, 검출 지표 이원화, 분류 CI·fold·seg-split 보고, test 열람 규칙 재정의 |
| **A6** 목표·진단 설계 | Opus | 성공기준, 로깅, 누수 진단 | ✅ **자기 프로젝트의 성공기준을 스스로 폐기**: `volume_only` 0.6421 로 v1 기준이 무의미함을 재현 → **E1 LAA-vs-CE·E2 부피 대비 증분·E3 SVO 제외 3클래스**로 대체(v1 원문·사유 보존). 통과가 보장된 S1 을 보완할 **S1b 신설**, ICC **CI 병기 의무화**(CI 가 임계값 포함). threshold 스윕 폭 0.0057 → 0.5 고정. **val 실패 사례의 절반은 라벨**(11명이 만성 병변 서명, Dice 0.172 vs 0.713). 테스트 42 → 46 |

## 결과 (테스트셋 1회 평가, 2026-09-09)

평가 설정은 **테스트셋을 보기 전에** `docs/00_project_charter.md` §C 에 고정했다: 체크포인트 `runs/seg_unet2d/best.pt`
(epoch 33, val 로만 선택) · threshold 0.5 · fp32 · TTA 없음 · 후처리 없음. 평가 후 재학습·재선택·기준 수정은 하지 않았다.

### 분할 (test 218명, 전원 병변 양성)
| 지표 | 사전 등록 기준 | val | **test** | 판정 |
|---|---|---|---|---|
| Dice (환자 단위 3D, GT 양성 평균) | ≥ 0.55 | 0.6855 | **0.6877** [0.654, 0.721] | **PASS** |
| val→test 일반화 \|Δ Dice\| | ≤ 0.06 | — | **0.0022** | **PASS** |
| 환자 단위 검출 (비어 있지 않은 출력) | ≥ 0.90 | 0.9725 | **0.9771** (213/218, FN 5) | **PASS** |
| 〃 (예측이 실제 병변에 접촉) | 보고 | 0.9312 | **0.9679** | — |
| 병변 부피 ICC(2,1) | ≥ 0.85 (CI 병기) | 0.9138 | **0.9839** [0.963, 0.992] | **PASS** |
| 부피 오차 | 보고 | — | MAE **5.21 mL** · MAPE 37.5 % · 중앙값 \|%오차\| **19.8 %** · bias −0.74 mL · LoA [−22.0, +20.5] | — |
| 소병변 <2 mL (n=60) | 보고 | — | Dice **0.5517** · 검출 0.950 · 중앙값 \|%오차\| 42.6 % | — |

병변 크기별 Dice: <2 mL **0.552** (n=60) · 2–10 mL **0.689** (n=57) · 10–50 mL **0.707** (n=60) · ≥50 mL **0.857** (n=41).

> **특이도는 보고하지 않는다.** 마스크 보유 1,451명이 전원 병변 양성이고 `acuteischaemicstroke` 관측값도 전부 1이라
> 음성 환자가 0명 → 환자 단위 특이도·PPV 는 이 데이터로 **계산 자체가 불가능**하다 (Astra 지적, A2 독립 확인).

### 병인 분류 (test 80명: LAA 35 / CE 22 / SVO 13 / Others 10, 예측 마스크 기준)
| 사전 등록 기준 | dev 5-fold CV | **test** | 판정 |
|---|---|---|---|
| **E1** LAA-vs-CE AUC > 0.55 (CI 하한 > 0.50 이어야 PASS) | 0.6709 [0.611, 0.730] | **0.5909** [0.438, 0.739] | **판정 불가** (CI 가 0.50 포함) |
| **E2** macro(`full`) − macro(`volume_only`) > 0 | +0.0752 | **+0.0352** (0.6687 − 0.6335) | **PASS** |
| **E3** SVO 제외 3클래스 macro > 0.55 | 0.6750 | **0.6170** [0.506, 0.726] | **PASS** |

**정직한 음성 결과.** test 에서 **부피 단독 기준선이 `full` 모델보다 LAA-vs-CE AUC 가 높다(0.6013 vs 0.5909)**.
dev 에서는 `full` 이 +0.084 앞섰는데 역전됐고, GT 마스크 오라클에서도 같은 역전이 나타난다(0.6234 vs 0.6351).
즉 **크기 지름길이 가장 약한 축에서 이 모델은 특징 1개짜리 기준선을 넘지 못했다.** E1 을 성공으로 주장하지 않는다.
SVO 의 높은 AUC(0.85)는 모델의 병인 판별력이 아니라 **TOAST 정의상 "병변 <1.5 cm"** 라는 라벨 정의의 재현이다.

### 한계 (결과와 무관하게 사전 등록)
1. **홀드아웃이 완전한 blind 가 아니다** — 층화 split 이 `etiology_code` 와 병변 크기 구간을 쓰므로 test 의 **라벨 메타데이터**가
   split 경계에 반영돼 있다. 또한 층화 split 을 만들려면 인덱스가 전 코호트 마스크 부피를 읽어야 한다(회피 불가).
   test 영상·마스크가 학습·모델 선택·평가에 쓰인 흔적은 0 이다 (A5a 지적 → A5b 판정, `CLAUDE.md` 에 규칙으로 명문화).
2. 분류 코호트는 **100 % Philips**, 임상변수 결측 425명(병변 크기 중앙값이 절반)이 제외된 낙관적 표본이다.
3. 분할 코호트 1,451명은 "숙련 rater 가 그릴 수 있었던 병변"으로 이미 필터링돼 있다.
4. 예측 마스크 dev CV 는 **분류 단계 CV** 이며 전체 파이프라인 OOF 가 아니다 (447명 중 367명이 분할 모델 학습에 사용됨).
5. E1 의 PASS 규칙은 test 표본 크기에서 실제 검정력이 **0.54** 다(판정 불가 0.40). 단일 데이터셋·단일 기관 결과다.
6. 학습 당시 좌우 반전 증강은 실제로는 **전후 반전**이었다(A5a 발견, `flip_axis: ap` 로 명시). 올바른 좌우 반전은
   3 시드에서 Dice +0.009 를 주지만 ICC 는 이득이 없어, 기준 확정 후 모델을 바꾸지 않기 위해 재학습하지 않았다.

## 빠른 시작 (서버)
```bash
git clone https://github.com/Elechun/JLK-.git && cd JLK-
python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]
bash scripts/run_all.sh          # 다운로드(2.8 GB) → 인덱스 → split → 전처리 → 테스트 → 학습 → 진단 → 분류
```
자세한 순서와 에이전트 실행 방법은 **`docs/01_server_runbook.md`**, 새 Claude Code 채팅에 붙여 넣을 프롬프트는 **`docs/02_handoff_prompt.md`**.

## 저장소 구조
```
.claude/agents/      A1~A6, Astra 에이전트 정의 (Claude Code 서브에이전트)
configs/             학습 하이퍼파라미터 (변경은 여기서만)
docs/                헌장(성공기준) · 서버 런북 · 인수인계 프롬프트 · 에이전트 보고서 · 발견 로그
scripts/             download_soop → build_index → make_splits → preprocess → train_seg → eval_seg → predict_masks → train_cls → diagnose
src/strokeai/        data(index/split/preprocess/dataset) · models(unet2d) · losses · metrics · features · train · utils
tests/               지표·split·모델/손실·전처리/특징·누수 방지 단위 테스트 (52개)
```

## 데이터 출처
Stroke Outcome Optimization Project (SOOP), OpenNeuro ds004889 v1.1.2, CC0. Rorden C, Absher J, Newman-Norlund R.
논문: *The stroke outcome optimization project: Acute ischemic strokes from a comprehensive stroke center*, Scientific Data (2024).
환자 데이터(`data/`)와 학습 산출물(`runs/`)은 커밋하지 않습니다.

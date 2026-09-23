# DWI 기반 급성 뇌경색 병변 분할 + 뇌졸중 병인 분류

> 공개 데이터(OpenNeuro SOOP, CC0)로
> (DWI → 병변 분할·부피 → 뇌졸중 유형 분류)를 처음부터 끝까지 직접 구현·검증하는 것이 목표입니다.
> 회사 제품과의 성능 비교나 우위 주장은 하지 않으며, 임상 사용을 위한 것이 아닙니다. JLK 제품은 "문제 구조의 참조점"으로만 언급합니다.

## 무엇을 하는가
| 단계 | 내용 | 산출물 |
|---|---|---|
| 1. 데이터 | OpenNeuro **ds004889 (SOOP)**: 급성 뇌졸중 1,715명 DWI(TRACE b1000)+ADC, 급성 병변 마스크 1,451명, 병인 라벨(LAA/CE/SVO/기타/원인불명), NIHSS·mRS | `data/index.csv`, `data/splits.json` (환자 단위, seed 2026) |
| 2. 분할 | 2D U-Net(2채널, 128²) → 환자 단위 3D Dice, 검출 민감도/특이도, 병변 **부피 ICC·Bland-Altman**, 소병변 Dice 별도 | `runs/seg_unet2d/`, `results/` |
| 3. 분류 | 병변 특징(부피·개수·편측성·다영역 proxy) + 임상변수 → 병인 4클래스. **크기 특징 포함/제외** 두 세트를 나란히 보고 | `results/cls_*.json` |
| 4. 검증 | 에이전트 A1~A6 + Astra 가 주제·데이터·전처리·모델·방법론·진단을 교차검증한 보고서 | `docs/agents/*.md`, `docs/findings_log.md` |

## 에이전트 구성 (참조 구성의 이름·모델 그대로, 주제만 교체)
| 에이전트 | 모델 | 담당 | 상태 / 주요 성과 |
|---|---|---|---|
| **A1** 주제·포지셔닝 | Opus | JLK 공고·제품군 확인, README 평가 | ✅ JBS-01K 와 1:1 과제 대응 확인. 갭 P0 3건(DICOM, 2.5D/3D, 민감도·특이도 보고) 지적 → 민감도·특이도는 `metrics.py` 에 반영 |
| **Astra** 방향성 교차검증 | Fable | 주제 탐색·대안 데이터셋·범위 판정, A1 결론 반박/동의 | ✅ **현 방향 유지+범위 축소** 판정. A1 P0 중 DICOM·특이도 반박(음성군 없음), 2.5D 는 P1. SVO 가 부피만으로 분리됨(중앙값 1.13 mL) → 부피 단독 기준선 의무화, Cryptogenic 제외 |
| **A2** 데이터 적합성 | Opus | SOOP 특성, 결측, confound | ⏳ 서버에서 실행 (인덱스 요약은 `data/index_summary.json`) |
| **A3** 전처리·통계 | Opus | RNG, split, 표시변환 | ⏳ 서버에서 실행 (split·전처리·테스트 코드는 준비됨) |
| **A4** 모델·최적화 | Opus | 학습 설정, 속도 | ⏳ 서버에서 실행 (CPU 스모크: 40명 15 s/epoch, grad clip 발동 0 %) |
| **A5a** 방법론 교차검증 | Codex (GPT) | 수치·알고리즘 | ⏳ 서버에서 실행 |
| **A5b** 방법론 교차검증 | Fable | 수치·알고리즘 (독립), A5a 반박 | ⏳ 서버에서 실행 |
| **A6** 목표·진단 설계 | Opus | 성공기준, 로깅, 누수 진단 | ⏳ 서버에서 실행 (사전 등록 기준은 `docs/00_project_charter.md`) |

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
tests/               지표·split·모델/손실·전처리/특징 단위 테스트 (15개)
```

## 데이터 출처
Stroke Outcome Optimization Project (SOOP), OpenNeuro ds004889 v1.1.2, CC0. Rorden C, Absher J, Newman-Norlund R.
논문: *The stroke outcome optimization project: Acute ischemic strokes from a comprehensive stroke center*, Scientific Data (2024).
환자 데이터(`data/`)와 학습 산출물(`runs/`)은 커밋하지 않습니다.

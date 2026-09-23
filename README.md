# DWI 기반 급성 뇌경색 병변 분할 + 뇌졸중 병인 분류

> 공개 데이터(OpenNeuro SOOP, CC0)로
> (DWI → 병변 분할·부피 → 뇌졸중 유형 분류)를 처음부터 끝까지 직접 구현·검증하는 것이 목표입니다.

## 무엇을 하는가
| 단계 | 내용 | 산출물 |
|---|---|---|
| 1. 데이터 | OpenNeuro **ds004889 (SOOP)**: 급성 뇌졸중 1,715명 DWI(TRACE b1000)+ADC, 급성 병변 마스크 1,451명, 병인 라벨(LAA/CE/SVO/기타/원인불명), NIHSS·mRS | `data/index.csv`, `data/splits.json` (환자 단위, seed 2026) |
| 2. 분할 | 2D U-Net(2채널, 128²) → 환자 단위 3D Dice, 검출 민감도/특이도, 병변 **부피 ICC·Bland-Altman**, 소병변 Dice 별도 | `runs/seg_unet2d/`, `results/` |
| 3. 분류 | 병변 특징(부피·개수·편측성·다영역 proxy) + 임상변수 → 병인 4클래스. **크기 특징 포함/제외** 두 세트를 나란히 보고 | `results/cls_*.json` |
| 4. 검증 | 에이전트 A1~A6 + Astra 가 주제·데이터·전처리·모델·방법론·진단을 교차검증한 보고서 | `docs/agents/*.md`, `docs/findings_log.md` |

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

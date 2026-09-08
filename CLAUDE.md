# strokeai — DWI 기반 급성 뇌경색 병변 분할 + 병인 분류 (JBS-01K 유사 파이프라인)

## 프로젝트 규칙
- 데이터: OpenNeuro ds004889 (SOOP, CC0). `data/raw/` 이하는 절대 커밋하지 않는다.
- 모든 무작위성은 `strokeai.utils.seed_everything(seed)` 로 고정하고, split 은 **환자(subject) 단위**로만 나눈다. 슬라이스 단위 split 금지.
- 평가 지표는 `strokeai/metrics.py` 의 함수만 사용한다 (학습 코드에 지표를 재구현하지 않는다).
- 테스트셋(`test`)은 최종 1회 평가 전까지 어떤 스크립트도 읽지 않는다. 모델 선택은 `val` 로만.
- 실험 로그는 `runs/<name>/log.jsonl` 에 JSON Lines 로 남긴다 (epoch, step, loss, lr, grad_norm, 시간).
- 보고서(`docs/agents/*.md`)에는 실제로 실행해 확인한 사실만 쓴다. 추정은 "추정" 이라고 표시한다.

## 개발 명령
```bash
pip install -e .[dev]
python scripts/download_soop.py --out data/raw/ds004889     # ≈2.8 GB
python scripts/build_index.py
python scripts/make_splits.py
python scripts/preprocess.py
python scripts/train_seg.py --config configs/seg_unet2d.yaml
python scripts/eval_seg.py --run runs/seg_unet2d --split val
python scripts/train_cls.py
pytest
```

## 에이전트 구성 (`.claude/agents/`)
| 에이전트 | 역할 |
|---|---|
| A1 topic-positioning | 채용공고·제품군 조사, 주제 정당화, README 평가 |
| Astra direction-review | 주제 탐색·방향성 독립 교차검증 (A1 결론 반박/동의) |
| A2 data-suitability | 데이터셋 비교, 메타데이터 결측, confound 탐색 |
| A3 preprocessing-stats | RNG, 환자 단위 split, 정규화/리샘플링, 기술통계 |
| A4 model-optimization | 모델·손실·옵티마이저·속도, 학습 설정 버그 추적 |
| A5a methodology-review-external | 외부 모델(Codex 등)에 의한 독립 수치·알고리즘 검증 |
| A5b methodology-review-internal | 독립 수치·알고리즘 검증 (A5a 와 상호 반박) |
| A6 goals-diagnostics | 성공 기준 사전 정의, 로깅, 누수/동일성 진단 |

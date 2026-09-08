# 서버 인수인계 런북 (여기서부터 이어서 하면 됨)

## 0. 현재 상태 (2026-09-08, 클라우드 세션에서 완료된 것)
- [x] 저장소 골격, 에이전트 정의(`.claude/agents/`), `CLAUDE.md`
- [x] A1 주제·포지셔닝 보고서 `docs/agents/A1_report.md`
- [x] Astra 방향성 교차검증 보고서 `docs/agents/Astra_report.md`
- [x] 데이터 다운로더(S3 직접, 2.8 GB) · 인덱스 · 환자 단위 split (seed 2026, sha256 `0f923acd…`)
- [x] 전처리 · 2D U-Net · 학습 루프 · 지표 · 병인 분류기 · 진단 스크립트 · 단위 테스트 15개 통과
- [x] 40명/2 epoch CPU 스모크 테스트 통과 (val Dice 0.22 — 스모크일 뿐, 성능 아님)
- [ ] **전체 데이터 전처리·학습 (서버에서)** ← 다음 단계
- [ ] A2 · A3 · A4 · A5a · A5b · A6 에이전트 실행과 보고서
- [ ] 테스트셋 1회 평가, README 결과표, 발견 로그 완성

`data/`, `runs/` 는 커밋되지 않으므로 서버에서 새로 만든다(자동).

## 1. 환경
```bash
git clone https://github.com/Elechun/JLK-.git && cd JLK-
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# GPU 서버: 먼저 CUDA 버전에 맞는 torch 설치 (예: pip install torch --index-url https://download.pytorch.org/whl/cu121)
pip install -e .[dev]
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
GPU 는 자동 감지된다 (`scripts/train_seg.py --device cuda|cpu` 로 강제 가능). GPU 에서는 `configs/seg_unet2d.yaml` 의 `batch_size` 를 64~128, `size` 를 192 로 올리는 것을 A4 가 검토한다.

## 2. 파이프라인 (한 번에)
```bash
bash scripts/run_all.sh
```
단계별로 돌리려면 `CLAUDE.md` 의 "개발 명령" 참고. 예상 시간(4코어 CPU 기준): 다운로드 2~5분, 인덱스 1분, 전처리 2분, 학습 12 epoch ≈ 70분(CPU) / GPU 는 수 분.

## 3. 에이전트 실행 순서 (Claude Code 에서)
각 에이전트는 `.claude/agents/<name>.md` 에 정의돼 있다. Claude Code 를 저장소 루트에서 열고 순서대로 위임한다.
```text
1. @a2-data-suitability      → docs/agents/A2_report.md   (인덱스 생성 직후)
2. @a3-preprocessing-stats   → docs/agents/A3_report.md   (전처리 직후, 재실행 해시 비교)
3. @a4-model-optimization    → docs/agents/A4_report.md   (학습 전 벤치마크 + --device cuda 추가)
4. 학습 실행 (run_all.sh 6단계)
5. @a6-goals-diagnostics     → docs/agents/A6_report.md   (val 진단, 성공기준 확정)  ※ test 금지
6. @a5a-methodology-review-external → docs/agents/A5a_report.md
   - 원래 설계: Codex(GPT) 로 실행. `codex exec "$(cat .claude/agents/a5a-methodology-review-external.md)"`
   - Codex 가 없으면 Claude 로 실행하고 보고서 머리에 "실행 모델" 을 적는다.
7. @a5b-methodology-review-internal → docs/agents/A5b_report.md (A5a 지적 항목별 판정)
8. 확정된 수정 반영 → 테스트 재실행 → 필요 시 재학습
9. 테스트셋 1회 평가 (run_all.sh 끝의 두 명령) → README 결과표 · docs/findings_log.md 갱신
10. @a1-topic-positioning 과 @astra-direction-review 로 README 최종 검토
```
에이전트 모델은 참조 구성대로 A1~A4·A6 = Opus, A5a = Codex(GPT), A5b = Fable(이 세션 모델), Astra = Fable.

## 4. 지켜야 할 규칙 (CLAUDE.md 요약)
- 테스트셋은 9단계 전까지 아무 스크립트도 읽지 않는다. 모델 선택은 val 로만.
- split 은 환자 단위, seed 2026. `data/splits.json` 의 sha256 이 `0f923acd2f8898d58c4c62cb598c4b8d88c17ab5eaab8a72ee4e73a7342d8f43` 와 다르면 데이터가 달라진 것이니 A3 에 알린다.
- 보고서에는 실행해 확인한 수치만 쓴다.

## 5. 알려진 이슈 / 첫 실행에서 볼 것
- SOOP 의 TRACE/ADC 는 4D `(H, W, S, 1)`, 마스크는 3D. 인덱스는 앞 3차원만 비교하도록 이미 수정됨.
- 급성 마스크 10명은 라벨값이 1이 아니라 2 또는 3 (`mask_unique_values` 열). 현재 `>0` 으로 이진화. A2 가 의미를 확인할 것.
- 병인 라벨: 영상 1,715명 중 participants.tsv 에 없는 210명 + `n/a` 425명. 분류기는 594+486명만 사용.
- 나이 `89+` 문자열 → 89 로 파싱(`age_clamped_89plus` 플래그).
- ~~스모크 테스트에서 grad clip(1.0) 발동 비율 0 %, grad norm p95 ≈ 0.8~~ → **폐기(A4 §6 / A6 확인, 2026-09-09)**.
  그 값은 40명·2 epoch CPU 스모크의 산물이다. 전체 학습(`runs/seg_unet2d`)에서는 **epoch 2 에 22.2 % 가 클립**되고
  grad-norm p95 가 **1.340** 까지 오르며, 40 epoch 평균 클립 비율은 **3.2 %**, 마지막 5 epoch 은 0.4~0.9 % 다 [측정].
  `grad_clip: 1.0` 은 그대로 두면 된다(1 / 5 / 100 비교에서 차이가 잡음 이내 — A4).

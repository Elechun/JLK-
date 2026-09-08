# A4 보고서 — 모델·손실·옵티마이저·속도, 학습 설정 확정

작성일: 2026-09-09 · 담당: **A4 model-optimization** · 실행 모델: **Opus** (Claude Code, GPU 서버 세션)
하드웨어: **NVIDIA RTX A6000 48 GB × 2 (`CUDA_VISIBLE_DEVICES=0` 로 1장만 사용) · CPU 48 코어 · RAM 125 GB**
환경: Python 3.10.12 · torch 2.5.1+cu121 · numpy 2.2.6 · scipy 1.15.3 · nibabel 5.4.2
데이터 스냅샷: `data/splits.json` 내부 sha256 = `0f923acd…` (헌장 값과 일치, 변경 없음)

표기 규칙: **[측정]** = 이 보고서의 명령을 실제로 실행해 얻은 값, **[추정]** = 측정값으로부터의 산술 추론.
아래 표의 모든 Dice·시간 수치는 **[측정]** 이며, "전체 학습 예상 시간" 절만 **[추정]** 이다.

> **테스트셋 취급**: 이 보고서의 모든 실험은 `train`(1015) + `val`(218) = **1,233명**만 사용했다.
> `test` 218명은 **전처리조차 하지 않았고**(`data/cache` 에 test 환자 0명) 어떤 스크립트도 읽지 않았다.
> 내가 추가한 `scripts/analysis/a4_*.py` 는 모두 `--which train val` 이 기본값이다.

---

## 0. 요약 판정

**판정: [파이프라인 배선은 정상. 다만 "속도"보다 훨씬 큰 문제가 정규화 클립에 있었고, 그것을 고치니 val Dice 가 0.595 → 0.683 으로 뛰었다. 학습 자체는 GPU 에서 5분 이내라 속도는 병목이 아니다.]**

| # | 발견 | 심각도 | 상태 |
|---|---|---|---|
| **A4-1** | **정규화 클립 ±6 이 성능을 크게 깎고 있었다** (A3 의 F3 이관 항목). clip 6 → 10 으로 바꾸자 동일 조건 24 epoch 에서 **val Dice 0.5950 → 0.6826 (+0.088)**, 부피 ICC 0.912 → 0.959. 시드 잡음 폭(±0.007)의 **12배** | **높음** | **수정함** — `configs` 에 `clip: 10.0` 확정, **전체 캐시 1,233명 재생성 완료** |
| **A4-2** | `configs/seg_unet2d.yaml` 의 `size` 키를 **학습·전처리 어느 코드도 읽지 않았다**. yaml 을 192 로 고쳐도 128² 캐시로 조용히 학습됨 | **중간** | **수정함** — `scripts/preprocess.py` 가 config 에서 `size`/`clip`/`clip_mode` 를 읽고, `train()` 이 캐시 해상도 불일치를 assert |
| **A4-3** | 런북의 "**grad clip 발동 0 %, p95 0.82**"는 40명·2 epoch CPU 스모크의 산물. 전체 데이터에서는 **epoch 2 에 22 % 가 클립**되고 p95 가 1.34 까지 오른다(clip=6 캐시에서는 26 % / p95 2.41) | **낮음** | 기록. `grad_clip: 1.0` 유지(1 / 5 / 100 비교에서 차이 < 잡음) |
| **A4-4** | GPU 경로 미구현: AMP·`channels_last`·`pin_memory`·`num_workers` 가 전혀 없었다 | **중간** | **수정함** — 전부 config 키로 추가. 연산 처리량 1,580 → 3,695 slices/s(**2.34배**), 실제 학습 epoch 11.18 s → **5.71 s(1.96배)** |
| **A4-5** | `num_workers > 0` 재현성: A3 수정 이후 nw 0 / 8 / 16 이 **로그 수준에서 완전히 동일**(train_loss·grad-norm·val Dice 모두 비트 단위 일치) | — | 확인함. `num_workers: 8` 채택 |
| **A4-6** | `persistent_workers` 를 켜면 **안 된다**: `ds.resample()` 이 부모에서만 돌아 워커는 첫 epoch 의 슬라이스 목록을 계속 재생한다 | **낮음**(잠복) | 코드에 주석으로 못 박음. config 키로 노출하지 않음 |
| **A4-7** | ADC 채널은 **Dice 에는 기여가 없고**(2ch 0.6826 vs TRACE-only 0.6803, 잡음 이내) **부피 ICC 에는 기여한다**(0.959 vs 0.813) | — | 2채널 유지. 근거는 §7 |
| **A4-8** | `size` 192 는 이득이 없다: 24 epoch 에서 0.6812 (128² 는 0.6826), epoch 시간 5.49 s → 10.59 s | — | `size: 128` 유지 |

---

## 1. 검토 대상 코드와 정적 리뷰

`src/strokeai/models/unet2d.py`, `src/strokeai/train.py`, `src/strokeai/losses.py`,
`src/strokeai/data/dataset.py`, `configs/seg_unet2d.yaml`.

| 항목 | 리뷰 결과 |
|---|---|
| `UNet2D` | base=16, depth=4 → **1,942,433 파라미터** [측정]. 인코더 4단 + bottleneck, `ConvTranspose2d` 업샘플, skip concat. 배선 이상 없음(§4 과적합 테스트로 확증) |
| `cosine_warmup` | warmup 구간에서 `base_lr*(step+1)/warmup`, 이후 cosine → `min_lr`. `warmup=0` 이어도 0으로 나누지 않는다(`step < 0` 이 성립하지 않아 분기 자체를 안 탐). §3 에서 실제 값 검증 |
| `soft_dice_loss` | 배치 전체 합산(micro-Dice). 빈 마스크 슬라이스가 0/0 을 만들지 않는 대신 **큰 병변이 작은 병변을 압도**하고 값이 배치 구성에 의존. §5 에서 정량화 |
| `SliceDataset` | A3 가 고친 항목별 결정론 RNG 확인. 증강 스케일/시프트가 `size=(2,1,1)` 로 채널 수를 하드코딩하고 있어 `x.shape[0]` 로 바꿈(2채널일 때 값은 완전히 동일 — ADC ablation 을 위해 필요) |
| `train()` | GPU 경로 전무(AMP·channels_last·pin_memory 없음), `x.to(device)` 가 blocking. **A4-4 로 수정** |
| `configs/seg_unet2d.yaml` | `size` 키가 어디서도 읽히지 않음. **A4-2 로 수정** |
| `predict_subject` | 환자 1명씩 fp16→fp32 변환 후 GPU 전송. val 218명 전체가 **1.3 s** [측정] 이라 최적화 가치 없음 |

---

## 2. 벤치마크 (a) 데이터로더 처리량 · (b) forward/backward 시간

### 2.1 순수 연산 (합성 텐서, 128², warmup 10 + 측정 30 step) [측정]
`results/a4/a4_bench_compute.json`

| batch | AMP | channels_last | cudnn.benchmark | s/step | slices/s | peak mem (MiB) |
|---:|---|---:|---:|---:|---:|---:|
| 64 | fp32 | ✗ | ✗ (현행 기본) | 0.04052 | 1,580 | 1,475 |
| 64 | fp32 | ✓ | ✗ | 0.03627 | 1,765 | 1,476 |
| 64 | bf16 | ✗ | ✗ | 0.03173 | 2,017 | 803 |
| 64 | **bf16** | **✓** | **✗** | **0.01732** | **3,695** | **793** |
| 64 | bf16 | ✓ | ✓ | 0.01696 | 3,774 | 1,737 |
| 64 | fp16 | ✓ | ✓ | 0.01806 | 3,544 | 1,739 |
| 128 | bf16 | ✓ | ✗ | 0.03277 | 3,906 | 1,558 |
| 256 | bf16 | ✓ | ✓ | 0.06318 | 4,052 | 6,912 |

**핵심**: 속도의 거의 전부는 `channels_last` 에서 나온다(bf16 기준 2,017 → 3,695 slices/s, **+83 %**).
`cudnn.benchmark` 는 **+2.1 %** 에 불과하다(3,695 → 3,774). 즉 **재현성을 포기할 이유가 없다** — §8 참조.
모델이 1.9 M 파라미터로 작아 batch 64 이상에서는 처리량이 포화한다(3,695 → 4,052, +10 %).

### 2.2 데이터로더 단독 (캐시는 이미 RAM 상주, 60 batch) [측정]
`results/a4/a4_bench_loader.json`

| batch | num_workers | slices/s | s/batch | 첫 배치 지연 |
|---:|---:|---:|---:|---:|
| 64 | 0 | 5,305 | 0.0121 | 0.088 s |
| 64 | 2 | 7,430 | 0.0086 | 0.077 s |
| 64 | **8** | **8,036** | **0.0080** | 0.046 s |
| 64 | 16 | 7,601 | 0.0084 | 0.042 s |
| 128 | 16 | 10,448 | 0.0123 | 0.139 s |

`SubjectCache` 가 전 split 을 RAM 에 올려두므로 로더는 **연산보다 빠르다**(5,305 vs 3,695 slices/s).
워커를 늘리는 이득은 "로더가 느려서"가 아니라 **연산과 겹치기 때문**이다. 워커 fork 비용은 첫 배치 0.05 s 수준이라
epoch 마다 워커를 새로 띄우는 현재 구조(= `persistent_workers=False`, A4-6 때문에 필수)의 오버헤드는 무시 가능하다.

### 2.3 실제 epoch (train 1015명, 실데이터) [측정]

| 설정 | train s/epoch | val s/epoch | 12 epoch 벽시계 | val Dice(12 ep) |
|---|---:|---:|---:|---:|
| 인계 config (bs 32, nw 0, fp32) | 12.79 | 1.68 | 178 s | 0.5789 |
| 최적 (bs 64, nw 8, bf16, channels_last, pin) | **5.48** | **1.32** | **86 s** | 0.5734 |
| 최적에서 AMP·channels_last 만 제거 | 11.04 | 1.72 | 157 s | 0.5679 |
| 최적에서 num_workers 만 0 | 6.88 | 1.28 | 102 s | 0.5734 |

위 4행은 clip = 6 캐시(stage-1)에서 측정했다. **확정 캐시(clip 10)에서 12 epoch 로 재측정한 값** [측정]:

| 설정 | train s/epoch | val s/epoch | 12 epoch 벽시계 | val Dice(12 ep) |
|---|---:|---:|---:|---:|
| 확정 (bs 64, nw 8, bf16, channels_last, pin) | **5.71** | 1.32 | **88 s** | 0.664604 |
| 위에서 AMP·channels_last 만 제거(fp32) | 11.18 | 1.71 | 159 s | 0.665876 |
| 위에서 num_workers 만 0 | 7.00 | 1.28 | 103 s | 0.664604 |
| 위에서 num_workers 만 16 | 6.39 | 1.29 | 96 s | 0.664604 |

**bf16 + channels_last 로 학습 구간이 1.96배 빨라지고 val Dice 는 0.0013 차이(잡음 이내)** 이다.

---

## 3. 벤치마크 (d) LR 스케줄이 의도대로 도는가 [측정]

bs 64 → **234 step/epoch** 이다. 아래는 60 epoch(총 14,040 step, `warmup_frac 0.05` → warmup **702 step**)로
검증한 값이다. 확정값인 40 epoch 에서도 동일한 형태이며 warmup 은 468 step(= 2 epoch)이 된다.

| 확인 항목 | 값 | 판정 |
|---|---|---|
| step 0 의 lr | 1.42e-06 | warmup 시작 ✅ |
| warmup 마지막 step(701)의 lr | 1.000e-03 | 정확히 base_lr 도달 ✅ |
| warmup 직후 step(702)의 lr | 1.000e-03 | 이음매 불연속 없음 ✅ |
| 최대 lr 위치 | step 701 (= epoch 3.0) | 의도대로 ✅ |
| peak 이후 단조 감소 | True | ✅ |
| 마지막 step 의 lr | 2.000e-05 | `min_lr` 2e-05 도달(오차 < base_lr 의 1 %) ✅ |

실제 학습 로그(`runs/a4_ep60_bs64/log.jsonl` 의 `lr_end`)와도 일치한다 [측정]:
epoch 0 끝 3.333e-04 → epoch 2 끝 1.000e-03(최대) → epoch 3 끝 9.993e-04(감소 시작) → 마지막 epoch 끝 2.000e-05.
회귀 테스트로 고정: `tests/test_train_options.py::test_lr_schedule_warms_up_then_decays_to_min_lr`.

---

## 4. 과적합 sanity check [측정]

배치 2개(= 슬라이스 128장, 양성 픽셀 비율 0.906 %)를 300 step 동안 암기시킨 결과
(`scripts/analysis/a4_sanity.py`, GPU 44 s):

| step | loss | batch0 hard Dice |
|---:|---:|---:|
| 0 | 1.5628 | 0.063 |
| 100 | 0.885 | 0.928 |
| 200 | 0.120 | 0.978 |
| 299 | **0.0255** | **0.993** |

**loss 가 1.56 → 0.026, hard Dice 0.99** → 모델·손실·옵티마이저·마스크 정렬 배선에 문제가 없다는 증거다.
(`tests/test_model_and_loss.py::test_overfit_two_batches` 로도 고정되어 있음.)

---

## 5. 손실 함수 수치 안정성 [측정]

### 5.1 극단 입력 (`bce_dice_loss`, smooth = 1.0)

| 케이스 | loss | 유한? | grad 유한? | grad abs max |
|---|---:|---|---|---:|
| 빈 마스크 + logit 0 | 1.6929 | ✅ | ✅ | 6.1e-05 |
| 빈 마스크 + logit −20 (완벽 예측) | 1.69e-05 | ✅ | ✅ | 2.1e-09 |
| 빈 마스크 + logit +20 (최악) | 21.000 | ✅ | ✅ | 1.2e-04 |
| 전면 마스크 + logit +20 | 2.1e-09 | ✅ | ✅ | 0.0 |
| **logit 1e4** (극단) | 10001.0 | ✅ | ✅ | 1.2e-04 |

`binary_cross_entropy_with_logits` 의 log-sum-exp 안정화 덕에 **NaN/Inf 가 한 건도 나오지 않았다**.
smooth=1 은 빈 마스크 배치에서 `1 − 1/1 = 0` 을 주므로 "빈 슬라이스 = 공짜 보상" 병리도 없다
(빈 배치·완벽 예측의 dice loss = 3.4e-05 [측정]).

### 5.2 배치 전체 합산(micro-Dice)의 영향 — A3/과제 항목 4

실제 배치 200개를 뽑아 **고정된 예측기**로 손실만 계산해 배치 구성 효과를 분리했다 [측정]:

| batch_size | 배치당 양성 픽셀 p5 / p50 / p95 | p95/p5 | dice(batch) 평균±SD | dice(sample) 평균±SD | dice(batch)–양성픽셀 상관 |
|---:|---|---:|---|---|---:|
| 32 | 1,187 / 2,940 / 4,882 | 4.12× | 0.9894 ± **0.0038** | 0.9905 ± 0.0031 | **−0.9996** |
| 64 | 3,318 / 6,066 / 8,850 | 2.67× | 0.9890 ± **0.0032** | 0.9903 ± 0.0024 | **−0.9996** |

- **영향의 크기**: 예측이 전혀 변하지 않아도 배치 구성만으로 Dice 항이 SD 0.0032(bs 64) 만큼 흔들린다.
  같은 조건에서 슬라이스별 평균 방식의 SD 는 0.0024 이므로, **배치합산 때문에 추가로 생기는 변동은 SD 0.001 수준**이다.
  실제 학습에서 이 변동이 문제였다면 `dice_reduction: sample` 이 더 나은 결과를 냈어야 하는데 그렇지 않았다(아래).
- **방향성**: 상관 −0.9996 은 "배치에 큰 병변이 들어오면 Dice 항이 작아진다"는 뜻이다. 즉 이 항은 사실상
  **큰 병변에 가중된 micro-Dice** 이고, 헌장이 별도 보고하도록 요구한 **소병변(<2 mL)에는 상대적으로 약한 신호**를 준다.
- **batch_size 를 키우면 이 변동이 줄어든다**(0.0038 → 0.0032). bs 64 를 고른 부수적 근거.
- **실측 비교**: 배치합산 대신 슬라이스별 Dice 평균(`dice_reduction: sample`)으로 학습하면 24 epoch val Dice
  **0.6772** (기본 0.6826) — 차이가 시드 잡음(±0.007) 이내라 **바꿀 근거가 없다**. 기본값(`batch`) 유지.
  다만 두 방식을 config 로 전환할 수 있게 만들어 두었고, 소병변 Dice 가 문제가 되면 A6 가 이 스위치를 쓸 수 있다.

### 5.3 클래스 불균형
전체 슬라이스의 병변 픽셀 비율 **0.339 %**, `neg_pos_ratio: 1.0` 샘플링 후 배치 내 비율 **0.91 %** [측정]
(약 110 : 1). `pos_weight: 8.0` 을 넣어 보았으나 24 epoch val Dice **0.6775** (기본 0.6826) 로 이득 없음 → `null` 유지.
`neg_pos_ratio: 2.0` 은 0.6865 로 미세하게 높지만 잡음 이내이고 epoch 시간이 5.49 → 8.01 s 로 46 % 늘어 채택하지 않는다.

---

## 6. 벤치마크 (c) grad norm 분포와 clip 발동 비율 [측정]

런북에 적힌 "grad clip 발동 0 %, p95 0.82" 는 **40명 · 2 epoch CPU 스모크**의 값이고, 전체 데이터에서는 다르다.

**확정 config 완주 run (`runs/a4_ep40_bs64`, clip 10 캐시)** [측정]:

| epoch | grad-norm 평균 | p95 | clip(1.0) 발동 비율 |
|---:|---:|---:|---:|
| 0 | 0.74 | 0.89 | 0.0 % |
| 1 | 0.54 | 0.64 | 0.0 % |
| **2** (loss 급락 시작) | 0.84 | **1.34** | **22.2 %** |
| 3 | 0.65 | 1.25 | 13.7 % |
| 4 | 0.57 | 1.17 | 8.5 % |
| 10 | 0.42 | 0.85 | 3.0 % |
| 20 | 0.33 | 0.71 | 2.6 % |
| 39 | 0.23 | 0.42 | 0.4 % |
| **40 epoch 전체 평균** | — | — | **3.2 %** |

clip = 6 캐시(stage-1)에서는 더 심해서 마지막 epoch 의 p95 가 2.41, 12 epoch 평균 발동률이 26 % 였다 —
**즉 클립을 넓혀 입력 포화를 없앤 것이 그래디언트도 안정시켰다**.

**grad clip 은 초반 2~4 epoch 에만 실제로 작동**하고 이후엔 거의 놀고 있다. clip 값을 1 / 5 / 100 으로 바꾼 비교
(24 epoch, clip=6 캐시)는 val Dice 0.5950 / 0.5941 / 0.5916 으로 **모두 시드 잡음 이내** — 성능 차이는 없다.
초반 폭주를 막는 보험으로서 비용이 0 이므로 **`grad_clip: 1.0` 을 유지**한다.

---

## 7. A3 이관 항목 처리

### 7.1 클립 실험 (A3 F3) — **가장 큰 발견**

**(1) 원본 z-분포 재측정** [측정] — train 150명을 무작위 추출해 **클립 전** robust z 를 다시 계산
(`scripts/analysis/a4_clip_stats.py`, 결과 `results/a4/a4_clip_stats.json`):

| 통계 (환자별 값의 평균) | TRACE | ADC |
|---|---:|---:|
| 병변 복셀 z 중앙값 | **+6.24** | +1.49 (중앙값 기준 −0.22) |
| 병변 복셀 z p99 | +10.53 | +3.84 |
| **병변 복셀 중 z > 6 비율** | **50.7 %** | 0.9 % |
| 병변 복셀 중 z > 10 비율 | 11.9 % | 0.7 % |
| 병변 복셀 중 z > 14 비율 | 1.9 % | 0.7 % |
| 비병변 뇌 복셀 중 z > 6 비율 | 1.9 % | 0.7 % |
| 비병변 뇌 복셀 중 z > 10 비율 | **0.06 %** | 0.6 % |
| 병변 복셀 중 z < −6 비율 | **0 %** | 0 % |

원본 격자에서 병변의 **절반**이 ±6 클립에 걸린다(A3 가 128² 캐시에서 잰 34.6 % 와의 차이는 리샘플링 평활 효과다.
같은 방식으로 캐시에서 재측정하면 clip6 캐시는 28.9 %, clip10 캐시는 5.5 %, clip14 는 1.8 % [측정]).
반대로 **정상 뇌 조직 중 +6 을 넘는 것은 1.9 % 뿐**이므로, 클립을 넓혀도 "이상치 억제" 기능은 거의 손해가 없다.
음의 방향은 병변이 전혀 도달하지 않으므로 무관하다.

**(2) 실제 학습 비교** — 4가지 캐시를 각각 새로 만들고(1,233명 × 4) 동일 config·동일 시드로 24 epoch 학습 [측정]:

| 캐시 | 병변 복셀 포화율(캐시 기준) | val Dice(best) | 검출 민감도 | 부피 ICC(2,1) | epoch 시간 |
|---|---:|---:|---:|---:|---:|
| `clip 6` (기존) | 28.9 % | 0.5950 | 0.963 | 0.912 | 5.58 s |
| **`clip 10`** | **5.5 %** | **0.6826** | 0.968 | **0.959** | 5.67 s |
| `clip 14` | 1.8 % | 0.6799 | 0.977 | 0.898 | 5.65 s |
| `pct 0.1` (환자별 p0.1–p99.9) | 5.1 % | 0.6804 | 0.954 | 0.909 | 5.66 s |

**결론**: 넓은 클립 3종이 모두 clip 6 을 **+0.085 이상** 앞선다. 시드 잡음이 ±0.007(§8.1)이므로 **압도적**이다.
셋 사이 차이(0.6826 / 0.6804 / 0.6799)는 잡음 이내라 가장 단순하고 ICC 가 가장 높은 **`clip: 10.0` (mad)** 를 채택했다.
백분위 방식은 환자마다 클립값이 달라져(캐시 상 TRACE max p10 4.84 ~ p90 11.78 [측정]) 채널 스케일이 환자 간에 흔들리는데,
이득이 없으므로 굳이 쓸 이유가 없다.

**캐시 재생성**: `configs/seg_unet2d.yaml` 에 `clip: 10.0` 을 넣고 **train+val 1,233명 전량을 재생성 완료**했다
(오류 0건, 24 프로세스로 약 20 초). 검증 [측정]:
- 재생성 캐시 60명 표본이 실험용 `data/cache_clip10` 과 **바이트 단위 동일**
- **마스크 배열과 `voxel_volume_mm3` 는 clip 6 캐시와 완전히 동일** (클립은 강도만 건드린다)
- 이전 캐시는 `data/cache_clip06/` 에 보존(실험 재현용). `_manifest.json` 에 `clip: 10.0, clip_mode: mad` 기록

### 7.2 `num_workers` 상향과 재현성 (A3 이관 2)

확정 캐시(clip 10)·확정 런타임(bf16 + channels_last)으로 **12 epoch 를 4번** 돌린 결과 [측정] (`runs/a4_infra_*`):

| 실행 | num_workers | train_loss(마지막 epoch) | grad_norm_p95 | val Dice | train s/epoch |
|---|---:|---:|---:|---:|---:|
| infra_opt | 8 | 0.1496461958138861 | 0.8245091408491134 | 0.664604448904985 | 5.71 |
| infra_opt_rep (동일 시드 재실행) | 8 | 0.1496461958138861 | 0.8245091408491134 | **0.664604448904985** | 5.71 |
| infra_opt_nw0 | 0 | 0.1496461958138861 | 0.8245091408491134 | **0.664604448904985** | 7.00 |
| infra_opt_nw16 | 16 | 0.1496461958138861 | 0.8245091408491134 | **0.664604448904985** | 6.39 |

**같은 시드 2회 실행의 val Dice 차이 = 0.000000**(헌장 기준 < 0.02 충족), 그리고 **워커 수를 0/8/16 중 무엇으로 바꿔도
train_loss·grad-norm·val Dice 가 소수점 15자리까지 같다**. A3 의 항목별 결정론 RNG 수정이 실제로 동작함을 확인했다.
(clip=6 캐시에서 돌린 stage-1 에서도 동일한 4중 일치를 확인했다: `runs/a4_stage1_clip6/a4_infra_*`, 모두 0.5734.)
`num_workers: 8` 을 채택한다 — nw 0 대비 epoch 7.00 s → 5.71 s(**18 % 단축**), nw 16 은 6.39 s 로 오히려 느리다.

### 7.3 ADC 채널 ablation (A3 이관 3) [측정] — 24 epoch, clip 10 캐시

| 입력 | 파라미터 | val Dice(best) | 검출 민감도 | **부피 ICC(2,1)** | epoch 시간 |
|---|---:|---:|---:|---:|---:|
| TRACE + ADC (2ch) | 1,942,433 | **0.6826** | 0.968 | **0.959** | 5.49 s |
| TRACE only (1ch) | 1,942,145 | 0.6803 | 0.968 | **0.813** | 5.55 s |
| ADC only (1ch) | 1,942,145 | 0.4449 | 0.931 | 0.970 | 5.55 s |

- **Dice 관점에서 ADC 는 기여가 없다**: 2ch − TRACE-only = **+0.0023**, 시드 잡음(±0.007) 이내.
  A3 가 관측한 "병변부 ADC z 중앙값 −0.19"(대비 약함)와 정확히 일치하는 결과다.
- **부피 ICC 에서는 차이가 크다**: 0.959 vs 0.813. 헌장 기준이 **ICC ≥ 0.85** 이므로 TRACE-only 는 이 기준을 놓친다.
  다만 ICC 는 시드 간 변동이 커서(동일 config 3시드에서 0.855 / 0.925 / 0.959 [측정]) **단일 시드 비교로 확정할 수 없다**.
  → **판정: ADC 채널 유지**(비용 0, Dice 손해 없음, ICC 에 유리할 가능성). ICC 차이의 확정은 **A6 로 이관**(다중 시드 필요).
- ADC 단독은 Dice 0.445 로 크게 떨어지지만 **0 은 아니다** — ADC 만으로도 병변 위치 정보는 있다는 뜻이며,
  ADC 채널이 "죽은 입력"은 아님을 보여준다.

### 7.4 `size` 192 (A3 이관 4) [측정]

192² 캐시(1,233명, clip 10)를 새로 만들어 동일 config·24 epoch 로 비교했다.

| size | 캐시 크기 | val Dice(best) | 부피 ICC | train s/epoch | val s/epoch |
|---:|---:|---:|---:|---:|---:|
| **128** | 2.5 GB | **0.6826** | 0.959 | **5.49** | 1.40 |
| 192 | 5.5 GB | 0.6812 | 0.951 | 10.59 | 2.50 |

**192 는 이득이 없고 epoch 시간이 1.9배**다. A3 가 이미 "128² 에서 빈 마스크 0건, 부피 보존비 p50 0.990,
최소 병변 3복셀"을 확인했으므로 128² 가 병변을 잃지 않는다는 근거도 있다. → **`size: 128` 유지.**

---

## 8. GPU 설정 최적화와 재현성 트레이드오프

### 8.1 시드 잡음 폭(비교의 유의미성 기준) [측정]
동일 config(clip 10, 24 epoch)를 **시드만 바꿔** 3회 실행:

| seed | val Dice(best) | 검출 민감도 | 부피 ICC |
|---:|---:|---:|---:|
| 2026 | 0.6826 | 0.968 | 0.959 |
| 7 | 0.6802 | 0.963 | 0.855 |
| 77 | 0.6761 | 0.959 | 0.925 |

**val Dice 의 시드 간 폭 = 0.0065 (SD ≈ 0.0033)**. 이 보고서에서 **0.007 미만의 Dice 차이는 "차이 없음"** 으로 읽는다.
부피 ICC 는 폭이 0.104 로 훨씬 커서 **단일 시드로 ICC 를 비교하면 안 된다**(A6 유의).

### 8.2 `cudnn.deterministic` / `benchmark` 판단

`seed_everything` 이 `cudnn.deterministic=True, benchmark=False` 를 설정한다. 이를 뒤집으면 [측정]:

| 설정 | slices/s (bs 64, bf16, channels_last) | 이득 | 재현성 |
|---|---:|---:|---|
| deterministic (현행) | 3,695 | — | 비트 단위 재현 ✅ |
| benchmark | 3,774 | **+2.1 %** | 보장 없음 ❌ |

**판정: `cudnn_benchmark: false` 유지.** 헌장이 "seed 고정 재실행 시 val Dice 차이 < 0.02"를 요구하는데,
얻는 것은 2 % 뿐이고 전체 학습이 어차피 5분대이므로 트레이드오프가 성립하지 않는다.
**재현성을 깨는 선택은 하지 않았다.** (필요하면 config 한 줄로 켤 수 있게 키만 열어 두었다.)

AMP(bf16)와 `channels_last` 는 **재현성을 깨지 않는다** — §7.2 표에서 bf16+channels_last 로 두 번 돌린 결과가
비트 단위로 같았다. 확정 캐시에서 bf16 vs fp32 의 12 epoch val Dice 는 **0.664604 vs 0.665876 (차이 0.0013, 잡음 이내)**,
epoch 시간은 5.71 s vs 11.18 s (**1.96배**) 다. (clip=6 캐시에서도 0.5734 vs 0.5679 로 같은 결론이었다.)
fp16 은 bf16 과 속도가 비슷한데 `GradScaler` 가 필요해 코드 경로만 복잡해지므로 **bf16** 을 택했다.

### 8.3 batch_size [측정]

| batch_size | 24 ep Dice | 40 ep Dice | 60 ep Dice | train s/epoch | 배치 구성에 의한 Dice 항 SD |
|---:|---:|---:|---:|---:|---:|
| 32 | — | 0.6925 | 0.6919 | 8.28 | 0.0038 |
| **64** | 0.6826 | 0.6847 | **0.6944** | **5.49** | **0.0032** |

40 epoch 에서는 bs 32 가 +0.008 앞서고 60 epoch 에서는 bs 64 가 +0.003 앞선다 — **방향이 뒤집히므로 잡음**이다
(잡음 폭 ±0.007). epoch 시간이 51 % 더 들고 손실 항의 배치 변동도 더 큰 bs 32 를 택할 이유가 없다 → **bs 64**.

### 8.4 그 밖에 측정했지만 채택하지 않은 것 (모두 24 epoch, clip 10)

| 변경 | val Dice | 기본값 대비 | 판정 |
|---|---:|---:|---|
| 기본 (lr 1e-3, base 16, dice=batch, pos_weight null, neg_pos 1.0) | 0.6826 | — | **채택** |
| lr 3e-3 | 0.6741 | −0.009 | 기각 |
| lr 3e-4 | 0.6697 | −0.013 | 기각 |
| `base_channels: 32` (7.76 M 파라미터) | 0.6830 | +0.000 | 기각(파라미터 4배, epoch 1.7배) |
| `dice_reduction: sample` | 0.6772 | −0.005 | 기각(잡음 이내) |
| `pos_weight: 8.0` | 0.6775 | −0.005 | 기각 |
| `neg_pos_ratio: 2.0` | 0.6865 | +0.004 | 기각(잡음 이내, epoch 1.46배) |

---

## 9. 수정 전/후 요약표 [측정]

| 축 | 수정 전 (인계 config, GPU) | 수정 후 (확정 config) | 배수 |
|---|---:|---:|---|
| train 시간 / epoch | 12.79 s | 5.49 s | **2.33× 빠름** |
| val 시간 / epoch | 1.68 s | 1.32 s | 1.27× |
| step 처리량(연산만) | 1,580 slices/s | 3,695 slices/s | **2.34×** |
| GPU peak 메모리 | 1,475 MiB | 793 MiB | 0.54× |
| val Dice (동일 캐시·12 epoch, fp32 → bf16) | 0.665876 | 0.664604 | ≈ 동일(차이 0.0013) |
| **val Dice (best epoch, 전체 학습)** | 0.5789 (12 ep, clip 6) | **0.6847** (40 ep, clip 10) | **+0.106** |
| 부피 ICC(2,1) (best epoch) | 0.962 (12 ep) | 0.914 (40 ep) | 둘 다 헌장 ≥ 0.85 충족 |
| 검출 민감도 (best epoch) | 0.917 | 0.972 | +0.055 |
| 같은 시드 재실행 val Dice 차이 | 미측정 | **0.000000** | 헌장 < 0.02 충족 |
| 전체 학습 벽시계 | 178 s (12 ep) | 279 s (40 ep) | epoch 3.3배에 시간 1.6배 |

---

## 10. 최종 학습 설정 확정

`configs/seg_unet2d.yaml` 전체를 아래로 확정했다. **모든 값에 위 측정 근거가 있다.**

| 키 | 값 | 근거 (모두 [측정]) |
|---|---|---|
| `seed` | 2026 | 헌장 |
| `size` | 128 | 192 는 Dice −0.001, epoch 시간 1.9배 (§7.4) |
| **`clip`** | **10.0** | clip 6 대비 **val Dice +0.088**, ICC +0.047 (§7.1) |
| `clip_mode` | mad | 백분위 방식과 Dice 차이 없음, 환자 간 스케일이 흔들리지 않음 (§7.1) |
| `base_channels` / `depth` | 16 / 4 | base 32 는 파라미터 4배에 Dice +0.000 (§8.4) |
| **`epochs`** | **40** | §10.1 |
| `batch_size` | 64 | bs 32 대비 epoch 시간 −34 %, Dice 차이 잡음 이내 (§8.3) |
| `lr` / `min_lr` / `warmup_frac` | 1e-3 / 2e-5 / 0.05 | 3e-3 −0.009, 3e-4 −0.013 (§8.4). 스케줄 동작 검증 §3 |
| `weight_decay` | 1e-4 | 변경하지 않음(측정 안 함 — 추가 검증은 A6/A5 몫) |
| `grad_clip` | 1.0 | 초반 20~30 % 발동, 값 1/5/100 차이 없음 → 무비용 보험 (§6) |
| `dice_weight` / `dice_reduction` | 1.0 / batch | `sample` 은 −0.005 (§5.2) |
| `pos_weight` | null | 8.0 은 −0.005 (§5.3) |
| `neg_pos_ratio` | 1.0 | 2.0 은 +0.004(잡음 이내)에 epoch 1.46배 (§5.3) |
| `augment` / `threshold` | true / 0.5 | 변경하지 않음. threshold 재조정은 A6 진단 항목 |
| **`amp`** | **bf16** | fp32 대비 epoch 시간 1.96배 단축, Dice 차이 0.0013(잡음 이내), 비트 재현성 유지 (§8.2) |
| **`channels_last`** | **true** | 속도 이득의 대부분(+83 %) (§2.1) |
| `pin_memory` | true | non-blocking H2D 전송 |
| **`cudnn_benchmark`** | **false** | 이득 +2.1 % 뿐 → **재현성을 깨지 않는다** (§8.2) |
| `num_workers` / `prefetch_factor` | 8 / 4 | nw 0 대비 epoch 7.00 → 5.71 s(**−18 %**), nw 16(6.39 s)보다 빠름. 결과는 nw 0/8/16 비트 동일 (§7.2) |
| `threads` | 4 | 변경하지 않음 |


### 10.1 `epochs` 를 40 으로 정한 근거 [측정]

동일 config(clip 10, bs 64, seed 2026)로 epoch 예산만 바꿔 5번 학습했다. 각 행은 **그 run 안에서 val Dice 가
가장 높았던 epoch** 의 지표다.

| epochs | best epoch | val Dice | 검출 민감도 | 부피 ICC(2,1) | train+val 합계 |
|---:|---:|---:|---:|---:|---:|
| 24 | 22 | 0.6826 | 0.968 | **0.956** | 184 s |
| **40** | **33** | **0.6847** | **0.972** | **0.914** | **273 s** |
| 60 | 45 | 0.6944 | 0.968 | 0.866 | 418 s |
| 100 | 59 | 0.6947 | 0.950 | 0.857 | 691 s |
| 150 | 88 | 0.6944 | 0.954 | 0.828 | 1,052 s |

- **val Dice 는 epoch 60 에서 사실상 포화한다**(60 → 100 → 150 이 0.6944 / 0.6947 / 0.6944 로 제자리).
  train loss 는 계속 내려가므로(150 epoch 마지막 0.0848 vs 40 epoch 마지막 0.1250) 그 뒤는 과적합 구간이다.
- **그런데 부피 ICC 는 epoch 를 늘릴수록 단조 감소한다**(0.956 → 0.914 → 0.866 → 0.857 → 0.828).
  5개 run 에서 방향이 한 번도 뒤집히지 않아 **잡음으로 보기 어렵다**.
- 헌장 기준은 Dice ≥ 0.55 **와** ICC ≥ 0.85 **둘 다**이다. Dice 는 24 epoch 만 해도 크게 넘지만 ICC 는 하한이 가깝다.
  60 epoch 의 ICC 0.866 은 하한까지 여유가 **0.016** 인데 **ICC 의 시드 간 변동 폭이 ±0.05**(§8.1)라 test 에서 하한을
  놓칠 수 있다. 40 epoch 은 Dice 를 0.010(시드 잡음의 약 3σ) 양보하는 대신 ICC 여유를 **0.064** 로 4배 늘린다.
- 따라서 **`epochs: 40`** 을 채택한다. Dice 를 최대화하고 싶다면 60 이 대안이지만, 그 선택은 **ICC 하한 미달 위험을
  받아들이는 결정**이므로 A6 의 성공기준 검토를 거쳐야 한다.

**주의(A6 이관)**: 이 파이프라인은 매 epoch val 로 best 를 고른다. epoch 를 150 으로 늘리면 val 평가가 150번이 되어
**val 에 대한 선택 과적합**이 커진다(위 표에서 150 epoch 의 best 는 ep88 이지만 마지막 epoch 값은 0.6797 로 0.015 낮다).
40 epoch 은 이 선택 횟수도 억제한다.

### 10.2 전체 학습 예상 시간

**[측정]** 확정 config 로 실제 완주한 40 epoch run(`runs/a4_ep40_bs64`, clip 10 캐시):
train 5.49 s/epoch + val 1.31 s/epoch → **train+val 합계 273 s**, 캐시 로딩·체크포인트 저장 포함 **벽시계 279 s**.

**[추정]** 오케스트레이터가 `python scripts/train_seg.py --config configs/seg_unet2d.yaml --run runs/seg_unet2d` 로
전체 학습을 돌릴 때:

| 항목 | 시간 |
|---|---:|
| `SubjectCache` 로딩 (train 1015 + val 218 = 1,233 npz → RAM 2,028 MiB) | **3.4 s** [측정] |
| 학습 40 epoch (5.49 s/epoch) | 220 s [측정] |
| val 평가 40회 (1.31 s/회) | 52 s [측정] |
| 체크포인트 저장 등 | 약 4 s |
| **합계** | **약 4분 40초 (≈ 5분)** [추정, 완주 실측 279 s 와 일치] |

참고로 인계 config(12 epoch, CPU 기준값)를 GPU 에서 그대로 돌리면 178 s 였다. 즉 **epoch 를 3.3배 늘리고도
전체 시간은 1.6배**밖에 늘지 않는다. 런북의 "GPU 는 수 분" 표현은 맞았지만, 그 여유를 epoch 수와 클립 수정에 쓴 것이
이번 A4 의 실질적인 결론이다.

> **주의**: `data/cache` 는 이제 **clip = 10** 캐시다. 이전 clip = 6 캐시로 학습하면 Dice 가 0.09 낮게 나온다.
> `scripts/preprocess.py` 는 config 에서 `clip` 을 읽으므로, config 를 바꿨다면 반드시 캐시를 다시 만들어야 한다
> (`train()` 이 `size` 불일치는 assert 로 잡지만, **`clip` 불일치는 잡지 못한다** — 캐시에 clip 값이 기록되지 않기 때문.
> `data/cache/_manifest.json` 의 `clip` 필드로 육안 확인할 것. → A6 이관 항목).

---

## 11. 산출물

### 11.1 코드 변경
| 파일 | 변경 |
|---|---|
| `configs/seg_unet2d.yaml` | 확정 설정으로 전면 갱신(§10). 새 키: `clip`, `clip_mode`, `amp`, `channels_last`, `pin_memory`, `cudnn_benchmark`, `prefetch_factor`, `dice_reduction` |
| `src/strokeai/train.py` | AMP(bf16/fp16 + GradScaler)·`channels_last`·`pin_memory`+non-blocking 전송·`prefetch_factor`·`channels`(ADC ablation)·`cudnn_benchmark` 스위치, **캐시 해상도 assert**, `amp_context()` 헬퍼, start 로그에 런타임 설정 기록 |
| `src/strokeai/losses.py` | `soft_dice_loss(reduction="batch"|"sample")` + 배치합산의 성질을 측정값으로 문서화 |
| `src/strokeai/data/preprocess.py` | `robust_zscore(clip, clip_mode)` / `preprocess_subject(clip, clip_mode)` — 클립이 하드코딩 6.0 이었음 |
| `src/strokeai/data/dataset.py` | `SliceDataset(channels=[...])` 채널 부분집합, 증강 RNG 를 채널 수에 무관하게 수정(2채널 값은 불변) |
| `scripts/preprocess.py` | `--config` 로 `size`/`clip`/`clip_mode` 기본값을 config 에서 읽음, manifest 에 기록 |
| `scripts/run_all.sh` | 4단계에서 `--size 128` 하드코딩 제거 → `--config configs/seg_unet2d.yaml` |
| `scripts/eval_seg.py`, `scripts/predict_masks.py` | run 의 `config.json` 에 `channels` 가 있으면 그대로 사용(ablation run 평가용) |
| `tests/test_train_options.py` (신규, 7개) | 클립 모드 3종, 채널 부분집합, 1채널 학습 end-to-end, `amp_context` CPU no-op, LR 스케줄 |

### 11.2 분석 스크립트 · 산출물
| 경로 | 내용 |
|---|---|
| `scripts/analysis/a4_sanity.py` | LR 스케줄 검증, 손실 극단값·배치구성 실측, 과적합 sanity → `results/a4/a4_sanity.json` |
| `scripts/analysis/a4_bench.py` | 연산/로더/실epoch 벤치 → `results/a4/a4_bench_*.json` |
| `scripts/analysis/a4_clip_stats.py` | 원본 NIfTI 에서 클립 전 z 분포 재계산 → `results/a4/a4_clip_stats.json` |
| `scripts/analysis/a4_experiments.py` | 실험 하네스(그룹: infra/epochs/clipgrad/clip/adc/size/lr/loss/capacity) → `results/a4/a4_exp_*.json` |
| `runs/a4_*` | stage-2(clip 10) 실험 run 24개. stage-1(clip 6) 은 `runs/a4_stage1_clip6/` 에 보존 |
| `data/cache` | **확정 캐시(clip 10, size 128, 1,233명)** |
| `data/cache_clip06`, `data/cache_clip10`, `data/cache_clip14`, `data/cache_pct01`, `data/cache_192_clip10` | 클립·해상도 비교용 캐시(재현용, gitignore) |

### 11.3 테스트
`pytest` **42개 전부 통과** (기존 35 + A4 신규 7). 실행: `PYTHONPATH=src .venv/bin/python -m pytest -q`.

---

## 12. A6 로 이관하는 항목

1. **`epochs` 40 vs 60 의 Dice–ICC 트레이드오프**(§10.1). Dice 최대화가 목표면 60, ICC 안전마진이 목표면 40.
   성공기준 소유자가 판단할 것. 다중 시드로 ICC 를 재추정하면 결론이 더 단단해진다.
2. **ADC 채널의 ICC 기여**(§7.3): 2ch 0.959 vs TRACE-only 0.813 은 단일 시드 비교이고 ICC 시드 변동이 ±0.05 다.
   ADC 를 유지하기로 했지만, 근거를 굳히려면 각 3시드가 필요하다.
3. **부피 ICC 의 시드 변동이 크다**(0.855~0.959, §8.1). 헌장의 "ICC ≥ 0.85" 를 단일 run 으로 판정하면 위험하다.
   진단 단계에서 ICC 의 부트스트랩 CI 를 함께 보고할 것을 권한다.
4. **`clip` 불일치는 코드가 잡지 못한다**(§10.2 주의). `data/cache/_manifest.json` 의 `clip` 필드를 진단
   스크립트에서 확인하도록 넣으면 좋다.
5. **소병변(<2 mL) 신호 약화**(§5.2): 손실의 Dice 항이 micro-Dice 라 큰 병변에 가중된다(상관 −0.9996).
   헌장이 소병변 Dice 를 별도 보고하도록 요구하므로, 소병변 성능이 미달이면 `dice_reduction: sample` 스위치가 준비돼 있다.
6. **val 선택 과적합**: 매 epoch val 평가로 best 를 고르는 구조. 40 epoch 에서도 best(ep33)와 마지막(ep39)의
   Dice 차이가 0.001 이라 현재는 문제가 크지 않지만, epoch 를 늘리면 커진다(150 epoch 에서 0.015).
7. **런북 §5 의 "grad clip 발동 0 %, p95 0.82" 문장은 폐기 대상**(§6). 실제로는 초반 epoch 에 20~30 % 발동한다.

# A5b 보고서 — 방법론 교차검증(독립) · A5a 17개 지적 판정

작성일: 2026-09-09 · 담당: **A5b methodology-review-internal** · **실행 모델: Fable** (Claude Code, GPU 서버 세션)
대상: `src/strokeai/`, `scripts/`, `configs/seg_unet2d.yaml`, `tests/`, 헌장 v2, 런북, README, A1/Astra/A2/A3/A4/A6 보고서, 발견 로그, `results/`, `runs/seg_unet2d/`
절차: **A5a 보고서를 읽지 않은 상태에서** 코드·산출물을 먼저 독립 검토(§1)했고, 그 다음에 `docs/agents/A5a_report.md` 를 읽어 항목별로 판정했다(§3).
표기: **[측정]** = 이 보고서의 스크립트를 실행해 얻은 값, **[추정]** = 측정값으로부터의 추론.

> **테스트셋 취급**: test 218명의 영상·마스크·예측·지표를 전혀 읽지도 계산하지도 않았다. test ID 는 교집합·캐시 검사에만 썼다.
> 분류 코호트의 test 라벨 보유 수(80명)는 `data/splits_strata.csv`(이미 저장소에 있는 층화표)에서 산술로 나온 것이며, 새로 라벨을 조회하지 않았다.
> 이 보고서의 학습 실험(`runs/a5b_flip_*`)은 전부 train/val 캐시만 사용했다.

---

## 0. 요약 판정

| 질문 | 판정 |
|---|---|
| A5a 17개 지적 | **동의 13 · 부분동의 4 · 반박 0** (§3 판정표). 높음 4건: ① 부분동의 ② 부분동의 ③ 동의 ④ 부분동의 |
| 재학습이 필요한가 | **아니오.** 좌우축 오류는 "명칭 오류"이지 기능 결함이 아니다. 해부학적 좌우 반전으로 재학습하면 val Dice **+0.009**(3 시드 전부 양수)지만 부트스트랩 SE(0.019)의 절반이고 ICC 는 차이가 없다 [측정]. 재학습은 val 선택을 한 번 더 하는 일이며 이득이 그 비용을 정당화하지 못한다 |
| 테스트셋 평가를 진행해도 되는가 | **[조건부 승인]** — 조건 6개는 §6. 핵심: 수정된 코드(특징·지표·분류 CI)로 평가하고, 헌장의 "결과를 보기 전 확정" 류 문구는 이미 "dev 분석 후·test 전 개정"으로 정정했으며, `runs/seg_unet2d/best.pt` 만 test 에 올린다 |

**확정된 문제와 조치(코드)**

| # | 문제 | 근거 | 조치 |
|---|---|---|---|
| B1 | **캐시 좌우축**: 캐시 `(S,H,W)` 에서 **H 가 좌우(RAS x), W 가 전후**인데 `features.py` 는 W 를 좌우로 읽었다 | 합성 NIfTI(좌측 병변) 를 실제 `preprocess_subject` 에 통과 → `laterality_left_frac` **0.5**(기대 1.0); train/val 1,233명 GT 마스크의 **H 저측 분율이 0/1 에 79.7 % 집중** vs W 는 50.4 % (뇌졸중은 편측이므로 좌우축은 양극에 몰려야 한다) [측정, `results/a5b/orientation.json`] | `features.py` H = 좌우로 수정(x = 좌우, y = 전후), 전처리→특징 관통 회귀 테스트 2개 |
| B2 | 증강 "left-right flip" 이 실제로는 **전후 반전** | 위와 동일 축 계약 | `flip_axis` config 키(`lr`/`ap`/`both`/`none`). `ap` 로 재학습 시 `best.pt` **sha256 바이트 동일**(리팩터가 동작 보존) [측정]. §4 실험 뒤 **`ap` 유지** |
| B3 | `detection_sensitivity` 는 "비어 있지 않은 출력률" — val 9/218 이 **Dice 0 인데 TP** | `eval_val.json` 재집계 [측정] | `detection_sensitivity_overlap`(pred∩GT≠∅, val **0.9312**) 병기, 헌장 S2 보고 의무 |
| B4 | 분류 결과에 클래스별·3클래스 CI 없음, pooled/fold 구분 없음, OOF 감사 자료 없음 | `results/cls_*.json` 필드 검사 | `train_cls.py`: 클래스별 CI, 3클래스 부트스트랩, fold 별 AUC, seg-split 별 AUC, `*.oof.json` sidecar, 클래스 결측 재표본 드롭 규칙 |
| B5 | `dice_binary` 가 shape 불일치를 브로드캐스팅으로 허용(예: (1,3) vs (3,1) → 3.0); GT 양성 0명일 때 민감도 0 | 재현 [측정] | shape 검사 `ValueError`, 민감도 NaN |
| B6 | "어떤 스크립트도 test 를 읽지 않았다"가 문자 그대로 거짓 | `index.py` 전 코호트 마스크 복셀 판독(층화 split 의 전제), `a2_volumes.py` split 후 1,451명 재판독 | CLAUDE.md 규칙을 범위 명시형으로 재정의, `a2_volumes.py` 기본 train+val, 전 코호트 마스크 판독 스크립트 allow-list 회귀 테스트 |
| B7 | `preprocess.py` "float16 storage stays exact" 오기, `a6_threshold.py` 확률을 float16 으로 저장 | 1.2345 → 1.234375 | 주석 정정, fp32 로 변경 |

pytest: **46 → 52 passed** [측정].

---

## 1. 독립 검토 (A5a 를 읽기 전) — 지표·손실·split·통계

### 1.1 지표 (`metrics.py`) [측정, `scripts/analysis/a5b_checks.py`]
| 항목 | 검증 | 결과 |
|---|---|---|
| ICC(2,1) | 제곱합을 직접 쓴 독립 구현과 20회 무작위 대조 + val 218명 | 최대 차 **1e-15**, val 0.9138 동일 |
| Dice·부피·LoA·MAPE | `eval_val.json` 218명 재집계 | Dice 0.6855 [0.649, 0.721], LoA [−53.76, +46.94], MAPE 51.1 % 재현 |
| 부트스트랩 | 환자 단위 재표본, seed 고정, ICC 는 쌍이 함께 이동 | 정상. seed 를 결과마다 기록해야 함(§3 항목 9) |
| 검출 | `pred.any()` 기반 | **의미상 결함**(§0 B3) |
| Dice 빈 GT | NaN, 양성만 평균 | 타당 |

### 1.2 손실·학습 루프
- `bce_dice_loss` = BCE + micro-Dice(batch). 수식·경계값 정상. `sample` 은 슬라이스 평균이며 환자 Dice 가 아님 — A4 판정 유지.
- lr 스케줄은 로그와 공식 일치(A6 이 0.0 오차 확인, 내가 재확인하지 않음 — A5a 가 2.7e-19 확인).
- AMP bf16 학습, 손실은 fp32, 평가는 fp32 경로(`eval_seg.py`). best 선택은 val Dice(bf16) 로 이뤄졌고 fp32 재평가 0.6855 — 차이 0.0008, 문제 없음.

### 1.3 split
- `make_split(elig, 2026)` 재실행 → sha256 **일치**(`0f923acd…`), train/val/test 1015/218/218, 교집합 0 [측정].
- 층화 키 = `etiology_code | 부피구간(<2 mL)` → test 라벨·부피 **메타데이터**가 경계에 반영(알려진 한계). 이 메타데이터의 출처가 `index.py` 의 마스크 복셀 판독이라는 점이 **부트스트랩 관계**다: 인덱스 없이는 층화 split 자체를 만들 수 없다(§3 항목 2).

### 1.4 분류 파이프라인 (`train_cls.py`)
- Imputer·Scaler 가 Pipeline 안에서 fold 별 적합 → 누수 없음. `cross_val_predict` 열 순서 = `sorted(set(y))` 로 정확.
- 분류 코호트 447 = seg-train **367** + seg-val **80** [측정]. 예측 마스크는 단일 `best.pt` 산출 → 분류 단계 CV(§3 항목 4).
- 지적 사항: 클래스별 CI·3클래스 CI 부재, pooled OOF 만 보고 → **B4** 로 수정.

### 1.5 데이터 축 계약 (독립 발견 → A5a 3번과 동일)
`load_canonical` 후 `(X=L→R, Y=P→A, Z)`; `preprocess_subject` 가 `H, W, S = tr.shape` 로 놓고 `(S, 2, H, W)` 로 저장 → **H = 좌우**. `features.py` 는 `S, H, W = m.shape; zs, ys, xs = np.nonzero(m); left = xs < W/2` → W(전후)를 좌우로 사용. `dataset.py` 의 `x[:, :, ::-1]` 은 W 반전 = 전후 반전. **[측정]** `scripts/analysis/a5b_orientation.py`:

| 검사 | 값 |
|---|---|
| 합성 좌측 병변 → 캐시에서 병변 H 범위 [3,7]/32(저측), W 범위 [13,18]/32(중앙) | H 저측 = 좌 |
| 기존 `laterality_left_frac` | **0.5** (기대 1.0) |
| 실제 1,233명: H 저측 분율이 0.05 이내로 0 또는 1 인 비율 | **79.7 %** (히스토그램 양끝 504/522) |
| 실제 1,233명: W 저측 분율 동일 기준 | 50.4 % (한쪽 끝 641 — 후방 편중) |
| raw 축 코드 → canonical | `LAS` → `RAS` (5명 표본) |

---

## 2. test 접근 이력의 정확한 기술 (독립 감사)
| 경로 | 읽은 것 | 시점 | 모델·평가 투입 |
|---|---|---|---|
| `scripts/build_index.py` → `index.py` | 전 1,451명 급성 마스크 **복셀**(부피·성분수·고유값), 영상은 헤더만 | split 이전(필수 전제) | 층화 split 경계에만 |
| `scripts/analysis/a2_volumes.py` | 전 1,451명 급성 마스크 복셀(부피) | split **이후**(02:55 KST) | A2 전체 코호트 기술통계. 학습·선택·threshold·특징 세트 선정에 미사용 |
| `data/cache*` 6개, `runs/`, `results/` | — | — | **test 산출물 0건** [측정: `diagnose.py` `test_subjects_in_cache: 0`, 캐시 1,233개] |

→ 정확한 표현: **"부피·라벨 메타데이터에는 노출됐으나 모델 성능 평가에는 쓰인 흔적이 없는 홀드아웃"**. CLAUDE.md·헌장 한계 1 을 이 표현으로 고쳤고, `a2_volumes.py` 는 기본값이 train+val, 회귀 테스트가 split 필터 없는 전 코호트 마스크 판독 스크립트를 막는다(allow-list: `build_index.py`, `download_soop.py`).

---

## 3. A5a 판정표 (17 항목 × 판정 × 근거)

| # | A5a 지적 (심각도) | 판정 | 근거 [측정/수식] | 조치 |
|---|---|---|---|---|
| 1 | v2 는 dev 결과 의존 개정, "결과 보기 전 확정" 은 부정확 (높음) | **부분동의** | 동의: git 순서(학습 06:44 → dev 결과 커밋 `638a397` 06:47 → 헌장 v2 07:08)와 A6 의 검정력 계산이 dev 0.655 를 참값으로 놓은 것은 사실. 반박(범위): 헌장·A6 어디에도 "결과를 보기 전"이라는 문구는 없고 "test 미열람 상태에서 확정"이라고만 썼으며 이는 참이다. test 산출물은 0건. 따라서 부정행위가 아니라 **표기의 정밀도** 문제 | 헌장 머리말을 "dev 분석 후·test 전 개정"으로 명시, v1 판정(`no_size` macro > 0.60)을 test 에서 병기하도록 추가 |
| 2 | "어떤 스크립트도 test 를 읽지 않았다"는 거짓; 캐시 0명은 미열람 증명 아님 (높음) | **부분동의** | 동의: §2 표. 반박(심각도): `index.py` 의 판독은 층화 split 의 **전제 조건**이라 회피 불가(부트스트랩)이고, A2 재판독은 부피 1개 스칼라의 기술통계로 어떤 선택에도 들어가지 않았다. 누수의 정의(학습·선택·평가로의 정보 유입)에서는 **0** | CLAUDE.md 규칙 재정의(허용 범위 명시), `a2_volumes.py` 기본 dev, allow-list 회귀 테스트, 헌장 한계 1 정정 |
| 3 | RAS 좌우축이 특징·증강에서 뒤바뀜 (높음) | **동의** | §1.5 [측정]. Dice·부피는 영향 없음(영상·마스크 동시 변환) — 확인. 분류 영향: 수정 후 예측 `full` LAA-vs-CE **0.6551 → 0.6709**, GT 0.6409 → 0.6693, macro 불변(0.7167 → 0.7173) [측정] | `features.py` 수정, `flip_axis` config, 관통 테스트, §4 재학습 실험 |
| 4 | 예측 마스크 dev CV 는 전체 파이프라인 OOF 가 아님 (높음) | **부분동의** | 동의(구조): 367/80 [측정]. seg-train 부분의 예측 마스크는 더 깨끗하다: Dice **0.758 vs 0.706**, log 부피 MAE **0.20 vs 0.33**, ρ 0.957 vs 0.891 [측정]. 반박(영향): pooled OOF AUC 를 두 부분으로 나누면 `full` macro **0.717(seg-train) vs 0.720(seg-val)**, LAA-vs-CE 0.634 vs 0.822 — 낙관 방향의 차이가 없고 GT 마스크 대조군도 같은 패턴(0.731 vs 0.718) → **AUC 로의 편향은 검출되지 않음**. test 는 정직한 예측을 쓰므로 test 판정에 무관(분류기가 더 깨끗한 특징으로 학습된 것은 보수적으로 작용) | `cv_by_seg_split` 을 표준 출력에 추가, 헌장 한계 6. k-fold 분할 재학습(OOF 마스크)은 이득 근거가 없어 **철회** |
| 5 | E1 검정력 0.93 은 PASS 규칙의 검정력이 아님; "두 오류 동시 최소" 불성립 (중간) | **동의** | 실제 PASS 규칙(점>0.55 ∧ 백분위 부트스트랩 CI 하한>0.5)을 시뮬레이션(binormal, n 35/22, 300회 × 2,000 재표본): 참값 0.655 → **PASS 0.54 / 판정불가 0.40 / FAIL 0.06**; 참값 0.5 → PASS 0.017; 0.60 → 0.26; 0.70 → 0.76 [측정, `scripts/analysis/a5b_e1_power.py`]. 정규근사 0.568 과 일치. 오류 합 최소는 0.55 가 아님(수식: A5a) | 헌장 E1 근거 정정 + 종합 판정 규칙 추가 |
| 6 | 3클래스·클래스별 CI 부재 (중간) | **동의** | JSON 필드 검사 [측정] | B4. 재실행 결과 3클래스 `full` **0.6750 [0.622, 0.722]**, `volume_only` 0.5125 [0.466, 0.558] |
| 7 | 선택 편향 0.0037 은 편향 추정치가 아님 (중간) | **부분동의** | 동의: best−last5 는 같은 val 위의 관측 차이. 반박(정도): 40회 평가는 인접 epoch 간 상관이 극히 높아(마지막 5 epoch SD 0.0025) 최대값 편향의 크기는 그 SD 규모로 **[추정]** 작다. A5a 의 1,270회 epoch 평가는 hyper-parameter 탐색 run 이며 최종 체크포인트 선택은 1 run 40회 | 헌장 §C 에 "선택 편향 미측정" 은 A6 문구 대신 이 표로 기록. 별도 보정 없음 |
| 8 | pooled OOF vs fold 평균 (중간) | **동의** | fold 평균 재계산: `full` pred macro pooled 0.7173 / fold 평균 0.7192, `volume_only` 0.6421 / 0.6496 [측정] — 방향은 pooled 가 낮음 | `fold_macro_auc_ovr`·`fold_mean_*` 출력, 명칭 "pooled OOF" |
| 9 | CI 보장 범위·클래스 결측 재표본 (중간) | **동의** | seed 별 ICC CI 하한 0.790~0.798 [A5a 측정, 재실행 안 함]. 클래스 결측: dev n=447 에서 드롭 **0회** [측정]; test n≈80 에서도 발생 확률 ≈ (1−10/80)^80 ≈ 2e-5 [수식] | 재표본 규칙 확정: 클래스가 빠진 재표본은 드롭하고 `n_boot_effective` 기록, seed 기록 |
| 10 | `scale_invariant` 정의·증거 불일치 (중간) | **동의** | age 는 "년" 단위. 예측 마스크 `centroid_z` ρ = **0.2636** (GT 0.1722) [측정] | 헌장 E5 문구 정정, 세트는 고정(사후 변경 금지) |
| 11 | "학습 없이", "완전 소멸", "GT 상한" 과장 (중간) | **동의** | `volume_only` 는 1-특징 로지스틱; 3클래스 `volume_only` LAA-vs-CE 0.596 잔여; `full` LAA-vs-CE 는 예측(0.6709) > GT(0.6693) | 헌장 문구 정정 |
| 12 | ADC 대비 → 라벨 오류 인과 (중간) | **동의** | A6 §6 의 수치는 재현되나 인과 근거는 없음. 나는 원본 판독을 하지 않았다 | 발견 로그에 기록; 헌장에는 인과 문구 없음(변경 불필요) |
| 13 | 검출 = 비어 있지 않은 출력률; README 낡음; PPV 문구 (중간) | **동의** | Dice 0 & pred_pos **9/218** [측정] | B3, 헌장 S2·한계 4, README 정정 |
| 14 | micro-Dice 의존성 (낮음) | **동의** | 수식 동일. 조치 불필요(A4 결정 유지) | — |
| 15 | float16 "exact" 오기, a6 스윕 fp16, 재현성 범위 (낮음) | **동의** | 1.2345 → 1.234375 | B7 |
| 16 | `dice_binary` 브로드캐스팅, 민감도 0, LoA 해석 (낮음) | **동의** | 재현 | B5 + 테스트 |
| 17 | 플래그 8명은 중복 합산, 합집합 5 (낮음) | **동의** | eq 3 ⊂ subset 5, 합집합 **5** [측정] | 발견 로그 기록 |

---

## 4. 재학습 판정 — `flip_axis` 비교 실험 [측정, `scripts/analysis/a5b_flip_experiment.py`]
동일 config(40 epoch, bf16, clip 10, seed 만 변경), train/val 캐시만 사용.

| run | flip | seed | best ep | val Dice | 마지막5 평균 | 민감도 | ICC [CI] | <2 mL Dice/검출 |
|---|---|---:|---:|---:|---:|---:|---|---|
| `a5b_flip_ap_s2026` (= 현 best.pt, sha 동일) | ap | 2026 | 33 | **0.6847** | 0.6811 | 0.9725 | **0.9137** [0.793, 0.990] | 0.538 / 0.950 |
| `a5b_flip_lr_s2026` | lr | 2026 | 39 | **0.6936** | 0.6907 | 0.9725 | 0.8500 [0.598, 0.991] | 0.558 / 0.933 |
| `a5b_flip_none_s2026` | none | 2026 | 30 | 0.6930 | 0.6912 | 0.9817 | 0.8239 [0.523, 0.991] | 0.562 / 0.950 |
| `a5b_flip_ap_s7` | ap | 7 | 29 | 0.6779 | 0.6722 | 0.9587 | 0.8483 | 0.538 / 0.900 |
| `a5b_flip_lr_s7` | lr | 7 | 29 | 0.6891 | 0.6866 | 0.9633 | 0.8417 | 0.557 / 0.917 |
| `a5b_flip_ap_s77` | ap | 77 | 30 | 0.6825 | 0.6810 | 0.9587 | 0.8390 | 0.566 / 0.917 |
| `a5b_flip_lr_s77` | lr | 77 | 33 | 0.6901 | 0.6868 | 0.9633 | 0.8853 | 0.545 / 0.900 |

- Dice: `lr` **0.6909 ± 0.0024** vs `ap` **0.6817 ± 0.0035** (3 시드). 시드별 짝 차이 **+0.0089 / +0.0112 / +0.0076** — 방향 일관. 크기는 부트스트랩 SE 0.019 의 **0.5배**, S1b 폭(0.06)의 1/6.
- ICC: `lr` 0.859 ± 0.023 vs `ap` 0.867 ± 0.041 → 차이 없음. **현 best.pt 의 ICC 0.914 는 7 run 중 최대값**(범위 0.824~0.914) → S3 는 test 에서 놓칠 가능성이 실질적이며, 이는 재학습으로 고칠 수 없는 시드 변동이다(헌장 §C 에 기록).
- `none` 이 `lr` 과 같다 → 반전 증강의 기여 자체가 작고, **전후 반전은 오히려 −0.009** [추정: 시드 1개].

**판정: 재학습 불필요.** 이유 ① 기능 결함이 아닌 명칭 오류(영상·마스크 동시 반전이라 라벨 불일치 없음) ② 이득이 잡음 이내 ③ 재학습 후 채택은 val 로 모델을 한 번 더 고르는 것이며 A6 가 40 epoch 을 ICC 여유 때문에 택한 논리와 충돌(lr s2026 의 val ICC 0.850 은 S3 임계값 위에 간신히 걸침) ④ 헌장 §C 가 체크포인트 sha 를 고정했고 그 run 은 바이트 재현된다.
**대안(오케스트레이터 선택지)**: 그래도 좌우 반전 모델을 원하면 `flip_axis: lr`, seed 2026, 40 epoch 으로 `runs/seg_unet2d_lr` 을 만들고 **그것 하나만** test 에 올린다(두 모델을 다 올리면 두 번째 열람). 그 경우 dev 분류 결과·헌장 §C·A6 §9 예상치를 그 run 으로 다시 산출해야 한다(약 15분). 나는 권장하지 않는다.

**헤드룸 제안과 철회**
- (유지) 새 학습에는 `flip_axis: lr` 이 맞다(+0.009, 3 시드 일관). 이 프로젝트 범위에서는 적용하지 않는다.
- (철회) "좌우 반전으로 ICC 도 개선"은 근거 없음(측정 차이 없음).
- (철회) k-fold 분할 재학습으로 OOF 예측 마스크를 만드는 안: seg-train/seg-val 부분 AUC 차이가 없어 이득 근거가 없다.

---

## 5. 분류 결과 재계산 (좌우축 수정 후, dev 5-fold pooled OOF, seed 2026) [측정]
| 세트 | 예측 macro [CI] | 예측 LAA-vs-CE [CI] | GT macro | GT LAA-vs-CE | 3클래스 예측 [CI] |
|---|---|---|---:|---:|---|
| `volume_only` | 0.6421 [0.615, 0.670] | 0.5869 | 0.6421 | 0.5830 | 0.5125 [0.466, 0.558] |
| `full` | **0.7173 [0.683, 0.752]** | **0.6709 [0.611, 0.730]** | 0.7291 | 0.6693 | **0.6750 [0.622, 0.722]** |
| `no_size` | 0.7096 | 0.6677 | 0.7320 | 0.6751 | 0.6873 |
| `scale_invariant` | 0.6522 | 0.6434 | 0.6536 | 0.6193 | 0.6641 |
| Δ `full`−`volume_only` | **+0.0752 [+0.039, +0.110]** | +0.084 [+0.021, +0.149] | +0.087 | +0.086 | — |

좌우축 수정으로 바뀐 것은 `laterality_left_frac`·`bilateral` 을 포함한 `full`·`no_size` 뿐이며(다른 세트는 소수점 4자리까지 동일), 변화는 LAA-vs-CE 축에 집중(+0.016 예측, +0.028 GT). 이는 정확한 편측성 정보가 LAA/CE 판별에 조금 더 유용하다는 신호이지만 CI 폭(±0.06) 안이다.

---

## 6. 테스트셋 평가 판정 — **[조건부 승인]**
승인 조건(모두 충족 시 A6 §9.2 의 명령을 **그대로** 1회 실행):
1. 코드 상태 = 이 보고서 시점(§7 파일)·`pytest` **52 passed**. `features.py`·`metrics.py`·`train_cls.py` 수정본으로 평가한다(예측 마스크 특징은 자동으로 수정본 사용).
2. 체크포인트는 `runs/seg_unet2d/best.pt`(sha `e10d9268…`) **하나만**. `runs/a5b_flip_*` 는 test 에 올리지 않는다.
3. 헌장 §C 대로 threshold 0.5·min-voxels 0·TTA 없음·fp32. `configs/seg_unet2d.yaml` 의 `flip_axis: ap` 는 학습 당시 동작 기록이며 평가에는 영향이 없다.
4. 보고서에 S2 의 두 검출률(비어 있지 않은 출력률 / 병변 접촉률)과 모든 AUC 의 CI·fold 값·v1 판정(`no_size` macro > 0.60)을 함께 싣는다. E1 은 PASS/판정불가/FAIL 을 §3 항목 5 의 실제 검정력(0.54)과 함께 해석한다.
5. test 열람 이력은 §2 표현("메타데이터 노출, 평가 미사용")으로 쓴다. "한 번도 읽지 않았다"는 문장은 어느 문서에도 남기지 않는다.
6. test 결과를 본 뒤 재학습·재선택·기준 수정 금지(A6 §9.4 그대로). S3 가 실패하면 §4 의 시드 변동(0.82~0.91)을 근거로 "실패"로 보고한다.

A6(승인)와 A5a(보류)가 갈린 지점에 대한 내 입장: A5a 의 지적은 대부분 **문구·보고 범위·검정력 해석**의 문제이고 그것들은 이 보고서에서 고쳤다. 실질적 결함(좌우축)은 Dice·부피에 영향이 없고 분류 특징은 수정했으며, 재학습 이득은 잡음 이내로 측정됐다. 따라서 보류를 유지할 근거는 남아 있지 않다.

---

## 7. 수정·추가한 파일
| 파일 | 변경 |
|---|---|
| `src/strokeai/features.py` | H = 좌우 축 계약(x = 좌우, y = 전후), 모듈 docstring |
| `src/strokeai/data/dataset.py` | 축 계약 문서화, `LR_AXIS`/`AP_AXIS`, `flip_axis` (`lr`/`ap`/`both`/`none`) |
| `src/strokeai/train.py` | `flip_axis` 를 config 에서 읽어 로그·데이터셋에 전달, `predict_subject` TTA 축 동일 규약, `overlap_pos` 산출 |
| `src/strokeai/metrics.py` | `dice_binary` shape 검사, 민감도 NaN, `detection_sensitivity_overlap`·`n_pred_nonempty_but_no_overlap` |
| `src/strokeai/data/preprocess.py` | float16 주석 정정 |
| `scripts/train_cls.py` | 클래스별 CI·3클래스 부트스트랩·fold AUC·seg-split AUC·OOF sidecar·클래스 결측 드롭 규칙·docstring |
| `scripts/analysis/a2_volumes.py` | 기본 train+val, `--include-test` |
| `scripts/analysis/a6_threshold.py` | 확률 fp32 |
| `scripts/analysis/a5b_orientation.py` / `a5b_flip_experiment.py` / `a5b_e1_power.py` / `a5b_checks.py` | **신규** 재현 스크립트 → `results/a5b/{orientation,flip_experiment,e1_power,checks}.json` |
| `configs/seg_unet2d.yaml` | `flip_axis: ap` (+주석) |
| `tests/test_preprocess_features.py` | 좌우 테스트 정정 + 전처리 관통 테스트(RAS·LAS) |
| `tests/test_dataset_rng.py` | 축별 동시 반전 테스트, `flip_axis` 축 검증 테스트 |
| `tests/test_metrics.py` | shape 검사·민감도 NaN·접촉률 테스트 |
| `tests/test_no_test_leakage.py` | `a2_volumes.py` 기본값, 전 코호트 마스크 판독 allow-list |
| `CLAUDE.md`, `docs/01_server_runbook.md` | test 열람 규칙 재정의 |
| `docs/00_project_charter.md` | v2 머리말(dev 후 개정), S2, E1(검정력)·E2·E3·E4·E5, v1→v2 이력 문구, 한계 1·4·5·6, §C(체크포인트 판정·증강 축·특징 코드·S3 기대) |
| `README.md` | 검출·특이도·분류 문구, 테스트 수, A5b 상태 |
| `docs/findings_log.md` | A5b 행 |
| `runs/seg_unet2d/eval_val.json`, `results/diag_seg_unet2d/` | fp32 val 재평가(수치 동일 + 접촉률 필드) |
| `results/cls_gt_logreg.json`, `results/cls_pred_logreg.json` (+`.oof.json`) | 좌우축 수정 후 재계산 |

재현: `PYTHONPATH=src .venv/bin/python -m pytest -q` → **52 passed**;
`scripts/analysis/a5b_orientation.py --tmp <scratch>`, `a5b_checks.py`, `a5b_e1_power.py --n-sim 300`,
`CUDA_VISIBLE_DEVICES=0 scripts/analysis/a5b_flip_experiment.py`(7 run ≈ 35분).

## 8. A5b 자신의 한계
- flip 비교는 시드 3개다. Dice 차이의 방향은 일관되나 크기의 CI 는 넓다(시드 SD 0.003 기준 ±0.005).
- 선택 편향(항목 7)은 측정하지 않았다 — 환자별 epoch 예측이 저장돼 있지 않다.
- ADC 대비 기반 라벨 품질(항목 12)은 원본 판독으로 검증하지 않았다.
- E1 검정력 시뮬레이션은 binormal 점수 모델·추정 표본수(35/22) 가정이다.

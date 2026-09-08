# A3 보고서 — 전처리·재현성·기술통계

작성일: 2026-09-09 · 담당: **A3 preprocessing-stats** · 실행 모델: **Opus** (Claude Code, GPU 서버 세션)
데이터 스냅샷: `data/splits.json` 내부 `sha256` = **0f923acd2f8898d58c4c62cb598c4b8d88c17ab5eaab8a72ee4e73a7342d8f43** (헌장 값과 일치)
· `data/splits.json` 파일 sha256 = `b5b707044af638754fa3b8bc0c6e51d69170c9f1901217f624bd758e1bbd81c9`
· `data/index.csv` 파일 sha256 = `fb0e8c10558d7a74cd573d8f4e77e68219bde140d1270b21f02205ab2d8a4bd2` (A2 값과 동일)
· `data/splits_strata.csv` sha256 = `ca2babfd8c2dc831b094c080352c4200b13b72c9cc91394aa58fbd21f270920c`
환경: Python 3.10.12 · numpy 2.2.6 · scipy 1.15.3 · nibabel 5.4.2 · pandas 2.3.3 · torch 2.5.1+cu121 · RTX A6000 ×2 / 48 core
표기 규칙: **[측정]** = 이 보고서의 명령을 직접 실행해 얻은 값, **[추정]** = 측정값으로부터의 추론

> **테스트셋 취급**: 이 보고서의 모든 수치는 `train`(1015) + `val`(218) = **1,233명**에서만 계산했다.
> `test` **218명의 영상·마스크는 한 번도 읽지 않았다**(전처리·통계·플래그 모두 없음). test ID 는 교집합 검사에만 사용했다.
> 내가 추가한 분석 스크립트(`scripts/analysis/a3_*.py`)도 `--which train val` 이 기본값이며 test 는 명시적으로 지정해야만 읽는다.

---

## 0. 요약 판정

**판정: [재현성은 확인됨 — 단 전처리에 실제 데이터 손상 버그 2건이 있었고, 둘 다 수정했다. 학습 전에 캐시 45명분 재생성이 필요하다.]**

| # | 발견 | 심각도 | 상태 |
|---|---|---|---|
| **F1** | `robust_zscore` 가 **MAD = 0 을 방어하지 않음**. SOOP ADC 맵은 TRACE 전경의 절반 이상이 단일 상수 채움값인 경우가 있어 MAD 가 정확히 0 → `1e-6` 로 나눠 **ADC 채널 전체가 ±6 포화**. train+val **44/1233명(3.6 %)**, 최악 sub-683 은 전체 복셀의 **64.4 %** 가 클립값 | **높음** | **수정함** |
| **F2** | `preprocess_subject` 가 마스크↔TRACE affine 은 검사하지만 **ADC↔TRACE affine 은 검사하지 않음**. sub-235 의 ADC 격자는 TRACE 대비 **영상 중심 6.6 mm, 모서리 15.5~27.3 mm 어긋남** → 두 채널이 다른 해부 위치로 학습에 들어감 | **중간** | **수정함** (TRACE 격자로 리샘플) |
| **F3** | 정규화 클립 `±6` 이 **너무 좁다**: 병변 복셀의 평균 **34.6 %** 가 TRACE 채널에서 +6 으로 포화, **397/1233명은 병변의 50 % 이상이 포화**. 병변 내부 명암 정보가 소실 | **중간** | **미수정**(전체 캐시 무효화 → A4 실험 항목으로 이관) |
| **F4** | `SliceDataset` 이 증강에 인스턴스 RNG 하나를 씀 → `num_workers>0` 이면 RNG 가 워커로 **fork** 되어 워커마다 같은 증강열을 재생하고 결과가 워커 수에 의존(재현 불가). 현재 config 는 `num_workers: 0` 이라 잠복 상태 | **중간** | **수정함** (항목별 결정론적 RNG) |
| **F5** | split 이 **재현됨**: seed 2026 로 3회 재실행 → `splits.json`·`splits_strata.csv` **바이트 단위 동일**, sha256 헌장 값과 일치 | — | 확인 |
| **F6** | 전처리도 **재현됨**: 1,233명 전량 재실행 결과 **1,188명이 바이트 단위 동일**(`np.savez` 는 결정론적). 달라진 45명은 전부 F1·F2 수정 효과이며 **마스크 배열은 1,233명 전원 불변** | — | 확인 |
| **F7** | 정규화는 **구조적으로 환자별 자기 통계만** 사용(`robust_zscore(x, fg)` 의 입력은 해당 환자 볼륨뿐). 코호트 평균/표준편차를 쓰는 경로 없음 | — | 확인 |
| **F8** | 마스크는 리샘플 후에도 **1,233명 전원 이진 {0,1}**(order=0 최근접), 좌표계도 일치. 정량 근거: 정규화 TRACE 평균이 병변 내부 4.85 vs **좌우반전 병변 2.23** vs 전체 −0.37, **1,227/1,233명(99.5 %)에서 병변 > 반전 병변** | — | 확인 |
| **F9** | 층화가 `test` 병변 크기를 사용한다(`stratum()` 이 `mask_acute_ml` 참조). 인덱스 생성 시점에 이미 계산된 메타데이터라 영상 재열람은 없지만, **엄밀히는 test 라벨 정보가 split 경계에 반영**된 상태 | 낮음(정보) | 기록만 |
| **F10** | `data/index.csv` 에 **`adc_affine_matches_trace` 열이 없음** → 적격성 필터가 ADC 기하 불일치를 잡지 못함. 지금 고치면 sub-235 가 탈락해 **splits sha256 이 바뀌므로 의도적으로 고치지 않았다** | 낮음 | 미수정(근거 기재) |
| **F11** | 결합 마스크 `*_desc-lesion_mask.nii.gz` 는 **어떤 코드도 복셀을 읽지 않음**(존재 여부 플래그 `has_mask_any` 만). A2 의 "사용 금지" 준수 확인 | — | 확인 |
| **F12** | `acuteischaemicstroke` 는 분류 특징에 **들어 있지 않음**(`train_cls.py` 의 `CLIN = [age, sex_male, nihss]`). A2 권고 이미 충족 | — | 확인 |
| **F13** | 128² 리샘플 안전성 재확인: 빈 마스크 **0건**, 부피 보존비 중앙값 **0.990**(p1 0.836 / p99 1.132), 최소 병변 **3복셀** | 낮음 | 확인 |
| **F14** | 급성 = 만성 마스크 완전 동일 **3명**, 급성 ⊂ 만성 **5명** (train+val 중 만성 마스크 보유 168명 기준). 플래그 파일 생성 | 중간 | 이관(A4/A6) |
| **F15** | `split.py` 의 `fillna().astype(bool)` 이 pandas FutureWarning 발생(object dtype 무음 다운캐스트). 리팩터 후 **split 해시 불변** 확인 | 낮음 | 수정함 |

---

## 1. 재현성 검증

### 1.1 split (같은 seed → 같은 결과)

```bash
cd /home/user1/Desktop/Multi-Agent/JLK
PYTHONPATH=src .venv/bin/python scripts/make_splits.py --out /tmp/a3/rerun1/splits.json
PYTHONPATH=src .venv/bin/python scripts/make_splits.py --out /tmp/a3/rerun2/splits.json
sha256sum data/splits.json /tmp/a3/rerun*/splits.json
```

| 파일 | sha256 |
|---|---|
| `data/splits.json` (기존) | `b5b707044af638754fa3b8bc0c6e51d69170c9f1901217f624bd758e1bbd81c9` |
| 재실행 1 | 동일 |
| 재실행 2 | 동일 |
| 재실행 3 (F15 리팩터 후) | 동일 |
| 내부 `sha256` 필드 | `0f923acd…3a8434` = 헌장 값 |

**[측정]** train 1015 / val 218 / test 218, 합 1451 = `n_eligible`. 교집합은 3쌍 모두 **공집합**, 중복 ID 0건,
`sha256_of({train,val,test})` 재계산값이 파일에 적힌 값과 일치. → `tests/test_no_test_leakage.py` 에 회귀 테스트로 고정.

**환자 단위 확인 [측정]**: `splits.json` 의 원소는 전부 `sub-<n>` 형태의 **환자 ID**이며, 슬라이스 단위 항목이 없다.
`SliceDataset` 은 `SubjectCache(cache_dir, splits[w])` 로 만들어진 환자 집합 안에서만 슬라이스를 펼치므로,
같은 환자의 슬라이스가 두 split 에 걸칠 구조적 경로가 없다(`tests/test_dataset_rng.py::test_slice_table_is_patient_level`).

### 1.2 전처리 산출물 (.npz) 재실행 해시 비교

먼저 **표본 40명**(seed 20260909 무작위 추출)으로 2회 재실행:

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess.py \
  --splits /tmp/a3/sample_splits.json --out /tmp/a3/cache_repro  --workers 20 --which train
PYTHONPATH=src .venv/bin/python scripts/preprocess.py \
  --splits /tmp/a3/sample_splits.json --out /tmp/a3/cache_repro2 --workers 8  --which train
```

**[측정]** 40/40 이 **배열 단위·바이트 단위 모두 동일**. 워커 수(20 vs 8)를 바꿔도 동일 → `np.savez` 는 결정론적이고
전처리에 난수가 없다. 표본 40명 결합 배열 해시 = `b76ba25a445fddbdd9e2d5308b60365cb0b8bb86a23bed53d918633c447567b1`.

전처리가 1,233명 **13초**(44 워커)로 끝나므로 표본에 그치지 않고 **전량 재실행**해 비교했다:

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess.py --out /tmp/a3/cache_full --size 128 --workers 44 --which train val
```

| 비교 | 결과 |
|---|---|
| 배열 단위 동일 | **1,188 / 1,233** |
| 달라진 환자 | **45명** = F1(44명) ∪ F2(sub-235) — 전부 이번 수정의 의도된 효과 |
| **마스크 배열** 변화 | **0명** (라벨은 전혀 건드리지 않음) |
| 바이트 단위 동일(수정 무관 1,188명) | 1,188 / 1,188 |
| 현재 `data/cache` 결합 해시 | `b126df15b957dcd479503791314a18868394affbef7613192f463ea75fdce901` |
| 수정 후 캐시 결합 해시 | `33f1cd3f112ba823861e86b41dc032a20b1cf0c8a2d27d6152b8dfd399c28c03` |

> 결합 해시 = 환자 ID 오름차순으로 `sha256(sid ‖ sha256(img,mask,voxel_volume_mm3,orig_shape,zooms,scale))` 을 누적한 값.

**규칙 준수**: 기존 `data/cache/` 는 **삭제·덮어쓰지 않았다**. 재생성 대상 45명 목록은
`results/a3/stale_cache_subjects.txt` 에 있고, 아래 한 줄이면 그 45개만 다시 만들어진다(약 2초).

```bash
cd /home/user1/Desktop/Multi-Agent/JLK
while read sid; do rm -f "data/cache/$sid.npz"; done < results/a3/stale_cache_subjects.txt
PYTHONPATH=src .venv/bin/python scripts/preprocess.py --size 128 --workers 44 --which train val
```

### 1.3 정규화 통계가 test 를 보지 않는가 (코드 구조 검증)

**[측정 — 코드 검토]** `src/strokeai/data/preprocess.py::robust_zscore(x, fg)` 의 입력은 **그 환자의 볼륨 하나**뿐이다.
`preprocess_subject` 는 파일 3개(TRACE/ADC/급성 마스크)만 인자로 받고 다른 환자·다른 split 을 참조할 방법이 없다.
학습 경로(`src/strokeai/train.py`)에도 데이터셋 수준 평균/표준편차, 채널 통계, 배치 정규화 통계의 사전 계산이 없다.
→ **설계상(by construction) 누수 불가.** 저장소 전체에서 `sp["test"]` / `splits["test"]` 를 참조하는 곳은
`make_splits.py`(생성), `diagnose.py`(개수만), `eval_seg.py --split test`, `train_cls.py --final`, `run_all.sh` 의 **echo 문**뿐이다.

---

## 2. 오케스트레이터의 `preprocess.py --which` 수정 판정

### 판정: **[타당]** — 다만 그것만으로는 충분하지 않았다(아래 2건을 추가로 막았다).

**근거 [측정]**

1. 수정 전 코드는 `sp["train"] + sp["val"] + sp["test"]` 를 전처리해 **파이프라인 4단계에서 홀드아웃 218명의 영상·마스크를 실제로 디스크에서 읽었다**. 이는 CLAUDE.md "테스트셋은 최종 1회 평가 전까지 어떤 스크립트도 읽지 않는다" 를 직접 위반한다. `--which` 기본값을 `["train","val"]` 로 두고 `run_all.sh` 4단계를 `--which train val` 로 바꾼 것은 **최소 변경으로 위반을 제거하는 올바른 수정**이다.
2. 현재 상태가 그 수정이 실효적임을 증명한다: `data/cache/` 에 `sub-*.npz` 가 **정확히 1,233개**, test 218명 중 캐시된 환자 **0명** **[측정]**.
3. 아래 세 경로에도 test 조기 열람이 남아 있지 않음을 확인했다 **[측정]**:
   - `predict_masks.py --which` 기본값 = `["train","val"]`
   - `eval_seg.py --split` 기본값 = `"val"`
   - `train_cls.py` 는 `--final` 없이는 `sp["test"]` 를 만지지 않고, `--final` 도 dev(train+val) 로 학습 후 test 를 **1회** 채점
   - `run_all.sh` 안에서 `--which test` / `--split test` 가 등장하는 줄은 전부 `echo` (실행되지 않음)
4. **불충분했던 부분**: `--which` 는 "언제 읽느냐" 만 고쳤을 뿐, "무엇을 만드느냐"의 결함(F1·F2)은 남아 있었다. 또한 수정 자체에 회귀 방지 장치가 없어 다음 사람이 `run_all.sh` 를 되돌리면 조용히 재발한다 → `tests/test_no_test_leakage.py` 5개 테스트로 **CLI 기본값과 run_all.sh 내용을 고정**했다.
5. **부작용 없음**: `--which` 는 하이퍼파라미터가 아니고 splits/index 를 건드리지 않는다. split sha256 불변 확인.

---

## 3. 발견한 버그 상세

### F1 [높음] MAD = 0 → ADC 채널 포화

**증상 [측정]** train+val 1,233명 중 **44명**에서 TRACE 전경 위 ADC 값의 MAD 가 **정확히 0**.
원인은 ADC 맵이 뇌 외곽(두개골·목·마스킹 영역)에 **단일 상수 채움값**을 갖고, TRACE 기준 전경이 그 영역을 넉넉히 포함하기 때문이다.
전경 복셀 중 최빈값 비율의 코호트 분포는 중앙값 0.237 / p95 0.491 이고, **0.5 를 넘는 44명이 정확히 MAD = 0 인 집합과 일치**한다.

예: `sub-683` — 전경 ADC 의 51.1 % 가 값 `-4.936`, 따라서 중앙값 = 채움값, MAD = 0 → `(x-med)/1e-6` →
**캐시 ADC 채널의 64.4 % 가 |z| ≥ 6**. `sub-703` 63.1 %, `sub-955` 22.6 %, …

**수정** `robust_stats()` 신설: MAD > 0 이면 **기존 식 그대로**(비트 단위 동일 보장), MAD = 0 이면 채움값을 제외하고 재계산 →
그래도 0 이면 IQR/1.349 → std → 1.0 순으로 폴백.

**효과 [측정]** 44명만 변경(다른 1,189명 바이트 동일).
전체 볼륨 ADC 클립 비율: 평균 0.86 % → **0.164 %**, 최대 64.4 % → **18.4 %**.
표본 확인: `sub-51` |z|≥6 비율 20.4 % → **0.00 %**, `sub-571` 18.3 % → **0.00 %**, TRACE 채널과 마스크는 비트 단위 불변.

### F2 [중간] ADC↔TRACE 기하 미검증 (sub-235)

**증상 [측정]** `preprocess_subject` 는 `assert np.allclose(aff_t, aff_m)` 로 마스크만 검사했다.
train+val 1,233명의 헤더를 전수 감사한 결과:

| 환자 | raw affine 불일치 | canonical 후 불일치 | 격자 변위 |
|---|---|---|---|
| `sub-1233` | 예 (최대 237.2 mm) | **아니오** | 코너·중심 모두 **0.00 mm** — `as_closest_canonical` 이 축 순서/뒤집힘을 정상 해소 (오탐) |
| `sub-235` | 예 (11.1 mm) | **예** | **중심 6.57 mm, 코너 15.48~27.33 mm** — 진짜 어긋남 |

즉 canonical 변환은 제 역할을 하고 있으나(오탐 1건 흡수), **진짜 어긋난 1건은 아무도 잡지 못했다**.
sub-235 는 TRACE 와 다른 위치의 ADC 를 2번 채널로 붙여 학습에 들어가 있었다.

**수정** canonical affine 이 다르거나 shape 이 다르면 `nibabel.processing.resample_from_to(order=1)` 로
**ADC 를 TRACE 격자에 리샘플**한 뒤 스택한다. split 을 바꾸지 않으므로 sha256 이 보존된다(F10 참조).
TRACE/ADC 전경 Dice: 0.761 → **0.800**.

### F3 [중간] 클립 ±6 이 병변을 포화시킨다 — **미수정, A4 이관**

**[측정]** (1,233명, 현재 캐시 = 수정 후 캐시 모두 동일한 결과)

| 지표 | 값 |
|---|---|
| 병변 복셀 중 TRACE 채널이 +6 에 포화된 비율 (환자 평균) | **0.346** |
| 병변의 50 % 이상이 포화된 환자 | **397 / 1233 (32.2 %)** |
| 병변의 10 % 이상이 포화된 환자 | **852 / 1233 (69.1 %)** |
| 병변 내부 정규화 TRACE 중앙값 (환자별) | p25 4.33 / **p50 5.63** / p75 6.00 |
| 비교: 전경(비병변 포함) 중 +6 포화 비율 | 1.6 % |

병변/비병변 분리 자체는 남아 있으나(34.6 % vs 1.6 %), **병변 내부 명암과 "얼마나 밝은가" 정보가 잘린다**.
지금 고치면 캐시 1,233개 전부가 무효가 되므로 A3 단독으로 바꾸지 않았다.
**권고(A4)**: `clip=6` 을 유지한 기준선과 `clip=10` 또는 백분위 기반 스케일(p50/p99.5) 변형을 val Dice 로 1회 비교.
전처리가 13초, 학습이 GPU 에서 수 분이므로 비용이 거의 없다.

### F4 [중간] 증강 RNG 가 DataLoader 워커에 fork 된다

`SliceDataset.__getitem__` 이 `self.rng` 를 썼다. `num_workers>0` 이면 워커 프로세스마다 **동일한 상태의 RNG 사본**이
생겨 (a) 워커별로 같은 증강열이 반복되고 (b) 결과가 워커 수·스케줄링에 의존해 **재현이 깨진다**.
현재 `configs/seg_unet2d.yaml` 은 `num_workers: 0` 이라 아직 발현하지 않았지만, A4 가 GPU 처리량을 위해
`num_workers` 를 올리는 것이 자연스러운 다음 수순이라 지뢰였다.

**수정** 항목별 결정론적 생성기 `np.random.default_rng((seed, epoch, index))` 사용.
→ 워커 수와 무관, 조회 순서와 무관, 에폭마다 다름. (`tests/test_dataset_rng.py` 4개 테스트로 고정)
`neg_pos_ratio` 음성 샘플링은 그대로 `self.rng`(메인 프로세스, 에폭 1회)로 유지했다. **하이퍼파라미터는 변경하지 않았다.**

---

## 4. A2 이관 항목 처리 결과

| # | A2 이관 항목 | A3 처리 |
|---|---|---|
| 1 | `(mk > 0)` 이진화 유지, "마스크 값 2 또는 3 인 10명" 기록 | **완료.** `preprocess.py` 모듈 docstring 에 근거와 subject 블록(503–509, 1252–1257)을 명시했고, `preprocess_subject` 가 `flags["mask_max_value"]` 를 반환해 `data/cache/_manifest.json` 에 자동 기록된다. **[측정]** train+val 에서 값≠1 인 환자 **9명**(sub-503/504/505/507/508/1252/1253 = 값 2, sub-1255/1257 = 값 3) → 나머지 1명은 test 에 있다(코호트 총 10명, A2 값과 일치). 리샘플 후 전원 이진 |
| 2 | 결합 마스크 `*_desc-lesion_mask.nii.gz` 사용 금지 | **확인 완료.** 저장소 전체에서 이 파일의 **복셀을 읽는 코드가 없다**. `index.py` 가 존재 여부(`has_mask_any`, 적격 1,451명 중 1,450명 보유)만 기록할 뿐이며 적격성·전처리·특징·학습 어디에도 쓰이지 않는다. docstring 에 금지 사유(sub-507 = {1,4})를 못박았다 |
| 3 | `acute == chronic` 플래그 | **완료(train+val 범위).** 만성 마스크 보유 168명(적격 1,451명 중에서는 198명)에 대해 복셀 단위 전수 비교: **완전 동일 3명**(`sub-235`, `sub-1720`, `sub-1737`), **급성 ⊂ 만성 5명**(위 3명 + `sub-1095`, `sub-1132`). 급성∩만성 > 0 인 환자는 59명이고 그 중 Dice 중앙값은 0.0093(정상적인 인접 접촉 수준). 플래그 파일: `results/a3/acute_chronic_flags.csv`. **test 218명은 규칙에 따라 계산하지 않았다** — 같은 스크립트를 `--which test` 로 최종 평가 단계에서 실행하면 된다 |
| 4 | 128² 리샘플 안전성 재확인 | **완료.** 빈 마스크 **0건**(native 도 0건), 부피 보존비 min 0.667 / p1 0.836 / **p50 0.990** / p99 1.132 / max 1.381, 리샘플 후 최소 병변 **3복셀**(`sub-1666`, `sub-1684`, native 10·11복셀). 0.9 미만 42명, 1.2 초과 5명 |
| 5 | 부피 오차를 mL 와 % 둘 다 | **완료.** `metrics.VolumeAgreement` 에 `mape_pct`, `median_abs_pct_err` 를 **추가**(기존 필드 불변). 근거 **[측정]**: native 복셀 부피 **1.449 ~ 8.610 mm³ (5.9배)**, 병변 부피 **0.067 ~ 557 mL**(train p5 0.36 / p50 8.2 / p95 132) → mL 단독 MAE 는 상위 몇 명이 지배한다 |
| 6 | `acuteischaemicstroke` 제외 | **확인 완료.** `train_cls.py` 의 임상 특징은 `CLIN = ["age","sex_male","nihss"]` 뿐이고 `features.py` 의 15개 병변 특징에도 없다. 이미 충족 |

---

## 5. split 별 기술통계 (train / val 만)

> **test = 218명. 규칙에 따라 개수 외에는 어떤 통계도 산출하지 않았다.**

재현: `PYTHONPATH=src .venv/bin/python scripts/analysis/a3_stats.py --workers 32 --out results/a3`
(출력: `results/a3/split_stats.json`, `results/a3/per_subject_stats.csv`)

### 5.1 슬라이스 · 병변 양성 슬라이스

| 항목 | train | val |
|---|---|---|
| 환자 수 | **1,015** | **218** |
| 총 슬라이스 | **25,963** | **5,593** |
| 슬라이스/환자 (p5 / p50 / p95, 범위) | 24 / **26** / 27 (21–30) | 24 / **26** / 28 (23–32) |
| 병변 양성 슬라이스 수 | 7,497 | 1,622 |
| **양성 슬라이스 비율(풀링)** | **0.2888** | **0.2900** |
| 환자별 양성 비율 (p5 / p25 / p50 / p75 / p95) | 0.071 / 0.125 / **0.269** / 0.423 / 0.615 | 0.042 / 0.154 / **0.259** / 0.423 / 0.615 |
| 환자별 양성 비율 범위 | 0.038 – 0.852 | 0.037 – 0.769 |

→ train/val 의 양성 슬라이스 비율 차이는 **0.0012** 로 층화가 잘 작동했다.
`neg_pos_ratio: 1.0` 은 학습 시 양성:음성을 1:1 로 맞추므로, 에폭당 학습 슬라이스는 대략 2 × 7,497 ≈ **15,000장** **[추정]**.

### 5.2 병변 부피 (마스크 = GT, 리샘플 후 캐시 기준)

| 지표 | train | val |
|---|---|---|
| 병변 부피 mL (p5 / p25 / **p50** / p75 / p95) | 0.36 / 1.69 / **8.20** / 34.05 / 132.4 | 0.41 / 1.79 / **9.82** / 35.64 / 126.2 |
| 최소 / 최대 mL | 0.067 / 523.7 | 0.067 / 557.3 |
| 평균 mL | 30.05 | 31.90 |
| **뇌 전경 대비 % (p5 / p50 / p95)** | 0.011 / **0.261** / 4.79 | 0.014 / **0.299** / 4.11 |
| 리샘플 후 병변 복셀 수 (p5 / p50 / p95) | 16 / 394 / 6,314 | 21 / 468 / 6,409 |
| **소병변(< 2 mL) 비율** | **0.2798** | **0.2752** |
| 전경(두부) 부피 mL (p50) | 3,145 | 3,166 |

> "뇌 전경"은 `foreground()`(TRACE p99 의 5 %) 기준이라 **두개골·두피·목을 포함한 두부 전경**이다(중앙값 3,145 mL, 실제 뇌 실질보다 크다).
> 절대 mL 이 복셀 크기 5.9배 차이에 흔들리는 문제를 보정하기 위한 **상대 척도로만** 쓴다. **[측정, 단 "뇌 부피"로 해석하지 말 것]**

### 5.3 강도 분포

정규화 후(모델이 실제로 보는 값, 클립 ±6):

| 지표 (환자별 값의 분위수) | train p5 / p50 / p95 | val p5 / p50 / p95 |
|---|---|---|
| TRACE z 전체 중앙값 | −1.41 / **−1.12** / −0.87 | −1.35 / **−1.12** / −0.89 |
| ADC z 전체 중앙값 | −2.06 / **−1.21** / −0.67 | −2.03 / **−1.08** / −0.67 |
| **TRACE z 병변 중앙값** | 1.94 / **5.63** / 6.00 | 2.26 / **5.71** / 6.00 |
| **ADC z 병변 중앙값** | −0.75 / **−0.19** / 0.32 | −0.73 / **−0.15** / 0.68 |

→ TRACE 는 병변이 강하게 밝고(정상 −1.1 vs 병변 +5.6), **ADC 는 병변에서 겨우 −0.19** 로 대비가 매우 약하다.
ADC 채널의 기여도는 A4 가 채널 ablation 으로 확인할 가치가 있다 **[추정]**.
TRACE 병변 p75 이상이 6.00(클립값)인 것이 F3 의 근거다.

정규화 **전** 원 강도(뇌 전경 중앙값) — 환자 간 스케일 이질성:

| 지표 | train (p5 / p50 / p95, 최소–최대) | val |
|---|---|---|
| TRACE raw 중앙값 | 138 / **219** / 388 (13 – 33,398) | 145 / **215** / 394 (13 – 1,580) |
| ADC raw 중앙값 | 344 / **4,679** / 12,236 (−4.7 – 192,147) | 0.7 / **4,966** / 11,374 (−4.9 – 17,313) |

→ ADC 원 강도는 환자 간 **p95/p5 = 69배**(TRACE 는 2.8배). 음수 중앙값(채움값)도 있다.
**환자별 로버스트 정규화는 선택이 아니라 필수**이며, 동시에 F1 이 왜 터졌는지도 설명한다 **[측정]**.

### 5.4 복셀 크기 · 코호트 구성

| 지표 | train | val |
|---|---|---|
| native 복셀 부피 mm³ (min / p50 / max) | 1.449 / 6.272 / 8.610 (**5.94배**) | 2.153 / 6.272 / 7.336 |
| 리샘플 후 복셀 부피 mm³ (min / p50 / max) | 17.72 / 22.47 / 28.71 (1.62배) | 17.62 / 22.47 / 28.71 |
| 장비 | Philips 1,001 / GE 10 / Siemens 4 | Philips 213 / Siemens 3 / GE 2 |
| 자장 | 1.5T 806 / 3T 209 | 1.5T 173 / 3T 45 |
| 병인 라벨 | LAA 162 / CE 105 / SVO 58 / OtherDet 42 / Cryptogenic 283 / 없음 365 | LAA 35 / CE 22 / SVO 13 / OtherDet 10 / Cryptogenic 60 / 없음 78 |

층화 결과 병인 구성비가 train:val 에서 거의 정확히 유지된다(`data/splits_strata.csv` 와 일치).
장비는 층화 변수가 아니었지만 결과적으로 비율이 비슷하다(Philips 98.6 % vs 97.7 %) **[측정]**.

### 5.5 마스크 · 좌표계 검증

재현: `PYTHONPATH=src .venv/bin/python scripts/analysis/a3_checks.py --workers 32 --out results/a3`

| 검사 | 결과 (1,233명) |
|---|---|
| 캐시 img shape | 전원 `(S, 2, 128, 128)`, S ∈ [21, 32] |
| 마스크 값 집합 ⊆ {0,1} | **1,233 / 1,233** |
| img 유한값 / 값 범위 | 전원 유한, [−6.0, +6.0] |
| 리샘플 후 빈 마스크 | **0명** |
| 원본 orientation | 전원 `LAS` → canonical `RAS` 전원 성공 |
| 마스크 affine == TRACE affine (canonical) | **1,233 / 1,233** |
| ADC affine == TRACE affine (canonical) | 1,232 / 1,233 (sub-235 → F2 수정으로 리샘플) |
| 정렬 대조검정: 병변 TRACE z > **좌우반전** 병변 | **1,227 / 1,233 (99.5 %)**, 중앙 대비 +2.12 |
| 정렬 대조검정: 병변 TRACE z > **전치(H↔W)** 병변 | **1,227 / 1,233** |

→ 마스크는 리샘플 후에도 이진이고, 영상과 **해부학적으로 같은 위치**를 가리킨다.
(마스크는 `order=0` 최근접, 영상은 `order=1` 선형이지만 `scipy.ndimage.zoom` 은 두 경우에 **동일한 좌표 매핑**을 쓰므로 서브픽셀 어긋남이 생기지 않는다.)

---

## 6. 수정·추가한 파일

| 파일 | 변경 |
|---|---|
| `src/strokeai/data/preprocess.py` | **F1** `robust_stats()` 신설 + MAD=0 폴백 / **F2** ADC↔TRACE affine 검사 후 `resample_to_grid()` / 리샘플 후 마스크 이진·shape assert / `flags` 반환 / A2 결정(값 2·3, 결합 마스크 금지) docstring 명문화 |
| `scripts/preprocess.py` | `flags` 를 npz 에서 분리해 `_manifest.json` 에 집계(`flag_counts`). npz 내용은 불변 = 재현 해시 유지 |
| `src/strokeai/data/dataset.py` | **F4** 항목별 결정론적 증강 RNG, `epoch` 추적 |
| `src/strokeai/data/split.py` | **F15** `_flag()` 헬퍼로 FutureWarning 제거(해시 불변), ADC affine 을 적격성에 넣지 **않은** 이유 주석화 |
| `src/strokeai/metrics.py` | A2 항목 5 — `VolumeAgreement.mape_pct`, `.median_abs_pct_err` 추가(기존 필드·동작 불변) |
| `tests/test_preprocess_geometry.py` | **신규 7개** — MAD=0 폴백, 레거시 비트 동일성, 마스크 이진화(값 3), 격자 일치, ADC 리샘플, 마스크 affine 불일치 예외, 전처리 결정론 |
| `tests/test_no_test_leakage.py` | **신규 7개** — CLI 기본값 3종, `run_all.sh` 내용, split 무교집합·헌장 해시, **캐시에 test 환자 없음** |
| `tests/test_dataset_rng.py` | **신규 5개** — 환자 단위 슬라이스 테이블, 증강 재현·순서 무관, 에폭 간 변화, seed 동일성, 영상·마스크 동시 플립 |
| `tests/test_metrics.py` | +1 — 부피 오차 % 보고 |
| `scripts/analysis/a3_checks.py` | **신규** 캐시 무결성·기하·급성/만성 감사 (기본 `--which train val`) |
| `scripts/analysis/a3_stats.py` | **신규** split 별 기술통계 (기본 `--which train val`) |
| `results/a3/*` | 산출물: `checks.json`, `split_stats.json`, `cache_check.csv`, `geom_check.csv`, `per_subject_stats.csv`, `acute_chronic_flags.csv`, `stale_cache_subjects.txt` |

**최종 테스트**: `PYTHONPATH=src .venv/bin/python -m pytest` → **35 passed in 4.29s** (기존 15 → 35, +20).

---

## 7. 후속 에이전트 이관 항목

**A4 (모델·최적화)**
1. **학습 전에 캐시 45명분을 재생성할 것** (§1.2 명령, 약 2초). 하지 않으면 44명의 ADC 채널이 포화 이진맵이고 1명은 어긋난 ADC 로 학습된다.
2. **F3 클립 실험**: `preprocess.py` 의 `clip=6.0` 을 10 또는 백분위 스케일로 바꾼 변형을 val Dice 로 1회 비교. 전처리 13초 + GPU 학습 수 분이므로 비용이 거의 없다. **바꾸기로 하면 캐시 1,233개 전량 재생성 필요.**
3. **`num_workers` 를 올려도 안전하다**(F4 수정 완료). 다만 올린 뒤 같은 seed 2회 실행으로 val Dice 재현을 실제로 확인할 것 — 헌장 기준은 "재실행 시 val Dice 차이 < 0.02".
4. **ADC 채널 ablation 권고**: 병변 내부 ADC z 중앙값이 −0.19(TRACE 는 +5.63)로 대비가 약하다. TRACE 단독 vs TRACE+ADC 를 비교해 2채널이 실제로 값을 하는지 확인.
5. `batch_size`·`size` 상향 검토 시, 캐시는 128² 로 고정되어 있으므로 `size` 를 192 로 올리려면 **전처리부터 다시** 해야 한다(약 20초 예상 **[추정]**).

**A6 (목표·진단)**
1. **부피 오차는 mL 과 % 를 함께 보고**(`mape_pct`, `median_abs_pct_err` 추가됨). 병변 부피가 0.067–557 mL 로 4자릿수에 걸쳐 있어 mL 단독 MAE 는 상위 몇 명이 지배한다.
2. **F14 라벨 품질 플래그**: 급성 = 만성 완전 동일 3명, 급성 ⊂ 만성 5명(train+val). 실패 사례 분석에서 이들이 상위에 오르는지 확인하고, 성공 기준 계산 시 민감도 분석(제외/포함)을 붙일 것. 파일 `results/a3/acute_chronic_flags.csv`.
3. **소병변 구간의 실제 한계**: 리샘플 후 최소 병변이 **3복셀**이다. 3–20복셀 구간에서 Dice 는 이산적으로 튄다(1복셀 차이가 Dice 를 0.1 이상 움직임) → `dice_by_gt_volume_ml` 의 `[0,2)` 구간은 **부피 오차와 검출 여부를 같이** 보고할 것.
4. **F9 층화 caveat 를 진단 보고서에 명시**: split 층화가 test 환자의 병변 크기 구간(<2 mL / ≥2 mL)을 사용했다. 영상 재열람은 없었지만 완전한 blind holdout 은 아니다.
5. 최종 평가 단계에서 `a3_checks.py --which test` 와 `a3_stats.py --which test` 를 **그때 처음** 실행해 test 코호트의 무결성·분포를 train/val 과 비교할 것(사전 등록된 절차로).

**오케스트레이터**
- `data/index.csv` 를 언젠가 재생성한다면 `build_index.py` 에 `adc_affine_matches_trace` 열을 넣는 것을 권장한다(F10). 단 **적격성 필터에는 넣지 말 것** — sub-235 가 빠지면 splits sha256 이 바뀌어 헌장 값과 불일치한다. 지금은 전처리 단계의 리샘플로 해결되어 있다.

---

## 8. 재현 명령 모음

```bash
cd /home/user1/Desktop/Multi-Agent/JLK
export PYTHONPATH=src

# 1) split 재현 (해시가 0f923acd… 여야 함)
.venv/bin/python scripts/make_splits.py --out /tmp/a3/rerun1/splits.json
sha256sum data/splits.json /tmp/a3/rerun1/splits.json

# 2) 전처리 재현 (기존 캐시를 건드리지 않고 별도 디렉터리로)
.venv/bin/python scripts/preprocess.py --out /tmp/a3/cache_full --size 128 --workers 44 --which train val
# → 배열 해시 비교는 보고서 §1.2 참조. 마스크는 반드시 1,233명 전원 불변이어야 한다.

# 3) 무결성·기하·급성/만성 감사  (test 는 읽지 않음)
.venv/bin/python scripts/analysis/a3_checks.py --workers 32 --out results/a3

# 4) split 별 기술통계          (test 는 읽지 않음)
.venv/bin/python scripts/analysis/a3_stats.py --workers 32 --out results/a3

# 5) 회귀 테스트
.venv/bin/python -m pytest -q      # 35 passed

# 6) [A4 가 학습 전에 1회] 버그 수정으로 낡아진 45명분 캐시만 재생성
while read sid; do rm -f "data/cache/$sid.npz"; done < results/a3/stale_cache_subjects.txt
.venv/bin/python scripts/preprocess.py --size 128 --workers 44 --which train val
```

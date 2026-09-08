# Astra 보고서 — 주제·방향성 독립 교차검증

작성일: 2026-09-08 · 담당: Astra direction-review · 절차: ①독자 조사 → ②A1 판정 → ③리스크 → ④최종 판정
조사 방식: WebSearch + 로컬 데이터(`data/index.csv`, `data/raw/ds004889/*`) 직접 집계. WebFetch 는 github.com 1건만 성공, 나머지(PMC·Nature·arXiv·j-stroke·medRxiv·jlkgroup·phdkim·PubMed) **접근 불가** → 검색 요약 기반이며 확실도를 표기함.

## 1. 결론 (5줄)

1. **최종 판정: [현 방향 유지 + 범위 축소]**. SOOP 로 "DWI 분할 + 병인 분류"를 하는 주제 자체는 JLK 의 JBS-01K(JoS 2024 논문: U-Net 분할 + EfficientNetV2 분류, DWI+AF 입력)와 구조가 같아 적합하다. 다만 A1 이 P0 로 올린 DICOM I/O·2.5D·민감도/특이도는 **"너무 어렵지 않게"에 비추어 과잉**이며, 특히 민감도/특이도는 이 데이터에서 **정의 자체가 안 선다**(음성 환자 없음).
2. SOOP 마스크는 **수동(3명 숙련 rater, MRIcroGL, ADC/TRACE 위에서 직접 트레이싱)**으로 확인됨(검색 요약, 확실도 중). 반자동 아님. 단 10례는 값이 {0,2}/{0,3} 이라 이진화가 필요하고, 158례는 뇌경색 진단인데 마스크가 없다(음성 아님 → 환자 단위 특이도 계산 불가).
3. 병인 4클래스 라벨은 마스크 보유자 중 **527명뿐**(CE149/LAA232/SVO84/Other62). SVO 중앙값 부피 1.13 mL vs 나머지 10~23 mL → **SVO 는 부피만으로 거의 분리**되므로 "분류 정확도"는 LAA-vs-CE 를 따로 보고해야 오해가 없다.
4. 채용담당자 관점 최대 리스크는 "JLK 제품을 재현/능가했다"로 읽히는 것과, 클리니션+AI DSC 0.742(A1 인용)를 모델 Dice 와 같은 축에 놓는 것. 두 가지 모두 문구로 차단 가능.
5. 1주 MVP: 2D U-Net(TRACE+ADC 2ch) 환자 단위 split → Dice·병변 부피 오차 → 부피+임상변수 로지스틱/GBM 병인 분류(LAA/CE/SVO/Other, macro-F1 + LAA-vs-CE AUC 별도) → README. DICOM·3D·CT 는 "의도적 제외"로 명시.

## 2. 독자 조사 (A1 읽기 전)

### 2-1. JLK 핵심 과제·제품 구조 (언론·논문 검색 요약, 원문 미대조)

| 항목 | 내용 | 확실도 |
|---|---|---|
| JBS-01K (=JLK-DWI 계열) | DWI 3D 분석으로 뇌경색 병변 자동 분할 + TOAST 유형(LAA/SVO/CE/Others) 분류. 식약처 3등급, 혁신의료기기 통합심사, NECA 비급여 승인 | 높음(다수 기사) |
| JBS-01K 기반 논문 (Ryu 등, J Stroke 2024; 저자 3인 JLK 소속) | 3개 병원 2,988명, **U-Net 분할 + EfficientNetV2 분류**, 입력 DWI(+AF 유무). DWI 단독 일치율 59.3–60.7%, AUC LAA 0.69–0.72 / SVO 0.83–0.90 / CE 0.79–0.82. 라벨은 신경과 전문의가 MRI 기반 TOAST 로 판정 | 중(초록·스니펫) |
| 나머지 포트폴리오 | JLK-LVO(CTA, FDA), JLK-ICH(NCCT, FDA), JLK-CTP(CTP, FDA), JLK-PWI(FDA), JLK-NCCT(FDA 2026-03), JLK-CTL/CTI/WMH/CMB/LAC/UIA 등 11종 | 중 |
| 전문연구요원 공고 | "AI 빅데이터 연구원(전문연구요원), 석사 이상, 경력무관, 강남구" 메타데이터만 확인. 담당업무·우대 원문 **미확인** | 낮음 |

시사점: 회사의 DWI 제품은 **"분할 → 부피 → 유형분류"** 3단이며 분류 성능은 회사 논문에서도 중간 수준(AUC 0.7~0.9). 개인 프로젝트에서 낮은 분류 성능이 나와도 "회사 논문과 같은 난이도 구조"라고 설명 가능.

### 2-2. 공개 데이터셋 후보

| 데이터셋 | 모달리티·라벨 | 규모 | 접근·라이선스 | 확실도 |
|---|---|---|---|---|
| **SOOP ds004889** (현재) | TRACE+ADC(+T1/FLAIR), 급성/만성 마스크(수동), TOAST 병인, NIHSS/mRS | 영상 1715, 급성 마스크 1451, 병인 4클래스 527(+Crypto 403) | OpenNeuro 즉시, **CC0** | 높음(로컬 확인) |
| ISLES'22 | DWI+ADC+FLAIR, 전문가 마스크 | 학습 250 (테스트 150 비공개) | Zenodo, **CC BY 4.0** | 높음 |
| ISLES'24 | NCCT/CTA/CTP + 후속 MRI 최종경색 마스크, 임상변수 | 학습 149 | Zenodo, **CC BY-NC(-SA) 4.0** (표기 불일치) | 중 |
| ATLAS v2.0 | T1w(만성) 수동 마스크 | 학습 655 (테스트 마스크 비공개) | NITRC/INDI, 신청·DUA 성격(세부 미확인) | 중 |
| RSNA ICH 2019 | NCCT, 슬라이스 단위 출혈 5유형 라벨(마스크 없음) | 25k+ 스캔 | Kaggle 규약(비상업·경진대회 목적) | 중 |
| PhysioNet CT-ICH | NCCT, 출혈 마스크 | 82 | PhysioNet, CC BY 4.0 | 높음 |
| CQ500 | NCCT, 스캔 단위 라벨(BHX 로 bbox 확장) | 491 | **CC BY-NC-SA** | 높음 |
| BHSD | NCCT, 3D 다중클래스 출혈 마스크 | 192(픽셀) + 2200(슬라이스) | GitHub/HF, 라이선스 미확인 | 낮음 |

### 2-3. 후보별 프로젝트 난이도·차별성·리스크

| 후보 → 프로젝트 | 난이도 | JLK 대응 제품 | 차별성 | 리스크 |
|---|---|---|---|---|
| SOOP: DWI 분할+부피+병인 분류 | **중** | JBS-01K/JLK-DWI | 병인 라벨+임상변수까지 있는 유일한 공개 DWI 셋 → 분류 단계까지 가능 | 단일기관·Philips 1.5T 편중(1676/1715), 5 mm 두께, 소병변 다수 |
| ISLES'22: DWI 분할 | 하~중 | JLK-DWI | nnU-Net 벤치마크 풍부 → 비교 쉬움 | n=250, 분류 불가, "챌린지 따라하기"로 보일 수 있음 |
| ISLES'24: CT→최종경색 예측 | **상** | JLK-CTP | 신선함 | n=149, 다중모달 정합 필요, NC 라이선스 |
| ATLAS: 만성 T1 분할 | 하~중 | JLK-LAC(원거리) | 규모 큼 | 급성 DWI 와 거리, DUA |
| RSNA/CQ500/BHSD: 출혈 검출 | 하~중 | JLK-ICH | 대규모, 튜토리얼 많음 | 2D 분류 튜토리얼과 구별 어려움, 비상업 라이선스 |

→ SOOP 이 "회사와 비슷한 주제 유경험자" 목표에 가장 가깝고 라이선스(CC0)도 가장 깨끗하다. ISLES'22 는 **분할 모델 외부검증용 보조**로만 가치 있음(추가 시).

## 3. A1 보고서 항목별 판정

| A1 항목 | 판정 | 근거 |
|---|---|---|
| 결론1 "주제 적합성 매우 높음, JBS-01K 와 1:1 대응" | **동의** | JoS 2024 논문 파이프라인(U-Net→EfficientNetV2, DWI+AF)이 본 과제와 같은 3단 구조. 단 회사는 분류 입력이 **영상(DWI 슬라이스+마스크)** 이고 본 프로젝트는 **부피·형태 특징+임상변수(표 형식)** 라 "같은 문제, 다른 접근"으로 써야 정확 |
| 결론3 "publication 이 자격요건" | **부분동의** | A1 도 인정하듯 **다른 공고(AI알고리즘 연구원)** 문구. 전문연구요원 공고 원문 미확인이므로 "필수"로 단정 금지. 지원 전략을 바꾸는 근거로는 부족 |
| P0-1 DICOM I/O | **반박(범위 과잉)** | 데이터가 NIfTI 이고 회사 논문도 NIfTI/BIDS 급 처리. NIfTI→DICOM 역변환은 "다뤄봤다" 진술을 위한 인위적 장치로, 면접에서 오히려 약점. 필요하면 README 한 줄("PACS 연동은 범위 밖") 로 충분 |
| P0-2 2.5D/3D | **부분동의** | 3D U-Net 은 5 mm 두께·24~26 슬라이스 데이터에 비용 대비 이득이 작음. 2.5D 는 **인접 3슬라이스를 채널로 쌓는 것뿐**(코드 10줄 내외)이라 P1 로 두고 여력 시 1회 비교 실험 |
| P0-3 민감도·특이도·환자 단위 검출률 | **반박(정의 불가)** | 마스크 보유 1451례 모두 병변 있음(`acute_mask_empty=0`); 마스크 없는 264례 중 158례는 뇌경색 진단(=음성 아님). **환자 단위 특이도는 계산할 수 없다.** 대체: 연결성분(병변) 단위 검출률·정밀도, 부피 오차(mL, Bland-Altman) — 이것이 JLK-DWI "부피 측정" 과 직접 접점 |
| P1-4 클래스 불균형·macro-F1·Cryptogenic 방침 | **동의** | 4클래스 527례, 최소 클래스 62. macro-F1/per-class recall 필수. Cryptogenic 은 정의상 "판정 불가"라 **학습에서 제외** 권장(Others 병합 금지) |
| P1-5 추론 시간 | 동의 | 1줄 비용 |
| P1-6 site-holdout | **반박** | 단일기관(Prisma Health-Upstate)·Philips 1676/1715. 사이트 변수 없음. 대신 **3.0T(358) vs 1.5T(1357) 홀드아웃**이 유일한 현실적 일반화 축이나 MVP 범위 밖 |
| P2 CT/PACS/MLOps 제외, 인허가 문서 모사 금지 | **동의** | |
| JBS-01K "DSC 0.523→0.742" 인용 | **반박(해석 오류)** | 이 수치는 **판독의 단독 vs 판독의+AI 의 DSC** (reader study)이지 모델 단독 Dice 가 아님. 본 프로젝트 Dice 와 같은 축에 두면 채용담당자가 바로 지적할 수 있음. 비교 축이 필요하면 공개 벤치마크(nnU-Net DWI 외부검증 Dice ≈0.81, PMC11950494 검색 요약)를 "참고치"로만 |
| README 포지셔닝 문구 | **동의(수정 2건)** | ① "임상 변수(나이·성별·NIHSS)"에 mRS 는 **퇴원 시 지표라 병인 예측 입력으로 부적절**(누수·인과 역전) → 제외 명시. ② "JBS-01K 와 같은 문제 구조" 뒤에 "접근 방식(특징 기반 분류)은 다름"을 한 구절 추가 |

종합: A1 의 갭 분석은 "제품 스펙 대비" 관점에서는 타당하나, 사용자 목표("유경험자 되기, 너무 어렵지 않게")를 기준으로 하면 **P0 3개 중 2개는 강등, 1개는 재정의**가 맞다.

## 4. 방향성 리스크와 완화책

| # | 리스크 | 실측 근거 | 완화책 |
|---|---|---|---|
| R1 | 마스크 라벨 값 이질 | 10례가 {0,2}/{0,3} | 로드 시 `mask>0` 이진화, 전처리 로그에 건수 기록 |
| R2 | 마스크 없는 뇌경색 158례를 음성으로 오해 | crosstab 확인 | 해당 264례는 **분할 학습·평가에서 제외**, README 에 "음성 케이스 없음 → 특이도 미보고" 명시 |
| R3 | 소병변·얇은 병변에서 2D Dice 급락 | <1 mL 16.7%, ≤2 슬라이스 13.5% | Dice 를 부피 4분위별로 층화 보고; 환자 평균 Dice 와 전체 voxel Dice 둘 다 제시 |
| R4 | SVO 라벨이 부피와 얽힘(정의상 <1.5 cm) | SVO 중앙값 1.13 mL, 63% 가 1.77 mL(1.5 cm 구) 미만; 타 클래스 14~18% | (a) 부피 단독 베이스라인을 먼저 보고 → 추가 특징의 증분만 주장 (b) **LAA-vs-CE 2진 AUC 를 주 지표**로 병기 (c) "SVO 분리는 정의상 쉬움"을 본문에 선언 |
| R5 | 병인 라벨 신뢰도·버전 불일치 | 논문 스니펫 CE 343 vs tsv CE 159(확실도 낮음) | tsv v1.1.2 기준으로만 보고, 라벨은 임상의 TOAST 판정이며 검증 불가함을 명시 |
| R6 | 임상변수 결측 | 4클래스+마스크 527례 중 NIHSS 등 확보 527(양호), race/bmi 결측 다수 | 입력은 age·sex·NIHSS·priorstroke 로 제한; mRS 제외(R7) |
| R7 | 결과 변수 누수 | `gs_rankin_6isdeath` 는 퇴원 시 결과 | 분류 입력에서 제외, 문서화 |
| R8 | 지표 해석 오해(회사 수치와 직접 비교) | A1 의 DSC 0.742 인용 | "직접 비교 불가" 문구 + 비교표 미작성 |
| R9 | 채용담당자 오해: "JLK 문제를 풀었다/제품 재현" | — | 첫 문단에 "개인 학습 프로젝트, 과제 구조 참조점" 고정(A1 문구 채택) |
| R10 | 스캐너 편중으로 일반화 주장 불가 | Philips 98% | 일반화 주장 자체를 하지 않고 "한계" 절에 기록; 여력 시 3.0T 홀드아웃 |
| R11 | 4D NIfTI(마지막 축 1) 처리 실수 | 1715 전부 ndim=4 | `squeeze` 후 shape 검증 테스트 1개 |

## 5. 최종 판정

**[현 방향 유지 + 범위 축소]**. 주제(SOOP DWI 분할→부피→병인 분류)는 JLK 의 대표 과제 구조와 일치하고, CC0·즉시 접근·병인 라벨 보유라는 점에서 대체 후보(ISLES/ATLAS/출혈 셋) 대비 우위가 분명하므로 주제 변경·확장의 이유가 없다. 반면 A1 의 P0 는 "제품 스펙 흉내" 방향으로 범위를 키우며, 그중 민감도/특이도는 이 데이터에서 성립하지 않고 DICOM I/O 는 인위적이다. 사용자 목표는 "유경험자"이지 "제품 수준"이 아니므로, 분할·부피·분류 3단을 **끝까지 돌려서 정직하게 보고하는 것**에 집중하고, 확장 항목은 README "의도적 제외" 절로 옮기는 것이 완성도와 방어력 모두에 유리하다.

### 1주 MVP (서버 진행 시)
1. 전처리: 1451 마스크 보유 환자만, TRACE+ADC 2ch, 마스크 `>0` 이진화, 환자 단위 split(기존 `splits.json` 유지), 슬라이스 단위 z-score.
2. 2D U-Net(Dice+BCE) 1회 학습(≤20 epoch) → val 로 threshold 선택 → test 1회: 환자별 Dice(부피 4분위 층화), 병변 부피 오차(mL, Bland-Altman 1장).
3. 분류: 4클래스 527례(Cryptogenic 제외), 특징 = 예측 마스크 부피·성분 수·최대 성분 부피·좌우/ADC 평균 + age·sex·NIHSS·priorstroke → 로지스틱/GBM, macro-F1·per-class recall + **LAA-vs-CE AUC** + 부피 단독 베이스라인.
4. README: 결과표 2개, 한계(단일기관·Philips·특이도 미보고·SVO 정의 의존) 및 "의도적 제외"(DICOM·3D·CT) 절.
5. 여력 시에만: 2.5D(3슬라이스 채널) 1회 비교, 추론 시간 1줄.

## 6. 출처
- https://www.monews.co.kr/news/articleView.html?idxno=327851 (JBS-01K NECA)
- https://www.dailymedi.com/news/news_view.php?ca_id=22&wr_id=921463 (JBS-01K reader study 수치)
- https://healthairegister.com/products/jlk-inc-jbs-01k/ (JBS-01K 개요)
- https://pubmed.ncbi.nlm.nih.gov/38836277/ , https://www.j-stroke.org/journal/view.php?doi=10.5853%2Fjos.2024.00535 (Ryu 등 2024, 접근 불가·스니펫)
- https://www.koreabiomed.com/news/articleView.html?idxno=26760 , idxno=31083 (JLK FDA 현황, 11종 포트폴리오)
- https://www.pharmnews.com/news/articleView.html?idxno=251168 (JLK-LVO FDA)
- https://phdkim.net/board/postgraduate-career/23 (전문연구요원 공고, 접근 불가)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11297183/ , https://www.nature.com/articles/s41597-024-03667-5 (SOOP 논문, 접근 불가·스니펫: 3 rater 수동, MRIcroGL)
- https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo/ (접근 성공: 1,714명 중 1,449 확진, 마스크 TRACE 위 수동)
- https://zenodo.org/records/7960856 , https://www.nature.com/articles/s41597-022-01875-5 (ISLES'22, CC BY 4.0)
- https://zenodo.org/records/16813698 , https://github.com/ezequieldlrosa/isles24 (ISLES'24, CC BY-NC)
- https://atlas.grand-challenge.org/ , https://www.nature.com/articles/s41597-022-01401-7 (ATLAS v2.0)
- https://physionet.org/content/ct-ich/1.3.0/ (CT-ICH, CC BY 4.0) · http://headctstudy.qure.ai/dataset (CQ500, CC BY-NC-SA) · https://arxiv.org/abs/2308.11298 (BHSD)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11950494/ (nnU-Net DWI 외부검증 Dice 0.81, 스니펫)
- 로컬: /home/user/jlk-/data/index.csv, index_summary.json, splits_strata.csv, data/raw/ds004889/{README.md,participants.json,CHANGES}

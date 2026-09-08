# A1 보고서 — 주제·포지셔닝 (JLK 전문연구요원 지원용)

작성일: 2026-09-08 · 담당: A1 topic-positioning
조사 방식: WebSearch (WebFetch 는 이 세션의 egress 프록시에서 **전 도메인 차단** — 아래 "접근 불가" 참조)

---

## 1. 결론 (5줄 요약)

1. **주제 적합성: 매우 높음(확인).** JLK 의 대표 제품 **JBS-01K 는 DWI 기반 뇌경색 병변 분할 + 뇌졸중 유형(TOAST) 분류**이며, 본 프로젝트(SOOP/ds004889, DWI 분할 + 병인 분류)와 과제 구조가 1:1 로 대응한다.
2. **공고 원문은 확보 실패(잡코리아·김박사넷·회사 careers·rndjob 모두 프록시 차단).** 다만 검색 메타데이터로 "인공지능(AI) 빅데이터 연구원(전문연구요원), 석사 이상, 경력무관, 서울 강남구, 병역특례" 는 확인.
3. 다른 JLK 공고에서 확인된 자격요건은 **DL/ML/CV/의료영상 전공·학위 + 해당 분야 학회/학술지 publication + Python 또는 C/C++**, 우대는 **의료영상 top-tier publication / AI 의료 유경험 / EMR·PACS 경험**.
4. **가장 큰 갭 3가지: (P0) DICOM 입출력, (P0) 3D(또는 2.5D) 모델, (P0) 민감도·특이도 등 임상 성능지표 보고.** 현재 프로젝트는 NIfTI·2D U-Net·Dice 중심이라 제품 스펙과 어긋난다.
5. 포지셔닝은 "JLK 문제를 푼다"가 아니라 **"JBS-01K 와 동일한 과제 구조를 공개 데이터로 처음부터 끝까지 직접 돌려본 사람"** 으로 잡는 것이 정확하고 방어 가능하다.

---

## 2. 채용공고 조사 결과

### 2-1. 접근 시도 결과

| URL | 결과 |
|---|---|
| https://www.jobkorea.co.kr/Recruit/GI_Read/46756089 | **접근 불가** (egress 프록시 차단) |
| https://phdkim.net/board/postgraduate-career/23 | **접근 불가** |
| https://www.jlkgroup.com/careers/ | **접근 불가** |
| https://jlk.recruitin.co.kr/jobs | **접근 불가** |
| https://rndjob.or.kr/... | **접근 불가** |
| https://ai.pusan.ac.kr/ (JLK 채용 공지) | **접근 불가** |
| WebFetch 전반 (en.wikipedia.org 테스트 포함) | **전 도메인 차단** → 아래 내용은 모두 **검색엔진 요약 기반**, 원문 대조 불가 |

> ⚠️ 아래 2-2/2-3 은 검색 스니펫·요약에서 얻은 것이며 **공고 원문 대조를 하지 못했다**. 지원 직전 사용자가 원문을 직접 열어 재확인해야 한다.

### 2-2. 해당 공고 (확인된 범위)

| 항목 | 내용 | 확실도 |
|---|---|---|
| 공고명 | [제이엘케이] 인공지능(AI) 빅데이터 연구원 (전문연구요원) | 확인 |
| 마감 | 2025-02-28 (해당 회차) | 확인 |
| 고용형태 | 병역특례(전문연구요원) | 확인 |
| 학력 | 대학원 **석사 졸업 이상** | 확인 |
| 경력 | 무관 | 확인 |
| 근무지 | 서울 강남구 (AI R&D 센터) | 확인 |
| 직종 태그 | 데이터엔지니어 / 머신러닝엔지니어 / 앱개발자 / 솔루션·SI | 확인 |
| 키워드 | 빅데이터, 솔루션, 딥러닝, 인공지능 | 확인 |
| 담당업무 상세 문구 | **미확인** (원문 접근 불가) | — |
| 우대사항 상세 문구 | **미확인** | — |

### 2-3. JLK 의 다른 AI 연구직 공고에서 확인된 요건 (같은 조직 기준선으로 사용)

| 구분 | 내용 | 출처 성격 |
|---|---|---|
| 자격요건 | Deep Learning / Machine Learning / Computer Vision / **Medical Image** 관련 전공 및 학위 | AI알고리즘 연구원 공고 |
| 자격요건 | 해당 분야 **학회 및 학술지 publication** | 동상 |
| 자격요건 | **C/C++, Python 중 1개 이상** 사용 가능 | 동상 |
| 우대사항 | 의료영상 분석 관련 **top-tier 학회/학술지 publication** | 동상 |
| 우대사항 | **AI 의료 유경험자**, **뇌 관련 비즈니스 경험자**, **EMR/PACS** 등 병원 대상 경험 | 2023 공채 기사 |
| 전형 | 서류 → 실무진 면접 → 임원 면접 | 2023 공채 기사 |
| 조직 | 서울 AI R&D 센터 / 청주 본사 / 미국 Santa Clara / 일본 도쿄 법인 | 회사 소개 |

> 추정(근거 있음): "빅데이터 연구원" 명칭과 데이터엔지니어/ML엔지니어 태그로 보아, **영상 딥러닝 단독이 아니라 영상+임상 메타데이터(빅데이터) 결합 분석**이 업무 범위에 포함될 가능성이 높다. 본 프로젝트의 "병변 특징 + 임상 변수(NIHSS, mRS, 나이/성별) → 병인 분류" 구성은 이 해석에 잘 맞는다.

---

## 3. JLK 제품군 정리

**모두 언론 보도 기반이며, 식약처/FDA/PMDA 원 DB 대조는 하지 못했다(WebFetch 차단).**

| 제품 | 입력 모달리티 | 출력 | 인허가·급여 현황 | 확실도 |
|---|---|---|---|---|
| **JBS-01K** (뇌경색 유형분류) | **MRI DWI** (2D·3D 시각화 분석) | **뇌경색 병변 분할 + 뇌졸중 유형(TOAST: LAA/SVO/CE/Others) 분류** | 식약처 3등급 허가 2018-08(국내 최초 뇌경색 진단보조) / 혁신의료기기 **통합심사 통과 2022-12** → 2023-01 비급여 사용 / **비급여 수가 54,300원** / NECA 승인 / 일본·태국 시판허가 | 확인(보도) |
| **JLK-DWI** | MRI DWI | DWI 고신호강도 영역 검출 + **부피 측정** | 일본 PMDA 인허가 획득 (2025-04 보도) | 확인(보도) |
| **JLK-FLAIR** | MRI FLAIR | 고신호강도 영역 분석·시각화 + 부피 측정 | 일본 PMDA 인허가 획득 | 확인(보도) |
| **JLK-SWI** | 뇌 MRI SWI | 미확인 | 일본 PMDA 인허가 획득 | 부분확인 |
| **JLK-PWI** | MRI 관류영상(PWI) | ischemic core / hypoperfusion 영역 **정량 지표** | 일본 PMDA 인허가 획득 / FDA 신청 | 확인(보도) |
| **JLK-CTP** | CT 관류영상 | 허혈 코어·구제가능 영역 정량 → 재개통 시술 판단 지원 | **FDA 510(k) 2024-10** (JLK 3번째 FDA) / 일본 PMDA / 신의료기술평가 **유예 지정 2026-06** | 확인(보도) |
| **JLK-LVO** | **CTA** (CT 혈관조영) | 대혈관폐색(LVO) 검출 | **FDA 510(k) 승인** (뇌졸중 솔루션 중 첫 FDA) / 혁신의료기기 통합심사 통과 2025-04 | 확인(보도) |
| **JLK-ICH** | **NCCT** (비조영 CT) | 두개내출혈 검출 | **FDA 510(k) 2025-01** | 확인(보도) |
| **JLK-NCCT** | NCCT | 비조영 CT 기반 뇌졸중 분석 | **FDA 510(k) 2026-03** / 일본 PMDA 획득 | 확인(보도) |
| **JLK-CTL** | **NCCT** | 조영제 없이 **LVO 가능성 예측** (응급 선별) | 혁신의료기기 **통합심사 통과** | 확인(보도) |
| JLK-AILink / SDH 솔루션 | 미확인 | 미확인 | FDA clearance 보도 있음 | **미확인** |
| JBS-04K(뇌출혈), JBA-01K(뇌동맥류), JPC-01K(전립선암) | 미확인 | 미확인 | 혁신의료기기 통합심사 대상 언급 | 부분확인 |

### JBS-01K 보고 성능 (보도 기준, 원 논문 미대조)

| 지표 | 값 |
|---|---|
| 병변 검출 민감도 — 의료진 단독 | 74.6% |
| 병변 검출 민감도 — 의료진 + AI | 90.6% |
| 병변 검출 민감도 — AI 단독 | 98.0% |
| **병변 분할 DSC — 의료진 단독 → AI 병용** | **0.523 → 0.742** |
| 판독자가 놓친 사례 중 AI 추가 검출 | 79.6% |

> 실무적 의미: **DSC 0.74 대**가 JLK 가 대외적으로 인용하는 급성기 DWI 병변 분할 수준이다. 본 프로젝트의 2D U-Net 결과를 이 수치와 나란히 놓을 수 있으면(같은 데이터가 아니므로 "직접 비교 불가"를 명시한 채) 포트폴리오의 설득력이 크게 오른다.
>
> ⚠️ 상충 정보: 검색 요약에서 "NCCT 기반 두개내출혈 민감도 98.7% / 특이도 88.5%" 와 "JLK-NCCT 민감도 78.5% / 특이도 90.3%" 가 각각 나왔다. 어느 제품 수치인지 **원문 미확인** — 인용하지 말 것.

---

## 4. 요구 역량 ↔ 프로젝트 매핑

| JLK 요구 역량 | 근거 | 본 프로젝트에서 충족되는 요소 | 상태 |
|---|---|---|---|
| Medical Image 딥러닝 | 자격요건 | DWI(TRACE b1000)+ADC 2채널 U-Net 병변 분할 | ✅ 충족 |
| **뇌졸중 도메인** | 우대(뇌 관련 경험) | 급성 뇌경색, NIHSS/mRS/TOAST 병인 | ✅ 충족(핵심 강점) |
| 병변 정량화(부피 산출) | JLK-DWI/FLAIR 제품 핵심 출력 | 분할 마스크 → 병변 부피 계산 | ✅ 충족 |
| 영상+임상 데이터 결합 분석 | "빅데이터 연구원" 직무 성격 | 병변 특징 + 나이/성별/NIHSS → 병인 분류 | ✅ 충족 |
| Python | 자격요건 | 전체 파이프라인 Python | ✅ 충족 |
| 재현성/실험관리 | 연구직 기본 | seed 고정, 환자 단위 split, runs/log.jsonl | ✅ 충족 |
| **TOAST 유형 분류** | JBS-01K 출력 | LAA/SVO/CE/Others 분류 | ✅ 충족(가장 차별적) |
| **3D 볼륨 분석** | JBS-01K "2D·3D 분석" 명시 | 2D U-Net 만 계획 | ❌ **갭 P0** |
| **DICOM 입출력** | 제품은 PACS 상 DICOM 처리 | ds004889 는 NIfTI | ❌ **갭 P0** |
| **민감도/특이도 기반 성능보고** | 인허가·임상 보고 관례 | Dice 중심 | ❌ **갭 P0** |
| 외부/다기관 일반화 검증 | 인허가 요건 | 단일 데이터셋 내부 split | ❌ 갭 P1 |
| 추론 속도/응급 워크플로우 | "골든타임" 제품 특성 | 미측정 | ❌ 갭 P1 |
| CT 계열(NCCT/CTA/CTP) | 제품군 절반 이상 | 없음 | ⚠️ 갭 P2(범위 밖으로 두는 게 합리적) |
| PACS/EMR 연동 | 우대사항 | 없음 | ⚠️ 갭 P2 |
| MLOps/배포 | 산업 R&D 일반 | 없음 | ⚠️ 갭 P2 |
| 학회/학술지 publication | **자격요건(필수 문구)** | 개인 프로젝트로 대체 불가 | ⚠️ **구조적 갭** |

### 갭 우선순위와 권장 조치

**P0 — 반드시 넣을 것 (비용 대비 효과 최고)**
1. **DICOM 처리 최소 1회 경유.** NIfTI 원본을 pydicom 으로 DICOM 시리즈 write → 다시 read → 전처리 파이프라인 투입하는 짧은 경로를 만든다. "DICOM 을 다뤄봤다"는 진술이 사실이 된다.
2. **2.5D 또는 얕은 3D 변형 1개 추가.** 인접 슬라이스 스택(2.5D) 만으로도 "3D 문맥을 고려했다"가 성립하고, 2D 대비 성능 비교표가 그대로 실험 설계 능력의 증거가 된다.
3. **평가지표에 민감도·특이도·환자 단위 검출률 추가.** Dice 만으로는 제품/인허가 언어와 접점이 없다. JBS-01K 보도치(민감도, DSC)와 같은 축을 쓰면 대화가 된다.

**P1 — 여력 되면**
4. 병인 분류를 **클래스 불균형 + 다중 클래스 지표(macro-F1, per-class recall)** 로 보고. Cryptogenic 처리 방침(제외/Others 병합)을 명시.
5. 추론 시간(케이스당 초) 측정 1줄 보고.
6. 사이트/스캐너 정보가 participants.tsv 에 있으면 **site-holdout 일반화** 실험 (A2 와 협의).

**P2 — 하지 말 것 / 문서로만 언급**
7. CT 모달리티 확장, PACS 연동, 풀 MLOps 는 개인 프로젝트 범위를 넘고 완성도만 떨어뜨린다. README 의 "한계 및 향후" 절에 **의도적으로 제외했다고 명시**하는 편이 낫다.
8. 인허가 문서 모사(임상성능시험 계획서 흉내)는 **권장하지 않음** — 허위성 오해 소지. 대신 "성능 보고 형식을 인허가 관례에 맞췄다" 수준으로만.

---

## 5. README 상단 포지셔닝 문구 (제안)

> **strokeai — DWI 기반 급성 뇌경색 병변 분할 및 뇌졸중 병인 분류**
>
> 공개 데이터셋 **OpenNeuro SOOP (ds004889, CC0)** 의 급성 뇌졸중 환자 DWI(TRACE b1000)·ADC 영상과 임상 메타데이터를 사용해, (1) 급성 뇌경색 병변을 자동 분할하고 병변 부피를 산출한 뒤, (2) 병변 특징과 임상 변수(나이·성별·NIHSS)로 뇌졸중 병인(LAA / CE / SVO / Others)을 분류하는 파이프라인을 처음부터 끝까지 구현한 **개인 학습 프로젝트**입니다.
>
> 이 과제 구성은 국내 의료 AI 기업 제이엘케이(JLK)의 **JBS-01K**(DWI 기반 뇌경색 병변 분할 + 뇌졸중 유형 분류)와 같은 문제 구조를 가지며, 본 저장소는 해당 유형의 문제를 직접 다뤄본 경험을 보이기 위한 것입니다. **상용 제품의 재현이나 성능 비교가 목적이 아니며**, 임상적 사용을 의도하지 않습니다.
>
> 모든 실험은 환자 단위 분할, 고정 시드, 테스트셋 1회 평가 원칙 아래 수행되었고, 실행 로그와 실패한 시도까지 `docs/` 에 남겨 두었습니다.

작성 시 지킨 원칙: ① "공개 데이터·개인 프로젝트" 를 첫 문단에 명시 ② JLK 제품은 **문제 구조의 참조점**으로만 언급하고 성능 우위를 주장하지 않음 ③ 임상 사용 부인 문구 포함 ④ 재현성 원칙을 마지막 문장에 배치해 연구직 평가자가 볼 지점을 만든다.

---

## 6. 남은 확인 필요 사항 (사용자 직접 확인 권장)

1. 잡코리아 46756089 공고 **원문의 담당업무·우대사항 문구** — 이 보고서의 최대 공백.
2. 자격요건의 "학회/학술지 publication" 이 **필수인지 우대인지** — 필수라면 지원 전략 자체가 달라진다.
3. JBS-01K 성능 수치의 **원 논문**(Journal of Stroke 등) — 인용하려면 원문 필요.
4. participants.tsv 의 사이트/스캐너 변수 유무 (A2 담당).

---

## 7. 출처

- https://www.jobkorea.co.kr/Recruit/GI_Read/46756089 — **접근 불가**(검색 메타데이터만)
- https://www.jobkorea.co.kr/company/43403590/recruit — 접근 불가
- https://phdkim.net/board/postgraduate-career/23 — **접근 불가**
- https://www.jlkgroup.com/careers/ — **접근 불가**
- https://www.jlkgroup.com/en/about/ — 접근 불가(검색 요약만)
- https://jlk.recruitin.co.kr/jobs — 접근 불가
- https://ai.pusan.ac.kr/ai/60711/subview.do — 접근 불가
- https://cse.pusan.ac.kr/bbs/cse/2616/718513/download.do (JLK 상반기 채용 안내 PDF) — 접근 불가, 검색 요약으로 자격요건 확보
- https://www.rndjob.or.kr/info/eview01.asp?gno=00162420 / gno=00154153 — 접근 불가
- https://www.rocketpunch.com/companies/jlk-1/jobs — 접근 불가
- https://www.saramin.co.kr/zf_user/company-info/view/csn/dGJHczlpN0NvYWNkaEhUbGRXZXdKUT09 — 검색 요약
- https://www.asiae.co.kr/article/2023121109472147892 (JLK 2023 공채, 우대사항·전형절차)
- https://www.asiae.co.kr/article/2022122315054860358 (JBS-01K 혁신의료기기 선정)
- https://www.monews.co.kr/news/articleView.html?idxno=327529 (JBS-01K 건강보험 수가)
- https://www.monews.co.kr/news/articleView.html?idxno=327851 (JBS-01K NECA 승인)
- https://www.hankyung.com/article/202310310308i (뇌졸중 AI 유형분류 수가 54,300원)
- https://www.dailymedi.com/news/news_view.php?ca_id=22&wr_id=921463 (JBS-01K 성능: 민감도/DSC)
- https://www.k-health.com/news/articleView.html?idxno=37536 (2018 식약처 3등급 허가)
- https://www.hkbiocon.com/bbs/board.php?bo_table=news&wr_id=446 (JBS-01K 일본·태국 시판허가)
- https://www.pharmnews.com/news/articleView.html?idxno=251168 (JLK-LVO FDA)
- https://www.hankyung.com/article/202410171435i (JLK-CTP FDA, 3번째 승인)
- https://www.pharmnews.com/news/articleView.html?idxno=256456 (JLK-ICH FDA)
- https://www.hankyung.com/article/202603267491P (JLK-NCCT FDA 510(k))
- https://pharm.edaily.co.kr/news/read?newsId=02673206642337512 (JLK-CTL 혁신의료기기 통합심사)
- https://www.biotimes.co.kr/news/articleView.html?idxno=20931 (JLK-LVO 혁신의료기기)
- https://www.biotimes.co.kr/news/articleView.html?idxno=20566 (JLK-DWI 일본 PMDA)
- https://www.biotimes.co.kr/news/articleView.html?idxno=20481 (JLK-FLAIR 일본 PMDA)
- https://www.thebionews.net/news/articleView.html?idxno=18518 (JLK-SWI 일본 PMDA)
- https://www.hankyung.com/article/202409059120P (JLK-CTP·PWI 일본 PMDA 신청)
- https://www.mt.co.kr/thebio/2026/06/09/2026060909592644763 (JLK-CTP 신의료기술평가 유예)
- https://grand-challenge.org/aiforradiology/company/jlk-inc/ (JLK 제품 인허가 요약, 검색 요약)
- https://welldone-interview.co.kr/interview/jlk-interview-questions (JLK 면접 질문 정리, 검색 요약 — 3자 제작물이므로 사실 근거로는 미채택)

> 모든 WebFetch 시도가 egress 프록시에서 차단되어, 위 언론 기사들은 **검색엔진 요약 경유**로만 확인했다. 인용 수치를 외부에 제시하기 전 원문 재확인 필요.

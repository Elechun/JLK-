# 서버의 새 Claude Code 채팅에 붙여 넣을 인수인계 프롬프트

아래 블록을 그대로 첫 메시지로 붙여 넣으세요.

```text
이 저장소는 제이엘케이(JLK) 전문연구요원 지원용 개인 포트폴리오 프로젝트다. 먼저 CLAUDE.md, README.md,
docs/00_project_charter.md, docs/01_server_runbook.md, docs/agents/A1_report.md, docs/agents/Astra_report.md 를 읽어라.

현재 상태: 코드·에이전트 정의·A1/Astra 보고서·단위 테스트(15개 통과)·CPU 스모크 테스트까지 끝났고,
전체 데이터 학습과 A2~A6 에이전트 실행은 아직이다. 이 서버에서 이어서 진행한다.

해야 할 일 (docs/01_server_runbook.md 3절 순서 그대로):
1. 환경 확인: python -c "import torch; print(torch.cuda.is_available())"; pip install -e .[dev]
2. bash scripts/run_all.sh 의 1~5단계(다운로드→인덱스→split→전처리→pytest)를 실행하고,
   data/splits.json 의 sha256 이 0f923acd2f8898d58c4c62cb598c4b8d88c17ab5eaab8a72ee4e73a7342d8f43 인지 확인해라.
3. .claude/agents/ 의 에이전트를 순서대로 위임해 보고서를 만들어라:
   a2-data-suitability → a3-preprocessing-stats → a4-model-optimization → (학습 실행) → a6-goals-diagnostics
   → a5a-methodology-review-external (Codex 가 있으면 codex exec 로, 없으면 Claude 로 실행하고 보고서에 실행 모델 표기)
   → a5b-methodology-review-internal. 각 보고서는 docs/agents/<ID>_report.md.
4. 확정된 수정을 반영하고 테스트를 다시 돌린 뒤, 필요하면 재학습.
5. 테스트셋 평가는 A6 사인오프 후 단 1회: scripts/eval_seg.py --split test, train_cls.py --final.
6. README 의 에이전트 표 "상태 / 주요 성과" 열과 docs/findings_log.md 를 실제 결과로 채우고,
   a1-topic-positioning 과 astra-direction-review 로 README 를 최종 검토해라.

규칙: 테스트셋은 5단계 전까지 절대 읽지 않는다. split 은 환자 단위. 보고서에는 실행해 확인한 수치만 쓴다.
에이전트 모델은 A1~A4·A6 = Opus, A5a = Codex(GPT), A5b·Astra = Fable 을 유지한다.
작업은 브랜치 claude/medical-ai-agent-project-v2vrkv 에서 이어서 하고 단계마다 커밋해라.
```

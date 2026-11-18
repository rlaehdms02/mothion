
## 6. 팀 번호 유지 버전 (최종)
- 파일명을 팀 스토리보드 번호(A001 다리 뻗기, A002 팔 뻗기, A003 힙힌지)로 고정하여 재실행
- 24 Clip 모두 AUTO_PASS → PASS → Leakage check: OK
- 교재 csv를 수정하지 않았으므로 action_label은 교재 이름(bend_return / leg_raise_lower / walk_turn_walk)으로 기록됨 → 번호로만 해석
- A002(팔 뻗기)는 교재 leg_raise_lower 기준(다리 관절)으로 QA되어 손목 품질은 검사되지 않음
  (A006 기준 실행에서는 손목 포함 low 최대 0.034로 문제 없음을 확인)

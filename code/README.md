# 코드 구성

기존 main/sub와 실험 구현은 code/old에 원본 그대로 보관했습니다.
code/utils에는 캐시 생성, 인코더, 기하 분해, 결과 통합, 정렬 loss를 남겼습니다.
code/main 및 code/sub는 두 paper의 새 매핑에 맞춰 작성할 위치입니다.
현재 실행 파일은 old/main 및 old/sub에 보관되어 있습니다.
데이터·가중치·캐시·results는 기존 위치를 유지합니다.

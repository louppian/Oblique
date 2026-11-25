# 공통 코드

- build_embedding_cache.py: CLIP/CARZero × COCO/NIH의 image, text, noise 캐시 생성. 이미지 배치 256, 텍스트 배치 1024.
- encoders.py: 모델 로딩·전처리·임베딩 추출과 NIH 프롬프트.
- geometry.py: 평균·노이즈 평면 추정과 `u = alpha*rhat + beta*bhat + v`의 3항 분해. 전체 잔차 v는 두 기준 방향에 직교합니다.
- output.py: 세부 결과를 CSV 한 개와 PNG 한 개로 통합 저장.
- positive_analysis.py: 세 항의 크기·분산·점수 기여, 평균–노이즈 평면 기준 코사인 범위, 성분 제거 후 재정규화 분석.
- coordinate_geometry.py: conformity, imageness, 평면 밖 잔차 크기 계산.

기존 실험·학습·검증 코드와 이전 main/sub는 ../old에 보관합니다.

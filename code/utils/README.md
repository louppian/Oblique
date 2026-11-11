# 공통 코드

- build_embedding_cache.py: CLIP/CARZero × COCO/NIH의 image, text, noise 캐시 생성. 이미지 배치 256, 텍스트 배치 1024.
- encoders.py: 모델 로딩·전처리·임베딩 추출과 NIH 프롬프트.
- geometry.py: 평균·노이즈 평면과 평면 밖 최대 분산 방향 추정, 네 항 분해.
- output.py: 세부 결과를 CSV 한 개와 PNG 한 개로 통합 저장.
- residual_alignment_loss.py: 기존 평균–노이즈 평면 밖 질환 방향 정렬 loss. 새 정의의 e 선택과 별도로 유지한 기존 loss 함수.

기존 실험·학습·검증 코드와 이전 main/sub는 ../old에 보관합니다.

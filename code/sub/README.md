# 보조분석 실행

각 번호의 Python 파일을 직접 실행한다. 설정과 경로는 코드에 포함되어 있으며, 경로는 저장소 root를 기준으로 해석한다. 각 실행 결과는 `results/sub/<코드 이름>.png`와 같은 이름의 `.csv`에 저장한다.

| 코드 | outline | 분석 |
|---|---|---|
| 01_domain_shift.py | 5.1 / S1 | COCO–NIH 도메인 차이, 기존 손상 캐시, 합성 비의료 텍스트, MNIST |
| 02_image_examples.py | 5.2 / S2 | 노이즈 근접도 분위수 및 동일 conformity 구간의 실제 이미지 |
| 03_attack_detection.py | 5.3 / S3 | 기존 exp09의 ResNet34–PGD, 예산 1·2·4·8/255, 탐지와 좌표 변화 |
| 04_negation_geometry.py | 5.4 / S4 | 14개 질환 × 10개 문형의 긍정→부정 이동 분해 |
| 05_negation_score_decomposition.py | 5.4 / S5 | POS·NEG·PNC, 네 항 점수 기여, 성분 제거, 질환 분리 |
| 06_medical_model_comparison.py | 5.5 / S6 | CARZero·Bi-MCQ 전역 인코더의 공통/자체 노이즈 기준 기하 |

`code/utils/supplementary_analysis.py`가 실행 절차를 제공한다. 네 항은 main과 동일한 `geometry.fit_frame`, `geometry.components`를 호출한다. 평균–노이즈 평면 밖 최대 분산 한 축을 e로, 나머지를 delta로 정의한다. main/05와 동일하게 seed 0·1·2, 기준 이미지 2,000장, 나머지 평가 영상을 사용한다. 이미지 사례는 seed 0을 사용한다. 공격은 기존 프로토콜의 기준 200장·평가 500장, ResNet34–PGD 예산 4개를 사용한다.

부정 분석에서 질환 양성을 양성 라벨로 두며 POS=z·t+, NEG=−z·t−, PNC=z·(t+−t−)를 사용한다. AUC는 연속 점수로 계산한다. F1·MCC·Recall의 임계값은 기준 2,000장에 대한 Youden 기준으로 정하고 평가 집합에 고정한다. PNC 임계값 0과 기준 평균 위치를 뺀 PNC도 별도로 기록한다. 기준 집합에 한 클래스만 있으면 보정 임계값 및 해당 F1·MCC·Recall은 결측으로 기록한다. 성분 제거 후 재정규화한다. 평면 밖/델타 공간 투영은 이미지와 텍스트에 동일한 이미지 기준 투영을 적용한다. 질환–문형 단위 및 질환 단위 상관을 구분하고 BH 보정값을 기록한다.

Bi-MCQ는 외부 소스의 `build_BiMCQ_model` 및 제공된 YAML을 사용하고 최종 체크포인트의 전체 모델 키를 strict 로딩한다. 최종 가중치로 덮이는 초기 M3AE 가중치 읽기는 생략한다. 같은 NIH 영상·문장·노이즈 입력을 사용하며 생성한 Bi-MCQ 임베딩은 메모리에서 처리한다. 공통 노이즈 기준은 CARZero 노이즈이고, 자체 기준에서는 각 모델의 노이즈 방향을 사용한다.

MNIST는 `data/mnist`, Tiny-ImageNet은 `data/tiny-imagenet-200/val`에서 읽는다. 입력이 빠진 조건은 CSV의 availability 행과 그림 제목에 표시한다. 현재 로컬 실행은 COCO·NIH와 합성 텍스트를 대상으로 수행하며 MNIST·Tiny-ImageNet 조건은 입력 준비 후 같은 코드를 재실행한다. 5.6의 확장 학습은 별도 실험이다.

07_alignment_uncertainty.py는 CARZero only encoder, CARZero cross attention, CARZero only encoder + delta alignment를 비교한다. NIH 공식 test 환자와 별도 학습·검증 환자를 사용하며, seed 설정 없이 10 epoch를 실행한다. 구현은 code/utils/medical_alignment_uncertainty.py, 실행·loss·uncertainty 정의는 alignment_protocol.md에 있다. 결과는 results/sub/07_alignment_uncertainty.png 및 .csv, 선택한 텍스트 가중치는 ckpt/carzero_encoder_delta_alignment.pt에 저장한다.

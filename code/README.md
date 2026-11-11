# 논문 실험 실행 코드

프로젝트 root에서 아래 Python 파일을 직접 실행합니다. CLI 옵션 없이 각 파일 위쪽 CONFIG를 사용합니다. main과 sub의 각 실행 파일은 `results/main/<코드명>.csv`, `.png` 또는 `results/sub/<코드명>.csv`, `.png`만 결과물로 남깁니다. 여러 세부 표는 CSV의 analysis/table 열로 구분하고 그림은 하나의 PNG 안에 배치합니다. 반복 실행은 해당 코드의 결과 두 파일을 갱신합니다.

## Main Experiment

| 논문 절 | 실행 파일 | 내용 |
|---|---|---|
| 4.1 | main/01_noise_geometry.py | 이미지·텍스트 노이즈 집중, 평균–노이즈 배치, 축 기하 |
| 4.2 | main/02_component_magnitude.py | 2×2 공통 캐시에서 네 항 크기·분산·교차항·코사인 범위 |
| 4.3 | main/03_alignment_retrieval.py | CLIP-only/conformity/imageness 학습과 네 residual 분석을 한 번의 학습에서 계산 |
| 4.4 | main/04_medical_negation.py | NIH 캐시에서 POS/NEG/PNC, 평균 위치, 네 항 질환 분리, 부정 이동 |

## Sub Experiment

| 논문 절 | 실행 파일 | 내용 |
|---|---|---|
| 5.1 | sub/01_noise_stability.py | 노이즈 분포·표본 수·seed, 문자·단어 비교 |
| 5.2 | sub/02_corruption_paths.py | 손상 궤적과 같은 기울기의 무작위 기준점 대조 |
| 5.3 | sub/03_domain_shift.py | COCO/NIH 교차 도메인, 손상, MNIST·비의료 텍스트 |
| 5.4 | sub/04_mean_shift.py | 축 정렬 측정과 평균 이동 분석 |
| 5.4 | sub/04_reference_controls.py | COCO 캐시에서 같은 차원의 random/PCA 및 잔차 점수 대조 |
| 5.5 | sub/05_attack_detection.py | 기존 공격 종류·분류기의 탐지 및 예산별 기하 |
| 5.6 | sub/06_medical_model_comparison.py | 기존 CARZero/Bi-MCQ 체크포인트의 전역 인코더 기하·문형별 비교 |
| 5.6 | sub/06_medical_alignment_training.py | 기존 질환 잔차 정렬 loss로 텍스트 인코더 학습, 환자 분할 평가 |

## 실행 예시

```bash
python code/utils/build_embedding_cache.py
python code/main/02_component_magnitude.py
python code/main/04_medical_negation.py
python code/main/03_alignment_retrieval.py
```

캐시는 CLIP·CARZero × COCO·NIH 각 폴더에 image.npz, text.npz, noise.npz를 사용합니다. main/02, main/04, sub/04_reference_controls는 이 캐시를 직접 읽습니다. 노이즈 분포·손상·공격 실험은 기존 인코더 및 조건별 캐시 함수를 사용합니다. main/03은 가중치를 업데이트하므로 학습 시점별 임베딩을 계산합니다.

## 공통 계산과 기존 코드

공유 구현은 code/utils에 있습니다. 기존 1_analysis.py, 2_application.py, 3_alignment.py 등의 함수와 클래스는 유지하고 실행 진입점을 main/sub로 분리했습니다. residual_core.py는 기존 새 정의의 학습·분해 구현입니다. experiment_runner.py는 세부 결과를 통합하고 cached_analysis.py는 공통 캐시의 네 항 분석을 수행합니다.

main/sub 실행 중 세부 CSV·PNG는 cache/work의 임시 폴더에 생성된 후 통합되고 정리됩니다. 의료 학습의 재개·평가용 checkpoint와 학습 메타데이터는 cache/residual_alignment_training에 보관됩니다. utils의 캐시 생성 코드는 NPZ를 저장하는 준비 단계입니다.

## 평가 규칙

- main/02와 main/04의 기준 이미지와 평가 이미지는 seed별로 분리합니다. COCO 텍스트는 이미지 ID로 함께 나눕니다. NIH 텍스트 축은 고정된 긍정 문형 집합으로 추정하고 CSV에 reference 정책을 기록합니다.
- main/04의 F1·MCC는 고정 임계값 0으로 계산하며 threshold_policy 열에 기록합니다. AUC는 질환 양성 라벨을 사용합니다. NEG는 부정 점수에 음수를 취해 질환 양성 방향으로 맞춥니다.
- 의료 모델 비교는 기존 전역 임베딩 분석입니다. 별도의 cross-attention 출력 비교나 Bi-MCQ와 정렬 loss의 공동 재학습 결과는 이 코드의 출력에 포함되어 있지 않습니다.
- sub/04_reference_controls는 학습 전 고정 임베딩 대조입니다. 학습에 따른 initial_fixed/epoch_current 차이는 main/03에서 평가합니다.

## 자산

가중치는 ckpt, 데이터는 data, 외부 코드는 third_party 아래에서 읽습니다. CARZero baseline은 ckpt/best_model.ckpt, Bi-MCQ 비교는 ckpt/BiMCQ_CARZero_best_model.ckpt 및 기존 모델 config를 사용합니다. 도메인 보조분석은 data/mnist의 IDX 자료, 공격 분석은 data/tiny-imagenet-200과 분류기 가중치를 추가로 사용합니다. 새 정의를 위해 별도의 new definition 폴더를 사용하지 않습니다.

## 검증

```bash
python code/utils/verify_layout.py
python code/utils/verify_residual.py
```

검증은 작은 합성 임베딩에서 수치 재구성, 의료 지표, 부분공간 대조와 CSV·PNG 한 쌍 출력 규칙을 확인합니다. 실제 모델 학습 및 전체 데이터 성능 평가는 해당 실행 파일로 수행합니다.

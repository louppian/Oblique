# CARZero 정렬 학습과 uncertainty 비교

실행 파일은 `code/sub/07_alignment_uncertainty.py`이며 인자를 붙이지 않고 실행한다. 구현은 `code/utils/medical_alignment_uncertainty.py`에 있다. 모든 경로는 저장소 root를 기준으로 해석한다.

| 조건 | 가중치와 학습 | 평가 점수 |
|---|---|---|
| CARZero only encoder | `ckpt/best_model.ckpt`의 이미지·텍스트 인코더 | 정규화된 전역 임베딩의 내적 |
| CARZero cross attention | 같은 체크포인트의 이미지·텍스트 인코더 및 양방향 fusion head | 원본 i2t·t2i 출력의 평균 |
| CARZero only encoder + alignment | 이미지 인코더 고정, 텍스트 인코더에 δ 정렬 loss 적용 | 정규화된 전역 임베딩의 내적 |

NIH 공식 test 환자를 test에 유지하고 나머지 환자의 15%를 검증에 배정한다. 무작위 seed 설정과 반복 실행을 생략하고, 한 번 생성한 환자 분할을 `cache/carzero/nih/alignment_split.npz`에 보관한다. 이미지 전역 임베딩과 cross-attention 점수도 같은 캐시 폴더에 저장한다.

학습 영상의 원본 CARZero 임베딩과 기존 노이즈 캐시로 네 항의 기준을 추정한다. 세 조건 모두 이 기준을 공유한다. $Q_3$의 열은 $\operatorname{span}(\hat r,\hat b)$의 직교 기저 두 개와 평면 밖 첫 번째 PCA 방향이며, $P_\delta=I-Q_3Q_3^\top$이다.

$$a_d=P_\delta\left(\mathbb E_\mathrm{train}[z\mid y_d=1]-\mathbb E_\mathrm{train}[z\mid y_d=0]\right),\quad w_{d,k}=P_\delta(t^+_{d,k}-t^-_{d,k}).$$

$$L_\mathrm{align}=\frac{1}{140}\sum_{d,k}\left(1-\frac{a_d^\top w_{d,k}}{\|a_d\|\|w_{d,k}\|}\right).$$

AdamW, learning rate $10^{-5}$, weight decay $0.01$, 문항 쌍 batch 14로 총 10 epoch를 학습한다. 매 epoch 모든 140개 문항 쌍을 사용한다. 검증 문항별 PNC AUC 평균으로 최적 epoch를 선택하며 초기 상태도 후보에 포함한다. 선택한 텍스트 가중치와 기준은 `ckpt/carzero_encoder_delta_alignment.pt`에 저장한다.

각 조건에서 POS는 긍정 점수, NEG는 부정 점수의 음수, PNC는 긍정−부정 점수이다. 질환별 양성 라벨을 공통으로 사용한다. 검증 Youden 임계값을 test에 적용하여 F1·MCC·recall을 계산한다. CSV에는 문항별 성능과 10개 문형 평균 점수의 질환별 성능을 함께 기록한다. 그림은 후자의 14개 질환 macro 평균을 사용한다.

Uncertainty 분석은 PNC 오분류에 대한 AUROC와 risk–coverage로 평가한다. 입력 지표는 $-h$, 학습 분포에서의 $\alpha$, $\beta$, $e$ 좌표 편차, 질환 방향 밖의 δ 크기, 네 항의 점수 상쇄도, 검증 임계값에서의 거리를 사용한다. 수치가 클수록 uncertainty가 크도록 정의한다. AUROC 0.5 초과는 해당 점수가 오분류와 같은 방향으로 연결됨을 의미한다. AURC는 작을수록 오분류가 높은 uncertainty 영역에 모인다.

네 항 상쇄도는 $1-|\sum_j g_j|/\sum_j|g_j|$이다. Cross-attention 조건에서는 인코더의 네 항 지표가 fusion 이후 오분류와 연결되는지를 평가한다. 인코더 내적에 대한 정확한 네 항 분해와 nonlinear fusion 출력의 예측을 구분해 CSV의 score_space에 기록한다. 여기서 측정하는 것은 오분류와 기하 지표의 관계이며 epistemic·aleatoric uncertainty의 추정에는 별도의 관측·모형 가정이 필요하다.

최종 분석 산출물은 `results/sub/07_alignment_uncertainty.png`와 `.csv`이다. CSV에는 실제 환자 분할, 10 epoch 학습 이력, 선택 epoch, 성능, 방향 정렬, uncertainty 평가를 포함한다.

# Uncertainty 분해 실행

`09_uncertainty_decomposition.py`를 인자 없이 실행한다. 결과는 `results/sub/09_uncertainty_decomposition.png`와 `.csv`이다. 모델과 표본은 CARZero encoder, CARZero cross attention, CARZero encoder + 정렬의 세 조건에서 공유한다. 기존 학습·검증·test 환자 분할과 10 epoch 정렬 가중치를 사용한다.

## 모델 예측 분해

기존 텍스트 dropout과 cross-attention fusion dropout을 활성화하여 20회 예측한다. 이미지 인코더의 dropout 확률을 실행 중 점검하며 현재 체크포인트 구성은 0이다. 이 MC 근사는 활성화된 텍스트·fusion dropout이 만드는 모델 변동을 대상으로 한다. 난수 seed 설정은 생략한다. 이는 seed별 재학습과 구분되는 stochastic forward sampling이다.

질환별 10개 긍정−부정 점수를 평균한 PNC 점수에 검증 자료로 학습한 비음수 기울기의 sigmoid 보정을 적용한다. 모든 MC draw에 같은 보정을 사용한다. 기존 검증 집합은 정렬 epoch 선택에도 사용됐다. Test 표본은 보정에 사용되는 자료와 환자 단위로 분리된다.

$$\bar p=\mathbb E_m p_m,\qquad U_{\mathrm{total}}=H(\bar p),\qquad U_{\mathrm{conditional}}=\mathbb E_m H(p_m),\qquad U_{\mathrm{MC}}=H(\bar p)-\mathbb E_m H(p_m).$$

`epistemic_mc`는 MC dropout 근사에서의 모델–예측 mutual information이다. `expected_conditional_entropy`는 해당 예측 모형에서의 평균 조건부 엔트로피이며, aleatoric의 모델 기반 근사로 해석한다. 고유한 생물학적·판독자 uncertainty의 분리는 그에 맞는 관측·라벨 모형을 추가로 요구한다. Bernoulli 예측 분산에서도 $\operatorname{Var}_m(p_m)+\mathbb E_m[p_m(1-p_m)]=\bar p(1-\bar p)$를 확인한다. 10회와 20회 추정값의 차이도 기록한다.

## 관측 변동

NIH test 1,024장 중 128장에 Gaussian pixel noise 표준편차 0.05, 0.1, 0.2를 각각 5회 적용한다. 기존 resize 이후, normalize 이전의 [0,1] grayscale pixel에 노이즈를 넣고 [0,1]로 clip한다. 비교군은 같은 128장의 원본이다. $k$는 관측 realization, $m$은 dropout draw이다.

$$H(\mathbb E_{k,m}p_{mk})=\mathbb E_{k,m}H(p_{mk})+\mathbb E_k\left[H(\mathbb E_m p_{mk})-\mathbb E_mH(p_{mk})\right]+\left[H(\mathbb E_{k,m}p_{mk})-\mathbb E_kH(\mathbb E_mp_{mk})\right].$$

세 항을 평균 조건부 엔트로피, 관측별 모델 변동, 관측 realization 간 정보량으로 기록한다. 합의 수치적 일치를 확인한다. 관측 항은 지정한 Gaussian 관측 모형에 대한 민감도이다. 유한 MC 표본의 추정 변동도 포함될 수 있다.

관측 정보량에는 유한 MC 표본으로 각 관측의 평균 확률을 추정한 오차가 섞인다. 20개 draw 전체의 값 $J_{20}$과 서로 겹치지 않는 두 10개 draw 집합의 평균 $\bar J_{10}$을 구하여 $2J_{20}-\bar J_{10}$를 일차 편향 보정 추정값으로 함께 기록한다. 이 보정은 매끄러운 함수에서 편향이 표본 수의 역수에 비례한다는 근사를 사용한다. 작은 음수 추정값은 유한 표본 오차를 드러내도록 유지한다. 원래 합의 항등식은 보정 전 항들에 적용한다. 그림의 관측 정보량 패널에는 보정 추정값을 표시한다.

## OOD

같은 NIH test 1,024장과 COCO 256장, uniform grayscale noise 256장을 비교한다. COCO와 noise의 NIH 질환 라벨은 미지정으로 유지한다. 세 uncertainty 값의 OOD AUROC, 평균 확률, 확신도 0.95 이상 비율을 확인한다. 비교 표본과 확률 보정은 세 모델에 공통으로 적용한다. 이미지의 네 항 좌표는 공통으로 기록한다.

## 네 항과 모델 변동

이미지 기준을 고정하고 각 dropout draw의 텍스트 차이에 대해 네 항 점수 $g_r,g_b,g_e,g_\delta$를 구한다. 분산과 공분산을 계산하여 $\operatorname{Var}(\sum_jg_j)=\sum_j\operatorname{Cov}(g_j,\sum_kg_k)$를 확인한다. Cross-attention의 nonlinear fusion 점수는 별도로 평가하며 이 내적 항등식은 encoder 점수에 적용한다.

## 기준 추정 변동

전체 학습 환자를 cluster 단위로 100회 bootstrap하고 노이즈 임베딩도 독립적으로 재표집한다. 각 반복에서 $\hat r,\hat b,e$를 다시 추정한다. 원래 기준과의 각도, 좌표의 표준편차·95% percentile 구간, 각 항 점수의 변동을 기록한다. 고정된 encoder에서 네 항의 합이 원래 내적과 일치하는지도 확인한다. 이 과정은 분해 기준의 추정 안정성을 평가한다.

## 산출물과 캐시

각 평가 영상·질환의 확률과 uncertainty, 집단별 요약, OOD AUROC, 관측 분해, 네 항의 공분산, bootstrap 결과를 하나의 CSV에 저장한다. 특징·MC draws·bootstrap 캐시는 `cache/carzero/nih/uncertainty_decomposition`에 보관하여 결과 집계 시 재사용한다.

MC dropout 근거: [Gal & Ghahramani, 2016](https://proceedings.mlr.press/v48/gal16.html).

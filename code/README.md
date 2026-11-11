# 주분석 실행

root의 code/main에 있는 번호별 Python 파일을 직접 실행합니다. 각 파일은 results/main에 같은 이름의 CSV 한 개와 PNG 한 개를 저장합니다.

| 코드 | 분석 |
|---|---|
| 01_noise_concentration.py | 이미지·텍스트 노이즈 집중과 기준 방향 안정성 |
| 02_distribution_geometry.py | 평균–노이즈 평면의 이미지·텍스트 분포와 기하 |
| 03_corruption_response.py | 손상 전후 c, n, h 변화 |
| 04_reference_controls.py | 같은 conformity·같은 기울기의 무작위 기준점 대조 |
| 05_positive_score_decomposition.py | 새 네 항의 크기·분산·코사인 범위·긍정 점수 분해·성분 제거 |

분해는 u = alpha*rhat + beta*bhat + e_u + delta_u를 사용합니다. rhat과 bhat은 참조 집합에서 추정한 고정 방향이며, e_u는 평균–노이즈 평면 밖 최대 분산 방향으로의 투영입니다. 나머지는 delta_u입니다. 코드의 공통 좌표 c, n, h는 code/utils/coordinate_geometry.py에, 네 항 정의는 code/utils/geometry.py에 있습니다.

05는 code/utils/positive_analysis.py에서 CLIP·CARZero × COCO·NIH의 2×2 조합, seed 0·1·2를 계산합니다. 참조 이미지 2,000장과 평가 이미지를 분리하며 COCO 텍스트도 대응 이미지에 따라 분리합니다. NIH는 긍정 문항 140개를 기준 및 평가에 사용합니다. 성분 크기와 분산은 이미지·텍스트 모두 CSV에 기록하고 그림에는 이미지 결과를 표시합니다. 점수는 이미지의 네 성분과 전체 긍정 텍스트의 내적으로 분해합니다.

성분 제거는 이미지에서 alpha*rhat, beta*bhat, e_u, delta_u를 각각 뺀 뒤 재정규화합니다. 추가로 이미지 평면 밖 공간과 이미지 delta 공간에 이미지·텍스트를 함께 투영하고 재정규화합니다. 모든 변형은 같은 평가 문항을 사용합니다. CSV에는 문항별 AUC·분리도·점수 기여와 seed별 재구성 검증을 보관합니다. 코사인 범위는 평가 배열의 이웃한 표본 쌍에서 평면 또는 평면+e 투영을 고정하여 계산합니다.

01–04는 보관된 입력 생성·대조·그리기 절차를 재사용하며, 좌표 계산은 현재 공통 좌표 구현에 연결됩니다. 05는 새 네 항 분석을 직접 호출합니다. PNG는 기존 서체·색상 체계를 따릅니다. 파일 경로는 root 기준으로 해석하고 data·ckpt·cache를 공통으로 사용합니다.

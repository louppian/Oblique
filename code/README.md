# 주분석 실행

root에서 Python 파일을 직접 실행합니다. 설정은 code/main/_shared.py의 CONFIG를 사용합니다. CLI 옵션은 사용하지 않습니다.

| 실행 코드 | 분석 | 기존 PNG 비교 기준 |
|---|---|---|
| main/01_noise_concentration.py | 이미지·텍스트 노이즈 집중과 안정성 | old/analysis/fig1_noise_convergence.png 및 기존 exp01·exp02 |
| main/02_distribution_geometry.py | 이미지·텍스트 분포 위치와 축 기하 | old/analysis/fig2_image_planes.png, fig3_text_planes.png |
| main/03_corruption_response.py | 손상 전후 좌표 반응 | old/analysis/fig6_perturbation_clip.png, fig7_perturbation_carzero.png |
| main/04_reference_controls.py | 동일 기울기·conformity 대조 | old/analysis/fig8_null_test.png |
| main/05_positive_score_decomposition.py | 긍정 점수의 기여·분리·제거 및 새 성분 크기 | old/analysis/fig10_decomposition.png, fig11_component_removal.png |

각 코드는 results/main에 같은 이름의 CSV 한 개와 PNG 한 개를 저장합니다. 여러 원본 패널은 종횡비를 유지하여 한 PNG에 통합합니다. CSV는 analysis와 table 열로 세부 표를 구분합니다.

기존 분석·그리기 함수는 code/old/utils에서 재사용합니다. code/main/_shared.py는 완료된 2×2 이미지·텍스트 캐시를 기존 함수에 연결합니다. 기존 점수 분해 패널은 평균 중심 M/V/A/U 정의를 유지하고, 새 네 항의 크기·분산 패널을 별도로 추가합니다. 손상·노이즈 조건별 임베딩은 해당 조건으로 계산하고 cache에 보관합니다.

현재 분포 그림과 기존 긍정 점수 패널은 원문의 자기 도메인(CLIP–COCO, CARZero–NIH)을 중심으로 구성하며 이미지 분포 그림에 교차 도메인을 함께 표시합니다. 새 성분 크기 분석은 2×2 전체를 사용합니다. Outline의 새 정의에 따른 모든 성분 제거·대조 패널까지 확장하는 작업은 별도 보완 대상입니다.

기존 실행 코드 전체는 code/old에 보관합니다. 데이터·가중치·캐시는 root의 data·ckpt·cache를 사용합니다.

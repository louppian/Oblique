# OrthoBasis

COCO 고정 CLIP 임베딩의 공통 기저·잔차 정렬 실험 코드 및 결과.

## 최종 실행

프로젝트 루트 `D:/Oblique`에서:

```powershell
C:/Users/USER/anaconda3/envs/torch_env/python.exe code/OrthoBasis/coco_basis_orth_learned_scale_30.py
C:/Users/USER/anaconda3/envs/torch_env/python.exe code/OrthoBasis/coco_uncertainty_learned_scale.py
```

입력은 프로젝트의 `cache/clip/coco` 및 기존 split을 사용합니다. 출력과 비교 가중치는 이 폴더의 `results`에서 읽고 씁니다.

## 조건과 결과

- `axis_only_loss_vs_rotation.py`: 축 대응·행렬 직교 기준.
- `coco_shared_residual_30.py`: 공통 평면 제거 후 양성 잔차 내적 정렬.
- `coco_residual_direction_clip_30.py`: 잔차 방향 contrastive + 행렬 전체 직교.
- `coco_basis_orth_residual_clip_30.py`: 잔차 방향 contrastive + 기저만 정규직교.
- `coco_basis_orth_learned_scale_30.py`: 최종 조건, logit scale 공동 학습.
- `coco_clip_weight_diagnostics.py`: train/validation/test 정답·오답 점수 진단.
- `coco_uncertainty_comparison.py`: 고정 temperature 0.07 평가 기록.
- `coco_uncertainty_clip_scale.py`: 원본 CLIP scale 100 평가.
- `coco_uncertainty_learned_scale.py`: 최종 공동 학습 scale 평가 및 기존 조건 비교.

학습: train/validation/test 이미지 3500/500/1000, batch 512쌍, Adam lr=0.001, identity 초기화, 최종 epoch 30 사용. 최종 모델 scale은 100에서 시작하고 상한 100, 학습 후 약 95.003입니다.

최종 검색 R@1: I→T 58.00%, T→I 42.73%. 최종 uncertainty 결과는 `results/coco_uncertainty_learned_scale/metrics.csv`입니다. I→T uncertainty는 동일 이미지의 캡션 확률을 합산한 뒤 예측하므로 캡션 R@1과 구분합니다.

학습 스크립트의 baseline entropy는 초기 TAU=0.07로 기록됩니다. 조건별 scale을 반영한 uncertainty 비교에는 별도 평가 스크립트의 metrics.csv를 사용합니다. geometry 행의 orth_image/orth_text는 전체 행렬 진단값이며, 기저만 직교하는 실행의 training orth_loss와 정의가 다릅니다.

가중치 maps.pt, epoch별 comparison.csv, protocol, 그림과 쿼리별 평가 결과를 함께 보관했습니다. migration_manifest.json에 원본 경로와 결과 파일 SHA256을 기록했습니다. 기존 위치의 자료도 보존했습니다.

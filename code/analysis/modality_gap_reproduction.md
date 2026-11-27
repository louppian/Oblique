# Chowers et al. (CVPR 2026) 재현

CIFAR-10·CIFAR-100·A-OKVQA 본문과 ImageNet·MS-COCO 부록을 대상으로 한다. 현재 완료한 계산과 데이터 접근이 필요한 항목을 구분해 기록했다.

## 원문·모델·폴더

원문: [Is the Modality Gap a Bug or a Feature? A Robustness Perspective](https://arxiv.org/html/2603.29080v2). 로컬 본문 PDF와 arXiv v2의 부록 19쪽을 대조했다. 논문은 CIFAR-10(ViT-L/14), CIFAR-100(ViT-B/16), A-OKVQA(ViT-B/16·SigLIP), ImageNet(ViT-B/16·ViT-L/14·SigLIP), MS-COCO(ViT-L/14·SigLIP)를 해당 강건성 실험에 사용한다.

산출물: `code/analysis/<데이터셋>_<모델>/modality_gap/`의 `reproduce.py`, `report.md`, `metrics.csv`, `figure*.png`. 임베딩은 `cache/clip/`, 원본 데이터는 `data/`, 체크포인트는 `ckpt/`에 저장한다. A-OKVQA 문장 변경 실행 코드는 `code/utils/modality_gap_rephrasing.py`이고 결과는 같은 모델 폴더에 모인다.

## 기존 형식의 3항 분해 산출물

각 모델 폴더 최상위에 `decomposition.py`, `decompostion.md`, `figure1.png`, `figure2.png`, `figure3.png`를 생성했다. 기존 COCO와 같은 평균·입력 노이즈·잔차 분해, 추정량 표 2개, 히스토그램·산점도·두 평면 그림 3개이며 수치는 소수점 네 자리다.

- [CIFAR10_VIT-L_14 3항 분해](CIFAR10_VIT-L_14/decompostion.md)
- [CIFAR100_VIT-B_16 3항 분해](CIFAR100_VIT-B_16/decompostion.md)
- [AOKVQA_VIT-B_16 3항 분해](AOKVQA_VIT-B_16/decompostion.md)
- [AOKVQA_SigLIP_VIT-B_16 3항 분해](AOKVQA_SigLIP_VIT-B_16/decompostion.md)
- [COCO_VIT-L_14_224 3항 분해](COCO_VIT-L_14_224/decompostion.md)
- [COCO_SigLIP_VIT-B_16 3항 분해](COCO_SigLIP_VIT-B_16/decompostion.md)

COCO 224px 결과는 `COCO_VIT-L_14_224`, 기존 336px 결과는 `COCO_VIT-L_14`에 저장한다. 입력 노이즈 인코딩과 원본 임베딩은 cache/에 저장하며 Git 업로드에서 제외한다.

## Gaussian 결과

아래 수치는 저장된 소수점 네 자리 비율을 백분율로 표시했다. 원본과 전체 gap 이동을 비교한다.

| 모델 | 잡음 표준편차 | 원본 정확도 (%) | 이동 후 정확도 (%) | 원본 예측 유지율 (%) | 이동 후 예측 유지율 (%) |
| --- | --- | --- | --- | --- | --- |
| CIFAR10_VIT-L_14 | 0.0000 | 95.2000 | 95.2000 | 100.0000 | 100.0000 |
| CIFAR10_VIT-L_14 | 0.0225 | 59.4100 | 84.8400 | 60.2600 | 86.3800 |
| CIFAR100_VIT-B_16 | 0.0000 | 68.4000 | 68.4000 | 100.0000 | 100.0000 |
| CIFAR100_VIT-B_16 | 0.0225 | 29.2400 | 44.6900 | 32.4300 | 50.9400 |
| AOKVQA_VIT-B_16 | 0.0000 | 57.2900 | 57.2900 | 100.0000 | 100.0000 |
| AOKVQA_VIT-B_16 | 0.0225 | 39.5300 | 51.2600 | 46.1900 | 68.5000 |
| AOKVQA_SigLIP_VIT-B_16 | 0.0000 | 58.9500 | 58.9500 | 100.0000 | 100.0000 |
| AOKVQA_SigLIP_VIT-B_16 | 0.0225 | 40.1300 | 52.1900 | 47.0400 | 68.5800 |

## A-OKVQA 문장 변경

| 모델 | 원본 정확도 (%) | 이동 후 정확도 (%) |
| --- | --- | --- |
| AOKVQA_VIT-B_16 | 61.3400 | 81.5600 |
| AOKVQA_SigLIP_VIT-B_16 | 61.4400 | 85.6400 |

## MS-COCO 부록

잡음 표준편차 0.0100. threshold=0.0000은 전체 텍스트 공간에 직교하는 방향이며, full rank에서는 이동량이 수치 오차 수준이다. threshold를 키우면 더 많은 방향으로 이동하며 정확도와 강건성의 교환 관계가 생긴다.

| 모델 | 특이값 임계값 | 잡음 없는 정확도 (%) | 잡음 추가 정확도 (%) | 예측 유지율 (%) |
| --- | --- | --- | --- | --- |
| COCO_VIT-L_14_224 | 0.0000 | 56.1000 | 49.7700 | 50.2100 |
| COCO_VIT-L_14_224 | 2.0000 | 56.1200 | 51.1000 | 54.5400 |
| COCO_VIT-L_14_224 | 5.0000 | 55.1000 | 50.8400 | 55.7900 |
| COCO_SigLIP_VIT-B_16 | 0.0000 | 65.4600 | 56.2000 | 46.2500 |
| COCO_SigLIP_VIT-B_16 | 2.0000 | 64.8800 | 57.9500 | 51.4200 |
| COCO_SigLIP_VIT-B_16 | 5.0000 | 59.9400 | 55.3000 | 58.4200 |

## 결과 해석

CIFAR와 A-OKVQA의 Gaussian 실험에서는 정확한 직교 방향으로 이동할 때 잡음 없는 예측을 유지하면서 잡음 강건성이 증가했다. COCO에서는 특이값 임계값을 높여 이동량을 늘리면 강건성 상승과 함께 잡음 없는 검색 정확도가 낮아지는 구간이 나타났다. A-OKVQA 문장 변경은 전체 이동보다 중간 이동에서 정확도가 더 높을 수 있으므로, 전체 이동 결과와 전체 곡선을 함께 기록했다. 모든 실험은 인코더 가중치를 고정한 후처리 평가다.

## 재현 조건과 남은 항목

- OpenAI CLIP ViT-B/16·ViT-L/14는 224px, SigLIP은 ViT-B/16 WebLI 224px을 사용했다. 기존 COCO ViT-L/14@336px 캐시와 구분해 새로 추출했다. 저자의 정확한 pretrained 식별자는 본문에 생략되어 있다.
- 최초 임베딩만 단위 정규화한다. 이동 후 점수는 `image·text − ||text||² / 2`이며, 후보 사이에 공통인 영상 제곱 노름을 생략한 제곱 유클리드 거리 순위다.
- 본문 그림의 잡음 표기를 표준편차로 해석했다. 부록의 σ² 범례를 문자 그대로 해석하면 다른 조건이 되므로 저자 코드 확인 항목으로 남겼다.
- A-OKVQA는 질문별 PCA 및 train·validation 답변 목록을 구현 선택으로 명시했다.
- ImageNet 원본은 `data/imagenet`에 다운로드하도록 준비했다. 공식 서버 연결 시간초과, 공식 Hugging Face 저장소는 기존 계정에 403을 반환했다. 계정 접근 동의 후 `code/utils/prepare_imagenet_validation.py`로 50,000장을 받고 모델별 실행 코드에 연결한다.
- ImageNet Fig. 13·18 및 Fig. 15의 400개 원문 프롬프트 실험은 대기 상태다. ImageNet의 분산 5% 임계값 정의도 원문 그림과 추가 대조할 항목이다. Fig. 15의 정확한 프롬프트 목록은 원문에 제공되지 않았다.
- 따라서 현재 결과는 명시한 체크포인트·분할·잡음 해석에서 실행한 재현 결과다. ImageNet까지의 전체 완료 여부는 위 대기 항목과 함께 판단한다.

# Embedding cache

프로젝트 root에서 실행합니다.

```bash
python code/utils/build_embedding_cache.py
```

CLIP·CARZero × COCO·NIH의 2×2 구성입니다.

```text
cache/
├── clip/
│   ├── coco/
│   └── nih/
└── carzero/
    ├── coco/
    └── nih/
```

각 데이터셋 폴더에 image.npz, text.npz, noise.npz를 저장합니다. 총 12개 파일입니다.

- image.npz: 정규화 전·후 이미지 임베딩, 이미지 ID, root 상대경로, NIH 라벨.
- text.npz: 정규화 전·후 텍스트 임베딩, ID, 실제 문장, COCO 이미지 대응 또는 NIH 질환·문형·극성.
- noise.npz: 이미지 노이즈와 텍스트 노이즈의 정규화 전·후 임베딩, ID, 실제 노이즈 문장.

COCO는 val2017 전체 이미지와 캡션, NIH는 기존 NIH_test_data.csv 목록과 14개 질환 × 10개 문형 × 긍정·부정 문장을 사용합니다. 이미지 노이즈는 uniform RGB, 텍스트 노이즈는 무작위 문자 다음 무작위 단어 순서입니다.

파일 내부 metadata에 입력 목록, 모델·가중치 정보, 전처리, 노이즈 seed와 개수를 저장합니다. 동일 조건은 재사용하고 조건이 다른 기존 파일은 경로를 표시하며 중단합니다. 저장은 atomic NPZ이며 읽기는 load_cache 또는 numpy.load(..., allow_pickle=False)를 사용합니다.

CLIP 가중치는 ckpt 아래의 open_clip_model.safetensors 또는 ViT-B-32.pt를 사용하며 QuickGELU를 적용합니다. CARZero는 ckpt/best_model.ckpt와 third_party/BI_MCQ를 사용합니다. 캐시 저장 함수와 실행 코드는 build_embedding_cache.py 한 파일에 들어 있습니다. 인코더 추출 함수는 기존 code/utils/1_analysis.py를 사용합니다.


공통 실험 함수는 이 폴더에 두며 실행 코드는 ../main, ../sub에 있습니다. 자세한 구성과 실행 방법은 ../README.md를 참고하세요.

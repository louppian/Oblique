# Embedding cache

Run from the project root:

```bash
python code/utils/build_embedding_cache.py
```

Edit CONFIG in that file to select model/dataset pairs. Defaults: CLIP–COCO val2017, CARZero–NIH test, CLIP–Tiny ImageNet validation.

Output: `cache/<model>/<dataset>/<identity>/image.npz`, `text.npz`, `noise.npz`.

- image: raw/unit embeddings, IDs, relative paths, NIH disease labels or Tiny ImageNet class IDs.
- text: raw/unit embeddings, IDs and exact strings; COCO image mapping, NIH disease/template/polarity, or Tiny ImageNet class mapping.
- noise: image_raw/image_unit and text_raw/text_unit, IDs and exact noise text. Image noise is uniform RGB; text rows are random characters then random words. Seed and counts are stored in metadata.

Identity includes checkpoint SHA256, encoder settings, ordered input IDs and texts, image file sizes/mtimes, labels and noise settings. Matching files are reused. NPZ writes are atomic and uncompressed. Read with `load_cache` or `numpy.load(..., allow_pickle=False)`.

CLIP uses local `ckpt/**/open_clip_model.safetensors` or `ckpt/ViT-B-32.pt`, with QuickGELU. CARZero uses `ckpt/best_model.ckpt` and `third_party/BI_MCQ`. The dataset manifest is prepared independently for each dataset. Existing inference adapters are loaded from `code/1_analysis.py`.

NIH text contains 14 diseases × 10 templates × positive/negative polarity. Tiny ImageNet text uses `A photo of <first class synonym>.` for each class. This cache contains global embeddings; reference splits and geometric projections are computed by the experiments using saved IDs.

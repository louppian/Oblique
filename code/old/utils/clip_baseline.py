"""CLIP COCO 검색 baseline. eda.py의 캐시 재사용, CLI 없음.

설정은 CONFIG에서 변경한다. 동일 이미지의 모든 캐시 캡션을 정답으로
취급한다. 전체 캐시 후보군을 사용하므로 1k/5-caption 프로토콜과 구분한다.
동점은 후보의 캐시 행 번호가 작은 순서로 처리한다.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())
import csv
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    model: str = "ViT-B-32"
    pretrained: str = "openai"
    query_chunk: int = 128


CONFIG = Config()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("임베딩은 유한한 [N, D] 행렬이어야 합니다.")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("영벡터 임베딩이 있습니다.")
    return np.ascontiguousarray(x / norm)


def validate(img, txt, txt_img_row, image_ids, txt_ann_ids):
    if img.shape[1] != txt.shape[1] or not len(img) or not len(txt):
        raise ValueError("이미지/텍스트 임베딩의 차원 또는 개수가 잘못됐습니다.")
    for values, n, name in [(image_ids, len(img), "image_ids"),
                            (txt_ann_ids, len(txt), "txt_ann_ids"),
                            (txt_img_row, len(txt), "txt_img_row")]:
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"잘못된 {name}: 정수 벡터와 임베딩 행 수를 확인하세요.")
    if len(np.unique(image_ids)) != len(img) or len(np.unique(txt_ann_ids)) != len(txt):
        raise ValueError("중복 이미지 ID 또는 캡션 annotation ID가 있습니다.")
    if np.any(txt_img_row < 0) or np.any(txt_img_row >= len(img)):
        raise ValueError("캡션의 이미지 행 매핑이 범위를 벗어났습니다.")
    if np.any(np.bincount(txt_img_row, minlength=len(img)) == 0):
        raise ValueError("정답 캡션이 없는 이미지가 있습니다.")


def ranks_from_scores(scores, best_indices):
    best_scores = scores[np.arange(len(scores)), best_indices]
    higher = (scores > best_scores[:, None]).sum(axis=1)
    earlier_ties = ((scores == best_scores[:, None]) &
                    (np.arange(scores.shape[1])[None, :] < best_indices[:, None])).sum(axis=1)
    return higher + earlier_ties + 1


def evaluate_retrieval(img, txt, txt_img_row, query_chunk=128):
    if query_chunk < 1:
        raise ValueError("query_chunk는 양수여야 합니다.")
    positive_captions = [np.flatnonzero(txt_img_row == i) for i in range(len(img))]
    image_ranks = np.empty(len(img), dtype=np.int64)
    text_ranks = np.empty(len(txt), dtype=np.int64)
    text_auc = np.full(len(txt), np.nan, dtype=np.float64)
    for start in range(0, len(img), query_chunk):
        stop = min(start + query_chunk, len(img))
        scores = img[start:stop] @ txt.T
        best = np.array([positives[np.argmax(scores[j, positives])]
                         for j, positives in enumerate(positive_captions[start:stop])])
        image_ranks[start:stop] = ranks_from_scores(scores, best)
    for start in range(0, len(txt), query_chunk):
        stop = min(start + query_chunk, len(txt))
        scores = txt[start:stop] @ img.T
        positive = txt_img_row[start:stop]
        text_ranks[start:stop] = ranks_from_scores(scores, positive)
        if len(img) > 1:
            pos_score = scores[np.arange(len(scores)), positive]
            text_auc[start:stop] = (
                (scores < pos_score[:, None]).sum(axis=1)
                + ((scores == pos_score[:, None]).sum(axis=1) - 1) / 2
            ) / (len(img) - 1)
    return image_ranks, text_ranks, text_auc


def summarize(ranks):
    return {**{f"R@{k}_percent": float(100 * np.mean(ranks <= k)) for k in (1, 5, 10)},
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks))}


def main(config=CONFIG):
    cache = config.root / "cache" / f"{config.model}_{config.pretrained}.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"eda.py 임베딩 캐시가 필요합니다: {cache}")
    with np.load(cache, allow_pickle=False) as data:
        img, txt = normalize(data["img"]), normalize(data["txt"])
        image_ids, txt_ann_ids = data["image_ids"].copy(), data["txt_ann_ids"].copy()
        txt_img_row = data["txt_img_row"].copy()
    validate(img, txt, txt_img_row, image_ids, txt_ann_ids)
    ir, tr, auc = evaluate_retrieval(img, txt, txt_img_row, config.query_chunk)
    digest = hashlib.sha256()
    with cache.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "model": config.model, "pretrained": config.pretrained,
        "protocol": "COCO val2017: all cached images and all cached captions",
        "cache_path": str(cache),
        "n_images": len(img), "n_captions": len(txt),
        "captions_per_image_min": int(np.bincount(txt_img_row).min()),
        "captions_per_image_max": int(np.bincount(txt_img_row).max()),
        "tie_break": "descending cosine, ascending candidate cache row",
        "image_to_text": summarize(ir), "text_to_image": summarize(tr),
        "caption_image_matching_AUROC": float(auc.mean()) if len(img) > 1 else None,
        "training": "none; cached pretrained CLIP embeddings",
        "cache_sha256": digest.hexdigest(),
    }
    out = config.root / "results" / "clip_baseline" / f"{config.model}_{config.pretrained}"
    out.mkdir(parents=True, exist_ok=True)
    # JSON은 float를 소수점 세 자리로 출력하며, 계산에는 원래 정밀도를 쓴다.
    def json_three_decimals(value):
        if isinstance(value, dict):
            return "{" + ", ".join(json.dumps(k) + ": " + json_three_decimals(v)
                                    for k, v in value.items()) + "}"
        return f"{value:.3f}" if isinstance(value, float) else json.dumps(value, ensure_ascii=False)
    (out / "summary.json").write_text(json_three_decimals(result) + "\n", encoding="utf-8")
    with (out / "retrieval_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction", "R@1_percent", "R@5_percent", "R@10_percent", "median_rank", "mean_rank"])
        for direction in ("image_to_text", "text_to_image"):
            writer.writerow([direction, *[f"{v:.3f}" for v in result[direction].values()]])
    with (out / "query_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction", "query_id", "positive_image_id", "rank", "matching_AUROC"])
        writer.writerows(("image_to_text", int(i), int(i), int(r), "") for i, r in zip(image_ids, ir))
        writer.writerows(("text_to_image", int(a), int(image_ids[i]), int(r),
                          f"{v:.3f}" if np.isfinite(v) else "")
                         for a, i, r, v in zip(txt_ann_ids, txt_img_row, tr, auc))
    print(f"CLIP {config.model} / {config.pretrained}; images={len(img)}, captions={len(txt)}")
    for direction in ("image_to_text", "text_to_image"):
        print(direction + ": " + ", ".join(f"{k}={v:.3f}" for k, v in result[direction].items()))
    if result["caption_image_matching_AUROC"] is not None:
        print(f"Caption-image matching AUROC={result['caption_image_matching_AUROC']:.3f}")
    print(f"결과 폴더: {out}")
    return result

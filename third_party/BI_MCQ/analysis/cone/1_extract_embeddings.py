#!/usr/bin/env python3
"""
임베딩 추출 스크립트
===================
CARZero zero-shot / Bi-MCQ 모델에서 pre-cross-attention 임베딩을 추출하여 .pt 파일로 저장.
이미 존재하는 파일은 skip.

출력 파일:
    zs_img_emb_g.pt      (N, 768)  zero-shot 이미지 임베딩
    zs_text_pos.pt        (14, 768) zero-shot "There is X"
    zs_text_neg.pt        (14, 768) zero-shot "There is no X"
    bimcq_img_emb_g.pt    (N, 768)  Bi-MCQ 이미지 임베딩
    bimcq_text_pos.pt     (14, 768) Bi-MCQ "There is X"
    bimcq_text_neg.pt     (14, 768) Bi-MCQ "There is no X"
    labels.npy            (N, 14)   바이너리 라벨
    image_names.json      list[str] 이미지 파일명
"""

from pathlib import Path
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정 — 절대경로로 지정. 다른 컴퓨터에서는 여기만 수정.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE = Path("/shared/home/mai/Taehun/Uncertainty")
PROJECT = BASE / "BI_MCQ"
WEIGHT_ROOT = BASE / "MICCAI_2025/CARZero"

# 체크포인트
CKPT_ZEROSHOT = WEIGHT_ROOT / "checkpoints/best_model.ckpt"
CKPT_BIMCQ = WEIGHT_ROOT / "checkpoints/BiMCQ_CARZero_best_model.ckpt"

# NIH 데이터
IMAGE_ROOT = BASE / "data/NIH"
LABEL_CSV = IMAGE_ROOT / "Data_Entry_2017.csv"
TEST_CSV = IMAGE_ROOT / "NIH_test_data.csv"

# Bi-MCQ config (Bi-MCQ 모델 로딩에 필요)
BIMCQ_CONFIG = PROJECT / "configs/chest14_finetuning_Bi-MCQ.yaml"

# 출력 디렉토리
OUTPUT_DIR = PROJECT / "analysis/cone/embeddings"

DEVICE = "cuda:0"
BATCH_SIZE = 1024

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]

TERMS = [
    "Atelectasis", "Cardiomegaly", "Pleural Effusion",
    "Pulmonary Infiltration", "Pulmonary Mass", "Lung Nodule",
    "Pneumonia", "Pneumothorax", "Pulmonary Consolidation",
    "Pulmonary Edema", "Pulmonary Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]

import pandas as pd  # load_data에서만 사용


# ── 데이터 ────────────────────────────────────────────────
def load_data():
    """NIH test set 이미지 경로 + 14-class 바이너리 라벨."""
    source = pd.read_csv(LABEL_CSV, usecols=["Image Index", "Finding Labels"])
    test = pd.read_csv(TEST_CSV, usecols=["Image Index"])

    source["Image Index"] = source["Image Index"].astype(str).str.strip()
    test["Image Index"] = test["Image Index"].astype(str).str.strip()
    source = source.set_index("Image Index")
    names = test["Image Index"].tolist()

    findings = (
        source.loc[names, "Finding Labels"]
        .str.replace("_", " ", regex=False)
        .str.split("|")
        .map(lambda vs: {v.strip() for v in vs})
    )

    labels = np.column_stack([
        findings.map(lambda vs: int(c in vs)).to_numpy()
        for c in CLASSES
    ])

    image_map = {}
    for path in IMAGE_ROOT.glob("images*/*/*.png"):
        image_map[path.name] = path

    missing = [n for n in names if n not in image_map]
    if missing:
        raise FileNotFoundError(f"이미지 누락: {missing[:10]}")

    paths = [str(image_map[n]) for n in names]
    print(f"[DATA] 테스트 이미지 {len(paths):,}개 / 질환 14개")
    return paths, labels, names


# ── CARZero 모듈 import ──────────────────────────────────
def _import_carzero():
    repo = next(
        (p for p in [PROJECT, PROJECT / "Bi-MCQ"]
         if (p / "CARZero/CARZero.py").is_file()),
        None,
    )
    if repo is None:
        raise FileNotFoundError("BI_MCQ 아래에 CARZero 소스가 없습니다.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import CARZero
    return CARZero, repo


# ── 모델 로드: zero-shot ─────────────────────────────────
def load_zeroshot_model(device):
    """checkpoint의 hyper_parameters로 모델 빌드 + state_dict 로드."""
    CARZero_module, repo = _import_carzero()
    from CARZero import builder

    prev = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)
        ckpt = torch.load(str(CKPT_ZEROSHOT), map_location="cpu", weights_only=False)
        cfg = ckpt["hyper_parameters"]
        ckpt_dict = ckpt["state_dict"]
        model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(cfg)

        expected = model.state_dict()
        fixed = {}
        for k, v in ckpt_dict.items():
            new_key = k
            for prefix in ["CARZero_model.", "CARZero."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            if new_key in expected:
                fixed[new_key] = v

        missing = set(expected.keys()) - set(fixed.keys())
        if missing:
            print(f"  [WARN] zero-shot: {len(missing)}개 키 누락 (strict=False)")
            model.load_state_dict(fixed, strict=False)
        else:
            model.load_state_dict(fixed, strict=True)
    finally:
        os.chdir(prev)

    model.to(device).eval()
    print(f"[MODEL] zero-shot 로드 완료: {CKPT_ZEROSHOT.name}")
    return model, CARZero_module


# ── 모델 로드: Bi-MCQ ────────────────────────────────────
def load_bimcq_model(device):
    """Bi-MCQ config로 모델 초기화 + 체크포인트 가중치 로드."""
    from omegaconf import OmegaConf

    CARZero_module, repo = _import_carzero()

    cfg = OmegaConf.load(BIMCQ_CONFIG)
    prev = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)
        try:
            model = CARZero_module.load_CARZero(
                name=str(WEIGHT_ROOT / "pretrain_model/CARZero_best_model.ckpt"),
                device="cpu",
                multi=True,
                cfg=cfg,
            )
        except TypeError:
            print("  [WARN] load_CARZero에 multi 파라미터 없음. 수동 빌드.")
            from CARZero import builder
            ckpt_pretrain = torch.load(
                str(WEIGHT_ROOT / "pretrain_model/CARZero_best_model.ckpt"),
                map_location="cpu", weights_only=False,
            )
            pretrain_cfg = ckpt_pretrain["hyper_parameters"]
            model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(pretrain_cfg)
    finally:
        os.chdir(prev)

    ckpt = torch.load(CKPT_BIMCQ, map_location="cpu", weights_only=False)
    expected = model.state_dict()
    state = {}
    for key, value in ckpt["state_dict"].items():
        new_key = key
        for prefix in ["CARZero_model.", "CARZero."]:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        if new_key in expected:
            state[new_key] = value

    missing = set(expected.keys()) - set(state.keys())
    if missing:
        print(f"  [WARN] Bi-MCQ: {len(missing)}개 키 누락 (strict=False)")
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)

    model.to(device).eval()
    print(f"[MODEL] Bi-MCQ 로드 완료: {CKPT_BIMCQ.name}")
    return model, CARZero_module


# ── 임베딩 추출 ──────────────────────────────────────────
@torch.no_grad()
def extract_image_embeddings(model, paths, device, batch_size):
    all_g = []
    for i in tqdm(range(0, len(paths), batch_size), desc="이미지 임베딩"):
        batch_paths = paths[i : i + batch_size]
        imgs = model.process_img(batch_paths, device)
        img_emb_l, img_emb_g = model.image_encoder_forward(imgs)
        all_g.append(img_emb_g.cpu())
        del imgs, img_emb_l, img_emb_g
        torch.cuda.empty_cache()
    return torch.cat(all_g, dim=0)  # (N, 768)


@torch.no_grad()
def extract_text_embeddings(model, prompts, device):
    processed = model.process_text(prompts, device)
    text_emb_l, text_emb_g, sents = model.text_encoder_forward(
        processed["caption_ids"],
        processed["attention_mask"],
        processed["token_type_ids"],
    )
    return text_emb_g.cpu()  # (P, 768)


# ── 메인 ─────────────────────────────────────────────────
def main():
    # 필수 파일 확인
    for path in [CKPT_ZEROSHOT, CKPT_BIMCQ, LABEL_CSV, TEST_CSV, BIMCQ_CONFIG]:
        if not path.is_file():
            raise FileNotFoundError(f"필수 파일 없음: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {OUTPUT_DIR}")

    device = DEVICE
    batch_size = BATCH_SIZE

    # ── 어떤 파일이 필요한지 확인 ──
    files = {
        "labels":         OUTPUT_DIR / "labels.npy",
        "image_names":    OUTPUT_DIR / "image_names.json",
        "zs_img":         OUTPUT_DIR / "zs_img_emb_g.pt",
        "zs_text_pos":    OUTPUT_DIR / "zs_text_pos.pt",
        "zs_text_neg":    OUTPUT_DIR / "zs_text_neg.pt",
        "bimcq_img":      OUTPUT_DIR / "bimcq_img_emb_g.pt",
        "bimcq_text_pos": OUTPUT_DIR / "bimcq_text_pos.pt",
        "bimcq_text_neg": OUTPUT_DIR / "bimcq_text_neg.pt",
    }

    # 기존 cone_experiments.py가 생성한 파일 자동 감지 → 복사
    LEGACY_DIR = PROJECT / "analysis/cone"
    legacy_map = {
        "zs_img": LEGACY_DIR / "img_emb_g_zeroshot.pt",
    }
    for key, legacy_path in legacy_map.items():
        if not files[key].exists() and legacy_path.exists():
            import shutil
            shutil.copy2(legacy_path, files[key])
            print(f"  [COPY] 기존 파일 재사용: {legacy_path.name} → {files[key].name}")

    existing = {k: v.exists() for k, v in files.items()}
    for k, v in existing.items():
        tag = "SKIP" if v else "TODO"
        print(f"  [{tag}] {files[k].name}")

    if all(existing.values()):
        print("\n[DONE] 모든 임베딩이 이미 존재합니다. 추출 불필요.")
        return

    # ── 데이터 로드 (labels, names는 항상 필요) ──
    paths, labels, names = load_data()

    if not existing["labels"]:
        np.save(files["labels"], labels)
        print(f"[SAVE] {files['labels'].name}")

    if not existing["image_names"]:
        with open(files["image_names"], "w") as f:
            json.dump(names, f)
        print(f"[SAVE] {files['image_names'].name}")

    # ── 프롬프트 ──
    pos_prompts = [f"There is {t}" for t in TERMS]
    neg_prompts = [f"There is no {t}" for t in TERMS]

    # ── Zero-shot 추출 ──
    need_zs = not (existing["zs_img"] and existing["zs_text_pos"] and existing["zs_text_neg"])
    if need_zs:
        print("\n── Zero-shot 임베딩 추출 ──")
        model, _ = load_zeroshot_model(device)

        if not existing["zs_img"]:
            emb = extract_image_embeddings(model, paths, device, batch_size)
            torch.save(emb, files["zs_img"])
            print(f"[SAVE] {files['zs_img'].name}  shape={tuple(emb.shape)}")
            del emb

        if not existing["zs_text_pos"]:
            emb = extract_text_embeddings(model, pos_prompts, device)
            torch.save(emb, files["zs_text_pos"])
            print(f"[SAVE] {files['zs_text_pos'].name}  shape={tuple(emb.shape)}")

        if not existing["zs_text_neg"]:
            emb = extract_text_embeddings(model, neg_prompts, device)
            torch.save(emb, files["zs_text_neg"])
            print(f"[SAVE] {files['zs_text_neg'].name}  shape={tuple(emb.shape)}")

        del model
        torch.cuda.empty_cache()
    else:
        print("\n[SKIP] Zero-shot 임베딩 전부 존재")

    # ── Bi-MCQ 추출 ──
    need_bimcq = not (existing["bimcq_img"] and existing["bimcq_text_pos"] and existing["bimcq_text_neg"])
    if need_bimcq:
        print("\n── Bi-MCQ 임베딩 추출 ──")
        model, _ = load_bimcq_model(device)

        if not existing["bimcq_img"]:
            emb = extract_image_embeddings(model, paths, device, batch_size)
            torch.save(emb, files["bimcq_img"])
            print(f"[SAVE] {files['bimcq_img'].name}  shape={tuple(emb.shape)}")
            del emb

        if not existing["bimcq_text_pos"]:
            emb = extract_text_embeddings(model, pos_prompts, device)
            torch.save(emb, files["bimcq_text_pos"])
            print(f"[SAVE] {files['bimcq_text_pos'].name}  shape={tuple(emb.shape)}")

        if not existing["bimcq_text_neg"]:
            emb = extract_text_embeddings(model, neg_prompts, device)
            torch.save(emb, files["bimcq_text_neg"])
            print(f"[SAVE] {files['bimcq_text_neg'].name}  shape={tuple(emb.shape)}")

        del model
        torch.cuda.empty_cache()
    else:
        print("\n[SKIP] Bi-MCQ 임베딩 전부 존재")

    # ── 완료 ──
    print("\n[DONE] 추출 완료:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s}  {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
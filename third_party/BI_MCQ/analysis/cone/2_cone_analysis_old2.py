#!/usr/bin/env python3
"""
Cone Effect 분석 — 재설계 버전 (Exp 0a ~ 8)
============================================
Phase 0: Cone의 정체 (0a: random noise, 0b: MNIST)
Phase 1: 두 Cone의 구조 (1: image cone, 2: two-cone structure)
Phase 2: Gap 메커니즘 (3: gap decomposition, 4: r_z·r_Δt quality)
Phase 3: 질병별 성능 결정 요인 (5: AUC determinants)
Phase 4: 개입 실험 (6: m-removal variants)
Phase 5: Bi-MCQ 비교 (7: cone comparison, 8: cosine AUC)

사용법:
    python 2_cone_analysis.py                  # 전체 실행
    python 2_cone_analysis.py --exp 0a,0b      # Phase 0만
    python 2_cone_analysis.py --exp 3,4,5      # 핵심 메커니즘만
    python 2_cone_analysis.py --exp all        # 전체
"""

from pathlib import Path
import argparse
import json
import os
import sys
import struct
import gzip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE = Path("/shared/home/mai/Taehun/Uncertainty")
PROJECT = BASE / "BI_MCQ"
WEIGHT_ROOT = BASE / "MICCAI_2025/CARZero"

# 임베딩 디렉토리 (1_extract_embeddings.py 출력)
EMB_DIR = PROJECT / "analysis/cone/embeddings"

# 결과 출력 디렉토리
OUT_DIR = PROJECT / "analysis/cone/results"

# 체크포인트 (Exp 0a, 0b에서 모델 로드용)
CKPT_ZEROSHOT = WEIGHT_ROOT / "checkpoints/best_model.ckpt"

# MNIST 데이터 경로
MNIST_DIR = Path("/shared/home/mai/Taehun/Uncertainty/Uncertain-Aware Learning/mnist")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  공통 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _norm(t):
    return F.normalize(t, dim=-1)


def _safe_auc(y, score):
    try:
        return roc_auc_score(y, score)
    except ValueError:
        return np.nan


def _save_csv(df, path):
    import pandas as pd
    df_out = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    df_out.to_csv(path, index=False)
    print(f"  [CSV] {path.name}")


def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {path.name}")


def load_embeddings():
    """저장된 임베딩과 라벨을 dict로 반환."""
    data = {}
    data["labels"] = np.load(EMB_DIR / "labels.npy")
    data["zs_img"] = torch.load(EMB_DIR / "zs_img_emb_g.pt", map_location="cpu")
    data["zs_text_pos"] = torch.load(EMB_DIR / "zs_text_pos.pt", map_location="cpu")
    data["zs_text_neg"] = torch.load(EMB_DIR / "zs_text_neg.pt", map_location="cpu")
    data["bimcq_img"] = torch.load(EMB_DIR / "bimcq_img_emb_g.pt", map_location="cpu")
    data["bimcq_text_pos"] = torch.load(EMB_DIR / "bimcq_text_pos.pt", map_location="cpu")
    data["bimcq_text_neg"] = torch.load(EMB_DIR / "bimcq_text_neg.pt", map_location="cpu")
    print(f"[LOAD] 이미지 {data['zs_img'].shape[0]:,}개, 질환 14개")
    return data


def compute_cone(z):
    """z: (N, D) L2-normalized → m_hat, R, cos_zm, r, r_norms"""
    m = z.mean(dim=0)
    R = m.norm().item()
    m_hat = m / m.norm()
    cos_zm = (z @ m_hat).numpy()
    proj = (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    r = z - proj
    r_norms = r.norm(dim=1).numpy()
    return m_hat, R, cos_zm, r, r_norms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 0a/0b용: ViT 이미지 인코더 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


def load_vit_encoder(device):
    """zero-shot 체크포인트에서 ViT 이미지 인코더만 로드."""
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

        model.load_state_dict(fixed, strict=False)
    finally:
        os.chdir(prev)

    model.to(device).eval()
    print(f"[MODEL] ViT encoder 로드 완료 (device={device})")
    return model


@torch.no_grad()
def extract_embeddings_from_tensors(model, images_tensor, device, batch_size=64):
    """
    (N, 3, 224, 224) 텐서 → ViT → img_emb_g (N, 768)
    model.image_encoder_forward() 사용.
    """
    all_g = []
    N = images_tensor.shape[0]
    for i in range(0, N, batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
        img_emb_l, img_emb_g = model.image_encoder_forward(batch)
        all_g.append(img_emb_g.cpu())
        del batch, img_emb_l, img_emb_g
        if device != "cpu":
            torch.cuda.empty_cache()
    return torch.cat(all_g, dim=0)


def load_mnist_images(n=1000):
    """MNIST idx 파일에서 이미지 로드 → (N, 3, 224, 224) 텐서."""
    # train 또는 test 이미지 로드
    img_path = MNIST_DIR / "train-images-idx3-ubyte"
    gz_path = MNIST_DIR / "train-images-idx3-ubyte.gz"

    if img_path.exists():
        with open(img_path, "rb") as f:
            raw = f.read()
    elif gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            raw = f.read()
    else:
        raise FileNotFoundError(f"MNIST 이미지 파일 없음: {img_path} 또는 {gz_path}")

    # IDX 포맷 파싱
    magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
    data = np.frombuffer(raw[16:], dtype=np.uint8).reshape(num_images, rows, cols)

    # n개 샘플링
    indices = np.random.RandomState(42).choice(num_images, min(n, num_images), replace=False)
    images = data[indices]  # (n, 28, 28)

    # 28x28 → 224x224, 1ch → 3ch, normalize to [0,1]
    imgs_t = torch.from_numpy(images).float() / 255.0  # (n, 28, 28)
    imgs_t = imgs_t.unsqueeze(1)  # (n, 1, 28, 28)
    imgs_t = imgs_t.repeat(1, 3, 1, 1)  # (n, 3, 28, 28)
    imgs_t = F.interpolate(imgs_t, size=(224, 224), mode="bilinear", align_corners=False)

    print(f"[MNIST] {imgs_t.shape[0]}개 이미지 로드 (28→224, 1ch→3ch)")
    return imgs_t


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 0a: Random Noise → R
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp0a(nih_R):
    print("\n" + "=" * 60)
    print("Exp 0a: Random Noise Cone (encoder bias test)")
    print("=" * 60)

    device = DEVICE
    model = load_vit_encoder(device)

    # 1000개 랜덤 노이즈 이미지
    torch.manual_seed(42)
    noise_imgs = torch.randn(1000, 3, 224, 224)
    print(f"  Synthetic noise: {noise_imgs.shape}")

    emb = extract_embeddings_from_tensors(model, noise_imgs, device)
    z = _norm(emb)
    m_hat, R, cos_zm, r, r_norms = compute_cone(z)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    print(f"  R_noise = {R:.6f}")
    print(f"  R_nih   = {nih_R:.6f}")
    print(f"  비율: R_noise / R_nih = {R / nih_R:.4f}")

    import pandas as pd
    df = pd.DataFrame([{
        "source": "random_noise",
        "N": 1000,
        "R": R,
        "cos_zm_mean": float(cos_zm.mean()),
        "cos_zm_std": float(cos_zm.std()),
        "r_norm_mean": float(r_norms.mean()),
        "r_norm_std": float(r_norms.std()),
    }, {
        "source": "nih_xray",
        "N": 25596,
        "R": nih_R,
        "cos_zm_mean": np.nan,
        "cos_zm_std": np.nan,
        "r_norm_mean": np.nan,
        "r_norm_std": np.nan,
    }])
    _save_csv(df, OUT_DIR / "exp0a_noise_cone.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) R 비교 bar
    bars = axes[0].bar(["Random Noise\n(1000)", "NIH X-ray\n(25596)"],
                       [R, nih_R], color=["salmon", "steelblue"], alpha=0.8)
    for bar, val in zip(bars, [R, nih_R]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"R={val:.4f}", ha="center", fontsize=10)
    axes[0].set_ylabel("R = ||mean(z)||")
    axes[0].set_title("Cone Concentration: Noise vs X-ray")
    axes[0].set_ylim(0, max(R, nih_R) * 1.3)

    # (b) cos(z,m) 분포 — noise만
    axes[1].hist(cos_zm, bins=60, edgecolor="black", alpha=0.7, color="salmon")
    axes[1].set_xlabel("cos(z_i, m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Random Noise: cos(z_i, m) Distribution\nmean={cos_zm.mean():.4f}")
    axes[1].axvline(cos_zm.mean(), color="red", ls="--")

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp0a_noise_cone.png")
    return R


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 0b: MNIST → R
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp0b(nih_R, noise_R):
    print("\n" + "=" * 60)
    print("Exp 0b: MNIST Cone (out-of-domain test)")
    print("=" * 60)

    device = DEVICE
    model = load_vit_encoder(device)

    mnist_imgs = load_mnist_images(n=1000)
    emb = extract_embeddings_from_tensors(model, mnist_imgs, device)
    z = _norm(emb)
    m_hat, R, cos_zm, r, r_norms = compute_cone(z)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    print(f"  R_mnist = {R:.6f}")
    print(f"  R_noise = {noise_R:.6f}")
    print(f"  R_nih   = {nih_R:.6f}")

    import pandas as pd
    df = pd.DataFrame([{
        "source": "mnist",
        "N": 1000,
        "R": R,
        "cos_zm_mean": float(cos_zm.mean()),
        "cos_zm_std": float(cos_zm.std()),
        "r_norm_mean": float(r_norms.mean()),
        "r_norm_std": float(r_norms.std()),
    }, {
        "source": "random_noise",
        "N": 1000,
        "R": noise_R,
        "cos_zm_mean": np.nan,
        "cos_zm_std": np.nan,
        "r_norm_mean": np.nan,
        "r_norm_std": np.nan,
    }, {
        "source": "nih_xray",
        "N": 25596,
        "R": nih_R,
        "cos_zm_mean": np.nan,
        "cos_zm_std": np.nan,
        "r_norm_mean": np.nan,
        "r_norm_std": np.nan,
    }])
    _save_csv(df, OUT_DIR / "exp0b_mnist_cone.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) 3-source R 비교
    sources = ["Random Noise\n(1000)", "MNIST\n(1000)", "NIH X-ray\n(25596)"]
    Rs = [noise_R, R, nih_R]
    colors = ["salmon", "orange", "steelblue"]
    bars = axes[0].bar(sources, Rs, color=colors, alpha=0.8)
    for bar, val in zip(bars, Rs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"R={val:.4f}", ha="center", fontsize=10)
    axes[0].set_ylabel("R = ||mean(z)||")
    axes[0].set_title("Cone Concentration Across Domains")
    axes[0].set_ylim(0, max(Rs) * 1.3)

    # (b) MNIST cos(z,m) 분포
    axes[1].hist(cos_zm, bins=60, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("cos(z_i, m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"MNIST: cos(z_i, m) Distribution\nmean={cos_zm.mean():.4f}")
    axes[1].axvline(cos_zm.mean(), color="red", ls="--")

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp0b_mnist_cone.png")
    return R


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 1: Image Cone Geometry (기존 유지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp1(data):
    print("\n" + "=" * 60)
    print("Exp 1: Image Cone Geometry (zero-shot)")
    print("=" * 60)

    z = _norm(data["zs_img"])
    m_hat, R, cos_zm, r, r_norms = compute_cone(z)

    import pandas as pd
    df = pd.DataFrame([{
        "N": z.shape[0], "dim": 768, "R": R,
        "cos_zm_mean": float(cos_zm.mean()),
        "cos_zm_std": float(cos_zm.std()),
        "cos_zm_min": float(cos_zm.min()),
        "cos_zm_max": float(cos_zm.max()),
        "r_norm_mean": float(r_norms.mean()),
        "r_norm_std": float(r_norms.std()),
    }])
    _save_csv(df, OUT_DIR / "exp1_cone_geometry.csv")
    print(f"  R = {R:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(cos_zm, bins=100, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("cos(z_i, m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Image Embedding Cone (R={R:.4f})")
    axes[0].axvline(cos_zm.mean(), color="red", ls="--", label=f"mean={cos_zm.mean():.4f}")
    axes[0].legend()

    axes[1].hist(r_norms, bins=100, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("||r_i|| (residual norm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Component Distribution")
    axes[1].axvline(r_norms.mean(), color="red", ls="--", label=f"mean={r_norms.mean():.4f}")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp1_cone_geometry.png")
    return {"z": z, "m_hat": m_hat, "R": R}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 2: Two-Cone Structure (기존 Exp 2 + Exp 9 통합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp2(data, m_hat_img):
    print("\n" + "=" * 60)
    print("Exp 2: Two-Cone Structure (Image + Text)")
    print("=" * 60)

    t_pos_n = _norm(data["zs_text_pos"])  # (14, 768)
    t_neg_n = _norm(data["zs_text_neg"])  # (14, 768)

    # 텍스트 cone: 28개 텍스트 임베딩의 평균
    all_text = torch.cat([t_pos_n, t_neg_n], dim=0)  # (28, 768)
    m_text = all_text.mean(dim=0)
    R_text = m_text.norm().item()
    m_hat_text = m_text / m_text.norm()

    # 두 cone의 정렬도
    cos_alignment = (m_hat_img @ m_hat_text).item()

    print(f"  R_img  = {m_hat_img @ _norm(data['zs_img']).mean(dim=0) / (_norm(data['zs_img']).mean(dim=0).norm()):.4f}")
    print(f"  R_text = {R_text:.6f}")
    print(f"  cos(m̂_img, m̂_text) = {cos_alignment:.6f}")

    # Δt 분해: m̂_img, m̂_text 각각에 대해
    t_pos_raw = data["zs_text_pos"]
    t_neg_raw = data["zs_text_neg"]

    import pandas as pd
    rows = []
    for i, (cls, term) in enumerate(zip(CLASSES, TERMS)):
        cos_pn = (t_pos_n[i] @ t_neg_n[i]).item()
        dt = t_pos_raw[i] - t_neg_raw[i]
        dt_norm = dt.norm().item()

        # m̂_img 방향 분해
        dt_m_img_scalar = (dt @ m_hat_img).item()
        dt_m_img = dt_m_img_scalar * m_hat_img
        dt_r_img = dt - dt_m_img
        m_ratio_img = dt_m_img.norm().item() / (dt_norm + 1e-8)
        r_ratio_img = dt_r_img.norm().item() / (dt_norm + 1e-8)

        # m̂_text 방향 분해
        dt_m_text_scalar = (dt @ m_hat_text).item()
        dt_m_text = dt_m_text_scalar * m_hat_text
        dt_r_text = dt - dt_m_text
        m_ratio_text = dt_m_text.norm().item() / (dt_norm + 1e-8)
        r_ratio_text = dt_r_text.norm().item() / (dt_norm + 1e-8)

        # 텍스트 벡터의 m̂_img, m̂_text projection
        proj_pos_on_m_img = (t_pos_n[i] @ m_hat_img).item()
        proj_neg_on_m_img = (t_neg_n[i] @ m_hat_img).item()
        proj_pos_on_m_text = (t_pos_n[i] @ m_hat_text).item()
        proj_neg_on_m_text = (t_neg_n[i] @ m_hat_text).item()

        rows.append({
            "disease": cls, "term": term,
            "cos_pos_neg": cos_pn,
            "dt_norm": dt_norm,
            # m̂_img 기준
            "m_ratio_img": m_ratio_img, "r_ratio_img": r_ratio_img,
            "proj_pos_on_m_img": proj_pos_on_m_img,
            "proj_neg_on_m_img": proj_neg_on_m_img,
            # m̂_text 기준
            "m_ratio_text": m_ratio_text, "r_ratio_text": r_ratio_text,
            "proj_pos_on_m_text": proj_pos_on_m_text,
            "proj_neg_on_m_text": proj_neg_on_m_text,
        })

    # 전체 통계 행
    rows.append({
        "disease": "_GLOBAL",
        "term": "",
        "cos_pos_neg": float(np.mean([r["cos_pos_neg"] for r in rows[:14]])),
        "dt_norm": np.nan,
        "m_ratio_img": np.nan, "r_ratio_img": np.nan,
        "proj_pos_on_m_img": np.nan, "proj_neg_on_m_img": np.nan,
        "m_ratio_text": np.nan, "r_ratio_text": np.nan,
        "proj_pos_on_m_text": np.nan, "proj_neg_on_m_text": np.nan,
    })
    rows.append({
        "disease": "_CONE_ALIGN",
        "term": "",
        "cos_pos_neg": cos_alignment,  # cos(m̂_img, m̂_text)
        "dt_norm": R_text,             # R_text
        "m_ratio_img": np.nan, "r_ratio_img": np.nan,
        "proj_pos_on_m_img": np.nan, "proj_neg_on_m_img": np.nan,
        "m_ratio_text": np.nan, "r_ratio_text": np.nan,
        "proj_pos_on_m_text": np.nan, "proj_neg_on_m_text": np.nan,
    })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp2_two_cone.csv")

    # ── 시각화: 4-panel ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x = np.arange(14)
    df14 = df.head(14)

    # (a) cos(pos, neg) per disease
    axes[0, 0].bar(x, df14["cos_pos_neg"], color="steelblue", alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("cos(t_pos, t_neg)")
    axes[0, 0].set_title("Positive-Negative Text Similarity")
    axes[0, 0].axhline(df14["cos_pos_neg"].mean(), color="red", ls="--",
                        label=f'mean={df14["cos_pos_neg"].mean():.4f}')
    axes[0, 0].legend()

    # (b) Δt decomposition: m̂_img vs m̂_text
    w = 0.2
    axes[0, 1].bar(x - w, df14["m_ratio_img"], w, label="m_ratio (m̂_img)", color="coral")
    axes[0, 1].bar(x, df14["m_ratio_text"], w, label="m_ratio (m̂_text)", color="orange")
    axes[0, 1].bar(x + w, df14["r_ratio_img"], w, label="r_ratio (m̂_img)", color="teal")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].set_title("Δt Decomposition: m̂_img vs m̂_text")
    axes[0, 1].legend(fontsize=7)

    # (c) Pos/Neg projection on m̂_img
    w2 = 0.35
    axes[1, 0].bar(x - w2/2, df14["proj_pos_on_m_img"], w2, label="pos", alpha=0.8)
    axes[1, 0].bar(x + w2/2, df14["proj_neg_on_m_img"], w2, label="neg", alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("t · m̂_img")
    axes[1, 0].set_title("Text Projection on Image Cone Axis")
    axes[1, 0].legend()

    # (d) Cone alignment annotation
    axes[1, 1].text(0.5, 0.6,
                    f"cos(m̂_img, m̂_text) = {cos_alignment:.4f}",
                    transform=axes[1, 1].transAxes,
                    fontsize=16, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    axes[1, 1].text(0.5, 0.35,
                    f"R_text = {R_text:.4f}",
                    transform=axes[1, 1].transAxes,
                    fontsize=14, ha="center", va="center")
    axes[1, 1].text(0.5, 0.15,
                    "m̂_img ≈ m̂_text?" if cos_alignment > 0.8
                    else ("m̂_img ≠ m̂_text" if cos_alignment < 0.3
                          else "m̂_img ~ m̂_text (moderate)"),
                    transform=axes[1, 1].transAxes,
                    fontsize=12, ha="center", va="center", color="gray")
    axes[1, 1].set_title("Cone Alignment Summary")
    axes[1, 1].axis("off")

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp2_two_cone.png")
    return {"m_hat_text": m_hat_text, "R_text": R_text, "cos_alignment": cos_alignment}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 3: Gap Decomposition (NEW — gap = m_term + r_term 실측)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp3(data, z, m_hat):
    """
    gap_i = z_i · Δt = (z_i·m̂)(Δt·m̂) + r_z_i · r_Δt
    m_term과 r_term을 실측하여 cone 소거를 검증.
    """
    print("\n" + "=" * 60)
    print("Exp 3: Gap Decomposition (m_term vs r_term)")
    print("=" * 60)

    t_pos_raw = data["zs_text_pos"]
    t_neg_raw = data["zs_text_neg"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])
    labels = data["labels"]

    # z 분해
    z_dot_m = (z @ m_hat)  # (N,)
    r_z = z - z_dot_m.unsqueeze(1) * m_hat.unsqueeze(0)  # (N, 768)

    import pandas as pd
    rows = []
    all_m_frac = []

    for d, cls in enumerate(CLASSES):
        dt = t_pos_raw[d] - t_neg_raw[d]
        dt_dot_m = (dt @ m_hat).item()
        r_dt = dt - dt_dot_m * m_hat  # (768,)

        # 전체 이미지에 대해 m_term, r_term 계산
        m_term = (z_dot_m * dt_dot_m).numpy()  # (N,)
        r_term = (r_z @ r_dt).numpy()          # (N,)
        gap = m_term + r_term

        # gap 중 m_term이 차지하는 비율
        m_frac = np.abs(m_term) / (np.abs(m_term) + np.abs(r_term) + 1e-12)
        mean_m_frac = float(m_frac.mean())
        all_m_frac.append(mean_m_frac)

        # 부호 일치율: m_term과 r_term이 같은 방향인 비율
        sign_agree = float((np.sign(m_term) == np.sign(r_term)).mean())

        # gap으로 AUC 계산 (검증용 — exp5와 동일해야)
        gap_auc = _safe_auc(labels[:, d], gap)

        rows.append({
            "disease": cls,
            "dt_dot_m": dt_dot_m,
            "mean_abs_m_term": float(np.abs(m_term).mean()),
            "mean_abs_r_term": float(np.abs(r_term).mean()),
            "m_fraction": mean_m_frac,
            "r_fraction": 1.0 - mean_m_frac,
            "sign_agreement": sign_agree,
            "gap_auc_pnc": gap_auc,
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp3_gap_decomposition.csv")

    print(f"  m_fraction 평균: {np.mean(all_m_frac):.4f}")
    print(f"  → gap의 {np.mean(all_m_frac)*100:.1f}%가 m_term, {(1-np.mean(all_m_frac))*100:.1f}%가 r_term")
    print(df[["disease", "mean_abs_m_term", "mean_abs_r_term", "m_fraction"]].to_string(index=False))

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    # (a) m_term vs r_term 비중 stacked bar
    axes[0].bar(x, df["m_fraction"], label="m_term fraction", color="coral", alpha=0.8)
    axes[0].bar(x, df["r_fraction"], bottom=df["m_fraction"],
                label="r_term fraction", color="teal", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("fraction of |gap|")
    axes[0].set_title(f"Gap Composition: m_term vs r_term\nmean m_frac={np.mean(all_m_frac):.4f}")
    axes[0].legend()

    # (b) |m_term| vs |r_term| 절대값 비교
    w = 0.35
    axes[1].bar(x - w/2, df["mean_abs_m_term"], w, label="|m_term|", color="coral", alpha=0.8)
    axes[1].bar(x + w/2, df["mean_abs_r_term"], w, label="|r_term|", color="teal", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("mean absolute value")
    axes[1].set_title("Absolute Magnitude: m_term vs r_term")
    axes[1].legend()

    # (c) Δt·m̂ per disease (m_term의 source)
    axes[2].bar(x, df["dt_dot_m"], color="coral", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Δt · m̂")
    axes[2].set_title("Text Difference Projection on Cone Axis\n(source of m_term)")
    axes[2].axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp3_gap_decomposition.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 4: r_z · r_Δt Quality (NEW — 실제 discrimination 메커니즘)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp4(data, z, m_hat):
    """
    gap ≈ r_z · r_Δt 이므로, r_z와 r_Δt의 관계를 분석.
    - pos 이미지의 cos(r_z, r̂_Δt) vs neg 이미지의 cos(r_z, r̂_Δt)
    - ||r_Δt|| per disease
    """
    print("\n" + "=" * 60)
    print("Exp 4: r_z · r_Δt Quality Analysis")
    print("=" * 60)

    labels = data["labels"]
    t_pos_raw = data["zs_text_pos"]
    t_neg_raw = data["zs_text_neg"]

    # z → r_z
    r_z = z - (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)  # (N, 768)
    r_z_n = _norm(r_z)  # 방향만

    import pandas as pd
    rows = []

    for d, cls in enumerate(CLASSES):
        dt = t_pos_raw[d] - t_neg_raw[d]
        dt_dot_m = (dt @ m_hat).item()
        r_dt = dt - dt_dot_m * m_hat
        r_dt_norm = r_dt.norm().item()
        r_dt_hat = r_dt / (r_dt.norm() + 1e-8)

        # cos(r_z, r̂_Δt) for each image
        cos_r = (r_z_n @ r_dt_hat).numpy()  # (N,)

        pos_mask = labels[:, d] == 1
        neg_mask = labels[:, d] == 0

        cos_r_pos = cos_r[pos_mask]
        cos_r_neg = cos_r[neg_mask]

        # r_term = r_z · r_Δt = ||r_z|| * ||r_Δt|| * cos(r_z, r_Δt)
        # discrimination: pos 이미지의 cos_r > neg 이미지의 cos_r?
        sep = float(cos_r_pos.mean() - cos_r_neg.mean()) if pos_mask.sum() > 0 else np.nan
        auc_cos_r = _safe_auc(labels[:, d], cos_r)

        rows.append({
            "disease": cls,
            "r_dt_norm": r_dt_norm,
            "cos_r_pos_mean": float(cos_r_pos.mean()) if pos_mask.sum() > 0 else np.nan,
            "cos_r_neg_mean": float(cos_r_neg.mean()) if neg_mask.sum() > 0 else np.nan,
            "cos_r_separation": sep,
            "auc_cos_r": auc_cos_r,
            "cos_r_pos_std": float(cos_r_pos.std()) if pos_mask.sum() > 0 else np.nan,
            "cos_r_neg_std": float(cos_r_neg.std()) if neg_mask.sum() > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp4_residual_quality.csv")

    print(df[["disease", "r_dt_norm", "cos_r_separation", "auc_cos_r"]].to_string(index=False))

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    # (a) ||r_Δt|| per disease
    axes[0].bar(x, df["r_dt_norm"], color="teal", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("||r_Δt||")
    axes[0].set_title("Negation Signal Strength in Residual Space")

    # (b) cos(r_z, r̂_Δt) pos vs neg
    w = 0.35
    axes[1].bar(x - w/2, df["cos_r_pos_mean"], w, label="Positive images", alpha=0.8, color="coral")
    axes[1].bar(x + w/2, df["cos_r_neg_mean"], w, label="Negative images", alpha=0.8, color="steelblue")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("mean cos(r_z, r̂_Δt)")
    axes[1].set_title("Residual Alignment: Pos vs Neg Images")
    axes[1].legend()
    axes[1].axhline(0, color="black", lw=0.5)

    # (c) AUC from cos(r_z, r̂_Δt)
    axes[2].bar(x, df["auc_cos_r"], color="teal", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("AUC from Residual Alignment\n(= negation discrimination in r-space)")
    axes[2].axhline(0.5, color="red", ls="--", alpha=0.5)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp4_residual_quality.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 5: AUC 결정 요인 분석 (기존 Exp 5+7 통합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp5(data, z, m_hat, exp4_df):
    """
    14개 질병의 PNC AUC를 설명하는 변수 비교:
    - ||r_Δt||: negation 신호 강도
    - cos_r_separation: r-space pos/neg 분리도
    - cone_offset: 양성 이미지 cone 이탈 정도
    - cos(pos,neg): 텍스트 유사도
    """
    print("\n" + "=" * 60)
    print("Exp 5: AUC Determinants (multi-variable analysis)")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    cos_zm = (z @ m_hat).numpy()
    sim_pos = (z @ t_pos_n.T).numpy()
    sim_neg = (z @ t_neg_n.T).numpy()
    gap = sim_pos - sim_neg

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        pos_mask = labels[:, d] == 1
        pnc_auc = _safe_auc(labels[:, d], gap[:, d])
        cos_pn = (t_pos_n[d] @ t_neg_n[d]).item()
        cos_offset = cos_zm[pos_mask].mean() - cos_zm.mean() if pos_mask.sum() > 0 else np.nan

        rows.append({
            "disease": cls,
            "pnc_auc": pnc_auc,
            "cos_pos_neg": cos_pn,
            "cone_offset": cos_offset,
            "r_dt_norm": exp4_df.iloc[d]["r_dt_norm"],
            "cos_r_separation": exp4_df.iloc[d]["cos_r_separation"],
            "auc_cos_r": exp4_df.iloc[d]["auc_cos_r"],
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp5_auc_determinants.csv")

    # 각 변수와 PNC AUC의 상관
    predictors = {
        "||r_Δt||": ("r_dt_norm", df["r_dt_norm"]),
        "cos_r_sep": ("cos_r_separation", df["cos_r_separation"]),
        "cone_offset": ("cone_offset", df["cone_offset"]),
        "cos(pos,neg)": ("cos_pos_neg", df["cos_pos_neg"]),
    }

    print("\n  PNC AUC 상관 분석:")
    corr_results = {}
    for name, (col, vals) in predictors.items():
        rho, p = spearmanr(vals, df["pnc_auc"])
        corr_results[name] = (rho, p)
        print(f"    {name:15s}  Spearman ρ={rho:+.4f}  p={p:.4f}")

    # ── 시각화: 2x2 scatter ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, (col, vals)) in enumerate(predictors.items()):
        ax = axes[idx]
        rho, p = corr_results[name]
        ax.scatter(vals, df["pnc_auc"], s=60, zorder=5, color="steelblue", edgecolors="white")
        for i, cls in enumerate(CLASSES):
            ax.annotate(cls, (vals.iloc[i], df["pnc_auc"].iloc[i]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")
        ax.set_xlabel(name)
        ax.set_ylabel("PNC AUC")
        ax.set_title(f"{name} vs PNC AUC\nSpearman ρ={rho:+.4f} (p={p:.3f})")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

        # regression line
        if np.isfinite(vals).all() and np.isfinite(df["pnc_auc"]).all():
            coeffs = np.polyfit(vals, df["pnc_auc"], 1)
            x_fit = np.linspace(vals.min(), vals.max(), 50)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", alpha=0.5)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp5_auc_determinants.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 6: m-Removal Variants (기존 Exp 6 확장 = Exp 10 통합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp6(data, z, m_hat_img, m_hat_text):
    """
    4가지 m-removal 변형:
    (a) image만 m̂_img 제거
    (b) text만 m̂_img 제거
    (c) 양쪽 다 m̂_img 제거
    (d) image에 m̂_img, text에 m̂_text 각각 제거
    """
    print("\n" + "=" * 60)
    print("Exp 6: m-Removal Variants")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    # 원본 gap
    sim_pos_orig = (z @ t_pos_n.T).numpy()
    sim_neg_orig = (z @ t_neg_n.T).numpy()
    gap_orig = sim_pos_orig - sim_neg_orig

    # m̂_img 제거 함수
    def remove_m(vecs, m_hat):
        proj = (vecs @ m_hat).unsqueeze(-1) * m_hat.unsqueeze(0)
        return _norm(vecs - proj)

    # (a) image만 m̂_img 제거
    z_rm = remove_m(z, m_hat_img)
    gap_a = (z_rm @ t_pos_n.T - z_rm @ t_neg_n.T).numpy()

    # (b) text만 m̂_img 제거
    tp_rm = remove_m(t_pos_n, m_hat_img)
    tn_rm = remove_m(t_neg_n, m_hat_img)
    gap_b = (z @ tp_rm.T - z @ tn_rm.T).numpy()

    # (c) 양쪽 다 m̂_img 제거
    gap_c = (z_rm @ tp_rm.T - z_rm @ tn_rm.T).numpy()

    # (d) image에 m̂_img, text에 m̂_text 각각
    tp_rm_text = remove_m(t_pos_n, m_hat_text)
    tn_rm_text = remove_m(t_neg_n, m_hat_text)
    gap_d = (z_rm @ tp_rm_text.T - z_rm @ tn_rm_text.T).numpy()

    import pandas as pd
    rows = []
    conditions = {
        "original": gap_orig,
        "img_only": gap_a,
        "text_only": gap_b,
        "both_m_img": gap_c,
        "img_m_img_text_m_text": gap_d,
    }

    for d, cls in enumerate(CLASSES):
        row = {"disease": cls}
        for cond_name, gap_arr in conditions.items():
            row[f"pnc_{cond_name}"] = _safe_auc(labels[:, d], gap_arr[:, d])
        rows.append(row)

    df = pd.DataFrame(rows)

    # delta columns
    for cond in ["img_only", "text_only", "both_m_img", "img_m_img_text_m_text"]:
        df[f"delta_{cond}"] = df[f"pnc_{cond}"] - df["pnc_original"]

    _save_csv(df, OUT_DIR / "exp6_m_removal_variants.csv")

    # 평균 출력
    print("\n  PNC AUC 평균:")
    for cond in conditions:
        mean_val = df[f"pnc_{cond}"].mean()
        if cond == "original":
            print(f"    {cond:30s}  {mean_val:.4f}")
        else:
            delta = mean_val - df["pnc_original"].mean()
            print(f"    {cond:30s}  {mean_val:.4f}  (Δ={delta:+.4f})")

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    # (a) 5조건 PNC AUC 비교 (grouped bar)
    cond_names = ["original", "img_only", "text_only", "both_m_img", "img_m_img_text_m_text"]
    cond_labels = ["Original", "Rm img m̂_i", "Rm text m̂_i", "Rm both m̂_i", "Rm img m̂_i + text m̂_t"]
    colors_cond = ["steelblue", "coral", "orange", "red", "purple"]
    n_conds = len(cond_names)
    w = 0.8 / n_conds
    for ci, (cond, lbl) in enumerate(zip(cond_names, cond_labels)):
        offset = (ci - n_conds/2 + 0.5) * w
        axes[0].bar(x + offset, df[f"pnc_{cond}"], w, label=lbl, alpha=0.8, color=colors_cond[ci])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=6)
    axes[0].set_ylabel("PNC AUC")
    axes[0].set_title("m-Removal Variants: PNC AUC")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0].legend(fontsize=6, loc="upper right")

    # (b) ΔAUC per variant (mean bar)
    delta_cols = ["delta_img_only", "delta_text_only", "delta_both_m_img", "delta_img_m_img_text_m_text"]
    delta_labels = ["Rm img m̂_i", "Rm text m̂_i", "Rm both m̂_i", "Rm img m̂_i\n+ text m̂_t"]
    delta_means = [df[c].mean() for c in delta_cols]
    bar_colors = ["coral" if v < 0 else "green" for v in delta_means]
    axes[1].bar(range(len(delta_cols)), delta_means, color=bar_colors, alpha=0.8)
    axes[1].set_xticks(range(len(delta_cols)))
    axes[1].set_xticklabels(delta_labels, fontsize=8)
    axes[1].set_ylabel("mean ΔAUC")
    axes[1].set_title("Average AUC Change by Removal Strategy")
    axes[1].axhline(0, color="black", lw=0.5)
    for i, v in enumerate(delta_means):
        axes[1].text(i, v + 0.001 * np.sign(v), f"{v:+.4f}", ha="center", fontsize=9)

    # (c) per-disease delta for best variant
    # 자동으로 가장 좋은 variant 선택
    best_idx = np.argmax([abs(m) for m in delta_means])
    best_col = delta_cols[best_idx]
    best_label = delta_labels[best_idx]
    bar_colors_d = ["green" if v > 0 else "red" for v in df[best_col]]
    axes[2].bar(x, df[best_col], color=bar_colors_d, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("ΔAUC")
    axes[2].set_title(f"Per-disease ΔAUC: {best_label}")
    axes[2].axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp6_m_removal_variants.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 7: ZS vs Bi-MCQ Cone (기존 Exp 4 확장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp7(data, zs_z, zs_m_hat, zs_R):
    print("\n" + "=" * 60)
    print("Exp 7: Zero-shot vs Bi-MCQ Cone Comparison")
    print("=" * 60)

    # Bi-MCQ image cone
    z_bm = _norm(data["bimcq_img"])
    m_hat_bm, R_bm, cos_zm_bm, _, _ = compute_cone(z_bm)

    # Bi-MCQ text cone
    tp_bm = _norm(data["bimcq_text_pos"])
    tn_bm = _norm(data["bimcq_text_neg"])
    all_text_bm = torch.cat([tp_bm, tn_bm], dim=0)
    m_text_bm = all_text_bm.mean(dim=0)
    m_hat_text_bm = m_text_bm / m_text_bm.norm()

    # ZS text cone
    tp_zs = _norm(data["zs_text_pos"])
    tn_zs = _norm(data["zs_text_neg"])
    all_text_zs = torch.cat([tp_zs, tn_zs], dim=0)
    m_text_zs = all_text_zs.mean(dim=0)
    m_hat_text_zs = m_text_zs / m_text_zs.norm()

    # Cone alignment
    align_zs = (zs_m_hat @ m_hat_text_zs).item()
    align_bm = (m_hat_bm @ m_hat_text_bm).item()

    cos_zm_zs = (zs_z @ zs_m_hat).numpy()

    # text cos(pos, neg)
    cos_pn_zs = np.array([(tp_zs[i] @ tn_zs[i]).item() for i in range(14)])
    cos_pn_bm = np.array([(tp_bm[i] @ tn_bm[i]).item() for i in range(14)])

    import pandas as pd
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append({
            "disease": cls,
            "zs_cos_pos_neg": cos_pn_zs[i],
            "bimcq_cos_pos_neg": cos_pn_bm[i],
            "delta_cos_pos_neg": cos_pn_bm[i] - cos_pn_zs[i],
        })
    summary = pd.DataFrame(rows)

    global_row = pd.DataFrame([{
        "disease": "_GLOBAL",
        "zs_cos_pos_neg": float(cos_pn_zs.mean()),
        "bimcq_cos_pos_neg": float(cos_pn_bm.mean()),
        "delta_cos_pos_neg": float(cos_pn_bm.mean() - cos_pn_zs.mean()),
    }])
    cone_row = pd.DataFrame([{
        "disease": "_CONE_R",
        "zs_cos_pos_neg": zs_R,
        "bimcq_cos_pos_neg": R_bm,
        "delta_cos_pos_neg": R_bm - zs_R,
    }])
    align_row = pd.DataFrame([{
        "disease": "_CONE_ALIGN",
        "zs_cos_pos_neg": align_zs,
        "bimcq_cos_pos_neg": align_bm,
        "delta_cos_pos_neg": align_bm - align_zs,
    }])
    df = pd.concat([summary, global_row, cone_row, align_row], ignore_index=True)
    _save_csv(df, OUT_DIR / "exp7_zs_vs_bimcq.csv")

    print(f"  Zero-shot  R={zs_R:.4f}  align={align_zs:.4f}")
    print(f"  Bi-MCQ     R={R_bm:.4f}  align={align_bm:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Cone distribution
    axes[0].hist(cos_zm_zs, bins=80, alpha=0.5, density=True,
                 label=f"Zero-shot (R={zs_R:.4f})")
    axes[0].hist(cos_zm_bm, bins=80, alpha=0.5, density=True,
                 label=f"Bi-MCQ (R={R_bm:.4f})")
    axes[0].set_xlabel("cos(z_i, m)")
    axes[0].set_title("Cone Distribution: Zero-shot vs Bi-MCQ")
    axes[0].legend()

    # (b) cos(pos,neg)
    x = np.arange(14)
    w = 0.35
    axes[1].bar(x - w/2, cos_pn_zs, w, label="Zero-shot", alpha=0.8)
    axes[1].bar(x + w/2, cos_pn_bm, w, label="Bi-MCQ", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("cos(t_pos, t_neg)")
    axes[1].set_title("Pos-Neg Text Similarity")
    axes[1].legend()

    # (c) Cone alignment comparison
    axes[2].bar(["Zero-shot", "Bi-MCQ"], [align_zs, align_bm],
                color=["steelblue", "coral"], alpha=0.8)
    axes[2].set_ylabel("cos(m̂_img, m̂_text)")
    axes[2].set_title("Image-Text Cone Alignment")
    for i, v in enumerate([align_zs, align_bm]):
        axes[2].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
    axes[2].set_ylim(0, 1.0)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp7_zs_vs_bimcq.png")
    return {"R_bm": R_bm, "z_bm": z_bm, "m_hat_bm": m_hat_bm}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 8: ZS vs Bi-MCQ Cosine AUC (기존 유지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp8(data, zs_z):
    print("\n" + "=" * 60)
    print("Exp 8: Bi-MCQ Cosine-based AUC (Embedding Quality)")
    print("=" * 60)

    labels = data["labels"]

    zs_tp = _norm(data["zs_text_pos"])
    zs_tn = _norm(data["zs_text_neg"])
    zs_gap = (zs_z @ zs_tp.T - zs_z @ zs_tn.T).numpy()

    bm_z = _norm(data["bimcq_img"])
    bm_tp = _norm(data["bimcq_text_pos"])
    bm_tn = _norm(data["bimcq_text_neg"])
    bm_gap = (bm_z @ bm_tp.T - bm_z @ bm_tn.T).numpy()

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        y = labels[:, d]
        rows.append({
            "disease": cls,
            "zs_auc_pos": _safe_auc(y, (zs_z @ zs_tp.T).numpy()[:, d]),
            "zs_auc_neg": _safe_auc(y, -(zs_z @ zs_tn.T).numpy()[:, d]),
            "zs_auc_pnc": _safe_auc(y, zs_gap[:, d]),
            "bimcq_auc_pos": _safe_auc(y, (bm_z @ bm_tp.T).numpy()[:, d]),
            "bimcq_auc_neg": _safe_auc(y, -(bm_z @ bm_tn.T).numpy()[:, d]),
            "bimcq_auc_pnc": _safe_auc(y, bm_gap[:, d]),
            "delta_pnc": _safe_auc(y, bm_gap[:, d]) - _safe_auc(y, zs_gap[:, d]),
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp8_bimcq_cosine_auc.csv")

    mean_zs = df["zs_auc_pnc"].mean()
    mean_bm = df["bimcq_auc_pnc"].mean()
    print(f"  PNC AUC 평균: ZS={mean_zs:.4f}  Bi-MCQ={mean_bm:.4f}  Δ={mean_bm - mean_zs:+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    w = 0.35
    axes[0].bar(x - w/2, df["zs_auc_pnc"], w, label="Zero-shot", alpha=0.8, color="steelblue")
    axes[0].bar(x + w/2, df["bimcq_auc_pnc"], w, label="Bi-MCQ", alpha=0.8, color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("PNC AUC (cosine)")
    axes[0].set_title(f"Cosine PNC AUC: ZS vs Bi-MCQ\nmean Δ={mean_bm - mean_zs:+.4f}")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0].legend()

    colors = ["green" if v > 0 else "red" for v in df["delta_pnc"]]
    axes[1].bar(x, df["delta_pnc"], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("ΔAUC (Bi-MCQ − ZS)")
    axes[1].set_title("Per-disease AUC Improvement")
    axes[1].axhline(0, color="black", lw=0.5)

    w2 = 0.25
    axes[2].bar(x - w2, df["bimcq_auc_pos"], w2, label="POS", alpha=0.8)
    axes[2].bar(x, df["bimcq_auc_neg"], w2, label="NEG", alpha=0.8)
    axes[2].bar(x + w2, df["bimcq_auc_pnc"], w2, label="PNC", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Bi-MCQ Cosine AUCs")
    axes[2].axhline(0.5, color="red", ls="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp8_bimcq_cosine_auc.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID_EXPS = {"0a", "0b", "1", "2", "3", "4", "5", "6", "7", "8"}


def main():
    parser = argparse.ArgumentParser(description="Cone Effect Analysis (Exp 0a~8)")
    parser.add_argument("--exp", type=str, default="all",
                        help="실험 번호 (예: 0a,0b,1 또는 all)")
    args = parser.parse_args()

    if args.exp == "all":
        exps = VALID_EXPS.copy()
    else:
        exps = set(e.strip() for e in args.exp.split(","))
        invalid = exps - VALID_EXPS
        if invalid:
            raise ValueError(f"잘못된 실험 번호: {invalid}. 사용 가능: {sorted(VALID_EXPS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {OUT_DIR}")

    # ── Phase 0: 0a, 0b는 모델 로드 필요 (GPU) ──
    # 나머지는 저장된 임베딩 사용

    # 먼저 기본 임베딩 로드 (1-8에 필요)
    need_emb = exps & {"1", "2", "3", "4", "5", "6", "7", "8"}
    data = None
    if need_emb:
        required = ["labels.npy", "zs_img_emb_g.pt", "zs_text_pos.pt", "zs_text_neg.pt"]
        if exps & {"7", "8"}:
            required += ["bimcq_img_emb_g.pt", "bimcq_text_pos.pt", "bimcq_text_neg.pt"]
        for name in required:
            if not (EMB_DIR / name).exists():
                raise FileNotFoundError(
                    f"임베딩 파일 없음: {EMB_DIR / name}\n"
                    f"먼저 1_extract_embeddings.py 를 실행하세요."
                )
        data = load_embeddings()

    # z, m_hat 준비 (대부분 실험에서 필요)
    z = m_hat = R = None
    if data is not None:
        z = _norm(data["zs_img"])
        m = z.mean(dim=0)
        R = m.norm().item()
        m_hat = m / m.norm()

    # ── Exp 1 ──
    if "1" in exps:
        exp1_out = exp1(data)
        z, m_hat, R = exp1_out["z"], exp1_out["m_hat"], exp1_out["R"]

    # ── Exp 0a (R 필요 → exp1 이후) ──
    noise_R = None
    if "0a" in exps:
        nih_R = R if R is not None else 0.6927  # fallback
        noise_R = exp0a(nih_R)

    # ── Exp 0b (noise_R 필요 → 0a 이후) ──
    if "0b" in exps:
        nih_R = R if R is not None else 0.6927
        nr = noise_R if noise_R is not None else 0.0
        exp0b(nih_R, nr)

    # ── Exp 2 ──
    m_hat_text = None
    if "2" in exps:
        exp2_out = exp2(data, m_hat)
        m_hat_text = exp2_out["m_hat_text"]

    # m_hat_text fallback (exp6에서 필요)
    if m_hat_text is None and data is not None:
        t_pos_n = _norm(data["zs_text_pos"])
        t_neg_n = _norm(data["zs_text_neg"])
        all_t = torch.cat([t_pos_n, t_neg_n], dim=0)
        mt = all_t.mean(dim=0)
        m_hat_text = mt / mt.norm()

    # ── Exp 3 ──
    if "3" in exps:
        exp3(data, z, m_hat)

    # ── Exp 4 ──
    exp4_df = None
    if "4" in exps:
        exp4_df = exp4(data, z, m_hat)

    # ── Exp 5 (exp4 결과 필요) ──
    if "5" in exps:
        if exp4_df is None:
            print("  [INFO] Exp 5 requires Exp 4 results. Running Exp 4 first...")
            exp4_df = exp4(data, z, m_hat)
        exp5(data, z, m_hat, exp4_df)

    # ── Exp 6 ──
    if "6" in exps:
        exp6(data, z, m_hat, m_hat_text)

    # ── Exp 7 ──
    if "7" in exps:
        exp7(data, z, m_hat, R)

    # ── Exp 8 ──
    if "8" in exps:
        exp8(data, z)

    # ── 완료 ──
    print("\n" + "=" * 60)
    print("완료! 결과 파일:")
    for f in sorted(OUT_DIR.glob("exp*")):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
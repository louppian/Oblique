#!/usr/bin/env python3
"""
Cone Effect 분석 — Exp 1-8
==========================
1_extract_embeddings.py 가 생성한 임베딩 파일(.pt)을 읽어
GPU 없이 전체 분석을 수행한다.

각 실험: CSV 1개 + PNG 1개.

사용법:
    python 2_cone_analysis.py              # 전체 실행
    python 2_cone_analysis.py --exp 6,7,8  # 신규 실험만
"""

from pathlib import Path
import argparse
import json

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

# 임베딩 디렉토리 (1_extract_embeddings.py 출력)
EMB_DIR = PROJECT / "analysis/cone/embeddings"

# 결과 출력 디렉토리
OUT_DIR = PROJECT / "analysis/cone/results"

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
#  공통: 임베딩 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_embeddings():
    """저장된 임베딩과 라벨을 dict로 반환."""
    data = {}
    data["labels"] = np.load(EMB_DIR / "labels.npy")                          # (N, 14)
    data["zs_img"] = torch.load(EMB_DIR / "zs_img_emb_g.pt", map_location="cpu")   # (N, 768)
    data["zs_text_pos"] = torch.load(EMB_DIR / "zs_text_pos.pt", map_location="cpu")  # (14, 768)
    data["zs_text_neg"] = torch.load(EMB_DIR / "zs_text_neg.pt", map_location="cpu")  # (14, 768)
    data["bimcq_img"] = torch.load(EMB_DIR / "bimcq_img_emb_g.pt", map_location="cpu")
    data["bimcq_text_pos"] = torch.load(EMB_DIR / "bimcq_text_pos.pt", map_location="cpu")
    data["bimcq_text_neg"] = torch.load(EMB_DIR / "bimcq_text_neg.pt", map_location="cpu")
    print(f"[LOAD] 이미지 {data['zs_img'].shape[0]:,}개, 질환 14개")
    return data


def _norm(t):
    """L2 정규화."""
    return F.normalize(t, dim=1)


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 1: Cone Geometry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp1(data):
    print("\n" + "=" * 60)
    print("Exp 1: Cone Geometry (zero-shot)")
    print("=" * 60)

    z = _norm(data["zs_img"])          # (N, 768)
    N = z.shape[0]
    m = z.mean(dim=0)
    R = m.norm().item()
    m_hat = m / m.norm()

    cos_zm = (z @ m_hat).numpy()
    proj = (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    r = z - proj
    r_norms = r.norm(dim=1).numpy()

    import pandas as pd
    df = pd.DataFrame([{
        "N": N, "dim": 768, "R": R,
        "cos_zm_mean": float(cos_zm.mean()),
        "cos_zm_std": float(cos_zm.std()),
        "cos_zm_min": float(cos_zm.min()),
        "cos_zm_max": float(cos_zm.max()),
        "r_norm_mean": float(r_norms.mean()),
        "r_norm_std": float(r_norms.std()),
    }])
    _save_csv(df, OUT_DIR / "exp1_cone_geometry.csv")

    print(f"  R = {R:.6f}")
    print(f"  cos(z,m): mean={cos_zm.mean():.4f}, std={cos_zm.std():.4f}")
    print(f"  ||r||:    mean={r_norms.mean():.4f}, std={r_norms.std():.4f}")

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
#  Exp 2: Text Decomposition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp2(data, m_hat):
    print("\n" + "=" * 60)
    print("Exp 2: Text Embedding Decomposition")
    print("=" * 60)

    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])
    t_pos_raw = data["zs_text_pos"]
    t_neg_raw = data["zs_text_neg"]

    import pandas as pd
    rows = []
    for i, (cls, term) in enumerate(zip(CLASSES, TERMS)):
        cos_pn = (t_pos_n[i] @ t_neg_n[i]).item()

        dt = t_pos_raw[i] - t_neg_raw[i]
        dt_norm = dt.norm().item()

        dt_m_scalar = (dt @ m_hat).item()
        dt_m = dt_m_scalar * m_hat
        dt_r = dt - dt_m
        dt_m_norm = dt_m.norm().item()
        dt_r_norm = dt_r.norm().item()

        m_ratio = dt_m_norm / (dt_norm + 1e-8)
        r_ratio = dt_r_norm / (dt_norm + 1e-8)

        rows.append({
            "disease": cls, "term": term,
            "cos_pos_neg": cos_pn,
            "dt_norm": dt_norm,
            "dt_m_norm": dt_m_norm, "dt_r_norm": dt_r_norm,
            "m_ratio": m_ratio, "r_ratio": r_ratio,
            "proj_pos_on_m": (t_pos_n[i] @ m_hat).item(),
            "proj_neg_on_m": (t_neg_n[i] @ m_hat).item(),
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp2_text_decomposition.csv")
    print(df[["disease", "cos_pos_neg", "m_ratio", "r_ratio"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(14)

    axes[0].bar(x, df["cos_pos_neg"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("cos(t_pos, t_neg)")
    axes[0].set_title("Positive-Negative Text Embedding Similarity")
    axes[0].axhline(df["cos_pos_neg"].mean(), color="red", ls="--",
                    label=f'mean={df["cos_pos_neg"].mean():.4f}')
    axes[0].legend()

    w = 0.35
    axes[1].bar(x - w / 2, df["m_ratio"], w, label="m component", color="coral")
    axes[1].bar(x + w / 2, df["r_ratio"], w, label="r component", color="teal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("||Δt_comp|| / ||Δt||")
    axes[1].set_title("Direction of Pos-Neg Difference (m vs r)")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp2_text_decomposition.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 3: Negation vs Cone Position
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp3(data, z, m_hat):
    print("\n" + "=" * 60)
    print("Exp 3: Per-image Negation Discrimination vs Cone Position")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    cos_zm = (z @ m_hat).numpy()
    sim_pos = (z @ t_pos_n.T).numpy()
    sim_neg = (z @ t_neg_n.T).numpy()
    gap = sim_pos - sim_neg

    abs_gap_mean = np.abs(gap).mean(axis=1)
    rho_s, p_s = spearmanr(cos_zm, abs_gap_mean)
    print(f"  전체 Spearman: ρ={rho_s:.4f}, p={p_s:.2e}")

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        gap_d = gap[:, d]
        abs_gap_d = np.abs(gap_d)
        rho, p = spearmanr(cos_zm, abs_gap_d)

        pos_mask = labels[:, d] == 1
        neg_mask = labels[:, d] == 0

        correct = np.zeros(len(cos_zm), dtype=bool)
        correct[pos_mask] = gap_d[pos_mask] > 0
        correct[neg_mask] = gap_d[neg_mask] < 0

        median_cos = np.median(cos_zm)
        near_mask = cos_zm > median_cos
        far_mask = cos_zm <= median_cos
        acc_near = correct[near_mask].mean() if near_mask.sum() > 0 else np.nan
        acc_far = correct[far_mask].mean() if far_mask.sum() > 0 else np.nan

        rows.append({
            "disease": cls,
            "spearman_rho": rho, "spearman_p": p,
            "pos_count": int(pos_mask.sum()),
            "acc_near_axis": acc_near, "acc_far_axis": acc_far,
            "acc_diff": acc_far - acc_near,
            "auc_gap": _safe_auc(labels[:, d], gap_d),
            "mean_gap_pos": float(gap_d[pos_mask].mean()) if pos_mask.sum() > 0 else np.nan,
            "mean_gap_neg": float(gap_d[neg_mask].mean()) if neg_mask.sum() > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp3_negation_vs_cone.csv")
    print(df[["disease", "spearman_rho", "acc_near_axis", "acc_far_axis", "acc_diff", "auc_gap"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_plot = min(5000, len(cos_zm))
    idx = np.random.RandomState(42).choice(len(cos_zm), n_plot, replace=False)
    axes[0].scatter(cos_zm[idx], abs_gap_mean[idx], alpha=0.1, s=3)
    axes[0].set_xlabel("cos(z_i, m)")
    axes[0].set_ylabel("mean |gap|")
    axes[0].set_title(f"Cone Position vs Discrimination\nSpearman ρ={rho_s:.4f}")

    x = np.arange(14)
    colors = ["green" if v > 0 else "red" for v in df["acc_diff"]]
    axes[1].bar(x, df["acc_diff"], color=colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("acc_far - acc_near")
    axes[1].set_title("Accuracy Diff (positive = far better)")
    axes[1].axhline(0, color="black", lw=0.5)

    axes[2].bar(x, df["auc_gap"], color="steelblue", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("AUROC (gap as score)")
    axes[2].set_title("Negation-based Classification")
    axes[2].axhline(0.5, color="red", ls="--")

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp3_negation_vs_cone.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 4: Zero-shot vs Bi-MCQ Cone
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp4(data, zs_z, zs_m_hat, zs_R):
    print("\n" + "=" * 60)
    print("Exp 4: Zero-shot vs Bi-MCQ Cone Comparison")
    print("=" * 60)

    # Bi-MCQ cone
    z_bm = _norm(data["bimcq_img"])
    m_bm = z_bm.mean(dim=0)
    R_bm = m_bm.norm().item()
    m_hat_bm = m_bm / m_bm.norm()
    cos_zm_bm = (z_bm @ m_hat_bm).numpy()

    # Bi-MCQ text cos(pos, neg)
    tp_bm = _norm(data["bimcq_text_pos"])
    tn_bm = _norm(data["bimcq_text_neg"])
    cos_pn_bm = np.array([(tp_bm[i] @ tn_bm[i]).item() for i in range(14)])

    # Zero-shot text cos(pos, neg)
    tp_zs = _norm(data["zs_text_pos"])
    tn_zs = _norm(data["zs_text_neg"])
    cos_pn_zs = np.array([(tp_zs[i] @ tn_zs[i]).item() for i in range(14)])
    cos_zm_zs = (zs_z @ zs_m_hat).numpy()

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

    # 전체 요약 행 추가
    global_row = pd.DataFrame([{
        "disease": "_GLOBAL",
        "zs_cos_pos_neg": float(cos_pn_zs.mean()),
        "bimcq_cos_pos_neg": float(cos_pn_bm.mean()),
        "delta_cos_pos_neg": float(cos_pn_bm.mean() - cos_pn_zs.mean()),
    }])
    # cone R도 별도 행
    cone_row = pd.DataFrame([{
        "disease": "_CONE_R",
        "zs_cos_pos_neg": zs_R,
        "bimcq_cos_pos_neg": R_bm,
        "delta_cos_pos_neg": R_bm - zs_R,
    }])
    df = pd.concat([summary, global_row, cone_row], ignore_index=True)
    _save_csv(df, OUT_DIR / "exp4_zs_vs_bimcq.csv")

    print(f"  Zero-shot  R={zs_R:.4f}  cos(pos,neg) mean={cos_pn_zs.mean():.4f}")
    print(f"  Bi-MCQ     R={R_bm:.4f}  cos(pos,neg) mean={cos_pn_bm.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(cos_zm_zs, bins=80, alpha=0.5, density=True,
                 label=f"Zero-shot (R={zs_R:.4f})")
    axes[0].hist(cos_zm_bm, bins=80, alpha=0.5, density=True,
                 label=f"Bi-MCQ (R={R_bm:.4f})")
    axes[0].set_xlabel("cos(z_i, m)")
    axes[0].set_title("Cone Distribution: Zero-shot vs Bi-MCQ")
    axes[0].legend()

    x = np.arange(14)
    w = 0.35
    axes[1].bar(x - w / 2, cos_pn_zs, w, label="Zero-shot", alpha=0.8)
    axes[1].bar(x + w / 2, cos_pn_bm, w, label="Bi-MCQ", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("cos(t_pos, t_neg)")
    axes[1].set_title("Pos-Neg Text Similarity")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp4_zs_vs_bimcq.png")
    return {"R_bm": R_bm, "z_bm": z_bm, "m_hat_bm": m_hat_bm}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 5: Disease Cone vs AUC (zero-shot cosine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp5(data, z, m_hat):
    print("\n" + "=" * 60)
    print("Exp 5: Disease-level Cone Position vs AUC (zero-shot)")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    cos_zm = (z @ m_hat).numpy()
    r = z - (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    r_norms = r.norm(dim=1).numpy()

    sim_pos = (z @ t_pos_n.T).numpy()
    sim_neg = (z @ t_neg_n.T).numpy()
    gap = sim_pos - sim_neg

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        pos_mask = labels[:, d] == 1
        mean_cos_pos = cos_zm[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        mean_cos_all = cos_zm.mean()

        rows.append({
            "disease": cls,
            "pos_count": int(pos_mask.sum()),
            "pos_pct": pos_mask.sum() / len(labels) * 100,
            "mean_cos_pos": mean_cos_pos,
            "mean_cos_all": mean_cos_all,
            "cos_offset": mean_cos_pos - mean_cos_all,
            "mean_rnorm_pos": r_norms[pos_mask].mean() if pos_mask.sum() > 0 else np.nan,
            "mean_rnorm_all": r_norms.mean(),
            "auc_pos": _safe_auc(labels[:, d], sim_pos[:, d]),
            "auc_neg": _safe_auc(labels[:, d], -sim_neg[:, d]),
            "auc_pnc": _safe_auc(labels[:, d], gap[:, d]),
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp5_disease_cone_auc.csv")
    print(df[["disease", "pos_pct", "cos_offset", "auc_pos", "auc_neg", "auc_pnc"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    axes[0].bar(x, df["cos_offset"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("cos(pos_imgs, m) - cos(all, m)")
    axes[0].set_title("Positive Images: Cone Offset")
    axes[0].axhline(0, color="black", lw=0.5)

    axes[1].scatter(df["cos_offset"], df["auc_pnc"], s=60, zorder=5)
    for i, cls in enumerate(CLASSES):
        axes[1].annotate(cls, (df["cos_offset"].iloc[i], df["auc_pnc"].iloc[i]),
                         fontsize=7, ha="center", va="bottom")
    axes[1].set_xlabel("cos(pos_imgs, m) - cos(all, m)")
    axes[1].set_ylabel("PNC AUC")
    axes[1].set_title("Cone Offset vs Classification AUC")

    w = 0.25
    axes[2].bar(x - w, df["auc_pos"], w, label="POS", alpha=0.8)
    axes[2].bar(x, df["auc_neg"], w, label="NEG", alpha=0.8)
    axes[2].bar(x + w, df["auc_pnc"], w, label="PNC", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Zero-shot Classification AUCs")
    axes[2].axhline(0.5, color="red", ls="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp5_disease_cone_auc.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 6: m 제거 실험 (인과 증거)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp6(data, z, m_hat):
    """
    z' = z - (z·m̂)m̂  로 cone 축 성분을 제거한 뒤
    cosine 기반 PNC AUC가 올라가는지 확인.
    """
    print("\n" + "=" * 60)
    print("Exp 6: m-Removal Experiment (Causal Evidence)")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    # 원본 유사도
    sim_pos_orig = (z @ t_pos_n.T).numpy()
    sim_neg_orig = (z @ t_neg_n.T).numpy()
    gap_orig = sim_pos_orig - sim_neg_orig

    # m 제거
    proj = (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)  # (N, 768)
    z_rm = z - proj                                         # (N, 768)
    z_rm_n = _norm(z_rm)                                    # 재정규화

    # 텍스트에서도 m 제거 (동일 조건)
    t_pos_rm = t_pos_n - (t_pos_n @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    t_neg_rm = t_neg_n - (t_neg_n @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    t_pos_rm_n = _norm(t_pos_rm)
    t_neg_rm_n = _norm(t_neg_rm)

    sim_pos_rm = (z_rm_n @ t_pos_rm_n.T).numpy()
    sim_neg_rm = (z_rm_n @ t_neg_rm_n.T).numpy()
    gap_rm = sim_pos_rm - sim_neg_rm

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        y = labels[:, d]
        auc_pnc_orig = _safe_auc(y, gap_orig[:, d])
        auc_pnc_rm = _safe_auc(y, gap_rm[:, d])
        auc_pos_orig = _safe_auc(y, sim_pos_orig[:, d])
        auc_pos_rm = _safe_auc(y, sim_pos_rm[:, d])
        auc_neg_orig = _safe_auc(y, -sim_neg_orig[:, d])
        auc_neg_rm = _safe_auc(y, -sim_neg_rm[:, d])

        # gap의 분산도 체크 (m 제거 시 gap이 더 분리되는가)
        gap_std_orig = float(np.std(gap_orig[:, d]))
        gap_std_rm = float(np.std(gap_rm[:, d]))

        rows.append({
            "disease": cls,
            "auc_pnc_orig": auc_pnc_orig,
            "auc_pnc_m_removed": auc_pnc_rm,
            "delta_auc_pnc": auc_pnc_rm - auc_pnc_orig,
            "auc_pos_orig": auc_pos_orig,
            "auc_pos_m_removed": auc_pos_rm,
            "auc_neg_orig": auc_neg_orig,
            "auc_neg_m_removed": auc_neg_rm,
            "gap_std_orig": gap_std_orig,
            "gap_std_m_removed": gap_std_rm,
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp6_m_removal.csv")

    mean_orig = df["auc_pnc_orig"].mean()
    mean_rm = df["auc_pnc_m_removed"].mean()
    print(f"  PNC AUC 평균: 원본={mean_orig:.4f}  m제거={mean_rm:.4f}  Δ={mean_rm - mean_orig:+.4f}")
    print(df[["disease", "auc_pnc_orig", "auc_pnc_m_removed", "delta_auc_pnc"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    # (a) PNC AUC 비교: 원본 vs m 제거
    w = 0.35
    axes[0].bar(x - w / 2, df["auc_pnc_orig"], w, label="Original", alpha=0.8, color="steelblue")
    axes[0].bar(x + w / 2, df["auc_pnc_m_removed"], w, label="m removed", alpha=0.8, color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("PNC AUC")
    axes[0].set_title(f"PNC AUC: Original vs m-Removed\nmean Δ={mean_rm - mean_orig:+.4f}")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0].legend()

    # (b) ΔAUC bar
    colors = ["green" if v > 0 else "red" for v in df["delta_auc_pnc"]]
    axes[1].bar(x, df["delta_auc_pnc"], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("ΔAUC (m-removed − original)")
    axes[1].set_title("Per-disease AUC Change After m-Removal")
    axes[1].axhline(0, color="black", lw=0.5)

    # (c) gap std 비교
    axes[2].bar(x - w / 2, df["gap_std_orig"], w, label="Original", alpha=0.8, color="steelblue")
    axes[2].bar(x + w / 2, df["gap_std_m_removed"], w, label="m removed", alpha=0.8, color="coral")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("std(gap)")
    axes[2].set_title("Gap Variance: Original vs m-Removed")
    axes[2].legend()

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp6_m_removal.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 7: cos(pos,neg) vs AUC scatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp7(data, z, m_hat):
    """
    질환별 cos(pos,neg)와 zero-shot PNC AUC의 상관.
    cos(pos,neg)가 높을수록 PNC AUC가 낮은가?
    """
    print("\n" + "=" * 60)
    print("Exp 7: Text Similarity vs AUC Correlation")
    print("=" * 60)

    labels = data["labels"]
    t_pos_n = _norm(data["zs_text_pos"])
    t_neg_n = _norm(data["zs_text_neg"])

    cos_pn = np.array([(t_pos_n[i] @ t_neg_n[i]).item() for i in range(14)])

    sim_pos = (z @ t_pos_n.T).numpy()
    sim_neg = (z @ t_neg_n.T).numpy()
    gap = sim_pos - sim_neg

    import pandas as pd
    pnc_aucs = np.array([_safe_auc(labels[:, d], gap[:, d]) for d in range(14)])

    rho, p_val = spearmanr(cos_pn, pnc_aucs)
    r_pear, p_pear = pearsonr(cos_pn, pnc_aucs)

    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append({
            "disease": cls,
            "cos_pos_neg": cos_pn[i],
            "pnc_auc": pnc_aucs[i],
        })
    # 전체 통계 행
    rows.append({
        "disease": "_STATS",
        "cos_pos_neg": float(rho),       # Spearman ρ
        "pnc_auc": float(p_val),         # p-value
    })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp7_text_sim_vs_auc.csv")

    print(f"  Spearman: ρ={rho:.4f}, p={p_val:.4f}")
    print(f"  Pearson:  r={r_pear:.4f}, p={p_pear:.4f}")
    print(df.head(14)[["disease", "cos_pos_neg", "pnc_auc"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cos_pn, pnc_aucs, s=80, zorder=5, color="steelblue", edgecolors="white")
    for i, cls in enumerate(CLASSES):
        ax.annotate(cls, (cos_pn[i], pnc_aucs[i]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    # 회귀선
    if np.isfinite(cos_pn).all() and np.isfinite(pnc_aucs).all():
        coeffs = np.polyfit(cos_pn, pnc_aucs, 1)
        x_fit = np.linspace(cos_pn.min(), cos_pn.max(), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", alpha=0.5)

    ax.set_xlabel("cos(t_pos, t_neg)")
    ax.set_ylabel("Zero-shot PNC AUC")
    ax.set_title(f"Text Similarity vs Classification Performance\n"
                 f"Spearman ρ={rho:.4f} (p={p_val:.4f}), Pearson r={r_pear:.4f}")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    _save_fig(fig, OUT_DIR / "exp7_text_sim_vs_auc.png")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exp 8: Bi-MCQ cosine PNC AUC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def exp8(data, zs_z, zs_m_hat):
    """
    Bi-MCQ 임베딩에서 cosine 기반 POS/NEG/PNC AUC 계산.
    zero-shot과 동일 방식으로 비교 → cone 감소가 cosine 판별에 미치는 효과.
    """
    print("\n" + "=" * 60)
    print("Exp 8: Bi-MCQ Cosine-based AUC (Embedding Quality)")
    print("=" * 60)

    labels = data["labels"]

    # zero-shot cosine AUC
    zs_tp = _norm(data["zs_text_pos"])
    zs_tn = _norm(data["zs_text_neg"])
    zs_sim_pos = (zs_z @ zs_tp.T).numpy()
    zs_sim_neg = (zs_z @ zs_tn.T).numpy()
    zs_gap = zs_sim_pos - zs_sim_neg

    # Bi-MCQ cosine AUC
    bm_z = _norm(data["bimcq_img"])
    bm_tp = _norm(data["bimcq_text_pos"])
    bm_tn = _norm(data["bimcq_text_neg"])
    bm_sim_pos = (bm_z @ bm_tp.T).numpy()
    bm_sim_neg = (bm_z @ bm_tn.T).numpy()
    bm_gap = bm_sim_pos - bm_sim_neg

    import pandas as pd
    rows = []
    for d, cls in enumerate(CLASSES):
        y = labels[:, d]
        rows.append({
            "disease": cls,
            "zs_auc_pos": _safe_auc(y, zs_sim_pos[:, d]),
            "zs_auc_neg": _safe_auc(y, -zs_sim_neg[:, d]),
            "zs_auc_pnc": _safe_auc(y, zs_gap[:, d]),
            "bimcq_auc_pos": _safe_auc(y, bm_sim_pos[:, d]),
            "bimcq_auc_neg": _safe_auc(y, -bm_sim_neg[:, d]),
            "bimcq_auc_pnc": _safe_auc(y, bm_gap[:, d]),
            "delta_pnc": _safe_auc(y, bm_gap[:, d]) - _safe_auc(y, zs_gap[:, d]),
        })

    df = pd.DataFrame(rows)
    _save_csv(df, OUT_DIR / "exp8_bimcq_cosine_auc.csv")

    mean_zs = df["zs_auc_pnc"].mean()
    mean_bm = df["bimcq_auc_pnc"].mean()
    print(f"  PNC AUC 평균: zero-shot={mean_zs:.4f}  Bi-MCQ={mean_bm:.4f}  Δ={mean_bm - mean_zs:+.4f}")
    print(df[["disease", "zs_auc_pnc", "bimcq_auc_pnc", "delta_pnc"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(14)

    # (a) PNC AUC 비교
    w = 0.35
    axes[0].bar(x - w / 2, df["zs_auc_pnc"], w, label="Zero-shot", alpha=0.8, color="steelblue")
    axes[0].bar(x + w / 2, df["bimcq_auc_pnc"], w, label="Bi-MCQ", alpha=0.8, color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("PNC AUC (cosine)")
    axes[0].set_title(f"Cosine PNC AUC: ZS vs Bi-MCQ\nmean Δ={mean_bm - mean_zs:+.4f}")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0].legend()

    # (b) ΔAUC bar
    colors = ["green" if v > 0 else "red" for v in df["delta_pnc"]]
    axes[1].bar(x, df["delta_pnc"], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("ΔAUC (Bi-MCQ − ZS)")
    axes[1].set_title("Per-disease AUC Improvement")
    axes[1].axhline(0, color="black", lw=0.5)

    # (c) 3종 AUC — Bi-MCQ
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
def main():
    parser = argparse.ArgumentParser(description="Cone Effect Analysis (Exp 1-8)")
    parser.add_argument("--exp", type=str, default="all",
                        help="실험 번호 (예: 1,2,3 또는 all)")
    args = parser.parse_args()

    if args.exp == "all":
        exps = set(range(1, 9))
    else:
        exps = set(int(e) for e in args.exp.split(","))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {OUT_DIR}")

    # 임베딩 파일 확인
    required = ["labels.npy", "zs_img_emb_g.pt", "zs_text_pos.pt", "zs_text_neg.pt"]
    if exps & {4, 8}:
        required += ["bimcq_img_emb_g.pt", "bimcq_text_pos.pt", "bimcq_text_neg.pt"]
    for name in required:
        if not (EMB_DIR / name).exists():
            raise FileNotFoundError(
                f"임베딩 파일 없음: {EMB_DIR / name}\n"
                f"먼저 1_extract_embeddings.py 를 실행하세요."
            )

    data = load_embeddings()

    # ── Exp 1 ──
    if 1 in exps:
        exp1_out = exp1(data)
        z, m_hat, R = exp1_out["z"], exp1_out["m_hat"], exp1_out["R"]
    else:
        # 다른 실험에 z, m_hat 필요
        z = _norm(data["zs_img"])
        m = z.mean(dim=0)
        R = m.norm().item()
        m_hat = m / m.norm()

    # ── Exp 2 ──
    if 2 in exps:
        exp2(data, m_hat)

    # ── Exp 3 ──
    if 3 in exps:
        exp3(data, z, m_hat)

    # ── Exp 4 ──
    if 4 in exps:
        exp4(data, z, m_hat, R)

    # ── Exp 5 ──
    if 5 in exps:
        exp5(data, z, m_hat)

    # ── Exp 6 (신규) ──
    if 6 in exps:
        exp6(data, z, m_hat)

    # ── Exp 7 (신규) ──
    if 7 in exps:
        exp7(data, z, m_hat)

    # ── Exp 8 (신규) ──
    if 8 in exps:
        exp8(data, z, m_hat)

    # ── 완료 ──
    print("\n" + "=" * 60)
    print("완료! 결과 파일:")
    for f in sorted(OUT_DIR.glob("exp*")):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

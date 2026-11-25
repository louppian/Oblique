#!/usr/bin/env python3
"""
Cone Effect → Negation Discrimination Failure: Experiments 1-5
=============================================================
CARZero zero-shot 모델에서 이미지 임베딩의 cone 구조를 측정하고,
그것이 긍정/부정 프롬프트 판별 실패와 어떻게 연결되는지 실험.

사용법:
    python cone_experiments.py --exp 1      # Exp 1만 실행
    python cone_experiments.py --exp 1,2,3  # Exp 1,2,3 실행
    python cone_experiments.py --exp all    # 전체 실행

서버 경로:
    /shared/home/mai/Taehun/Uncertainty/ 기준
"""

from pathlib import Path
import os
import sys
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ── 경로 설정 ──────────────────────────────────────────
BASE = Path("/shared/home/mai/Taehun/Uncertainty")
PROJECT = BASE / "BI_MCQ"
WEIGHT_ROOT = BASE / "MICCAI_2025/CARZero"

# CARZero zero-shot 가중치
CKPT_ZEROSHOT = WEIGHT_ROOT / "checkpoints/best_model.ckpt"
# Bi-MCQ fine-tuned 가중치
CKPT_BIMCQ = WEIGHT_ROOT / "checkpoints/BiMCQ_CARZero_best_model.ckpt"
# M3AE pretrain (ViT 초기화에 필요)
M3AE = WEIGHT_ROOT / "pretrain_model/VITB-16-M3AE_last.ckpt"

# NIH 데이터
IMAGE_ROOT = BASE / "data/NIH"
LABEL_CSV = IMAGE_ROOT / "Data_Entry_2017.csv"
TEST_CSV = IMAGE_ROOT / "NIH_test_data.csv"

# 출력
OUTPUT_DIR = PROJECT / "analysis/cone"

DEVICE = "cuda:0"
BATCH_SIZE = 256   # VRAM에 맞게 조절

# 14개 질환
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]

# 프롬프트에 사용하는 용어 (CARZero/Bi-MCQ 관례)
TERMS = [
    "Atelectasis", "Cardiomegaly", "Pleural Effusion",
    "Pulmonary Infiltration", "Pulmonary Mass", "Lung Nodule",
    "Pneumonia", "Pneumothorax", "Pulmonary Consolidation",
    "Pulmonary Edema", "Pulmonary Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]


# ── 데이터 로드 ────────────────────────────────────────
def load_data():
    """NIH test set 이미지 경로와 라벨 로드."""
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

    paths = [str(image_map[name]) for name in names]
    print(f"[DATA] 테스트 이미지 {len(paths):,}개 / 질환 14개")
    return paths, labels, names


# ── 모델 로드 ──────────────────────────────────────────
def _import_carzero():
    """CARZero 모듈 import 및 repo 경로 반환."""
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


def load_carzero_model(checkpoint_path, device=DEVICE):
    """CARZero 모델을 zero-shot 모드로 로드.

    서버의 load_CARZero가 config.yaml을 요구하는 경우를 대비해,
    checkpoint의 hyper_parameters에서 cfg를 직접 추출하여 모델을 빌드.
    """
    CARZero_module, repo = _import_carzero()
    from CARZero import builder

    prev = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        cfg = ckpt["hyper_parameters"]
        ckpt_dict = ckpt["state_dict"]

        model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(cfg)

        expected = model.state_dict()
        fixed = {}
        for k, v in ckpt_dict.items():
            # 접두사 자동 감지: CARZero. 또는 CARZero_model.
            new_key = k
            for prefix in ["CARZero_model.", "CARZero."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            if new_key in expected:
                fixed[new_key] = v

        missing = set(expected.keys()) - set(fixed.keys())
        if missing:
            print(f"  [WARN] {len(missing)}개 키 누락 (strict=False로 로드)")
            model.load_state_dict(fixed, strict=False)
        else:
            model.load_state_dict(fixed, strict=True)
    finally:
        os.chdir(prev)

    model.to(device).eval()
    print(f"[MODEL] CARZero 로드 완료: {checkpoint_path.name}")
    return model, CARZero_module


# ── 임베딩 추출 ────────────────────────────────────────
@torch.no_grad()
def extract_image_embeddings(model, paths, device=DEVICE, batch_size=BATCH_SIZE):
    """
    이미지 임베딩 추출 (pre-cross-attention).
    Returns:
        img_emb_g: (N, 768) global embeddings (ViT [CLS], 정규화 전)
    """
    all_g = []
    for i in tqdm(range(0, len(paths), batch_size), desc="이미지 임베딩"):
        batch_paths = paths[i:i+batch_size]
        imgs = model.process_img(batch_paths, device)
        img_emb_l, img_emb_g = model.image_encoder_forward(imgs)
        all_g.append(img_emb_g.cpu())
        del imgs, img_emb_l, img_emb_g
        torch.cuda.empty_cache()
    return torch.cat(all_g, dim=0)  # (N, 768)


@torch.no_grad()
def extract_text_embeddings(model, prompts, device=DEVICE):
    """
    텍스트 임베딩 추출 (pre-cross-attention).
    Returns:
        text_emb_g: (len(prompts), 768) global embeddings
    """
    processed = model.process_text(prompts, device)
    text_emb_l, text_emb_g, sents = model.text_encoder_forward(
        processed["caption_ids"],
        processed["attention_mask"],
        processed["token_type_ids"],
    )
    return text_emb_g.cpu()  # (P, 768)


# ── Exp 1: Cone 기하학 측정 ────────────────────────────
def exp1_cone_geometry(img_emb_g, output_dir):
    """
    이미지 임베딩의 cone 구조 측정.
    - R = ||mean(z_i)|| where z_i are L2-normalized
    - cos(z_i, m) 분포
    """
    print("\n" + "="*60)
    print("Exp 1: Cone Geometry")
    print("="*60)

    # L2 정규화
    z = F.normalize(img_emb_g, dim=1)  # (N, 768)
    N = z.shape[0]

    # mean과 R
    m = z.mean(dim=0)  # (768,)
    R = m.norm().item()
    m_hat = m / m.norm()

    # cos(z_i, m) 분포
    cos_zm = (z @ m_hat).numpy()  # (N,)

    # r_i = z_i - (z_i · m_hat) * m_hat
    proj = (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)  # (N, 768)
    r = z - proj
    r_norms = r.norm(dim=1).numpy()  # (N,)

    results = {
        "N": N,
        "dim": 768,
        "R": R,
        "cos_zm_mean": float(cos_zm.mean()),
        "cos_zm_std": float(cos_zm.std()),
        "cos_zm_min": float(cos_zm.min()),
        "cos_zm_max": float(cos_zm.max()),
        "r_norm_mean": float(r_norms.mean()),
        "r_norm_std": float(r_norms.std()),
    }
    print(f"  R = ||mean(z_i)|| = {R:.6f}")
    print(f"  cos(z_i, m): mean={cos_zm.mean():.6f}, std={cos_zm.std():.6f}")
    print(f"  ||r_i||: mean={r_norms.mean():.6f}, std={r_norms.std():.6f}")

    # 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(cos_zm, bins=100, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("cos(z_i, m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Image Embedding Cone (R={R:.4f})")
    axes[0].axvline(cos_zm.mean(), color="red", linestyle="--", label=f"mean={cos_zm.mean():.4f}")
    axes[0].legend()

    axes[1].hist(r_norms, bins=100, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("||r_i|| (residual norm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Component Distribution")
    axes[1].axvline(r_norms.mean(), color="red", linestyle="--", label=f"mean={r_norms.mean():.4f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_cone_geometry.png", dpi=150)
    plt.close()

    with open(output_dir / "exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  → {output_dir / 'exp1_cone_geometry.png'}")
    return z, m, m_hat, R, results


# ── Exp 2: 텍스트 임베딩 긍정/부정 분해 ───────────────
def exp2_text_decomposition(model, m_hat, output_dir, device=DEVICE):
    """
    긍정/부정 프롬프트의 텍스트 임베딩을 이미지 cone 축(m) 기준으로 분해.
    - cos(t_pos, t_neg) per disease
    - Δt = t_pos - t_neg 의 m 성분 vs r 성분
    """
    print("\n" + "="*60)
    print("Exp 2: Text Embedding Decomposition")
    print("="*60)

    pos_prompts = [f"There is {t}" for t in TERMS]
    neg_prompts = [f"There is no {t}" for t in TERMS]

    t_pos = extract_text_embeddings(model, pos_prompts, device)  # (14, 768)
    t_neg = extract_text_embeddings(model, neg_prompts, device)  # (14, 768)

    # L2 정규화
    t_pos_n = F.normalize(t_pos, dim=1)
    t_neg_n = F.normalize(t_neg, dim=1)

    rows = []
    for i, (cls, term) in enumerate(zip(CLASSES, TERMS)):
        # cos(t_pos, t_neg)
        cos_pn = (t_pos_n[i] @ t_neg_n[i]).item()

        # Δt = t_pos - t_neg (정규화 전 raw 차이)
        dt = t_pos[i] - t_neg[i]
        dt_norm = dt.norm().item()

        # m 축 방향 분해
        dt_m_scalar = (dt @ m_hat).item()
        dt_m = dt_m_scalar * m_hat
        dt_r = dt - dt_m

        dt_m_norm = dt_m.norm().item()
        dt_r_norm = dt_r.norm().item()

        # 비율
        m_ratio = dt_m_norm / (dt_norm + 1e-8)
        r_ratio = dt_r_norm / (dt_norm + 1e-8)

        # 각 임베딩의 m축 투영
        proj_pos = (t_pos_n[i] @ m_hat).item()
        proj_neg = (t_neg_n[i] @ m_hat).item()

        rows.append({
            "disease": cls,
            "term": term,
            "cos_pos_neg": cos_pn,
            "dt_norm": dt_norm,
            "dt_m_norm": dt_m_norm,
            "dt_r_norm": dt_r_norm,
            "m_ratio": m_ratio,
            "r_ratio": r_ratio,
            "proj_pos_on_m": proj_pos,
            "proj_neg_on_m": proj_neg,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp2_text_decomposition.csv", index=False)

    print(df[["disease", "cos_pos_neg", "m_ratio", "r_ratio"]].to_string(index=False))
    print(f"\n  cos(pos, neg) 평균: {df['cos_pos_neg'].mean():.6f}")
    print(f"  Δt의 m 비율 평균: {df['m_ratio'].mean():.6f}")
    print(f"  Δt의 r 비율 평균: {df['r_ratio'].mean():.6f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(CLASSES))

    axes[0].bar(x, df["cos_pos_neg"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("cos(t_pos, t_neg)")
    axes[0].set_title("Positive-Negative Text Embedding Similarity")
    axes[0].axhline(df["cos_pos_neg"].mean(), color="red", linestyle="--",
                    label=f'mean={df["cos_pos_neg"].mean():.4f}')
    axes[0].legend()

    width = 0.35
    axes[1].bar(x - width/2, df["m_ratio"], width, label="m component", color="coral")
    axes[1].bar(x + width/2, df["r_ratio"], width, label="r component", color="teal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("||Δt_component|| / ||Δt||")
    axes[1].set_title("Direction of Pos-Neg Difference (m vs r)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_text_decomposition.png", dpi=150)
    plt.close()
    print(f"  → {output_dir / 'exp2_text_decomposition.png'}")

    return t_pos, t_neg, t_pos_n, t_neg_n, df


# ── Exp 3: 이미지별 부정 판별력 vs Cone 위치 ──────────
def exp3_negation_vs_cone(z, m_hat, t_pos_n, t_neg_n, labels, output_dir):
    """
    이미지별:
    - cone 위치: cos(z_i, m)
    - 부정 판별 간격: cos(z_i, t_pos) - cos(z_i, t_neg)
    → 축에 가까운 이미지일수록 판별 간격이 작은가?
    """
    print("\n" + "="*60)
    print("Exp 3: Per-image Negation Discrimination vs Cone Position")
    print("="*60)

    cos_zm = (z @ m_hat).numpy()  # (N,)

    # 각 질환에 대해 cos(z_i, t_pos_d) - cos(z_i, t_neg_d)
    # z: (N, 768), t_pos_n/t_neg_n: (14, 768)
    sim_pos = (z @ t_pos_n.T).numpy()  # (N, 14)
    sim_neg = (z @ t_neg_n.T).numpy()  # (N, 14)
    gap = sim_pos - sim_neg            # (N, 14) — 양수면 pos가 더 가까움

    # 전체 aggregate
    abs_gap_mean = np.abs(gap).mean(axis=1)  # (N,) 14개 질환 평균 |gap|

    from scipy.stats import spearmanr, pearsonr

    rho_s, p_s = spearmanr(cos_zm, abs_gap_mean)
    rho_p, p_p = pearsonr(cos_zm, abs_gap_mean)
    print(f"  [전체] cos(z,m) vs mean|gap|:")
    print(f"    Spearman: ρ={rho_s:.4f}, p={p_s:.2e}")
    print(f"    Pearson:  r={rho_p:.4f}, p={p_p:.2e}")

    # 질환별 상세
    disease_results = []
    for d, cls in enumerate(CLASSES):
        gap_d = gap[:, d]     # (N,)
        abs_gap_d = np.abs(gap_d)

        rho, p = spearmanr(cos_zm, abs_gap_d)

        # 양성 이미지 vs 음성 이미지의 cone 위치
        pos_mask = labels[:, d] == 1
        neg_mask = labels[:, d] == 0

        # 올바른 판별 비율
        # 양성이면 gap > 0이 정답, 음성이면 gap < 0이 정답
        correct = np.zeros(len(cos_zm), dtype=bool)
        correct[pos_mask] = gap_d[pos_mask] > 0
        correct[neg_mask] = gap_d[neg_mask] < 0

        # cone 축에 가까운 이미지 (cos > median) vs 먼 이미지에서 정답률 비교
        median_cos = np.median(cos_zm)
        near_mask = cos_zm > median_cos
        far_mask = cos_zm <= median_cos

        acc_near = correct[near_mask].mean() if near_mask.sum() > 0 else np.nan
        acc_far = correct[far_mask].mean() if far_mask.sum() > 0 else np.nan

        # AUC (gap을 점수로 사용한 분류 성능)
        try:
            auc_gap = roc_auc_score(labels[:, d], gap_d)
        except ValueError:
            auc_gap = np.nan

        disease_results.append({
            "disease": cls,
            "spearman_rho": rho,
            "spearman_p": p,
            "pos_count": int(pos_mask.sum()),
            "acc_near_axis": acc_near,
            "acc_far_axis": acc_far,
            "acc_diff": acc_far - acc_near,  # 양수면 축에서 먼 것이 더 정확
            "auc_gap": auc_gap,
            "mean_gap_pos": float(gap_d[pos_mask].mean()) if pos_mask.sum() > 0 else np.nan,
            "mean_gap_neg": float(gap_d[neg_mask].mean()) if neg_mask.sum() > 0 else np.nan,
        })

    df_disease = pd.DataFrame(disease_results)
    df_disease.to_csv(output_dir / "exp3_negation_vs_cone.csv", index=False)

    print("\n질환별 결과:")
    print(df_disease[["disease", "spearman_rho", "spearman_p", "acc_near_axis", "acc_far_axis", "acc_diff", "auc_gap"]].to_string(index=False))

    # 시각화 1: scatter (전체)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) cos(z,m) vs |gap| — 랜덤 서브샘플
    n_plot = min(5000, len(cos_zm))
    idx = np.random.choice(len(cos_zm), n_plot, replace=False)
    axes[0].scatter(cos_zm[idx], abs_gap_mean[idx], alpha=0.1, s=3)
    axes[0].set_xlabel("cos(z_i, m) — cone position")
    axes[0].set_ylabel("mean |gap| (pos-neg discrimination)")
    axes[0].set_title(f"Cone Position vs Discrimination\nSpearman ρ={rho_s:.4f}")

    # (b) 질환별 acc_diff bar
    x = np.arange(len(CLASSES))
    colors = ["green" if v > 0 else "red" for v in df_disease["acc_diff"]]
    axes[1].bar(x, df_disease["acc_diff"], color=colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("acc_far - acc_near")
    axes[1].set_title("Accuracy Difference\n(positive = far from axis is more accurate)")
    axes[1].axhline(0, color="black", linewidth=0.5)

    # (c) 질환별 AUC from gap
    axes[2].bar(x, df_disease["auc_gap"], color="steelblue", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("AUROC (using pos-neg gap as score)")
    axes[2].set_title("Negation-based Classification")
    axes[2].axhline(0.5, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_negation_vs_cone.png", dpi=150)
    plt.close()
    print(f"  → {output_dir / 'exp3_negation_vs_cone.png'}")

    return df_disease


# ── Exp 4: CARZero vs Bi-MCQ Cone 비교 ────────────────
def exp4_carzero_vs_bimcq(paths, labels, output_dir, device=DEVICE):
    """
    Bi-MCQ fine-tuned 모델에서 같은 cone 측정을 반복.
    pre-cross-attention 임베딩에서 cone이 변했는지 확인.
    """
    print("\n" + "="*60)
    print("Exp 4: CARZero vs Bi-MCQ Cone Comparison")
    print("="*60)

    if not CKPT_BIMCQ.exists():
        print(f"  [SKIP] Bi-MCQ 가중치 없음: {CKPT_BIMCQ}")
        return None

    from omegaconf import OmegaConf

    config_path = PROJECT / "configs/chest14_finetuning_Bi-MCQ.yaml"
    if not config_path.exists():
        print(f"  [SKIP] Bi-MCQ config 없음: {config_path}")
        return None

    CARZero_module, repo = _import_carzero()
    cfg = OmegaConf.load(config_path)

    prev = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)
        # cfg를 사용해 multi=True 로드 시도 (사용자 로컬 수정 버전)
        try:
            model = CARZero_module.load_CARZero(
                name=str(WEIGHT_ROOT / "pretrain_model/CARZero_best_model.ckpt"),
                device="cpu",
                multi=True,
                cfg=cfg,
            )
        except TypeError:
            # load_CARZero에 multi/cfg 파라미터가 없는 경우 — 수동 빌드
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

    # Bi-MCQ 가중치 로드
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
        print(f"  [WARN] Bi-MCQ {len(missing)}개 키 누락 (strict=False로 로드)")
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("  Bi-MCQ 가중치 로드 완료")

    # 이미지 임베딩 추출
    bimcq_emb_g = extract_image_embeddings(model, paths, device)

    # Cone 측정
    z_bimcq = F.normalize(bimcq_emb_g, dim=1)
    m_bimcq = z_bimcq.mean(dim=0)
    R_bimcq = m_bimcq.norm().item()
    m_hat_bimcq = m_bimcq / m_bimcq.norm()
    cos_zm_bimcq = (z_bimcq @ m_hat_bimcq).numpy()

    # 텍스트 임베딩도 추출
    pos_prompts = [f"There is {t}" for t in TERMS]
    neg_prompts = [f"There is no {t}" for t in TERMS]
    t_pos_bimcq = extract_text_embeddings(model, pos_prompts, device)
    t_neg_bimcq = extract_text_embeddings(model, neg_prompts, device)
    t_pos_n_bimcq = F.normalize(t_pos_bimcq, dim=1)
    t_neg_n_bimcq = F.normalize(t_neg_bimcq, dim=1)

    cos_pn_bimcq = [(t_pos_n_bimcq[i] @ t_neg_n_bimcq[i]).item() for i in range(14)]

    results = {
        "R_bimcq": R_bimcq,
        "cos_zm_mean_bimcq": float(cos_zm_bimcq.mean()),
        "cos_zm_std_bimcq": float(cos_zm_bimcq.std()),
        "cos_pn_mean_bimcq": float(np.mean(cos_pn_bimcq)),
    }

    with open(output_dir / "exp4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Bi-MCQ R = {R_bimcq:.6f}")
    print(f"  Bi-MCQ cos(z,m) mean = {cos_zm_bimcq.mean():.6f}")
    print(f"  Bi-MCQ cos(pos,neg) mean = {np.mean(cos_pn_bimcq):.6f}")

    del model, bimcq_emb_g
    torch.cuda.empty_cache()

    return results, z_bimcq, m_hat_bimcq, t_pos_n_bimcq, t_neg_n_bimcq


# ── Exp 5: 질환별 Cone 위치와 AUC ─────────────────────
def exp5_disease_cone_vs_auc(z, m_hat, t_pos_n, t_neg_n, labels, output_dir):
    """
    질환별:
    - 해당 질환 양성 이미지들의 평균 cos(z_i, m)
    - 양성 이미지들의 r_norm 평균
    - POS AUC, NEG AUC, PNC AUC (gap 기반)
    """
    print("\n" + "="*60)
    print("Exp 5: Disease-level Cone Position vs AUC")
    print("="*60)

    cos_zm = (z @ m_hat).numpy()  # (N,)
    r = z - (z @ m_hat).unsqueeze(1) * m_hat.unsqueeze(0)
    r_norms = r.norm(dim=1).numpy()  # (N,)

    sim_pos = (z @ t_pos_n.T).numpy()  # (N, 14)
    sim_neg = (z @ t_neg_n.T).numpy()  # (N, 14)
    gap = sim_pos - sim_neg

    rows = []
    for d, cls in enumerate(CLASSES):
        pos_mask = labels[:, d] == 1

        # 양성 이미지의 cone 위치
        mean_cos_pos = cos_zm[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        mean_cos_all = cos_zm.mean()
        mean_rnorm_pos = r_norms[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        mean_rnorm_all = r_norms.mean()

        # AUC: pos similarity as score
        try:
            auc_pos = roc_auc_score(labels[:, d], sim_pos[:, d])
        except ValueError:
            auc_pos = np.nan

        # AUC: neg similarity as score (낮을수록 양성이면 1-neg)
        try:
            auc_neg = roc_auc_score(labels[:, d], -sim_neg[:, d])
        except ValueError:
            auc_neg = np.nan

        # AUC: gap as score
        try:
            auc_pnc = roc_auc_score(labels[:, d], gap[:, d])
        except ValueError:
            auc_pnc = np.nan

        # 빈도
        pos_pct = pos_mask.sum() / len(labels) * 100

        rows.append({
            "disease": cls,
            "pos_count": int(pos_mask.sum()),
            "pos_pct": pos_pct,
            "mean_cos_pos": mean_cos_pos,
            "mean_cos_all": mean_cos_all,
            "cos_pos_minus_all": mean_cos_pos - mean_cos_all,
            "mean_rnorm_pos": mean_rnorm_pos,
            "mean_rnorm_all": mean_rnorm_all,
            "auc_pos": auc_pos,
            "auc_neg": auc_neg,
            "auc_pnc": auc_pnc,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp5_disease_cone_auc.csv", index=False)

    print(df[["disease", "pos_pct", "mean_cos_pos", "cos_pos_minus_all", "auc_pos", "auc_neg", "auc_pnc"]].to_string(index=False))

    from scipy.stats import spearmanr

    # 상관분석: cos_pos_minus_all vs AUC
    valid = df.dropna(subset=["cos_pos_minus_all", "auc_pnc"])
    if len(valid) > 3:
        rho, p = spearmanr(valid["cos_pos_minus_all"], valid["auc_pnc"])
        print(f"\n  cos(pos_imgs, m) - cos(all, m) vs PNC AUC:")
        print(f"    Spearman ρ={rho:.4f}, p={p:.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 양성 이미지의 cone 위치 vs 전체
    x = np.arange(len(CLASSES))
    axes[0].bar(x, df["cos_pos_minus_all"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("cos(pos_imgs, m) - cos(all, m)")
    axes[0].set_title("Positive Images: Cone Offset from Mean")
    axes[0].axhline(0, color="black", linewidth=0.5)

    # (b) scatter: cos_pos_minus_all vs auc_pnc
    axes[1].scatter(df["cos_pos_minus_all"], df["auc_pnc"], s=60, zorder=5)
    for i, cls in enumerate(CLASSES):
        axes[1].annotate(cls, (df["cos_pos_minus_all"].iloc[i], df["auc_pnc"].iloc[i]),
                        fontsize=7, ha="center", va="bottom")
    axes[1].set_xlabel("cos(pos_imgs, m) - cos(all, m)")
    axes[1].set_ylabel("PNC AUC (gap-based)")
    axes[1].set_title("Cone Offset vs Classification AUC")

    # (c) 3종 AUC 비교
    width = 0.25
    axes[2].bar(x - width, df["auc_pos"], width, label="POS AUC", alpha=0.8)
    axes[2].bar(x, df["auc_neg"], width, label="NEG AUC", alpha=0.8)
    axes[2].bar(x + width, df["auc_pnc"], width, label="PNC AUC", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Zero-shot Classification AUCs")
    axes[2].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "exp5_disease_cone_auc.png", dpi=150)
    plt.close()
    print(f"  → {output_dir / 'exp5_disease_cone_auc.png'}")

    return df


# ── 메인 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cone Effect Experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="실험 번호 (1,2,3,4,5 또는 all)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size

    if args.exp == "all":
        exps = {1, 2, 3, 4, 5}
    else:
        exps = set(int(e) for e in args.exp.split(","))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {OUTPUT_DIR}")

    # 데이터 로드
    paths, labels, names = load_data()

    # CARZero zero-shot 로드 (Exp 1-3, 5에 필요)
    if exps & {1, 2, 3, 5}:
        model, CARZero_module = load_carzero_model(CKPT_ZEROSHOT, device)

        # 이미지 임베딩 추출 (캐시)
        emb_cache = OUTPUT_DIR / "img_emb_g_zeroshot.pt"
        if emb_cache.exists():
            print(f"[CACHE] 이미지 임베딩 로드: {emb_cache}")
            img_emb_g = torch.load(emb_cache, map_location="cpu")
        else:
            img_emb_g = extract_image_embeddings(model, paths, device, batch_size)
            torch.save(img_emb_g, emb_cache)
            print(f"[CACHE] 이미지 임베딩 저장: {emb_cache}")

    # ── Exp 1 ──
    if 1 in exps:
        z, m, m_hat, R, exp1_results = exp1_cone_geometry(img_emb_g, OUTPUT_DIR)
    elif exps & {2, 3, 5}:
        # Exp 2,3,5에서 m_hat 필요
        z = F.normalize(img_emb_g, dim=1)
        m = z.mean(dim=0)
        m_hat = m / m.norm()

    # ── Exp 2 ──
    if 2 in exps:
        t_pos, t_neg, t_pos_n, t_neg_n, exp2_df = exp2_text_decomposition(
            model, m_hat, OUTPUT_DIR, device)
    elif exps & {3, 5}:
        # Exp 3,5에서 t_pos_n, t_neg_n 필요
        pos_prompts = [f"There is {t}" for t in TERMS]
        neg_prompts = [f"There is no {t}" for t in TERMS]
        t_pos = extract_text_embeddings(model, pos_prompts, device)
        t_neg = extract_text_embeddings(model, neg_prompts, device)
        t_pos_n = F.normalize(t_pos, dim=1)
        t_neg_n = F.normalize(t_neg, dim=1)

    # ── Exp 3 ──
    if 3 in exps:
        exp3_df = exp3_negation_vs_cone(z, m_hat, t_pos_n, t_neg_n, labels, OUTPUT_DIR)

    # ── Exp 4 ──
    if 4 in exps:
        exp4_result = exp4_carzero_vs_bimcq(paths, labels, OUTPUT_DIR, device)

        # zero-shot과 비교 시각화
        if exp4_result is not None:
            exp4_data, z_bimcq, m_hat_bimcq, t_pos_n_bimcq, t_neg_n_bimcq = exp4_result

            # Exp 1 결과 로드
            if not (1 in exps):
                with open(OUTPUT_DIR / "exp1_results.json") as f:
                    exp1_results = json.load(f)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # zero-shot
            if exps & {1}:
                cos_zm_zs = (z @ m_hat).numpy()
            else:
                z_zs = F.normalize(img_emb_g, dim=1)
                m_zs = z_zs.mean(dim=0)
                cos_zm_zs = (z_zs @ (m_zs / m_zs.norm())).numpy()

            cos_zm_bm = (z_bimcq @ m_hat_bimcq).numpy()

            axes[0].hist(cos_zm_zs, bins=80, alpha=0.5, label=f"Zero-shot (R={exp1_results['R']:.4f})", density=True)
            axes[0].hist(cos_zm_bm, bins=80, alpha=0.5, label=f"Bi-MCQ (R={exp4_data['R_bimcq']:.4f})", density=True)
            axes[0].set_xlabel("cos(z_i, m)")
            axes[0].set_title("Cone Distribution: Zero-shot vs Bi-MCQ")
            axes[0].legend()

            # cos(pos, neg) 비교
            if 2 in exps:
                cos_pn_zs = exp2_df["cos_pos_neg"].values
            else:
                pos_prompts = [f"There is {t}" for t in TERMS]
                neg_prompts = [f"There is no {t}" for t in TERMS]
                tp = extract_text_embeddings(model, pos_prompts, device)
                tn = extract_text_embeddings(model, neg_prompts, device)
                tp_n = F.normalize(tp, dim=1)
                tn_n = F.normalize(tn, dim=1)
                cos_pn_zs = np.array([(tp_n[i] @ tn_n[i]).item() for i in range(14)])

            cos_pn_bm = np.array([(t_pos_n_bimcq[i] @ t_neg_n_bimcq[i]).item() for i in range(14)])
            x = np.arange(14)
            w = 0.35
            axes[1].bar(x - w/2, cos_pn_zs, w, label="Zero-shot", alpha=0.8)
            axes[1].bar(x + w/2, cos_pn_bm, w, label="Bi-MCQ", alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
            axes[1].set_ylabel("cos(t_pos, t_neg)")
            axes[1].set_title("Pos-Neg Text Similarity")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "exp4_comparison.png", dpi=150)
            plt.close()
            print(f"  → {OUTPUT_DIR / 'exp4_comparison.png'}")

    # ── Exp 5 ──
    if 5 in exps:
        exp5_df = exp5_disease_cone_vs_auc(z, m_hat, t_pos_n, t_neg_n, labels, OUTPUT_DIR)

    # 정리
    if exps & {1, 2, 3, 5}:
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("완료! 결과:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f}")
    print("="*60)


if __name__ == "__main__":
    main()
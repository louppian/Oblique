#!/usr/bin/env python3
"""
2_cone_analysis.py — Bias-Corrected Cone Analysis for CARZero/Bi-MCQ
=====================================================================

Core insight
------------
Canonical geometry: b is the cone vertex, r is the image-embedding
mean/base-center estimate, and m is the axis line through b and r.
The estimates are b_hat = normalized mean noise embedding, r_hat = mean
normalized image embedding, and m_hat = normalize(r_hat - b_hat).
For orthogonal decomposition, axis_basis removes the b_hat component
from m_hat; it is a basis vector, not the geometric axis itself.

For normalized image embeddings z_i and disease text difference Δt_d:
  z_i = r_hat + δ_i
  gap_id = r_hat·Δt_d + δ_i·Δt_d
The centered image deviation δ_i is decomposed along b_hat, axis_basis,
and image-specific residual δ'_i. The hypothesis that the first three
terms are non-discriminative is tested empirically; it is not implied
by the algebra alone.

Hypothesis → Experiment map
----------------------------
  H0  노이즈 → 인코더 → 꼭지점 (cone의 기원)
      Exp  1  Image noise → ViT → b̂_img
      Exp  2  Text noise  → BioBERT → b̂_text
      Exp  3  OOD baseline (MNIST + non-medical text)

  H1  의료 데이터도 cone을 형성하고, 축이 존재한다
      Exp  4  Image cone 구조
      Exp  5  Text cone 구조

  H2  gap 분해 → cone 구조가 상수항으로 지배
      Exp  6  Per-disease gap 기초 통계
      Exp  7  Gap 4-term 분해

  H3  AUC는 residual space 분리도가 결정
      Exp  8  Content separation vs AUC

  H4  꼭지점·축 제거는 AUC에 거의 영향 없다
      Exp  9  Bias removal intervention

  H5  Δt는 cone 밖에 살고, "no"는 텍스트를 꼭지점으로 민다
      Exp 10  Δt geometry (image + text cone projection)
      Exp 11  "no" insertion → text movement analysis

  H6  Bi-MCQ 학습은 cone을 풀고 content 정렬을 높인다
      Exp 12  ZS vs Bi-MCQ 비교

  종합
      Exp 13  Summary correlations

GPU_EXPS (1, 2, 3) require model loading;
set RUN_GPU_EXPS = False below to skip them.
"""

import os, sys, struct, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Paths  (edit if your layout differs)
# ──────────────────────────────────────────────────────────────────────────
BASE        = Path("/shared/home/mai/Taehun/Uncertainty")
PROJECT     = BASE / "BI_MCQ"
WEIGHT_ROOT = BASE / "MICCAI_2025/CARZero"

BASE_DIR    = PROJECT / "analysis/cone"
EMB_DIR     = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH   = WEIGHT_ROOT / "checkpoints/best_model.ckpt"
CARZERO_SRC = PROJECT / "CARZero"
MNIST_DIR   = Path("/shared/home/mai/Taehun/Uncertainty/"
                   "Uncertain-Aware Learning/mnist")

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def compute_pnc_auc(D):
    """Compute PNC AUC from embeddings: gap = cos(z, t+) - cos(z, t-) per disease.
    Returns dict {disease_name: AUC}."""
    from sklearn.metrics import roc_auc_score
    z  = F.normalize(D["zs_img"].float(),      dim=-1)
    tp = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    lb = D["labels"]
    auc = {}
    for di, dis in enumerate(DISEASES):
        gap = (z @ tp[di] - z @ tn[di]).numpy()
        y   = lb[:, di]
        try:
            auc[dis] = roc_auc_score(y, gap)
        except ValueError:
            auc[dis] = np.nan
    return auc

# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def load_embeddings():
    """Load pre-extracted embeddings (from 1_extract_embeddings.py)."""
    d = {}
    # zero-shot  (legacy fallback: img_emb_g_zeroshot.pt)
    zs_img_path = EMB_DIR / "zs_img_emb_g.pt"
    if not zs_img_path.exists():
        zs_img_path = EMB_DIR / "img_emb_g_zeroshot.pt"
    d["zs_img"]      = torch.load(zs_img_path,                   map_location="cpu")
    d["zs_text_pos"] = torch.load(EMB_DIR / "zs_text_pos.pt",    map_location="cpu")
    d["zs_text_neg"] = torch.load(EMB_DIR / "zs_text_neg.pt",    map_location="cpu")
    # Bi-MCQ
    d["bimcq_img"]      = torch.load(EMB_DIR / "bimcq_img_emb_g.pt", map_location="cpu")
    d["bimcq_text_pos"] = torch.load(EMB_DIR / "bimcq_text_pos.pt",  map_location="cpu")
    d["bimcq_text_neg"] = torch.load(EMB_DIR / "bimcq_text_neg.pt",  map_location="cpu")
    d["labels"] = np.load(EMB_DIR / "labels.npy")
    return d


def R_metric(emb: torch.Tensor) -> float:
    """R = ‖mean(x̂_i)‖  where x̂_i = x_i/‖x_i‖.
    R → 1 : tight cone.   R → 0 : uniform on sphere."""
    return F.normalize(emb.float(), dim=-1).mean(0).norm().item()


def estimate_cone_axis(r_hat: torch.Tensor, b_hat: torch.Tensor):
    """Return m-hat and an orthogonal axis basis from vertex b-hat to base center r-hat.

    m is the line through b and r; m-hat is its unit direction.  axis_basis
    is m-hat with its b-hat component removed, used only for orthogonal
    component decomposition.
    """
    b = F.normalize(b_hat.float(), dim=0)
    axis_vec = r_hat.float() - b
    if axis_vec.norm() < 1e-8:
        raise ValueError("Cone axis is undefined: r_hat and b_hat coincide.")
    m_hat = F.normalize(axis_vec, dim=0)
    axis_raw = m_hat - torch.dot(m_hat, b) * b
    if axis_raw.norm() < 1e-8:
        raise ValueError("Cannot separate vertex and axis: m_hat is parallel to b_hat.")
    axis_basis = F.normalize(axis_raw, dim=0)
    return m_hat, axis_basis


def _save_csv(df, name):
    p = RESULTS_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  → {p}")

def _save_fig(fig, name):
    p = RESULTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ──────────────────────────────────────────────────────────────────────────
# Model helpers  (used only in Exp 1 / 2 / 3)
# ──────────────────────────────────────────────────────────────────────────

def _import_carzero():
    """기존 코드와 동일: PROJECT 아래에서 CARZero 소스를 찾아 sys.path에 추가."""
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


def _load_full_model():
    """Load CARZero model (ViT + BioBERT) from checkpoint."""
    CARZero_module, repo = _import_carzero()
    from CARZero import builder

    prev = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
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

    model.to(DEVICE).eval()
    return model


def _vit_encode(model, imgs: torch.Tensor) -> torch.Tensor:
    """imgs [N,3,224,224] → img_emb_g [N,768]"""
    parts = []
    with torch.no_grad():
        for i in range(0, len(imgs), BATCH_SIZE):
            _, g = model.image_encoder_forward(imgs[i:i+BATCH_SIZE].to(DEVICE))
            parts.append(g.cpu())
    return torch.cat(parts, 0)


def _text_encode(model, texts: list) -> torch.Tensor:
    """list[str] → text_emb_g [N,768]"""
    parts = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            inp   = model.process_text(batch, DEVICE)
            _, g, _ = model.text_encoder_forward(
                inp["caption_ids"], inp["attention_mask"], inp["token_type_ids"])
            parts.append(g.cpu())
    return torch.cat(parts, 0)


def _read_mnist(n=1000):
    """MNIST idx 파일에서 이미지 로드 → (N, 3, 224, 224) 텐서.  (기존 코드 동일)"""
    import gzip as _gzip

    img_path = MNIST_DIR / "train-images-idx3-ubyte"
    gz_path  = MNIST_DIR / "train-images-idx3-ubyte.gz"

    if img_path.exists():
        with open(img_path, "rb") as f:
            raw = f.read()
    elif gz_path.exists():
        with _gzip.open(gz_path, "rb") as f:
            raw = f.read()
    else:
        raise FileNotFoundError(f"MNIST 이미지 파일 없음: {img_path} 또는 {gz_path}")

    # IDX 포맷 파싱
    magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
    data = np.frombuffer(raw[16:], dtype=np.uint8).reshape(num_images, rows, cols)

    # n개 샘플링
    indices = np.random.RandomState(42).choice(num_images, min(n, num_images), replace=False)
    images = data[indices]  # (n, 28, 28)

    # 28x28 → 224x224, 1ch → 3ch, normalize
    imgs_t = torch.from_numpy(images).float() / 255.0  # (n, 28, 28)
    imgs_t = imgs_t.unsqueeze(1)                        # (n, 1, 28, 28)
    imgs_t = imgs_t.repeat(1, 3, 1, 1)                  # (n, 3, 28, 28)
    imgs_t = F.interpolate(imgs_t, size=(224, 224), mode="bilinear", align_corners=False)

    # ImageNet normalization
    mu  = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    sig = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_t = (imgs_t - mu) / sig

    print(f"[MNIST] {imgs_t.shape[0]}개 이미지 로드 (28→224, 1ch→3ch)")
    return imgs_t


# ══════════════════════════════════════════════════════════════════════════
# H0: 노이즈 → 인코더 → 꼭지점 (cone의 기원)
# ══════════════════════════════════════════════════════════════════════════

def exp_1():
    """Exp 1: Random noise → ViT → R_noise, save b̂_img.
    무의미 입력이 고정점(꼭지점)으로 수렴함을 보인다."""
    print("=" * 60)
    print("Exp 1  Random Noise → ViT  (Image Bias Direction)")
    print("=" * 60)

    model = _load_full_model()

    torch.manual_seed(42)
    noise = torch.randn(1000, 3, 224, 224)
    mu  = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    sig = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    noise = (noise - mu) / sig

    emb = _vit_encode(model, noise)                              # [1000, 768]
    R   = R_metric(emb)

    emb_n    = F.normalize(emb.float(), dim=-1)
    b_hat    = emb_n.mean(0)
    b_hat    = b_hat / b_hat.norm()
    torch.save(b_hat, EMB_DIR / "b_hat_img.pt")

    cos_vals = (emb_n @ b_hat).numpy()
    print(f"  R_noise = {R:.6f}")
    print(f"  cos(emb, b̂)  mean={cos_vals.mean():.4f}  std={cos_vals.std():.6f}")
    print(f"  Saved b̂_img → {EMB_DIR / 'b_hat_img.pt'}")

    _save_csv(pd.DataFrame([dict(source="random_noise", N=1000, R=R,
                                  cos_mean=cos_vals.mean(), cos_std=cos_vals.std())]),
              "exp_1_img_noise_bias")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cos_vals, bins=50, edgecolor="k", alpha=.7, color="steelblue")
    ax.axvline(cos_vals.mean(), c="red", ls="--", label=f"mean={cos_vals.mean():.4f}")
    ax.set(xlabel="cos(noise_emb, b̂_img)", ylabel="count",
           title=f"Exp 1 — Noise → ViT  (R={R:.4f})")
    ax.legend()
    _save_fig(fig, "exp_1_img_noise_bias")

    del model; torch.cuda.empty_cache()


def exp_2():
    """Exp 2: Random text → BioBERT → R, save b̂_text.
    A: char-level gibberish  (500)  — broad charset, wide length
    B: token-level gibberish (500)  — random vocab words spliced
    텍스트 인코더에도 동일한 꼭지점이 존재함을 보인다."""
    print("\n" + "=" * 60)
    print("Exp 2  Random Text → BioBERT  (Text Bias Direction)")
    print("=" * 60)

    model = _load_full_model()

    # ── A: char-level gibberish (500) ────────────────────────
    random.seed(42)
    chars_wide = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,;:!?-+=/\\@#$%&*()[]{}~"
    )
    gib_char = [
        "".join(random.choice(chars_wide)
                for _ in range(random.randint(20, 80)))
        for _ in range(500)
    ]

    # ── B: token-level gibberish (500) ───────────────────────
    word_pool = [
        "apple","neutron","cactus","flicker","prism","quartz","wagon",
        "myth","oxygen","cobalt","delta","ember","frost","glyph","helix",
        "ivory","jungle","karma","lemon","mosaic","nebula","orchid",
        "pulse","ripple","sigma","thorn","umbra","vortex","wax","zenith",
        "blaze","cipher","dusk","echo","flux","grove","haze","iris",
        "jade","knot","latch","mirth","nook","opal","plume","reef",
        "sable","tide","umber","vale","whisk","yarn","zinc","abyss",
        "bolt","crux","dew","elm","fang","gust","husk","ion","jolt",
        "keel","lobe","musk","nib","ore","pith","quill","rind","sap",
        "tuft","urge","vim","wren","yew","zeal","arc","brim","cleft",
    ]
    gib_token = [
        " ".join(random.choice(word_pool)
                 for _ in range(random.randint(2, 15)))
        for _ in range(500)
    ]

    # ── encode ───────────────────────────────────────────────
    noise_texts = gib_char + gib_token                     # 1000

    emb = _text_encode(model, noise_texts)

    n_a, n_b = 500, 500

    R_char  = R_metric(emb[:n_a])
    R_token = R_metric(emb[n_a:])
    R_noise = R_metric(emb)

    # b̂_text from all 1000 noise samples
    noise_n = F.normalize(emb.float(), dim=-1)
    b_hat_t = noise_n.mean(0); b_hat_t = b_hat_t / b_hat_t.norm()
    torch.save(b_hat_t, EMB_DIR / "b_hat_text.pt")

    print(f"  R_char_gib     = {R_char:.6f}")
    print(f"  R_token_gib    = {R_token:.6f}")
    print(f"  R_noise (A+B)  = {R_noise:.6f}")
    print(f"  Saved b̂_text → {EMB_DIR / 'b_hat_text.pt'}")

    _save_csv(pd.DataFrame([
        dict(source="char_gibberish",  N=n_a,   R=R_char),
        dict(source="token_gibberish", N=n_b,   R=R_token),
        dict(source="noise_all",       N=1000,  R=R_noise),
    ]), "exp_2_text_noise_bias")

    # ── plot ─────────────────────────────────────────────────
    cos_all = (noise_n @ b_hat_t).numpy()

    # debug: cos가 낮은 char_gib 샘플 확인
    cos_char = cos_all[:n_a]
    low_idx = np.argsort(cos_char)[:20]
    print("\n  [DEBUG] char_gib — lowest cos samples:")
    for i in low_idx:
        print(f"    cos={cos_char[i]:.4f}  len={len(gib_char[i]):3d}  text='{gib_char[i]}'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cos_all[:n_a],   bins=40, alpha=.6,
            label=f"char gib   R={R_char:.4f}",  color="red")
    ax.hist(cos_all[n_a:],   bins=40, alpha=.6,
            label=f"token gib  R={R_token:.4f}", color="orange")
    ax.set(xlabel="cos(text_emb, b̂_text)", ylabel="count",
           title="Exp 2 — Text Encoder Bias Direction")
    ax.legend()
    _save_fig(fig, "exp_2_text_noise_bias")

    del model; torch.cuda.empty_cache()


def exp_3():
    """Exp 3: OOD baselines — MNIST → ViT, non-medical text → BioBERT.
    도메인 무관 입력도 같은 꼭지점으로 수렴함을 보인다 (Exp 1/2와 대조)."""
    print("\n" + "=" * 60)
    print("Exp 3  OOD Baselines (MNIST + Non-Medical Text)")
    print("=" * 60)

    model = _load_full_model()
    b_img  = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    b_text = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")

    # ── MNIST → ViT ─────────────────────────────────────────
    imgs      = _read_mnist(1000)
    emb_mnist = _vit_encode(model, imgs)
    R_mnist   = R_metric(emb_mnist)
    emb_mn    = F.normalize(emb_mnist.float(), dim=-1)
    cos_mnist = (emb_mn @ b_img).numpy()

    print(f"  R_mnist = {R_mnist:.6f}")
    print(f"  cos(MNIST, b̂_img)  mean={cos_mnist.mean():.4f}  std={cos_mnist.std():.4f}")

    # ── Non-medical text → BioBERT ──────────────────────────
    random.seed(123)
    ood_templates = [
        "the {} {} the {} {}",
        "{} and {} went to the {}",
        "a {} of {} on the {}",
        "{} is better than {}",
        "yesterday the {} was {}",
        "how to {} a {} quickly",
        "the {} {} near the {}",
        "{} {} {} in the {}",
    ]
    ood_nouns = [
        "cat","dog","house","river","mountain","car","book","garden",
        "bridge","forest","ocean","window","guitar","planet","robot",
        "bicycle","dragon","castle","diamond","feather","hammer","island",
        "jacket","kitchen","ladder","mirror","notebook","painting","queen",
        "rocket","sandwich","telescope","umbrella","village","waterfall",
    ]
    ood_verbs = [
        "jumped","crossed","found","built","watched","carried","opened",
        "discovered","followed","created","measured","designed","explored",
    ]
    ood_adjs = [
        "small","ancient","bright","cold","dark","elegant","fast",
        "gentle","heavy","invisible","loud","mysterious","narrow",
    ]
    all_words = ood_nouns + ood_verbs + ood_adjs
    non_med = []
    for _ in range(1000):
        tmpl = random.choice(ood_templates)
        n_slots = tmpl.count("{}")
        words = [random.choice(all_words) for _ in range(n_slots)]
        non_med.append(tmpl.format(*words))

    emb_nm    = _text_encode(model, non_med)
    R_nm      = R_metric(emb_nm)
    emb_nm_n  = F.normalize(emb_nm.float(), dim=-1)
    cos_nm    = (emb_nm_n @ b_text).numpy()

    print(f"  R_non_med = {R_nm:.6f}")
    print(f"  cos(non_med, b̂_text)  mean={cos_nm.mean():.4f}  std={cos_nm.std():.4f}")

    _save_csv(pd.DataFrame([
        dict(source="MNIST",           N=len(emb_mnist), R=R_mnist,
             mean_cos_b=cos_mnist.mean(), std_cos_b=cos_mnist.std()),
        dict(source="non_medical_ood",  N=len(emb_nm),    R=R_nm,
             mean_cos_b=cos_nm.mean(),    std_cos_b=cos_nm.std()),
    ]), "exp_3_ood_baseline")

    # ── plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cos_mnist, bins=50, edgecolor="k", alpha=.7, color="orange")
    axes[0].axvline(cos_mnist.mean(), c="red", ls="--",
                    label=f"mean={cos_mnist.mean():.4f}")
    axes[0].set(xlabel="cos(MNIST_emb, b̂_img)", ylabel="count",
                title=f"MNIST → ViT  (R={R_mnist:.4f})")
    axes[0].legend()

    axes[1].hist(cos_nm, bins=50, edgecolor="k", alpha=.7, color="green")
    axes[1].axvline(cos_nm.mean(), c="red", ls="--",
                    label=f"mean={cos_nm.mean():.4f}")
    axes[1].set(xlabel="cos(non_med_emb, b̂_text)", ylabel="count",
                title=f"Non-Medical Text → BioBERT  (R={R_nm:.4f})")
    axes[1].legend()

    fig.suptitle("Exp 3 — OOD Baselines", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_3_ood_baseline")

    del model; torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# H1: 의료 데이터도 cone을 형성하고, 축이 존재한다
# ══════════════════════════════════════════════════════════════════════════

def exp_4():
    """Exp 4: Image cone structure — R, axis direction, δ decomposition.
    r_hat is the image-embedding base-center estimate.
    m_hat = normalize(r_hat - b_hat) is the geometric axis direction.
    axis_dir is its orthogonalized basis for the component decomposition.
    z = r_hat + delta; delta = delta_b + delta_axis + delta_prime."""
    print("\n" + "=" * 60)
    print("Exp 4  Image Cone Structure, Bias-Data Alignment & Axis Direction")
    print("=" * 60)

    D       = load_embeddings()
    b_img   = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    b_text  = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")

    zn   = F.normalize(D["zs_img"].float(),      dim=-1)
    tpn  = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tnn  = F.normalize(D["zs_text_neg"].float(),  dim=-1)

    R_xray = R_metric(D["zs_img"])
    R_tp   = R_metric(D["zs_text_pos"])
    R_tn   = R_metric(D["zs_text_neg"])

    # 밑중심 r_hat (이미지 임베딩 평균) 및 그 방향
    r_hat  = zn.mean(0)                                            # [768]
    R_val  = r_hat.norm().item()
    r_hat_dir = r_hat / r_hat.norm()
    m_tp   = F.normalize(tpn.mean(0, keepdim=True), dim=-1).squeeze()
    m_tn   = F.normalize(tnn.mean(0, keepdim=True), dim=-1).squeeze()

    # 기하학적 축 m_hat = normalize(r_hat - b_hat); axis_dir는 분해용 직교 기저
    m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

    pairs = {
        "cos(b̂_img, r̂_xray_dir)": (b_img,    r_hat_dir),
        "cos(b̂_text, m̂_t_pos)":   (b_text,   m_tp),
        "cos(b̂_text, m̂_t_neg)":   (b_text,   m_tn),
        "cos(b̂_img, b̂_text)":     (b_img,    b_text),
        "cos(r̂_xray_dir, m̂_t_pos)": (r_hat_dir, m_tp),
        "cos(b̂_img, m̂_t_pos)":    (b_img,    m_tp),
        "cos(b̂_text, r̂_xray_dir)": (b_text, r_hat_dir),
        "cos(b̂_img, m̂_axis)":     (b_img, m_hat),
        "cos(m̂_axis, r̂_xray_dir)": (m_hat, r_hat_dir),
        "cos(b̂_img, e_m)":         (b_img, axis_dir),
        "cos(axis_dir, m̂_t_pos)":  (axis_dir, m_tp),
    }
    vals = {k: (a @ b).item() for k, (a, b) in pairs.items()}

    # z = r_hat + delta, 3-way decompose r
    r_all      = zn - r_hat                                        # [N, 768]
    r_b_vals   = (r_all @ b_img).numpy()                           # r·b̂_img [N]
    r_a_vals   = (r_all @ axis_dir).numpy()                        # r·axis  [N]
    r_prime    = (r_all
                  - torch.from_numpy(r_b_vals).unsqueeze(1).float() * b_img
                  - torch.from_numpy(r_a_vals).unsqueeze(1).float() * axis_dir)
    rp_norm    = r_prime.norm(dim=-1).numpy()

    rows = [dict(metric="R_xray",            value=R_xray),
            dict(metric="R_text_pos",        value=R_tp),
            dict(metric="R_text_neg",        value=R_tn),
            dict(metric="‖m‖ (=R)",          value=R_val),
            dict(metric="r·b̂_img mean",      value=float(r_b_vals.mean())),
            dict(metric="r·b̂_img std",       value=float(r_b_vals.std())),
            dict(metric="r·axis mean",       value=float(r_a_vals.mean())),
            dict(metric="r·axis std",        value=float(r_a_vals.std())),
            dict(metric="‖r'‖ mean",         value=float(rp_norm.mean())),
            dict(metric="‖r'‖ std",          value=float(rp_norm.std()))]
    rows += [dict(metric=k, value=v) for k, v in vals.items()]
    _save_csv(pd.DataFrame(rows), "exp_4_img_cone_structure")

    for r in rows:
        print(f"  {r['metric']:30s} = {r['value']:.6f}")

    # ── plot ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # (a) R bar
    r_dict = {"noise": None, "MNIST": None,
              "X-ray": R_xray, "t_pos": R_tp, "t_neg": R_tn}
    try:
        r_dict["noise"] = pd.read_csv(RESULTS_DIR / "exp_1_img_noise_bias.csv")["R"].iloc[0]
    except Exception:
        pass
    try:
        df3 = pd.read_csv(RESULTS_DIR / "exp_3_ood_baseline.csv")
        mnist_row = df3[df3["source"] == "MNIST"]
        if len(mnist_row):
            r_dict["MNIST"] = mnist_row["R"].iloc[0]
    except Exception:
        pass
    r_dict = {k: v for k, v in r_dict.items() if v is not None}
    cmap = {"noise": "red", "MNIST": "orange"}
    colors = [cmap.get(n, "steelblue") for n in r_dict]
    axes[0].barh(list(r_dict.keys()), list(r_dict.values()),
                 color=colors, edgecolor="k")
    axes[0].set_xlabel("R"); axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Cone Concentration R")
    for i, (n, v) in enumerate(r_dict.items()):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=8)

    # (b) r projections histogram
    axes[1].hist(r_b_vals, bins=50, alpha=.6,
                 label=f"r·b̂_img  μ={r_b_vals.mean():.4f}", color="salmon")
    axes[1].hist(r_a_vals, bins=50, alpha=.6,
                 label=f"r·axis   μ={r_a_vals.mean():.4f}", color="steelblue")
    axes[1].set(xlabel="projection", ylabel="count",
                title="r decomposition: 꼭지점 vs 축 projections")
    axes[1].legend(fontsize=7)

    # (c) ‖r'‖ histogram
    axes[2].hist(rp_norm, bins=50, edgecolor="k", alpha=.7, color="green")
    axes[2].axvline(rp_norm.mean(), c="red", ls="--",
                    label=f"mean={rp_norm.mean():.4f}")
    axes[2].set(xlabel="‖r'‖ (이미지 고유 잔차)", ylabel="count",
                title="Image-specific residual norm")
    axes[2].legend()

    # (d) alignment matrix (5×5)
    dirs  = [b_img, b_text, r_hat_dir, m_hat, axis_dir]
    names = ["b̂_img", "b̂_text", "r̂_xray dir", "m̂ axis", "e_m basis"]
    mat   = np.array([[(a @ b).item() for b in dirs] for a in dirs])
    im = axes[3].imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_xticks(range(5)); axes[3].set_xticklabels(names, fontsize=7, rotation=45)
    axes[3].set_yticks(range(5)); axes[3].set_yticklabels(names, fontsize=7)
    for i in range(5):
        for j in range(5):
            axes[3].text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
    axes[3].set_title("Direction Alignment")
    fig.colorbar(im, ax=axes[3], shrink=.8)

    fig.suptitle("Exp 4 — Image Cone Structure, Bias & Axis", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, "exp_4_img_cone_structure")


def exp_5():
    """Exp 5: Text cone structure — axis, δ decomposition, t⁺ vs t⁻ position.
    텍스트 임베딩(t⁺, t⁻)도 b̂_text 중심의 cone을 형성하며,
    t⁻(부정)가 t⁺(긍정)보다 꼭지점에 더 가까움을 보인다."""
    print("\n" + "=" * 60)
    print("Exp 5  Text Cone Structure (t⁺ vs t⁻ in Text Cone)")
    print("=" * 60)

    D      = load_embeddings()
    b_text = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")
    b_img  = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")

    tp = F.normalize(D["zs_text_pos"].float(), dim=-1)   # [14, 768]
    tn = F.normalize(D["zs_text_neg"].float(), dim=-1)   # [14, 768]

    # ── Text cone 구조 ──────────────────────────────────────
    all_text    = torch.cat([tp, tn], dim=0)              # [28, 768]
    r_hat_text  = all_text.mean(0)                        # [768]
    R_text_all  = r_hat_text.norm().item()
    r_hat_text_dir = r_hat_text / r_hat_text.norm()

    m_hat_text, axis_dir_text = estimate_cone_axis(r_hat_text, b_text)

    # Image axis (cross-cone 비교용)
    zn         = F.normalize(D["zs_img"].float(), dim=-1)
    r_hat_img  = zn.mean(0)
    m_hat_img, axis_dir_img = estimate_cone_axis(r_hat_img, b_img)

    # ── δ decomposition: t = r_hat_text + δ ─────────────────
    delta_all = all_text - r_hat_text                     # [28, 768]
    delta_b   = (delta_all @ b_text)                      # [28]
    delta_a   = (delta_all @ axis_dir_text)               # [28]
    delta_prime = (delta_all
                   - delta_b.unsqueeze(1) * b_text
                   - delta_a.unsqueeze(1) * axis_dir_text)  # [28, 768]

    # t⁺ = indices 0:14,  t⁻ = indices 14:28
    cos_tp_b = (tp @ b_text).numpy()
    cos_tn_b = (tn @ b_text).numpy()

    # ── per-disease table ────────────────────────────────────
    rows = []
    for di, dis in enumerate(DISEASES):
        rows.append(dict(
            disease=dis,
            cos_tpos_b_text=cos_tp_b[di],
            cos_tneg_b_text=cos_tn_b[di],
            delta_cos_b_text=cos_tn_b[di] - cos_tp_b[di],
            delta_b_tpos=delta_b[di].item(),
            delta_b_tneg=delta_b[14 + di].item(),
            delta_axis_tpos=delta_a[di].item(),
            delta_axis_tneg=delta_a[14 + di].item(),
            delta_prime_norm_tpos=delta_prime[di].norm().item(),
            delta_prime_norm_tneg=delta_prime[14 + di].norm().item(),
            cos_tpos_tneg=(tp[di] @ tn[di]).item(),
        ))

    df = pd.DataFrame(rows)

    # ── summary ──────────────────────────────────────────────
    summary_rows = [
        dict(metric="R_text_all (tp+tn)",        value=R_text_all),
        dict(metric="R_text_pos",                value=R_metric(D["zs_text_pos"])),
        dict(metric="R_text_neg",                value=R_metric(D["zs_text_neg"])),
        dict(metric="cos(b̂_text, r̂_text_dir)",  value=(b_text @ r_hat_text_dir).item()),
        dict(metric="cos(b̂_text, b̂_img)",       value=(b_text @ b_img).item()),
        dict(metric="cos(m̂_text, m̂_img)",       value=(m_hat_text @ m_hat_img).item()),
        dict(metric="cos(axis_text, axis_img)",  value=(axis_dir_text @ axis_dir_img).item()),
        dict(metric="cos(t⁺, b̂_text) mean",     value=float(cos_tp_b.mean())),
        dict(metric="cos(t⁻, b̂_text) mean",     value=float(cos_tn_b.mean())),
        dict(metric="delta_cos mean (t⁻ - t⁺)",  value=float((cos_tn_b - cos_tp_b).mean())),
    ]
    _save_csv(pd.DataFrame(summary_rows), "exp_5_text_cone_summary")
    _save_csv(df, "exp_5_text_cone_per_disease")

    for r in summary_rows:
        print(f"  {r['metric']:35s} = {r['value']:.6f}")
    print()
    for _, r in df.iterrows():
        print(f"  {r['disease']:20s}  cos(t⁺,b̂)={r['cos_tpos_b_text']:.4f}  "
              f"cos(t⁻,b̂)={r['cos_tneg_b_text']:.4f}  "
              f"Δcos={r['delta_cos_b_text']:+.4f}")

    # ── plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # (a) cos(t, b̂_text): t⁺ vs t⁻ paired scatter
    axes[0].scatter(cos_tp_b, cos_tn_b, s=60, edgecolors="k", c="steelblue")
    for _, r in df.iterrows():
        axes[0].annotate(r["disease"][:4],
                         (r["cos_tpos_b_text"], r["cos_tneg_b_text"]), fontsize=7)
    lim = [min(cos_tp_b.min(), cos_tn_b.min()) - 0.02,
           max(cos_tp_b.max(), cos_tn_b.max()) + 0.02]
    axes[0].plot(lim, lim, "k--", lw=.5, alpha=.5)
    axes[0].set(xlabel="cos(t⁺, b̂_text)", ylabel="cos(t⁻, b̂_text)",
                title="t⁻ closer to vertex than t⁺")

    # (b) δ decomposition: t⁺ vs t⁻ bar chart
    xi = np.arange(len(DISEASES)); w = .35
    axes[1].bar(xi - w/2, df["delta_b_tpos"],  w, label="t⁺ δ·b̂", color="steelblue")
    axes[1].bar(xi + w/2, df["delta_b_tneg"],  w, label="t⁻ δ·b̂", color="coral")
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[1].set_ylabel("δ · b̂_text"); axes[1].legend(fontsize=7)
    axes[1].axhline(0, c="k", lw=.5)
    axes[1].set_title("Vertex projection: t⁺ vs t⁻")

    # (c) residual norm: t⁺ vs t⁻
    axes[2].bar(xi - w/2, df["delta_prime_norm_tpos"], w,
                label="t⁺ ‖δ'‖", color="steelblue")
    axes[2].bar(xi + w/2, df["delta_prime_norm_tneg"], w,
                label="t⁻ ‖δ'‖", color="coral")
    axes[2].set_xticks(xi)
    axes[2].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[2].set_ylabel("‖δ'‖ (text residual)"); axes[2].legend(fontsize=7)
    axes[2].set_title("Text residual norm: t⁺ vs t⁻")

    # (d) cross-cone alignment matrix
    dirs  = [b_text, b_img, r_hat_text_dir, m_hat_text, axis_dir_text]
    names = ["b̂_text", "b̂_img", "r̂_text", "m̂_text", "e_text"]
    mat   = np.array([[(a @ b).item() for b in dirs] for a in dirs])
    im = axes[3].imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_xticks(range(5)); axes[3].set_xticklabels(names, fontsize=7, rotation=45)
    axes[3].set_yticks(range(5)); axes[3].set_yticklabels(names, fontsize=7)
    for i in range(5):
        for j in range(5):
            axes[3].text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
    axes[3].set_title("Text Cone Direction Alignment")
    fig.colorbar(im, ax=axes[3], shrink=.8)

    fig.suptitle("Exp 5 — Text Cone Structure", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_5_text_cone_structure")


# ══════════════════════════════════════════════════════════════════════════
# H2: gap 분해 → cone 구조가 상수항으로 지배
# ══════════════════════════════════════════════════════════════════════════

def exp_6():
    """Exp 6: Per-disease zero-shot gap statistics."""
    print("\n" + "=" * 60)
    print("Exp 6  Per-Disease Zero-Shot Gap")
    print("=" * 60)

    D  = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    z  = F.normalize(D["zs_img"].float(),      dim=-1)
    tp = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    lb = D["labels"]

    rows = []
    for di, dis in enumerate(DISEASES):
        gap = (z @ tp[di] - z @ tn[di]).numpy()
        mp  = lb[:, di] == 1;  mn = lb[:, di] == 0
        rows.append(dict(
            disease=dis,
            gap_mean=gap.mean(), gap_std=gap.std(),
            gap_pos_mean=gap[mp].mean() if mp.sum() else np.nan,
            gap_neg_mean=gap[mn].mean() if mn.sum() else np.nan,
            positive_rate=(gap > 0).mean(),
            PNC_AUC=pnc_auc[dis]))

    df = pd.DataFrame(rows)
    _save_csv(df, "exp_6_per_disease_gap")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ds = df.sort_values("gap_mean")
    axes[0].barh(ds["disease"], ds["gap_mean"], xerr=ds["gap_std"],
                 color="steelblue", edgecolor="k", alpha=.7)
    axes[0].set_xlabel("gap mean ± std"); axes[0].set_title("Per-Disease Gap")

    rho, p = stats.spearmanr(df["gap_mean"], df["PNC_AUC"])
    print(f"  PNC AUC (computed from embeddings):")
    for _, r in df.iterrows():
        print(f"    {r['disease']:20s}  AUC={r['PNC_AUC']:.4f}")
    axes[1].scatter(df["gap_mean"], df["PNC_AUC"], s=60, edgecolors="k", c="steelblue")
    for _, r in df.iterrows():
        axes[1].annotate(r["disease"][:4], (r["gap_mean"], r["PNC_AUC"]), fontsize=7)
    axes[1].set(xlabel="gap mean", ylabel="PNC AUC",
                title=f"Gap vs AUC  ρ={rho:.3f}  p={p:.4f}")
    fig.suptitle("Exp 6 — Per-Disease Gap", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_6_per_disease_gap")


def exp_7():
    """Exp 7: Four-term gap decomposition around the image base-center estimate.
    z = r_hat + delta_i. Decompose centered delta_i and Δt in the
    orthogonal basis (b_hat, axis_dir), then reconstruct the gap."""
    print("\n" + "=" * 60)
    print("Exp 7  4-Term Gap Decomposition (m + 꼭지점 + 축 + 잔차)")
    print("=" * 60)

    D     = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    b_img = torch.load(EMB_DIR / "b_hat_img.pt", map_location="cpu")
    z     = F.normalize(D["zs_img"].float(),      dim=-1)
    tp    = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn    = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    dt    = tp - tn                                              # Δt  [14,768]

    # z = r_hat + delta
    r_hat = z.mean(0)                                            # [768]
    r     = z - r_hat                                            # [N, 768]

    m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

    # 3-way decompose r
    r_b     = r @ b_img                                          # [N]
    r_a     = r @ axis_dir                                       # [N]
    r_prime = (r - r_b.unsqueeze(1) * b_img
                 - r_a.unsqueeze(1) * axis_dir)                  # [N, 768]

    # 3-way decompose Δt
    dt_b     = dt @ b_img                                        # [14]
    dt_a     = dt @ axis_dir                                     # [14]
    dt_prime = (dt - dt_b.unsqueeze(1) * b_img
                   - dt_a.unsqueeze(1) * axis_dir)               # [14, 768]

    rows = []
    for di, dis in enumerate(DISEASES):
        gap     = (z @ dt[di]).numpy()
        r_center_term  = (r_hat @ dt[di]).item()                        # scalar (constant)
        vtx     = (r_b * dt_b[di]).numpy()                       # 꼭지점항 [N]
        ax      = (r_a * dt_a[di]).numpy()                       # 축항 [N]
        res     = (r_prime @ dt_prime[di]).numpy()                # 잔차항 [N]

        recon   = r_center_term + vtx + ax + res
        err     = np.abs(gap - recon).mean()

        # fractions (by mean absolute contribution)
        abs_m   = abs(r_center_term)
        abs_vtx = np.abs(vtx).mean()
        abs_ax  = np.abs(ax).mean()
        abs_res = np.abs(res).mean()
        total   = abs_m + abs_vtx + abs_ax + abs_res + 1e-10

        nondiscrim_frac = (abs_m + abs_vtx + abs_ax) / total

        rows.append(dict(
            disease=dis,
            gap_mean=gap.mean(),
            r_center_term=r_center_term,
            vertex_term_mean=vtx.mean(),
            axis_term_mean=ax.mean(),
            residual_term_mean=res.mean(),
            base_center_frac=abs_m / total,
            vertex_frac=abs_vtx / total,
            axis_frac=abs_ax / total,
            residual_frac=abs_res / total,
            nondiscrim_frac=nondiscrim_frac,
            dt_b_img=dt_b[di].item(),
            dt_axis=dt_a[di].item(),
            norm_dt_prime=dt_prime[di].norm().item(),
            recon_error=err,
            PNC_AUC=pnc_auc[dis]))

    df = pd.DataFrame(rows)
    _save_csv(df, "exp_7_gap_decomposition")

    for _, r in df.iterrows():
        print(f"  {r['disease']:20s}  center={r['r_center_term']:+.5f}  "
              f"vtx={r['vertex_term_mean']:+.5f}  ax={r['axis_term_mean']:+.5f}  "
              f"res={r['residual_term_mean']:+.5f}  "
              f"nondiscrim={r['nondiscrim_frac']:.3f}")

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xi = np.arange(len(DISEASES)); w = .2
    axes[0].bar(xi - 1.5*w, [rows[i]["r_center_term"] for i in range(14)],
                w, label="r̂·Δt (constant)", color="salmon")
    axes[0].bar(xi - 0.5*w, df["vertex_term_mean"],
                w, label="꼭지점항",       color="orange")
    axes[0].bar(xi + 0.5*w, df["axis_term_mean"],
                w, label="축항",           color="gold")
    axes[0].bar(xi + 1.5*w, df["residual_term_mean"],
                w, label="잔차항 r'·Δt'",  color="steelblue")
    axes[0].set_xticks(xi)
    axes[0].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[0].set_ylabel("mean"); axes[0].legend(fontsize=7)
    axes[0].axhline(0, c="k", lw=.5)
    axes[0].set_title("gap = 중심점 상수항 + 꼭지점 + 축 + 잔차")

    # stacked fraction bar
    axes[1].bar(xi, df["base_center_frac"], label="중심점 상수항", color="salmon")
    axes[1].bar(xi, df["vertex_frac"],  bottom=df["base_center_frac"],
                label="꼭지점항",  color="orange")
    axes[1].bar(xi, df["axis_frac"],
                bottom=df["base_center_frac"] + df["vertex_frac"],
                label="축항",     color="gold")
    axes[1].bar(xi, df["residual_frac"],
                bottom=df["base_center_frac"] + df["vertex_frac"] + df["axis_frac"],
                label="잔차항",   color="steelblue")
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[1].set_ylabel("fraction"); axes[1].legend(fontsize=7)
    axes[1].set_title("Fraction of |gap| by term")

    rho, p = stats.spearmanr(df["residual_term_mean"], df["PNC_AUC"])
    axes[2].scatter(df["residual_term_mean"], df["PNC_AUC"],
                    s=60, edgecolors="k", c="steelblue")
    for _, r in df.iterrows():
        axes[2].annotate(r["disease"][:4],
                         (r["residual_term_mean"], r["PNC_AUC"]), fontsize=7)
    axes[2].set(xlabel="잔차항 mean (r'·Δt')", ylabel="PNC AUC",
                title=f"Residual term vs AUC  ρ={rho:.3f}  p={p:.4f}")

    fig.suptitle("Exp 7 — 4-Term Gap Decomposition", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_7_gap_decomposition")


# ══════════════════════════════════════════════════════════════════════════
# H3: AUC는 residual space 분리도가 결정
# ══════════════════════════════════════════════════════════════════════════

def exp_8():
    """Exp 8: Content-space separation predicts classification.
    3-way: r' = r에서 b̂_img, axis 모두 제거한 이미지 고유 잔차
    cos_r_prime_sep = E[cos(r', Δt')]_y=1 − E[cos(r', Δt')]_y=0"""
    print("\n" + "=" * 60)
    print("Exp 8  Content-Space Separation vs PNC AUC (2-way vs 3-way)")
    print("=" * 60)

    D     = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    b_img = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    z     = F.normalize(D["zs_img"].float(),      dim=-1)
    tp    = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn    = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    dt    = tp - tn;  lb = D["labels"]

    r_hat = z.mean(0)
    m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

    # --- 2-way: b̂_img만 제거 ---
    def _rm1(emb, b):
        return emb - (emb @ b).unsqueeze(-1) * b

    r_2way   = _rm1(z, b_img)
    s_2way   = _rm1(dt, b_img)
    r_2way_n = F.normalize(r_2way, dim=-1)
    s_2way_n = F.normalize(s_2way, dim=-1)

    # --- 3-way: b̂_img + axis 제거 → r', Δt' ---
    def _rm2(emb, b1, b2):
        return emb - (emb @ b1).unsqueeze(-1) * b1 - (emb @ b2).unsqueeze(-1) * b2

    r_prime   = _rm2(z, b_img, axis_dir)
    dt_prime  = _rm2(dt, b_img, axis_dir)
    r_prime_n = F.normalize(r_prime, dim=-1)
    dt_prime_n = F.normalize(dt_prime, dim=-1)

    rows = []
    for di, dis in enumerate(DISEASES):
        mp = lb[:, di] == 1; mn = lb[:, di] == 0

        # 2-way
        cr1  = (r_2way_n @ s_2way_n[di]).numpy()
        sep1 = cr1[mp].mean() - cr1[mn].mean() if mp.sum() else np.nan

        # 3-way (r' and Δt')
        cr2  = (r_prime_n @ dt_prime_n[di]).numpy()
        sep2 = cr2[mp].mean() - cr2[mn].mean() if mp.sum() else np.nan

        # original space
        dt_n = F.normalize(dt[di:di+1], dim=-1).squeeze()
        cr0  = (z @ dt_n).numpy()
        sep0 = cr0[mp].mean() - cr0[mn].mean() if mp.sum() else np.nan

        rows.append(dict(disease=dis,
                         cos_r_sep_2way=sep1,
                         cos_r_prime_sep_3way=sep2,
                         cos_orig_sep=sep0,
                         PNC_AUC=pnc_auc[dis]))

    df = pd.DataFrame(rows)
    rho1, p1 = stats.spearmanr(df["cos_r_sep_2way"],        df["PNC_AUC"])
    rho2, p2 = stats.spearmanr(df["cos_r_prime_sep_3way"],  df["PNC_AUC"])
    rho0, p0 = stats.spearmanr(df["cos_orig_sep"],          df["PNC_AUC"])
    print(f"  ρ (2-way, −b̂_img):          {rho1:+.4f}  p={p1:.6f}")
    print(f"  ρ (3-way, −b̂_img−axis):     {rho2:+.4f}  p={p2:.6f}")
    print(f"  ρ (original space):          {rho0:+.4f}  p={p0:.6f}")

    _save_csv(df, "exp_8_content_separation")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, rho, p, title in [
        (axes[0], "cos_r_sep_2way",        rho1, p1, "2-way (−b̂_img)"),
        (axes[1], "cos_r_prime_sep_3way",  rho2, p2, "3-way (−b̂_img−axis)"),
        (axes[2], "cos_orig_sep",          rho0, p0, "Original space"),
    ]:
        ax.scatter(df[col], df["PNC_AUC"], s=60, edgecolors="k", c="steelblue")
        for _, r in df.iterrows():
            ax.annotate(r["disease"][:4], (r[col], r["PNC_AUC"]), fontsize=7)
        ax.set(xlabel=col, ylabel="PNC AUC",
               title=f"{title}\nρ={rho:.3f}  p={p:.4f}")
    fig.suptitle("Exp 8 — Content-Space Separation vs PNC AUC", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_8_content_separation")


# ══════════════════════════════════════════════════════════════════════════
# H4: 꼭지점·축 제거는 AUC에 거의 영향 없다
# ══════════════════════════════════════════════════════════════════════════

def exp_9():
    """Exp 9: Bias removal → recompute AUC.
    Variants:
      A  remove b̂_img from z              (꼭지점 제거)
      B  remove b̂_text from t             (텍스트 bias 제거)
      C  remove b̂_img + b̂_text            (양쪽 bias 제거)
      D  remove b̂_img + axis from z       (꼭지점 + 축 동시 제거)
      E  original (baseline)"""
    print("\n" + "=" * 60)
    print("Exp 9  Bias Removal Intervention (꼭지점 + 축)")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score

    D     = load_embeddings()
    b_img = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    b_txt = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")
    z     = F.normalize(D["zs_img"].float(),      dim=-1)
    tp    = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn    = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    lb    = D["labels"]

    r_hat = z.mean(0)
    m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

    def _rm1(emb, b):
        return F.normalize(emb - (emb @ b).unsqueeze(-1) * b, dim=-1)

    def _rm2(emb, b1, b2):
        return F.normalize(emb - (emb @ b1).unsqueeze(-1) * b1
                               - (emb @ b2).unsqueeze(-1) * b2, dim=-1)

    def _aucs(z_, tp_, tn_):
        out = {}
        for di, dis in enumerate(DISEASES):
            g = (z_ @ tp_[di] - z_ @ tn_[di]).numpy()
            y = lb[:, di]
            out[dis] = roc_auc_score(y, g) if 0 < y.sum() < len(y) else np.nan
        return out

    aE = _aucs(z, tp, tn)                                          # original
    aA = _aucs(_rm1(z, b_img), tp, tn)                             # −b̂_img
    aB = _aucs(z, _rm1(tp, b_txt), _rm1(tn, b_txt))               # −b̂_text
    aC = _aucs(_rm1(z, b_img), _rm1(tp, b_txt), _rm1(tn, b_txt))  # −b̂_img−b̂_text
    aD = _aucs(_rm2(z, b_img, axis_dir), tp, tn)                   # −b̂_img−axis

    rows = []
    for dis in DISEASES:
        rows.append(dict(
            disease=dis,
            AUC_original=aE[dis],
            AUC_rm_b_img=aA[dis],          dAUC_b_img=aA[dis]-aE[dis],
            AUC_rm_b_text=aB[dis],         dAUC_b_text=aB[dis]-aE[dis],
            AUC_rm_both_bias=aC[dis],      dAUC_both_bias=aC[dis]-aE[dis],
            AUC_rm_vertex_axis=aD[dis],    dAUC_vertex_axis=aD[dis]-aE[dis]))

    df = pd.DataFrame(rows)
    _save_csv(df, "exp_9_bias_removal")

    print(f"  Mean AUC  original       : {df['AUC_original'].mean():.4f}")
    print(f"  Mean AUC  −b̂_img        : {df['AUC_rm_b_img'].mean():.4f}"
          f"  (Δ={df['dAUC_b_img'].mean():+.4f})")
    print(f"  Mean AUC  −b̂_text       : {df['AUC_rm_b_text'].mean():.4f}"
          f"  (Δ={df['dAUC_b_text'].mean():+.4f})")
    print(f"  Mean AUC  −b̂_img−b̂_text : {df['AUC_rm_both_bias'].mean():.4f}"
          f"  (Δ={df['dAUC_both_bias'].mean():+.4f})")
    print(f"  Mean AUC  −b̂_img−axis   : {df['AUC_rm_vertex_axis'].mean():.4f}"
          f"  (Δ={df['dAUC_vertex_axis'].mean():+.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xi = np.arange(len(DISEASES)); w = .17
    axes[0].bar(xi-2*w,   df["AUC_original"],       w, label="Original",        color="steelblue")
    axes[0].bar(xi-1*w,   df["AUC_rm_b_img"],       w, label="−b̂_img",         color="coral")
    axes[0].bar(xi,        df["AUC_rm_b_text"],      w, label="−b̂_text",        color="gold")
    axes[0].bar(xi+1*w,   df["AUC_rm_both_bias"],   w, label="−b̂_img−b̂_text",  color="mediumpurple")
    axes[0].bar(xi+2*w,   df["AUC_rm_vertex_axis"], w, label="−b̂_img−axis",    color="green")
    axes[0].set_xticks(xi)
    axes[0].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=6)
    axes[0].set_title("AUC per Variant")

    for col, mk, c, lb_ in [
        ("dAUC_b_img",       "o", "coral",        "−b̂_img"),
        ("dAUC_b_text",      "s", "gold",         "−b̂_text"),
        ("dAUC_both_bias",   "^", "mediumpurple", "−b̂_img−b̂_text"),
        ("dAUC_vertex_axis", "D", "green",        "−b̂_img−axis")]:
        axes[1].scatter(df[col], df["AUC_original"], s=50, marker=mk, c=c, label=lb_)
    axes[1].axvline(0, c="k", ls="--", lw=.5)
    axes[1].set(xlabel="ΔAUC", ylabel="AUC (original)",
                title="Bias-Removal Effect on AUC")
    axes[1].legend(fontsize=7)

    fig.suptitle("Exp 9 — Bias Removal Intervention (꼭지점 + 축)", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_9_bias_removal")


# ══════════════════════════════════════════════════════════════════════════
# H5: Δt는 cone 밖에 살고, "no"는 텍스트를 꼭지점으로 민다
# ══════════════════════════════════════════════════════════════════════════

def exp_10():
    """Exp 10: Δt geometry per disease — image cone + text cone projections.
    Δt = t⁺ − t⁻ 의 방향을 image cone과 text cone 양쪽 기저로 분해한다."""
    print("\n" + "=" * 60)
    print("Exp 10  Δt Geometry (Image + Text Cone Projections)")
    print("=" * 60)

    D     = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    b_img = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    b_txt = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")
    z  = F.normalize(D["zs_img"].float(), dim=-1)
    tp = F.normalize(D["zs_text_pos"].float(), dim=-1)
    tn = F.normalize(D["zs_text_neg"].float(), dim=-1)
    dt = tp - tn

    # Image cone axis
    r_hat_img = z.mean(0)
    m_hat_img, axis_dir_img = estimate_cone_axis(r_hat_img, b_img)

    # Text cone axis
    all_text    = torch.cat([tp, tn], dim=0)
    r_hat_text  = all_text.mean(0)
    m_hat_text, axis_dir_text = estimate_cone_axis(r_hat_text, b_txt)

    rows = []
    for di, dis in enumerate(DISEASES):
        dtn = F.normalize(dt[di:di+1], dim=-1).squeeze()

        # Image cone decomposition of Δt
        dt_prime_img = (dt[di] - (dt[di] @ b_img) * b_img
                                - (dt[di] @ axis_dir_img) * axis_dir_img)

        # Text cone decomposition of Δt
        dt_prime_txt = (dt[di] - (dt[di] @ b_txt) * b_txt
                                - (dt[di] @ axis_dir_text) * axis_dir_text)

        rows.append(dict(
            disease=dis,
            norm_dt=dt[di].norm().item(),
            # image cone projections
            cos_dt_b_img=(dtn @ b_img).item(),
            cos_dt_axis_img=(dtn @ axis_dir_img).item(),
            dt_b_img_proj=(dt[di] @ b_img).item(),
            dt_axis_img_proj=(dt[di] @ axis_dir_img).item(),
            norm_dt_prime_img=dt_prime_img.norm().item(),
            # text cone projections
            cos_dt_b_text=(dtn @ b_txt).item(),
            cos_dt_axis_text=(dtn @ axis_dir_text).item(),
            dt_b_text_proj=(dt[di] @ b_txt).item(),
            dt_axis_text_proj=(dt[di] @ axis_dir_text).item(),
            norm_dt_prime_text=dt_prime_txt.norm().item(),
            # cross
            cos_tpos_tneg=(tp[di] @ tn[di]).item(),
            cos_tpos_b_text=(tp[di] @ b_txt).item(),
            cos_tneg_b_text=(tn[di] @ b_txt).item(),
            frac_dt_prime_img=(dt_prime_img.norm() / (dt[di].norm() + 1e-10)).item(),
            frac_dt_prime_text=(dt_prime_txt.norm() / (dt[di].norm() + 1e-10)).item(),
            PNC_AUC=pnc_auc[dis]))

    df = pd.DataFrame(rows)
    _save_csv(df, "exp_10_delta_t_geometry")

    # ── print ──
    for _, r in df.iterrows():
        print(f"  {r['disease']:20s}  "
              f"cos(Δt,b̂_img)={r['cos_dt_b_img']:+.4f}  "
              f"cos(Δt,b̂_text)={r['cos_dt_b_text']:+.4f}  "
              f"‖Δt'_img‖/‖Δt‖={r['frac_dt_prime_img']:.4f}  "
              f"‖Δt'_text‖/‖Δt‖={r['frac_dt_prime_text']:.4f}")

    # ── plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: image cone
    rho1, p1 = stats.spearmanr(df["cos_dt_b_img"],    df["PNC_AUC"])
    rho2, p2 = stats.spearmanr(df["cos_dt_axis_img"], df["PNC_AUC"])
    rho3, p3 = stats.spearmanr(df["cos_tpos_tneg"],   df["PNC_AUC"])

    for ax, col, rho, p, xl, c in [
        (axes[0,0], "cos_dt_b_img",    rho1, p1, "cos(Δt, b̂_img)",   "coral"),
        (axes[0,1], "cos_dt_axis_img", rho2, p2, "cos(Δt, axis_img)", "green"),
        (axes[0,2], "cos_tpos_tneg",   rho3, p3, "cos(t⁺, t⁻)",      "gold")]:
        ax.scatter(df[col], df["PNC_AUC"], s=60, edgecolors="k", c=c)
        for _, r in df.iterrows():
            ax.annotate(r["disease"][:4], (r[col], r["PNC_AUC"]), fontsize=7)
        ax.set(xlabel=xl, ylabel="PNC AUC",
               title=f"{xl} vs AUC  ρ={rho:.3f}  p={p:.4f}")

    # Bottom row: text cone
    rho4, p4 = stats.spearmanr(df["cos_dt_b_text"],    df["PNC_AUC"])
    rho5, p5 = stats.spearmanr(df["cos_dt_axis_text"], df["PNC_AUC"])
    rho6, p6 = stats.spearmanr(df["cos_tneg_b_text"],  df["PNC_AUC"])

    for ax, col, rho, p, xl, c in [
        (axes[1,0], "cos_dt_b_text",    rho4, p4, "cos(Δt, b̂_text)",   "mediumpurple"),
        (axes[1,1], "cos_dt_axis_text", rho5, p5, "cos(Δt, axis_text)", "teal"),
        (axes[1,2], "cos_tneg_b_text",  rho6, p6, "cos(t⁻, b̂_text)",   "brown")]:
        ax.scatter(df[col], df["PNC_AUC"], s=60, edgecolors="k", c=c)
        for _, r in df.iterrows():
            ax.annotate(r["disease"][:4], (r[col], r["PNC_AUC"]), fontsize=7)
        ax.set(xlabel=xl, ylabel="PNC AUC",
               title=f"{xl} vs AUC  ρ={rho:.3f}  p={p:.4f}")

    fig.suptitle("Exp 10 — Δt Geometry (Image + Text Cone)", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_10_delta_t_geometry")


def exp_11():
    """Exp 11: "no" insertion → text movement analysis.
    t⁻ − t⁺ 벡터를 text cone 기저 (b̂_text, axis_text, residual)로 분해하여
    "no"가 텍스트 임베딩을 꼭지점(무의미 방향)으로 미는 효과를 정량화한다."""
    print("\n" + "=" * 60)
    print('Exp 11  "no" Insertion → Text Movement Analysis')
    print("=" * 60)

    D      = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    b_text = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")
    b_img  = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")

    tp = F.normalize(D["zs_text_pos"].float(), dim=-1)    # [14, 768]
    tn = F.normalize(D["zs_text_neg"].float(), dim=-1)    # [14, 768]
    z  = F.normalize(D["zs_img"].float(),      dim=-1)

    # Text cone basis
    all_text    = torch.cat([tp, tn], dim=0)
    r_hat_text  = all_text.mean(0)
    m_hat_text, axis_dir_text = estimate_cone_axis(r_hat_text, b_text)

    # Image cone basis
    r_hat_img = z.mean(0)
    m_hat_img, axis_dir_img = estimate_cone_axis(r_hat_img, b_img)

    # Movement vector: "no" insertion moves t⁺ → t⁻
    movement = tn - tp                                     # [14, 768]
    mv_norm  = movement.norm(dim=-1)                       # [14]

    # Decompose into text cone basis
    mv_b_text = (movement @ b_text)                        # [14]
    mv_a_text = (movement @ axis_dir_text)                 # [14]
    mv_prime_text = (movement
                     - mv_b_text.unsqueeze(1) * b_text
                     - mv_a_text.unsqueeze(1) * axis_dir_text)
    mv_prime_text_norm = mv_prime_text.norm(dim=-1)

    # Decompose into image cone basis
    mv_b_img = (movement @ b_img)                          # [14]
    mv_a_img = (movement @ axis_dir_img)                   # [14]

    rows = []
    for di, dis in enumerate(DISEASES):
        norm_i = mv_norm[di].item() + 1e-10
        rows.append(dict(
            disease=dis,
            movement_norm=mv_norm[di].item(),
            # text cone decomposition
            mv_b_text=mv_b_text[di].item(),
            mv_axis_text=mv_a_text[di].item(),
            mv_prime_text_norm=mv_prime_text_norm[di].item(),
            mv_b_text_frac=abs(mv_b_text[di].item()) / norm_i,
            mv_axis_text_frac=abs(mv_a_text[di].item()) / norm_i,
            mv_prime_text_frac=mv_prime_text_norm[di].item() / norm_i,
            # image cone decomposition
            mv_b_img=mv_b_img[di].item(),
            mv_axis_img=mv_a_img[di].item(),
            # direct cos with b̂_text
            cos_tpos_b_text=(tp[di] @ b_text).item(),
            cos_tneg_b_text=(tn[di] @ b_text).item(),
            delta_cos_b_text=(tn[di] @ b_text - tp[di] @ b_text).item(),
            PNC_AUC=pnc_auc[dis]))

    df = pd.DataFrame(rows)
    _save_csv(df, "exp_11_no_insertion")

    # ── print ──
    print(f"\n  Movement decomposition (text cone basis):")
    print(f"  {'disease':20s}  {'mv·b̂_text':>10s}  {'mv·axis':>10s}  "
          f"{'‖mv_prime‖':>10s}  {'Δcos(b̂)':>10s}")
    for _, r in df.iterrows():
        print(f"  {r['disease']:20s}  {r['mv_b_text']:+10.5f}  "
              f"{r['mv_axis_text']:+10.5f}  "
              f"{r['mv_prime_text_norm']:10.5f}  "
              f"{r['delta_cos_b_text']:+10.5f}")

    mean_frac_b    = df["mv_b_text_frac"].mean()
    mean_frac_axis = df["mv_axis_text_frac"].mean()
    mean_frac_res  = df["mv_prime_text_frac"].mean()
    print(f"\n  Mean fraction:  b̂_text={mean_frac_b:.4f}  "
          f"axis={mean_frac_axis:.4f}  residual={mean_frac_res:.4f}")
    print(f"  All diseases: mv·b̂_text > 0?  "
          f"{(df['mv_b_text'] > 0).all()}  "
          f"(mean={df['mv_b_text'].mean():.5f})")

    # ── plot ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    xi = np.arange(len(DISEASES))

    # (a) stacked decomposition bar
    axes[0].bar(xi, df["mv_b_text"].abs(), label="|mv·b̂_text|", color="coral")
    axes[0].bar(xi, df["mv_axis_text"].abs(),
                bottom=df["mv_b_text"].abs(), label="|mv·axis|", color="gold")
    axes[0].bar(xi, df["mv_prime_text_norm"],
                bottom=df["mv_b_text"].abs() + df["mv_axis_text"].abs(),
                label="‖mv'‖", color="steelblue")
    axes[0].set_xticks(xi)
    axes[0].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[0].set_ylabel("|component|"); axes[0].legend(fontsize=7)
    axes[0].set_title('"no" movement decomposition\n(text cone basis)')

    # (b) mv·b̂_text per disease (signed)
    colors_b = ["coral" if v > 0 else "steelblue" for v in df["mv_b_text"]]
    axes[1].bar(xi, df["mv_b_text"], color=colors_b, edgecolor="k", alpha=.7)
    axes[1].axhline(0, c="k", lw=.5)
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([d[:4] for d in DISEASES], rotation=60, fontsize=7)
    axes[1].set_ylabel("mv · b̂_text")
    axes[1].set_title('"no" pushes toward vertex?\n(positive = toward b̂_text)')

    # (c) delta_cos(b̂_text) vs AUC
    rho_dc, p_dc = stats.spearmanr(df["delta_cos_b_text"], df["PNC_AUC"])
    axes[2].scatter(df["delta_cos_b_text"], df["PNC_AUC"],
                    s=60, edgecolors="k", c="mediumpurple")
    for _, r in df.iterrows():
        axes[2].annotate(r["disease"][:4],
                         (r["delta_cos_b_text"], r["PNC_AUC"]), fontsize=7)
    axes[2].set(xlabel="Δcos(b̂_text) = cos(t⁻,b̂) − cos(t⁺,b̂)",
                ylabel="PNC AUC",
                title=f'"no" vertex shift vs AUC\nρ={rho_dc:.3f}  p={p_dc:.4f}')

    # (d) cos(t⁺, b̂) vs cos(t⁻, b̂) paired
    axes[3].scatter(df["cos_tpos_b_text"], df["cos_tneg_b_text"],
                    s=60, edgecolors="k", c="steelblue")
    for _, r in df.iterrows():
        axes[3].annotate(r["disease"][:4],
                         (r["cos_tpos_b_text"], r["cos_tneg_b_text"]), fontsize=7)
    lim = [min(df["cos_tpos_b_text"].min(), df["cos_tneg_b_text"].min()) - 0.02,
           max(df["cos_tpos_b_text"].max(), df["cos_tneg_b_text"].max()) + 0.02]
    axes[3].plot(lim, lim, "k--", lw=.5, alpha=.5)
    axes[3].set(xlabel="cos(t⁺, b̂_text)", ylabel="cos(t⁻, b̂_text)",
                title="t⁻ consistently closer to vertex")

    fig.suptitle('Exp 11 — "no" Insertion Effect on Text Embeddings', fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_11_no_insertion")


# ══════════════════════════════════════════════════════════════════════════
# H6: Bi-MCQ 학습은 cone을 풀고 content 정렬을 높인다
# ══════════════════════════════════════════════════════════════════════════

def exp_12():
    """Exp 12: Bi-MCQ vs Zero-Shot — 꼭지점/축/잔차 3-way comparison."""
    print("\n" + "=" * 60)
    print("Exp 12  Bi-MCQ vs Zero-Shot: 꼭지점/축/잔차 Analysis")
    print("=" * 60)

    D     = load_embeddings()
    b_img = torch.load(EMB_DIR / "b_hat_img.pt", map_location="cpu")

    def _stats(emb, label):
        n     = F.normalize(emb.float(), dim=-1)
        r_hat = n.mean(0)
        R     = r_hat.norm().item()

        m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

        r    = n - r_hat
        r_b  = (r @ b_img).numpy()
        r_a  = (r @ axis_dir).numpy()
        rp   = (r - torch.from_numpy(r_b).unsqueeze(1).float() * b_img
                  - torch.from_numpy(r_a).unsqueeze(1).float() * axis_dir)
        rpn  = rp.norm(dim=-1).numpy()

        return dict(model=label, R=R,
                    cos_b_img_m_hat=(b_img @ m_hat).item(),
                    r_b_mean=float(r_b.mean()), r_b_std=float(r_b.std()),
                    r_axis_mean=float(r_a.mean()), r_axis_std=float(r_a.std()),
                    r_prime_norm_mean=float(rpn.mean()),
                    r_prime_norm_std=float(rpn.std()))

    zs_row = _stats(D["zs_img"],    "zero_shot")
    bm_row = _stats(D["bimcq_img"], "bi_mcq")

    # text Δt projections
    for tag, tp_key, tn_key, row in [
        ("zero_shot",  "zs_text_pos",    "zs_text_neg",    zs_row),
        ("bi_mcq",     "bimcq_text_pos", "bimcq_text_neg", bm_row)]:
        tp = F.normalize(D[tp_key].float(), dim=-1)
        tn = F.normalize(D[tn_key].float(), dim=-1)
        dt = tp - tn
        b  = (dt @ b_img).numpy()
        row["dt_b_mean"] = float(b.mean()); row["dt_b_std"] = float(b.std())

    df = pd.DataFrame([zs_row, bm_row])
    _save_csv(df, "exp_12_bimcq_comparison")
    print(df.to_string(index=False))

    # ── plot: 3-way decomposition histograms ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for emb_key, label, color in [
            ("zs_img", "ZS", "steelblue"), ("bimcq_img", "BiMCQ", "coral")]:
        n     = F.normalize(D[emb_key].float(), dim=-1)
        r_hat = n.mean(0)
        m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

        r   = n - r_hat
        r_b = (r @ b_img).numpy()
        r_a = (r @ axis_dir).numpy()
        rp  = (r - torch.from_numpy(r_b).unsqueeze(1).float() * b_img
                 - torch.from_numpy(r_a).unsqueeze(1).float() * axis_dir)
        rpn = rp.norm(dim=-1).numpy()

        axes[0].hist(r_b,  bins=50, alpha=.5, color=color,
                     label=f"{label} r·b̂  μ={r_b.mean():.3f}")
        axes[1].hist(r_a,  bins=50, alpha=.5, color=color,
                     label=f"{label} r·axis μ={r_a.mean():.3f}")
        axes[2].hist(rpn,  bins=50, alpha=.5, color=color,
                     label=f"{label} ‖r'‖ μ={rpn.mean():.3f}")

    axes[0].set(xlabel="r · b̂_img",  ylabel="count", title="꼭지점 projection")
    axes[1].set(xlabel="r · axis",    ylabel="count", title="축 projection")
    axes[2].set(xlabel="‖r'‖",        ylabel="count", title="이미지 고유 잔차 norm")
    for ax in axes:
        ax.legend(fontsize=7)

    fig.suptitle("Exp 12 — Bi-MCQ vs ZS: 3-Way Decomposition", fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_12_bimcq_comparison")


# ══════════════════════════════════════════════════════════════════════════
# 종합
# ══════════════════════════════════════════════════════════════════════════

def exp_13():
    """Exp 13: Summary correlation table — what predicts PNC AUC?
    Includes 2-way (b̂_img only) and 3-way (b̂_img + axis) metrics,
    plus text cone metrics."""
    print("\n" + "=" * 60)
    print("Exp 13  Summary Correlation Table (2-way + 3-way + text cone)")
    print("=" * 60)

    D     = load_embeddings()
    pnc_auc = compute_pnc_auc(D)
    b_img = torch.load(EMB_DIR / "b_hat_img.pt",  map_location="cpu")
    b_txt = torch.load(EMB_DIR / "b_hat_text.pt", map_location="cpu")
    z   = F.normalize(D["zs_img"].float(),      dim=-1)
    tp  = F.normalize(D["zs_text_pos"].float(),  dim=-1)
    tn  = F.normalize(D["zs_text_neg"].float(),  dim=-1)
    dt  = tp - tn;  lb = D["labels"]

    # z = r_hat + delta
    r_hat = z.mean(0)
    r_all = z - r_hat

    # Image axis
    m_hat, axis_dir = estimate_cone_axis(r_hat, b_img)

    # Text axis
    all_text    = torch.cat([tp, tn], dim=0)
    r_hat_text  = all_text.mean(0)
    m_hat_text, axis_dir_text = estimate_cone_axis(r_hat_text, b_txt)

    # ── 2-way decomposition (b̂_img only) ──
    alpha = z  @ b_img
    r_img = z  - alpha.unsqueeze(1) * b_img
    r_n   = F.normalize(r_img, dim=-1)
    beta  = dt @ b_img
    s_txt = dt - beta.unsqueeze(1) * b_img
    s_n   = F.normalize(s_txt, dim=-1)

    # ── 3-way decomposition (b̂_img + axis) ──
    r_b  = r_all @ b_img
    r_a  = r_all @ axis_dir
    r_prime = (r_all - r_b.unsqueeze(1) * b_img
                     - r_a.unsqueeze(1) * axis_dir)
    r_prime_n = F.normalize(r_prime, dim=-1)

    dt_b  = dt @ b_img
    dt_a  = dt @ axis_dir
    dt_prime = (dt - dt_b.unsqueeze(1) * b_img
                   - dt_a.unsqueeze(1) * axis_dir)
    dt_prime_n = F.normalize(dt_prime, dim=-1)

    # ── "no" movement metrics ──
    movement   = tn - tp
    mv_b_text  = (movement @ b_txt)

    met = {}
    for di, dis in enumerate(DISEASES):
        mp = lb[:, di] == 1; mn = lb[:, di] == 0
        gap = (z @ dt[di]).numpy()
        bt  = (alpha * beta[di]).numpy()
        ct  = (r_img @ s_txt[di]).numpy()
        # 2-way cos_r_separation
        cr  = (r_n @ s_n[di]).numpy()
        cos_r_sep_2way = (cr[mp].mean() - cr[mn].mean()) if mp.sum() else np.nan
        # 3-way cos_r_prime_separation
        cr3 = (r_prime_n @ dt_prime_n[di]).numpy()
        cos_r_prime_sep_3way = (cr3[mp].mean() - cr3[mn].mean()) if mp.sum() else np.nan

        dtn = F.normalize(dt[di:di+1], dim=-1).squeeze()

        # 4-term gap decomposition fractions
        r_center_term = (r_hat @ dt[di]).item()
        vtx = (r_b * dt_b[di]).numpy()
        ax  = (r_a * dt_a[di]).numpy()
        res = (r_prime @ dt_prime[di]).numpy()

        met[dis] = dict(
            gap_mean=gap.mean(),
            gap_sep=gap[mp].mean() - gap[mn].mean() if mp.sum() else np.nan,
            bias_term_mean=bt.mean(),
            content_term_mean=ct.mean(),
            # 2-way metrics
            cos_r_sep_2way=cos_r_sep_2way,
            beta_d=beta[di].item(),
            norm_s_d=s_txt[di].norm().item(),
            # 3-way metrics
            cos_r_prime_sep_3way=cos_r_prime_sep_3way,
            r_center_term=r_center_term,
            vertex_term_mean=vtx.mean(),
            axis_term_mean=ax.mean(),
            residual_term_mean=res.mean(),
            dt_b_img=dt_b[di].item(),
            dt_axis=dt_a[di].item(),
            norm_dt_prime=dt_prime[di].norm().item(),
            # common
            norm_dt=dt[di].norm().item(),
            cos_dt_b_img=(dtn @ b_img).item(),
            cos_dt_axis=(dtn @ axis_dir).item(),
            cos_tpos_tneg=(tp[di] @ tn[di]).item(),
            # text cone metrics
            cos_tpos_b_text=(tp[di] @ b_txt).item(),
            cos_tneg_b_text=(tn[di] @ b_txt).item(),
            delta_cos_b_text=(tn[di] @ b_txt - tp[di] @ b_txt).item(),
            mv_b_text=mv_b_text[di].item(),
            cos_dt_b_text=(dtn @ b_txt).item(),
            cos_dt_axis_text=(dtn @ axis_dir_text).item(),
            PNC_AUC=pnc_auc[dis])

    df_met  = pd.DataFrame.from_dict(met, orient="index")
    df_met.index.name = "disease"

    target = df_met["PNC_AUC"]
    corrs  = []
    for col in df_met.columns:
        if col == "PNC_AUC":
            continue
        rho, p = stats.spearmanr(df_met[col], target)
        corrs.append(dict(metric=col, spearman_rho=rho, p_value=p))
    df_corr = pd.DataFrame(corrs).sort_values("spearman_rho", ascending=False, key=abs)

    _save_csv(df_corr,               "exp_13_summary_correlations")
    _save_csv(df_met.reset_index(),  "exp_13_all_metrics")

    print("\n  Spearman ρ with PNC AUC (|ρ| desc.):")
    for _, r in df_corr.iterrows():
        sig = "***" if r["p_value"] < .001 else "**" if r["p_value"] < .01 \
              else "*" if r["p_value"] < .05 else ""
        print(f"    {r['metric']:30s}  ρ={r['spearman_rho']:+.4f}  "
              f"p={r['p_value']:.6f} {sig}")

    # ── plot: 3 panels ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: correlation bar chart
    dc = df_corr.sort_values("spearman_rho", ascending=True, key=abs)
    colors = ["red" if p < .05 else "gray" for p in dc["p_value"]]
    axes[0].barh(dc["metric"], dc["spearman_rho"], color=colors, edgecolor="k", alpha=.7)
    axes[0].axvline(0, c="k", lw=.5)
    axes[0].set_xlabel("Spearman ρ with PNC AUC")
    axes[0].set_title("Metric Correlations  (red = p < 0.05)")

    # Panel 2: best predictor scatter
    best = df_corr.iloc[0]
    bm   = best["metric"]
    axes[1].scatter(df_met[bm], df_met["PNC_AUC"], s=60, edgecolors="k", c="steelblue")
    for dis in DISEASES:
        axes[1].annotate(dis[:4],
                         (df_met.loc[dis, bm], df_met.loc[dis, "PNC_AUC"]), fontsize=7)
    axes[1].set(xlabel=bm, ylabel="PNC AUC",
                title=f"Best predictor: {bm}\n"
                      f"ρ={best['spearman_rho']:.3f}  p={best['p_value']:.6f}")

    # Panel 3: 2-way vs 3-way separation scatter
    axes[2].scatter(df_met["cos_r_sep_2way"], df_met["cos_r_prime_sep_3way"],
                    s=60, edgecolors="k", c="green")
    for dis in DISEASES:
        axes[2].annotate(dis[:4],
                         (df_met.loc[dis, "cos_r_sep_2way"],
                          df_met.loc[dis, "cos_r_prime_sep_3way"]), fontsize=7)
    rho_cmp, p_cmp = stats.spearmanr(df_met["cos_r_sep_2way"],
                                      df_met["cos_r_prime_sep_3way"])
    axes[2].set(xlabel="cos_r_sep (2-way: −b̂_img only)",
                ylabel="cos_r_prime_sep (3-way: −b̂_img−axis)",
                title=f"2-way vs 3-way separation\nρ={rho_cmp:.3f}  p={p_cmp:.4f}")
    mn_v = min(df_met["cos_r_sep_2way"].min(), df_met["cos_r_prime_sep_3way"].min())
    mx_v = max(df_met["cos_r_sep_2way"].max(), df_met["cos_r_prime_sep_3way"].max())
    axes[2].plot([mn_v, mx_v], [mn_v, mx_v], "k--", lw=.5, alpha=.5)

    fig.suptitle("Exp 13 — What Predicts Negation Classification? (2-way + 3-way + text cone)",
                 fontsize=13)
    fig.tight_layout(); _save_fig(fig, "exp_13_summary")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

RUN_EXPS = None  # 실행할 실험 리스트. None이면 전체 실행. 예: ["5", "11"]

EXP_MAP = {
    "1":  exp_1,   "2":  exp_2,   "3":  exp_3,
    "4":  exp_4,   "5":  exp_5,
    "6":  exp_6,   "7":  exp_7,
    "8":  exp_8,
    "9":  exp_9,
    "10": exp_10,  "11": exp_11,
    "12": exp_12,
    "13": exp_13,
}
ALL_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


def main():
    run_list = RUN_EXPS if RUN_EXPS else ALL_ORDER
    for eid in run_list:
        EXP_MAP[eid]()

    print("\n" + "=" * 60)
    print(f"Done.  Results → {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
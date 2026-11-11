#!/usr/bin/env python3
"""섹션 1 — 기하와 노이즈 분석: 실험 함수 6개.
exp01: 이미지 노이즈 수렴, 모멘트 대조, 표본 수/seed 안정성.
exp02: 문자열/단어열 노이즈 수렴과 꼭지점 안정성.
exp03: 두 축의 기하, 구면 시각화와 실제 이미지 몽타주.
exp04: 손상 강도 응답, 유형별 혼합 궤적, 교차 꼭지점과 유형 내 반복 대조.
exp05: 기하 조건을 맞춘 무작위 기준점 대조.
exp06: COCO/NIH 교차 도메인, 손상 탐지, MNIST/비의료 텍스트.
r=normalize(mean(normalize(z))), b=normalize(mean(normalize(noise))).
conformity 축 O→r, Imageness 축 b→r. h=((z-b)·(r-b))/||r-b||².
Config를 수정하고 python analysis.py로 실행한다. CLI 인자/JSON 결과 없음.
결과: <root>/results/section_experiments/analysis/ — 실험별 CSV와 PNG 하나.
내부 계산은 원 정밀도, CSV/그림 수치는 소수점 세 자리.
입력 annotation JSON을 읽고 임베딩 NPZ는 <root>/cache에 저장한다.
Dependencies: numpy pandas scipy Pillow matplotlib, 모델 추론: torch open_clip_torch/CARZero.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 설정: 여기만 수정한다.
# =============================================================================
@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    base: Path = ROOT
    carzero_project: Path = ROOT / 'third_party' / 'BI_MCQ'
    carzero_weight_root: Path = ROOT / 'ckpt'
    carzero_ckpt: str = "best_model.ckpt"
    nih_root: Path = ROOT / 'data' / 'NIH'
    carzero_embedding_dir: Path = ROOT / 'cache' / 'embeddings'
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    encoders: tuple[str, ...] = ("clip", "carzero")
    device: str | None = None
    batch_size: int = 256
    seeds: tuple[int, ...] = (0, 1, 2)
    n_reference: int = 2000
    n_noise: int = 1000
    n_text_noise_each: int = 500
    n_ood: int = 3000
    noise_std_sweep: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    constant_levels: tuple[float, ...] = tuple(round(0.01 * k, 2) for k in range(1, 101))
    n_perturb_images: int = 500
    bootstrap_repeats: int = 400
    null_directions: int = 1000
    conformity_caliper: float = 0.010
    montage_k: int = 6
    caption_chunk: int = 2048
    recompute: bool = False
    make_figures: bool = True
    gaussian_levels: tuple[float, ...] = (0.000, 0.050, 0.100, 0.200, 0.400)
    blur_levels: tuple[float, ...] = (0.000, 1.000, 2.000, 4.000, 8.000)
    phase_levels: tuple[float, ...] = (0.000, 0.250, 0.500, 0.750, 1.000)
    noise_sample_sizes: tuple[int, ...] = (50, 100, 250, 500, 1000)
    matched_noise_std: float = 0.200
    trajectory_levels: tuple[float, ...] = (0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000)
    n_trajectory_images: int = 200
    mnist_root: Path = ROOT / 'data' / 'mnist'
    n_mnist: int = 1000
    n_nonmedical_text: int = 1000
    bimcq_ckpt: str = 'BiMCQ_CARZero_best_model.ckpt'
    bimcq_config: Path = ROOT / 'third_party' / 'BI_MCQ' / 'configs' / 'chest14_finetuning_Bi-MCQ.yaml'
    figure_dpi: int = 180


CONFIG = Config()
EPS = np.sqrt(np.finfo(np.float64).eps)
CACHE_VERSION = "conformity-imageness-v2"   # 기존 CLIP noise / perturbation cache 재사용
LEGACY_UNIFORM_STD = 0.2                    # 이전 uniform cache key와의 호환용 값
REFERENCE_KEY = "uniform"
FAMILIES = ("gaussian", "blur", "patch_shuffle", "phase")
SCORES = ("c", "h", "n")
SCORE_LABEL = dict(c="Δ conformity c", h="Δ imageness h", n="Δ noise proximity n")
DOMAIN = dict(clip="coco", carzero="nih")
OTHER = dict(coco="nih", nih="coco")
ENC_LABEL = dict(clip="CLIP ViT-B/32", carzero="CARZero")
DATA_LABEL = dict(coco="COCO", nih="NIH")
RESULT_GROUP = 'section_experiments'


def result_directory(root, section):
    return Path(root) / 'results' / RESULT_GROUP / section

NIH_CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]
NIH_TERMS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Pulmonary Infiltration", "Pulmonary Mass",
             "Lung Nodule", "Pneumonia", "Pneumothorax", "Pulmonary Consolidation", "Pulmonary Edema",
             "Pulmonary Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]
# 1:1 긍정 / 부정 템플릿. {x}: 소문자 질환명, {X}: 문장 맨 앞이라 첫 글자만 대문자.
TEMPLATES = [
    ("There is {x}.", "There is no {x}."),
    ("{X} is present.", "No {x} is present."),
    ("An image with {x}.", "An image without {x}."),
    ("There is evidence of {x}.", "There is no evidence of {x}."),
    ("The image shows {x}.", "The image shows no {x}."),
    ("{X} is seen in the image.", "No {x} is seen in the image."),
    ("Signs of {x} are observed.", "No signs of {x} are observed."),
    ("The image shows evidence of {x}.", "The image shows no evidence of {x}."),
    ("{X} is detected in the image.", "{X} is not detected in the image."),
    ("The image is positive for {x}.", "The image is negative for {x}."),
]
WORD_POOL = ("apple neutron cactus flicker prism quartz wagon myth oxygen cobalt delta ember frost glyph helix "
             "ivory jungle karma lemon mosaic nebula orchid pulse ripple sigma thorn umbra vortex wax zenith "
             "blaze cipher dusk echo flux grove haze iris jade knot latch mirth nook opal plume reef sable tide "
             "umber vale whisk yarn zinc abyss bolt crux dew elm fang gust husk ion jolt keel lobe musk nib ore "
             "pith quill rind sap tuft urge vim wren yew zeal arc brim cleft").split()
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-+=/\\@#$%&*()[]{}~"

# 색: dataviz 기본 categorical 순서
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED = "#52514e", "#898781"
ENC_COLOR = dict(clip=C1, carzero=C2)
SCORE_COLOR = dict(c=C1, h=C2, n=C3)
FAMILY_COLOR = dict(gaussian=C1, blur=C2, patch_shuffle=C3, phase=C4)
plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "lines.linewidth": 2, "legend.frameon": False})


def sentence(template: str, term: str) -> str:
    x = term.lower()
    return template.replace("{X}", x[:1].upper() + x[1:]).replace("{x}", x)


def nih_prompts(polarity: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(문장, 질환 index, 템플릿 index). polarity: 'pos' 또는 'neg'."""
    k = 0 if polarity == "pos" else 1
    texts, d_idx, l_idx = [], [], []
    for d, term in enumerate(NIH_TERMS):
        for l, pair in enumerate(TEMPLATES):
            texts.append(sentence(pair[k], term)); d_idx.append(d); l_idx.append(l)
    return texts, np.asarray(d_idx), np.asarray(l_idx)


# =============================================================================
# 공통 유틸리티
# =============================================================================
def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Embedding contains NaN or infinity.")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("Cannot normalize a zero / near-zero vector.")
    return x / norms


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(normalize(a) * normalize(b), axis=-1), -1, 1)))








def save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)




def legend_below(ax, ncol: int = 2, offset: float = -0.17, **kw):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol, fontsize=8, **kw)


def band(ax, x, mean, low, high, color, label, **kw):
    ax.fill_between(x, low, high, color=color, alpha=0.2, lw=0)
    ax.plot(x, mean, color=color, label=label, **kw)


def fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def legacy_fingerprint(array: np.ndarray) -> str:
    """2_cone_analysis.py의 fingerprint (기존 perturbation cache key 호환)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode()); h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def cached_array(cache: Path, recompute: bool, stem: str, metadata: dict,
                 compute: Callable[[], np.ndarray]) -> np.ndarray:
    manifest = json.dumps(metadata, sort_keys=True, default=str)
    token = hashlib.sha256(manifest.encode()).hexdigest()[:20]
    path = cache / f"{stem}_{token}.npz"
    if path.exists() and not recompute:
        with np.load(path, allow_pickle=False) as z:
            if str(z["manifest"].item()) != manifest:
                raise ValueError(f"Cache metadata mismatch: {path}")
            raw = z["raw"].copy()
    else:
        raw = np.asarray(compute(), dtype=np.float32)
        save_npz(path, raw=raw, manifest=np.array(manifest))
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid encoder output / cache: {path}")
    normalize(raw)
    return raw


def bootstrap_means(X: np.ndarray, repeats: int, rng: np.random.Generator) -> np.ndarray:
    result = []
    for first in range(0, repeats, 32):
        k = min(32, repeats - first)
        weights = rng.multinomial(len(X), np.full(len(X), 1 / len(X)), size=k)
        result.append((weights @ X) / len(X))
    return np.concatenate(result)


def bootstrap_mean_ci(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    means = bootstrap_means(v.reshape(-1, 1), repeats, rng)[:, 0]
    return tuple(float(q) for q in np.quantile(means, [0.025, 0.975]))


def pooled(points: list, replicates: list) -> dict:
    """seed별 점추정의 평균 + 모든 seed bootstrap 반복을 합친 95% 분위수 구간."""
    reps = np.concatenate([np.atleast_1d(np.asarray(r, dtype=float)) for r in replicates])
    reps = reps[np.isfinite(reps)]
    return dict(mean=float(np.nanmean(points)),
                ci_low=float(np.quantile(reps, 0.025)) if len(reps) else np.nan,
                ci_high=float(np.quantile(reps, 0.975)) if len(reps) else np.nan)


def pooled_matrix(points: list, replicates: list) -> dict[str, np.ndarray]:
    reps = np.concatenate(replicates, axis=0)
    return dict(mean=np.nanmean(points, axis=0),
                ci_low=np.nanquantile(reps, 0.025, axis=0), ci_high=np.nanquantile(reps, 0.975, axis=0))


def partial_rank_corr(h, severity, c, image_ids) -> float:
    """Within-image partial Spearman association controlling conformity (인과 효과 추정치 아님)."""
    values = np.column_stack([stats.rankdata(h), stats.rankdata(severity), stats.rankdata(c)])
    _, inverse = np.unique(image_ids, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.column_stack([np.bincount(inverse, weights=values[:, j]) / counts for j in range(3)])
    values -= means[inverse]
    H, Y, C = values.T
    cc = C @ C
    if cc > EPS:
        H = H - C * ((C @ H) / cc); Y = Y - C * ((C @ Y) / cc)
    denom = np.linalg.norm(H) * np.linalg.norm(Y)
    return float(H @ Y / denom) if denom > EPS else np.nan


def within_bin_sd(labels: np.ndarray, H: np.ndarray):
    _, group = np.unique(labels, return_inverse=True)
    H2 = np.asarray(H, dtype=float).reshape(len(H), -1)
    onehot = np.zeros((group.max() + 1, len(group))); onehot[group, np.arange(len(group))] = 1
    means = (onehot @ H2) / onehot.sum(1, keepdims=True)
    result = np.sqrt(np.mean((H2 - means[group]) ** 2, axis=0))
    return result if np.ndim(H) > 1 else float(result[0])


def per_image_spearman(X: np.ndarray, levels: np.ndarray) -> np.ndarray:
    XR = stats.rankdata(X, axis=1); XR = XR - XR.mean(1, keepdims=True)
    LR = stats.rankdata(levels); LR = LR - LR.mean()
    denom = np.linalg.norm(XR, axis=1) * np.linalg.norm(LR)
    return np.where(denom > EPS, (XR @ LR) / np.where(denom > EPS, denom, 1), np.nan)


def auroc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """양성 점수가 클수록 1. 동점은 1/2."""
    s = np.concatenate([score_pos, score_neg])
    ranks = stats.rankdata(s)
    n1, n0 = len(score_pos), len(score_neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


# =============================================================================
# 기하: 축과 좌표
# =============================================================================
@dataclass
class Axes:
    """r_I: 편차 기준 평균(정규화 전). r_hat: 축 끝점. b: 단위 노이즈점."""
    r_I: np.ndarray
    b: np.ndarray
    R_I: float = field(init=False)
    r_hat: np.ndarray = field(init=False)
    gamma: float = field(init=False)
    m_hat: np.ndarray = field(init=False)
    e: np.ndarray = field(init=False)
    length: float = field(init=False)

    def __post_init__(self):
        self.r_I, self.b = np.asarray(self.r_I, dtype=np.float64), normalize(self.b)
        self.R_I = float(np.linalg.norm(self.r_I))
        if self.R_I <= EPS:
            raise ValueError("Mean vector is (near) zero.")
        self.r_hat = self.r_I / self.R_I
        self.gamma = float(self.b @ self.r_hat)
        self.length = float(np.linalg.norm(self.r_hat - self.b))
        if self.length <= EPS or abs(self.gamma) >= 1 - 1e-9:
            raise ValueError("Imageness axis undefined: r_hat and b are (anti)parallel.")
        self.m_hat = (self.r_hat - self.b) / self.length
        self.e = normalize(self.r_hat - self.gamma * self.b)

    def coords(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        Z = normalize(Z)
        c, n = Z @ self.r_hat, Z @ self.b
        D = Z - self.r_I
        db, de = D @ self.b, D @ self.e
        resid = np.sqrt(np.clip(np.sum(D * D, 1) - db ** 2 - de ** 2, 0, None))
        return dict(c=c, n=n, h=(c - n + 1 - self.gamma) / (2 * (1 - self.gamma)),
                    delta_b=db, delta_e=de, resid=resid)

    def project_out(self, X: np.ndarray, axes: str) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = X - np.outer(X @ self.b, self.b)
        if axes == "be":
            Y = Y - np.outer(X @ self.e, self.e)
        return Y


def geometry_stats(Zref: np.ndarray, Unoise: np.ndarray, Zeval: np.ndarray) -> dict:
    """tab3 한 행의 양. bootstrap에서도 같은 함수를 쓴다."""
    A = Axes(Zref.mean(0), Unoise.mean(0))
    S = A.coords(Zeval)
    q01 = np.quantile(S["c"], 0.01)
    ang = np.degrees(np.arccos(np.clip(S["n"], -1, 1)))
    return dict(R_b=float(np.linalg.norm(Unoise.mean(0))), R_I=A.R_I, gamma=A.gamma,
                angle_b_r_deg=float(np.degrees(np.arccos(np.clip(A.gamma, -1, 1)))),
                phi_deg=float(angle(A.r_hat, A.m_hat)),
                gamma_percentile=float(100 * np.mean(S["c"] < A.gamma)),
                frac_noise_c_above_q01=float(np.mean(Unoise @ A.r_hat > q01)),
                nearest_data_angle_deg=float(ang.min()), data_angle_q01_deg=float(np.quantile(ang, 0.01)),
                delta_b_sd=float(S["delta_b"].std()), delta_e_sd=float(S["delta_e"].std()),
                resid_norm_mean=float(S["resid"].mean()))


# =============================================================================
# 인코더
# =============================================================================
class ClipEncoder:
    name = "clip"

    def __init__(self, cfg: Config):
        # 환경변수(HF_HOME, HF_HUB_OFFLINE)는 건드리지 않는다: 프로세스 전체에 적용되어 CARZero / timm의
        # HF 캐시 위치까지 바꾸기 때문. open_clip은 cache_dir을 직접 받으므로 download.py가 받은
        # root/weights 캐시를 그대로 쓰고, 인터넷이 없으면 huggingface_hub가 이 캐시로 대체한다.
        import torch
        import open_clip
        self.torch, self.cfg = torch, cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained, force_quick_gelu=True, cache_dir=str(cfg.root / 'ckpt'))
        self.model = self.model.to(self.device).eval(); self.model.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model)
        norm = next((t for t in reversed(self.preprocess.transforms) if hasattr(t, "mean") and hasattr(t, "std")), None)
        if norm is None:
            raise ValueError("Cannot find the CLIP Normalize transform.")
        self.mean = np.asarray(norm.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(norm.std, dtype=np.float32).reshape(1, 1, 3)
        size = self.model.visual.image_size
        self.hw = (size, size) if isinstance(size, int) else tuple(size)
        patch = getattr(self.model.visual, "patch_size", None)
        self.patch = int(patch[0] if isinstance(patch, (tuple, list)) else patch) if patch else find_patch_size(self.model.visual)
        self.signature = dict(model=cfg.clip_model, pretrained=cfg.clip_pretrained,
                              torch=importlib.metadata.version("torch"),
                              open_clip=importlib.metadata.version("open_clip_torch"),
                              size=list(self.hw), mean=self.mean.ravel().tolist(), std=self.std.ravel().tolist())

    def pixels(self, path) -> np.ndarray:
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return np.clip(tensor.numpy().transpose(1, 2, 0) * self.std + self.mean, 0, 1).astype(np.float32)

    def encode_pixels(self, pixels: Iterable[np.ndarray]) -> np.ndarray:
        parts, batch = [], []
        def flush():
            if batch:
                x = (np.stack(batch) - self.mean) / self.std
                x = self.torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)
                with self.torch.inference_mode():
                    parts.append(self.model.encode_image(x, normalize=False).float().cpu().numpy())
                batch.clear()
        for image in pixels:
            if image.shape != (*self.hw, 3):
                raise ValueError(f"Unexpected pixel shape: {image.shape}")
            batch.append(np.asarray(image, dtype=np.float32))
            if len(batch) == self.cfg.batch_size:
                flush()
        flush()
        return np.concatenate(parts)

    def encode_paths(self, paths: list) -> np.ndarray:
        return self.encode_pixels(self.pixels(p) for p in paths)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), self.cfg.batch_size * 4):
            t = self.tokenizer(texts[i:i + self.cfg.batch_size * 4]).to(self.device)
            with self.torch.inference_mode():
                parts.append(self.model.encode_text(t, normalize=False).float().cpu().numpy())
        return np.concatenate(parts)

    def text_truncation(self, texts: list[str]) -> float:
        """시작 / 끝 토큰을 포함한 길이가 77을 넘는 비율."""
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            return float("nan")
        return float(np.mean([len(encode(t)) + 2 > 77 for t in texts]))


def find_patch_size(module) -> int:
    """kernel == stride인 첫 Conv2d (ViT patch embedding)에서 패치 크기를 읽는다."""
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == m.stride and m.kernel_size[0] > 1:
            return int(m.kernel_size[0])
    raise ValueError("Cannot infer the patch size of the image encoder.")


def load_carzero_model(cfg: Config, device: str):
    """1_extract_embeddings.py의 load_zeroshot_model과 같은 절차."""
    import torch
    repo = next((p for p in [cfg.carzero_project, cfg.carzero_project / "Bi-MCQ"]
                 if (p / "CARZero/CARZero.py").is_file()), None)
    if repo is None:
        raise FileNotFoundError("CARZero source not found under carzero_project.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from CARZero import builder
    from CARZero.models import transformer_backbones
    original_load_weight = transformer_backbones.load_weight
    def load_pretrain(model, path):
        resolved = cfg.carzero_weight_root / Path(path).name
        weights = torch.load(str(resolved if resolved.is_file() else path), map_location="cpu", weights_only=False)
        model.load_state_dict(transformer_backbones.update_weights(model, weights["state_dict"]))
        return model
    prev = Path.cwd()
    try:
        os.chdir(cfg.carzero_weight_root)
        ckpt = torch.load(str(cfg.carzero_weight_root / cfg.carzero_ckpt), map_location="cpu", weights_only=False)
        transformer_backbones.load_weight = load_pretrain
        try:
            model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(ckpt["hyper_parameters"])
        finally:
            transformer_backbones.load_weight = original_load_weight
        expected = model.state_dict(); fixed = {}
        for k, v in ckpt["state_dict"].items():
            new_key = k
            for prefix in ["CARZero_model.", "CARZero."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]; break
            if new_key in expected:
                fixed[new_key] = v
        missing = set(expected) - set(fixed)
        if missing:
            # 1_extract_embeddings.py와 같은 처리. 어떤 모듈이 빠졌는지 접두어별로 기록한다.
            groups = pd.Series([".".join(k.split(".")[:2]) for k in missing]).value_counts()
            print(f"  [WARN] CARZero: {len(missing)} keys missing (strict=False): " +
                  ", ".join(f"{g} ({n})" for g, n in groups.items()))
        model.load_state_dict(fixed, strict=not missing)
    finally:
        os.chdir(prev)
    return model.to(device).eval()


class CarzeroEncoder:
    """CARZero 전역 임베딩. 모든 이미지 입력은 PNG 경로 → model.process_img를 거친다."""
    name = "carzero"
    loader: Callable = staticmethod(load_carzero_model)   # 검증 시 가짜 모델 주입 지점

    def __init__(self, cfg: Config):
        import torch
        self.torch, self.cfg = torch, cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = type(self).loader(cfg, self.device)
        self.tmp = Path(tempfile.mkdtemp(prefix="carzero_png_", dir=cfg.root / "cache"))
        probe = self.tmp / "probe.png"
        Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(probe)
        with torch.inference_mode():
            x = self.model.process_img([str(probe)], self.device)
        self.hw = tuple(int(v) for v in x.shape[-2:])
        self.channels = int(x.shape[1])
        self.patch = find_patch_size(self.model)
        ckpt = cfg.carzero_weight_root / cfg.carzero_ckpt
        info = ckpt.stat() if ckpt.exists() else None
        self.signature = dict(model="CARZero", ckpt=str(ckpt), size_bytes=info.st_size if info else None,
                              mtime_ns=info.st_mtime_ns if info else None, input_hw=list(self.hw),
                              patch=self.patch, torch=importlib.metadata.version("torch"))
        print(f"  CARZero input {self.hw}, channels {self.channels}, patch {self.patch}")

    def pixels(self, path) -> np.ndarray:
        """원본을 RGB로 읽어 입력 해상도로 resize한 [0,1] 픽셀 (손상 실험용)."""
        with Image.open(path) as image:
            img = image.convert("RGB").resize(self.hw[::-1], Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255

    def _encode_batch(self, paths: list) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.model.process_img([str(p) for p in paths], self.device)
            _, g = self.model.image_encoder_forward(x)
        return g.float().cpu().numpy()

    def encode_paths(self, paths: list) -> np.ndarray:
        return np.concatenate([self._encode_batch(paths[i:i + self.cfg.batch_size])
                               for i in range(0, len(paths), self.cfg.batch_size)])

    def encode_pixels(self, pixels: Iterable[np.ndarray]) -> np.ndarray:
        parts, batch = [], []
        def flush():
            if batch:
                paths = []
                for k, image in enumerate(batch):
                    p = self.tmp / f"x{k:04d}.png"
                    arr = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(arr if arr.ndim == 2 or arr.shape[2] == 3 else arr[..., 0]).save(p)
                    paths.append(p)
                parts.append(self._encode_batch(paths)); batch.clear()
        for image in pixels:
            batch.append(image)
            if len(batch) == self.cfg.batch_size:
                flush()
        flush()
        return np.concatenate(parts)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), self.cfg.batch_size * 4):
            with self.torch.inference_mode():
                p = self.model.process_text(texts[i:i + self.cfg.batch_size * 4], self.device)
                _, g, _ = self.model.text_encoder_forward(p["caption_ids"], p["attention_mask"], p["token_type_ids"])
            parts.append(g.float().cpu().numpy())
        return np.concatenate(parts)

    def text_truncation(self, texts: list[str]) -> float:
        """attention mask가 최대 길이를 모두 채운 비율 (잘림 가능성)."""
        with self.torch.inference_mode():
            p = self.model.process_text(texts, self.device)
        mask = p["attention_mask"].cpu().numpy()
        return float(np.mean(mask.sum(1) >= mask.shape[1]))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# =============================================================================
# 데이터
# =============================================================================
class Datasets:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        coco = cfg.root / "data" / "coco"
        with (coco / "annotations" / "captions_val2017.json").open(encoding="utf-8") as f:
            ann = json.load(f)
        self.coco_images = {int(x["id"]): x for x in ann["images"]}
        self.coco_dir = coco / "val2017"
        captions = sorted(ann["annotations"], key=lambda x: x["id"])
        self.coco_caption_by_ann = {int(x["id"]): x["caption"].strip() for x in captions}
        self.coco_first_caption = {}
        for x in captions:
            self.coco_first_caption.setdefault(int(x["image_id"]), x["caption"].strip())
        self._nih = None

    # ---------------- COCO ----------------
    def coco_path(self, image_id: int) -> Path:
        return self.coco_dir / self.coco_images[int(image_id)]["file_name"]

    # ---------------- NIH (1_extract_embeddings.load_data와 같은 규칙) ----------------
    @property
    def nih(self) -> dict:
        if self._nih is None:
            root = self.cfg.nih_root
            source = pd.read_csv(root / "Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels"])
            test = (pd.read_csv(root/'NIH_test_data.csv',usecols=['Image Index']) if (root/'NIH_test_data.csv').is_file() else pd.DataFrame({'Image Index':(root/'test_list.txt').read_text().splitlines()}))
            source["Image Index"] = source["Image Index"].astype(str).str.strip()
            names = test["Image Index"].astype(str).str.strip().tolist()
            source = source.set_index("Image Index")
            findings = (source.loc[names, "Finding Labels"].str.replace("_", " ", regex=False)
                        .str.split("|").map(lambda vs: {v.strip() for v in vs}))
            labels = np.column_stack([findings.map(lambda vs: int(c in vs)).to_numpy() for c in NIH_CLASSES])
            image_map = {p.name: p for p in root.rglob("*.png")}
            missing = [n for n in names if n not in image_map]
            if missing:
                raise FileNotFoundError(f"NIH images missing: {missing[:10]}")
            self._nih = dict(names=names, paths=[image_map[n] for n in names], labels=labels,
                             findings=source.loc[names, "Finding Labels"].tolist())
            print(f"  NIH test: {len(names):,} images, {len(NIH_CLASSES)} classes")
        return self._nih


# =============================================================================
# 컨텍스트: 임베딩 캐시
# =============================================================================
class Context:
    def __init__(self, cfg: Config, encoders: dict | None = None, out_name: str = "analysis"):
        self.cfg = cfg
        self.out = result_directory(cfg.root, out_name)
        self.cache = cfg.root / "cache" / "oblique_cone"
        self.out.mkdir(parents=True, exist_ok=True); self.cache.mkdir(parents=True, exist_ok=True)
        self.legacy_cache = cfg.root / "cache" / "conformity_imageness"
        self._encoders = dict(encoders or {})
        self.data = Datasets(cfg)
        self.summary: dict[str, Any] = {}
        self._images, self._splits, self.noise, self.text_noise = {}, {}, {}, {}
        self.axes, self.text_axes, self.perturb = {}, {}, {}
        if cfg.n_reference < 2 or cfg.n_noise < 2 or cfg.bootstrap_repeats < 2 or cfg.null_directions < 1:
            raise ValueError("Sizes are too small.")
        if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
            raise ValueError("Seeds must be distinct.")

    def encoder(self, name: str):
        if name not in self._encoders:
            self._encoders[name] = ClipEncoder(self.cfg) if name == "clip" else CarzeroEncoder(self.cfg)
        return self._encoders[name]

    def cached(self, stem: str, metadata: dict, compute) -> np.ndarray:
        cache = self.legacy_cache if metadata.get("version") == CACHE_VERSION else self.cache
        cache.mkdir(parents=True, exist_ok=True)
        return cached_array(cache, self.cfg.recompute, stem, metadata, compute)

    # ---------------- 이미지 임베딩 ----------------
    def images(self, enc: str, dataset: str) -> dict:
        """dict(Z, ids, paths[, labels]). 자기 도메인은 전체, 교차 칸은 n_ood개 고정 표본."""
        key = (enc, dataset)
        if key in self._images:
            return self._images[key]
        cfg = self.cfg
        if dataset == "coco":
            ids_all = np.array(sorted(self.data.coco_images), dtype=int)
            paths_all = [self.data.coco_path(i) for i in ids_all]
            labels = None
        else:
            nih = self.data.nih
            ids_all = np.arange(len(nih["names"])); paths_all = nih["paths"]; labels = nih["labels"]
        if DOMAIN[enc] != dataset:
            pick = np.sort(np.random.default_rng(7001).choice(len(ids_all), min(cfg.n_ood, len(ids_all)), replace=False))
        else:
            pick = np.arange(len(ids_all))
        ids = ids_all[pick]; paths = [paths_all[i] for i in pick]
        raw = None
        if enc == "clip" and dataset == "coco":
            eda = cfg.root / "cache" / f"{cfg.clip_model}_{cfg.clip_pretrained}.npz"
            if eda.exists():
                with np.load(eda, allow_pickle=False) as z:
                    eda_raw, eda_ids = z["img"].copy(), z["image_ids"].astype(int)
                if np.array_equal(eda_ids, ids):
                    raw = eda_raw.astype(np.float32)
                    self.clip_clean_hash = legacy_fingerprint(eda_raw) + legacy_fingerprint(eda_ids)
        if enc == "carzero" and dataset == "nih":
            emb = cfg.carzero_embedding_dir / "zs_img_emb_g.pt"; names_file = cfg.carzero_embedding_dir / "image_names.json"
            if emb.exists() and names_file.exists() and json.loads(names_file.read_text()) == self.data.nih["names"]:
                import torch
                raw = torch.load(emb, map_location="cpu").float().numpy()
                self.check_carzero_against_extraction(paths, raw)
        if raw is None:
            encoder = self.encoder(enc)
            meta = dict(kind="images", dataset=dataset, encoder=encoder.signature, n=len(paths),
                        ids_hash=fingerprint(np.asarray(ids)))
            raw = self.cached(f"img_{enc}_{dataset}", meta, lambda: encoder.encode_paths(paths))
        if len(raw) != len(ids):
            raise ValueError(f"Image embedding count mismatch for {key}.")
        if enc == 'clip' and dataset == 'coco' and not hasattr(self, 'clip_clean_hash'):
            self.clip_clean_hash = legacy_fingerprint(raw) + legacy_fingerprint(ids)
        out = dict(Z=normalize(raw), ids=ids, paths=paths, pick=pick,
                   labels=None if labels is None else labels[pick])
        self._images[key] = out
        return out

    def check_carzero_against_extraction(self, paths: list, raw: np.ndarray, k: int = 32) -> None:
        """지금 로드한 CARZero가 1_extract_embeddings.py가 추출한 임베딩을 재현하는지 확인한다.
        이미지: 같은 경로 k장을 다시 인코딩. 텍스트: 원래 프롬프트 "There is {TERM}"."""
        encoder = self.encoder("carzero")
        idx = np.linspace(0, len(paths) - 1, min(k, len(paths))).astype(int)
        cos_img = np.sum(normalize(encoder.encode_paths([paths[i] for i in idx])) * normalize(raw[idx]), 1)
        report = dict(image_min_cos=float(cos_img.min()), image_mean_cos=float(cos_img.mean()))
        pos = self.cfg.carzero_embedding_dir / "zs_text_pos.pt"
        if pos.exists():
            import torch
            ref = torch.load(pos, map_location="cpu").float().numpy()
            new = encoder.encode_texts([f"There is {t}" for t in NIH_TERMS])
            cos_txt = np.sum(normalize(new) * normalize(ref), 1)
            report.update(text_min_cos=float(cos_txt.min()), text_mean_cos=float(cos_txt.mean()))
        self.summary["carzero_reproduces_extraction"] = report
        print(f"  CARZero vs 1_extract_embeddings: {report}")
        if report["image_min_cos"] < 0.999 or report.get("text_min_cos", 1.0) < 0.999:
            raise ValueError("Loaded CARZero does not reproduce the stored embeddings (cos < 0.999). "
                             "Check the checkpoint / missing keys before continuing.")

    def split(self, enc: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """자기 도메인 이미지의 (reference, evaluation) 행 번호."""
        key = (enc, seed)
        if key not in self._splits:
            n = len(self.images(enc, DOMAIN[enc])["Z"])
            if n <= self.cfg.n_reference + 10:
                raise ValueError("Not enough images for a reference / evaluation split.")
            perm = np.random.default_rng(seed).permutation(n)
            self._splits[key] = (perm[:self.cfg.n_reference], perm[self.cfg.n_reference:])
        return self._splits[key]

    # ---------------- 텍스트 임베딩 ----------------
    def domain_texts(self, enc: str) -> dict:
        """CLIP: COCO 캡션 전체.  CARZero: NIH 긍정 템플릿 140문장."""
        key = ("texts", enc)
        if key in self._images:
            return self._images[key]
        cfg = self.cfg
        if enc == "clip":
            eda = cfg.root / "cache" / f"{cfg.clip_model}_{cfg.clip_pretrained}.npz"
            if eda.exists():
                with np.load(eda, allow_pickle=False) as z:
                    raw, ann_ids, img_ids = z["txt"].astype(np.float32), z["txt_ann_ids"].astype(int), z["image_ids"].astype(int)
                    rows = z["txt_img_row"].astype(int)
                texts = [self.data.coco_caption_by_ann[int(a)] for a in ann_ids]
                image_ids = img_ids[rows]
            else:
                items = sorted(self.data.coco_caption_by_ann.items())
                ann_ids = np.array([a for a, _ in items]); texts = [t for _, t in items]
                caption_image = {}
                with (cfg.root / "data/coco/annotations/captions_val2017.json").open(encoding="utf-8") as f:
                    for x in json.load(f)["annotations"]:
                        caption_image[int(x["id"])] = int(x["image_id"])
                image_ids = np.array([caption_image[int(a)] for a in ann_ids])
                encoder = self.encoder("clip")
                meta = dict(kind="captions", encoder=encoder.signature, n=len(texts))
                raw = self.cached("txt_clip_coco", meta, lambda: encoder.encode_texts(texts))
            out = dict(T=normalize(raw), texts=texts, image_ids=image_ids)
        else:
            texts, d_idx, l_idx = nih_prompts("pos")
            encoder = self.encoder("carzero")
            meta = dict(kind="nih_pos_templates", encoder=encoder.signature, texts=texts)
            raw = self.cached("txt_carzero_nih_pos", meta, lambda: encoder.encode_texts(texts))
            out = dict(T=normalize(raw), texts=texts, disease=d_idx, template=l_idx)
        self._images[key] = out
        return out

    # ---------------- 노이즈 ----------------
    def noise_sources(self) -> list[tuple[str, str, float]]:
        sources = [(f"gaussian_s{s:.3f}", "gaussian", float(s)) for s in self.cfg.noise_std_sweep]
        return sources + [(REFERENCE_KEY, "uniform", float(1 / np.sqrt(12))),
                          ("uniform_gray", "uniform_gray", float(1 / np.sqrt(12)))]

    def image_noise(self, enc: str, key: str, family: str, sigma: float, seed: int) -> np.ndarray:
        cache_key = (enc, key, seed)
        if cache_key in self.noise:
            return self.noise[cache_key]
        cfg, encoder = self.cfg, self.encoder(enc)
        h, w = encoder.hw
        def generate():
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1101]))
            for _ in range(cfg.n_noise):
                if family == "gaussian":
                    x = 0.5 + sigma * rng.standard_normal((h, w, 3))
                elif family == "uniform":
                    x = rng.uniform(0, 1, (h, w, 3))
                else:
                    x = np.repeat(rng.uniform(0, 1, (h, w, 1)), 3, axis=2)
                yield np.clip(x, 0, 1).astype(np.float32)
        if enc == "clip" and family in ("gaussian", "uniform"):
            # 이전 버전과 같은 metadata → 기존 noise cache 재사용
            meta = dict(version=CACHE_VERSION, kind="noise", family=family, seed=seed, n=cfg.n_noise,
                        gaussian_std=sigma if family == "gaussian" else LEGACY_UNIFORM_STD,
                        encoder=encoder.signature, pixel_range=[0, 1])
            stem = f"noise_{family}_{seed}"
        else:
            meta = dict(kind="noise", family=family, sigma=sigma, seed=seed, n=cfg.n_noise,
                        encoder=encoder.signature, pixel_range=[0, 1])
            stem = f"noise_{enc}_{family}_{seed}"
        U = normalize(self.cached(stem, meta, lambda: encoder.encode_pixels(generate())))
        self.noise[cache_key] = U
        return U

    def reference_noise(self, enc: str, seed: int) -> np.ndarray:
        return self.image_noise(enc, REFERENCE_KEY, "uniform", float(1 / np.sqrt(12)), seed)

    def text_noise_strings(self, seed: int) -> tuple[list[str], list[str]]:
        rng = random.Random(1000 + seed)
        n = self.cfg.n_text_noise_each
        chars = ["".join(rng.choice(CHARSET) for _ in range(rng.randint(20, 75))) for _ in range(n)]
        words = [" ".join(rng.choice(WORD_POOL) for _ in range(rng.randint(2, 15))) for _ in range(n)]
        return chars, words

    def text_noise_embeddings(self, enc: str, seed: int) -> dict:
        key = (enc, seed)
        if key not in self.text_noise:
            chars, words = self.text_noise_strings(seed)
            encoder = self.encoder(enc)
            meta = dict(kind="text_noise", seed=seed, encoder=encoder.signature, n=len(chars) + len(words),
                        char_len=[20, 75], word_len=[2, 15])
            raw = self.cached(f"txtnoise_{enc}_{seed}", meta, lambda: encoder.encode_texts(chars + words))
            U = normalize(raw)
            self.text_noise[key] = dict(U=U, char=U[:len(chars)], word=U[len(chars):], strings=chars + words)
        return self.text_noise[key]

    def pooled_b(self, enc: str, modality: str = "image") -> np.ndarray:
        if modality == "image":
            U = np.concatenate([self.reference_noise(enc, s) for s in self.cfg.seeds])
        else:
            U = np.concatenate([self.text_noise_embeddings(enc, s)["U"] for s in self.cfg.seeds])
        return normalize(U.mean(0))


def _banner(tag: str, title: str):
    print(f"\n{'=' * 70}\n{tag}  {title}\n{'=' * 70}")


# =============================================================================
# A1. 이미지 노이즈 수렴  →  fig1, tab1, tab2
# =============================================================================
def _section_a1(ctx: Context) -> None:
    _banner("A1", "Image-noise convergence: sigma sweep, gray noise, constant images")
    cfg = ctx.cfg
    sources = ctx.noise_sources()
    keys = [k for k, _, _ in sources]
    tab1, tab2, curves = [], [], {}
    for enc in cfg.encoders:
        encoder = ctx.encoder(enc); h, w = encoder.hw
        const_meta = dict(kind="constant", levels=list(cfg.constant_levels), encoder=encoder.signature)
        if enc == "clip":
            const_meta.update(version=CACHE_VERSION)
        const_U = normalize(ctx.cached("constant" if enc == "clip" else f"constant_{enc}", const_meta,
                                       lambda: encoder.encode_pixels(np.full((h, w, 3), v, np.float32)
                                                                     for v in cfg.constant_levels)))
        Zdom = ctx.images(enc, DOMAIN[enc])["Z"]
        pts = {k: [] for k in ("cos_axis", "angle_b_r", "R_b", "angle_to_uniform", "const")}
        reps = {k: [] for k in ("cos_axis", "angle_b_r", "const")}
        noise_med, image_med = [], []
        for seed in cfg.seeds:
            ref, evaluation = ctx.split(enc, seed)
            rng = np.random.default_rng(seed + 5101)
            Zref = Zdom[ref]; r_hat = normalize(Zref.mean(0))
            r_boot = normalize(bootstrap_means(Zref, cfg.bootstrap_repeats, rng))
            U = {k: ctx.image_noise(enc, k, f, s, seed) for k, f, s in sources}
            b = {k: normalize(U[k].mean(0)) for k in keys}
            b_boot = {k: normalize(bootstrap_means(U[k], cfg.bootstrap_repeats, rng)) for k in keys}
            m = {k: normalize(r_hat - b[k]) for k in keys}
            m_boot = {k: normalize(r_boot - b_boot[k]) for k in keys}
            pts["cos_axis"].append([float(m[k] @ m[REFERENCE_KEY]) for k in keys])
            reps["cos_axis"].append(np.column_stack([np.sum(m_boot[k] * m_boot[REFERENCE_KEY], 1) for k in keys]))
            pts["angle_b_r"].append([float(angle(b[k], r_hat)) for k in keys])
            reps["angle_b_r"].append(np.column_stack([angle(b_boot[k], r_boot) for k in keys]))
            pts["R_b"].append([float(np.linalg.norm(U[k].mean(0))) for k in keys])
            pts["angle_to_uniform"].append([float(angle(b[k], b[REFERENCE_KEY])) for k in keys])
            pts["const"].append(const_U @ b[REFERENCE_KEY])
            reps["const"].append(b_boot[REFERENCE_KEY] @ const_U.T)
            noise_med.append(float(np.median(U[REFERENCE_KEY] @ b[REFERENCE_KEY])))
            image_med.append(float(np.median(Zdom[evaluation] @ b[REFERENCE_KEY])))
        ca, ab = pooled_matrix(pts["cos_axis"], reps["cos_axis"]), pooled_matrix(pts["angle_b_r"], reps["angle_b_r"])
        R_b, to_uni = np.mean(pts["R_b"], 0), np.mean(pts["angle_to_uniform"], 0)
        for j, (key, family, sigma) in enumerate(sources):
            bs = [normalize(ctx.image_noise(enc, key, family, sigma, s).mean(0)) for s in cfg.seeds]
            seed_angle = max([float(angle(bs[i], bs[k])) for i in range(len(bs)) for k in range(i + 1, len(bs))],
                             default=np.nan)
            tab1.append(dict(encoder=enc, key=key, family=family, sigma=sigma, R_b=float(R_b[j]),
                             cos_axis_to_uniform_mean=ca["mean"][j], cos_axis_to_uniform_ci_low=ca["ci_low"][j],
                             cos_axis_to_uniform_ci_high=ca["ci_high"][j], angle_b_r_deg_mean=ab["mean"][j],
                             angle_b_r_deg_ci_low=ab["ci_low"][j], angle_b_r_deg_ci_high=ab["ci_high"][j],
                             angle_b_to_uniform_b_deg=float(to_uni[j]), max_angle_between_seed_b_deg=seed_angle))
        const = pooled_matrix(pts["const"], reps["const"])
        for v, mu, lo, hi in zip(cfg.constant_levels, const["mean"], const["ci_low"], const["ci_high"]):
            tab2.append(dict(encoder=enc, pixel_value=float(v), cos_to_b_mean=mu, cos_to_b_ci_low=lo,
                             cos_to_b_ci_high=hi, noise_sample_median_cos_to_b=float(np.mean(noise_med)),
                             data_median_cos_to_b=float(np.mean(image_med))))
        curves[enc] = dict(ca=ca, ab=ab, const=const, noise_med=float(np.mean(noise_med)),
                           image_med=float(np.mean(image_med)))
        gray = keys.index("uniform_gray")
        print(f"  {enc}: R_b(uniform) = {R_b[keys.index(REFERENCE_KEY)]:.3f}, "
              f"angle(b_gray, b_uniform) = {to_uni[gray]:.3f} deg")
    tab1 = save_csv(ctx.out, "tab1_sigma_sweep", tab1)
    tab2 = save_csv(ctx.out, "tab2_constant_images", tab2)
    ctx.summary["A1"] = {enc: dict(
        R_b_uniform=float(tab1[(tab1.encoder == enc) & (tab1.key == REFERENCE_KEY)].R_b.iloc[0]),
        angle_gray_to_uniform_deg=float(tab1[(tab1.encoder == enc) & (tab1.key == "uniform_gray")].angle_b_to_uniform_b_deg.iloc[0]),
        constant_max_cos_to_b=float(tab2[tab2.encoder == enc].cos_to_b_mean.max()),
        noise_sample_median_cos_to_b=curves[enc]["noise_med"]) for enc in cfg.encoders}
    if cfg.make_figures:
        n_g = len(cfg.noise_std_sweep); x = np.asarray(cfg.noise_std_sweep)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for enc in cfg.encoders:
            col, cv = ENC_COLOR[enc], curves[enc]
            band(axes[0], x, cv["ca"]["mean"][:n_g], cv["ca"]["ci_low"][:n_g], cv["ca"]["ci_high"][:n_g],
                 col, ENC_LABEL[enc], marker="o", ms=4)
            band(axes[1], x, cv["ab"]["mean"][:n_g], cv["ab"]["ci_low"][:n_g], cv["ab"]["ci_high"][:n_g],
                 col, ENC_LABEL[enc], marker="o", ms=4)
            axes[1].axhline(cv["ab"]["mean"][keys.index(REFERENCE_KEY)], color=col, ls="--", lw=1)
            band(axes[2], np.asarray(cfg.constant_levels), cv["const"]["mean"], cv["const"]["ci_low"],
                 cv["const"]["ci_high"], col, f"{ENC_LABEL[enc]}: constant image")
            axes[2].axhline(cv["noise_med"], color=col, ls="--", lw=1, label=f"{ENC_LABEL[enc]}: noise samples (median)")
            axes[2].axhline(cv["image_med"], color=col, ls=":", lw=1.2, label=f"{ENC_LABEL[enc]}: in-domain images (median)")
        axes[0].axhline(1, color=MUTED, ls="--", lw=1)
        axes[0].set(xlabel="Gaussian noise σ (pixel scale)", ylabel=r"cos($\hat m_\sigma$, $\hat m_{uniform}$)",
                    title="(a) Imageness axis vs. uniform RGB reference")
        axes[1].set(xlabel="Gaussian noise σ (pixel scale)", ylabel=r"angle($b_\sigma$, $\hat r$) (deg)",
                    title="(b) Noise point vs. mean direction (dashed: uniform)")
        axes[2].set(xlabel="Constant pixel value", ylabel=r"cos(f(constant), $b$)", title="(c) Constant images vs. b")
        legend_below(axes[0]); legend_below(axes[1]); legend_below(axes[2], ncol=2)
        save_figure(ctx.out, fig, "fig1_noise_convergence")


# =============================================================================
# A2–A4. 이미지 / 텍스트 축 기하  →  tab3, fig2, fig3
# =============================================================================
def text_split(ctx: Context, enc: str, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """COCO 캡션: 이미지 reference / evaluation 분할을 따른다. NIH 템플릿(140): 전체를 양쪽에 사용."""
    texts = ctx.domain_texts(enc)
    if enc == "clip":
        ref_rows, _ = ctx.split(enc, seed)
        ref_ids = set(ctx.images(enc, "coco")["ids"][ref_rows].tolist())
        in_ref = np.array([int(i) in ref_ids for i in texts["image_ids"]])
        return np.flatnonzero(in_ref), np.flatnonzero(~in_ref), True
    allrows = np.arange(len(texts["T"]))
    return allrows, allrows, False


def _section_geometry(ctx: Context) -> None:
    _banner("A2–A4", "Image and text axis geometry")
    cfg = ctx.cfg
    rows = []
    for enc in cfg.encoders:
        encoder = ctx.encoder(enc)
        for modality in ("image", "text"):
            points, replicates = [], []
            for seed in cfg.seeds:
                rng = np.random.default_rng(seed + 5202)
                if modality == "image":
                    Z = ctx.images(enc, DOMAIN[enc])["Z"]; ref, ev = ctx.split(enc, seed)
                    Zref, Zev, U = Z[ref], Z[ev], ctx.reference_noise(enc, seed)
                    ctx.axes[(enc, seed)] = Axes(Zref.mean(0), U.mean(0))
                else:
                    T = ctx.domain_texts(enc)["T"]; ref, ev, _ = text_split(ctx, enc, seed)
                    Zref, Zev, U = T[ref], T[ev], ctx.text_noise_embeddings(enc, seed)["U"]
                    ctx.text_axes[(enc, seed)] = Axes(Zref.mean(0), U.mean(0))
                point = geometry_stats(Zref, U, Zev)
                A = Axes(Zref.mean(0), U.mean(0)); S = A.coords(Zev)
                if not np.isclose(point["phi_deg"], 90 - point["angle_b_r_deg"] / 2, atol=1e-6):
                    raise AssertionError("phi identity failed.")
                if not np.allclose((S["c"] - S["n"] + 1 - A.gamma) / (2 * (1 - A.gamma)),
                                   (normalize(Zev) - A.b) @ A.m_hat / A.length):
                    raise AssertionError("h identity failed.")
                reps = []
                for _ in range(cfg.bootstrap_repeats):
                    reps.append(geometry_stats(Zref[rng.integers(0, len(Zref), len(Zref))],
                                               U[rng.integers(0, len(U), len(U))],
                                               Zev[rng.integers(0, len(Zev), len(Zev))]))
                points.append(point); replicates.append(pd.DataFrame(reps))
            row = dict(encoder=enc, modality=modality, domain=DOMAIN[enc],
                       n_reference=len(Zref), n_evaluation=len(Zev))
            for q in points[0]:
                p = pooled([pt[q] for pt in points], [r[q].to_numpy() for r in replicates])
                row.update({f"{q}_mean": p["mean"], f"{q}_ci_low": p["ci_low"], f"{q}_ci_high": p["ci_high"]})
            if modality == "text":
                tn = ctx.text_noise_embeddings(enc, cfg.seeds[0])
                row["R_b_char"] = float(np.mean([np.linalg.norm(ctx.text_noise_embeddings(enc, s)["char"].mean(0)) for s in cfg.seeds]))
                row["R_b_word"] = float(np.mean([np.linalg.norm(ctx.text_noise_embeddings(enc, s)["word"].mean(0)) for s in cfg.seeds]))
                row["noise_truncation_frac"] = encoder.text_truncation(tn["strings"])
                row["split_by_image"] = bool(text_split(ctx, enc, cfg.seeds[0])[2])
                cos_b = [float(ctx.axes[(enc, s)].b @ ctx.text_axes[(enc, s)].b) for s in cfg.seeds]
                cos_m = [float(ctx.axes[(enc, s)].m_hat @ ctx.text_axes[(enc, s)].m_hat) for s in cfg.seeds]
                row["cos_bI_bT"], row["cos_mI_mT"] = float(np.mean(cos_b)), float(np.mean(cos_m))
            rows.append(row)
            print(f"  {enc}/{modality}: gamma = {row['gamma_mean']:.3f}, angle(b, r) = {row['angle_b_r_deg_mean']:.3f}, "
                  f"phi = {row['phi_deg_mean']:.3f}, nearest data angle = {row['nearest_data_angle_deg_mean']:.3f}")
    tab3 = save_csv(ctx.out, "tab3_axis_geometry", rows)
    ctx.summary["A3"] = {f"{r.encoder}/{r.modality}": dict(R_b=r.R_b_mean, R_I=r.R_I_mean, gamma=r.gamma_mean,
                                                         phi_deg=r.phi_deg_mean, gamma_percentile=r.gamma_percentile_mean,
                                                         nearest_data_angle_deg=r.nearest_data_angle_deg_mean)
                         for r in tab3.itertuples()}
    if cfg.make_figures:
        figure_planes(ctx)


def plane_scatter(ax, A: Axes, groups: list, title: str, xlab: str, ylab: str):
    rng = np.random.default_rng(7202)
    allc = []
    for Z, label, color, k in groups:
        S = A.coords(Z); i = rng.choice(len(Z), min(len(Z), k), replace=False)
        ax.scatter(S["c"][i], S["n"][i], s=4, color=color, alpha=0.3, lw=0, label=label)
        allc.extend([S["c"][i].min(), S["n"][i].min()])
    lo = min(allc + [A.gamma]) - 0.03; hi = 1.02
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1, label="h = 0.5 (c = n)")
    ax.plot([A.gamma, 1], [1, A.gamma], color=INK, ls=":", lw=1)
    ax.scatter([1, A.gamma], [A.gamma, 1], color=INK, s=36, zorder=5)
    ax.annotate(r"$\hat r$", (1, A.gamma), xytext=(6, -10), textcoords="offset points")
    ax.annotate("b", (A.gamma, 1), xytext=(6, -10), textcoords="offset points")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel=xlab, ylabel=ylab, title=title)


def figure_planes(ctx: Context) -> None:
    cfg, seed = ctx.cfg, ctx.cfg.seeds[0]
    fig, axes = plt.subplots(1, len(cfg.encoders), figsize=(6.5 * len(cfg.encoders), 6), squeeze=False)
    for ax, enc in zip(axes[0], cfg.encoders):
        A = ctx.axes[(enc, seed)]; dom = DOMAIN[enc]
        Zd = ctx.images(enc, dom)["Z"][ctx.split(enc, seed)[1]]
        Zo = ctx.images(enc, OTHER[dom])["Z"]; U = ctx.reference_noise(enc, seed)
        plane_scatter(ax, A, [(Zd, f"{DATA_LABEL[dom]} (in-domain)", C1, 3000),
                              (Zo, f"{DATA_LABEL[OTHER[dom]]} (other domain)", C3, 3000),
                              (U, "Uniform-noise samples", C2, 1000)],
                      f"{ENC_LABEL[enc]}: image (c, n) plane", r"$c = z^\top\hat r$", r"$n = z^\top b$")
        legend_below(ax)
    save_figure(ctx.out, fig, "fig2_image_planes")
    fig, axes = plt.subplots(1, len(cfg.encoders), figsize=(6.5 * len(cfg.encoders), 6), squeeze=False)
    for ax, enc in zip(axes[0], cfg.encoders):
        A = ctx.text_axes[(enc, seed)]; tn = ctx.text_noise_embeddings(enc, seed)
        T = ctx.domain_texts(enc)["T"]
        label = "COCO captions" if enc == "clip" else "NIH positive templates (140)"
        plane_scatter(ax, A, [(T, label, C1, 3000), (tn["char"], "Random-character noise", C2, 500),
                              (tn["word"], "Random-word noise", C4, 500)],
                      f"{ENC_LABEL[enc]}: text (c, n) plane", r"$c_T = t^\top\hat r_T$", r"$n_T = t^\top b_T$")
        legend_below(ax)
    save_figure(ctx.out, fig, "fig3_text_planes")


# =============================================================================
# A5. b 근접 몽타주  →  fig4, fig5, tab4
# =============================================================================
def _section_a5(ctx: Context) -> None:
    _banner("A5", "Images nearest to / farthest from the noise point")
    cfg, seed, rows = ctx.cfg, ctx.cfg.seeds[0], []
    for fig_no, enc in zip((4, 5), cfg.encoders):
        dom = DOMAIN[enc]; imgs = ctx.images(enc, dom); ev = ctx.split(enc, seed)[1]
        S = ctx.axes[(enc, seed)].coords(imgs["Z"][ev])
        order = np.argsort(-S["n"]); k = cfg.montage_k; N = len(order)
        groups = [("top (closest to b)", order[:k])]
        for q in (0.25, 0.50, 0.75):
            mid = int(round(q * (N - 1)))
            groups.append((f"{int(q * 100)}% by n", order[max(0, mid - k // 2): max(0, mid - k // 2) + k]))
        groups.append(("bottom (farthest from b)", order[-k:][::-1]))
        for gname, idx in groups:
            for rank, i in enumerate(idx):
                row = ev[i]
                text = (ctx.data.coco_first_caption.get(int(imgs["ids"][row]), "") if dom == "coco"
                        else ctx.data.nih["findings"][int(imgs["pick"][row])])
                rows.append(dict(encoder=enc, dataset=dom, group=gname, rank=rank, image=str(imgs["paths"][row].name),
                                 c=float(S["c"][i]), n=float(S["n"][i]), h=float(S["h"][i]),
                                 angle_to_b_deg=float(np.degrees(np.arccos(np.clip(S["n"][i], -1, 1)))), text=text))
        if cfg.make_figures:
            fig, axes = plt.subplots(len(groups), k, figsize=(2.6 * k, 2.9 * len(groups)), squeeze=False)
            for r, (gname, idx) in enumerate(groups):
                for col in range(k):
                    ax = axes[r, col]; ax.axis("off")
                    if col >= len(idx):
                        continue
                    i = idx[col]; row = ev[i]
                    with Image.open(imgs["paths"][row]) as im:
                        ax.imshow(im.convert("RGB"))
                    text = (ctx.data.coco_first_caption.get(int(imgs["ids"][row]), "") if dom == "coco"
                            else ctx.data.nih["findings"][int(imgs["pick"][row])])
                    ax.set_title(f"n={S['n'][i]:.3f}  h={S['h'][i]:.3f}\n{text[:38]}", fontsize=7)
                axes[r, 0].text(-0.08, 0.5, gname, transform=axes[r, 0].transAxes, rotation=90,
                                ha="right", va="center", fontsize=9)
            fig.suptitle(f"{ENC_LABEL[enc]} × {DATA_LABEL[dom]}: images ordered by n = zᵀb (seed {seed})")
            save_figure(ctx.out, fig, f"fig{fig_no}_noise_proximity_{enc}")
    save_csv(ctx.out, "tab4_noise_proximity_images", rows)


# =============================================================================
# A6. 손상 단조성  →  fig6, fig7, tab5, tab6
# =============================================================================
def shuffle_levels(hw: tuple, patch: int) -> tuple[int, ...]:
    """(1, 2, 4, 정렬 격자, 비정렬 격자). 정렬: 블록 = 패치 정수배, 비정렬: 블록이 패치 배수가 아닌 가장 가까운 큰 격자."""
    H = hw[0]
    aligned = H // patch
    unaligned = next(g for g in range(aligned + 1, H + 1) if H % g == 0 and (H // g) % patch != 0)
    return (1, 2, 4, aligned, unaligned)


def perturb_pixels(pixels: np.ndarray, family: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32)
    if family == "gaussian":
        out = pixels + severity * rng.standard_normal(pixels.shape)
    elif family == "blur":
        out = ndimage.gaussian_filter(pixels, sigma=(severity, severity, 0), mode="reflect")
    elif family == "patch_shuffle":
        grid = int(severity); H, W, C = pixels.shape
        if grid < 1 or H % grid or W % grid:
            raise ValueError(f"Patch grid {grid} must divide {H}×{W}.")
        blocks = pixels.reshape(grid, H // grid, grid, W // grid, C).transpose(0, 2, 1, 3, 4).reshape(grid * grid, H // grid, W // grid, C)
        out = blocks[rng.permutation(grid * grid)].reshape(grid, grid, H // grid, W // grid, C).transpose(0, 2, 1, 3, 4).reshape(pixels.shape)
    elif family == "phase":
        if severity == 0:
            return pixels.copy()
        H, W = pixels.shape[:2]
        theta = np.angle(np.fft.fft2(rng.standard_normal((H, W))))
        for i in ([0, H // 2] if H % 2 == 0 else [0]):
            for j in ([0, W // 2] if W % 2 == 0 else [0]):
                theta[i, j] = 0
        out = np.fft.ifft2(np.fft.fft2(pixels, axes=(0, 1)) * np.exp(1j * severity * theta[:, :, None]), axes=(0, 1)).real
    else:
        raise ValueError(family)
    return np.clip(out, 0, 1).astype(np.float32)


def family_levels(cfg: Config, encoder) -> dict:
    return dict(gaussian=cfg.gaussian_levels, blur=cfg.blur_levels,
                patch_shuffle=shuffle_levels(encoder.hw, encoder.patch), phase=cfg.phase_levels)


def _section_a6(ctx: Context) -> None:
    _banner("A6", "Perturbation response with fixed axes")
    cfg = ctx.cfg
    tab5, tab6 = [], []
    for fig_no, enc in zip((6, 7), cfg.encoders):
        encoder = ctx.encoder(enc); dom = DOMAIN[enc]; imgs = ctx.images(enc, dom)
        levels_by_family = family_levels(cfg, encoder)
        for seed in cfg.seeds:
            ev = ctx.split(enc, seed)[1]; n = min(cfg.n_perturb_images, len(ev))
            rows = np.random.default_rng(seed + 2404).choice(ev, n, replace=False)
            paths = [imgs["paths"][r] for r in rows]
            ids = imgs["ids"][rows]; A = ctx.axes[(enc, seed)]
            ctx.perturb[(enc, seed)] = {}
            for family_id, (family, levels) in enumerate(levels_by_family.items()):
                def gen(family=family, levels=levels, family_id=family_id):
                    for p, image_id in zip(paths, ids):
                        pix = encoder.pixels(p)
                        for li, sev in enumerate(levels):
                            key = 0 if family in ("gaussian", "phase") else li
                            rng = np.random.default_rng(np.random.SeedSequence([seed, int(image_id), 2404, family_id, key]))
                            yield perturb_pixels(pix, family, sev, rng)
                if enc == "clip":
                    # 이전 버전과 같은 metadata → 기존 perturbation cache 재사용
                    meta = dict(version=CACHE_VERSION, kind="perturbation", seed=seed, family=family,
                                levels=list(levels), files=[[int(i), p.stat().st_size, p.stat().st_mtime_ns] for i, p in zip(ids, paths)],
                                encoder=encoder.signature, clean_cache_hash=ctx.clip_clean_hash)
                else:
                    meta = dict(kind="perturbation", seed=seed, family=family, levels=list(levels),
                                files=[[str(p.name), p.stat().st_size] for p in paths], encoder=encoder.signature)
                raw = ctx.cached(f"perturb_{enc}_{family}_{seed}" if enc != "clip" else f"perturb_{family}_{seed}",
                                 meta, lambda: encoder.encode_pixels(gen()))
                L = len(levels)
                if raw.shape[0] != n * L:
                    raise ValueError("Perturbation cache has a wrong shape.")
                Z = normalize(raw); S = A.coords(Z)
                mats = {k: S[k].reshape(n, L) for k in ("c", "h", "n", "resid")}
                ctx.perturb[(enc, seed)][family] = dict(Z=Z, n_images=n, levels=np.asarray(levels, float),
                                                       ids=np.repeat(ids, L), sev=np.tile(np.asarray(levels, float), n),
                                                       mats=mats, delta={k: mats[k] - mats[k][:, :1] for k in mats})
            print(f"  {enc} seed={seed}: {n} images × 4 families")
        for family, levels in levels_by_family.items():
            runs = [ctx.perturb[(enc, s)][family] for s in cfg.seeds]
            rng = np.random.default_rng(6303)
            for j, sev in enumerate(levels):
                row = dict(encoder=enc, family=family, level_index=j, severity=float(sev),
                           n_image_instances=sum(r["n_images"] for r in runs))
                for k in SCORES:
                    d = np.concatenate([r["delta"][k][:, j] for r in runs])
                    lo, hi = bootstrap_mean_ci(d, cfg.bootstrap_repeats, rng)
                    row.update({f"delta_{k}_mean": float(d.mean()), f"delta_{k}_ci_low": lo, f"delta_{k}_ci_high": hi})
                for k in ("c", "n"):
                    row[f"{k}_mean"] = float(np.concatenate([r["mats"][k][:, j] for r in runs]).mean())
                tab5.append(row)
            for k in SCORES:
                rho = np.concatenate([per_image_spearman(r["mats"][k], np.asarray(levels, float)) for r in runs])
                v = rho[np.isfinite(rho)]; lo, hi = bootstrap_mean_ci(v, cfg.bootstrap_repeats, rng)
                tab6.append(dict(encoder=enc, family=family, score=k, n_image_instances=len(rho), n_valid=len(v),
                                 mean_spearman=float(v.mean()), ci_low=lo, ci_high=hi,
                                 frac_negative=float(np.mean(v < 0))))
        if cfg.make_figures:
            sub_all = pd.DataFrame([r for r in tab5 if r["encoder"] == enc])
            fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
            for ax, family in zip(axes, FAMILIES):
                sub = sub_all[sub_all.family == family]; x = sub.level_index.to_numpy()
                for k in SCORES:
                    band(ax, x, sub[f"delta_{k}_mean"], sub[f"delta_{k}_ci_low"], sub[f"delta_{k}_ci_high"],
                         SCORE_COLOR[k], SCORE_LABEL[k], marker="o", ms=5)
                ax.axhline(0, color=MUTED, lw=0.8)
                ax.set_xticks(x, [f"{v:.3f}" for v in sub.severity])
                ax.set(xlabel=dict(gaussian="Added noise σ", blur="Blur σ (px)",
                                   patch_shuffle=f"Shuffle grid (aligned = {encoder.hw[0] // encoder.patch})",
                                   phase="Phase randomization")[family],
                       title=family.replace("_", " "))
            axes[0].set_ylabel(f"{ENC_LABEL[enc]}: mean change from clean")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
            save_figure(ctx.out, fig, f"fig{fig_no}_perturbation_{enc}")
    save_csv(ctx.out, "tab5_perturbation_by_level", tab5)
    tab6 = save_csv(ctx.out, "tab6_monotonicity", tab6)
    ctx.summary["A6"] = {f"{r.encoder}/{r.family}/{r.score}": r.mean_spearman for r in tab6.itertuples()}


# =============================================================================
# A7. 무작위 기준점 대조  →  fig8, tab7
# =============================================================================
def _section_a7(ctx: Context) -> None:
    _banner("A7", "Noise point vs. angle-matched random reference points")
    cfg = ctx.cfg
    rows, viol = [], {}
    for enc in cfg.encoders:
        fam = {f: dict(actual=[], boot=[], null=[], tail=[]) for f in FAMILIES}
        nat = dict(actual=[], boot=[], null=[], tail=[])
        Zdom = ctx.images(enc, DOMAIN[enc])["Z"]
        for seed in cfg.seeds:
            A = ctx.axes[(enc, seed)]; rng = np.random.default_rng(seed + 2505)
            D = rng.standard_normal((cfg.null_directions, len(A.r_hat)))
            D = normalize(D - np.outer(D @ A.r_hat, A.r_hat))
            null_b = A.gamma * A.r_hat + np.sqrt(max(0.0, 1 - A.gamma ** 2)) * D
            null_m = (A.r_hat - null_b) / A.length
            if not (np.allclose(null_b @ A.r_hat, A.gamma) and np.allclose(np.linalg.norm(A.r_hat - null_b, axis=1), A.length)):
                raise AssertionError("Null construction failed.")
            offset = np.sum(null_b * null_m, axis=1)
            Zev = Zdom[ctx.split(enc, seed)[1]]; S = A.coords(Zev)
            labels = np.floor((S["c"] - S["c"].min()) / cfg.conformity_caliper).astype(int)
            actual = within_bin_sd(labels, S["h"]); null_sd = within_bin_sd(labels, (Zev @ null_m.T - offset) / A.length)
            boot = []
            for _ in range(cfg.bootstrap_repeats):
                k = rng.integers(0, len(Zev), len(Zev)); boot.append(within_bin_sd(labels[k], S["h"][k]))
            nat["actual"].append(actual); nat["boot"].append(boot); nat["null"].append(null_sd)
            nat["tail"].append((1 + np.sum(null_sd >= actual)) / (1 + len(null_sd)))
            for family, d in ctx.perturb[(enc, seed)].items():
                Sx = A.coords(d["Z"]); L = len(d["levels"])
                act = partial_rank_corr(Sx["h"], d["sev"], Sx["c"], d["ids"])
                H = (d["Z"] @ null_m.T - offset) / A.length
                nr = np.array([partial_rank_corr(H[:, k], d["sev"], Sx["c"], d["ids"]) for k in range(len(null_b))])
                bt = []
                for _ in range(cfg.bootstrap_repeats):
                    pick = rng.integers(0, d["n_images"], d["n_images"]); rr = (pick[:, None] * L + np.arange(L)).ravel()
                    bt.append(partial_rank_corr(Sx["h"][rr], d["sev"][rr], Sx["c"][rr], np.repeat(np.arange(len(pick)), L)))
                v = nr[np.isfinite(nr)]
                fam[family]["actual"].append(act); fam[family]["boot"].append(bt); fam[family]["null"].append(nr)
                fam[family]["tail"].append((1 + np.sum(np.abs(v) >= abs(act))) / (1 + len(v)))
            print(f"  {enc} seed={seed}: done")
        for family in FAMILIES:
            f = fam[family]; null_all = np.concatenate(f["null"]); null_all = null_all[np.isfinite(null_all)]
            p = pooled(f["actual"], f["boot"])
            rows.append(dict(encoder=enc, measurement="partial_spearman_h_vs_severity_given_c", family=family,
                             actual_mean=p["mean"], actual_ci_low=p["ci_low"], actual_ci_high=p["ci_high"],
                             null_mean=float(null_all.mean()), null_abs_q975=float(np.quantile(np.abs(null_all), 0.975)),
                             tail_fraction_max_over_seeds=float(max(f["tail"])), n_null_per_seed=cfg.null_directions))
            viol[(enc, family)] = null_all
        null_all = np.concatenate(nat["null"]); p = pooled(nat["actual"], nat["boot"])
        rows.append(dict(encoder=enc, measurement="within_c_bin_sd_of_h", family="clean",
                         actual_mean=p["mean"], actual_ci_low=p["ci_low"], actual_ci_high=p["ci_high"],
                         null_mean=float(null_all.mean()), null_abs_q975=float(np.quantile(null_all, 0.975)),
                         tail_fraction_max_over_seeds=float(max(nat["tail"])), n_null_per_seed=cfg.null_directions))
    tab7 = save_csv(ctx.out, "tab7_null_summary", rows)
    ctx.summary["A7"] = {f"{r.encoder}/{r.family}": dict(actual=r.actual_mean, null_q975=r.null_abs_q975,
                                                       tail=r.tail_fraction_max_over_seeds) for r in tab7.itertuples()}
    if cfg.make_figures:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
        ax = axes[0]; width = 0.38
        for k, enc in enumerate(cfg.encoders):
            pos = np.arange(len(FAMILIES)) + (k - 0.5) * width
            parts = ax.violinplot([viol[(enc, f)] for f in FAMILIES], positions=pos, widths=width * 0.9, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(ENC_COLOR[enc]); body.set_alpha(0.18); body.set_edgecolor("none")
            sub = tab7[(tab7.encoder == enc) & (tab7.family != "clean")].set_index("family").loc[list(FAMILIES)]
            ax.errorbar(pos, sub.actual_mean, yerr=[sub.actual_mean - sub.actual_ci_low, sub.actual_ci_high - sub.actual_mean],
                        fmt="o", color=ENC_COLOR[enc], ms=7, capsize=3, label=f"{ENC_LABEL[enc]}: actual b (violin: null)")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(np.arange(len(FAMILIES)), [f.replace("_", " ") for f in FAMILIES])
        ax.set(ylabel="Partial Spearman ρ (h vs severity | c)", ylim=(-1.05, 1.05),
               title="(a) Perturbation: b vs. angle-matched null points")
        legend_below(ax)
        ax = axes[1]
        for k, enc in enumerate(cfg.encoders):
            r = tab7[(tab7.encoder == enc) & (tab7.family == "clean")].iloc[0]
            ax.errorbar([k], [r.actual_mean / r.null_mean],
                        yerr=[[(r.actual_mean - r.actual_ci_low) / r.null_mean], [(r.actual_ci_high - r.actual_mean) / r.null_mean]],
                        fmt="o", color=ENC_COLOR[enc], ms=8, capsize=4, label=f"{ENC_LABEL[enc]}: actual / null mean")
            ax.scatter([k], [r.null_abs_q975 / r.null_mean], marker="_", s=400, color=MUTED,
                       label="null 97.5% / null mean" if k == 0 else None)
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(cfg.encoders)), [ENC_LABEL[e] for e in cfg.encoders])
        ax.set(ylabel="Within-c-bin SD of h (ratio to null mean)", title="(b) Clean images: spread along the axis",
               xlim=(-0.6, len(cfg.encoders) - 0.4))
        legend_below(ax)
        save_figure(ctx.out, fig, "fig8_null_test")


# =============================================================================
# A8. 2×2 교차와 epistemic / aleatoric  →  fig9, tab8
# =============================================================================
def _section_a8(ctx: Context) -> None:
    _banner("A8", "2×2 cross: other domain vs pixel corruption")
    cfg = ctx.cfg
    rows, scatter = [], {}
    score_names = ("h", "c", "n", "resid")
    # 양성 = 불확실한 쪽. h, c는 작을수록 불확실 → 부호 반전해 "클수록 불확실"로 맞춘다.
    sign = dict(h=-1, c=-1, n=1, resid=1)
    for enc in cfg.encoders:
        dom = DOMAIN[enc]; Zd = ctx.images(enc, dom)["Z"]; Zo = ctx.images(enc, OTHER[dom])["Z"]
        tasks = {"domain shift: other domain": None}
        tasks.update({f"corruption: {f}": f for f in FAMILIES})
        tasks["corruption: non-gaussian pooled"] = "pooled"
        res = {(t, s): ([], []) for t in tasks for s in score_names}
        for seed in cfg.seeds:
            A = ctx.axes[(enc, seed)]; rng = np.random.default_rng(seed + 8808)
            clean = A.coords(Zd[ctx.split(enc, seed)[1]]); other = A.coords(Zo)
            for t, fam in tasks.items():
                if fam is None:
                    pos, neg = other, clean
                else:
                    fams = [f for f in FAMILIES if f != "gaussian"] if fam == "pooled" else [fam]
                    cols = [ctx.perturb[(enc, seed)][f] for f in fams]
                    pos = {k: np.concatenate([d["mats"][k][:, -1] for d in cols]) for k in score_names}
                    neg = {k: np.concatenate([d["mats"][k][:, 0] for d in cols]) for k in score_names}
                for s in score_names:
                    a, b = sign[s] * pos[s], sign[s] * neg[s]
                    point = auroc(a, b); boot = []
                    for _ in range(cfg.bootstrap_repeats):
                        boot.append(auroc(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]))
                    res[(t, s)][0].append(point); res[(t, s)][1].append(boot)
            if seed == cfg.seeds[0]:
                d = ctx.perturb[(enc, seed)]
                scatter[enc] = dict(clean=clean, other=other,
                                    corrupt={k: np.concatenate([d[f]["mats"][k][:, -1] for f in FAMILIES if f != "gaussian"])
                                             for k in score_names})
        for (t, s), (pts, bts) in res.items():
            p = pooled(pts, bts)
            rows.append(dict(encoder=enc, task=t, score=s, oriented="higher = more uncertain",
                             auroc_mean=p["mean"], auroc_ci_low=p["ci_low"], auroc_ci_high=p["ci_high"],
                             other_domain_is_training_ood=(enc == "carzero")))
    tab8 = save_csv(ctx.out, "tab8_epistemic_aleatoric", rows)
    ctx.summary["A8"] = {f"{r.encoder}/{r.task}/{r.score}": r.auroc_mean for r in tab8.itertuples()}
    if cfg.make_figures:
        fig, axes = plt.subplots(1, 1 + len(cfg.encoders), figsize=(6 + 5.2 * len(cfg.encoders), 4.8),
                                 gridspec_kw=dict(width_ratios=[1.6] + [1] * len(cfg.encoders)))
        ax = axes[0]; tasks_plot = ["domain shift: other domain", "corruption: non-gaussian pooled"]
        colors = dict(h=C2, c=C1, n=C3, resid=C4); width = 0.18
        xlabels = []
        for g, (enc, task) in enumerate([(e, t) for e in cfg.encoders for t in tasks_plot]):
            xlabels.append(f"{ENC_LABEL[enc]}\n{task.split(':')[0]}")
            for k, s in enumerate(score_names):
                r = tab8[(tab8.encoder == enc) & (tab8.task == task) & (tab8.score == s)].iloc[0]
                ax.bar(g + (k - 1.5) * width, r.auroc_mean - 0.5, width * 0.9, bottom=0.5, color=colors[s],
                       yerr=[[r.auroc_mean - r.auroc_ci_low], [r.auroc_ci_high - r.auroc_mean]], capsize=2,
                       label={"h": "−h", "c": "−c", "n": "n", "resid": "‖δ′‖"}[s] if g == 0 else None)
        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
        ax.set(ylabel="AUROC (positive = other domain / corrupted)", ylim=(0, 1), title="(a) Coordinates for domain shift and corruption")
        legend_below(ax, ncol=4)
        for ax, enc in zip(axes[1:], cfg.encoders):
            sc = scatter[enc]; rng = np.random.default_rng(1)
            for key, label, color in [("clean", "clean (in-domain)", C1), ("corrupt", "corrupted (max, non-gaussian)", C2),
                                      ("other", "other domain", C3)]:
                d = sc[key]; i = rng.choice(len(d["h"]), min(2000, len(d["h"])), replace=False)
                ax.scatter(d["h"][i], d["resid"][i], s=4, alpha=0.3, color=color, lw=0, label=label)
            ax.axvline(0.5, color=MUTED, ls="--", lw=1)
            ax.set(xlabel="imageness h", ylabel="residual ‖δ′‖", title=f"({'bc'[cfg.encoders.index(enc)]}) {ENC_LABEL[enc]}")
            legend_below(ax, ncol=1)
        save_figure(ctx.out, fig, "fig9_epistemic_aleatoric")


# =============================================================================
# A9–A11. 긍정 문장 점수 분해, 분리도–AUC, 상관, 성분 제거  →  fig10, fig11, tab9–tab11
# =============================================================================
def decomposition_units(ctx: Context, enc: str) -> dict:
    """단위별(캡션 / 질환×템플릿) AUC, 네 항, 분리도, 지표. 전체 이미지 평균 r_I와 pooled b를 쓴다."""
    cfg = ctx.cfg
    imgs = ctx.images(enc, DOMAIN[enc]); Z = imgs["Z"]
    texts = ctx.domain_texts(enc); T = texts["T"]
    A = Axes(Z.mean(0), ctx.pooled_b(enc, "image"))
    b_T = ctx.pooled_b(enc, "text")
    D = Z - A.r_I; db, de = D @ A.b, D @ A.e
    E_absb, E_abse = np.abs(db).mean(), np.abs(de).mean()
    Pz = {"orig": Z, "rm_b": normalize(A.project_out(Z, "b")), "rm_be": normalize(A.project_out(Z, "be"))}
    Pt = {"orig": T, "rm_b": normalize(A.project_out(T, "b")), "rm_be": normalize(A.project_out(T, "be"))}
    T_rm_bT = normalize(T - np.outer(T @ b_T, b_T))
    tb, te = T @ A.b, T @ A.e
    M = T @ A.r_I
    nT = len(T)
    if enc == "clip":
        row_of = {int(i): r for r, i in enumerate(imgs["ids"])}
        pos_row = np.array([row_of[int(i)] for i in texts["image_ids"]])
        units = dict(kind="caption", n=nT)
    else:
        labels = imgs["labels"]; units = dict(kind="disease_template", n=nT)
    out = {k: np.full(nT, np.nan) for k in ("auc", "E_abs_U", "dmu_orig", "dmu_rm_b", "dmu_rm_be", "mean_s",
                                              "auc_rm_b_img", "auc_rm_be_img", "auc_rm_bT_text", "auc_rm_both",
                                              "recon_err")}
    Pd = A.project_out(D, "be")
    variants = dict(auc=(Z, T), auc_rm_b_img=(Pz["rm_b"], T), auc_rm_be_img=(Pz["rm_be"], T),
                    auc_rm_bT_text=(Z, T_rm_bT), auc_rm_both=(Pz["rm_b"], T_rm_bT))
    for start in range(0, nT, cfg.caption_chunk):
        sl = slice(start, min(nT, start + cfg.caption_chunk)); cols = np.arange(sl.start, sl.stop)
        S = Z @ T[sl].T                                                  # (N, chunk)
        V = np.outer(db, tb[sl]); Aa = np.outer(de, te[sl])
        U = S - M[sl][None, :] - V - Aa
        U_direct = Pd @ A.project_out(T[sl], "be").T
        out["recon_err"][sl] = np.abs(U - U_direct).max(0)
        out["E_abs_U"][sl] = np.abs(U).mean(0); out["mean_s"][sl] = S.mean(0)
        for name, (Zv, Tv) in variants.items():
            Sv = Zv @ Tv[sl].T
            if enc == "clip":
                sp = Sv[pos_row[sl], np.arange(len(cols))]
                out[name][sl] = ((Sv < sp[None, :]).sum(0) + 0.5 * ((Sv == sp[None, :]).sum(0) - 1)) / (len(Z) - 1)
            else:
                for jj, j in enumerate(cols):
                    y = labels[:, texts["disease"][j]].astype(bool)
                    out[name][j] = auroc(Sv[y, jj], Sv[~y, jj])
        for key in ("orig", "rm_b", "rm_be"):
            Sk = Pz[key] @ Pt[key][sl].T
            if enc == "clip":
                sp = Sk[pos_row[sl], np.arange(len(cols))]
                out[f"dmu_{key}"][sl] = sp - (Sk.sum(0) - sp) / (len(Z) - 1)
            else:
                for jj, j in enumerate(cols):
                    y = labels[:, texts["disease"][j]].astype(bool)
                    out[f"dmu_{key}"][j] = Sk[y, jj].mean() - Sk[~y, jj].mean()
    if np.nanmax(out["recon_err"]) > 1e-6:
        raise AssertionError(f"Four-term reconstruction error {np.nanmax(out['recon_err'])}")
    absM, absV, absA = np.abs(M), np.abs(tb) * E_absb, np.abs(te) * E_abse
    Dn = absM + absV + absA + out["E_abs_U"] + 1e-10
    out.update(M=M, frac_M=absM / Dn, frac_V=absV / Dn, frac_A=absA / Dn, frac_U=out["E_abs_U"] / Dn,
               cos_t_bI=tb, cos_t_e=te, norm_Pt=np.linalg.norm(A.project_out(T, "be"), axis=1),
               cos_t_bT=T @ b_T)
    out["units"] = units; out["axes"] = A
    if enc == "carzero":
        out["disease"] = texts["disease"]; out["template"] = texts["template"]
    return out


def _section_a9(ctx: Context) -> None:
    _banner("A9–A11", "Positive-text score decomposition, separation vs AUC, component removal")
    cfg = ctx.cfg; rng = np.random.default_rng(9909)
    tab9, tab10, tab11, data = [], [], [], {}
    metrics = ["dmu_orig", "dmu_rm_b", "dmu_rm_be", "mean_s", "M", "cos_t_bI", "cos_t_e", "norm_Pt", "cos_t_bT",
               "frac_M", "frac_V", "frac_A", "frac_U"]
    for enc in cfg.encoders:
        u = decomposition_units(ctx, enc); data[enc] = u
        frame = pd.DataFrame({k: u[k] for k in ["auc"] + metrics + ["auc_rm_b_img", "auc_rm_be_img", "auc_rm_bT_text", "auc_rm_both"]})
        if enc == "carzero":
            frame["disease"] = [NIH_CLASSES[d] for d in u["disease"]]; frame["template"] = u["template"]
            unit_frames = {"disease_template (140)": frame,
                           "disease (14, template mean)": frame.groupby("disease", sort=False).mean(numeric_only=True)}
        else:
            unit_frames = {"caption (25k)": frame}
        for col in ("frac_M", "frac_V", "frac_A", "frac_U"):
            lo, hi = bootstrap_mean_ci(frame[col].to_numpy(), cfg.bootstrap_repeats, rng)
            tab9.append(dict(encoder=enc, unit="all", term=col[-1], mean_fraction=float(frame[col].mean()), ci_low=lo, ci_high=hi))
        if enc == "carzero":
            for d, sub in frame.groupby("disease", sort=False):
                for col in ("frac_M", "frac_V", "frac_A", "frac_U"):
                    lo, hi = bootstrap_mean_ci(sub[col].to_numpy(), cfg.bootstrap_repeats, rng)
                    tab9.append(dict(encoder=enc, unit=d, term=col[-1], mean_fraction=float(sub[col].mean()), ci_low=lo, ci_high=hi))
        for unit, f in unit_frames.items():
            rows = []
            for mtr in metrics:
                ok = np.isfinite(f[mtr]) & np.isfinite(f["auc"])
                rho, p = stats.spearmanr(f.loc[ok, mtr], f.loc[ok, "auc"]) if ok.sum() > 2 else (np.nan, np.nan)
                rows.append(dict(encoder=enc, unit=unit, n_units=int(ok.sum()), metric=mtr, rho=float(rho), p=float(p)))
            q = bh_qvalues(np.array([r["p"] for r in rows]))
            for r, qq in zip(rows, q):
                r["q_bh"] = float(qq)
            tab10.extend(rows)
        removal_unit = unit_frames["caption (25k)"] if enc == "clip" else unit_frames["disease (14, template mean)"]
        for variant in ("auc_rm_b_img", "auc_rm_be_img", "auc_rm_bT_text", "auc_rm_both"):
            dlt = (frame[variant] - frame["auc"]).to_numpy(); lo, hi = bootstrap_mean_ci(dlt, cfg.bootstrap_repeats, rng)
            tab11.append(dict(encoder=enc, unit="all", variant=variant, auc_original=float(frame["auc"].mean()),
                              auc_variant=float(frame[variant].mean()), delta_mean=float(dlt.mean()), ci_low=lo, ci_high=hi))
        if enc == "carzero":
            for d, sub in frame.groupby("disease", sort=False):
                for variant in ("auc_rm_b_img", "auc_rm_be_img", "auc_rm_bT_text", "auc_rm_both"):
                    dlt = (sub[variant] - sub["auc"]).to_numpy(); lo, hi = bootstrap_mean_ci(dlt, cfg.bootstrap_repeats, rng)
                    tab11.append(dict(encoder=enc, unit=d, variant=variant, auc_original=float(sub["auc"].mean()),
                                      auc_variant=float(sub[variant].mean()), delta_mean=float(dlt.mean()), ci_low=lo, ci_high=hi))
        print(f"  {enc}: units = {len(frame)}, mean AUC = {frame['auc'].mean():.3f}, "
              f"rho(dmu_rm_be, AUC) = {[r['rho'] for r in tab10 if r['encoder'] == enc and r['metric'] == 'dmu_rm_be'][0]:.3f}")
        data[enc]["frame"] = frame
    tab9 = save_csv(ctx.out, "tab9_decomposition", tab9)
    tab10 = save_csv(ctx.out, "tab10_metric_auc_correlation", tab10)
    tab11 = save_csv(ctx.out, "tab11_component_removal", tab11)
    ctx.summary["A9"] = {f"{r.encoder}/{r.unit}/{r.metric}": dict(rho=r.rho, q=r.q_bh)
                         for r in tab10.itertuples() if r.metric in ("dmu_orig", "dmu_rm_b", "dmu_rm_be")}
    if cfg.make_figures:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        ax = axes[0]; term_color = dict(M=C1, V=C2, A=C4, U=C3)
        groups = [("clip", "all", "CLIP: COCO captions"), ("carzero", "all", "CARZero: NIH all")] + \
                 [("carzero", d, d) for d in NIH_CLASSES]
        x = np.arange(len(groups)); bottom = np.zeros(len(groups))
        for term in ("M", "V", "A", "U"):
            vals = []
            for enc, unit, _ in groups:
                sel = tab9[(tab9.encoder == enc) & (tab9.unit == unit) & (tab9.term == term)]
                vals.append(float(sel.mean_fraction.iloc[0]) if len(sel) else np.nan)
            ax.bar(x, vals, bottom=bottom, color=term_color[term], width=0.8, label=term, edgecolor="white", lw=0.5)
            bottom += np.nan_to_num(vals)
        ax.set_xticks(x, [g[2] for g in groups], rotation=70, fontsize=7)
        ax.set(ylabel="Share of |M|+E|V|+E|A|+E|U|", title="(a) Four-term contribution (positive text)")
        legend_below(ax, ncol=4, offset=-0.48)
        ax = axes[1]; f = data["carzero"]["frame"].groupby("disease", sort=False).mean(numeric_only=True)
        ax.scatter(f["dmu_rm_be"], f["auc"], color=C2, s=40, zorder=3)
        for name, r in f.iterrows():
            ax.annotate(name[:5], (r["dmu_rm_be"], r["auc"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        rho = tab10[(tab10.encoder == "carzero") & (tab10.unit.str.startswith("disease (")) & (tab10.metric == "dmu_rm_be")].rho.iloc[0]
        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.set(xlabel="Δμ(s) in residual space (3-way)", ylabel="AUC", title=f"(b) NIH: residual separation vs AUC (ρ = {rho:.3f})")
        ax = axes[2]; f = data["clip"]["frame"]
        hb = ax.hexbin(f["dmu_rm_be"], f["auc"], gridsize=45, cmap="Blues", mincnt=1, bins="log")
        fig.colorbar(hb, ax=ax, label="captions (log)")
        rho = tab10[(tab10.encoder == "clip") & (tab10.metric == "dmu_rm_be")].rho.iloc[0]
        ax.set(xlabel="Δμ(s) in residual space (3-way)", ylabel="Caption-image matching AUC",
               title=f"(c) COCO: residual separation vs AUC (ρ = {rho:.3f})")
        save_figure(ctx.out, fig, "fig10_decomposition")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 2]))
        names = dict(auc_rm_b_img="−b (image)", auc_rm_be_img="−b, −e (image)", auc_rm_bT_text="−b_T (text)",
                     auc_rm_both="−b (image), −b_T (text)")
        vcol = dict(auc_rm_b_img=C1, auc_rm_be_img=C2, auc_rm_bT_text=C3, auc_rm_both=C4)
        sub = tab11[(tab11.encoder == "clip") & (tab11.unit == "all")]
        axes[0].bar(range(len(sub)), sub.delta_mean, color=[vcol[v] for v in sub.variant],
                    yerr=[sub.delta_mean - sub.ci_low, sub.ci_high - sub.delta_mean], capsize=3)
        axes[0].set_xticks(range(len(sub)), [names[v] for v in sub.variant], rotation=20, fontsize=8)
        axes[0].axhline(0, color=MUTED, lw=0.8); axes[0].set(ylabel="ΔAUC (mean over captions)", title="(a) COCO")
        sub = tab11[(tab11.encoder == "carzero") & (tab11.unit != "all")]
        width = 0.2; diseases = list(dict.fromkeys(sub.unit))
        for k, v in enumerate(names):
            s2 = sub[sub.variant == v].set_index("unit").loc[diseases]
            axes[1].bar(np.arange(len(diseases)) + (k - 1.5) * width, s2.delta_mean, width, color=vcol[v], label=names[v],
                        yerr=[s2.delta_mean - s2.ci_low, s2.ci_high - s2.delta_mean], capsize=1.5)
        axes[1].set_xticks(range(len(diseases)), diseases, rotation=60, fontsize=7)
        axes[1].axhline(0, color=MUTED, lw=0.8)
        axes[1].set(ylabel="ΔAUC (mean over 10 templates)", title="(b) NIH (CARZero)")
        legend_below(axes[1], ncol=4, offset=-0.42)
        save_figure(ctx.out, fig, "fig11_component_removal")


# =============================================================================
# Main
# =============================================================================





# =============================================================================
# 실험별 출력: 여러 내부 표/그림을 CSV/PNG 한 쌍으로 묶는다.
# =============================================================================
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from itertools import combinations
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import brentq

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix', 'font.size': 10})
_ACTIVE_OUTPUT = None


class ExperimentOutput:
    """table 열로 내부 표를 구분한다. 모든 원자료/통계는 같은 CSV에 보존한다."""
    def __init__(self, out, name):
        self.out, self.name = Path(out), name
        self.tables, self.images = [], []

    def __enter__(self):
        global _ACTIVE_OUTPUT
        if _ACTIVE_OUTPUT is not None:
            raise RuntimeError('실험 출력 컨텍스트를 중첩할 수 없습니다.')
        self.out.mkdir(parents=True, exist_ok=True)
        _ACTIVE_OUTPUT = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_OUTPUT
        _ACTIVE_OUTPUT = None
        if exc_type:
            for im in self.images:
                im.close()
            return False
        if not self.tables:
            raise RuntimeError(f'{self.name}: 저장할 실험 결과가 없습니다.')
        frame = pd.concat(self.tables, ignore_index=True, sort=False)
        csv_path = self.out / f'{self.name}.csv'
        staging = ROOT / 'cache' / 'output_staging'
        staging.mkdir(parents=True, exist_ok=True)
        fd, temporary_csv = tempfile.mkstemp(prefix=self.name+'_', suffix='.csv.tmp', dir=staging)
        os.close(fd)
        temp = Path(temporary_csv)
        frame.to_csv(temp, index=False, float_format='%.3f')
        temp.replace(csv_path)
        if not self.images:
            fig, ax = plt.subplots(figsize=(10, 4))
            count = frame.groupby('table', sort=False).size()
            ax.barh(count.index, count.values, color=C1)
            ax.set(title=self.name, xlabel='Recorded rows')
            _capture_figure(self, fig)
        width = min(max(im.width for im in self.images), 3600)
        panels = [im.resize((width, max(1, round(im.height*width/im.width))), Image.Resampling.LANCZOS)
                  for im in self.images]
        canvas = Image.new('RGB', (width, sum(im.height for im in panels)+24*(len(panels)-1)), 'white')
        y = 0
        for im in panels:
            canvas.paste(im, (0, y)); y += im.height+24
        png_path = self.out / f'{self.name}.png'
        fd, temporary_png = tempfile.mkstemp(prefix=self.name+'_', suffix='.png.tmp', dir=staging)
        os.close(fd)
        temporary = Path(temporary_png)
        try:
            canvas.save(temporary, format='PNG'); temporary.replace(png_path)
        finally:
            temporary.unlink(missing_ok=True)
        for im in self.images + panels + [canvas]:
            im.close()
        print(f'  {self.name}: CSV {len(frame)} rows / PNG one', flush=True)


def _capture_figure(output, fig):
    for ax in fig.axes:
        axes=(ax.xaxis,ax.yaxis)+((ax.zaxis,) if hasattr(ax,'zaxis') else ())
        for axis in axes:
            # 명시적인 범주/수준 이름은 유지하고 자동 숫자 눈금만 세 자리로 맞춘다.
            if not axis.get_units() and not isinstance(axis.get_major_locator(),matplotlib.ticker.FixedLocator):
                axis.set_major_formatter(FormatStrFormatter('%.3f'))
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    buf.seek(0)
    with Image.open(buf) as im:
        output.images.append(im.convert('RGB').copy())
    buf.close(); plt.close(fig)


def save_csv(out, name, rows):
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if _ACTIVE_OUTPUT is None:
        raise RuntimeError('실험 함수 안에서 CSV를 저장해야 합니다.')
    collected = frame.copy(); collected.insert(0, 'table', name)
    _ACTIVE_OUTPUT.tables.append(collected)
    return frame


def save_figure(out, fig, name):
    if _ACTIVE_OUTPUT is None:
        raise RuntimeError('실험 함수 안에서 그림을 저장해야 합니다.')
    _capture_figure(_ACTIVE_OUTPUT, fig)


@contextmanager
def suppress_outputs():
    """단독 실행한 뒤쪽 실험의 선행 계산을 메모리에서 준비한다."""
    global _ACTIVE_OUTPUT
    previous = _ACTIVE_OUTPUT
    dummy = ExperimentOutput(Path('.'), '_preparation')
    _ACTIVE_OUTPUT = dummy
    try:
        yield
    finally:
        _ACTIVE_OUTPUT = previous
        for im in dummy.images:
            im.close()


def experiment(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(ctx, *args, **kwargs):
            with ExperimentOutput(ctx.out, name):
                return fn(ctx, *args, **kwargs)
        return wrapped
    return decorate


def ensure_geometry(ctx):
    if any((enc, seed) not in ctx.axes for enc in ctx.cfg.encoders for seed in ctx.cfg.seeds):
        with suppress_outputs():
            _section_geometry(ctx)


def ensure_corruptions(ctx):
    ensure_geometry(ctx)
    if any((enc, seed) not in ctx.perturb for enc in ctx.cfg.encoders for seed in ctx.cfg.seeds):
        with suppress_outputs():
            _section_a6(ctx)


MATCHED_FAMILIES = ('gaussian_matched', 'uniform_matched', 'gray_matched')


def matched_noise_pixels(hw, family, sigma, rng):
    """[0,1], 평균 0.5, 모집단 표준편차 sigma. 채널 의존성은 유형의 일부다.
    Gaussian은 [0,1] truncated normal: clipping 후 분산을 섞어 비교하지 않는다.
    """
    if not 0 < sigma < 1/np.sqrt(12):
        raise ValueError('matched_noise_std는 0과 균일[0,1] 표준편차 사이여야 합니다.')
    shape = (*hw, 1 if family == 'gray_matched' else 3)
    if family == 'gaussian_matched':
        def sd(scale):
            limit = 0.5/scale
            return scale*np.sqrt(stats.truncnorm.var(-limit, limit))
        scale = brentq(lambda x: sd(x)-sigma, sigma, 20*sigma)
        bound = 0.5/scale
        pixels = stats.truncnorm.rvs(-bound, bound, loc=0.5, scale=scale, size=shape, random_state=rng)
    elif family in ('uniform_matched', 'gray_matched'):
        radius = np.sqrt(3)*sigma
        pixels = rng.uniform(0.5-radius, 0.5+radius, shape)
    else:
        raise ValueError(family)
    if shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return np.asarray(pixels, dtype=np.float32)


def matched_noise_embeddings(ctx, enc, family, seed):
    encoder, cfg = ctx.encoder(enc), ctx.cfg
    def generate():
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3151, MATCHED_FAMILIES.index(family)]))
        for _ in range(cfg.n_noise):
            yield matched_noise_pixels(encoder.hw, family, cfg.matched_noise_std, rng)
    meta = dict(kind='moment_matched_noise_v1', family=family, n=cfg.n_noise, seed=seed,
                sigma=cfg.matched_noise_std, encoder=encoder.signature)
    return normalize(ctx.cached(f'matched_{enc}_{family}', meta, lambda: encoder.encode_pixels(generate())))


@experiment('exp01_image_noise_convergence')
def exp01_image_noise_convergence(ctx):
    """분포별 수렴점, 같은 모멘트 대조, 표본 수/독립 seed별 추정 안정성."""
    _section_a1(ctx)
    rows, pair_rows = [], []
    for enc in ctx.cfg.encoders:
        family_seeds = {f: [] for f in MATCHED_FAMILIES}
        for seed in ctx.cfg.seeds:
            uniform = normalize(ctx.reference_noise(enc, seed).mean(0))
            for family in MATCHED_FAMILIES:
                U = matched_noise_embeddings(ctx, enc, family, seed)
                b = normalize(U.mean(0)); family_seeds[family].append(b)
                encoder = ctx.encoder(enc)
                probe = matched_noise_pixels(encoder.hw, family, ctx.cfg.matched_noise_std,
                                             np.random.default_rng(seed+401))
                for n in sorted(set(min(n, len(U)) for n in ctx.cfg.noise_sample_sizes) | {len(U)}):
                    bn = normalize(U[:n].mean(0))
                    rows.append(dict(encoder=enc, seed=seed, family=family, n_noise=n,
                                     Rb=float(np.linalg.norm(U[:n].mean(0))),
                                     angle_to_full_sample_deg=float(angle(bn, b)),
                                     angle_to_uniform_rgb_deg=float(angle(bn, uniform)),
                                     target_pixel_mean=0.500, target_pixel_sd=ctx.cfg.matched_noise_std,
                                     observed_probe_pixel_mean=float(probe.mean()), observed_probe_pixel_sd=float(probe.std())))
            for left, right in combinations(MATCHED_FAMILIES, 2):
                pair_rows.append(dict(encoder=enc, seed=seed, family_a=left, family_b=right,
                                      angle_deg=float(angle(family_seeds[left][-1], family_seeds[right][-1]))))
        for family, vertices in family_seeds.items():
            for i, j in combinations(range(len(vertices)), 2):
                pair_rows.append(dict(encoder=enc, family_a=family, family_b=family,
                                      seed_a=ctx.cfg.seeds[i], seed_b=ctx.cfg.seeds[j],
                                      angle_deg=float(angle(vertices[i], vertices[j]))))
    df = save_csv(ctx.out, 'matched_moments_and_sample_size', rows)
    pairs = save_csv(ctx.out, 'vertex_pair_angles', pair_rows)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for (enc, family), d in df.groupby(['encoder', 'family']):
        g = d.groupby('n_noise').angle_to_full_sample_deg.mean()
        axs[0].plot(g.index, g.values, marker='o', label=f'{enc}/{family}')
    axs[0].set(xlabel='Number of noise samples', ylabel='Angle to full-sample estimate (deg)')
    axs[0].legend(fontsize=7)
    if len(pairs):
        d = pairs.dropna(subset=['seed']) if 'seed' in pairs else pairs.iloc[:0]
        if len(d):
            g = d.groupby(['encoder','family_a','family_b']).angle_deg.mean()
            axs[1].barh(['/'.join(k) for k in g.index], g.values)
    axs[1].set(xlabel='Angle between type-specific noise vertices (deg)')
    save_figure(ctx.out, fig, 'matched_noise_controls')


@experiment('exp02_text_noise_convergence')
def exp02_text_noise_convergence(ctx):
    """문자열/단어열/합친 노이즈의 집중도와 표본 수·seed별 꼭지점 안정성."""
    rows, pairs = [], []
    for enc in ctx.cfg.encoders:
        vertices = {}
        for seed in ctx.cfg.seeds:
            U = ctx.text_noise_embeddings(enc, seed)
            for family in ('char','word','U'):
                X = U[family]; b = normalize(X.mean(0)); vertices[(family, seed)] = b
                for n in sorted(set(min(n, len(X)) for n in ctx.cfg.noise_sample_sizes) | {len(X)}):
                    bn = normalize(X[:n].mean(0))
                    lo, hi = bootstrap_mean_ci(X[:n] @ bn, ctx.cfg.bootstrap_repeats, np.random.default_rng(seed+44))
                    rows.append(dict(encoder=enc, seed=seed, family=family, n_noise=n,
                                     Rb=float(np.linalg.norm(X[:n].mean(0))),
                                     angle_to_full_sample_deg=float(angle(bn,b)),
                                     mean_cos_to_vertex=float(np.mean(X[:n] @ bn)),
                                     ci_low=lo, ci_high=hi))
            pairs.append(dict(encoder=enc, seed=seed, comparison='char_vs_word',
                              angle_deg=float(angle(vertices[('char',seed)],vertices[('word',seed)]))))
        for family in ('char','word','U'):
            for left,right in combinations(ctx.cfg.seeds,2):
                pairs.append(dict(encoder=enc, family=family, seed_a=left, seed_b=right,
                                  comparison='independent_seeds', angle_deg=float(angle(vertices[(family,left)],vertices[(family,right)]))))
    df=save_csv(ctx.out,'concentration',rows); save_csv(ctx.out,'vertex_angles',pairs)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for (enc,family),d in df.groupby(['encoder','family']):
        g=d.groupby('n_noise').mean(numeric_only=True)
        axs[0].plot(g.index,g.Rb,marker='o',label=f'{enc}/{family}')
        axs[1].plot(g.index,g.angle_to_full_sample_deg,marker='o',label=f'{enc}/{family}')
    axs[0].set(xlabel='Noise sample count',ylabel='Resultant length Rb');axs[0].legend(fontsize=7)
    axs[1].set(xlabel='Noise sample count',ylabel='Angle to full-sample estimate (deg)')
    save_figure(ctx.out,fig,'text_noise')


@experiment('exp03_axis_geometry')
def exp03_axis_geometry(ctx):
    """conformity O→r와 Imageness b→r의 기하, 구면 그림, 실제 표본 몽타주."""
    _section_geometry(ctx); _section_a5(ctx)
    fig=plt.figure(figsize=(12,10)); rows=[]
    for k,(enc,modality) in enumerate((e,m) for e in ctx.cfg.encoders for m in ('image','text')):
        if k>=4: break
        A=(ctx.axes if modality=='image' else ctx.text_axes)[(enc,ctx.cfg.seeds[0])]
        # b/r가 놓인 평면의 좌표계를 사용한다. 단위구면 위 끝점의 위치를 보존한다.
        u,v=np.meshgrid(np.linspace(0,2*np.pi,36),np.linspace(0,np.pi,22))
        ax=fig.add_subplot(2,2,k+1,projection='3d')
        ax.plot_wireframe(np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v),color='#bbbbbb',alpha=0.18,lw=0.4)
        r=np.array([1.,0.,0.]);b=np.array([A.gamma,np.sqrt(1-A.gamma**2),0.]);origin=np.zeros(3)
        for start,end,color,label in ((origin,r,C1,'Conformity O→r'),(b,r,C2,'Imageness b→r' if modality=='image' else 'Textness b→r')):
            ax.quiver(*start,*(end-start),color=color,arrow_length_ratio=0.08,label=label)
        for point,label in ((origin,'O'),(b,'b'),(r,'r')):
            ax.scatter(*point,color=INK,s=22);ax.text(*(point+0.04),label)
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),title=f'{enc}/{modality}',box_aspect=(1,1,1));ax.legend(fontsize=7)
        rows.append(dict(encoder=enc,modality=modality,seed=ctx.cfg.seeds[0],
                         r_norm=float(np.linalg.norm(A.r_hat)),b_norm=float(np.linalg.norm(A.b)),
                         h_at_b=float(A.coords(A.b[None])['h'][0]),h_at_r=float(A.coords(A.r_hat[None])['h'][0])))
    save_csv(ctx.out,'endpoint_checks',rows);save_figure(ctx.out,fig,'sphere_axes')


def trajectory_metrics(Z, b):
    """개별 경로의 노이즈점 접근과 실제 임베딩 공간의 길이를 측정한다."""
    n=np.einsum('nld,d->nl',Z,b)
    distance=np.sqrt(np.maximum(0,2-2*n))
    increments=np.linalg.norm(np.diff(Z,axis=1),axis=2)
    displacement=np.linalg.norm(Z[:,-1]-Z[:,0],axis=1)
    length=increments.sum(1)
    straightness=np.divide(displacement,length,out=np.full_like(length,np.nan),where=length>EPS)
    return dict(n=n,distance=distance,path_length=length,displacement=displacement,straightness=straightness)


@experiment('exp04_noise_trajectories')
def exp04_noise_trajectories(ctx):
    """기존 손상 응답 + 같은 이미지의 유형별 혼합 경로/자기·교차 꼭지점 접근.
    endpoint는 개별 노이즈 표본이다. 평균 꼭지점까지 거리가 0이라는 가정은 두지 않는다.
    """
    ensure_geometry(ctx);_section_a6(ctx)
    cfg=ctx.cfg;levels=np.asarray(cfg.trajectory_levels,float)
    if len(levels)<3 or not np.all(np.diff(levels)>0) or levels[0]!=0 or levels[-1]!=1:
        raise ValueError('trajectory_levels는 0에서 1까지 증가하며 세 수준 이상이어야 합니다.')
    summaries=[]; details=[]; crossings=[]; separations=[]; paths=[]; within_rows=[]
    for enc in cfg.encoders:
        encoder=ctx.encoder(enc);imgs=ctx.images(enc,DOMAIN[enc]);L=len(levels)
        for seed in cfg.seeds:
            ev=ctx.split(enc,seed)[1]
            chosen=np.random.default_rng(seed+3501).choice(ev,min(cfg.n_trajectory_images,len(ev)),replace=False)
            r=ctx.axes[(enc,seed)].r_hat
            vertices={f:normalize(matched_noise_embeddings(ctx,enc,f,seed).mean(0)) for f in MATCHED_FAMILIES}
            family_paths={};repeat_paths={}
            for fi,family in enumerate(MATCHED_FAMILIES):
                distort=[]
                def generate():
                    for row in chosen:
                        clean=encoder.pixels(imgs['paths'][row])
                        rng=np.random.default_rng(np.random.SeedSequence([seed,int(imgs['ids'][row]),3601,fi]))
                        noise=matched_noise_pixels(encoder.hw,family,cfg.matched_noise_std,rng)
                        for alpha in levels:
                            x=(1-alpha)*clean+alpha*noise
                            distort.append(float(np.sqrt(np.mean((x-clean)**2))))
                            yield x.astype(np.float32)
                meta=dict(kind='matched_blend_trajectory_v1',encoder=encoder.signature,seed=seed,family=family,
                          levels=levels.tolist(),sigma=cfg.matched_noise_std,
                          files=[[str(imgs['paths'][j]),imgs['paths'][j].stat().st_size,imgs['paths'][j].stat().st_mtime_ns] for j in chosen])
                Z=normalize(ctx.cached(f'trajectory_{enc}_{family}',meta,lambda:encoder.encode_pixels(generate()))).reshape(len(chosen),L,-1)
                # 픽셀 왜곡은 캐시 재사용 시에도 같은 생성 경로로 재구성한다.
                if not distort:
                    for _ in generate(): pass
                A=Axes(r,vertices[family]);S=A.coords(Z.reshape(-1,Z.shape[-1]));T=trajectory_metrics(Z,vertices[family])
                family_paths[family]=Z
                for i,row in enumerate(chosen):
                    rho=float(per_image_spearman(T['distance'][i:i+1],levels)[0])
                    paths.append(dict(encoder=enc,seed=seed,family=family,image_id=int(imgs['ids'][row]),
                                      path_length=float(T['path_length'][i]),displacement=float(T['displacement'][i]),
                                      straightness=float(T['straightness'][i]),distance_severity_spearman=rho,
                                      strictly_nonincreasing_distance=bool(np.all(np.diff(T['distance'][i])<=EPS))))
                    for li,alpha in enumerate(levels):
                        q=i*L+li
                        details.append(dict(encoder=enc,seed=seed,family=family,image_id=int(imgs['ids'][row]),alpha=float(alpha),
                                            pixel_rmse=distort[q],c=float(S['c'][q]),h_own=float(S['h'][q]),
                                            n_own=float(S['n'][q]),distance_own=float(T['distance'][i,li])))
                for li,alpha in enumerate(levels):
                    distance=T['distance'][:,li];lo,hi=bootstrap_mean_ci(distance,cfg.bootstrap_repeats,np.random.default_rng(seed+li+43))
                    summaries.append(dict(encoder=enc,seed=seed,family=family,alpha=float(alpha),
                                          distance_own_mean=float(distance.mean()),ci_low=lo,ci_high=hi,
                                          delta_n_own_mean=float((T['n'][:,li]-T['n'][:,0]).mean()),
                                          h_own_mean=float(S['h'].reshape(len(chosen),L)[:,li].mean())))
                for reference,b in vertices.items():
                    cross=trajectory_metrics(Z,b)
                    for li,alpha in enumerate(levels):
                        crossings.append(dict(encoder=enc,seed=seed,input_family=family,reference_family=reference,
                                              alpha=float(alpha),distance_mean=float(cross['distance'][:,li].mean()),
                                              delta_n_mean=float((cross['n'][:,li]-cross['n'][:,0]).mean())))
                # 유형 차이와 개별 노이즈 표본 차이를 구분하는 같은 유형의 독립 반복.
                def repeat_generate():
                    for row in chosen:
                        clean=encoder.pixels(imgs['paths'][row])
                        rng=np.random.default_rng(np.random.SeedSequence([seed,int(imgs['ids'][row]),3602,fi]))
                        noise=matched_noise_pixels(encoder.hw,family,cfg.matched_noise_std,rng)
                        for alpha in levels:yield ((1-alpha)*clean+alpha*noise).astype(np.float32)
                repeat_meta=dict(meta,replicate=1)
                Zrepeat=normalize(ctx.cached(f'trajectory_repeat_{enc}_{family}',repeat_meta,
                                  lambda:encoder.encode_pixels(repeat_generate()))).reshape(len(chosen),L,-1)
                repeat_paths[family]=Zrepeat
                within=np.linalg.norm(Z-Zrepeat,axis=2)
                for li,alpha in enumerate(levels):
                    within_rows.append(dict(encoder=enc,seed=seed,family=family,alpha=float(alpha),
                                            within_type_separation_mean=float(within[:,li].mean())))
            for left,right in combinations(MATCHED_FAMILIES,2):
                distances=np.linalg.norm(family_paths[left]-family_paths[right],axis=2)
                within=(np.linalg.norm(family_paths[left]-repeat_paths[left],axis=2)+
                        np.linalg.norm(family_paths[right]-repeat_paths[right],axis=2))/2
                for li,alpha in enumerate(levels):
                    lo,hi=bootstrap_mean_ci(distances[:,li],cfg.bootstrap_repeats,np.random.default_rng(seed+li+79))
                    contrast=distances[:,li]-within[:,li]
                    clo,chi=bootstrap_mean_ci(contrast,cfg.bootstrap_repeats,np.random.default_rng(seed+li+81))
                    separations.append(dict(encoder=enc,seed=seed,family_a=left,family_b=right,alpha=float(alpha),
                                            mean_path_separation=float(distances[:,li].mean()),ci_low=lo,ci_high=hi,
                                            within_type_separation_mean=float(within[:,li].mean()),
                                            between_minus_within=float(contrast.mean()),contrast_ci_low=clo,contrast_ci_high=chi,
                                            vertex_angle_deg=float(angle(vertices[left],vertices[right]))))
    summary=save_csv(ctx.out,'type_trajectory_summary',summaries)
    save_csv(ctx.out,'individual_trajectory_coordinates',details);save_csv(ctx.out,'individual_path_metrics',paths)
    cross=save_csv(ctx.out,'cross_vertex_approach',crossings);sep=save_csv(ctx.out,'path_separation',separations)
    save_csv(ctx.out,'within_type_repeated_noise_paths',within_rows)
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for (enc,family),d in summary.groupby(['encoder','family']):
        g=d.groupby('alpha').mean(numeric_only=True);lab=f'{enc}/{family}'
        axs[0,0].plot(g.index,g.distance_own_mean,marker='o',label=lab)
        axs[0,1].plot(g.index,g.h_own_mean,marker='o',label=lab)
    axs[0,0].set(xlabel='Noise mixture fraction',ylabel='Distance to own-type vertex');axs[0,0].legend(fontsize=6)
    axs[0,1].set(xlabel='Noise mixture fraction',ylabel='Own-axis Imageness coordinate')
    final=cross[cross.alpha==1].groupby(['encoder','input_family','reference_family']).delta_n_mean.mean().unstack('reference_family')
    im=axs[1,0].imshow(final.values,cmap='coolwarm');fig.colorbar(im,ax=axs[1,0])
    axs[1,0].set_xticks(range(len(final.columns)),final.columns,rotation=35,ha='right')
    axs[1,0].set_yticks(range(len(final)),['/'.join(k) for k in final.index]);axs[1,0].set(title='Endpoint change in noise cosine: cross vertices')
    for i in range(len(final)):
        for j in range(len(final.columns)):axs[1,0].text(j,i,f'{final.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    for (enc,left,right),d in sep.groupby(['encoder','family_a','family_b']):
        g=d.groupby('alpha').mean(numeric_only=True);axs[1,1].plot(g.index,g.between_minus_within,label=f'{enc}/{left}/{right}')
    axs[1,1].axhline(0,color=INK,lw=0.7)
    axs[1,1].set(xlabel='Noise mixture fraction',ylabel='Between-type minus within-type path separation');axs[1,1].legend(fontsize=6)
    save_figure(ctx.out,fig,'type_trajectories')


@experiment('exp05_random_reference_controls')
def exp05_random_reference_controls(ctx):
    """실제 노이즈점과 동일 gamma/축 길이의 무작위 기준점을 비교한다."""
    ensure_corruptions(ctx);_section_a7(ctx)


def read_mnist_pixels(root, maximum, hw):
    """다운로드 없이 기존 IDX/IDX.gz 파일을 읽는다. 파일 위치는 Config에서 설정."""
    import gzip
    import struct
    roots=(Path(root),Path(root)/'raw',Path(root)/'MNIST'/'raw')
    choices=[p/name for p in roots for name in ('train-images-idx3-ubyte','train-images-idx3-ubyte.gz','t10k-images-idx3-ubyte','t10k-images-idx3-ubyte.gz')]
    path=next((p for p in choices if p.is_file()),None)
    if path is None:raise FileNotFoundError(f'MNIST IDX 파일을 찾을 수 없습니다: {root}')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f:
        magic,n,h,w=struct.unpack('>IIII',f.read(16))
        if magic!=2051:raise ValueError('MNIST 이미지 IDX magic이 잘못되었습니다.')
        X=np.frombuffer(f.read(min(n,maximum)*h*w),np.uint8).reshape(min(n,maximum),h,w)
    for image in X:
        yield np.asarray(Image.fromarray(image).convert('RGB').resize(hw[::-1],Image.Resampling.BILINEAR),np.float32)/255


@experiment('exp06_domain_shift')
def exp06_domain_shift(ctx):
    """도메인 이동과 픽셀 손상을 별도 평가하고 MNIST/비의료 텍스트도 비교한다."""
    ensure_corruptions(ctx);_section_a8(ctx)
    rows=[]
    templates=['the {} {} the {} {}','{} and {} went to the {}','a {} of {} on the {}',
               '{} is better than {}','yesterday the {} was {}','how to {} a {} quickly',
               'the {} {} near the {}','{} {} {} in the {}']
    words=('cat dog house river mountain car book garden bridge forest ocean window guitar planet robot '
           'bicycle dragon castle diamond feather hammer island jacket kitchen ladder mirror notebook painting queen '
           'rocket sandwich telescope umbrella village waterfall jumped crossed found built watched carried opened '
           'discovered followed created measured designed explored small ancient bright cold dark elegant fast '
           'gentle heavy invisible loud mysterious narrow').split()
    rng=random.Random(123);nonmedical=[]
    for _ in range(ctx.cfg.n_nonmedical_text):
        template=rng.choice(templates)
        nonmedical.append(template.format(*(rng.choice(words) for _ in range(template.count('{}')))))
    for enc in ctx.cfg.encoders:
        encoder=ctx.encoder(enc)
        U=normalize(encoder.encode_pixels(read_mnist_pixels(ctx.cfg.mnist_root,ctx.cfg.n_mnist,encoder.hw)))
        T=normalize(encoder.encode_texts(nonmedical))
        for seed in ctx.cfg.seeds:
            for source,X,modality in (('MNIST',U,'image'),('nonmedical_text',T,'text')):
                A=(ctx.axes if modality=='image' else ctx.text_axes)[(enc,seed)];S=A.coords(X)
                rows.append(dict(encoder=enc,seed=seed,source=source,modality=modality,n_samples=len(X),
                                 R=float(np.linalg.norm(X.mean(0))),mean_n=float(S['n'].mean()),
                                 mean_c=float(S['c'].mean()),mean_h=float(S['h'].mean()),
                                 mean_angle_to_b_deg=float(angle(X,A.b).mean())))
    df=save_csv(ctx.out,'original_ood_baselines',rows)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for source,d in df.groupby('source'):
        g=d.groupby('encoder').mean(numeric_only=True)
        axs[0].plot(g.index,g.R,marker='o',label=source);axs[1].plot(g.index,g.mean_n,marker='o',label=source)
    axs[0].set(ylabel='Resultant length R');axs[0].legend();axs[1].set(ylabel='Mean noise proximity n')
    save_figure(ctx.out,fig,'mnist_nonmedical')


EXPERIMENTS=(exp01_image_noise_convergence,exp02_text_noise_convergence,exp03_axis_geometry,
             exp04_noise_trajectories,exp05_random_reference_controls,exp06_domain_shift)


def run_analysis(config=CONFIG, encoders=None):
    ctx=Context(config,encoders)
    try:
        # 두 도메인과 MNIST 자산은 모든 계산을 시작하기 전에 확인한다.
        ctx.data.nih
        next(iter(read_mnist_pixels(config.mnist_root,config.n_mnist,(28,28))))
        for fn in EXPERIMENTS:fn(ctx)
    finally:
        for encoder in ctx._encoders.values():
            if hasattr(encoder,'close'):encoder.close()
    print(f'완료: {ctx.out}',flush=True)
    return ctx.out

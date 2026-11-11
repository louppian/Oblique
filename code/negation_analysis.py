"""부정 프롬프트 잔차 정렬 분석 — 이 파일을 실행합니다.
CLI 인자 없이 아래 Config의 경로를 실행 서버에 맞게 수정합니다.
기존 Oblique의 NIH 데이터, CARZero 가중치 및 캐시 형식을 사용합니다.
다른 분석 .py 파일 없이 데이터 로딩부터 CSV 저장까지 실행합니다.
출력: root/results/section_experiments/negation_alignment/exp08_negation_residual_alignment.csv
학습용 residual_alignment_loss도 포함하며 기본 실행은 분석입니다.
필요 패키지: numpy pandas scipy Pillow matplotlib torch 및 기존 CARZero 환경.
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

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    base: Path = ROOT
    carzero_project: Path = ROOT / 'third_party' / 'BI_MCQ'
    carzero_weight_root: Path = ROOT / 'ckpt'
    carzero_ckpt: str = 'best_model.ckpt'
    nih_root: Path = ROOT / 'data' / 'NIH'
    carzero_embedding_dir: Path = ROOT / 'cache' / 'embeddings'
    clip_model: str = 'ViT-B-32'
    clip_pretrained: str = 'openai'
    encoders: tuple[str, ...] = ('clip', 'carzero')
    device: str | None = None
    batch_size: int = 256
    seeds: tuple[int, ...] = (0, 1, 2)
    n_reference: int = 2000
    n_noise: int = 1000
    n_text_noise_each: int = 500
    n_ood: int = 3000
    noise_std_sweep: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
    constant_levels: tuple[float, ...] = tuple((round(0.01 * k, 2) for k in range(1, 101)))
    n_perturb_images: int = 500
    bootstrap_repeats: int = 400
    null_directions: int = 1000
    conformity_caliper: float = 0.01
    montage_k: int = 6
    caption_chunk: int = 2048
    recompute: bool = False
    make_figures: bool = True
    gaussian_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4)
    blur_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    phase_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    noise_sample_sizes: tuple[int, ...] = (50, 100, 250, 500, 1000)
    matched_noise_std: float = 0.2
    trajectory_levels: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    n_trajectory_images: int = 200
    mnist_root: Path = ROOT / 'data' / 'mnist'
    n_mnist: int = 1000
    n_nonmedical_text: int = 1000
    bimcq_ckpt: str = 'BiMCQ_CARZero_best_model.ckpt'
    bimcq_config: Path = ROOT / 'third_party' / 'BI_MCQ' / 'configs' / 'chest14_finetuning_Bi-MCQ.yaml'
    figure_dpi: int = 180

EPS = np.sqrt(np.finfo(np.float64).eps)

CACHE_VERSION = 'conformity-imageness-v2'

LEGACY_UNIFORM_STD = 0.2

REFERENCE_KEY = 'uniform'

DOMAIN = dict(clip='coco', carzero='nih')

RESULT_GROUP = 'section_experiments'

def result_directory(root, section):
    return Path(root) / 'results' / RESULT_GROUP / section

NIH_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

NIH_TERMS = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

TEMPLATES = [('There is {x}.', 'There is no {x}.'), ('{X} is present.', 'No {x} is present.'), ('An image with {x}.', 'An image without {x}.'), ('There is evidence of {x}.', 'There is no evidence of {x}.'), ('The image shows {x}.', 'The image shows no {x}.'), ('{X} is seen in the image.', 'No {x} is seen in the image.'), ('Signs of {x} are observed.', 'No signs of {x} are observed.'), ('The image shows evidence of {x}.', 'The image shows no evidence of {x}.'), ('{X} is detected in the image.', '{X} is not detected in the image.'), ('The image is positive for {x}.', 'The image is negative for {x}.')]

WORD_POOL = 'apple neutron cactus flicker prism quartz wagon myth oxygen cobalt delta ember frost glyph helix ivory jungle karma lemon mosaic nebula orchid pulse ripple sigma thorn umbra vortex wax zenith blaze cipher dusk echo flux grove haze iris jade knot latch mirth nook opal plume reef sable tide umber vale whisk yarn zinc abyss bolt crux dew elm fang gust husk ion jolt keel lobe musk nib ore pith quill rind sap tuft urge vim wren yew zeal arc brim cleft'.split()

CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-+=/\\@#$%&*()[]{}~'

def sentence(template: str, term: str) -> str:
    x = term.lower()
    return template.replace('{X}', x[:1].upper() + x[1:]).replace('{x}', x)

def nih_prompts(polarity: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(문장, 질환 index, 템플릿 index). polarity: 'pos' 또는 'neg'."""
    k = 0 if polarity == 'pos' else 1
    texts, d_idx, l_idx = ([], [], [])
    for d, term in enumerate(NIH_TERMS):
        for l, pair in enumerate(TEMPLATES):
            texts.append(sentence(pair[k], term))
            d_idx.append(d)
            l_idx.append(l)
    return (texts, np.asarray(d_idx), np.asarray(l_idx))

def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Embedding contains NaN or infinity.')
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError('Cannot normalize a zero / near-zero vector.')
    return x / norms

def save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)

def fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def legacy_fingerprint(array: np.ndarray) -> str:
    """2_cone_analysis.py의 fingerprint (기존 perturbation cache key 호환)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def cached_array(cache: Path, recompute: bool, stem: str, metadata: dict, compute: Callable[[], np.ndarray]) -> np.ndarray:
    manifest = json.dumps(metadata, sort_keys=True, default=str)
    token = hashlib.sha256(manifest.encode()).hexdigest()[:20]
    path = cache / f'{stem}_{token}.npz'
    if path.exists() and (not recompute):
        with np.load(path, allow_pickle=False) as z:
            if str(z['manifest'].item()) != manifest:
                raise ValueError(f'Cache metadata mismatch: {path}')
            raw = z['raw'].copy()
    else:
        raw = np.asarray(compute(), dtype=np.float32)
        save_npz(path, raw=raw, manifest=np.array(manifest))
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError(f'Invalid encoder output / cache: {path}')
    normalize(raw)
    return raw

class ClipEncoder:
    name = 'clip'

    def __init__(self, cfg: Config):
        import torch
        import open_clip
        self.torch, self.cfg = (torch, cfg)
        self.device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(cfg.clip_model, pretrained=cfg.clip_pretrained, cache_dir=str(cfg.root / 'ckpt'))
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model)
        norm = next((t for t in reversed(self.preprocess.transforms) if hasattr(t, 'mean') and hasattr(t, 'std')), None)
        if norm is None:
            raise ValueError('Cannot find the CLIP Normalize transform.')
        self.mean = np.asarray(norm.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(norm.std, dtype=np.float32).reshape(1, 1, 3)
        size = self.model.visual.image_size
        self.hw = (size, size) if isinstance(size, int) else tuple(size)
        patch = getattr(self.model.visual, 'patch_size', None)
        self.patch = int(patch[0] if isinstance(patch, (tuple, list)) else patch) if patch else find_patch_size(self.model.visual)
        self.signature = dict(model=cfg.clip_model, pretrained=cfg.clip_pretrained, torch=importlib.metadata.version('torch'), open_clip=importlib.metadata.version('open_clip_torch'), size=list(self.hw), mean=self.mean.ravel().tolist(), std=self.std.ravel().tolist())

    def pixels(self, path) -> np.ndarray:
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert('RGB'))
        return np.clip(tensor.numpy().transpose(1, 2, 0) * self.std + self.mean, 0, 1).astype(np.float32)

    def encode_pixels(self, pixels: Iterable[np.ndarray]) -> np.ndarray:
        parts, batch = ([], [])

        def flush():
            if batch:
                x = (np.stack(batch) - self.mean) / self.std
                x = self.torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)
                with self.torch.inference_mode():
                    parts.append(self.model.encode_image(x, normalize=False).float().cpu().numpy())
                batch.clear()
        for image in pixels:
            if image.shape != (*self.hw, 3):
                raise ValueError(f'Unexpected pixel shape: {image.shape}')
            batch.append(np.asarray(image, dtype=np.float32))
            if len(batch) == self.cfg.batch_size:
                flush()
        flush()
        return np.concatenate(parts)

    def encode_paths(self, paths: list) -> np.ndarray:
        return self.encode_pixels((self.pixels(p) for p in paths))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), self.cfg.batch_size * 4):
            t = self.tokenizer(texts[i:i + self.cfg.batch_size * 4]).to(self.device)
            with self.torch.inference_mode():
                parts.append(self.model.encode_text(t, normalize=False).float().cpu().numpy())
        return np.concatenate(parts)

    def text_truncation(self, texts: list[str]) -> float:
        """시작 / 끝 토큰을 포함한 길이가 77을 넘는 비율."""
        encode = getattr(self.tokenizer, 'encode', None)
        if encode is None:
            return float('nan')
        return float(np.mean([len(encode(t)) + 2 > 77 for t in texts]))

def find_patch_size(module) -> int:
    """kernel == stride인 첫 Conv2d (ViT patch embedding)에서 패치 크기를 읽는다."""
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == m.stride and (m.kernel_size[0] > 1):
            return int(m.kernel_size[0])
    raise ValueError('Cannot infer the patch size of the image encoder.')

def load_carzero_model(cfg: Config, device: str):
    """1_extract_embeddings.py의 load_zeroshot_model과 같은 절차."""
    import torch
    repo = next((p for p in [cfg.carzero_project, cfg.carzero_project / 'Bi-MCQ'] if (p / 'CARZero/CARZero.py').is_file()), None)
    if repo is None:
        raise FileNotFoundError('CARZero source not found under carzero_project.')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from CARZero import builder
    prev = Path.cwd()
    try:
        os.chdir(cfg.carzero_weight_root)
        ckpt = torch.load(str(cfg.carzero_weight_root / cfg.carzero_ckpt), map_location='cpu', weights_only=False)
        model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(ckpt['hyper_parameters'])
        expected = model.state_dict()
        fixed = {}
        for k, v in ckpt['state_dict'].items():
            new_key = k
            for prefix in ['CARZero_model.', 'CARZero.']:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            if new_key in expected:
                fixed[new_key] = v
        missing = set(expected) - set(fixed)
        if missing:
            groups = pd.Series(['.'.join(k.split('.')[:2]) for k in missing]).value_counts()
            print(f'  [WARN] CARZero: {len(missing)} keys missing (strict=False): ' + ', '.join((f'{g} ({n})' for g, n in groups.items())))
        model.load_state_dict(fixed, strict=not missing)
    finally:
        os.chdir(prev)
    return model.to(device).eval()

class CarzeroEncoder:
    """CARZero 전역 임베딩. 모든 이미지 입력은 PNG 경로 → model.process_img를 거친다."""
    name = 'carzero'
    loader: Callable = staticmethod(load_carzero_model)

    def __init__(self, cfg: Config):
        import torch
        self.torch, self.cfg = (torch, cfg)
        self.device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = type(self).loader(cfg, self.device)
        self.tmp = Path(tempfile.mkdtemp(prefix='carzero_png_', dir=cfg.root / 'cache'))
        probe = self.tmp / 'probe.png'
        Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(probe)
        with torch.inference_mode():
            x = self.model.process_img([str(probe)], self.device)
        self.hw = tuple((int(v) for v in x.shape[-2:]))
        self.channels = int(x.shape[1])
        self.patch = find_patch_size(self.model)
        ckpt = cfg.carzero_weight_root / cfg.carzero_ckpt
        info = ckpt.stat() if ckpt.exists() else None
        self.signature = dict(model='CARZero', ckpt=str(ckpt), size_bytes=info.st_size if info else None, mtime_ns=info.st_mtime_ns if info else None, input_hw=list(self.hw), patch=self.patch, torch=importlib.metadata.version('torch'))
        print(f'  CARZero input {self.hw}, channels {self.channels}, patch {self.patch}')

    def pixels(self, path) -> np.ndarray:
        """원본을 RGB로 읽어 입력 해상도로 resize한 [0,1] 픽셀 (손상 실험용)."""
        with Image.open(path) as image:
            img = image.convert('RGB').resize(self.hw[::-1], Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255

    def _encode_batch(self, paths: list) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.model.process_img([str(p) for p in paths], self.device)
            _, g = self.model.image_encoder_forward(x)
        return g.float().cpu().numpy()

    def encode_paths(self, paths: list) -> np.ndarray:
        return np.concatenate([self._encode_batch(paths[i:i + self.cfg.batch_size]) for i in range(0, len(paths), self.cfg.batch_size)])

    def encode_pixels(self, pixels: Iterable[np.ndarray]) -> np.ndarray:
        parts, batch = ([], [])

        def flush():
            if batch:
                paths = []
                for k, image in enumerate(batch):
                    p = self.tmp / f'x{k:04d}.png'
                    arr = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(arr if arr.ndim == 2 or arr.shape[2] == 3 else arr[..., 0]).save(p)
                    paths.append(p)
                parts.append(self._encode_batch(paths))
                batch.clear()
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
                _, g, _ = self.model.text_encoder_forward(p['caption_ids'], p['attention_mask'], p['token_type_ids'])
            parts.append(g.float().cpu().numpy())
        return np.concatenate(parts)

    def text_truncation(self, texts: list[str]) -> float:
        """attention mask가 최대 길이를 모두 채운 비율 (잘림 가능성)."""
        with self.torch.inference_mode():
            p = self.model.process_text(texts, self.device)
        mask = p['attention_mask'].cpu().numpy()
        return float(np.mean(mask.sum(1) >= mask.shape[1]))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

class Datasets:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._nih = None

    def coco_path(self, image_id: int) -> Path:
        return self.coco_dir / self.coco_images[int(image_id)]['file_name']

    @property
    def nih(self) -> dict:
        if self._nih is None:
            root = self.cfg.nih_root
            source = pd.read_csv(root / 'Data_Entry_2017.csv', usecols=['Image Index', 'Finding Labels'])
            test = pd.read_csv(root / 'NIH_test_data.csv', usecols=['Image Index'])
            source['Image Index'] = source['Image Index'].astype(str).str.strip()
            names = test['Image Index'].astype(str).str.strip().tolist()
            source = source.set_index('Image Index')
            findings = source.loc[names, 'Finding Labels'].str.replace('_', ' ', regex=False).str.split('|').map(lambda vs: {v.strip() for v in vs})
            labels = np.column_stack([findings.map(lambda vs: int(c in vs)).to_numpy() for c in NIH_CLASSES])
            image_map = {p.name: p for p in root.glob('images*/*/*.png')}
            missing = [n for n in names if n not in image_map]
            if missing:
                raise FileNotFoundError(f'NIH images missing: {missing[:10]}')
            self._nih = dict(names=names, paths=[image_map[n] for n in names], labels=labels, findings=source.loc[names, 'Finding Labels'].tolist())
            print(f'  NIH test: {len(names):,} images, {len(NIH_CLASSES)} classes')
        return self._nih

class Context:

    def __init__(self, cfg: Config, encoders: dict | None=None, out_name: str='analysis'):
        self.cfg = cfg
        self.out = result_directory(cfg.root, out_name)
        self.cache = cfg.root / 'cache' / 'oblique_cone'
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.legacy_cache = cfg.root / 'cache' / 'conformity_imageness'
        self._encoders = dict(encoders or {})
        self.data = Datasets(cfg)
        self.summary: dict[str, Any] = {}
        self._images, self._splits, self.noise, self.text_noise = ({}, {}, {}, {})
        self.axes, self.text_axes, self.perturb = ({}, {}, {})
        if cfg.n_reference < 2 or cfg.n_noise < 2 or cfg.bootstrap_repeats < 2 or (cfg.null_directions < 1):
            raise ValueError('Sizes are too small.')
        if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
            raise ValueError('Seeds must be distinct.')

    def encoder(self, name: str):
        if name not in self._encoders:
            self._encoders[name] = ClipEncoder(self.cfg) if name == 'clip' else CarzeroEncoder(self.cfg)
        return self._encoders[name]

    def cached(self, stem: str, metadata: dict, compute) -> np.ndarray:
        cache = self.legacy_cache if metadata.get('version') == CACHE_VERSION else self.cache
        cache.mkdir(parents=True, exist_ok=True)
        return cached_array(cache, self.cfg.recompute, stem, metadata, compute)

    def images(self, enc: str, dataset: str) -> dict:
        """dict(Z, ids, paths[, labels]). 자기 도메인은 전체, 교차 칸은 n_ood개 고정 표본."""
        key = (enc, dataset)
        if key in self._images:
            return self._images[key]
        cfg = self.cfg
        if dataset == 'coco':
            ids_all = np.array(sorted(self.data.coco_images), dtype=int)
            paths_all = [self.data.coco_path(i) for i in ids_all]
            labels = None
        else:
            nih = self.data.nih
            ids_all = np.arange(len(nih['names']))
            paths_all = nih['paths']
            labels = nih['labels']
        if DOMAIN[enc] != dataset:
            pick = np.sort(np.random.default_rng(7001).choice(len(ids_all), min(cfg.n_ood, len(ids_all)), replace=False))
        else:
            pick = np.arange(len(ids_all))
        ids = ids_all[pick]
        paths = [paths_all[i] for i in pick]
        raw = None
        if enc == 'clip' and dataset == 'coco':
            eda = cfg.root / 'cache' / f'{cfg.clip_model}_{cfg.clip_pretrained}.npz'
            if eda.exists():
                with np.load(eda, allow_pickle=False) as z:
                    eda_raw, eda_ids = (z['img'].copy(), z['image_ids'].astype(int))
                if np.array_equal(eda_ids, ids):
                    raw = eda_raw.astype(np.float32)
                    self.clip_clean_hash = legacy_fingerprint(eda_raw) + legacy_fingerprint(eda_ids)
        if enc == 'carzero' and dataset == 'nih':
            emb = cfg.carzero_embedding_dir / 'zs_img_emb_g.pt'
            names_file = cfg.carzero_embedding_dir / 'image_names.json'
            if emb.exists() and names_file.exists() and (json.loads(names_file.read_text()) == self.data.nih['names']):
                import torch
                raw = torch.load(emb, map_location='cpu').float().numpy()
                self.check_carzero_against_extraction(paths, raw)
        if raw is None:
            encoder = self.encoder(enc)
            meta = dict(kind='images', dataset=dataset, encoder=encoder.signature, n=len(paths), ids_hash=fingerprint(np.asarray(ids)))
            raw = self.cached(f'img_{enc}_{dataset}', meta, lambda: encoder.encode_paths(paths))
        if len(raw) != len(ids):
            raise ValueError(f'Image embedding count mismatch for {key}.')
        if enc == 'clip' and dataset == 'coco' and (not hasattr(self, 'clip_clean_hash')):
            self.clip_clean_hash = legacy_fingerprint(raw) + legacy_fingerprint(ids)
        out = dict(Z=normalize(raw), ids=ids, paths=paths, pick=pick, labels=None if labels is None else labels[pick])
        self._images[key] = out
        return out

    def check_carzero_against_extraction(self, paths: list, raw: np.ndarray, k: int=32) -> None:
        """지금 로드한 CARZero가 1_extract_embeddings.py가 추출한 임베딩을 재현하는지 확인한다.
        이미지: 같은 경로 k장을 다시 인코딩. 텍스트: 원래 프롬프트 "There is {TERM}"."""
        encoder = self.encoder('carzero')
        idx = np.linspace(0, len(paths) - 1, min(k, len(paths))).astype(int)
        cos_img = np.sum(normalize(encoder.encode_paths([paths[i] for i in idx])) * normalize(raw[idx]), 1)
        report = dict(image_min_cos=float(cos_img.min()), image_mean_cos=float(cos_img.mean()))
        pos = self.cfg.carzero_embedding_dir / 'zs_text_pos.pt'
        if pos.exists():
            import torch
            ref = torch.load(pos, map_location='cpu').float().numpy()
            new = encoder.encode_texts([f'There is {t}' for t in NIH_TERMS])
            cos_txt = np.sum(normalize(new) * normalize(ref), 1)
            report.update(text_min_cos=float(cos_txt.min()), text_mean_cos=float(cos_txt.mean()))
        self.summary['carzero_reproduces_extraction'] = report
        print(f'  CARZero vs 1_extract_embeddings: {report}')
        if report['image_min_cos'] < 0.999 or report.get('text_min_cos', 1.0) < 0.999:
            raise ValueError('Loaded CARZero does not reproduce the stored embeddings (cos < 0.999). Check the checkpoint / missing keys before continuing.')

    def split(self, enc: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """자기 도메인 이미지의 (reference, evaluation) 행 번호."""
        key = (enc, seed)
        if key not in self._splits:
            n = len(self.images(enc, DOMAIN[enc])['Z'])
            if n <= self.cfg.n_reference + 10:
                raise ValueError('Not enough images for a reference / evaluation split.')
            perm = np.random.default_rng(seed).permutation(n)
            self._splits[key] = (perm[:self.cfg.n_reference], perm[self.cfg.n_reference:])
        return self._splits[key]

    def domain_texts(self, enc: str) -> dict:
        """CLIP: COCO 캡션 전체.  CARZero: NIH 긍정 템플릿 140문장."""
        key = ('texts', enc)
        if key in self._images:
            return self._images[key]
        cfg = self.cfg
        if enc == 'clip':
            eda = cfg.root / 'cache' / f'{cfg.clip_model}_{cfg.clip_pretrained}.npz'
            if eda.exists():
                with np.load(eda, allow_pickle=False) as z:
                    raw, ann_ids, img_ids = (z['txt'].astype(np.float32), z['txt_ann_ids'].astype(int), z['image_ids'].astype(int))
                    rows = z['txt_img_row'].astype(int)
                texts = [self.data.coco_caption_by_ann[int(a)] for a in ann_ids]
                image_ids = img_ids[rows]
            else:
                items = sorted(self.data.coco_caption_by_ann.items())
                ann_ids = np.array([a for a, _ in items])
                texts = [t for _, t in items]
                caption_image = {}
                with (cfg.root / 'data/coco/annotations/captions_val2017.json').open(encoding='utf-8') as f:
                    for x in json.load(f)['annotations']:
                        caption_image[int(x['id'])] = int(x['image_id'])
                image_ids = np.array([caption_image[int(a)] for a in ann_ids])
                encoder = self.encoder('clip')
                meta = dict(kind='captions', encoder=encoder.signature, n=len(texts))
                raw = self.cached('txt_clip_coco', meta, lambda: encoder.encode_texts(texts))
            out = dict(T=normalize(raw), texts=texts, image_ids=image_ids)
        else:
            texts, d_idx, l_idx = nih_prompts('pos')
            encoder = self.encoder('carzero')
            meta = dict(kind='nih_pos_templates', encoder=encoder.signature, texts=texts)
            raw = self.cached('txt_carzero_nih_pos', meta, lambda: encoder.encode_texts(texts))
            out = dict(T=normalize(raw), texts=texts, disease=d_idx, template=l_idx)
        self._images[key] = out
        return out

    def noise_sources(self) -> list[tuple[str, str, float]]:
        sources = [(f'gaussian_s{s:.3f}', 'gaussian', float(s)) for s in self.cfg.noise_std_sweep]
        return sources + [(REFERENCE_KEY, 'uniform', float(1 / np.sqrt(12))), ('uniform_gray', 'uniform_gray', float(1 / np.sqrt(12)))]

    def image_noise(self, enc: str, key: str, family: str, sigma: float, seed: int) -> np.ndarray:
        cache_key = (enc, key, seed)
        if cache_key in self.noise:
            return self.noise[cache_key]
        cfg, encoder = (self.cfg, self.encoder(enc))
        h, w = encoder.hw

        def generate():
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1101]))
            for _ in range(cfg.n_noise):
                if family == 'gaussian':
                    x = 0.5 + sigma * rng.standard_normal((h, w, 3))
                elif family == 'uniform':
                    x = rng.uniform(0, 1, (h, w, 3))
                else:
                    x = np.repeat(rng.uniform(0, 1, (h, w, 1)), 3, axis=2)
                yield np.clip(x, 0, 1).astype(np.float32)
        if enc == 'clip' and family in ('gaussian', 'uniform'):
            meta = dict(version=CACHE_VERSION, kind='noise', family=family, seed=seed, n=cfg.n_noise, gaussian_std=sigma if family == 'gaussian' else LEGACY_UNIFORM_STD, encoder=encoder.signature, pixel_range=[0, 1])
            stem = f'noise_{family}_{seed}'
        else:
            meta = dict(kind='noise', family=family, sigma=sigma, seed=seed, n=cfg.n_noise, encoder=encoder.signature, pixel_range=[0, 1])
            stem = f'noise_{enc}_{family}_{seed}'
        U = normalize(self.cached(stem, meta, lambda: encoder.encode_pixels(generate())))
        self.noise[cache_key] = U
        return U

    def reference_noise(self, enc: str, seed: int) -> np.ndarray:
        return self.image_noise(enc, REFERENCE_KEY, 'uniform', float(1 / np.sqrt(12)), seed)

    def text_noise_strings(self, seed: int) -> tuple[list[str], list[str]]:
        rng = random.Random(1000 + seed)
        n = self.cfg.n_text_noise_each
        chars = [''.join((rng.choice(CHARSET) for _ in range(rng.randint(20, 75)))) for _ in range(n)]
        words = [' '.join((rng.choice(WORD_POOL) for _ in range(rng.randint(2, 15)))) for _ in range(n)]
        return (chars, words)

    def text_noise_embeddings(self, enc: str, seed: int) -> dict:
        key = (enc, seed)
        if key not in self.text_noise:
            chars, words = self.text_noise_strings(seed)
            encoder = self.encoder(enc)
            meta = dict(kind='text_noise', seed=seed, encoder=encoder.signature, n=len(chars) + len(words), char_len=[20, 75], word_len=[2, 15])
            raw = self.cached(f'txtnoise_{enc}_{seed}', meta, lambda: encoder.encode_texts(chars + words))
            U = normalize(raw)
            self.text_noise[key] = dict(U=U, char=U[:len(chars)], word=U[len(chars):], strings=chars + words)
        return self.text_noise[key]

    def pooled_b(self, enc: str, modality: str='image') -> np.ndarray:
        if modality == 'image':
            U = np.concatenate([self.reference_noise(enc, s) for s in self.cfg.seeds])
        else:
            U = np.concatenate([self.text_noise_embeddings(enc, s)['U'] for s in self.cfg.seeds])
        return normalize(U.mean(0))

from contextlib import contextmanager

from functools import wraps

from io import BytesIO

from itertools import combinations

from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import brentq

from pathlib import Path
import csv
import numpy as np

ALIGN_EPS = 1e-12


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), ALIGN_EPS)


def plane(mean, noise):
    b = unit(noise)
    e = mean - (mean @ b) * b
    if np.linalg.norm(noise) <= ALIGN_EPS or np.linalg.norm(e) <= ALIGN_EPS:
        raise ValueError('평면 정의에는 독립적인 평균 방향과 노이즈 방향이 필요합니다.')
    return np.stack((b, unit(e)), axis=1)


def residual(x, basis):
    return x - (x @ basis) @ basis.T


def fit_reference(z, labels, noise):
    """labels: [N,D]의 0/1. 평가에서는 고정된 학습 reference를 재사용한다."""
    z, labels = unit(z), np.asarray(labels)
    if labels.ndim != 2 or len(labels) != len(z) or not np.isin(labels, [0, 1]).all():
        raise ValueError('labels는 영상 행 순서와 일치하는 [N,D] 이진 배열이어야 합니다.')
    n = labels.sum(0)
    if np.any((n == 0) | (n == len(z))):
        raise ValueError('각 질환에 양성과 음성 영상이 모두 필요합니다.')
    mean = z.mean(0)
    basis = plane(mean, noise)
    difference = labels.T @ z / n[:, None] - (1-labels).T @ z / (len(z)-n)[:, None]
    return dict(mean=mean, basis=basis, disease_residual=residual(difference, basis))


def auc(y, score):
    """동점 평균 순위를 사용하는 AUROC."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    if y.all() or not y.any():
        return float('nan')
    order = np.argsort(score, kind='stable')
    _, start, count = np.unique(score[order], return_index=True, return_counts=True)
    rank = np.empty(len(score), dtype=float)
    rank[order] = np.repeat(start + (count+1)/2, count)
    n = y.sum()
    return float((rank[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def cosine(a, b):
    length = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / length) if length > ALIGN_EPS else float('nan')


def analyze(z, labels, positive, negative, disease_index, template_index,
            image_reference, text_noise, disease_names):
    """각 프롬프트 쌍당 한 행. 원래 크기의 정렬과 정규화 분리도를 함께 반환한다.
    image_reference: 기존 Exp08 재현은 전체 영상, 일반화 평가는 학습 영상에서 추정.
    평가 라벨의 질환 방향은 기술 통계 계산에만 사용한다.
    """
    z, tp, tn = unit(z), unit(positive), unit(negative)
    labels = np.asarray(labels)
    if tp.shape != tn.shape or z.shape[1] != tp.shape[1]:
        raise ValueError('영상과 긍정·부정 임베딩의 차원을 맞춰주세요.')
    if len(disease_index) != len(tp) or len(template_index) != len(tp):
        raise ValueError('프롬프트별 질환·문형 인덱스가 필요합니다.')
    if labels.shape != (len(z), len(disease_names)) or not np.isin(labels, [0, 1]).all():
        raise ValueError('영상 순서에 맞는 질환별 이진 라벨이 필요합니다.')
    qi = image_reference['basis']
    qt = plane(np.concatenate((tp, tn)).mean(0), text_noise)
    dt = tp-tn
    wt, wi = residual(dt, qt), residual(dt, qi)
    np.testing.assert_allclose(wt, residual(tp, qt)-residual(tn, qt), atol=1e-10)
    zi, delta = residual(z, qi), z-image_reference['mean']
    rows = []
    for j, d in enumerate(disease_index):
        y = labels[:, d].astype(bool)
        if y.all() or not y.any():
            raise ValueError(f'{disease_names[d]} 평가에 양성·음성 영상이 모두 필요합니다.')
        g = z @ dt[j]
        m = float(image_reference['mean'] @ dt[j])
        terms = (delta @ qi) * (dt[j] @ qi)
        u = residual(delta, qi) @ wi[j]
        centered = g-m
        np.testing.assert_allclose(g, m+terms.sum(1)+u, atol=1e-9)
        a = zi[y].mean(0)-zi[~y].mean(0)
        gap = lambda s: float(s[y].mean()-s[~y].mean())
        raw_gap = float(a @ wi[j])
        np.testing.assert_allclose(raw_gap, gap(u), atol=1e-9)
        normalized = unit(zi) @ unit(wi[j])
        norm2 = float(dt[j] @ dt[j])
        row = dict(disease=disease_names[d], template=int(template_index[j]),
                   n_positive=int(y.sum()), n_negative=int((~y).sum()), M=m,
                   auc=auc(y, g), auc_centered=auc(y, centered), auc_residual=auc(y, u),
                   gap_total=gap(g), gap_plane=gap(terms.sum(1)), gap_residual=raw_gap,
                   alignment_cosine=cosine(a, wi[j]), disease_residual_norm=float(np.linalg.norm(a)),
                   text_image_residual_norm=float(np.linalg.norm(wi[j])),
                   train_alignment_cosine=cosine(image_reference['disease_residual'][d], wi[j]),
                   dmu_rm_be=gap(normalized) if np.linalg.norm(wi[j]) > ALIGN_EPS and np.all(np.linalg.norm(zi, axis=1) > ALIGN_EPS) else float('nan'),
                   text_residual_energy_fraction=float(wt[j] @ wt[j]/norm2) if norm2 > ALIGN_EPS else float('nan'),
                   delta_text_norm=float(np.sqrt(norm2)))
        for name, score in (('raw', g), ('centered', centered)):
            pred = score >= 0
            row.update({f'{name}_positive_rate': float(pred.mean()),
                        f'{name}_sensitivity': float(pred[y].mean()),
                        f'{name}_specificity': float((~pred[~y]).mean())})
        np.testing.assert_allclose(row['gap_total'], row['gap_plane']+raw_gap, atol=1e-9)
        rows.append(row)
    return rows


def residual_alignment_loss(positive, negative, disease_index, basis, disease_residual):
    """PyTorch loss. basis/질환 방향은 학습 자료에서 추정한 고정 tensor.
    positive/negative [K,F]에는 gradient 유지. 영벡터 target은 가중치 0.
    loss_total = loss_clip + weight * residual_alignment_loss(...)
    """
    import torch.nn.functional as F
    q = basis.detach()
    target = disease_residual.detach()[disease_index]
    target = target-(target @ q) @ q.T
    dt = F.normalize(positive, dim=-1)-F.normalize(negative, dim=-1)
    w = dt-(dt @ q) @ q.T
    valid = (target.norm(dim=-1) > 1e-8).to(w.dtype)
    similarity = (F.normalize(w, dim=-1)*F.normalize(target, dim=-1)).sum(-1)
    return ((1-similarity)*valid).sum()/valid.sum().clamp_min(1)


def run(ctx):
    """기존 Exp08과 동일한 캐시·전체 영상 reference로 분석한다."""
    enc = ctx.encoder('carzero')
    imgs = ctx.images('carzero', 'nih')
    pos, diseases, templates = nih_prompts('pos')
    neg, _, _ = nih_prompts('neg')
    def texts(kind, prompts):
        return ctx.cached(f'txt_carzero_nih_{kind}',
                          dict(kind=f'nih_{kind}_templates', encoder=enc.signature, texts=prompts),
                          lambda: enc.encode_texts(prompts))
    ref = fit_reference(imgs['Z'], imgs['labels'], ctx.pooled_b('carzero', 'image'))
    rows = analyze(imgs['Z'], imgs['labels'], texts('pos', pos), texts('neg', neg),
                   diseases, templates, ref, ctx.pooled_b('carzero', 'text'), NIH_CLASSES)
    path = Path(ctx.out)/'exp08_negation_residual_alignment.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    config = Config(encoders=('carzero',), make_figures=False)
    ctx = Context(config, out_name='negation_alignment')
    try:
        print('NIH 부정 프롬프트 잔차 정렬 분석 시작', flush=True)
        print(f'저장 완료: {run(ctx)}', flush=True)
    finally:
        for encoder in ctx._encoders.values():
            if hasattr(encoder, 'close'):
                encoder.close()


if __name__ == '__main__':
    main()

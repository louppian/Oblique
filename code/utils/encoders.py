from __future__ import annotations

import importlib.metadata

import os

import shutil

import sys

import tempfile

from dataclasses import dataclass

from pathlib import Path

from typing import Callable, Iterable

import numpy as np

import pandas as pd

from PIL import Image

ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir()))

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
    (texts, d_idx, l_idx) = ([], [], [])
    for (d, term) in enumerate(NIH_TERMS):
        for (l, pair) in enumerate(TEMPLATES):
            texts.append(sentence(pair[k], term))
            d_idx.append(d)
            l_idx.append(l)
    return (texts, np.asarray(d_idx), np.asarray(l_idx))

class ClipEncoder:
    name = 'clip'

    def __init__(self, cfg: Config):
        import torch
        import open_clip
        (self.torch, self.cfg) = (torch, cfg)
        self.device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        (self.model, _, self.preprocess) = open_clip.create_model_and_transforms(cfg.clip_model, pretrained=cfg.clip_pretrained, force_quick_gelu=True, cache_dir=str(cfg.root / 'ckpt'))
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
        (parts, batch) = ([], [])

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
    from CARZero.models import transformer_backbones
    original_load_weight = transformer_backbones.load_weight

    def load_pretrain(model, path):
        resolved = cfg.carzero_weight_root / Path(path).name
        weights = torch.load(str(resolved if resolved.is_file() else path), map_location='cpu', weights_only=False)
        model.load_state_dict(transformer_backbones.update_weights(model, weights['state_dict']))
        return model
    prev = Path.cwd()
    try:
        os.chdir(cfg.carzero_weight_root)
        ckpt = torch.load(str(cfg.carzero_weight_root / cfg.carzero_ckpt), map_location='cpu', weights_only=False)
        transformer_backbones.load_weight = load_pretrain
        try:
            model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(ckpt['hyper_parameters'])
        finally:
            transformer_backbones.load_weight = original_load_weight
        expected = model.state_dict()
        fixed = {}
        for (k, v) in ckpt['state_dict'].items():
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
            print(f'  [WARN] CARZero: {len(missing)} keys missing (strict=False): ' + ', '.join((f'{g} ({n})' for (g, n) in groups.items())))
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
        (self.torch, self.cfg) = (torch, cfg)
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
            (_, g) = self.model.image_encoder_forward(x)
        return g.float().cpu().numpy()

    def encode_paths(self, paths: list) -> np.ndarray:
        return np.concatenate([self._encode_batch(paths[i:i + self.cfg.batch_size]) for i in range(0, len(paths), self.cfg.batch_size)])

    def encode_pixels(self, pixels: Iterable[np.ndarray]) -> np.ndarray:
        (parts, batch) = ([], [])

        def flush():
            if batch:
                paths = []
                for (k, image) in enumerate(batch):
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
                (_, g, _) = self.model.text_encoder_forward(p['caption_ids'], p['attention_mask'], p['token_type_ids'])
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

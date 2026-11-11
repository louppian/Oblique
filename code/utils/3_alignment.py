"""섹션 3 — 모달리티 정합: 실험 함수 4개.
exp10: 전체셋의 conformity 및 Imageness–Textness 축 정합도.
exp11: 평균 이동에 따른 C/c/h 분포 정합, CLIP 손실, 갭, 조건 외 예측.
exp12: CLIP-only / conformity / Imageness 정합 학습과 에폭별 지표·상관.
exp13: CARZero–Bi-MCQ의 현재/고정 꼭지점, 기하와 질환 점수 비교.
Config를 수정하고 python 3_alignment.py로 실행한다. 네 파일은 같은 폴더.
exp13만 재실행하려면 Config.experiments를 (13,)으로 설정한다.
기본: COCO val2017 이미지 4000/1000 분할, batch128, 10 epochs, seeds0/1/2.
같은 초기 가중치/배치 순서로 lambda0.100/1.000을 비교한다.
에폭 시작 전 동일 노이즈 입력으로 b를 재측정하고, 각 배치의 r로 축 로스를 계산한다.
L_axis=1-a_I·a_T, a_M=normalize(normalize(mean z_M)-b_M).
학습 기준축/PCA는 학습 이미지와 모든 학습 캡션으로 추정한다.
검색은 모든 정답 캡션을 인정하고 동점 오답을 앞에 두는 보수적 rank를 사용한다.
exp12 실행 시 4_residual.py의 네 함수도 같은 에폭 임베딩으로 실행한다.
결과: <root>/results/section_experiments/alignment/ 및 residual/.
실험별 CSV/PNG 하나, 표시 세 자리, CLI 인자/JSON 결과/checkpoint 출력 없음.
상관은 에폭별 관측 관계와 변화량 관계를 기록한다.
Dependencies: numpy pandas scipy Pillow matplotlib torch torchvision open_clip_torch.
Bi-MCQ 비교에는 기존 CARZero 소스/가중치, omegaconf가 필요하다.
"""
from __future__ import annotations
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())
import csv
import json  # COCO 원본 annotation 읽기 전용
import math
import os
import random
import numpy as np
from PIL import Image

torch = F = open_clip = None
import importlib
an = importlib.import_module("1_analysis")
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import spearmanr

@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    experiments: tuple[int, ...] = (10, 11, 12, 13)
    #experiments: tuple[int, ...] = (13,)
    output_directory: Path | None = None
    model: str = 'ViT-B-32'
    pretrained: str = 'openai'
    device: str | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 10
    batch_size: int = 128
    encode_batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    gradient_checkpointing: bool = True
    learning_rate_reciprocal: int = 100000
    weight_decay: float = 0.200
    warmup_fraction: float = 0.100
    max_grad_norm: float = 1.000
    axis_weights: tuple[float, ...] = (0.100, 1.000)
    methods: tuple[str, ...] = ('clip_only', 'conformity', 'imageness')
    data_mode: str = 'val2017_split'
    split_seed: int = 2026
    fallback_eval_images: int = 1000
    max_train_images: int = 4000
    n_noise_images_per_seed: int = 1000
    n_text_noise_each_per_seed: int = 500
    noise_seeds: tuple[int, ...] = (0,)
    geometry_noise_seed: int = 0
    control_seed: int = 2026
    random_subspace_repeats: int = 10
    pair_shuffle_repeats: int = 20
    neighbor_k: int = 10
    query_detail_epochs: tuple[int, ...] = (0, 10)
    reference_modes: tuple[str, ...] = ('initial_fixed', 'epoch_current')
    log_every_steps: int = 20
    alpha_grid: tuple[float, ...] = tuple(round(k / 20, 3) for k in range(-20, 21))
    shift_modes: tuple[str, ...] = ('image_only', 'text_only', 'both')
    mean_shift_full: bool = True
    mean_shift_heldout: bool = True
    mean_shift_reference_images: int = 2000
    mean_shift_evaluation_images: int = 1000
    histogram_bins: tuple[int, ...] = (50, 100, 200)
    histogram_pseudocount: float = 0.500
    ridge_penalty: float = 0.100
    loss_temperature: float = 0.010
    loss_chunk: int = 256
    run_bimcq: bool = True
    carzero_project: Path = ROOT / 'third_party' / 'BI_MCQ'
    carzero_weight_root: Path = ROOT / 'ckpt'
    carzero_ckpt: str = 'best_model.ckpt'
    bimcq_ckpt: str = 'BiMCQ_CARZero_best_model.ckpt'
    bimcq_config: Path = ROOT / 'third_party' / 'BI_MCQ' / 'configs' / 'chest14_finetuning_Bi-MCQ.yaml'
    nih_root: Path = ROOT / 'data' / 'NIH'
    carzero_embedding_dir: Path = ROOT / 'cache' / 'embeddings'

CONFIG = Config()
EPS = 0.000001

WORD_POOL = ('apple neutron cactus flicker prism quartz wagon myth oxygen cobalt delta ember frost glyph helix '
             'ivory jungle karma lemon mosaic nebula orchid pulse ripple sigma thorn umbra vortex wax zenith '
             'blaze cipher dusk echo flux grove haze iris jade knot latch mirth nook opal plume reef sable tide '
             'umber vale whisk yarn zinc abyss bolt crux dew elm fang gust husk ion jolt keel lobe musk nib ore '
             'pith quill rind sap tuft urge vim wren yew zeal arc brim cleft').split()

CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-+=/\\@#$%&*()[]{}~'

def require_dependencies():
    global torch, F, open_clip
    try:
        import torch as torch_module
        import torch.nn.functional as functional
        import open_clip as open_clip_module
    except ImportError as exc:
        raise RuntimeError('torch, torchvision, open_clip_torch, numpy, Pillow를 설치하세요.') from exc
    torch, F, open_clip = torch_module, functional, open_clip_module

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_coco(annotation_path, image_dir):
    annotation_path, image_dir = Path(annotation_path), Path(image_dir)
    if not annotation_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f'COCO 데이터 경로를 확인하세요: {annotation_path}, {image_dir}')
    ann = json.loads(annotation_path.read_text(encoding='utf-8'))
    captions = {}
    for caption in sorted(ann['annotations'], key=lambda x: x['id']):
        text = caption['caption'].strip()
        if text:
            captions.setdefault(int(caption['image_id']), []).append((int(caption['id']), text))
    records = []
    for image in sorted(ann['images'], key=lambda x: x['id']):
        image_id = int(image['id'])
        path = image_dir / image['file_name']
        if image_id not in captions:
            raise ValueError(f'캡션이 없는 이미지: {image_id}')
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(dict(id=image_id, path=str(path), captions=captions[image_id]))
    if len({x['id'] for x in records}) != len(records):
        raise ValueError('중복 COCO 이미지 ID가 있습니다.')
    return records

def load_splits(cfg):
    coco = cfg.root / 'data/coco'
    val = load_coco(coco / 'annotations/captions_val2017.json', coco / 'val2017')
    has_train = (coco / 'annotations/captions_train2017.json').is_file() and (coco / 'train2017').is_dir()
    if cfg.data_mode not in ('auto', 'train2017', 'val2017_split'):
        raise ValueError('data_mode는 auto / train2017 / val2017_split 중 하나입니다.')
    if cfg.data_mode == 'train2017' or (cfg.data_mode == 'auto' and has_train):
        train = load_coco(coco / 'annotations/captions_train2017.json', coco / 'train2017')
        evaluation = val
        protocol = 'train2017 training; all val2017 images/captions for evaluation'
    else:
        if not 1 <= cfg.fallback_eval_images <= len(val) - 2:
            raise ValueError('평가 이미지 수와 val2017 이미지 수를 확인하세요.')
        permutation = np.random.default_rng(cfg.split_seed).permutation(len(val))
        held_out = set(permutation[:cfg.fallback_eval_images].tolist())
        train = [x for i, x in enumerate(val) if i not in held_out]
        evaluation = [x for i, x in enumerate(val) if i in held_out]
        protocol = 'image-disjoint train/evaluation split within val2017'
    if cfg.max_train_images:
        if cfg.max_train_images < 2:
            raise ValueError('max_train_images는 0 또는 2 이상이어야 합니다.')
        chosen = np.random.default_rng(cfg.split_seed + 1).permutation(len(train))[:cfg.max_train_images]
        train = [train[i] for i in sorted(chosen)]
    if len(train) < 2 or not evaluation:
        raise ValueError('학습 이미지 2장 이상과 평가 이미지가 필요합니다.')
    if {r['id'] for r in train} & {r['id'] for r in evaluation}:
        raise ValueError('학습과 평가 이미지가 겹칩니다.')
    return train, evaluation, val, protocol

class ImageDataset:
    def __init__(self, records, preprocess):
        self.records, self.preprocess = records, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['path']) as image:
            return self.preprocess(image.convert('RGB'))

class PairDataset(ImageDataset):
    def __init__(self, records, preprocess, tokens):
        super().__init__(records, preprocess)
        self.tokens = tokens

    def __getitem__(self, index):
        return super().__getitem__(index), self.tokens[index]

def make_loader(dataset, cfg, device, batch_size, shuffle=False, seed=0, training=False):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=device.type == 'cuda',
        generator=generator, drop_last=training and len(dataset) >= batch_size,
        worker_init_fn=seed_worker,
    )

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

def autocast_context(cfg, device):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if cfg.amp and device.type == 'cuda' else nullcontext()

def unit_features(features):
    features = features.float()
    norms = features.norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or (norms == 0).any():
        raise ValueError('유한하지 않은 임베딩 또는 영벡터가 있습니다.')
    return features / norms

def encode_images(model, records, preprocess, cfg, device):
    model.eval()
    parts = []
    loader = make_loader(ImageDataset(records, preprocess), cfg, device, cfg.encode_batch_size)
    with torch.no_grad():
        for images in loader:
            with autocast_context(cfg, device):
                features = model.encode_image(images.to(device), normalize=False)
            parts.append(unit_features(features).cpu())
    return torch.cat(parts)

def encode_texts(model, texts, tokenizer, cfg, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(texts), cfg.encode_batch_size):
            tokens = tokenizer(texts[start:start + cfg.encode_batch_size]).to(device)
            with autocast_context(cfg, device):
                features = model.encode_text(tokens, normalize=False)
            parts.append(unit_features(features).cpu())
    return torch.cat(parts)

def noise_image_batches(model, preprocess, cfg, device, batch_size):
    size = model.visual.image_size
    height, width = (size, size) if isinstance(size, int) else tuple(size)
    norm = next((transform for transform in reversed(preprocess.transforms)
                 if hasattr(transform, 'mean') and hasattr(transform, 'std')), None)
    if norm is None:
        raise ValueError('CLIP 전처리에서 Normalize를 찾을 수 없습니다.')
    mean = torch.tensor(norm.mean, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=device).reshape(1, 3, 1, 1)
    for seed in cfg.noise_seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1101]))
        for start in range(0, cfg.n_noise_images_per_seed, batch_size):
            count = min(batch_size, cfg.n_noise_images_per_seed-start)
            pixels = rng.uniform(0, 1, (count, height, width, 3)).astype(np.float32)
            pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device)
            yield (pixels-mean)/std

def noise_text_strings(cfg):
    texts = []
    for seed in cfg.noise_seeds:
        rng = random.Random(1000+seed)
        chars = [''.join(rng.choice(CHARSET) for _ in range(rng.randint(20, 75)))
                 for _ in range(cfg.n_text_noise_each_per_seed)]
        words = [' '.join(rng.choice(WORD_POOL) for _ in range(rng.randint(2, 15)))
                 for _ in range(cfg.n_text_noise_each_per_seed)]
        texts.extend(chars+words)
    return texts

def encode_noise_embeddings(model, preprocess, tokenizer, cfg, device):
    model.eval()
    image_noise = []
    with torch.no_grad():
        for pixels in noise_image_batches(model, preprocess, cfg, device, cfg.encode_batch_size):
            with autocast_context(cfg, device):
                features = model.encode_image(pixels, normalize=False)
            image_noise.append(unit_features(features).cpu())
    text_noise = encode_texts(model, noise_text_strings(cfg), tokenizer, cfg, device)
    return torch.cat(image_noise), text_noise

def clip_contrastive_loss(image_features, text_features, logit_scale):
    z, t = unit_features(image_features), unit_features(text_features)
    if z.shape != t.shape or z.ndim != 2 or len(z) < 2:
        raise ValueError('이미지/텍스트의 [N,D] 형태가 같고 N이 2 이상이어야 합니다.')
    logits = logit_scale.float() * (z @ t.T)
    labels = torch.arange(len(z), device=z.device)
    clip_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return clip_loss

def axis_loss_from_means(mu_image, mu_text, mu_noise_image, mu_noise_text):
    """네 평균벡터를 통한 gradient를 유지한 L_axis = 1 - a_I dot a_T."""
    def direction(mu, mu_noise):
        r = unit_features(mu[None])[0]
        b = unit_features(mu_noise[None])[0]
        chord = r-b
        if chord.norm().item() <= 0.000001:
            raise ValueError('b와 r가 너무 가까워 축 방향을 추정할 수 없습니다.')
        return unit_features(chord[None])[0]
    ai = direction(mu_image, mu_noise_image)
    at = direction(mu_text, mu_noise_text)
    return 1-(ai*at).sum()

def estimate_epoch_vertices(model, preprocess, tokenizer, cfg, device):
    """에폭 시작 모델의 꼭지점을 계산한 뒤, 에폭 내 고정 anchor로 반환한다."""
    was_training = model.training
    try:
        noise_cfg = replace(cfg, amp=False, noise_seeds=(cfg.geometry_noise_seed,))
        ni, nt = encode_noise_embeddings(model, preprocess, tokenizer, noise_cfg, device)
        vertices = {}
        with torch.no_grad():
            for name, noise in (('image', ni), ('text', nt)):
                mu = noise.float().mean(dim=0).to(device)
                vertices[name] = unit_features(mu[None])[0].detach()
        return vertices
    finally:
        model.train(was_training)

def batch_axis_loss(image_features, text_features, vertices):
    """고정 b와 현재 배치의 정규화 평균 r로 두 축의 방향을 정합한다."""
    zi, ti = unit_features(image_features), unit_features(text_features)
    return axis_loss_from_means(zi.mean(dim=0), ti.mean(dim=0),
                               vertices['image'], vertices['text'])

def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith('bias') else decay).append(parameter)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': cfg.weight_decay},
         {'params': no_decay, 'weight_decay': 0.000}],
        lr=1 / cfg.learning_rate_reciprocal, betas=(0.900, 0.980),
    )

def lr_at_step(step, total, warmup_fraction, base_lr):
    warmup = int(total * warmup_fraction)
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return base_lr * (1 + math.cos(math.pi * min(progress, 1))) / 2

def optimizer_step(model, optimizer, scaler, cfg, amp_enabled):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    if not torch.isfinite(norm) and not amp_enabled:
        raise FloatingPointError('비유한 gradient가 발생했습니다.')
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        model.logit_scale.clamp_(max=math.log(100))
    return int(scaler.get_scale() < before)

@contextmanager
def preserve_diagnostic_state(model):
    """진단 forward/backward의 RNG와 buffers를 복원한다."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    buffers = [(value, value.detach().clone()) for value in model.buffers()]
    training = model.training
    def reset():
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)
        model.train(training)
    try:
        yield reset
    finally:
        reset()
        model.zero_grad(set_to_none=True)


# ------------------------- 공통 수치 분석 -------------------------
def unit(x, allow_zero=False):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('임베딩에 비유한 값이 있습니다.')
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not allow_zero and np.any(norm <= EPS):
        raise ValueError('평균/임베딩 방향이 정의되지 않습니다.')
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > EPS)


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def basis(columns):
    x = np.column_stack(columns)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if not len(s) or s[0] <= EPS:
        return np.empty((x.shape[0], 0))
    return u[:, s > EPS * s[0]]


def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    return centered.T @ centered / len(x)


def fit_frame(z, t, ni, nt):
    """실제 기준표본은 train. 평가표본은 축/PCA 추정에 사용하지 않는다."""
    z, t, ni, nt = (unit(x) for x in (z, t, ni, nt))
    frame = {}
    for modality, clean, noise in [('image', z, ni), ('text', t, nt)]:
        mu, nu = clean.mean(0), noise.mean(0)
        r, b = unit(mu), unit(nu)
        ell = float(np.linalg.norm(r - b))
        if ell <= EPS:
            raise ValueError(f'{modality}: 노이즈점과 평균 방향이 같아 축이 정의되지 않습니다.')
        e = unit(r - float(r @ b) * b, allow_zero=True)
        frame[modality] = dict(mu=mu, r=r, b=b, a=(r-b)/ell, e=e,
                               R=float(np.linalg.norm(mu)), Rb=float(np.linalg.norm(nu)), ell=ell)
    i, tframe = frame['image'], frame['text']
    frame['spaces'] = {
        'conformity': basis([i['r'], tframe['r']]),
        'imageness': basis([i['a'], tframe['a']]),
        'image_cone': basis([i['b'], i['r']]),
        'text_cone': basis([tframe['b'], tframe['r']]),
        'joint_cone': basis([i['b'], i['r'], tframe['b'], tframe['r']]),
    }
    # 모달리티를 같은 가중치로 두고 각 모달리티 평균을 제거한 PCA.
    cov = (covariance(z) + covariance(t)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    frame['pca'] = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    return frame


def angle(x, y):
    return float(np.degrees(np.arccos(np.clip(unit(x) @ unit(y), -1, 1))))


def geometry(z, t, frame):
    mu_i, mu_t = z.mean(0), t.mean(0)
    ri, rt = unit(mu_i), unit(mu_t)
    bi, bt = frame['image']['b'], frame['text']['b']
    ai, at = unit(ri-bi), unit(rt-bt)
    return dict(conformity_cosine=float(ri @ rt), imageness_textness_cosine=float(ai @ at),
                conformity_angle_deg=angle(ri, rt), imageness_textness_angle_deg=angle(ai, at),
                centroid_gap=float(np.linalg.norm(mu_i-mu_t)), R_image=float(np.linalg.norm(mu_i)),
                R_text=float(np.linalg.norm(mu_t)), Rb_image=frame['image']['Rb'],
                Rb_text=frame['text']['Rb'], vertex_angle_deg=angle(bi, bt),
                noise_axis_length_image=float(np.linalg.norm(ri-bi)),
                noise_axis_length_text=float(np.linalg.norm(rt-bt)))


def query_stats(scores, mapping):
    """연결된 모든 캡션을 정답으로 인정한다. 오답과의 동점은 보수적으로 처리한다."""
    scores = np.asarray(scores, dtype=np.float64)
    mapping = np.asarray(mapping, dtype=np.int64)
    n, m = scores.shape
    if n < 2 or mapping.shape != (m,) or np.any(mapping < 0) or np.any(mapping >= n):
        raise ValueError('검색 후보 수와 대응 행을 확인하세요.')
    positive = np.arange(n)[:, None] == mapping[None, :]
    if not positive.any(1).all():
        raise ValueError('정답 캡션이 없는 이미지가 있습니다.')
    result = {}
    for direction, s, mask in [('i2t', scores, positive), ('t2i', scores.T, positive.T)]:
        p = np.where(mask, s, -np.inf).argmax(1)
        q = np.where(~mask, s, -np.inf).argmax(1)
        rows = np.arange(len(s))
        best, worst = s[rows, p], s[rows, q]
        ranks = 1 + ((s >= best[:, None]) & ~mask).sum(1)
        auc = (((s < best[:, None]) & ~mask).sum(1)
               + 0.5 * ((s == best[:, None]) & ~mask).sum(1)) / (~mask).sum(1)
        result[direction] = dict(positive=p, negative=q, ranks=ranks, margin=best-worst, auc=auc)
    return result


def retrieval_summary(q):
    row = {}
    for direction, values in q.items():
        for k in (1, 5, 10):
            row[f'{direction}_R{k}_percent'] = float(100*np.mean(values['ranks'] <= k))
        row[f'{direction}_median_rank'] = float(np.median(values['ranks']))
        row[f'{direction}_margin_mean'] = float(values['margin'].mean())
        row[f'{direction}_AUROC_percent'] = float(100*values['auc'].mean())
    return row


def weighted_pearson(x, y, weights=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[good], y[good], w[good]
    if len(x) < 2:
        return float('nan')
    w = w / w.sum()
    x, y = x - w @ x, y - w @ y
    denominator = np.sqrt((w @ (x*x)) * (w @ (y*y)))
    return float(w @ (x*y) / denominator) if denominator > EPS*EPS else float('nan')


def residual(x, q):
    return x - (x @ q) @ q.T


def nearest_neighbors(x, k):
    x = unit(x, allow_zero=True)
    if np.any(np.linalg.norm(x, axis=1) <= EPS):
        return None
    scores = x @ x.T
    np.fill_diagonal(scores, -np.inf)
    return np.argsort(-scores, axis=1, kind='stable')[:, :min(k, len(x)-1)]


def neighbor_overlap(current, original):
    if current is None or original is None:
        return float('nan')
    return float(np.mean([len(set(a) & set(b))/len(a) for a, b in zip(current, original)]))


def text_per_image(t, mapping, n):
    result = np.zeros((n, t.shape[1]))
    np.add.at(result, mapping, t)
    return result / np.bincount(mapping, minlength=n)[:, None]


def spectral_stats(cov):
    eigenvalues = np.maximum(np.linalg.eigvalsh((cov+cov.T)/2), 0)
    singular = np.sqrt(eigenvalues)
    if singular.sum() <= EPS:
        return dict(effective_rank=0.000, participation_ratio=0.000)
    p = singular[singular > 0] / singular.sum()
    return dict(effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
                participation_ratio=float(eigenvalues.sum()**2 / np.sum(eigenvalues**2)))


COMMON_COLUMNS = ['record_type', 'run_id', 'seed', 'method', 'lambda_axis', 'epoch',
                  'reference_mode', 'reference_source', 'n_train_images', 'n_train_captions',
                  'n_evaluation_images', 'n_evaluation_captions', 'noise_seed', 'batch_size',
                  'n_optimizer_updates', 'train_loss_clip', 'train_loss_axis', 'train_loss',
                  'anchor_refreshes_per_epoch', 'model', 'pretrained','n_epochs_config',
                  'learning_rate_reciprocal','weight_decay','warmup_fraction','skipped_amp_steps',
                  'n_geometry_image_noise','n_geometry_text_noise','split_seed']
RETRIEVAL_COLUMNS = [f'{d}_{k}' for d in ('i2t', 't2i') for k in
                     ('R1_percent','R5_percent','R10_percent','median_rank','margin_mean','AUROC_percent')]
GEOMETRY_COLUMNS = ['conformity_cosine','imageness_textness_cosine','conformity_angle_deg',
                    'imageness_textness_angle_deg','centroid_gap','R_image','R_text','Rb_image','Rb_text',
                    'vertex_angle_deg','noise_axis_length_image','noise_axis_length_text']


class OutputTable:
    def __init__(self, output, name, columns):
        self.path = output / f'{name}.csv'
        self.png = output / f'{name}.png'
        self.columns = list(dict.fromkeys(COMMON_COLUMNS + columns))
        self.file = self.path.open('w', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns, extrasaction='raise')
        self.writer.writeheader()
        self.summaries = []

    def append(self, row):
        self.writer.writerow({key: (f'{float(value):.3f}' if np.isfinite(value) else '')
                              if isinstance(value, (float, np.floating)) else value
                              for key, value in row.items()})
        if row.get('record_type') != 'query':
            self.summaries.append(row.copy())

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


@dataclass
class Snapshot:
    metadata: dict
    z: np.ndarray
    t: np.ndarray
    scores: np.ndarray
    mapping: np.ndarray
    image_ids: np.ndarray
    caption_ids: np.ndarray
    frame: dict


class AnalysisContext:
    def __init__(self, cfg, output, initial):
        self.cfg, self.output, self.initial = cfg, Path(output), initial
        self.output.mkdir(parents=True, exist_ok=True)
        self.base_query = query_stats(initial.scores, initial.mapping)
        self.tables = {}
        self.controls = {}
        self.initial_neighbors = {}
        self.clip_queries = {}
        self.clip_margins = {}

    def table(self, name, columns):
        if name not in self.tables:
            self.tables[name] = OutputTable(self.output, name, columns)
        return self.tables[name]

    def frames(self, snapshot):
        for mode in self.cfg.reference_modes:
            yield mode, self.initial.frame if mode == 'initial_fixed' else snapshot.frame

    def meta(self, snapshot, mode='epoch_current'):
        return dict(**snapshot.metadata, reference_mode=mode, reference_source='training_images_and_captions')

    def random_basis(self, dimension, rank, trial):
        key = dimension, rank, trial
        if key not in self.controls:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.control_seed, rank, trial]))
            self.controls[key] = np.linalg.qr(rng.normal(size=(dimension, rank)), mode='reduced')[0]
        return self.controls[key]

    def close(self):
        for table in self.tables.values():
            table.close()


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'figure.dpi':140,'savefig.dpi':180})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(alpha=0.200)
    return fig, axes


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    import tempfile
    path=Path(path)
    results=next((p for p in path.parents if p.name=='results'),None)
    staging=(results.parent/'cache'/'oblique_cone') if results is not None else path.parent.parent/'plot_cache'
    staging.mkdir(parents=True,exist_ok=True)
    # 세 자리 표시보다 작은 변화를 과도하게 확대하지 않는다.
    for ax in fig.axes:
        low,high=ax.get_ylim()
        if high-low<0.010:
            midpoint=(low+high)/2;ax.set_ylim(midpoint-0.005,midpoint+0.005)
        if 'R@1' in ax.get_title() or 'Recall' in ax.get_ylabel():
            low,high=ax.get_ylim();low=max(0.,low);high=min(100.,high)
            if high<=low:low,high=0.,100.
            ax.set_ylim(low,high)
    fd,name=tempfile.mkstemp(prefix=path.stem+'_',suffix='.png.tmp',dir=staging)
    os.close(fd);temporary=Path(name)
    try:
        fig.savefig(temporary,format='png',dpi=180)
        temporary.replace(path)
    finally:
        plt.close(fig);temporary.unlink(missing_ok=True)


def label(row):
    return 'CLIP' if row['method'] == 'clip_only' else f"{row['method']} ({row['lambda_axis']:.3f})"


def plot_curves(ax, rows, key, title, predicate=lambda r: True):
    from matplotlib.ticker import MaxNLocator
    groups = {}
    for row in rows:
        if predicate(row) and np.isfinite(row.get(key, np.nan)):
            groups.setdefault(label(row), {}).setdefault(row['epoch'], []).append(row[key])
    for name, epochs in groups.items():
        x = sorted(epochs)
        mean = np.array([np.mean(epochs[e]) for e in x])
        sd = np.array([np.std(epochs[e], ddof=1) if len(epochs[e]) > 1 else 0 for e in x])
        line, = ax.plot(x, mean, marker='.', label=name)
        ax.fill_between(x, mean-sd, mean+sd, color=line.get_color(), alpha=0.120)
    ax.set(title=title, xlabel='Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if groups:
        ax.legend(fontsize=7)
    else:
        ax.text(0.500, 0.500, 'No eligible queries', transform=ax.transAxes,
                ha='center', va='center', fontsize=9)
















# ------------------------- 공유 학습/실행 -------------------------


def batch_conformity_loss(image_features, text_features):
    zi, ti = unit_features(image_features), unit_features(text_features)
    ri = unit_features(zi.mean(0)[None])[0]
    rt = unit_features(ti.mean(0)[None])[0]
    return 1-(ri*rt).sum()


def encode_snapshot(model,train,evaluation,preprocess,tokenizer,cfg,device,metadata):
    eval_texts=[caption[1] for record in evaluation for caption in record['captions']]
    train_texts=[caption[1] for record in train for caption in record['captions']]
    mapping=np.array([i for i,record in enumerate(evaluation) for _ in record['captions']],dtype=np.int64)
    caption_ids=np.array([caption[0] for record in evaluation for caption in record['captions']],dtype=np.int64)
    image_ids=np.array([record['id'] for record in evaluation],dtype=np.int64)
    if len(set(caption_ids)) != len(caption_ids):
        raise ValueError('중복 평가 캡션 ID가 있습니다.')
    with preserve_diagnostic_state(model):
        eval_cfg=replace(cfg,amp=False,noise_seeds=(cfg.geometry_noise_seed,))
        zi=unit(array(encode_images(model,train,preprocess,eval_cfg,device)))
        ti=unit(array(encode_texts(model,train_texts,tokenizer,eval_cfg,device)))
        z=unit(array(encode_images(model,evaluation,preprocess,eval_cfg,device)))
        t=unit(array(encode_texts(model,eval_texts,tokenizer,eval_cfg,device)))
        ni,nt=encode_noise_embeddings(model,preprocess,tokenizer,eval_cfg,device)
        frame=fit_frame(zi,ti,array(ni),array(nt))
    return Snapshot(metadata=metadata,z=z,t=t,scores=z @ t.T,mapping=mapping,
                    image_ids=image_ids,caption_ids=caption_ids,frame=frame)


def variants(cfg):
    result=[]
    if 'clip_only' in cfg.methods:
        result.append(('clip_only',0.000))
    for method in cfg.methods:
        if method != 'clip_only':
            result.extend((method,weight) for weight in cfg.axis_weights)
    return result


def metadata_for(cfg,train,evaluation,seed,method,weight,epoch,**extra):
    return dict(run_id=f'{method}_{weight:.3f}_seed_{seed}',seed=seed,method=method,
                lambda_axis=float(weight),epoch=epoch,n_train_images=len(train),
                n_train_captions=sum(len(r['captions']) for r in train),
                n_evaluation_images=len(evaluation),n_evaluation_captions=sum(len(r['captions']) for r in evaluation),
                noise_seed=cfg.geometry_noise_seed,batch_size=min(cfg.batch_size,len(train)),
                anchor_refreshes_per_epoch=1,model=cfg.model,pretrained=cfg.pretrained,
                n_epochs_config=cfg.epochs,learning_rate_reciprocal=cfg.learning_rate_reciprocal,
                weight_decay=cfg.weight_decay,warmup_fraction=cfg.warmup_fraction,
                n_geometry_image_noise=cfg.n_noise_images_per_seed,n_geometry_text_noise=2*cfg.n_text_noise_each_per_seed,
                split_seed=cfg.split_seed,**extra)


def run_training(model,initial_state,initial,context,train,evaluation,preprocess_train,
                 preprocess_eval,tokenizer,cfg,device,seed,method,weight):
    """한 모델을 학습하면서 네 h1 실험을 매 에폭 실행한다. 중복 학습하지 않는다."""
    model.load_state_dict(initial_state)
    seed_everything(seed)
    optimizer=build_optimizer(model,cfg)
    amp_enabled=cfg.amp and device.type=='cuda'
    try: scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    except (AttributeError,TypeError): scaler=torch.cuda.amp.GradScaler(enabled=amp_enabled)
    batch_size=min(cfg.batch_size,len(train))
    n_batches=len(train)//batch_size
    total_steps=cfg.epochs*n_batches
    global_step=0
    baseline=replace(initial,metadata=metadata_for(cfg,train,evaluation,seed,method,weight,0,
                                                  n_optimizer_updates=0))
    for experiment in _TRAINING_CALLBACKS:
        experiment(baseline,context,render=(cfg.epochs==0 and getattr(context,'is_last_run',True)))
    for epoch in range(1,cfg.epochs+1):
        print(f'{method} lambda={weight:.3f} seed={seed} epoch={epoch}: 꼭지점 재측정',flush=True)
        with preserve_diagnostic_state(model):
            anchors=estimate_epoch_vertices(model,preprocess_eval,tokenizer,cfg,device)
        rng=random.Random(seed*100000+epoch)
        texts=[rng.choice(record['captions'])[1] for record in train]
        loader=make_loader(PairDataset(train,preprocess_train,tokenizer(texts)),cfg,device,batch_size,
                           shuffle=True,seed=seed*100000+epoch,training=True)
        model.train()
        clip_sum=axis_sum=total_sum=0.000
        seen=skipped=0
        for step,(images,tokens) in enumerate(loader,1):
            lr=lr_at_step(global_step,total_steps,cfg.warmup_fraction,1/cfg.learning_rate_reciprocal)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            images,tokens=images.to(device),tokens.to(device)
            with autocast_context(cfg,device):
                zi=model.encode_image(images,normalize=False)
                ti=model.encode_text(tokens,normalize=False)
            lc=clip_contrastive_loss(zi,ti,model.logit_scale.exp())
            if method=='imageness':
                la=batch_axis_loss(zi,ti,anchors)
            elif method=='conformity':
                la=batch_conformity_loss(zi,ti)
            else:
                la=lc.new_zeros(())
            loss=lc+weight*la
            if not torch.isfinite(loss):
                raise FloatingPointError('학습 손실에 비유한 값이 있습니다.')
            scaler.scale(loss).backward()
            skipped+=optimizer_step(model,optimizer,scaler,cfg,amp_enabled)
            count=len(images)
            seen+=count;global_step+=1
            clip_sum+=float(lc.detach())*count
            axis_sum+=float(la.detach())*count
            total_sum+=float(loss.detach())*count
            if step%cfg.log_every_steps==0 or step==n_batches:
                print(f'  batch={step}/{n_batches} CLIP={float(lc.detach()):.3f} '
                      f'axis={float(la.detach()):.3f}',flush=True)
        meta=metadata_for(cfg,train,evaluation,seed,method,weight,epoch,
                          n_optimizer_updates=n_batches-skipped,skipped_amp_steps=skipped,train_loss_clip=clip_sum/seen,
                          train_loss_axis=axis_sum/seen,train_loss=total_sum/seen)
        snapshot=encode_snapshot(model,train,evaluation,preprocess_eval,tokenizer,cfg,device,meta)
        for experiment in _TRAINING_CALLBACKS:
            experiment(snapshot,context,render=(epoch==cfg.epochs and getattr(context,'is_last_run',True)))
        metrics=retrieval_summary(query_stats(snapshot.scores,snapshot.mapping))
        print(f'  I2T R1={metrics["i2t_R1_percent"]:.3f}% '
              f'T2I R1={metrics["t2i_R1_percent"]:.3f}%',flush=True)


def validate_config(cfg):
    if not cfg.methods or len(set(cfg.methods))!=len(cfg.methods) or not set(cfg.methods)<=set(('clip_only','conformity','imageness')):
        raise ValueError('methods는 clip_only/conformity/imageness 중 중복 없이 지정하세요.')
    if 'clip_only' not in cfg.methods:
        raise ValueError('동일 seed 대조를 위해 clip_only를 포함하세요.')
    if not cfg.seeds or len(set(cfg.seeds))!=len(cfg.seeds) or any(s<0 for s in cfg.seeds):
        raise ValueError('학습 seed 설정을 확인하세요.')
    if cfg.epochs<0 or cfg.batch_size<2 or cfg.encode_batch_size<1 or cfg.num_workers<0:
        raise ValueError('에폭/배치/worker 설정을 확인하세요.')
    if not cfg.axis_weights or any(not math.isfinite(w) or w<=0 for w in cfg.axis_weights):
        raise ValueError('axis_weights는 양의 유한값을 지정하세요.')
    if len({f'{w:.3f}' for w in cfg.axis_weights})!=len(cfg.axis_weights):
        raise ValueError('3자리 lambda 이름이 중복됩니다.')
    if cfg.random_subspace_repeats<1 or cfg.pair_shuffle_repeats<1 or cfg.neighbor_k<1:
        raise ValueError('대조군 반복 수와 이웃 수를 확인하세요.')
    if not cfg.reference_modes or len(set(cfg.reference_modes))!=len(cfg.reference_modes) or not set(cfg.reference_modes)<=set(('initial_fixed','epoch_current')):
        raise ValueError('reference_modes는 initial_fixed/epoch_current 중 지정하세요.')
    if cfg.learning_rate_reciprocal<1 or not 0<=cfg.warmup_fraction<1 or cfg.max_grad_norm<=0 or cfg.log_every_steps<1:
        raise ValueError('학습률/warmup/gradient/log 설정을 확인하세요.')
    if cfg.n_noise_images_per_seed<1 or cfg.n_text_noise_each_per_seed<1 or cfg.geometry_noise_seed<0:
        raise ValueError('노이즈 수/seed 설정을 확인하세요.')





# =============================================================================
# 섹션 3: 모달리티 정합. 4개 실험. H1 네 함수는 4_residual.py에 정의되어 있다.
# =============================================================================
from types import SimpleNamespace


def _analysis_config(cfg):
    fields=dict(root=cfg.root,clip_model=cfg.model,clip_pretrained=cfg.pretrained,device=cfg.device,
                seeds=cfg.noise_seeds,n_noise=cfg.n_noise_images_per_seed,
                n_text_noise_each=cfg.n_text_noise_each_per_seed,encoders=('clip',),
                batch_size=cfg.encode_batch_size)
    for name in ('carzero_project','carzero_weight_root','carzero_ckpt','bimcq_ckpt',
                 'bimcq_config','nih_root','carzero_embedding_dir'):
        if hasattr(cfg,name):fields[name]=getattr(cfg,name)
    return an.Config(**fields)


def _static_inputs(cfg,encoders=None):
    ctx=an.Context(_analysis_config(cfg),encoders,out_name='alignment')
    image=ctx.images('clip','coco');text=ctx.domain_texts('clip')
    lookup={int(v):i for i,v in enumerate(image['ids'])}
    mapping=np.array([lookup[int(v)] for v in text['image_ids']],dtype=np.int64)
    # 실제 annotation ID를 사용해 정답 매핑과 캐시의 순서를 보존한다.
    coco=cfg.root/'data/coco/annotations/captions_val2017.json'
    annotations=sorted(json.loads(coco.read_text())['annotations'],key=lambda a:a['id'])
    by_text={}
    for row in annotations:by_text.setdefault((int(row['image_id']),row['caption'].strip()),[]).append(int(row['id']))
    caption_ids=[]
    for iid,s in zip(text['image_ids'],text['texts']):
        ids=by_text.get((int(iid),s))
        if not ids:raise ValueError('캡션 임베딩과 annotation의 ID/문장이 맞지 않습니다.')
        caption_ids.append(ids.pop(0))
    clean=dict(image=image['Z'],text=text['T'],mapping=mapping,ids=image['ids'],
               annotation_ids=np.asarray(caption_ids,np.int64))
    noise=dict(image=np.concatenate([ctx.reference_noise('clip',s) for s in cfg.noise_seeds]),
               text=np.concatenate([ctx.text_noise_embeddings('clip',s)['U'] for s in cfg.noise_seeds]))
    return ctx,clean,noise


def _close_context(ctx):
    for encoder in ctx._encoders.values():
        if hasattr(encoder,'close'):encoder.close()
    ctx._encoders.clear()


def exp10_axis_alignment(cfg=CONFIG,encoders=None,payload=None):
    """전체셋 평균을 정규화한 conformity와 b→r 방향의 정합도를 직접 비교."""
    owned=payload is None
    ctx,clean,noise=_static_inputs(cfg,encoders) if owned else payload
    try:
        with an.ExperimentOutput(ctx.out,'exp10_axis_alignment'):
            axes={m:an.Axes(clean[m].mean(0),noise[m].mean(0)) for m in ('image','text')}
            ai,at=axes['image'],axes['text']
            c=float(ai.r_hat @ at.r_hat);h=float(ai.m_hat @ at.m_hat)
            ca=float(an.angle(ai.r_hat,at.r_hat));ha=float(an.angle(ai.m_hat,at.m_hat))
            row=dict(model=cfg.model,pretrained=cfg.pretrained,n_images=len(clean['image']),n_captions=len(clean['text']),
                     n_image_noise=len(noise['image']),n_text_noise=len(noise['text']),
                     conformity_cosine=c,imageness_textness_cosine=h,delta_cosine=h-c,
                     conformity_angle_deg=ca,imageness_textness_angle_deg=ha,angle_reduction_deg=ca-ha,
                     centroid_gap=float(np.linalg.norm(ai.r_I-at.r_I)))
            for m,A in axes.items():
                row.update({f'R_{m}':A.R_I,f'Rb_{m}':float(np.linalg.norm(noise[m].mean(0))),
                            f'gamma_{m}':A.gamma,f'axis_length_{m}':A.length})
            an.save_csv(ctx.out,'full_set_axis_alignment',[row])
            fig,axs=make_figure()
            for ax,values,title in ((axs[0,0],[c,h],'Axis cosine'),(axs[0,1],[ca,ha],'Axis angle (deg)')):
                bars=ax.bar(['Conformity','Imageness–Textness'],values,color=[an.C1,an.C2])
                ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4);ax.set_title(title)
            for ax,m in zip(axs[1],('image','text')):
                S=axes[m].coords(clean[m]);ax.scatter(S['c'],S['n'],s=2,alpha=0.15)
                ax.set(xlabel='Conformity c',ylabel='Noise proximity n',title=f'{m}: full-set coordinates')
            an.save_figure(ctx.out,fig,'axis_alignment')
        return row
    finally:
        if owned:_close_context(ctx)


def symmetric_clip_loss_numpy(z,t,temperature,chunk):
    """한 이미지당 첫 정답 캡션으로 고정한 대칭 CLIP 손실. logsumexp 분할 계산."""
    if len(z)!=len(t) or len(z)<2 or temperature<=0 or chunk<1:raise ValueError('CLIP loss 입력 크기/온도를 확인하세요.')
    n=len(z);diagonal=np.sum(z*t,axis=1)/temperature
    row_sum=np.empty(n);column_sum=np.full(n,-np.inf)
    for start in range(0,n,chunk):
        logits=z[start:start+chunk] @ t.T / temperature
        row_sum[start:start+chunk]=logsumexp(logits,axis=1)
        column_sum=np.logaddexp(column_sum,logsumexp(logits,axis=0))
    return float(((row_sum-diagonal).mean()+(column_sum-diagonal).mean())/2)


def _shift_save_csv(path,rows):
    return an.save_csv(Path(path).parent,Path(path).stem,rows)


def make_protocols(clean, cfg):
    n = len(clean['image'])
    protocols = []
    if cfg.run_full:
        protocols.append(('full', -1, np.arange(n), np.arange(n)))
    if cfg.run_heldout:
        if cfg.n_reference < 2 or cfg.n_evaluation < 2 or cfg.n_reference + cfg.n_evaluation > n:
            raise ValueError('heldout의 reference/evaluation 크기를 캐시 이미지 수에 맞추세요.')
        for seed in cfg.split_seeds:
            permutation = np.random.default_rng(seed).permutation(n)
            ref = np.sort(permutation[:cfg.n_reference])
            ev = np.sort(permutation[cfg.n_reference:cfg.n_reference + cfg.n_evaluation])
            protocols.append(('heldout', seed, ref, ev))
    return protocols

def extract_set(clean, image_indices):
    selected = np.isin(clean['mapping'], image_indices)
    text_indices = np.flatnonzero(selected)
    lookup = np.full(len(clean['image']), -1, dtype=int)
    lookup[image_indices] = np.arange(len(image_indices))
    mapping = lookup[clean['mapping'][selected]]
    first = np.array([np.flatnonzero(mapping == i)[0] for i in range(len(image_indices))])
    return dict(image=clean['image'][image_indices], text=clean['text'][text_indices],
                mapping=mapping, first=first, image_ids=clean['ids'][image_indices],
                annotation_ids=clean['annotation_ids'][text_indices])

def build_state(ref, ev, noise, alpha, mode, full):
    centres = {modality: ref[modality].mean(axis=0) for modality in ('image', 'text')}
    shifted_ref, shifted_ev, shifted_noise = {}, {}, {}
    axes = {}
    for modality in ('image', 'text'):
        amount = alpha if mode == 'both' or mode == f'{modality}_only' else 0.000
        shift = amount * centres[modality]
        shifted_ref[modality] = an.normalize(ref[modality] - shift)
        shifted_ev[modality] = an.normalize(ev[modality] - shift)
        shifted_noise[modality] = an.normalize(noise[modality] - shift)
        mu = shifted_ref[modality].mean(axis=0)
        r, b = an.normalize(mu), an.normalize(shifted_noise[modality].mean(axis=0))
        denominator = float(np.dot(r-b, r-b))
        if denominator <= EPS:
            raise ValueError(f'{modality}: b/r가 너무 가까워 imageness 축이 불안정합니다.')
        axes[modality] = dict(mu=mu, r=r, b=b, denominator=denominator,
                              gamma=float(np.dot(b, r)))
    coordinates = {}
    for modality in ('image', 'text'):
        z, axis = shifted_ev[modality], axes[modality]
        C = z @ axis['mu']
        if full:
            size = len(shifted_ref[modality])
            C = (size * C - 1) / (size - 1)  # 정확한 leave-one-out conformity
        coordinates[modality] = dict(C=C, c=z @ axis['r'],
                                    h=((z-axis['b']) @ (axis['r']-axis['b'])) / axis['denominator'])
    gap = float(np.linalg.norm(shifted_ev['image'].mean(axis=0) - shifted_ev['text'].mean(axis=0)))
    angle = float(np.degrees(np.arccos(np.clip(np.dot(axes['image']['r'], axes['text']['r']), -1, 1))))
    return dict(z=shifted_ev['image'], t=shifted_ev['text'], axes=axes,
                coordinates=coordinates, centroid_gap=gap, mean_axis_angle_deg=angle)

def fixed_h(state, baseline_axes):
    result = {}
    for modality, key in (('image', 'z'), ('text', 't')):
        z, axis = state[key], baseline_axes[modality]
        result[modality] = ((z-axis['b']) @ (axis['r']-axis['b'])) / axis['denominator']
    return result

def pairing_statistics(a, b, mapping, shuffle):
    errors = (a[mapping] - b) ** 2
    wrong = (a[shuffle[mapping]] - b) ** 2
    counts = np.bincount(mapping, minlength=len(a))
    mse = float(np.mean(np.bincount(mapping, weights=errors, minlength=len(a)) / counts))
    null_mse = float(np.mean(np.bincount(mapping, weights=wrong, minlength=len(a)) / counts))
    pair_a = a[mapping]
    correlation = float(np.corrcoef(pair_a, b)[0, 1]) if np.std(pair_a) > EPS and np.std(b) > EPS else float('nan')
    return dict(paired_mse=mse, shuffled_mse=null_mse, shuffle_advantage=null_mse-mse,
                pair_pearson=correlation, mean_difference=float(a.mean()-b.mean()),
                image_std=float(a.std()), text_std=float(b.std()))

def derangement(size, seed):
    permutation = np.random.default_rng(seed).permutation(size)
    shuffled = np.empty(size, dtype=int)
    shuffled[permutation] = np.roll(permutation, 1)
    return shuffled

def distribution_metrics(a, b, edges, pseudocount):
    pa = np.histogram(a, edges)[0].astype(float)
    pb = np.histogram(b, edges)[0].astype(float)
    if pa.sum() != len(a) or pb.sum() != len(b):
        raise ValueError('Histogram 범위를 벗어난 값이 있습니다.')
    # 모달리티별 표본 개수가 달라도 같은 혼합 비중의 smoothing.
    smoothing = pseudocount * len(pa) / (min(len(a), len(b)) + pseudocount * len(pa))
    pa, pb = pa/pa.sum(), pb/pb.sum()
    pa = (1-smoothing)*pa + smoothing/len(pa)
    pb = (1-smoothing)*pb + smoothing/len(pb)
    middle = (pa+pb)/2
    return dict(js=float((np.sum(pa*np.log(pa/middle)) + np.sum(pb*np.log(pb/middle)))/2),
                kl_image_text=float(np.sum(pa*np.log(pa/pb))),
                kl_text_image=float(np.sum(pb*np.log(pb/pa))))

def histogram_edges(values, bins):
    lower = min(float(np.min(v)) for v in values)
    upper = max(float(np.max(v)) for v in values)
    padding = max((upper-lower)/1000, EPS)
    return np.linspace(lower-padding, upper+padding, bins+1)

def _shift_exp1_baseline(clean, noise, cfg, loss_evaluator, out):
    rows = []
    for protocol, seed, ref_indices, ev_indices in make_protocols(clean, cfg):
        ref, ev = extract_set(clean, ref_indices), extract_set(clean, ev_indices)
        state = build_state(ref, ev, noise, 0.000, 'image_only', protocol == 'full')
        loss = loss_evaluator(state['z'], state['t'][ev['first']])
        row = dict(protocol=protocol, seed=seed, n_reference_images=len(ref_indices),
                   n_evaluation_images=len(ev_indices), n_evaluation_captions=len(ev['text']),
                   clip_loss=loss, centroid_gap=state['centroid_gap'],
                   mean_axis_angle_deg=state['mean_axis_angle_deg'],
                   gamma_image=state['axes']['image']['gamma'], gamma_text=state['axes']['text']['gamma'])
        shuffle = derangement(len(ev_indices), seed+9011)
        for name in ('C', 'c', 'h'):
            stats = pairing_statistics(state['coordinates']['image'][name], state['coordinates']['text'][name],
                                       ev['mapping'], shuffle)
            row.update({f'{name}_{key}': value for key, value in stats.items()})
        rows.append(row)
    _shift_save_csv(out / 'exp1_baseline.csv', rows)
    return rows

def _shift_exp2_sweeps(clean, noise, cfg, loss_evaluator, out):
    result, invalid, splits = [], [], []
    for protocol, seed, ref_indices, ev_indices in make_protocols(clean, cfg):
        ref, ev = extract_set(clean, ref_indices), extract_set(clean, ev_indices)
        splits.append(dict(protocol=protocol, seed=seed, reference_image_ids=ref['image_ids'].tolist(),
                           evaluation_image_ids=ev['image_ids'].tolist()))
        baseline = build_state(ref, ev, noise, 0.000, 'image_only', protocol == 'full')
        shuffle = derangement(len(ev_indices), seed+9011)
        # 모든 이동 조건에 공통인 고정 histogram 경계를 사용한다.
        states = []
        for mode in cfg.shift_modes:
            for alpha in cfg.alpha_grid:
                try:
                    state = build_state(ref, ev, noise, alpha, mode, protocol == 'full')
                    fixed = fixed_h(state, baseline['axes'])
                except ValueError as exc:
                    invalid.append(dict(protocol=protocol, seed=seed, mode=mode, alpha=alpha, reason=str(exc)))
                    continue
                loss = loss_evaluator(state['z'], state['t'][ev['first']])
                print(f'{protocol} seed={seed} {mode} alpha={alpha:.3f} '
                      f'loss={loss:.3f} gap={state["centroid_gap"]:.3f}', flush=True)
                # 전체 임베딩을 grid마다 보관하지 않고 좌표와 요약값만 유지한다.
                del state['z'], state['t']
                states.append(dict(mode=mode, alpha=alpha, state=state, fixed=fixed, loss=loss))
        if not states:
            _shift_save_csv(out / 'invalid_conditions.csv', invalid)
            raise ValueError(f'{protocol}, seed={seed}: 유효한 이동 조건이 없습니다. invalid_conditions.csv를 확인하세요.')
        for bins in cfg.histogram_bins:
            edges = {name: histogram_edges([item['state']['coordinates'][modality][name]
                                            for item in states for modality in ('image', 'text')], bins)
                     for name in ('C', 'c', 'h')}
            fixed_edges = histogram_edges([item['fixed'][modality] for item in states
                                            for modality in ('image', 'text')], bins)
            for item in states:
                state, alpha, mode = item['state'], item['alpha'], item['mode']
                for policy in ('moving', 'fixed'):
                    row = dict(protocol=protocol, seed=seed, shift_mode=mode, alpha=alpha,
                               anchor_policy=policy, histogram_bins=bins, clip_loss=item['loss'],
                               centroid_gap=state['centroid_gap'], mean_axis_angle_deg=state['mean_axis_angle_deg'])
                    for name in ('C', 'c', 'h'):
                        if name == 'h' and policy == 'fixed':
                            a, b, bins_edges = item['fixed']['image'], item['fixed']['text'], fixed_edges
                        else:
                            a, b = (state['coordinates'][modality][name] for modality in ('image', 'text'))
                            bins_edges = edges[name]
                        measures = distribution_metrics(a, b, bins_edges, cfg.histogram_pseudocount)
                        measures.update(pairing_statistics(a, b, ev['mapping'], shuffle))
                        row.update({f'{name}_{key}': value for key, value in measures.items()})
                    row['gamma_image'] = state['axes']['image']['gamma']
                    row['gamma_text'] = state['axes']['text']['gamma']
                    row['h_axis_norm_sq_image'] = state['axes']['image']['denominator']
                    row['h_axis_norm_sq_text'] = state['axes']['text']['denominator']
                    result.append(row)
        _shift_save_csv(out / 'exp2_sweep.csv', result)
    _shift_save_csv(out / 'invalid_conditions.csv', invalid)
    _shift_save_csv(out / 'split_ids.csv', [dict(protocol=r['protocol'], seed=r['seed'], split=k, image_id=int(v)) for r in splits for k in ('reference_image_ids','evaluation_image_ids') for v in r[k]])
    return result

def spearman(a, b):
    return float(spearmanr(a, b).statistic) if np.std(a) > EPS and np.std(b) > EPS else float('nan')

def ridge_predict(x_train, y_train, x_test, penalty):
    centre, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale <= EPS, 1, scale)
    x, xe = (x_train-centre)/scale, (x_test-centre)/scale
    y_mean = y_train.mean()
    coefficient = np.linalg.solve(x.T@x + penalty*np.eye(x.shape[1]), x.T@(y_train-y_mean))
    return y_mean + xe@coefficient

def _shift_exp3_comparison(sweep, cfg, out):
    groups = {}
    for row in sweep:
        key = (row['protocol'], row['seed'], row['anchor_policy'], row['histogram_bins'])
        groups.setdefault(key, []).append(row)
    minima, prediction_rows = [], []
    feature_sets = {'C': ('C_js', 'C_paired_mse'), 'c': ('c_js', 'c_paired_mse'),
                    'h': ('h_js', 'h_paired_mse'),
                    'C+h': ('C_js', 'C_paired_mse', 'h_js', 'h_paired_mse')}
    for key, group in groups.items():
        protocol, seed, policy, bins = key
        for mode in cfg.shift_modes:
            curve = sorted([r for r in group if r['shift_mode'] == mode], key=lambda r: r['alpha'])
            if len(curve) < 3:
                continue
            loss = np.array([r['clip_loss'] for r in curve])
            best_loss = curve[int(np.argmin(loss))]
            for metric in ('C_js', 'c_js', 'h_js', 'C_paired_mse', 'c_paired_mse', 'h_paired_mse'):
                values = np.array([r[metric] for r in curve])
                best = curve[int(np.argmin(values))]
                minima.append(dict(protocol=protocol, seed=seed, anchor_policy=policy, histogram_bins=bins,
                                   shift_mode=mode, metric=metric, alpha_loss_min=best_loss['alpha'],
                                   alpha_metric_min=best['alpha'], alpha_min_distance=abs(best['alpha']-best_loss['alpha']),
                                   loss_regret_at_metric_min=best['clip_loss']-best_loss['clip_loss'],
                                   loss_spearman=spearman(loss, values),
                                   metric_min=best[metric]))
        # alpha=0은 모든 조건에서 동일하므로 예측 CV에서 제외한다.
        usable = [r for r in group if abs(r['alpha']) > EPS]
        for target in ('clip_loss', 'centroid_gap'):
            for features_name, columns in feature_sets.items():
                for held_mode in cfg.shift_modes:
                    train = [r for r in usable if r['shift_mode'] != held_mode]
                    test = [r for r in usable if r['shift_mode'] == held_mode]
                    if len(train) < len(columns)+2 or len(test) < 2:
                        continue
                    # 학습 fold에서만 standardization; log1p는 사전 지정된 변환.
                    xt = np.log1p(np.array([[r[c] for c in columns] for r in train]))
                    xe = np.log1p(np.array([[r[c] for c in columns] for r in test]))
                    yt, ye = np.array([r[target] for r in train]), np.array([r[target] for r in test])
                    predicted = ridge_predict(xt, yt, xe, cfg.ridge_penalty)
                    sse, denominator = float(np.sum((predicted-ye)**2)), float(np.sum((ye-ye.mean())**2))
                    prediction_rows.append(dict(protocol=protocol, seed=seed, anchor_policy=policy,
                                                histogram_bins=bins, target=target, features=features_name,
                                                heldout_shift_mode=held_mode, n_train=len(train), n_test=len(test),
                                                rmse=float(np.sqrt(sse/len(test))),
                                                mae=float(np.mean(np.abs(predicted-ye))),
                                                r_squared=1-sse/denominator if denominator > EPS else float('nan')))
    _shift_save_csv(out / 'exp3_minima.csv', minima)
    _shift_save_csv(out / 'exp3_prediction.csv', prediction_rows)
    return minima, prediction_rows


def exp11_mean_shift(cfg=CONFIG,encoders=None,payload=None):
    """평균 이동에 따른 C/c/h 분포 정합, 손실, 갭, 최솟값 및 조건 외 예측.
    C는 집합 평균 cosine, c는 정규화 평균 방향 투영이다. 둘을 별도 열로 기록한다.
    """
    owned=payload is None
    ctx,clean,noise=_static_inputs(cfg,encoders) if owned else payload
    shifted_cfg=SimpleNamespace(run_full=cfg.mean_shift_full,run_heldout=cfg.mean_shift_heldout,
        split_seeds=cfg.seeds,n_reference=cfg.mean_shift_reference_images,n_evaluation=cfg.mean_shift_evaluation_images,
        alpha_grid=cfg.alpha_grid,shift_modes=cfg.shift_modes,histogram_bins=cfg.histogram_bins,
        histogram_pseudocount=cfg.histogram_pseudocount,ridge_penalty=cfg.ridge_penalty)
    loss=lambda z,t:symmetric_clip_loss_numpy(z,t,cfg.loss_temperature,cfg.loss_chunk)
    try:
        with an.ExperimentOutput(ctx.out,'exp11_mean_shift'):
            _shift_exp1_baseline(clean,noise,shifted_cfg,loss,ctx.out)
            sweep=_shift_exp2_sweeps(clean,noise,shifted_cfg,loss,ctx.out)
            minima,prediction=_shift_exp3_comparison(sweep,shifted_cfg,ctx.out)
            df=pd.DataFrame(sweep)
            selected=df[(df.anchor_policy=='moving')&(df.histogram_bins==cfg.histogram_bins[0])]
            fig,axs=__import__('matplotlib.pyplot',fromlist=['plt']).subplots(2,3,figsize=(15,9))
            for (protocol,mode),d in selected.groupby(['protocol','shift_mode']):
                g=d.groupby('alpha').mean(numeric_only=True);lab=f'{protocol}/{mode}'
                for ax,key in zip(axs.flat,('clip_loss','centroid_gap','C_js','c_js','h_js','h_paired_mse')):
                    ax.plot(g.index,g[key],label=lab);ax.set(xlabel='Mean shift coefficient α',ylabel=key)
            axs[0,0].legend(fontsize=6)
            an.save_figure(ctx.out,fig,'mean_shift_sweeps')
            if minima:
                md=pd.DataFrame(minima);fig,axs=make_figure()
                for ax,key in zip(axs.flat,('alpha_min_distance','loss_regret_at_metric_min','loss_spearman')):
                    g=md.groupby('metric')[key].mean();ax.barh(g.index,g.values);ax.set_xlabel(key)
                if prediction:
                    pr=pd.DataFrame(prediction);g=pr.groupby(['features','target']).rmse.mean()
                    axs[1,1].barh(['/'.join(k) for k in g.index],g.values);axs[1,1].set_xlabel('Held-out shift condition RMSE')
                an.save_figure(ctx.out,fig,'minima_and_prediction')
        return sweep
    finally:
        if owned:_close_context(ctx)


_TRAINING_DIAGNOSTICS=['geometry_set','axis_metric','retrieval_metric','n_epochs','n_epoch_changes',
                     'pearson','spearman','delta_pearson','delta_spearman',
                     'training_conformity_cosine','training_imageness_textness_cosine',
                     'image_vertex_change_deg','text_vertex_change_deg']


def _record_training_epoch(snapshot,context,render=False):
    table=context.table('exp12_alignment_training',GEOMETRY_COLUMNS+RETRIEVAL_COLUMNS+_TRAINING_DIAGNOSTICS)
    metrics=retrieval_summary(query_stats(snapshot.scores,snapshot.mapping))
    for mode,frame in context.frames(snapshot):
        geom=geometry(snapshot.z,snapshot.t,frame)
        row=dict(**context.meta(snapshot,mode),record_type='epoch',geometry_set='heldout_images_all_captions',
                 **geom,**metrics,
                 training_conformity_cosine=float(snapshot.frame['image']['r'] @ snapshot.frame['text']['r']),
                 training_imageness_textness_cosine=float(snapshot.frame['image']['a'] @ snapshot.frame['text']['a']),
                 image_vertex_change_deg=angle(snapshot.frame['image']['b'],context.initial.frame['image']['b']),
                 text_vertex_change_deg=angle(snapshot.frame['text']['b'],context.initial.frame['text']['b']))
        table.append(row)
        if snapshot.metadata['epoch']==context.cfg.epochs:
            runs=[r for r in table.summaries if r.get('record_type')=='epoch' and r['run_id']==row['run_id'] and r['reference_mode']==mode]
            runs=sorted(runs,key=lambda r:r['epoch'])
            for axis_metric in ('conformity_cosine','imageness_textness_cosine','centroid_gap'):
                for retrieval_metric in ('i2t_R1_percent','i2t_R5_percent','i2t_R10_percent',
                                         't2i_R1_percent','t2i_R5_percent','t2i_R10_percent'):
                    x=np.array([r[axis_metric] for r in runs]);y=np.array([r[retrieval_metric] for r in runs])
                    def corr(a,b,kind):
                        if len(a)<3 or np.std(a)<=EPS or np.std(b)<=EPS:return float('nan')
                        return float(stats.pearsonr(a,b).statistic if kind=='pearson' else stats.spearmanr(a,b).statistic)
                    table.append(dict(**context.meta(snapshot,mode),record_type='correlation',
                                      axis_metric=axis_metric,retrieval_metric=retrieval_metric,n_epochs=len(x),
                                      n_epoch_changes=max(0,len(x)-1),pearson=corr(x,y,'pearson'),spearman=corr(x,y,'spearman'),
                                      delta_pearson=corr(np.diff(x),np.diff(y),'pearson'),delta_spearman=corr(np.diff(x),np.diff(y),'spearman')))
    table.flush()
    if render:
        fig,axs=make_figure();p=lambda r:r.get('record_type')=='epoch' and r['reference_mode']=='epoch_current'
        for ax,key,title in ((axs[0,0],'imageness_textness_angle_deg','Imageness–Textness angle (deg)'),
                             (axs[0,1],'conformity_angle_deg','Conformity angle (deg)'),
                             (axs[1,0],'i2t_R1_percent','Image→text R@1 (%)'),
                             (axs[1,1],'t2i_R1_percent','Text→image R@1 (%)')):
            plot_curves(ax,table.summaries,key,title,p)
        save_figure(fig,table.png)


def exp12_alignment_training(config=CONFIG,model_factory=None):
    """CLIP-only / conformity / Imageness의 동일 조건 학습. 에폭 전 b 갱신,
    매 배치의 정규화 평균 r로 L=L_CLIP+λ(1−a_Iᵀa_T).
    한 학습 패스를 4_residual.py 네 실험과 공유한다. JSON/checkpoint 출력 없음.
    """
    global _TRAINING_CALLBACKS
    import importlib
    rs = importlib.import_module("4_residual")
    _TRAINING_CALLBACKS=(_record_training_epoch,)+rs.EXPERIMENTS
    validate_config(config);require_dependencies()
    device=torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train,evaluation,_,protocol=load_splits(config)
    if len(evaluation)<2:raise ValueError('검색 평가에는 이미지가 두 장 이상 필요합니다.')
    output=Path(config.output_directory or an.result_directory(config.root,'alignment'))
    residual_output=an.result_directory(config.root,'residual')
    output.mkdir(parents=True,exist_ok=True);residual_output.mkdir(parents=True,exist_ok=True)
    seed_everything(config.seeds[0])
    if model_factory is None:
        model,preprocess_train,preprocess_eval=open_clip.create_model_and_transforms(
            config.model,pretrained=config.pretrained,precision='fp32',cache_dir=str(config.root / 'ckpt'))
        tokenizer=open_clip.get_tokenizer(config.model)
    else:model,preprocess_train,preprocess_eval,tokenizer=model_factory(config)
    model=model.to(device).float();model.requires_grad_(True)
    if config.gradient_checkpointing:model.set_grad_checkpointing(True)
    initial_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    initial=encode_snapshot(model,train,evaluation,preprocess_eval,tokenizer,config,device,
                            metadata_for(config,train,evaluation,config.seeds[0],'clip_only',0.000,0))
    context=AnalysisContext(config,output,initial)
    # H1 표와 그림은 residual 폴더, 학습 궤적은 alignment 폴더에 저장한다.
    original_table=context.table
    def route_table(name,columns):
        if name.startswith('h1_'):
            if name not in context.tables:context.tables[name]=OutputTable(residual_output,name,columns)
            return context.tables[name]
        return original_table(name,columns)
    context.table=route_table
    try:
        print(f'{protocol}; train={len(train)}, eval={len(evaluation)}, runs={len(variants(config))*len(config.seeds)}',flush=True)
        scheduled=[(seed,method,weight) for seed in config.seeds for method,weight in variants(config)]
        for index,(seed,method,weight) in enumerate(scheduled):
            context.is_last_run=index==len(scheduled)-1
            run_training(model,initial_state,initial,context,train,evaluation,preprocess_train,
                         preprocess_eval,tokenizer,config,device,seed,method,weight)
    finally:context.close()
    expected=[output/'exp12_alignment_training.csv',output/'exp12_alignment_training.png']
    expected.extend(residual_output/f'{fn.__name__}.{ext}' for fn in rs.EXPERIMENTS for ext in ('csv','png'))
    missing=[str(p) for p in expected if not p.is_file()]
    if missing:raise AssertionError(f'실험 출력 누락: {missing}')
    print(f'완료: {output}, {residual_output}',flush=True)
    return output


def load_bimcq_model(cfg,device):
    """첨부 1_extract_embeddings.py의 Bi-MCQ 로더와 같은 multi/config 경로."""
    from omegaconf import OmegaConf
    repo=next((p for p in (cfg.carzero_project,cfg.carzero_project/'Bi-MCQ') if (p/'CARZero/CARZero.py').is_file()),None)
    if repo is None:raise FileNotFoundError(f'CARZero 소스: {cfg.carzero_project}')
    import sys
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    import CARZero as carzero
    from CARZero import builder
    checkpoint=cfg.carzero_weight_root/cfg.carzero_ckpt
    config=OmegaConf.load(cfg.bimcq_config)
    previous=Path.cwd()
    try:
        os.chdir(cfg.carzero_weight_root)
        try:
            model=carzero.load_CARZero(name=str(cfg.carzero_weight_root/'best_model.ckpt'),
                                      device='cpu',multi=True,cfg=config)
        except TypeError:
            ckpt=torch.load(cfg.carzero_weight_root/'best_model.ckpt',map_location='cpu',weights_only=False)
            model=builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(ckpt['hyper_parameters'])
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected=model.state_dict();fixed={}
        for key,value in ckpt['state_dict'].items():
            for prefix in ('CARZero_model.','CARZero.'):
                if key.startswith(prefix):key=key[len(prefix):];break
            if key in expected:fixed[key]=value
        missing=set(expected)-set(fixed)
        if missing:raise ValueError(f'Bi-MCQ checkpoint key 누락: {len(missing)}. 모델/config를 확인하세요.')
        model.load_state_dict(fixed,strict=True)
    finally:os.chdir(previous)
    return model.to(device).eval()


class BiMCQEncoder(an.CarzeroEncoder):
    name='bimcq'
    loader=staticmethod(load_bimcq_model)


def _bimcq_image_frame(Z, A):
    """질환별로 반복해서 사용할 이미지 투영을 한 번 계산한다."""
    D=Z-A.r_I
    Pz=A.project_out(Z,'be')
    norms=np.linalg.norm(Pz,axis=1)
    valid=norms>an.EPS
    zres=np.divide(Pz,norms[:,None],out=np.zeros_like(Pz),where=valid[:,None])
    return dict(D=D,db=D @ A.b,de=D @ A.e,Pd=A.project_out(D,'be'),
                zres=zres,residual_valid=valid)


def _bimcq_prompt_diagnostics(Z, A, frame, tp, tn, y):
    """0인 정·부정 차이는 동점 점수로 보존하고 정의되지 않은 비율만 NaN으로 기록한다."""
    dt=tp-tn
    score=Z @ dt
    M=float(A.r_I @ dt)
    V=frame['db']*float(dt @ A.b)
    Ae=frame['de']*float(dt @ A.e)
    tres=A.project_out(dt[None],'be')[0]
    Ur=frame['Pd'] @ tres
    error=float(np.max(np.abs(score-M-V-Ae-Ur)))
    if not np.isfinite(error) or error>0.000001:
        raise AssertionError('Bi-MCQ score decomposition failed.')
    magnitudes=np.array([abs(M),np.abs(V).mean(),np.abs(Ae).mean(),np.abs(Ur).mean()])
    denom=float(magnitudes.sum())
    if not np.isfinite(denom):
        raise ValueError('Bi-MCQ 점수 분해에 NaN 또는 infinity가 있습니다.')
    fraction_valid=denom>an.EPS
    fractions=magnitudes/denom if fraction_valid else np.full(4,np.nan)
    delta_norm=float(np.linalg.norm(dt))
    same=np.array_equal(tp,tn)
    status=('identical_text_embeddings' if same else
            'near_zero_text_difference' if delta_norm<=an.EPS else
            'near_zero_decomposition' if not fraction_valid else 'ok')
    # 잔차 방향이 없는 이미지가 있으면 기존 정의의 전체셋 분리도를 계산할 수 없다.
    has_classes=bool(y.any() and (~y).any())
    gap=np.nan
    if frame['residual_valid'].all() and has_classes:
        residual_score=frame['zres'] @ tres
        gap=float(residual_score[y].mean()-residual_score[~y].mean())
    pos_score=Z @ tp
    neg_score=Z @ tn
    return dict(auroc=an.auroc(score[y],score[~y]),
                positive_auroc=an.auroc(pos_score[y],pos_score[~y]),
                negative_auroc=an.auroc(neg_score[y],neg_score[~y]),
                text_pair_cosine=float(np.clip(tp @ tn,-1,1)),text_delta_norm=delta_norm,
                text_embeddings_equal=bool(same),decomposition_status=status,
                decomposition_magnitude=denom,fraction_valid=bool(fraction_valid),
                score_std=float(score.std()),score_min=float(score.min()),score_max=float(score.max()),
                residual_separation=gap,residual_valid_fraction=float(frame['residual_valid'].mean()),
                n_zero_image_residuals=int((~frame['residual_valid']).sum()),
                frac_M=float(fractions[0]),frac_V=float(fractions[1]),
                frac_A=float(fractions[2]),frac_U=float(fractions[3]),
                reconstruction_max_error=error)


def _bimcq_token_input_equal(encoder, texts, negtexts):
    """실제 인코더의 정·부정 token IDs, mask, token types가 같은지 기록한다."""
    model=getattr(encoder,'model',None)
    if model is None or not hasattr(model,'process_text'):
        return np.full(len(texts),np.nan)
    processed=model.process_text(texts+negtexts,encoder.device)
    equal=np.ones(len(texts),dtype=bool)
    for key in ('caption_ids','attention_mask','token_type_ids'):
        if key not in processed:
            continue
        value=processed[key]
        value=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        if len(value)!=2*len(texts):
            raise ValueError(f'Bi-MCQ token input batch mismatch: {key}')
        equal &= np.all(value[:len(texts)]==value[len(texts):],axis=tuple(range(1,value.ndim)))
    if 'caption_ids' not in processed:
        raise ValueError('Bi-MCQ process_text에서 caption_ids를 찾을 수 없습니다.')
    return equal


def exp13_bimcq_comparison(cfg=CONFIG,encoders=None):
    """같은 NIH 입력에서 CARZero/Bi-MCQ의 b를 각각 재측정한다.
    현재 b와 CARZero의 고정 b를 모두 보고해 기준점 변화와 분포 변화를 구분한다.
    """
    acfg=replace(_analysis_config(cfg),encoders=('carzero',))
    ctx=an.Context(acfg,encoders,out_name='alignment');rows=[];unit_rows=[]
    try:
        car=ctx.encoder('carzero')
        bm=(encoders or {}).get('bimcq')
        if bm is None:
            # CarzeroEncoder는 내부에서 torch를 가져오므로 이 실험의 로더도 실제 torch를 받는다.
            global torch
            import torch as torch_module
            torch=torch_module
            bm=BiMCQEncoder(replace(acfg,carzero_ckpt=acfg.bimcq_ckpt))
            ctx._encoders['bimcq']=bm
        imgs=ctx.images('carzero','nih');Z0=imgs['Z'];labels=imgs['labels']
        texts,didx,tidx=an.nih_prompts('pos');negtexts,_,_=an.nih_prompts('neg')
        # 기존 14개 단일 템플릿도 같은 CSV에 포함한다.
        legacy_pos=[f'There is {t}' for t in an.NIH_TERMS];legacy_neg=[f'There is no {t}' for t in an.NIH_TERMS]
        texts=texts+legacy_pos;negtexts=negtexts+legacy_neg
        didx=np.r_[didx,np.arange(14)];tidx=np.r_[tidx,np.full(14,-1)]
        U0=np.concatenate([ctx.reference_noise('carzero',s) for s in acfg.seeds]);b0=an.normalize(U0.mean(0))
        text_noise=sum((sum((list(x) for x in ctx.text_noise_strings(s)),[]) for s in acfg.seeds),[])
        btext0=None
        with an.ExperimentOutput(ctx.out,'exp13_bimcq_comparison'):
            for name,encoder in (('carzero',car),('bimcq',bm)):
                Z=Z0 if name=='carzero' else an.normalize(encoder.encode_paths(imgs['paths']))
                Tp=an.normalize(encoder.encode_texts(texts));Tn=an.normalize(encoder.encode_texts(negtexts))
                token_equal=_bimcq_token_input_equal(encoder,texts,negtexts)
                if name=='carzero':U=U0
                else:
                    def generate():
                        for seed in acfg.seeds:
                            rng=np.random.default_rng(np.random.SeedSequence([seed,1101]))
                            for _ in range(acfg.n_noise):yield rng.uniform(0,1,(*encoder.hw,3)).astype(np.float32)
                    U=an.normalize(encoder.encode_pixels(generate()))
                Ut=an.normalize(encoder.encode_texts(text_noise));b=an.normalize(U.mean(0));bt=an.normalize(Ut.mean(0))
                if btext0 is None:btext0=bt.copy()
                for policy,bi,btref in (('model_current',b,bt),('carzero_fixed',b0,btext0)):
                    A=an.Axes(Z.mean(0),bi);AT=an.Axes(np.concatenate([Tp,Tn]).mean(0),btref)
                    S=A.coords(Z)
                    frame=_bimcq_image_frame(Z,A)
                    rows.append(dict(model=name,anchor_policy=policy,n_images=len(Z),
                                     R=A.R_I,Rb=float(np.linalg.norm(U.mean(0))),gamma=A.gamma,
                                     conformity_cosine=float(A.r_hat @ AT.r_hat),imageness_textness_cosine=float(A.m_hat @ AT.m_hat),
                                     vertex_change_deg=float(an.angle(b,b0)),text_vertex_change_deg=float(an.angle(bt,btext0)),
                                     delta_b_sd=float(S['delta_b'].std()),delta_e_sd=float(S['delta_e'].std()),
                                     residual_norm_mean=float(S['resid'].mean())))
                    for j in range(len(Tp)):
                        y=labels[:,didx[j]].astype(bool)
                        diagnostics=_bimcq_prompt_diagnostics(Z,A,frame,Tp[j],Tn[j],y)
                        unit_rows.append(dict(model=name,anchor_policy=policy,disease=an.NIH_CLASSES[didx[j]],template=int(tidx[j]),
                                              positive_prompt=texts[j],negative_prompt=negtexts[j],
                                              token_input_equal=token_equal[j],**diagnostics))
            for name in ('carzero','bimcq'):
                flagged=[r for r in unit_rows if r['model']==name and r['anchor_policy']=='model_current'
                         and r['decomposition_status']!='ok']
                if flagged:
                    print(f'  {name}: {len(flagged)}/{len(Tp)}개 정·부정 쌍의 분해 진단값을 기록했습니다.',flush=True)
                    for row in flagged[:5]:
                        print(f"    {row['disease']} / template {row['template']}: {row['decomposition_status']}; "
                              f"tokens_equal={row['token_input_equal']}; {row['positive_prompt']!r} / {row['negative_prompt']!r}",flush=True)
            df=an.save_csv(ctx.out,'model_geometry',rows);uf=an.save_csv(ctx.out,'disease_template_scores',unit_rows)
            fig,axs=make_figure()
            current=df[df.anchor_policy=='model_current'].set_index('model')
            for ax,key in zip(axs.flat,('R','vertex_change_deg','residual_norm_mean','imageness_textness_cosine')):
                ax.bar(current.index,current[key],color=[an.C1,an.C2]);ax.set_title(key)
            an.save_figure(ctx.out,fig,'bimcq_geometry')
        return df
    finally:_close_context(ctx)


def run_alignment(config=CONFIG,encoders=None,model_factory=None):
    selected=tuple(config.experiments)
    if not selected or len(set(selected))!=len(selected) or any(n not in (10,11,12,13) for n in selected):
        raise ValueError('Config.experiments에는 10, 11, 12, 13 중 실행할 번호를 중복 없이 넣으세요.')
    # 필요한 자산을 긴 학습 전에 확인한다. 실제 데이터/모델 누락을 결과로 포장하지 않는다.
    if 13 in selected and config.run_bimcq and encoders is None:
        acfg=_analysis_config(config)
        for p in (acfg.carzero_weight_root/acfg.carzero_ckpt,acfg.carzero_weight_root/acfg.bimcq_ckpt,acfg.bimcq_config):
            if not p.is_file():raise FileNotFoundError(f'Bi-MCQ 비교에 필요한 파일: {p}')
    if 10 in selected or 11 in selected:
        payload=_static_inputs(config,encoders)
        try:
            if 13 in selected and config.run_bimcq:payload[0].data.nih
            if 10 in selected:exp10_axis_alignment(config,payload=payload)
            if 11 in selected:exp11_mean_shift(config,payload=payload)
        finally:_close_context(payload[0])
    if 12 in selected:exp12_alignment_training(config,model_factory)
    if 13 in selected and config.run_bimcq:exp13_bimcq_comparison(config,encoders)


_TRAINING_CALLBACKS=()
EXPERIMENTS=(exp10_axis_alignment,exp11_mean_shift,exp12_alignment_training,exp13_bimcq_comparison)

"""H1: 전역 축과 잔차 구조의 검색 기여. CLI/JSON 설정·출력 없음.
python h1_residual.py 로 실행하고 Config만 수정한다. COCO 원본 annotation JSON은 입력으로 읽는다.
한 학습 패스의 에폭별 임베딩을 네 실험이 공유한다. 실험별 CSV 하나/PNG 하나만 저장한다.
1) h1_exp1_subspace_retrieval: 축/평면/잔차 점수와 같은 차원 random/PCA 대조군.
2) h1_exp2_margin_decomposition: M,V,A_e,U 마진, 현재/초기 경쟁 후보 비교.
3) h1_exp3_pair_coordinates: C,c,h 정답 쌍과 이미지 단위 shuffle 비교.
4) h1_exp4_residual_structure: 중심 갭/분산/유효 랭크/잔차 이웃·대응 보존.
기본: COCO val 이미지 4000/1000 분할, 10 epochs, batch128, seeds0/1/2.
CLIP-only와 conformity/imageness 축 로스(lambda0.100/1.000)를 동일 초기 가중치로 비교한다.
학습 기준축·PCA는 학습 이미지/모든 학습 캡션으로 추정한다. 평가 이미지는 포함하지 않는다.
노이즈 꼭지점은 같은 uniform RGB 1000장과 문자열500개/단어열500개로 매 에폭 재계산한다.
초기 고정/현재 기준 평면을 함께 분석한다. float 출력 3자리, 내부 계산 원 정밀도.
검색 rank는 동점 오답을 앞에 두는 보수적 규칙. 정답은 이미지에 연결된 모든 캡션.
학습·데이터 진입부는 기존 train.py와 같은 방식이며, checkpoint/cache/JSON을 저장하지 않는다.
필요 패키지: torch, torchvision, open_clip_torch, numpy, Pillow, matplotlib.
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

@dataclass(frozen=True)
class Config:
    root: Path = ROOT
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
    temporary = path.with_suffix('.png.tmp')
    fig.savefig(temporary, format='png', dpi=180)
    plt.close(fig)
    temporary.replace(path)


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


def h1_exp1_subspace_retrieval(snapshot, context, render=False):
    """H1-1. 축/평면과 잔차의 raw dot/cosine 검색; rank-matched random/PCA 대조."""
    extra = ['subspace', 'control_trial', 'subspace_rank', 'score_variant',
             'zero_image_rows', 'zero_text_rows', 'valid_score', 'score_reconstruction_max_error']
    table = context.table('h1_exp1_subspace_retrieval', extra + RETRIEVAL_COLUMNS)
    for mode, frame in context.frames(snapshot):
        meta = context.meta(snapshot, mode)
        metrics = retrieval_summary(query_stats(snapshot.scores, snapshot.mapping))
        table.append(dict(**meta, record_type='summary', subspace='full', control_trial=-1,
                          subspace_rank=snapshot.z.shape[1], score_variant='original',
                          zero_image_rows=0, zero_text_rows=0, valid_score=True,
                          score_reconstruction_max_error=0.000, **metrics))
        spaces = [(name, -1, q) for name, q in frame['spaces'].items()]
        # 각 실제 부분공간의 rank에 맞춘 대조군. 2차원/4차원을 섞어 비교하지 않는다.
        for rank in sorted({q.shape[1] for q in frame['spaces'].values()}):
            spaces.append(('pca_rank_matched', -1, frame['pca'][:, :rank]))
            spaces.extend(('random_rank_matched', trial, context.random_basis(snapshot.z.shape[1], rank, trial))
                          for trial in range(context.cfg.random_subspace_repeats))
        for name, trial, q in spaces:
            zi, ti = snapshot.z @ q, snapshot.t @ q
            plane = snapshot.scores.copy() if q.shape[1]==snapshot.z.shape[1] else zi @ ti.T
            outside = snapshot.scores - plane
            nz = np.linalg.norm(residual(snapshot.z, q), axis=1)
            nt = np.linalg.norm(residual(snapshot.t, q), axis=1)
            for variant, score in [('plane_raw', plane), ('residual_raw', outside),
                                   ('residual_cosine', None)]:
                valid = variant != 'residual_cosine' or (np.all(nz > EPS) and np.all(nt > EPS))
                if variant == 'residual_cosine' and valid:
                    score = outside / (nz[:, None]*nt[None, :])
                row = dict(**meta, record_type='summary', subspace=name, control_trial=trial,
                           subspace_rank=q.shape[1], score_variant=variant,
                           zero_image_rows=int((nz <= EPS).sum()), zero_text_rows=int((nt <= EPS).sum()),
                           valid_score=bool(valid), score_reconstruction_max_error=float(np.max(np.abs(plane+outside-snapshot.scores))))
                if valid:
                    row.update(retrieval_summary(query_stats(score, snapshot.mapping)))
                table.append(row)
    table.flush()
    if render:
        fig, axes = make_figure()
        for ax, direction in zip(axes[0], ('i2t','t2i')):
            joint_ranks={(r['run_id'],r['epoch']):r['subspace_rank'] for r in table.summaries
                         if r['reference_mode']=='epoch_current' and r['subspace']=='joint_cone'}
            final = [r for r in table.summaries if r['epoch'] == context.cfg.epochs and
                     r['reference_mode'] == 'epoch_current' and r['score_variant'] in ('original','residual_cosine') and
                     r['subspace'] in ('full','joint_cone','random_rank_matched','pca_rank_matched') and r['valid_score'] and
                     (r['subspace'] in ('full','joint_cone') or r['subspace_rank']==joint_ranks[(r['run_id'],r['epoch'])])]
            names = list(dict.fromkeys(label(r) for r in final))
            variants = ['full','joint_cone','random_rank_matched','pca_rank_matched']
            width = 0.8/max(len(names),1)
            for j, run in enumerate(names):
                means = [np.mean([r[f'{direction}_R1_percent'] for r in final if label(r)==run and r['subspace']==v]) for v in variants]
                ax.bar(np.arange(4)-0.4+width/2+j*width, means, width, label=run)
            ax.set_xticks(range(4), ['Original','Cone residual','Random residual','PCA residual'], rotation=15)
            ax.set(title=f'{direction.upper()}: final R@1', ylabel='Recall (%)')
            if names: ax.legend(fontsize=7)
        predicate = lambda r: r['reference_mode']=='epoch_current' and r['subspace']=='joint_cone' and r['score_variant']=='residual_raw'
        plot_curves(axes[1,0],table.summaries,'i2t_R1_percent','I2T: raw residual',predicate)
        plot_curves(axes[1,1],table.summaries,'t2i_R1_percent','T2I: raw residual',predicate)
        fig.suptitle('H1-1: retrieval in the cone subspace and its complement')
        save_figure(fig, table.png)
    return table.path, table.png


def pair_terms(z, t, image_indices, text_indices, frame):
    zi, ti = z[image_indices], t[text_indices]
    f = frame['image']
    delta = zi-f['mu']
    m = ti @ f['mu']
    v = (delta @ f['b'])*(ti @ f['b'])
    a = (delta @ f['e'])*(ti @ f['e'])
    # U를 독립적으로 계산해서 reconstruction을 검사한다.
    q = frame['spaces']['image_cone']
    u = np.sum(residual(delta, q)*residual(ti, q), axis=1)
    direct = np.sum(zi*ti,axis=1)
    error = np.abs(direct-m-v-a-u)
    if error.max() > 0.00001:
        raise AssertionError(f'네 항 재구성 오류: {error.max():.3f}')
    return dict(M=m, V=v, A_e=a, U=u, full=direct, reconstruction_error=error)


def selected_pair_margins(snapshot, query, direction, frame):
    rows = np.arange(len(query['positive']))
    if direction == 'i2t':
        p = pair_terms(snapshot.z,snapshot.t,rows,query['positive'],frame)
        n = pair_terms(snapshot.z,snapshot.t,rows,query['negative'],frame)
    else:
        p = pair_terms(snapshot.z,snapshot.t,query['positive'],rows,frame)
        n = pair_terms(snapshot.z,snapshot.t,query['negative'],rows,frame)
    return {key: p[key]-n[key] for key in ('M','V','A_e','U','full')}


def h1_exp2_margin_decomposition(snapshot, context, render=False):
    """H1-2. 현재 hard negative와 초기 고정 후보의 M/V/A_e/U 마진 및 변화."""
    terms = ('M','V','A_e','U','full')
    extra = ['direction','candidate_policy','status','n_queries','query_id','positive_candidate_id',
             'negative_candidate_id','initial_correct','current_correct','initial_rank','current_rank',
             'reconstruction_max_error','U_full_margin_pearson','matched_clip_rank','matched_clip_correct',
             'status_vs_clip_only','matched_clip_margin_available','matched_clip_full_margin',
             'matched_clip_U_margin','delta_full_vs_clip_only','delta_U_vs_clip_only']
    extra += [f'{term}_margin' for term in terms]+[f'delta_{term}_margin' for term in terms]
    table = context.table('h1_exp2_margin_decomposition', extra)
    current = query_stats(snapshot.scores,snapshot.mapping)
    control_key=snapshot.metadata['seed'],snapshot.metadata['epoch']
    is_control=snapshot.metadata['method']=='clip_only'
    if is_control:
        context.clip_queries[control_key]=current
    control_queries=context.clip_queries.get(control_key)
    for mode, frame in context.frames(snapshot):
        for direction in ('i2t','t2i'):
            oldq,newq = context.base_query[direction],current[direction]
            initial_correct,current_correct = oldq['ranks']==1,newq['ranks']==1
            status = np.where(initial_correct,np.where(current_correct,'retained','lost'),np.where(current_correct,'gained','still_wrong'))
            clipq=control_queries[direction] if control_queries is not None else None
            status_clip=(np.where(clipq['ranks']==1,np.where(current_correct,'retained_vs_clip','lost_vs_clip'),
                                 np.where(current_correct,'gained_vs_clip','still_wrong_vs_clip')) if clipq is not None else None)
            policies=['current_hard_negative','initial_fixed_candidates']
            if clipq is not None:policies.append('clip_only_fixed_candidates')
            for policy in policies:
                chosen = newq if policy=='current_hard_negative' else oldq if policy=='initial_fixed_candidates' else clipq
                margins = selected_pair_margins(snapshot,chosen,direction,frame)
                old = selected_pair_margins(context.initial,chosen,direction,context.initial.frame)
                key=(*control_key,mode,direction,'initial_fixed_candidates' if policy=='initial_fixed_candidates' else 'current_hard_negative')
                if is_control:
                    context.clip_margins[key]=margins
                # 자체 current-hard 후보는 CLIP 대조군의 후보와 달라 직접 차감하지 않는다.
                clip_margins=context.clip_margins.get(key) if policy!='current_hard_negative' or is_control else None
                error = float(np.max(np.abs(margins['full']-sum(margins[k] for k in terms[:-1]))))
                if direction=='t2i' and np.max(np.abs(margins['M'])) > EPS:
                    raise AssertionError('텍스트 고정 검색에서 M 상쇄가 실패했습니다.')
                meta = dict(**context.meta(snapshot,mode),direction=direction,candidate_policy=policy)
                groups=['all','retained','lost','gained','still_wrong']
                if status_clip is not None:groups+=['retained_vs_clip','lost_vs_clip','gained_vs_clip','still_wrong_vs_clip']
                for group in groups:
                    mask = np.ones(len(status),bool) if group=='all' else status_clip==group if group.endswith('_vs_clip') else status==group
                    if not mask.any(): continue
                    row = dict(**meta,record_type='summary',status=group,n_queries=int(mask.sum()),
                               reconstruction_max_error=error,U_full_margin_pearson=weighted_pearson(margins['U'][mask],margins['full'][mask]))
                    for term in terms:
                        row[f'{term}_margin']=float(margins[term][mask].mean())
                        row[f'delta_{term}_margin']=float((margins[term]-old[term])[mask].mean())
                    row['matched_clip_margin_available']=clip_margins is not None
                    if clip_margins is not None:
                        row.update(matched_clip_full_margin=float(clip_margins['full'][mask].mean()),
                                   matched_clip_U_margin=float(clip_margins['U'][mask].mean()),
                                   delta_full_vs_clip_only=float((margins['full']-clip_margins['full'])[mask].mean()),
                                   delta_U_vs_clip_only=float((margins['U']-clip_margins['U'])[mask].mean()))
                    table.append(row)
                if snapshot.metadata['epoch'] in context.cfg.query_detail_epochs:
                    query_ids = snapshot.image_ids if direction=='i2t' else snapshot.caption_ids
                    candidate_ids = snapshot.caption_ids if direction=='i2t' else snapshot.image_ids
                    for j, query_id in enumerate(query_ids):
                        row=dict(**meta,record_type='query',status=status[j],n_queries=1,query_id=int(query_id),
                                 positive_candidate_id=int(candidate_ids[chosen['positive'][j]]),
                                 negative_candidate_id=int(candidate_ids[chosen['negative'][j]]),
                                 initial_correct=bool(initial_correct[j]),current_correct=bool(current_correct[j]),
                                 initial_rank=int(oldq['ranks'][j]),current_rank=int(newq['ranks'][j]),reconstruction_max_error=error)
                        for term in terms:
                            row[f'{term}_margin']=float(margins[term][j])
                            row[f'delta_{term}_margin']=float(margins[term][j]-old[term][j])
                        row['matched_clip_margin_available']=clip_margins is not None
                        if clipq is not None:
                            row.update(matched_clip_rank=int(clipq['ranks'][j]),matched_clip_correct=bool(clipq['ranks'][j]==1),
                                       status_vs_clip_only=status_clip[j])
                        if clip_margins is not None:
                            row.update(matched_clip_full_margin=float(clip_margins['full'][j]),matched_clip_U_margin=float(clip_margins['U'][j]),
                                       delta_full_vs_clip_only=float(margins['full'][j]-clip_margins['full'][j]),
                                       delta_U_vs_clip_only=float(margins['U'][j]-clip_margins['U'][j]))
                        table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,direction in enumerate(('i2t','t2i')):
            predicate=lambda r,d=direction:r['reference_mode']=='initial_fixed' and r['direction']==d and r['candidate_policy']=='initial_fixed_candidates' and r['status']=='all'
            plot_curves(axes[0,j],table.summaries,'delta_U_margin',f'{direction.upper()}: residual margin change',predicate)
            predicate=lambda r,d=direction:r['reference_mode']=='initial_fixed' and r['direction']==d and r['candidate_policy']=='clip_only_fixed_candidates' and r['status']=='lost_vs_clip'
            plot_curves(axes[1,j],table.summaries,'delta_U_vs_clip_only',f'{direction.upper()}: lost versus matched CLIP',predicate)
            axes[0,j].axhline(0,color='black',linewidth=0.8)
            axes[1,j].axhline(0,color='black',linewidth=0.8)
        fig.suptitle('H1-2: separate changing competitors from fixed-pair deformation')
        save_figure(fig,table.png)
    return table.path,table.png


def coordinate(x,frame,kind):
    c=x @ frame['r']
    if kind=='c': return c
    if kind=='C': return frame['R']*c
    return ((x-frame['b']) @ frame['a'])/frame['ell']


def h1_exp3_pair_coordinates(snapshot, context, render=False):
    """H1-3. 실제 정답 쌍의 C/c/h 대응과 이미지 단위 permutation 대조."""
    extra=['coordinate','n_pairs','weighting','paired_pearson','paired_mse','shuffled_pearson_mean',
           'shuffled_mse_mean','shuffled_mse_std','shuffle_advantage','image_coordinate_mean',
           'text_coordinate_mean','image_coordinate_std','text_coordinate_std']
    table=context.table('h1_exp3_pair_coordinates',extra)
    n=len(snapshot.z)
    # 캡션 수가 다른 이미지도 같은 총 가중치를 갖는다.
    weights=1/np.bincount(snapshot.mapping,minlength=n)[snapshot.mapping]
    weights=weights/weights.sum()
    for mode,frame in context.frames(snapshot):
        for kind in ('C','c','h'):
            xi=coordinate(snapshot.z,frame['image'],kind)
            xt=coordinate(snapshot.t,frame['text'],kind)
            paired=xi[snapshot.mapping]
            mse=float(weights @ ((paired-xt)**2))
            shuffled_mse=[];shuffled_corr=[]
            for trial in range(context.cfg.pair_shuffle_repeats):
                rng=np.random.default_rng(np.random.SeedSequence([context.cfg.control_seed,3101,trial]))
                perm=rng.permutation(n)
                shuffled=xi[perm[snapshot.mapping]]
                shuffled_mse.append(float(weights @ ((shuffled-xt)**2)))
                shuffled_corr.append(weighted_pearson(shuffled,xt,weights))
            xmean,ymean=float(weights @ paired),float(weights @ xt)
            row=dict(**context.meta(snapshot,mode),record_type='summary',coordinate=kind,n_pairs=len(xt),
                     weighting='equal_image_total_weight',paired_pearson=weighted_pearson(paired,xt,weights),paired_mse=mse,
                     shuffled_pearson_mean=float(np.nanmean(shuffled_corr)) if np.isfinite(shuffled_corr).any() else float('nan'),
                     shuffled_mse_mean=float(np.mean(shuffled_mse)),shuffled_mse_std=float(np.std(shuffled_mse,ddof=1)) if len(shuffled_mse)>1 else 0.000,
                     shuffle_advantage=float(np.mean(shuffled_mse)-mse),image_coordinate_mean=xmean,text_coordinate_mean=ymean,
                     image_coordinate_std=float(np.sqrt(weights @ ((paired-xmean)**2))),
                     text_coordinate_std=float(np.sqrt(weights @ ((xt-ymean)**2))))
            table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,kind in enumerate(('c','h')):
            predicate=lambda r,k=kind:r['reference_mode']=='epoch_current' and r['coordinate']==k
            plot_curves(axes[0,j],table.summaries,'paired_pearson',f'{kind}: positive-pair correlation',predicate)
            plot_curves(axes[1,j],table.summaries,'shuffle_advantage',f'{kind}: shuffled MSE - paired MSE',predicate)
            axes[1,j].axhline(0,color='black',linewidth=0.8)
        fig.suptitle('H1-3: coordinate matching versus image-level shuffled correspondence')
        save_figure(fig,table.png)
    return table.path,table.png


def h1_exp4_residual_structure(snapshot, context, render=False):
    """H1-4. 기하/평면·잔차 분산/중심화 effective rank/초기 잔차 이웃 보존."""
    structural=['modality','joint_subspace_rank','total_variance','plane_variance','residual_variance',
                'plane_variance_fraction','axis_parallel_variance','axis_perpendicular_variance',
                'centered_effective_rank','centered_participation_ratio','residual_effective_rank',
                'residual_participation_ratio','residual_direction_cosine_to_initial',
                'residual_neighbor_overlap_to_initial','neighbor_k','i2t_failure_fraction','t2i_failure_fraction']
    table=context.table('h1_exp4_residual_structure',structural+GEOMETRY_COLUMNS+RETRIEVAL_COLUMNS)
    qstats=query_stats(snapshot.scores,snapshot.mapping)
    metrics=retrieval_summary(qstats)
    geom=geometry(snapshot.z,snapshot.t,snapshot.frame)
    pooled_t=text_per_image(snapshot.t,snapshot.mapping,len(snapshot.z))
    baseline_t=text_per_image(context.initial.t,context.initial.mapping,len(context.initial.z))
    for mode,frame in context.frames(snapshot):
        q=frame['spaces']['joint_cone']
        for modality,x,base,neighbors_x,neighbors_base in (
            ('image',snapshot.z,context.initial.z,snapshot.z,context.initial.z),
            ('text',snapshot.t,context.initial.t,pooled_t,baseline_t)):
            cov=covariance(x)
            projection=q @ q.T
            residual_cov=cov-projection @ cov-cov @ projection+projection @ cov @ projection
            var=float(np.trace(cov));plane_var=float(np.trace(q.T @ cov @ q))
            spectral=spectral_stats(cov);rspectral=spectral_stats(residual_cov)
            a=frame[modality]['a'];parallel=float(a @ cov @ a)
            # 잔차 방향 변화는 두 시점에 동일한 초기 기준 평면을 적용해 측정한다.
            fixed_q=context.initial.frame['spaces']['joint_cone']
            cur_fixed=unit(residual(x,fixed_q),allow_zero=True)
            base_fixed=unit(residual(base,fixed_q),allow_zero=True)
            good=(np.linalg.norm(cur_fixed,axis=1)>EPS)&(np.linalg.norm(base_fixed,axis=1)>EPS)
            drift=float(np.sum(cur_fixed[good]*base_fixed[good],axis=1).mean()) if good.any() else float('nan')
            # 현재 평면을 쓰는 이웃 비교도 두 시점에 같은 평면을 적용한다.
            nn=nearest_neighbors(residual(neighbors_x,q),context.cfg.neighbor_k)
            original_nn=nearest_neighbors(residual(neighbors_base,q),context.cfg.neighbor_k)
            row=dict(**context.meta(snapshot,mode),record_type='summary',modality=modality,
                     joint_subspace_rank=q.shape[1],total_variance=var,plane_variance=plane_var,
                     residual_variance=float(np.trace(residual_cov)),plane_variance_fraction=plane_var/var if var>EPS else float('nan'),
                     axis_parallel_variance=parallel,axis_perpendicular_variance=var-parallel,
                     centered_effective_rank=spectral['effective_rank'],centered_participation_ratio=spectral['participation_ratio'],
                     residual_effective_rank=rspectral['effective_rank'],residual_participation_ratio=rspectral['participation_ratio'],
                     residual_direction_cosine_to_initial=drift,residual_neighbor_overlap_to_initial=neighbor_overlap(nn,original_nn),
                     neighbor_k=min(context.cfg.neighbor_k,len(snapshot.z)-1),
                     i2t_failure_fraction=float(np.mean(qstats['i2t']['ranks']>1)),
                     t2i_failure_fraction=float(np.mean(qstats['t2i']['ranks']>1)),**geom,**metrics)
            if abs(var-(1-np.linalg.norm(x.mean(0))**2)) > 0.00001:
                raise AssertionError('단위 임베딩의 분산 항등식이 실패했습니다.')
            table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,modality in enumerate(('image','text')):
            predicate=lambda r,m=modality:r['reference_mode']=='initial_fixed' and r['modality']==m
            plot_curves(axes[0,j],table.summaries,'residual_effective_rank',f'{modality}: centered residual effective rank',predicate)
            plot_curves(axes[1,j],table.summaries,'residual_neighbor_overlap_to_initial',f'{modality}: residual neighbor retention',predicate)
            axes[1,j].set_ylim(0,1.050)
        fig.suptitle('H1-4: preserve the initial reference when measuring residual deformation')
        save_figure(fig,table.png)
    return table.path,table.png


# ------------------------- 공유 학습/실행 -------------------------
EXPERIMENTS = (h1_exp1_subspace_retrieval, h1_exp2_margin_decomposition,
               h1_exp3_pair_coordinates, h1_exp4_residual_structure)


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
    for experiment in EXPERIMENTS:
        experiment(baseline,context,render=(cfg.epochs==0))
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
        for experiment in EXPERIMENTS:
            experiment(snapshot,context,render=(epoch==cfg.epochs))
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


def main(config=CONFIG,model_factory=None):
    """CLI 없이 모든 h1 실험을 실행한다. model_factory는 로컬 검증용 선택 인자."""
    validate_config(config)
    require_dependencies()
    device=torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train,evaluation,_,protocol=load_splits(config)
    if len(evaluation)<2:
        raise ValueError('검색 평가에는 이미지가 두 장 이상 필요합니다.')
    output=config.output_directory or config.root/'results'/'h1_residual'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    seed_everything(config.seeds[0])
    if model_factory is None:
        model,preprocess_train,preprocess_eval=open_clip.create_model_and_transforms(
            config.model,pretrained=config.pretrained,precision='fp32',cache_dir=str(config.root / 'ckpt'))
        tokenizer=open_clip.get_tokenizer(config.model)
    else:
        model,preprocess_train,preprocess_eval,tokenizer=model_factory(config)
    model=model.to(device).float()
    model.requires_grad_(True)
    if config.gradient_checkpointing:
        model.set_grad_checkpointing(True)
    initial_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    initial=encode_snapshot(model,train,evaluation,preprocess_eval,tokenizer,config,device,
                            metadata_for(config,train,evaluation,config.seeds[0],'clip_only',0.000,0))
    context=AnalysisContext(config,output,initial)
    try:
        print(f'{protocol}; train={len(train)} eval={len(evaluation)} '
              f'runs={len(variants(config))*len(config.seeds)}; output={output}',flush=True)
        for seed in config.seeds:
            for method,weight in variants(config):
                run_training(model,initial_state,initial,context,train,evaluation,preprocess_train,
                             preprocess_eval,tokenizer,config,device,seed,method,weight)
    finally:
        context.close()
    expected={f'{experiment.__name__}.{suffix}' for experiment in EXPERIMENTS for suffix in ('csv','png')}
    actual={p.name for p in output.iterdir()}
    if actual != expected:
        raise AssertionError(f'예상 출력 파일 수와 다릅니다: {sorted(actual)}')
    print(f'완료: {output} (실험별 CSV/PNG 각 하나)',flush=True)
    return output


if __name__ == '__main__':
    main()

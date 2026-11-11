#!/usr/bin/env python3
"""Conformity 축 정합 vs Imageness–Textness 축 정합. CONFIG 수정 후 실행.

1. Uniform RGB 이미지 1000장, 기존 analysis.py 방식의 텍스트 노이즈
   1000개(문자열 500 + 단어열 500)로 각 꼭지점 b_I, b_T를 추정한다.
2. eda.py 캐시의 전체 이미지/전체 캡션을 각각 정규화한 뒤 평균 내고
   다시 정규화해 r_I, r_T를 구한다. a_M = normalize(r_M - b_M).
3. r_I dot r_T와 a_I dot a_T를 비교한다. 축 방향은 O→r, b→r다.

사전학습 가중치는 eval/inference_mode로 사용한다.
캐시가 있으면 numpy와 matplotlib만 필요하다.
누락된 노이즈는 open_clip_torch로 한 번 추론해 캐시한다.
전체셋 캐시 기본 경로: <root>/cache/ViT-B-32_openai.npz (eda.py).
실행: python modality_gap_compare.py
출력: <root>/results/axis_alignment/<UTC timestamp>/
CSV/JSON/그림은 소수점 세 자리, axes_full_precision.npz는 원 정밀도.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())
import csv
import hashlib
import importlib.metadata
import json
import random

import numpy as np


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    model: str = 'ViT-B-32'
    pretrained: str = 'openai'
    seed: int = 0
    n_noise: int = 1000
    n_text_noise_each: int = 500
    batch_size: int = 256
    device: str = 'auto'  # auto / cpu / cuda
    clean_cache: Path | None = None
    image_noise_cache: Path | None = None
    text_noise_cache: Path | None = None
    encode_missing_noise: bool = True
    make_figures: bool = True


CONFIG = Config()
EPS = 1e-12
WORD_POOL = ('apple neutron cactus flicker prism quartz wagon myth oxygen cobalt delta ember frost glyph helix '
             'ivory jungle karma lemon mosaic nebula orchid pulse ripple sigma thorn umbra vortex wax zenith '
             'blaze cipher dusk echo flux grove haze iris jade knot latch mirth nook opal plume reef sable tide '
             'umber vale whisk yarn zinc abyss bolt crux dew elm fang gust husk ion jolt keel lobe musk nib ore '
             'pith quill rind sap tuft urge vim wren yew zeal arc brim cleft').split()
CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-+=/\\@#$%&*()[]{}~'


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2) or x.size == 0 or not np.isfinite(x).all():
        raise ValueError('임베딩은 유한한 [D] 또는 [N,D] 배열이어야 합니다.')
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= EPS):
        raise ValueError('영벡터 또는 방향을 추정할 수 없는 평균벡터입니다.')
    return x / norm


def unit_mean(raw):
    mu = normalize(raw).mean(axis=0)
    return mu, normalize(mu), float(np.linalg.norm(mu))


def cosine(a, b):
    return float(np.clip(np.dot(normalize(a), normalize(b)), -1, 1))


def angle(a, b):
    return float(np.degrees(np.arccos(cosine(a, b))))


def array_hash(x):
    x = np.ascontiguousarray(x)
    digest = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    digest.update(memoryview(x).cast('B'))
    return digest.hexdigest()


def json_value(x):
    if isinstance(x, Path):
        return json.dumps(str(x), ensure_ascii=False)
    if isinstance(x, dict):
        return '{' + ', '.join(json.dumps(str(k), ensure_ascii=False) + ': ' + json_value(v)
                               for k, v in x.items()) + '}'
    if isinstance(x, (tuple, list, np.ndarray)):
        return '[' + ', '.join(json_value(v) for v in x) + ']'
    if isinstance(x, (float, np.floating)):
        return f'{x:.3f}' if np.isfinite(x) else 'null'
    if isinstance(x, np.generic):
        return json.dumps(x.item())
    return json.dumps(x, ensure_ascii=False)


def save_json(path, value):
    Path(path).write_text(json_value(value) + '\n', encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f'{v:.3f}' if isinstance(v, (float, np.floating)) else v
                             for k, v in row.items()})


def text_noise_strings(seed, n_each):
    """analysis.py Context.text_noise_strings와 같은 RNG, 문자/단어 풀, 순서."""
    rng = random.Random(1000 + seed)
    chars = [''.join(rng.choice(CHARSET) for _ in range(rng.randint(20, 75))) for _ in range(n_each)]
    words = [' '.join(rng.choice(WORD_POOL) for _ in range(rng.randint(2, 15))) for _ in range(n_each)]
    return chars, words


def uniform_noise_pixels(seed, n, hw):
    """analysis.py image_noise의 uniform 분기와 동일한 샘플 순서."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1101]))
    for _ in range(n):
        yield rng.uniform(0, 1, (*hw, 3)).astype(np.float32)


def load_clean(cfg):
    path = cfg.clean_cache or cfg.root / 'cache' / f'{cfg.model}_{cfg.pretrained}.npz'
    with np.load(path, allow_pickle=False) as data:
        image, text = data['img'].copy(), data['txt'].copy()
        ids, annotation_ids = data['image_ids'].copy(), data['txt_ann_ids'].copy()
        mapping = data['txt_img_row'].copy()
    if image.ndim != 2 or text.ndim != 2 or image.shape[1] != text.shape[1]:
        raise ValueError('이미지/텍스트 캐시는 공통 차원을 가진 [N,D] 배열이어야 합니다.')
    if len(image) < 2 or len(text) < 2 or len(ids) != len(image) or len(annotation_ids) != len(text):
        raise ValueError('임베딩과 ID 개수를 확인하세요.')
    if len(mapping) != len(text) or mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError('txt_img_row는 캡션 수와 같은 길이의 정수 배열이어야 합니다.')
    if np.any(mapping < 0) or np.any(mapping >= len(image)):
        raise ValueError('txt_img_row가 이미지 범위를 벗어났습니다.')
    if np.any(np.bincount(mapping, minlength=len(image)) == 0):
        raise ValueError('전체셋 캐시에 캡션이 없는 이미지가 있습니다.')
    if ids.ndim != 1 or annotation_ids.ndim != 1 or len(np.unique(ids)) != len(ids) or len(np.unique(annotation_ids)) != len(annotation_ids):
        raise ValueError('이미지/캡션 ID는 중복 없는 1차원 배열이어야 합니다.')
    normalize(image); normalize(text)
    # 원본 출력과 이미 단위 정규화된 출력 모두 같은 평균 방향 계산을 지원한다.
    return dict(image=image, text=text, image_ids=ids, annotation_ids=annotation_ids, path=path)


def read_noise_cache(path, cfg, modality):
    with np.load(path, allow_pickle=False) as data:
        if 'raw' not in data or 'manifest' not in data:
            raise ValueError(f'raw/manifest가 없는 노이즈 캐시: {path}')
        metadata = json.loads(str(data['manifest'].item()))
        raw = data['raw'].copy()
    count = cfg.n_noise if modality == 'image' else 2*cfg.n_text_noise_each
    encoder = metadata.get('encoder', {})
    matches = (encoder.get('model') == cfg.model and encoder.get('pretrained') == cfg.pretrained
               and metadata.get('seed') == cfg.seed and metadata.get('n') == count)
    if modality == 'image':
        matches = matches and metadata.get('kind') == 'noise' and metadata.get('family') == 'uniform'
    else:
        matches = (matches and metadata.get('kind') == 'text_noise'
                   and metadata.get('char_len', [20, 75]) == [20, 75]
                   and metadata.get('word_len', [2, 15]) == [2, 15])
    if not matches:
        return None
    if raw.ndim != 2 or len(raw) != count:
        raise ValueError(f'노이즈 캐시 표본 수/형태가 잘못됐습니다: {path}')
    normalize(raw)
    return dict(raw=raw, path=path, manifest=metadata)


def find_noise_cache(cfg, modality):
    explicit = cfg.image_noise_cache if modality == 'image' else cfg.text_noise_cache
    if explicit is not None:
        result = read_noise_cache(explicit, cfg, modality)
        if result is None:
            raise ValueError(f'지정한 노이즈 캐시의 모델/seed/개수가 CONFIG와 다릅니다: {explicit}')
        return result
    stem = f'noise_uniform_{cfg.seed}' if modality == 'image' else f'txtnoise_clip_{cfg.seed}'
    found = []
    for directory in ('axis_alignment', 'conformity_imageness', 'oblique_cone', 'modality_gap_comparison'):
        for path in sorted((cfg.root / 'cache' / directory).glob(stem + '_*.npz')):
            result = read_noise_cache(path, cfg, modality)
            if result is not None:
                found.append(result)
    if not found:
        return None
    first = normalize(found[0]['raw'])
    if any(first.shape != item['raw'].shape or not np.allclose(first, normalize(item['raw']), rtol=0.00001, atol=0.000001)
           for item in found[1:]):
        field = f'{modality}_noise_cache'
        raise ValueError(f'서로 다른 {modality} 노이즈 캐시가 중복됩니다. CONFIG.{field}에 '
                         f'사용할 파일을 지정하세요: {[str(x["path"]) for x in found]}')
    return found[0]


class FrozenCLIP:
    def __init__(self, cfg):
        import torch
        import open_clip
        self.torch, self.batch_size = torch, cfg.batch_size
        if cfg.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA를 사용할 수 없습니다.')
        self.device = 'cuda' if cfg.device != 'cpu' and torch.cuda.is_available() else 'cpu'
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model, pretrained=cfg.pretrained, precision='fp32', cache_dir=str(cfg.root / 'ckpt'))
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(cfg.model)
        size = self.model.visual.image_size
        self.hw = (size, size) if isinstance(size, int) else tuple(size)
        norm = next((x for x in reversed(preprocess.transforms) if hasattr(x, 'mean') and hasattr(x, 'std')), None)
        if norm is None:
            raise ValueError('이미지 Normalize 전처리를 찾을 수 없습니다.')
        self.mean = np.asarray(norm.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(norm.std, dtype=np.float32).reshape(1, 1, 3)
        self.signature = dict(model=cfg.model, pretrained=cfg.pretrained,
                              torch=importlib.metadata.version('torch'),
                              open_clip=importlib.metadata.version('open_clip_torch'),
                              size=list(self.hw), mean=self.mean.ravel().tolist(), std=self.std.ravel().tolist())

    def encode_pixels(self, pixels):
        parts, batch = [], []
        def flush():
            if batch:
                values = (np.stack(batch)-self.mean)/self.std
                tensor = self.torch.from_numpy(values.transpose(0, 3, 1, 2).copy()).to(self.device)
                with self.torch.inference_mode():
                    parts.append(self.model.encode_image(tensor, normalize=False).float().cpu().numpy())
                batch.clear()
        for pixels_one in pixels:
            if pixels_one.shape != (*self.hw, 3):
                raise ValueError('노이즈 이미지 해상도가 모델 입력과 다릅니다.')
            batch.append(pixels_one)
            if len(batch) == self.batch_size:
                flush()
        flush()
        return np.concatenate(parts)

    def encode_texts(self, strings):
        parts = []
        for first in range(0, len(strings), self.batch_size*4):
            tokens = self.tokenizer(strings[first:first+self.batch_size*4]).to(self.device)
            with self.torch.inference_mode():
                parts.append(self.model.encode_text(tokens, normalize=False).float().cpu().numpy())
        return np.concatenate(parts)


def load_noise(cfg, encoder=None):
    noise = {modality: find_noise_cache(cfg, modality) for modality in ('image', 'text')}
    missing = [modality for modality, cached in noise.items() if cached is None]
    if missing:
        if not cfg.encode_missing_noise:
            raise FileNotFoundError(f'노이즈 캐시가 필요합니다: {missing}')
        encoder = encoder or FrozenCLIP(cfg)
        directory = cfg.root / 'cache/axis_alignment'
        directory.mkdir(parents=True, exist_ok=True)
        for modality in missing:
            print(f'{modality} 노이즈를 고정된 사전학습 인코더로 추론합니다.', flush=True)
            if modality == 'image':
                raw = encoder.encode_pixels(uniform_noise_pixels(cfg.seed, cfg.n_noise, encoder.hw))
                stem = f'noise_uniform_{cfg.seed}'
                metadata = dict(kind='noise', family='uniform', seed=cfg.seed, n=cfg.n_noise,
                                pixel_range=[0, 1], generator='analysis.py:SeedSequence([seed,1101])')
            else:
                chars, words = text_noise_strings(cfg.seed, cfg.n_text_noise_each)
                raw = encoder.encode_texts(chars+words)
                stem = f'txtnoise_clip_{cfg.seed}'
                metadata = dict(kind='text_noise', seed=cfg.seed, n=len(chars)+len(words),
                                char_len=[20, 75], word_len=[2, 15], generator='analysis.py:Random(1000+seed)')
            metadata['encoder'] = encoder.signature
            encoded = json.dumps(metadata, sort_keys=True)
            token = hashlib.sha256(encoded.encode()).hexdigest()[:20]
            path = directory / f'{stem}_{token}.npz'
            np.savez_compressed(path, raw=raw, manifest=np.array(encoded))
            noise[modality] = read_noise_cache(path, cfg, modality)
    return noise


def exp1_vertices(noise, out):
    """단위 노이즈 임베딩의 평균을 다시 정규화해 b_I, b_T를 추정."""
    result, rows = {}, []
    for modality in ('image', 'text'):
        mu, b, Rb = unit_mean(noise[modality]['raw'])
        result[modality] = dict(b=b, noise_mean=mu, Rb=Rb)
        rows.append(dict(modality=modality, n_noise=len(noise[modality]['raw']), Rb=Rb))
    save_csv(out / 'exp1_vertices.csv', rows)
    return result


def exp2_axes(clean, vertices, out):
    """전체 이미지와 전체 캡션으로 r와 b→r 방향을 추정."""
    result, rows = {}, []
    for modality in ('image', 'text'):
        mu, r, R = unit_mean(clean[modality])
        b = vertices[modality]['b']
        a = normalize(r-b)
        result[modality] = dict(**vertices[modality], mean=mu, r=r, a=a, R=R)
        rows.append(dict(modality=modality, n_clean=len(clean[modality]), R=R,
                         Rb=vertices[modality]['Rb'], cos_b_r=cosine(b, r),
                         angle_b_r_deg=angle(b, r), noise_axis_length=float(np.linalg.norm(r-b)),
                         angle_conformity_noise_axis_deg=angle(r, a)))
    save_csv(out / 'exp2_axes.csv', rows)
    return result


def exp3_axis_alignment(axes, out):
    """동일 공간에서 두 directed 축의 cosine과 각도를 직접 비교."""
    image, text = axes['image'], axes['text']
    c_cos, h_cos = cosine(image['r'], text['r']), cosine(image['a'], text['a'])
    c_angle, h_angle = angle(image['r'], text['r']), angle(image['a'], text['a'])
    rows = [dict(axes='conformity_image_text', cosine=c_cos, angle_deg=c_angle),
            dict(axes='imageness_textness', cosine=h_cos, angle_deg=h_angle)]
    save_csv(out / 'exp3_axis_alignment.csv', rows)
    summary = dict(conformity_cosine=c_cos, imageness_textness_cosine=h_cos,
                   delta_cosine=h_cos-c_cos, conformity_angle_deg=c_angle,
                   imageness_textness_angle_deg=h_angle, angle_reduction_deg=c_angle-h_angle)
    save_json(out / 'summary.json', summary)
    vectors = {f'{name}_{modality}': value for modality, record in axes.items()
               for name, value in record.items()}
    np.savez_compressed(out / 'axes_full_precision.npz', **vectors, **summary)
    return summary


def make_figure(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix',
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    labels = ['Conformity', 'Imageness–Textness']
    colours = ['#2563A6', '#D56A24']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, values, title, bounds in (
        (axes[0], [summary['conformity_cosine'], summary['imageness_textness_cosine']],
         'Axis cosine (higher = closer directions)', (-1, 1)),
        (axes[1], [summary['conformity_angle_deg'], summary['imageness_textness_angle_deg']],
         'Axis angle in degrees (lower = closer directions)', (0, 180))):
        bars = ax.bar(labels, values, color=colours, width=0.5)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4)
        ax.set_ylim(*bounds)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_title(title, fontsize=10)
    axes[0].axhline(0, color='#999999', linewidth=0.6)
    fig.suptitle('Full-set image–text axis alignment', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'axis_alignment.png', dpi=220)
    fig.savefig(out / 'axis_alignment.pdf')
    plt.close(fig)


def save_report(summary, clean, cfg, out):
    text = f'''# Conformity와 Imageness–Textness 축 정합

사전학습 {cfg.model} ({cfg.pretrained})의 공통 임베딩 공간에서 축 방향을 비교했다.
전체 이미지 {len(clean['image'])}개와 전체 캡션 {len(clean['text'])}개로 평균 방향을 추정했다.
노이즈는 seed {cfg.seed}의 균일 RGB 이미지 {cfg.n_noise}장과 기존 코드 방식의
무작위 문자열 {cfg.n_text_noise_each}개 + 무작위 단어열 {cfg.n_text_noise_each}개다.

개별 임베딩을 단위 정규화한 후 평균 내고 다시 정규화해 $r_I,r_T,b_I,b_T$를 구했다.
Conformity 방향은 $r_I,r_T$이고, Imageness/Textness 방향은
$a_I=(r_I-b_I)/\\|r_I-b_I\\|$, $a_T=(r_T-b_T)/\\|r_T-b_T\\|$다.

| 축 비교 | 코사인 | 각도 (도) |
|---|---:|---:|
| Conformity 이미지–텍스트 | {summary['conformity_cosine']:.3f} | {summary['conformity_angle_deg']:.3f} |
| Imageness–Textness | {summary['imageness_textness_cosine']:.3f} | {summary['imageness_textness_angle_deg']:.3f} |

코사인 차이(Imageness–Textness − Conformity): **{summary['delta_cosine']:.3f}**.
각도 감소(Conformity − Imageness–Textness): **{summary['angle_reduction_deg']:.3f}**도.
양수이면 노이즈 꼭지점을 기준으로 한 축이 더 가깝게 정렬된 결과다.

그림의 왼쪽은 축 방향 코사인, 오른쪽은 축 사이 각도다.
같은 공간에서 추정한 실제 축 벡터를 직접 비교한 수치다.
원 정밀도의 벡터와 측정값은 `axes_full_precision.npz`에 저장했다.
'''
    (out / 'report.md').write_text(text, encoding='utf-8')


def main(config=CONFIG, encoder=None):
    """세 실험을 모두 실행. encoder 인자는 테스트용 인코더 주입을 지원."""
    if config.seed < 0 or min(config.n_noise, config.n_text_noise_each, config.batch_size) < 1:
        raise ValueError('seed와 노이즈 수/batch_size를 확인하세요.')
    if config.device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device는 auto/cpu/cuda 중 하나여야 합니다.')
    clean = load_clean(config)
    noise = load_noise(config, encoder)
    if any(record['raw'].shape[1] != clean['image'].shape[1] for record in noise.values()):
        raise ValueError('노이즈와 전체셋의 임베딩 차원이 다릅니다.')
    out = config.root / 'results/axis_alignment' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    save_json(out / 'config.json', asdict(config))
    save_json(out / 'provenance.json', dict(clean_cache=clean['path'],
              image_sha256=array_hash(clean['image']), text_sha256=array_hash(clean['text']),
              noise={modality: dict(path=record['path'], raw_sha256=array_hash(record['raw']),
                                   manifest=record['manifest']) for modality, record in noise.items()}))
    chars, words = text_noise_strings(config.seed, config.n_text_noise_each)
    save_json(out / 'text_noise_strings.json', dict(chars=chars, words=words))
    print(f'전체셋: 이미지 {len(clean["image"])}개, 캡션 {len(clean["text"])}개', flush=True)
    vertices = exp1_vertices(noise, out)
    axes = exp2_axes(clean, vertices, out)
    summary = exp3_axis_alignment(axes, out)
    if config.make_figures:
        make_figure(summary, out)
    save_report(summary, clean, config, out)
    for name, value in summary.items():
        print(f'{name}: {value:.3f}', flush=True)
    print(f'완료: {out}', flush=True)
    return dict(output=out, axes=axes, summary=summary)

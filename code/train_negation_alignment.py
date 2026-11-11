"""CARZero 질환 잔차 정렬 학습. 이 파일 하나를 실행한다. CLI 없음.
TrainConfig: 프로젝트 root 상대경로, epoch/lr/batch 설정.
고정 이미지 encoder, 전체 text encoder를 질환 잔차 cosine loss로 학습.
환자 단위 분리. 기존 NIH test 목록이 있으면 해당 환자를 test로 보존.
그 외는 환자 15% test, 남은 환자 15% validation, 나머지 train.
훈련 자료로만 평면과 질환 target 추정; validation AUC로 checkpoint 선택.
best.pt는 text_encoder와 고정 reference를 저장한다. 기반 모델 로드 후
model.text_encoder.load_state_dict(checkpoint['text_encoder'])로 적용한다.
출력: root/results/residual_alignment_training/<시간>/
config.json, split.csv, best.pt, history.csv, test_before/after/comparison.csv,
disease_comparison.csv. 사전학습 모델(epoch 0)도 선택 후보에 포함.
필요 환경: 기존 CARZero 실행 환경 + numpy/pandas/scipy/Pillow/torch.
"""
from __future__ import annotations
import hashlib, importlib.metadata, json, os, random, shutil, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class TrainConfig:
    root: Path = ROOT
    carzero_project: Path = ROOT / 'third_party' / 'BI_MCQ'
    carzero_weight_root: Path = ROOT / 'ckpt'
    nih_root: Path = ROOT / 'data' / 'NIH'
    carzero_embedding_dir: Path = ROOT / 'cache' / 'embeddings'
    carzero_ckpt: str = 'best_model.ckpt'
    device: str | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    prompt_batch: int = 14
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 64
    n_noise: int = 1000

NIH_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

NIH_TERMS = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

TEMPLATES = [('There is {x}.', 'There is no {x}.'), ('{X} is present.', 'No {x} is present.'), ('An image with {x}.', 'An image without {x}.'), ('There is evidence of {x}.', 'There is no evidence of {x}.'), ('The image shows {x}.', 'The image shows no {x}.'), ('{X} is seen in the image.', 'No {x} is seen in the image.'), ('Signs of {x} are observed.', 'No signs of {x} are observed.'), ('The image shows evidence of {x}.', 'The image shows no evidence of {x}.'), ('{X} is detected in the image.', '{X} is not detected in the image.'), ('The image is positive for {x}.', 'The image is negative for {x}.')]

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

def find_patch_size(module) -> int:
    """kernel == stride인 첫 Conv2d (ViT patch embedding)에서 패치 크기를 읽는다."""
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == m.stride and (m.kernel_size[0] > 1):
            return int(m.kernel_size[0])
    raise ValueError('Cannot infer the patch size of the image encoder.')

def load_carzero_model(cfg: TrainConfig, device: str):
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

    def __init__(self, cfg: TrainConfig):
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

ALIGN_EPS = 1e-12

def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), ALIGN_EPS)

def residual(x, basis):
    return x - x @ basis @ basis.T

def auc(y, score):
    """동점 평균 순위를 사용하는 AUROC."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    if y.all() or not y.any():
        return float('nan')
    order = np.argsort(score, kind='stable')
    _, start, count = np.unique(score[order], return_index=True, return_counts=True)
    rank = np.empty(len(score), dtype=float)
    rank[order] = np.repeat(start + (count + 1) / 2, count)
    n = y.sum()
    return float((rank[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))

def cosine(a, b):
    length = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / length) if length > ALIGN_EPS else float('nan')

class ResidualAlignmentLoss(nn.Module):

    @torch.no_grad()
    def __init__(self, z_train, y_train, image_noise):
        super().__init__()
        z = F.normalize(z_train.detach().float(), dim=-1)
        y = y_train.detach().to(device=z.device, dtype=z.dtype)
        if z.ndim != 2 or y.ndim != 2 or len(z) != len(y):
            raise ValueError('z_train [N,F], y_train [N,D]의 행 순서를 맞춰주세요.')
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError('y_train에는 양성 1, 음성 0을 입력하세요.')
        count = y.sum(0)
        if torch.any((count == 0) | (count == len(z))):
            raise ValueError('학습 자료의 각 질환에 양성·음성 영상이 모두 필요합니다.')
        b = image_noise.detach().to(device=z.device, dtype=z.dtype)
        if b.shape != (z.shape[1],) or b.norm() < 1e-08:
            raise ValueError('image_noise는 길이가 F인 유효한 노이즈 방향이어야 합니다.')
        b = F.normalize(b, dim=0)
        mean = z.mean(0)
        e = mean - mean @ b * b
        if e.norm() < 1e-08:
            raise ValueError('평면을 정의할 독립적인 평균 방향과 노이즈 방향이 필요합니다.')
        q = torch.stack((b, F.normalize(e, dim=0)), dim=1)
        a = y.T @ z / count[:, None] - (1 - y).T @ z / (len(z) - count)[:, None]
        a = a - a @ q @ q.T
        self.register_buffer('basis', q)
        self.register_buffer('target', F.normalize(a, dim=-1))
        self.register_buffer('valid', a.norm(dim=-1) > 1e-08)

    def forward(self, positive, negative, disease_index):
        with torch.autocast(device_type=positive.device.type, enabled=False):
            dt = F.normalize(positive.float(), dim=-1) - F.normalize(negative.float(), dim=-1)
            w = dt - dt @ self.basis @ self.basis.T
            similarity = (F.normalize(w, dim=-1) * self.target[disease_index]).sum(-1)
            weight = self.valid[disease_index].to(w.dtype)
            return ((1 - similarity) * weight).sum() / weight.sum().clamp_min(1)

def prepare_data(cfg):
    frame = pd.read_csv(cfg.nih_root/'Data_Entry_2017.csv')
    frame['Image Index'] = frame['Image Index'].astype(str)
    if frame['Image Index'].duplicated().any():
        raise ValueError('NIH 영상 이름 중복을 확인해주세요.')
    patient = frame['Patient ID'].astype(str).to_numpy()
    rng = np.random.default_rng(cfg.seed)
    # 기존 NIH test 목록이 있으면 해당 환자를 모두 test로 보존한다.
    test_csv = cfg.nih_root/'NIH_test_data.csv'
    test_txt = cfg.nih_root/'test_list.txt'
    if test_csv.exists() or test_txt.exists():
        names = (pd.read_csv(test_csv)['Image Index'].astype(str).tolist() if test_csv.exists()
                 else test_txt.read_text().splitlines())
        if not set(names).issubset(set(frame['Image Index'])):
            raise ValueError('test 목록과 NIH annotation의 영상 이름을 확인해주세요.')
        test = np.isin(patient, np.unique(patient[frame['Image Index'].isin(names)]))
    else:
        ids = rng.permutation(np.unique(patient))
        test = np.isin(patient, ids[:max(1, int(len(ids)*cfg.test_fraction))])
    ids = rng.permutation(np.unique(patient[~test]))
    val = (~test) & np.isin(patient, ids[:max(1, int(len(ids)*cfg.validation_fraction))])
    frame['split'] = np.where(test, 'test', np.where(val, 'validation', 'train'))
    findings = frame['Finding Labels'].str.replace('_', ' ', regex=False).str.split('|').map(set)
    labels = np.column_stack([findings.map(lambda s: int(c in s)) for c in NIH_CLASSES])
    for split in ('train', 'validation', 'test'):
        y = labels[frame['split'].eq(split)]
        if len(y) == 0 or np.any((y.sum(0) == 0) | (y.sum(0) == len(y))):
            raise ValueError(f'{split}: 각 질환에 양성·음성 영상이 모두 필요합니다. 분할 설정을 확인해주세요.')
    paths = {}
    for p in cfg.nih_root.rglob('*.png'):
        if p.name in paths:
            raise ValueError(f'중복 영상 파일 이름: {p.name}')
        paths[p.name] = p
    missing = [n for n in frame['Image Index'] if n not in paths]
    if missing:
        raise FileNotFoundError(f'NIH 영상 {len(missing)}개 경로 확인 필요: {missing[:5]}')
    return frame, labels, [paths[n] for n in frame['Image Index']]


def encode_training_images(enc, frame, paths, cfg):
    cache = cfg.root/'cache'/'residual_alignment'
    cache.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(json.dumps(dict(encoder=enc.signature,
        names=frame['Image Index'].tolist(), files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]),
        sort_keys=True).encode()).hexdigest()[:20]
    path = cache/f'images_{token}.npy'
    if path.exists():
        z = np.load(path)
    else:
        parts = []
        for start in range(0, len(paths), cfg.batch_size):
            parts.append(enc.encode_paths(paths[start:start+cfg.batch_size]))
            if start % (cfg.batch_size*50) == 0:
                print(f'영상 추출 {start}/{len(paths)}', flush=True)
        z = np.concatenate(parts)
        np.save(path, z)
    if len(z) != len(frame) or not np.isfinite(z).all():
        raise ValueError('영상 임베딩 캐시의 행 수와 유효값을 확인해주세요.')
    return unit(z)


def image_noise_direction(enc, cfg):
    token = hashlib.sha256(json.dumps(dict(encoder=enc.signature, seeds=cfg.seeds, n=cfg.n_noise,
                                          family='uniform_rgb'), sort_keys=True).encode()).hexdigest()[:20]
    path = cfg.root/'cache'/'residual_alignment'/f'noise_{token}.npy'
    if path.exists():
        return np.load(path)
    parts = []
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        parts.append(unit(enc.encode_pixels(rng.uniform(0, 1, (*enc.hw, 3)).astype(np.float32)
                                            for _ in range(cfg.n_noise))))
    b = unit(np.concatenate(parts).mean(0))
    np.save(path, b)
    return b


def encode_tokens(model, tokens, indices):
    _, g, _ = model.text_encoder_forward(*(tokens[k][indices] for k in
                                           ('caption_ids', 'attention_mask', 'token_type_ids')))
    return g


@torch.no_grad()
def evaluate_training(model, tokens, z, y, criterion, diseases, templates, batch):
    model.eval()
    k = len(diseases)
    device = criterion.basis.device
    t = torch.cat([encode_tokens(model, tokens, torch.arange(i, min(i+batch, 2*k), device=device))
                   for i in range(0, 2*k, batch)])
    tp, tn = F.normalize(t[:k].float(), dim=-1), F.normalize(t[k:].float(), dim=-1)
    dt = tp-tn
    w = dt-(dt @ criterion.basis) @ criterion.basis.T
    q = criterion.basis.cpu().numpy()
    dt, w = dt.cpu().numpy(), w.cpu().numpy()
    r = residual(z, q)
    rows = []
    for j, d in enumerate(diseases):
        positive = y[:, d].astype(bool)
        a = r[positive].mean(0)-r[~positive].mean(0)
        score = z @ dt[j]
        rows.append(dict(disease=NIH_CLASSES[d], template=int(templates[j]),
                         auc=auc(positive, score), auc_residual=auc(positive, r @ w[j]),
                         alignment_cosine=cosine(a, w[j]), gap_residual=float(a @ w[j])))
    return pd.DataFrame(rows)


def train():
    cfg = TrainConfig()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    out = cfg.root/'results'/'residual_alignment_training'/__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (cfg.root/'cache').mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(cfg), default=str, indent=2), encoding='utf-8')
    frame, y, paths = prepare_data(cfg)
    frame[['Image Index', 'Patient ID', 'split']].to_csv(out/'split.csv', index=False)
    enc = CarzeroEncoder(cfg)
    try:
        z = encode_training_images(enc, frame, paths, cfg)
        b = image_noise_direction(enc, cfg)
        device, model = enc.device, enc.model
        masks = {s: frame['split'].eq(s).to_numpy() for s in ('train', 'validation', 'test')}
        criterion = ResidualAlignmentLoss(torch.as_tensor(z[masks['train']], device=device),
                                          torch.as_tensor(y[masks['train']], device=device),
                                          torch.as_tensor(b, device=device))
        pos, diseases, templates = nih_prompts('pos')
        neg, _, _ = nih_prompts('neg')
        tokens = model.process_text(pos+neg, device)
        model.requires_grad_(False)
        model.text_encoder.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.text_encoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        k = len(pos)
        d = torch.as_tensor(diseases, device=device, dtype=torch.long)
        baseline = evaluate_training(model, tokens, z[masks['test']], y[masks['test']], criterion, diseases, templates, cfg.prompt_batch)
        baseline.to_csv(out/'test_before.csv', index=False)
        best = float(evaluate_training(model, tokens, z[masks['validation']], y[masks['validation']],
                                      criterion, diseases, templates, cfg.prompt_batch).auc.mean())
        history = [dict(epoch=0, loss=float('nan'), validation_auc=best)]
        def save(epoch):
            torch.save(dict(text_encoder=model.text_encoder.state_dict(), reference=criterion.state_dict(),
                            epoch=epoch, validation_auc=best, config=asdict(cfg),
                            base_checkpoint=str(cfg.carzero_weight_root/cfg.carzero_ckpt)), out/'best.pt')
        save(0)
        for epoch in range(1, cfg.epochs+1):
            model.eval(); model.text_encoder.train()
            order = torch.randperm(k, device=device)
            total = torch.zeros((), device=device)
            for start in range(0, k, cfg.prompt_batch):
                idx = order[start:start+cfg.prompt_batch]
                t = encode_tokens(model, tokens, torch.cat((idx, idx+k)))
                loss = criterion(t[:len(idx)], t[len(idx):], d[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.text_encoder.parameters(), 1.0)
                optimizer.step()
                total += loss.detach()*len(idx)
            val = evaluate_training(model, tokens, z[masks['validation']], y[masks['validation']],
                                    criterion, diseases, templates, cfg.prompt_batch)
            score = float(val.auc.mean())
            history.append(dict(epoch=epoch, loss=float(total/k), validation_auc=score))
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            if score > best:
                best = score; save(epoch)
            print(f'epoch {epoch}/{cfg.epochs} loss={history[-1]["loss"]:.5f} validation AUC={score:.5f}', flush=True)
        checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=False)
        model.text_encoder.load_state_dict(checkpoint['text_encoder'])
        after = evaluate_training(model, tokens, z[masks['test']], y[masks['test']], criterion, diseases, templates, cfg.prompt_batch)
        after.to_csv(out/'test_after.csv', index=False)
        comparison = baseline.merge(after, on=['disease', 'template'], suffixes=('_before', '_after'))
        comparison['delta_auc'] = comparison.auc_after-comparison.auc_before
        comparison.to_csv(out/'test_comparison.csv', index=False)
        comparison.groupby('disease').mean(numeric_only=True).to_csv(out/'disease_comparison.csv')
        print(f'완료: {out}\n선택 epoch={checkpoint["epoch"]}, test AUC {baseline.auc.mean():.5f} → {after.auc.mean():.5f}', flush=True)
    finally:
        enc.close()


if __name__ == '__main__':
    train()

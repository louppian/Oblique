"""Chowers et al.: frozen embeddings, translated Euclidean retrieval."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[2]

def markdown_table(frame):
    labels={'model':'모델','noise_std':'잡음 표준편차','sigma':'잡음 표준편차','step':'이동 계수','gap':'gap 크기','threshold':'특이값 임계값','accuracy_before_pct':'원본 정확도 (%)','accuracy_after_pct':'이동 후 정확도 (%)','robustness_before_pct':'원본 예측 유지율 (%)','robustness_after_pct':'이동 후 예측 유지율 (%)','clean_accuracy_pct':'잡음 없는 정확도 (%)','noisy_accuracy_pct':'잡음 추가 정확도 (%)','robustness_pct':'예측 유지율 (%)'}
    lines=['| '+' | '.join(labels.get(c,c) for c in frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

class CifarParquet(Dataset):
    def __init__(self, path, transform):
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        self.data, self.transform = table.to_pydict(), transform
        self.label = 'fine_label' if 'fine_label' in self.data else 'label'
        metadata = json.loads(table.schema.metadata[b'huggingface'])
        self.classes = metadata['info']['features'][self.label]['names']
        self.classes = ['crab' if c == 'cra' else c for c in self.classes]
    def __len__(self): return len(self.data[self.label])
    def __getitem__(self, i):
        from PIL import Image
        from io import BytesIO
        image_key = 'img' if 'img' in self.data else 'image'
        image = Image.open(BytesIO(self.data[image_key][i]['bytes'])).convert('RGB')
        return self.transform(image), self.data[self.label][i]

def gap_direction(images, texts, threshold=0., singular_threshold=False):
    centered = texts.double() - texts.double().mean(0)
    _, singular, vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = singular.square() / singular.square().sum()
    keep = singular > threshold if singular_threshold else variance > max(threshold, 1e-12)
    gap = texts.double().mean(0) - images.double().mean(0)
    basis = vectors[keep]
    return (gap - (gap @ basis.T) @ basis).float(), gap.float(), int(keep.sum())

@torch.inference_mode()
def evaluate(images, texts, labels, out, threshold=0., draws=100, quantization=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    x, t = torch.as_tensor(images, device='cuda'), torch.as_tensor(texts, device='cuda')
    y = torch.as_tensor(labels, device='cuda')
    direction, gap, rank = gap_direction(x, t, threshold)
    steps = np.linspace(-2, 1, 13)
    clean = (x @ t.T - .5*t.square().sum(1)).argmax(1)
    clean_shifted = [(x @ t.T-.5*t.square().sum(1)+float(step)*(t@direction)).argmax(1) for step in steps]
    rows = []
    for sigma in (0., .005, .0225, .04):
        torch.manual_seed(2026)
        totals = torch.zeros((len(steps), 2), device='cuda', dtype=torch.float64)
        for draw in range(1 if sigma == 0 else draws):
            noisy = t + sigma*torch.randn_like(t)
            base = x @ noisy.T - .5*noisy.square().sum(1)
            offset = noisy @ direction
            for j, step in enumerate(steps):
                pred = (base + float(step)*offset).argmax(1)
                totals[j, 0] += (pred == y).double().mean()
                totals[j, 1] += (pred == clean_shifted[j]).double().mean()
        totals = (totals/(1 if sigma == 0 else draws)).cpu().numpy()
        for step, (accuracy, robustness) in zip(steps, totals):
            rows.append(dict(kind='gaussian', parameter=sigma, step=float(step), gap=float(torch.linalg.vector_norm(gap-float(step)*direction)), accuracy=float(accuracy), robustness=float(robustness)))
    if threshold == 0:
        assert all(torch.equal(pred, clean) for pred in clean_shifted), 'Clean nearest-neighbor invariance failed'
    for bins in ((32, 64, 128, 256) if quantization else ()):
        quantize = lambda z: (torch.round((z.clamp(-3, 3)+3)/6*bins)/bins)*6-3
        qx = quantize(x)
        for step in steps:
            qt = quantize(t-float(step)*direction)
            pred = (qx @ qt.T-.5*qt.square().sum(1)).argmax(1)
            rows.append(dict(kind='quantization', parameter=bins, step=float(step), gap=float(torch.linalg.vector_norm(gap-float(step)*direction)), accuracy=float((pred==y).float().mean()), robustness=float((pred==clean).float().mean())))
    import pandas as pd
    frame = pd.DataFrame(rows); frame.to_csv(out/'metrics.csv', index=False, float_format='%.4f')
    for number, kind in enumerate(('gaussian', 'quantization') if quantization else ('gaussian',), 1):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for parameter, group in frame[frame.kind==kind].groupby('parameter'):
            for ax, metric in zip(axes, ('accuracy', 'robustness')):
                ax.plot(group.gap, group[metric], '.-', label=f'{parameter:.4f}')
                ax.set(xlabel='Gap norm', ylabel=metric.title()); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        axes[0].legend(title='Noise std' if kind=='gaussian' else 'Intervals')
        fig.savefig(out/f'figure{number}.png', dpi=180); plt.close(fig)
    (out/'report.md').write_text(f'# Modality gap reproduction\n\nImages: {len(x)}; text candidates: {len(t)}; retained PCA rank: {rank}.\n\nFrozen normalized embeddings; text translation T − step × projected_gap; squared Euclidean nearest neighbor. Gaussian noise is added to text, with shared draws across translations, seed 2026, {draws} draws. Noise parameters are interpreted as standard deviations. Quantization uses uniform intervals on [−3, 3].\n\nPCA threshold is per-component explained variance > {threshold:.4f}. Exact projection uses numerical rank tolerance 0.0000 (internally 1e-12). Initial gap: {float(gap.norm()):.4f}; shift norm: {float(direction.norm()):.4f}.\n\nAccuracy is agreement with the data label; robustness is agreement with the original clean prediction.\n\n'+markdown_table(frame[frame.step.isin([0.,1.])])+'\n\nSource: https://arxiv.org/html/2603.29080v2\n', encoding='utf-8')
    return frame

def cifar(dataset, model_name):
    import open_clip
    from torchvision.datasets import CIFAR10, CIFAR100
    from torch.utils.data import DataLoader
    torch.set_num_threads(8)
    cache = ROOT/'cache/clip'/f'{dataset.lower()}_{model_name.lower().replace("-", "").replace("/", "")}'
    cache.mkdir(parents=True, exist_ok=True)
    out = ROOT/'code/analysis'/f'{dataset}_{model_name.replace("ViT", "VIT").replace("-14", "_14").replace("-16", "_16")}'/'modality_gap'
    if not (cache/'embeddings.npz').exists():
        model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=str(ROOT/'ckpt'/f'{model_name}.pt'), force_quick_gelu=True, device='cuda', load_weights_only=False)
        model.eval(); tokenizer = open_clip.get_tokenizer(model_name)
        data = CifarParquet(ROOT/'data/cifar'/f'{dataset.lower()}_test.parquet', transform)
        images, labels = [], []
        with torch.inference_mode(), torch.autocast('cuda'):
            for batch, target in DataLoader(data, batch_size=128, num_workers=4, pin_memory=True):
                images.append(model.encode_image(batch.cuda(non_blocking=True), normalize=True).float().cpu().numpy()); labels.append(target.numpy())
            texts = model.encode_text(tokenizer([f'a photo of a {c.replace("_", " ") }.' for c in data.classes]).cuda(), normalize=True).float().cpu().numpy()
        np.savez(cache/'embeddings.npz', images=np.concatenate(images), texts=texts, labels=np.concatenate(labels))
        (cache/'metadata.json').write_text(json.dumps(dict(dataset=dataset, split='test', model=model_name, pretrained='OpenAI', resolution=224, prompt='a photo of a {class}.', note='Checkpoint and prompt are explicit reproduction choices; paper does not give exact pretrained identifiers.'), indent=2), encoding='utf-8')
        del model; torch.cuda.empty_cache()
    data = np.load(cache/'embeddings.npz')
    evaluate(data['images'], data['texts'], data['labels'], out)
    print('Complete', out, flush=True)

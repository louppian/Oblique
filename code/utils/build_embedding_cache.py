"""Dataset embedding cache. NumPy arrays only; load with allow_pickle=False."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'code').is_dir() and (p/'third_party').is_dir())


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return dict(name=path.name, sha256=digest.hexdigest())


def embeddings(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected finite, nonempty [samples, dimensions] embeddings')
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError('Zero-length embedding')
    return raw, raw/norm


def save_cache(path, metadata, **arrays):
    """Atomic uncompressed NPZ; metadata and string columns use Unicode arrays."""
    path = Path(path)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError('Object arrays require conversion to numeric or Unicode arrays')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem+'_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)), **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    result['metadata'] = json.loads(str(result['metadata']))
    return result


def build_cache(encoder, model, dataset, image_paths, image_ids, texts, text_ids,
                noise_pixels, noise_texts, metadata, image_columns=None, text_columns=None):
    """encoder implements encode_paths, encode_texts, encode_pixels. Noise pixels: factory."""
    for name in (model, dataset):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Model/dataset must be single directory names')
    if len(image_paths) != len(image_ids) or len(texts) != len(text_ids):
        raise ValueError('Input and ID counts differ')
    if len(set(map(str, image_ids))) != len(image_ids) or len(set(map(str, text_ids))) != len(text_ids):
        raise ValueError('IDs must be unique within each modality')
    image_columns, text_columns = image_columns or {}, text_columns or {}
    if any(len(v) != len(image_ids) for v in image_columns.values()) or any(len(v) != len(texts) for v in text_columns.values()):
        raise ValueError('Metadata column row count differs')
    reserved = {'raw', 'unit', 'ids', 'paths', 'texts', 'metadata'}
    if reserved & (image_columns.keys() | text_columns.keys()):
        raise ValueError('Reserved cache column name')
    paths = [str(Path(p).resolve().relative_to(ROOT)) for p in image_paths]
    identity = dict(version=1, model=model, dataset=dataset, settings=metadata,
                    image_ids=list(map(str,image_ids)), paths=paths,
                    image_files=[(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in image_paths],
                    text_ids=list(map(str,text_ids)), texts=list(texts), noise_texts=list(noise_texts),
                    image_columns={k:np.asarray(v).tolist() for k,v in image_columns.items()},
                    text_columns={k:np.asarray(v).tolist() for k,v in text_columns.items()})
    directory = ROOT/'cache'/model/dataset
    metadata = json.loads(json.dumps(identity, sort_keys=True, ensure_ascii=False))
    for kind in ('image', 'text', 'noise'):
        path = directory/f'{kind}.npz'
        if path.is_file():
            existing = load_cache(path)
            if existing['metadata'] != metadata:
                raise ValueError(f'Cache metadata mismatch: {path}')
            print(f'Reuse {path}', flush=True)
            continue
        if kind == 'image':
            raw, unit = embeddings(encoder.encode_paths(image_paths))
            if len(raw) != len(image_ids): raise ValueError('Image encoder row count differs')
            save_cache(path,metadata,raw=raw,unit=unit,ids=np.asarray(image_ids,dtype=str),paths=paths,**image_columns)
        elif kind == 'text':
            raw, unit = embeddings(encoder.encode_texts(texts))
            if len(raw) != len(text_ids): raise ValueError('Text encoder row count differs')
            save_cache(path,metadata,raw=raw,unit=unit,ids=np.asarray(text_ids,dtype=str),texts=texts,**text_columns)
        else:
            raw_i, unit_i = embeddings(encoder.encode_pixels(noise_pixels()))
            raw_t, unit_t = embeddings(encoder.encode_texts(noise_texts))
            if len(raw_t) != len(noise_texts): raise ValueError('Noise text row count differs')
            save_cache(path,metadata,image_raw=raw_i,image_unit=unit_i,image_ids=np.arange(len(raw_i)),
                       text_raw=raw_t,text_unit=unit_t,text_ids=np.arange(len(raw_t)),texts=noise_texts)
        print(f'Saved {path}', flush=True)
    return directory


"""Run directly. Build three caches per configured model/dataset; edit CONFIG only."""
from pathlib import Path
import sys
import importlib
import json
import random
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(ROOT/'code'/'utils'))
import encoders as an
CONFIG = dict(pairs=(('clip','coco'),('clip','nih'),('carzero','coco'),('carzero','nih')),
              batch_size=256, seed=0, image_noise_count=1000, text_noise_each=500)


def dataset_inputs(dataset):
    if dataset == 'coco':
        root = ROOT/'data/coco'
        ann = json.loads((root/'annotations/captions_val2017.json').read_text())
        images = sorted(ann['images'],key=lambda r:r['id'])
        captions = sorted(ann['annotations'],key=lambda r:r['id'])
        return ([root/'val2017'/r['file_name'] for r in images], [str(r['id']) for r in images],
                [r['caption'].strip() for r in captions], [str(r['id']) for r in captions], {},
                dict(image_id=[str(r['image_id']) for r in captions]))
    if dataset == 'nih':
        import pandas as pd
        root = ROOT/'data/NIH'
        ann = pd.read_csv(root/'Data_Entry_2017.csv').set_index('Image Index')
        names = (pd.read_csv(root/'NIH_test_data.csv')['Image Index'].astype(str).str.strip().tolist()
                 if (root/'NIH_test_data.csv').is_file() else (root/'test_list.txt').read_text().splitlines())
        files = {p.name:p for p in root.rglob('*.png')}
        labels = np.asarray([[int(c in str(ann.loc[n,'Finding Labels']).replace('_',' ').split('|'))
                              for c in an.NIH_CLASSES] for n in names],dtype=np.uint8)
        texts, ids, diseases, templates, polarity = [], [], [], [], []
        for sign in ('pos','neg'):
            prompts, ds, ts = an.nih_prompts(sign)
            texts.extend(prompts); diseases.extend(ds); templates.extend(ts); polarity.extend([sign]*len(prompts))
            ids.extend(f'{sign}_{d}_{t}' for d,t in zip(ds,ts))
        return ([files[n] for n in names],names,texts,ids,dict(labels=labels),
                dict(disease=diseases,template=templates,polarity=polarity))
    raise ValueError(dataset)


def main():
    (ROOT/'cache').mkdir(exist_ok=True)
    cfg = replace(an.Config(),batch_size=CONFIG['batch_size'])
    for model,dataset in CONFIG['pairs']:
        inputs = dataset_inputs(dataset)
        if model == 'clip':
            import open_clip
            weights = sorted((ROOT/'ckpt').glob('**/open_clip_model.safetensors'))
            checkpoint = weights[0] if weights else ROOT/'ckpt/ViT-B-32.pt'
            cfg_model = replace(cfg,clip_model='ViT-B-32-quickgelu',clip_pretrained=str(checkpoint))
            encoder_class = an.ClipEncoder
        elif model == 'carzero':
            checkpoint = ROOT/'ckpt/best_model.ckpt'
            cfg_model, encoder_class = cfg, an.CarzeroEncoder
        else:
            raise ValueError(model)
        identity = file_identity(checkpoint)
        encoder = encoder_class(cfg_model)
        try:
            rng = random.Random(1000+CONFIG['seed'])
            n = CONFIG['text_noise_each']
            chars = [''.join(rng.choice(an.CHARSET) for _ in range(rng.randint(20,75))) for _ in range(n)]
            words = [' '.join(rng.choice(an.WORD_POOL) for _ in range(rng.randint(2,15))) for _ in range(n)]
            def pixels():
                rng_i = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'],1101]))
                for _ in range(CONFIG['image_noise_count']):
                    yield rng_i.uniform(0,1,(*encoder.hw,3)).astype(np.float32)
            signature = dict(encoder.signature)
            for key in ('ckpt','pretrained'):
                if key in signature: signature[key] = checkpoint.name
            metadata = dict(checkpoint=identity,encoder=signature,seed=CONFIG['seed'],
                            noise_family='uniform_rgb',image_noise_count=CONFIG['image_noise_count'],
                            text_noise_each=n,text_noise_order=['char','word'],
                            nih_class_order=an.NIH_CLASSES if dataset=='nih' else [])
            build_cache(encoder,model,dataset,*inputs[:4],pixels,chars+words,metadata,
                        image_columns=inputs[4],text_columns=inputs[5])
        finally:
            if hasattr(encoder,'close'): encoder.close()
            del encoder


if __name__ == '__main__':
    main()

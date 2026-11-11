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

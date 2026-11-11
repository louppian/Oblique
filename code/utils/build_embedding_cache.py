"""Run directly. Build three caches per configured model/dataset; edit CONFIG only."""
from pathlib import Path
import sys
import importlib
import json
import random
from dataclasses import replace
import numpy as np
from embedding_cache import ROOT, build_cache, file_identity

sys.path.insert(0, str(ROOT/'code'))
an = importlib.import_module('1_analysis')
CONFIG = dict(pairs=(('clip','coco'),('clip','nih'),('carzero','coco'),('carzero','nih')),
              batch_size=128, seed=0, image_noise_count=1000, text_noise_each=500)


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
        names = pd.read_csv(root/'NIH_test_data.csv')['Image Index'].astype(str).str.strip().tolist()
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
    cfg = replace(an.CONFIG,batch_size=CONFIG['batch_size'])
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

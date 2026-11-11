"""Shared execution adapter for the five fixed-encoder main analyses."""
from pathlib import Path
import sys, importlib
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr,"reconfigure"):sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import replace
from contextlib import contextmanager
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'code').is_dir() and (p/'third_party').is_dir())
sys.path.insert(0,str(ROOT/'code'/'old'/'utils'))
sys.path.insert(0,str(ROOT/'code'/'utils'))
from build_embedding_cache import load_cache
from output import workspace
an=importlib.import_module('1_analysis')
weights=sorted((ROOT/'ckpt').glob('**/open_clip_model.safetensors'))
CONFIG=replace(an.CONFIG,clip_model='ViT-B-32-quickgelu',clip_pretrained=str(weights[0] if weights else ROOT/'ckpt/ViT-B-32.pt'))

class Context(an.Context):
    def images(self,enc,dataset):
        key=(enc,dataset)
        if key in self._images:return self._images[key]
        data=load_cache(ROOT/'cache'/enc/dataset/'image.npz')
        ids=data['ids'].astype(int) if dataset=='coco' else np.arange(len(data['ids']))
        pick=np.arange(len(ids))
        if an.DOMAIN[enc]!=dataset:
            pick=np.sort(np.random.default_rng(7001).choice(len(ids),min(self.cfg.n_ood,len(ids)),replace=False))
        raw=data['raw'][pick]
        if enc=='clip' and dataset=='coco':self.clip_clean_hash=an.legacy_fingerprint(raw)+an.legacy_fingerprint(ids[pick])
        out=dict(Z=an.normalize(raw),ids=ids[pick],paths=[ROOT/str(s) for s in data['paths'][pick]],pick=pick,
                 labels=data['labels'][pick] if 'labels' in data else None)
        self._images[key]=out
        return out
    def domain_texts(self,enc):
        key=('texts',enc)
        if key in self._images:return self._images[key]
        dataset=an.DOMAIN[enc];d=load_cache(ROOT/'cache'/enc/dataset/'text.npz')
        if dataset=='coco':out=dict(T=an.normalize(d['raw']),texts=d['texts'].tolist(),image_ids=d['image_id'].astype(int))
        else:
            mask=d['polarity']=='pos';out=dict(T=an.normalize(d['raw'][mask]),texts=d['texts'][mask].tolist(),disease=d['disease'][mask],template=d['template'][mask])
        self._images[key]=out
        return out

def prepare_geometry(ctx):
    # Downstream experiments need fitted axes, while bootstrap reporting belongs to main/02.
    for enc in ctx.cfg.encoders:
        for seed in ctx.cfg.seeds:
            if (enc,seed) in ctx.axes and (enc,seed) in ctx.text_axes:continue
            ref,ev=ctx.split(enc,seed)
            z=ctx.images(enc,an.DOMAIN[enc])["Z"]
            ctx.axes[(enc,seed)]=an.Axes(z[ref].mean(0),ctx.reference_noise(enc,seed).mean(0))
            ref,ev,_=an.text_split(ctx,enc,seed)
            t=ctx.domain_texts(enc)["T"]
            ctx.text_axes[(enc,seed)]=an.Axes(t[ref].mean(0),ctx.text_noise_embeddings(enc,seed)["U"].mean(0))
an.ensure_geometry=prepare_geometry

def run(name,steps,config=CONFIG,include_magnitude=False):
    original=an.result_directory
    with workspace('main',name) as work:
        an.result_directory=lambda root,section:work/section
        ctx=Context(config,out_name=name)
        try:
            for label,functions in steps:
                print('SECTION '+label,flush=True)
                with an.ExperimentOutput(ctx.out,label):
                    for fn in functions:
                        method=getattr(an,fn);getattr(method,'__wrapped__',method)(ctx)
            if include_magnitude:
                import cached_analysis
                @contextmanager
                def staging(group,stem):
                    folder=work/stem;folder.mkdir();yield folder
                previous=cached_analysis.workspace;cached_analysis.workspace=staging
                try:cached_analysis.run_magnitude('new_components',(('clip','coco'),('clip','nih'),('carzero','coco'),('carzero','nih')),config.seeds,config.n_reference)
                finally:cached_analysis.workspace=previous
        finally:
            an.result_directory=original
            for encoder in ctx._encoders.values():
                if hasattr(encoder,'close'):encoder.close()

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

def capture_figure(output,fig):
    from io import BytesIO
    from PIL import Image
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            if axis.get_scale()!='linear' or axis.get_units() or isinstance(axis.get_major_locator(),FixedLocator):continue
            lo,hi=axis.get_view_interval()
            if 0<max(abs(lo),abs(hi))<0.01:
                formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((0,0))
            else:formatter=FormatStrFormatter('%.3f')
            axis.set_major_formatter(formatter)
    if fig.axes[0].get_xlabel() in ('Number of noise samples','Noise sample count'):
        text_noise=fig.axes[0].get_xlabel()=='Noise sample count'
        names={'gaussian_matched':'Gaussian','uniform_matched':'Uniform RGB','gray_matched':'Uniform gray','char':'Characters','word':'Words','U':'Combined'}
        styles={'gaussian_matched':'-','uniform_matched':'--','gray_matched':':','char':'-','word':'--','U':':'}
        for ax in fig.axes:
            for line in ax.lines:
                label=line.get_label()
                if '/' in label:
                    enc,family=label.split('/',1)
                    if enc in an.ENC_COLOR and family in names:
                        line.set_color(an.ENC_COLOR[enc]);line.set_linestyle(styles[family]);line.set_label(f'{an.ENC_LABEL[enc]}: {names[family]}')
            if ax.get_xlabel() in ('Number of noise samples','Noise sample count') or text_noise:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        fig.set_size_inches(13,4.8);fig.subplots_adjust(wspace=0.60,bottom=0.30)
        fig.axes[0].legend(loc='upper center',bbox_to_anchor=(0.5,-0.24),ncol=2,fontsize=8)
        if not text_noise:
            ax=fig.axes[1];labels=[tick.get_text() for tick in ax.get_yticklabels()]
            ax.set_yticks(ax.get_yticks(),[f"{an.ENC_LABEL[v.split('/')[0]]}\n{names[v.split('/')[1]]}–{names[v.split('/')[2]]}" for v in labels],fontsize=8)
            for patch,label in zip(ax.patches,labels):patch.set_facecolor(an.ENC_COLOR[label.split('/')[0]])
    with BytesIO() as buf:
        fig.savefig(buf,format='png',dpi=180,bbox_inches='tight');buf.seek(0)
        with Image.open(buf) as im:output.images.append(im.convert('RGB').copy())
    an.plt.close(fig)
an._capture_figure=capture_figure

def magnitude_figure(folder):
    import pandas as pd
    df=pd.read_csv(folder/'magnitude.csv')
    g=df[df.analysis_type=='components'].groupby(['model','dataset','modality','component'])[['mean_squared_norm','centered_variance']].mean()
    order=['Rhat','Bhat','E','Delta','cross_Rhat_Bhat']
    labels=[r'$\alpha\hat r$',r'$\beta\hat b$',r'$e_u$',r'$\delta_u$',r'$\hat r$–$\hat b$ cross term']
    fig,axes=an.plt.subplots(2,1,figsize=(16,10))
    for ax,key,title in zip(axes,['mean_squared_norm','centered_variance'],['(a) Mean squared component magnitude','(b) Centered variance and covariance']):
        frame=g[key].unstack().reindex(columns=order)
        frame.plot.bar(ax=ax,color=[an.C1,an.C2,an.C4,an.C3,an.MUTED])
        ax.set(title=title,xlabel='',ylabel='Contribution')
        ax.set_xticklabels([f'{m.upper()} / {d.upper()} / {t}' for m,d,t in frame.index],rotation=25,ha='right')
        ax.axhline(0,color=an.MUTED,lw=0.8)
        ax.legend(ax.containers,labels,ncol=5,loc='upper center',bbox_to_anchor=(0.5,-0.29))
    fig.tight_layout(h_pad=4)
    fig.savefig(folder/'magnitude.png',dpi=180,bbox_inches='tight');an.plt.close(fig)

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
                magnitude_figure(work/'new_components')
        finally:
            an.result_directory=original
            for encoder in ctx._encoders.values():
                if hasattr(encoder,'close'):encoder.close()

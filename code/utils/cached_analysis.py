"""Shared four-term geometry and medical analyses from the 2x2 embedding caches."""
from pathlib import Path
import csv
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_embedding_cache import ROOT, load_cache
from experiment_runner import workspace
from residual_core import fit_frame, components


def inputs(model,dataset):
    root=ROOT/'cache'/model/dataset
    values=[load_cache(root/f'{kind}.npz') for kind in ('image','text','noise')]
    if any(x['metadata']!=values[0]['metadata'] for x in values[1:]):
        raise ValueError(f'Cache input conditions differ: {root}')
    return values


def split_frame(image,text,noise,seed,n_reference):
    z,t=image['unit'].astype(float),text['unit'].astype(float)
    if not 2<=n_reference<len(z): raise ValueError('Reference size must leave evaluation images')
    order=np.random.default_rng(seed).permutation(len(z))
    ref,test=order[:n_reference],order[n_reference:]
    if 'image_id' in text:
        select=np.isin(text['image_id'],image['ids'][ref])
        evaluate=np.isin(text['image_id'],image['ids'][test])
        policy='image_disjoint_reference'
    else:
        select=text['polarity']=='pos';evaluate=np.ones(len(t),dtype=bool)
        policy='fixed_positive_prompt_reference'
    frame=fit_frame(z[ref],t[select],noise['image_unit'],noise['text_unit'])
    return frame,z[test],t[evaluate],test,policy


def table(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def run_magnitude(name,pairs,seeds,n_reference):
    rows=[]
    for model,dataset in pairs:
        image,text,noise=inputs(model,dataset)
        for seed in seeds:
            frame,z,t,_,policy=split_frame(image,text,noise,seed,n_reference)
            for modality,x in (('image',z),('text',t)):
                f=frame[modality];parts=components(x,f)
                base=dict(model=model,dataset=dataset,seed=seed,modality=modality,reference_policy=policy)
                energy={key:float(np.mean(np.sum(v*v,axis=1))) for key,v in parts.items()}
                variance={key:float(np.var(v,axis=0).sum()) for key,v in parts.items()}
                cross=float(2*np.mean(np.sum(parts['Rhat']*parts['Bhat'],axis=1)))
                covariance=float(2*np.mean(np.sum((parts['Rhat']-parts['Rhat'].mean(0))*(parts['Bhat']-parts['Bhat'].mean(0)),axis=1)))
                for key in parts:
                    rows.append(dict(**base,analysis_type='components',component=key,mean_squared_norm=energy[key],centered_variance=variance[key]))
                rows.append(dict(**base,analysis_type='components',component='cross_Rhat_Bhat',mean_squared_norm=cross,centered_variance=covariance))
                coeff=x@f['dual']; reconstruction=float(np.max(np.abs(sum(parts.values())-x)))
                error=abs(sum(energy.values())+cross-float(np.mean(np.sum(x*x,axis=1))))
                variance_error=abs(sum(variance.values())+covariance-float(np.var(x,axis=0).sum()))
                if max(reconstruction,error,variance_error)>1e-8:raise AssertionError('Four-term reconstruction')
                rows.append(dict(**base,analysis_type='summary',component='total',alpha_mean=float(coeff[:,0].mean()),alpha_variance=float(coeff[:,0].var()),
                    mean_squared_norm=float(np.mean(np.sum(x*x,axis=1))),centered_variance=float(np.var(x,axis=0).sum()),
                    reconstruction_error=reconstruction,energy_error=error,variance_error=variance_error,
                    e_eigenvalue=f['e_eigenvalue'],e_eigengap=f['e_eigengap']))
                # Same projection for both vectors; pair deterministic consecutive evaluation rows.
                other=np.roll(x,1,axis=0);p=(x@f['q'])@f['q'].T;v=np.roll(p,1,axis=0)
                middle=np.sum(p*v,axis=1);radius=np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1))*np.maximum(0,1-np.sum(v*v,axis=1)))
                actual=np.sum(x*other,axis=1)
                if np.any(actual<middle-radius-1e-6) or np.any(actual>middle+radius+1e-6):raise AssertionError('Cosine bounds')
                rows.append(dict(**base,analysis_type='cosine_bounds',component='plane',lower_mean=float(np.mean(middle-radius)),upper_mean=float(np.mean(middle+radius)),actual_mean=float(actual.mean())))
    with workspace('main',name) as work:
        table(work/'magnitude.csv',rows)
        import pandas as pd
        df=pd.DataFrame(rows);g=df[df.analysis_type=='components'].groupby(['model','dataset','modality','component'])[['mean_squared_norm','centered_variance']].mean()
        fig,axes=plt.subplots(2,1,figsize=(16,10))
        for ax,key in zip(axes,('mean_squared_norm','centered_variance')):
            g[key].unstack().plot.bar(ax=ax);ax.set_title(key);ax.legend(ncol=5)
        fig.tight_layout();fig.savefig(work/'magnitude.png',dpi=180);plt.close(fig)


def metrics(y,score):
    y=np.asarray(y,dtype=bool);p=y.sum();n=len(y)-p;pred=score>0
    auc=float((rankdata(score)[y].sum()-p*(p+1)/2)/(p*n)) if p and n else float('nan')
    tp=int(np.sum(y&pred));tn=int(np.sum(~y&~pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred))
    denominator=float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(auc=auc,f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                mcc=float((tp*tn-fp*fn)/np.sqrt(denominator)) if denominator else 0.,threshold=0.,
                threshold_policy='fixed_zero',n_positive=int(p),n_negative=int(n))


def run_medical(name,models,seeds,n_reference):
    rows=[]
    for model in models:
        image,text,noise=inputs(model,'nih')
        for seed in seeds:
            frame,z,_,test,policy=split_frame(image,text,noise,seed,n_reference)
            f=frame['image'];parts=components(z,f);labels=image['labels'][test];texts=text['unit'].astype(float)
            for i in np.flatnonzero(text['polarity']=='pos'):
                d=int(text['disease'][i]);template=int(text['template'][i])
                match=np.flatnonzero((text['polarity']=='neg')&(text['disease']==d)&(text['template']==template))
                if len(match)!=1:raise ValueError('Expected a unique negative prompt')
                tp,tn=texts[i],texts[match[0]];dt=tp-tn;y=labels[:,d].astype(bool)
                base=dict(model=model,seed=seed,disease=d,template=template,positive_prompt=str(text['texts'][i]),negative_prompt=str(text['texts'][match[0]]),reference_policy=policy)
                for kind,w in (('POS',tp),('NEG',-tn),('PNC',dt)):
                    score=z@w;terms={k:v@w for k,v in parts.items()}
                    if np.max(np.abs(sum(terms.values())-score))>1e-8:raise AssertionError('Score reconstruction')
                    row=dict(**base,score=kind,component='full',**metrics(y,score),mean_score=float(score.mean()),common_location=float(f['mu']@w))
                    for key,value in terms.items():
                        row[key+'_mean']=float(value.mean())
                        row[key+'_separation']=float(value[y].mean()-value[~y].mean()) if y.any() and (~y).any() else float('nan')
                    shift=tn-tp;tf=frame['text'];outside=shift-(shift@tf['q'])@tf['q'].T
                    row.update(negation_movement=float(np.linalg.norm(shift)),noise_approach=float(shift@tf['b']),
                        text_outside_squared_fraction=float(outside@outside/(shift@shift)) if shift@shift>0 else float('nan'))
                    rows.append(row)
                g=z@dt
                rows.append(dict(**base,score='PNC_centered',component='full',**metrics(y,g-f['mu']@dt)))
                for key,value in parts.items():rows.append(dict(**base,score='PNC',component=key,**metrics(y,value@dt)))
    with workspace('main',name) as work:
        table(work/'medical.csv',rows)
        import pandas as pd
        df=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(15,6))
        df[df.component=='full'].groupby(['model','score']).auc.mean().unstack().plot.bar(ax=axes[0]);axes[0].set(ylabel='AUC')
        df[df.score=='PNC'].groupby(['model','component']).auc.mean().unstack().plot.bar(ax=axes[1]);axes[1].set(ylabel='PNC AUC')
        fig.tight_layout();fig.savefig(work/'medical.png',dpi=180);plt.close(fig)


def run_reference_controls(name,models,config):
    import residual_core as core
    with workspace('sub',name) as work:
        for model in models:
            image,text,noise=inputs(model,'coco')
            for seed in config.seeds:
                frame,z,t,test,policy=split_frame(image,text,noise,seed,config.max_train_images)
                mask=np.isin(text['image_id'],image['ids'][test])
                lookup={str(v):i for i,v in enumerate(image['ids'][test])}
                mapping=np.asarray([lookup[str(v)] for v in text['image_id'][mask]])
                meta=dict(run_id=f'{model}_seed_{seed}',seed=seed,method='clip_only',lambda_axis=0.,epoch=0)
                snapshot=core.Snapshot(meta,z,t,z@t.T,mapping,image['ids'][test],text['ids'][mask],frame)
                context=core.AnalysisContext(config,work/f'{model}_{seed}',snapshot)
                try:
                    for fn in core.EXPERIMENTS:fn(snapshot,context,render=True)
                finally:context.close()

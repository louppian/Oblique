"""New four-component positive-score analysis; fixed references and cached embeddings."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_embedding_cache import load_cache, ROOT
from geometry import fit_frame, components, unit, residual
from output import workspace

PAIRS=(('clip','coco'),('clip','nih'),('carzero','coco'),('carzero','nih'))
TERMS=('Rhat','Bhat','E','Delta')
LABELS=(r'$\alpha\hat r$',r'$\beta\hat b$',r'$e_u$',r'$\delta_u$')
COLORS=('#2a78d6','#eb6834','#eda100','#1baf7a')
SEEDS=(0,1,2)
N_REFERENCE=2000
CHUNK=256
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})

def evaluate(scores,targets):
    """Per-prompt AUC and positive-minus-negative score gap, including tied ranks."""
    n,m=scores.shape
    if targets.ndim==1:
        pos=scores[targets,torch.arange(m,device=scores.device)]
        auc=((scores<pos).sum(0)+0.5*((scores==pos).sum(0)-1))/(n-1)
        gap=(pos*n-scores.sum(0))/(n-1)
    else:
        count=targets.sum(0);negative=n-count
        values,index=scores.sort(dim=0,stable=True)
        ranks=torch.arange(1,n+1,device=scores.device,dtype=scores.dtype)[:,None].expand(-1,m)
        start=torch.cat((torch.ones((1,m),device=scores.device,dtype=torch.bool),values[1:]!=values[:-1]),dim=0)
        end=torch.cat((values[1:]!=values[:-1],torch.ones((1,m),device=scores.device,dtype=torch.bool)),dim=0)
        lo=torch.where(start,ranks,0).cummax(0).values
        hi=torch.where(end,ranks,n+1).flip(0).cummin(0).values.flip(0)
        rank_sum=(((lo+hi)/2)*targets.gather(0,index)).sum(0)
        auc=(rank_sum-count*(count+1)/2)/(count*negative)
        gap=(scores*targets).sum(0)/count-(scores*(~targets)).sum(0)/negative
        valid=(count>0)&(negative>0)
        auc=torch.where(valid,auc,torch.nan);gap=torch.where(valid,gap,torch.nan)
    return auc,gap

def score_rows(z,t,parts,frame,targets,base,text_ids):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tensor=lambda a:torch.as_tensor(np.ascontiguousarray(a),device=device,dtype=torch.float32)
    zz,tt=tensor(z),tensor(t)
    pp={k:tensor(parts[k]) for k in TERMS}
    variants={'original':(zz,tt)}
    for key in TERMS:variants['remove_'+key+'_image']=(tensor(unit(z-parts[key])),tt)
    for label,q in [('outside_image_plane_both',frame['q']),('delta_image_space_both',frame['q3'])]:
        variants[label]=(tensor(unit(residual(z,q))),tensor(unit(residual(t,q))))
    targets=torch.as_tensor(targets,device=device)
    rows=[];reconstruction=0.
    with torch.inference_mode():
        for start in range(0,len(t),CHUNK):
            end=min(start+CHUNK,len(t));target=targets[start:end] if targets.ndim==1 else targets[:,start:end]
            text=tt[start:end].T;scores=zz@text
            terms=torch.stack([pp[k]@text for k in TERMS])
            error=(terms.sum(0)-scores).abs().max()
            shares=terms.abs().mean(1);shares=shares/shares.sum(0)
            term_metrics=[evaluate(s,target) for s in terms]
            variant_metrics=[evaluate(a@b[start:end].T,target) for a,b in variants.values()]
            packed=torch.stack([*shares,*[q[0] for q in term_metrics],*[q[1] for q in term_metrics],*[q[0] for q in variant_metrics],*[q[1] for q in variant_metrics]])
            data=packed.cpu().numpy();reconstruction=max(reconstruction,float(error.cpu()))
            vnames=list(variants);nv=len(vnames)
            for j,index in enumerate(range(start,end)):
                common=dict(**base,prompt_id=str(text_ids[index]))
                original_gap=data[12+nv,j]
                if not np.isclose(data[8:12,j].sum(),original_gap,atol=2e-5,equal_nan=True):raise AssertionError('Score-gap reconstruction')
                for k,key in enumerate(TERMS):
                    rows.append(dict(**common,analysis_type='score_component',component=key,absolute_share=float(data[k,j]),auc=float(data[4+k,j]),gap=float(data[8+k,j]),original_auc=float(data[12,j])))
                for k,name in enumerate(vnames):
                    rows.append(dict(**common,analysis_type='score_variant',variant=name,auc=float(data[12+k,j]),gap=float(data[12+nv+k,j]),original_auc=float(data[12,j]),delta_auc=float(data[12+k,j]-data[12,j])))
    if reconstruction>2e-5:raise AssertionError(f'Score reconstruction {reconstruction}')
    rows.append(dict(**base,analysis_type='score_check',score_reconstruction_error=reconstruction,n_images=len(z),n_prompts=len(t)))
    print(f"{base['model']}/{base['dataset']} seed={base['seed']}: {len(z)} images, {len(t)} positive prompts; score error={reconstruction:.2g}",flush=True)
    return rows

def run(name='05_positive_score_decomposition'):
    rows=[]
    with threadpool_limits(limits=4):
        for model,dataset in PAIRS:
            image,text,noise=[load_cache(ROOT/'cache'/model/dataset/f'{kind}.npz') for kind in ('image','text','noise')]
            if any(d['metadata']!=image['metadata'] for d in (text,noise)):raise ValueError('Cache metadata differs')
            z=unit(image['unit']);t=unit(text['unit'])
            for seed in SEEDS:
                order=np.random.default_rng(seed).permutation(len(z));ref,ev=order[:N_REFERENCE],order[N_REFERENCE:]
                if dataset=='coco':
                    select=np.isin(text['image_id'],image['ids'][ref]);test=np.isin(text['image_id'],image['ids'][ev])
                    positions={str(v):i for i,v in enumerate(image['ids'][ev])}
                    target=np.array([positions[str(v)] for v in text['image_id'][test]])
                    policy='image_disjoint_reference'
                else:
                    select=text['polarity']=='pos';test=select.copy()
                    target=image['labels'][ev][:,text['disease'][test].astype(int)].astype(bool)
                    policy='fixed_positive_prompt_reference'
                frame=fit_frame(z[ref],t[select],noise['image_unit'],noise['text_unit'])
                base=dict(model=model,dataset=dataset,seed=seed,definition='alpha_rhat+beta_bhat+e+delta',reference_policy=policy,n_reference_images=len(ref),n_reference_texts=int(select.sum()))
                for modality,x in [('image',z[ev]),('text',t[test])]:
                    f=frame[modality];parts=components(x,f);common=dict(**base,modality=modality)
                    energy={k:float(np.mean(np.sum(v*v,axis=1))) for k,v in parts.items()}
                    variance={k:float(np.var(v,axis=0).sum()) for k,v in parts.items()}
                    cross=float(2*np.mean(np.sum(parts['Rhat']*parts['Bhat'],axis=1)))
                    cov=float(2*np.mean(np.sum((parts['Rhat']-parts['Rhat'].mean(0))*(parts['Bhat']-parts['Bhat'].mean(0)),axis=1)))
                    reconstruction=float(np.max(np.abs(sum(parts.values())-x)))
                    energy_error=abs(sum(energy.values())+cross-1)
                    variance_error=abs(sum(variance.values())+cov-np.var(x,axis=0).sum())
                    if max(reconstruction,energy_error,variance_error)>1e-8:raise AssertionError('Component reconstruction')
                    for k in TERMS:rows.append(dict(**common,analysis_type='components',component=k,mean_squared_norm=energy[k],centered_variance=variance[k]))
                    rows.append(dict(**common,analysis_type='components',component='cross_Rhat_Bhat',mean_squared_norm=cross,centered_variance=cov))
                    rows.append(dict(**common,analysis_type='component_check',reconstruction_error=reconstruction,energy_error=energy_error,variance_error=variance_error,e_plane_error=float(np.max(np.abs(f['q'].T@f['e'])))))
                    other=np.roll(x,1,axis=0);actual=np.sum(x*other,axis=1)
                    for space,q in [('plane',f['q']),('plane_plus_E',f['q3'])]:
                        p=x@q;mid=np.sum(p*np.roll(p,1,axis=0),axis=1);length=np.maximum(0,1-np.sum(p*p,axis=1));radius=np.sqrt(length*np.roll(length,1))
                        lower,upper=mid-radius,mid+radius
                        if np.any(actual<lower-1e-8) or np.any(actual>upper+1e-8):raise AssertionError('Cosine bounds')
                        if space=='plane':prior=(lower,upper)
                        elif np.any(lower<prior[0]-1e-8) or np.any(upper>prior[1]+1e-8):raise AssertionError('Nested cosine bounds')
                        rows.append(dict(**common,analysis_type='cosine_bounds',space=space,pairing='consecutive_evaluation_rows',lower_mean=float(lower.mean()),upper_mean=float(upper.mean()),actual_mean=float(actual.mean())))
                ids=text.get('ids',np.arange(len(t)))[test]
                rows.extend(score_rows(z[ev],t[test],components(z[ev],frame['image']),frame['image'],target,base,ids))
    df=pd.DataFrame(rows)
    for (model,dataset,seed,key),g in df[df.analysis_type.eq('score_component')].groupby(['model','dataset','seed','component']):
        valid=g[['gap','original_auc']].dropna();rho,p=spearmanr(valid.gap,valid.original_auc)
        rows.append(dict(model=model,dataset=dataset,seed=seed,definition='alpha_rhat+beta_bhat+e+delta',analysis_type='gap_auc_correlation',component=key,rho=float(rho),p_value=float(p),n_prompts=len(valid)))
    df=pd.DataFrame(rows)
    with workspace('main',name) as work:
        df.to_csv(work/'new_definition.csv',index=False)
        plot(df,work/'new_definition.png')

def plot(df,path):
    fig,axes=plt.subplots(3,2,figsize=(18,15),layout='constrained')
    pair_index=pd.MultiIndex.from_tuples(PAIRS,names=['model','dataset'])
    pair_labels=[f'{m.upper()} / {d.upper()}' for m,d in PAIRS]
    g=df[df.analysis_type.eq('components') & df.modality.eq('image')].groupby(['model','dataset','component'])[['mean_squared_norm','centered_variance']].mean()
    for ax,key,title in zip(axes[0],['mean_squared_norm','centered_variance'],['(a) Image component magnitude','(b) Image component variation']):
        frame=g[key].unstack().reindex(pair_index).reindex(columns=[*TERMS,'cross_Rhat_Bhat'])
        frame.plot.bar(ax=ax,color=[*COLORS,'#898781'],rot=0)
        ax.set_xticklabels(pair_labels);ax.set(xlabel='',ylabel='Contribution',title=title)
        ax.axhline(0,color='gray',lw=.7);ax.legend(ax.containers,[*LABELS,r'$\hat r$–$\hat b$ cross'],ncol=3,fontsize=9)
    ax=axes[1,0];g=df[df.analysis_type.eq('cosine_bounds') & df.modality.eq('image')].groupby(['model','dataset','space'])[['lower_mean','upper_mean','actual_mean']].mean()
    for k,space in enumerate(['plane','plane_plus_E']):
        f=g.xs(space,level='space').reindex(pair_index);x=np.arange(4)+(k-.5)*.2
        ax.vlines(x,f.lower_mean,f.upper_mean,color=COLORS[k],lw=4,label='Plane' if k==0 else 'Plane + e')
        ax.scatter(x,f.actual_mean,color='black',s=18,label='Actual cosine' if k==0 else None)
    ax.set_xticks(range(4),pair_labels);ax.set(title='(c) Image cosine bounds: fixed projection',ylabel='Pair-average cosine');ax.legend(fontsize=9)
    ax=axes[1,1];g=df[df.analysis_type.eq('score_component')].groupby(['model','dataset','seed','component']).absolute_share.mean().groupby(['model','dataset','component']).mean().unstack().reindex(pair_index)[list(TERMS)]
    g.plot.bar(stacked=True,ax=ax,color=COLORS,rot=0);ax.set_xticklabels(pair_labels);ax.set(title='(d) Positive-score absolute contribution',xlabel='',ylabel='Share of sum of absolute terms',ylim=(0,1.13));ax.legend(ax.containers,LABELS,ncol=4,loc='upper center',fontsize=9)
    ax=axes[2,0];g=df[df.analysis_type.eq('score_component') & df.component.eq('Delta')].groupby(['model','dataset','prompt_id'])[['gap','original_auc']].mean()
    for (model,dataset),color in zip(PAIRS,COLORS):
        f=g.loc[(model,dataset)];pick=np.linspace(0,len(f)-1,min(2000,len(f)),dtype=int)
        ax.scatter(f.gap.iloc[pick],f.original_auc.iloc[pick],s=7,alpha=.35,color=color,label=f'{model.upper()} / {dataset.upper()}')
    ax.set(title=r'(e) $\delta$ score separation and original AUC',xlabel=r'Positive–negative mean gap of $\delta_z^\top t$',ylabel='Original AUC');ax.legend(fontsize=9)
    ax=axes[2,1];variants=['remove_'+k+'_image' for k in TERMS]+['outside_image_plane_both','delta_image_space_both']
    g=df[df.analysis_type.eq('score_variant')].groupby(['model','dataset','seed','variant']).delta_auc.mean().groupby(['model','dataset','variant']).mean().unstack().reindex(pair_index)[variants].T
    g.plot.bar(ax=ax,color=COLORS,rot=25);ax.set_xticklabels([r'$-\alpha\hat r$ (I)',r'$-\beta\hat b$ (I)',r'$-e$ (I)',r'$-\delta$ (I)','Outside plane (I,T)',r'$\delta$ space (I,T)'],ha='right');ax.set(title='(f) Component removal and renormalization',xlabel='',ylabel='Change in mean AUC');ax.legend(pair_labels,fontsize=8);ax.axhline(0,color='gray',lw=.7)
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(8)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/analysis/noise_residual_relation';OUT.mkdir(parents=True,exist_ok=True)
def unit(x):
    x=np.asarray(x,dtype=np.float64);return x/np.linalg.norm(x,axis=-1,keepdims=True)
def coords(u,b,r=None):
    r=unit(u.mean(0)) if r is None else r;g=r@b;c=u@r;n=u@b;beta=(n-g*c)/(1-g*g);s=np.arctan2((c-g*n)/np.sqrt(1-g*g),n);rho=np.sqrt(np.maximum(0,1-c*c-(1-g*g)*beta*beta))
    return s,rho,c,beta,n,r,g
rows=[];balanced=[]
for dataset,folder,ikey,nkey in [('NIH','carzero/nih','all_image.npz','noise.npz'),('COCO','clip/coco','image.npz','random_token_noise.npz')]:
    p=ROOT/'cache'/folder;noise=np.load(p/'noise.npz')
    for modality,key in [('image',ikey),('text','text.npz')]:
        u=unit(np.load(p/key)['unit']);b=unit(unit(noise['image_unit'] if modality=='image' else (np.load(p/nkey)['unit'] if dataset=='COCO' else noise['text_unit'])).mean(0))
        s,rho,c,beta,n,r,g=coords(u,b);corr=lambda a,b:float(np.corrcoef(a,b)[0,1])
        rows.append(dict(dataset=dataset,modality=modality,N=len(u),corr_s_rho=corr(s,rho),corr_beta_rho=corr(beta,rho),corr_beta_c=corr(beta,c),corr_c_rho=corr(c,rho),corr_n_rho=corr(n,rho),gamma=g,cov_beta_c=float(np.mean((beta-beta.mean())*(c-c.mean()))),identity_error=float(np.max(abs(rho*rho+c*c+(1-g*g)*beta*beta-1)))))
        for seed in range(20):
            ix=np.random.default_rng(seed).choice(len(u),300,replace=False)
            ss,rr,*_=coords(u[ix],b)
            balanced.append(dict(dataset=dataset,modality=modality,seed=seed,N=300,fixed_reference=corr(s[ix],rho[ix]),refit_mean=corr(ss,rr)))
f=pd.DataFrame(rows);f.to_csv(OUT/'full.csv',index=False);h=pd.DataFrame(balanced);h.to_csv(OUT/'balanced.csv',index=False)
print(f.to_string(index=False));print(h.groupby(['dataset','modality'])[['fixed_reference','refit_mean']].agg(['mean','min','max']).to_string())

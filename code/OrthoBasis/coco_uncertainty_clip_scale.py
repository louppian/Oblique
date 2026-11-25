from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from axis_only_loss_vs_rotation import rotation

torch.set_num_threads(8)
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent/'results/coco_uncertainty_clip_scale';out.mkdir(parents=True,exist_ok=True)
c=root/'cache/clip/coco';im=np.load(c/'image.npz');tx=np.load(c/'text.npz');sp=np.load(c/'residual_alignment/split.npz')
assert np.array_equal(im['ids'],sp['ids'])
lookup={str(v):j for j,v in enumerate(im['ids'])};owner=np.array([lookup[str(v)] for v in tx['image_id']])
ii=np.flatnonzero(sp['split']=='test');ti=np.flatnonzero(sp['split'][owner]=='test');local=np.full(len(im['ids']),-1);local[ii]=np.arange(len(ii));owner=torch.tensor(local[owner[ti]],dtype=torch.long,device='cuda')
x,t=[torch.tensor(d['unit'][ix],dtype=torch.float32,device='cuda') for d,ix in [(im,ii),(tx,ti)]]
w=torch.load(Path(__file__).resolve().parent/'results/coco_residual_direction_clip_30/maps.pt',map_location='cuda');bi,bt=w['image_basis'],w['text_basis']
bin,btn=bi.cpu().double().numpy(),bt.cpu().double().numpy();r=rotation(btn[:,0],bin[:,0]);r=rotation(r@btn[:,1],bin[:,1])@r;r=torch.tensor(r,dtype=torch.float32,device='cuda')
conditions={'original_clip':(F.normalize(x,dim=1),F.normalize(t,dim=1)),'axis':(F.normalize(x-(x@bi)@bi.T,dim=1),F.normalize((t-(t@bt)@bt.T)@r.T,dim=1))}
for name,folder in [('full_orth','coco_residual_direction_clip_30'),('basis_orth','coco_basis_orth_residual_clip_30')]:
    w=torch.load(Path(__file__).resolve().parent/('results/'+folder+'/maps.pt'),map_location='cuda');a=w['shared_basis'];z=[u@w[key].T for u,key in [(x,'image'),(t,'text')]]
    conditions[name]=tuple(F.normalize(v-(v@a)@a.T,dim=1) for v in z)
def auroc(y,s):
    ranks=pd.Series(s).rank(method='average').to_numpy();n=y.sum();return (ranks[y].sum()-n*(n+1)/2)/(n*(len(y)-n))
def aurc(err,unc):
    order=np.argsort(unc,kind='stable');e=err[order];return np.mean(np.cumsum(e)/np.arange(1,len(e)+1))
rows=[];bins=[]
for name,(xx,tt) in conditions.items():
    for direction,q,ca in [('I2T',xx,tt),('T2I',tt,xx)]:
        probs=[]
        for start in range(0,len(q),256):
            pp=F.softmax((q[start:start+256]@ca.T)*100.0,dim=1)
            if direction=='I2T':
                grouped=torch.zeros((len(pp),len(x)),device='cuda');grouped.scatter_add_(1,owner[None,:].expand(len(pp),-1),pp);pp=grouped
            probs.append(pp.cpu())
        pp=torch.cat(probs).numpy().astype('float64');truth=np.arange(len(x)) if direction=='I2T' else owner.cpu().numpy()
        pred=pp.argmax(1);conf=pp.max(1);correct=pred==truth;err=~correct;entropy=-(pp*np.log(np.clip(pp,1e-30,None))).sum(1)
        ece=0.
        for b in range(15):
            mask=(conf>=b/15)&((conf<(b+1)/15) if b<14 else (conf<=1))
            n=mask.sum();acc=correct[mask].mean() if n else np.nan;co=conf[mask].mean() if n else np.nan
            if n:ece+=n/len(conf)*abs(acc-co)
            bins.append(dict(condition=name,direction=direction,bin=b,count=n,confidence=co,accuracy=acc))
        target=pp[np.arange(len(pp)),truth]
        rows.append(dict(condition=name,direction=direction,accuracy=correct.mean(),confidence=conf.mean(),ece15=ece,nll=-np.log(np.clip(target,1e-30,None)).mean(),brier=(np.square(pp).sum(1)-2*target+1).mean(),entropy=entropy.mean(),error_auroc_entropy=auroc(err,entropy),error_auroc_1max=auroc(err,1-conf),aurc_entropy=aurc(err,entropy),aurc_1max=aurc(err,1-conf)))
        pd.DataFrame(dict(truth=truth,prediction=pred,correct=correct,confidence=conf,entropy=entropy)).to_csv(out/(name+'_'+direction+'_queries.csv'),index=False)
f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False);b=pd.DataFrame(bins);b.to_csv(out/'reliability_bins.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,d in zip(axes,['I2T','T2I']):
    ax.plot([0,1],[0,1],'k--',alpha=.5)
    for name in conditions:
        h=b[(b.direction==d)&(b.condition==name)&(b['count']>0)];ax.plot(h.confidence,h.accuracy,'o-',label=name)
    ax.set(title=d,xlabel='Confidence',ylabel='Accuracy',xlim=(0,1),ylim=(0,1));ax.legend()
fig.tight_layout();fig.savefig(out/'reliability.png',dpi=160)
(out/'protocol.txt').write_text('Test split; fixed pretrained CLIP exp(logit_scale)=100.0; ECE 15 equal-width bins; multiclass Brier summed over classes; natural-log NLL and entropy. I2T caption probabilities summed by owning image before argmax and all metrics (grouped accuracy differs from caption R@1). Error AUROC: error=positive, higher entropy or 1-max means uncertainty. AURC: mean cumulative error after sorting uncertainty ascending; lower better. No calibration fitting. Final epoch 30 for learned conditions.')
print(f.to_string(index=False))


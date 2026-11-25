from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from axis_only_loss_vs_rotation import rotation

torch.set_num_threads(8)
p=Path(__file__).resolve().parent/'results/coco_residual_direction_clip_30';c=Path(__file__).resolve().parents[2]/'cache/clip/coco'
w=torch.load(p/'maps.pt',map_location='cuda');bi,bt,a=w['image_basis'],w['text_basis'],w['shared_basis']
bin,btn=bi.cpu().double().numpy(),bt.cpu().double().numpy()
r=rotation(btn[:,0],bin[:,0]);r=rotation(r@btn[:,1],bin[:,1])@r;r=torch.tensor(r,dtype=torch.float32,device='cuda')
im,tx=np.load(c/'image.npz'),np.load(c/'text.npz');sp=np.load(c/'residual_alignment/split.npz')
assert np.array_equal(im['ids'],sp['ids'])
lookup={str(v):j for j,v in enumerate(im['ids'])};own=np.array([lookup[str(v)] for v in tx['image_id']])
x,t=[torch.tensor(d['unit'],dtype=torch.float32,device='cuda') for d in (im,tx)]
normalize=lambda z:F.normalize(z,dim=1)
base=(normalize(x-(x@bi)@bi.T),normalize((t-(t@bt)@bt.T)@r.T))
zx,zt=x@w['image'].T,t@w['text'].T
learn=(normalize(zx-(zx@a)@a.T),normalize(zt-(zt@a)@a.T))
rows=[];samples=[]
for split in ('train','validation','test'):
    ii=np.flatnonzero(sp['split']==split);ti=np.flatnonzero(sp['split'][own]==split)
    local=np.full(len(x),-1);local[ii]=np.arange(len(ii));owner=torch.tensor(local[own[ti]],device='cuda')
    for condition,(xx,tt) in [('axis',base),('learned',learn)]:
        xx,tt=xx[ii],tt[ti]
        for direction,q,ca in [('I2T',xx,tt),('T2I',tt,xx)]:
            values=[]
            for start in range(0,len(q),256):
                idx=torch.arange(start,min(start+256,len(q)),device='cuda');score=q[idx]@ca.T
                positive=owner[None,:]==idx[:,None] if direction=='I2T' else torch.arange(len(ca),device='cuda')[None,:]==owner[idx,None]
                pos=score.masked_fill(~positive,-torch.inf).max(1).values;neg=score.masked_fill(positive,-torch.inf).max(1).values
                posmean=(score*positive).sum(1)/positive.sum(1);negmean=(score*(~positive)).sum(1)/(~positive).sum(1)
                top=score.topk(10,dim=1).indices;correct=positive.gather(1,top)
                ce=-(F.log_softmax(score/.07,dim=1)*positive).sum(1)/positive.sum(1)
                values.append(torch.stack([correct[:,:k].any(1).float() for k in (1,5,10)]+[pos,neg,pos-neg,posmean,negmean,ce],dim=1))
            v=torch.cat(values).cpu().numpy();cols=['r1','r5','r10','best_positive','hardest_negative','margin','mean_positive','mean_negative','ce']
            rows.append(dict(split=split,condition=condition,direction=direction,**dict(zip(cols,v.mean(0).tolist()))))
            if split=='test':
                f=pd.DataFrame(v,columns=cols);f['condition']=condition;f['direction']=direction;f['query_index']=np.arange(len(f));samples.append(f)
f=pd.DataFrame(rows);f.to_csv(p/'weight_diagnostics.csv',index=False);pd.concat(samples).to_csv(p/'weight_diagnostics_test_queries.csv',index=False)
print(f.to_string(index=False))
g={}
for key,b in [('image',bi),('text',bt)]:
    o=w[key];sv=torch.linalg.svdvals(o);leak=(o@b)-a@(a.T@o@b)
    g[key]=dict(singular_min=sv.min().item(),singular_max=sv.max().item(),singular_rms_deviation=(sv-1).square().mean().sqrt().item(),plane_leak_fro=leak.norm().item(),orth_fro=(o.T@o-torch.eye(len(o),device='cuda')).norm().item())
for d in ['I2T','T2I']:
    b=samples[0 if d=='I2T' else 1];l=samples[2 if d=='I2T' else 3]
    g[d]=dict(lost=int(((b.r1==1)&(l.r1==0)).sum()),gained=int(((b.r1==0)&(l.r1==1)).sum()))
print(json.dumps(g,indent=2));(p/'weight_geometry.json').write_text(json.dumps(g,indent=2))

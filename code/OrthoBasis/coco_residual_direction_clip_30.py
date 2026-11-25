"""COCO: symmetric CLIP CE on normalized common-plane residuals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent/'results/coco_residual_direction_clip_30'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, BATCH, TAU = 30, 512, .07

def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def frame(x, noise):
    r, b = unit(x.mean(0)), unit(noise.mean(0))
    return np.stack((r, unit(b-(r@b)*r)), 1), b

def rotation(a, b):
    k = np.outer(b,a)-np.outer(a,b)
    return np.eye(len(a))+k+k@k/(1+a@b)

def shared_residual(x, t, wi, wt):
    a = torch.linalg.qr((wi@bi+wt@bt)*.5, mode='reduced').Q
    zi, zt = x@wi.T, t@wt.T
    return F.normalize(zi-(zi@a)@a.T,dim=1), F.normalize(zt-(zt@a)@a.T,dim=1)

@torch.no_grad()
def evaluate(x, t, owner, condition, epoch, rows):
    recalls, entropies = [], []
    for direction, queries, candidates in [('I2T',x,t),('T2I',t,x)]:
        hits = torch.zeros(3,device=DEVICE)
        entropy = torch.zeros((),device=DEVICE)
        for start in range(0,len(queries),256):
            ix = torch.arange(start,min(start+256,len(queries)),device=DEVICE)
            scores = queries[ix]@candidates.T
            top = scores.topk(10,dim=1).indices
            correct = owner[top]==ix[:,None] if direction=='I2T' else top==owner[ix,None]
            hits += torch.stack([correct[:,:k].any(1).sum() for k in (1,5,10)])
            logp = F.log_softmax(scores/TAU,dim=1)
            entropy -= (logp.exp()*logp).sum()
        values = torch.cat((hits/len(queries),(entropy/len(queries)).reshape(1))).cpu().tolist()
        recalls.extend(values[:3]);entropies.append(values[3])
        if rows is not None:
            rows.append(dict(kind='retrieval',condition=condition,epoch=epoch,direction=direction,
                             r1=values[0],r5=values[1],r10=values[2],entropy=values[3],
                             queries=len(queries),candidates=len(candidates)))
    return float(np.mean(recalls))

@torch.no_grad()
def geometry(wi,wt,condition,epoch,rows):
    ai,at = wi@bi,wt@bt
    cos = (F.normalize(ai,dim=0)*F.normalize(at,dim=0)).sum(0).clamp(-1,1)
    error = torch.stack(((ai-at).square().sum(),(wi.T@wi-eye).square().sum(),
                         (wt.T@wt-eye).square().sum()))
    gi = F.cosine_similarity(wi@bi[:,0],wi@ni,dim=0)
    gt = F.cosine_similarity(wt@bt[:,0],wt@nt,dim=0)
    values = torch.cat((torch.rad2deg(torch.acos(cos)),error,gi[None],gt[None])).cpu().tolist()
    rows.append(dict(kind='geometry',condition=condition,epoch=epoch,mean_axis_angle=values[0],
                     transverse_axis_angle=values[1],basis_loss=values[2],orth_image=values[3],
                     orth_text=values[4],gamma_image=values[5],gamma_text=values[6]))

if __name__ == '__main__':
    image=np.load(ROOT/'cache/clip/coco/image.npz');text=np.load(ROOT/'cache/clip/coco/text.npz')
    noise=np.load(ROOT/'cache/clip/coco/noise.npz');split=np.load(ROOT/'cache/clip/coco/residual_alignment/split.npz')
    assert np.array_equal(image['ids'],split['ids'])
    lookup={str(v):j for j,v in enumerate(image['ids'])}
    owner=np.array([lookup[str(v)] for v in text['image_id']]);split=split['split']
    x,t=image['unit'].astype('float64'),text['unit'].astype('float64')
    bi,ni=frame(x[split=='train'],noise['image_unit'].astype('float64'))
    bt,nt=frame(t[split[owner]=='train'],noise['text_unit'].astype('float64'))
    rotate=rotation(bt[:,0],bi[:,0]);rotate=rotation(rotate@bt[:,1],bi[:,1])@rotate
    np.testing.assert_allclose(rotate@bt,bi,atol=1e-10)
    np.testing.assert_allclose(rotate.T@rotate,np.eye(x.shape[1]),atol=1e-10)
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=DEVICE)
    x,t,bi,bt,ni,nt,rotate=map(tensor,(x,t,bi,bt,ni,nt,rotate))
    vi,vt=x-(x@bi)@bi.T,t-(t@bt)@bt.T
    eye=torch.eye(x.shape[1],device=DEVICE);groups={};rows=[]
    for name in ('train','validation','test'):
        ii=np.flatnonzero(split==name);ti=np.flatnonzero(split[owner]==name)
        local=np.full(len(x),-1);local[ii]=np.arange(len(ii))
        groups[name]=tuple(torch.as_tensor(a,device=DEVICE) for a in (ii,ti,local[owner[ti]]))
    ii,ti,local=groups['test']
    evaluate(x[ii],t[ti],local,'original_full',0,rows)
    evaluate(vi[ii],vt[ti],local,'original_residual',0,rows)
    evaluate(F.normalize(vi[ii],dim=1),F.normalize(vt[ti]@rotate.T,dim=1),local,'axis_rotation_cosine',0,rows)
    geometry(eye,eye,'original_residual',0,rows);geometry(eye,rotate,'axis_rotation',0,rows)
    wi=torch.nn.Parameter(eye.clone());wt=torch.nn.Parameter(eye.clone())
    optimizer=torch.optim.Adam([wi,wt],lr=1e-3)
    owner_gpu=torch.as_tensor(owner,device=DEVICE);best=-np.inf;chosen=0
    protocol=dict(model='cached CLIP ViT-B/32',seed=0,epochs=EPOCHS,batch=BATCH,lr=.001,tau=TAU,
                  train_images=3500,validation_images=500,test_images=1000,
                  maps='two bias-free 512x512 linear layers, identity initialized',
                  logits='normalize(vi).T normalize(vt)/0.07; vm=(I-AA.T) Om um; A=qr((Oi Bi+Ot Bt)/2)',
                  loss='0.5*(CE(image_to_text)+CE(text_to_image)) + ||Oi Bi-Ot Bt||_F^2 + sum ||Om.T Om-I||_F^2',
                  coefficients=dict(task=1,basis=1,orth=1),
                  reference='means from training images/captions; existing uniform RGB and random text noise cache',
                  axis_baseline='image identity; text two proper minimal rotations: mean then transverse noise axis',
                  selection='fixed final epoch 30; validation used for reporting only',
                  note='projection after mapping; same differentiable shared QR plane for both modalities in train, validation and test; tau=0.07 in CLIP CE; caption-pair batches; duplicate image owners treated as positives with uniform positive target')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    va,vta,vo=groups['validation'];train=groups['train'][1]
    for epoch in range(1,EPOCHS+1):
        order=train[torch.randperm(len(train),device=DEVICE)];total=torch.zeros(4,device=DEVICE)
        for ix in order.split(BATCH):
            ri,rt=shared_residual(x[owner_gpu[ix]],t[ix],wi,wt)
            logits=ri@rt.T/TAU
            positive=(owner_gpu[ix,None]==owner_gpu[ix][None,:]).float()
            target=positive/positive.sum(1,keepdim=True)
            task=-.5*((target*F.log_softmax(logits,dim=1)).sum(1).mean()+(target*F.log_softmax(logits.T,dim=1)).sum(1).mean())
            basis=(wi@bi-wt@bt).square().sum()
            orth=(wi.T@wi-eye).square().sum()+(wt.T@wt-eye).square().sum()
            loss=task+basis+orth
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=torch.stack((loss,task,basis,orth)).detach()*len(ix)
        with torch.no_grad():
            rv,tv=shared_residual(x[va],t[vta],wi,wt)
            score=evaluate(rv,tv,vo,'validation',epoch,rows)
        values=(total/len(train)).cpu().tolist()
        rows.append(dict(kind='training',condition='learned',epoch=epoch,loss=values[0],task_loss=values[1],
                         basis_loss=values[2],orth_loss=values[3],validation_mean_recall=score))
        geometry(wi,wt,'learned_epoch',epoch,rows)
        if epoch==EPOCHS:
            best=score;chosen=epoch;weights=(wi.detach().clone(),wt.detach().clone())
        pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
        print(f'epoch={epoch} loss={values[0]:.5f} task={values[1]:.5f} basis={values[2]:.5f} orth={values[3]:.5f} val={score:.5f}',flush=True)
    wi,wt=weights
    ri,rt=shared_residual(x[ii],t[ti],wi,wt)
    a=torch.linalg.qr((wi@bi+wt@bt)*.5,mode='reduced').Q
    assert max((ri@a).abs().max().item(),(rt@a).abs().max().item()) < 1e-5
    evaluate(ri,rt,local,'learned',chosen,rows)
    geometry(wi,wt,'learned',chosen,rows)
    torch.save(dict(image=wi.cpu(),text=wt.cpu(),image_basis=bi.cpu(),text_basis=bt.cpu(),shared_basis=a.cpu(),selected_epoch=chosen,protocol=protocol),OUT/'maps.pt')
    df=pd.DataFrame(rows);df.to_csv(OUT/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    r=df[(df.kind=='retrieval')&(df.condition!='validation')]
    for ax,direction in zip(axes[:2],('I2T','T2I')):
        r[r.direction==direction].set_index('condition')[['r1','r5','r10']].mul(100).plot.bar(ax=ax,rot=25)
        ax.set(title=direction,ylabel='Recall (%)',xlabel='')
    h=df[df.kind=='training'];axes[2].plot(h.epoch,h.validation_mean_recall*100,'o-')
    axes[2].axvline(chosen,color='gray',ls='--');axes[2].set(title='Validation tracking (no selection)',xlabel='Epoch',ylabel='Mean Recall (%)')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    print(r[['condition','direction','r1','r5','r10','entropy']].to_string(index=False),flush=True)
    print(df[(df.kind=='geometry')&(df.condition!='learned_epoch')].dropna(axis=1,how='all').to_string(index=False),flush=True)

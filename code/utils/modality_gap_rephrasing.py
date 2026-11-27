"""A-OKVQA Fig. 10 protocol, 500 sampled answer sets per question."""
from pathlib import Path
import json
import numpy as np
import torch,open_clip
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[2]

def run(siglip=False):
    torch.set_num_threads(8)
    name='SigLIP_VIT-B_16' if siglip else 'VIT-B_16'
    out=ROOT/'code/analysis'/f'AOKVQA_{name}'/'modality_gap'
    cache=ROOT/'cache/clip'/('aokvqa_siglip_vitb16' if siglip else 'aokvqa_vitb16')
    rows=json.loads((ROOT/'data/aokvqa/aokvqa_v1p0_val.json').read_text())
    train=json.loads((ROOT/'data/aokvqa/aokvqa_v1p0_train.json').read_text())
    bank=np.array(sorted({a for r in train+rows for a in r['choices']+r['direct_answers']}))
    a=np.load(cache/'embeddings.npz'); x=torch.tensor(a['images'],device='cuda'); t=torch.tensor(a['texts'],device='cuda')
    centered=t.double()-t.double().mean(1,keepdim=True); _,s,v=torch.linalg.svd(centered,full_matrices=False)
    g=t.double().mean(1)-x.double(); b=v*(s>s[:,:1]*1e-6).unsqueeze(-1)
    d=(g-torch.einsum('nkd,nk->nd',b,torch.einsum('nd,nkd->nk',g,b))).float()
    steps=np.linspace(-2,1,13)
    path=cache/'rephrasing.npz'
    counts=np.load(path)['counts'] if path.exists() else np.full((len(rows),len(steps)),-1,dtype=np.int32)
    if siglip:
        from open_clip.tokenizer import HFTokenizer
        model,_,_=open_clip.create_model_and_transforms('ViT-B-16-SigLIP',pretrained=str(ROOT/'ckpt/ViT-B-16-SigLIP/open_clip_pytorch_model.bin'),device='cuda')
        tokenize=HFTokenizer(str(ROOT/'ckpt/ViT-B-16-SigLIP'),context_length=64,clean='canonicalize')
    else:
        model,_,_=open_clip.create_model_and_transforms('ViT-B-16',pretrained=str(ROOT/'ckpt/ViT-B-16.pt'),force_quick_gelu=True,device='cuda',load_weights_only=False)
        tokenize=open_clip.get_tokenizer('ViT-B-16')
    model.eval()
    with torch.inference_mode():
        for i,r in enumerate(rows):
            if counts[i,0]>=0:continue
            rng=np.random.default_rng(2026+i)
            correct=r['direct_answers'] or [r['choices'][r['correct_choice_idx']]]
            excluded=set(a.strip().lower() for a in correct+[r['choices'][r['correct_choice_idx']]])
            wrong=bank[[a.strip().lower() not in excluded for a in bank]]
            sampled=np.empty((500,4),dtype=object); sampled[:,0]=rng.choice(correct,500)
            for k in range(500):sampled[k,1:]=rng.choice(wrong,3,replace=False)
            prompts=[r['question']+' '+answer for answer in sampled.ravel()]
            unique,inverse=np.unique(prompts,return_inverse=True); embeddings=[]
            with torch.autocast('cuda'):
                for start in range(0,len(unique),512):embeddings.append(model.encode_text(tokenize(unique[start:start+512].tolist()).cuda(),normalize=True).float())
            z=torch.cat(embeddings)[torch.tensor(inverse,device='cuda')].reshape(500,4,-1)
            base=(z*x[i]).sum(-1)-.5*z.square().sum(-1); offset=(z*d[i]).sum(-1)
            counts[i]=torch.stack([((base+float(step)*offset).argmax(-1)==0).sum() for step in steps]).cpu().numpy()
            if i%25==24 or i==len(rows)-1:
                np.savez(path,counts=counts);print(name,'rephrasing',i+1,'/',len(rows),flush=True)
    frame=pd.DataFrame(dict(step=steps,gap=[float((g.float()-float(step)*d).norm(dim=-1).mean()) for step in steps],accuracy=counts.sum(0)/(len(rows)*500)))
    frame.to_csv(out/'rephrasing.csv',index=False,float_format='%.4f')
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);ax.plot(frame.gap,frame.accuracy,'o-');ax.set(xlabel='Mean per-question gap norm',ylabel='Rephrased answer accuracy');ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'));fig.savefig(out/'figure2.png',dpi=180);plt.close(fig)
    with (out/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## Rephrasing (Fig. 10)\n\n500 answer sets per validation question. Correct option sampled from direct_answers, three distinct wrong options from the unique train+validation choices/direct_answers bank after excluding the question’s correct aliases. Answer-bank scope is an explicit implementation choice because the paper says entire answer bank without specifying splits. Clean four-option PCA and gap are retained for every rephrasing.\n\n```csv\n'+frame.to_csv(index=False,float_format='%.4f')+'```\n')
    print('Rephrasing complete',name,flush=True)

if __name__=='__main__':
    run(False)
    run(True)

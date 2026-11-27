from pathlib import Path
import json, sys
import numpy as np
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'code/utils'))
from modality_gap_reproduction import markdown_table
OUT=Path(__file__).resolve().parent
CACHE=ROOT/'cache/clip/aokvqa_vitb16'

class Images(Dataset):
    def __init__(self, rows, transform): self.rows,self.transform=rows,transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(ROOT/'data/coco/val2017'/f'{self.rows[i]["image_id"]:012d}.jpg') as image:
            return self.transform(image.convert('RGB'))

def main(siglip=False):
    global CACHE, OUT
    if siglip:
        CACHE=ROOT/'cache/clip/aokvqa_siglip_vitb16'
        OUT=ROOT/'code/analysis/AOKVQA_SigLIP_VIT-B_16/modality_gap'
        OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8); torch.manual_seed(2026); CACHE.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'data/aokvqa/aokvqa_v1p0_val.json').read_text())
    if not (CACHE/'embeddings.npz').exists():
        if siglip:
            from open_clip.tokenizer import HFTokenizer
            model,_,transform=open_clip.create_model_and_transforms('ViT-B-16-SigLIP',pretrained=str(ROOT/'ckpt/ViT-B-16-SigLIP/open_clip_pytorch_model.bin'),device='cuda',image_mean=(.5,.5,.5),image_std=(.5,.5,.5),image_resize_mode='squash')
            tokenizer=HFTokenizer(str(ROOT/'ckpt/ViT-B-16-SigLIP'),context_length=64,clean='canonicalize')
        else:
            model,_,transform=open_clip.create_model_and_transforms('ViT-B-16',pretrained=str(ROOT/'ckpt/ViT-B-16.pt'),force_quick_gelu=True,device='cuda',load_weights_only=False)
            tokenizer=open_clip.get_tokenizer('ViT-B-16')
        model.eval(); images=[]; texts=[]
        prompts=[r['question']+' '+choice for r in rows for choice in r['choices']]
        with torch.inference_mode(),torch.autocast('cuda'):
            for batch in DataLoader(Images(rows,transform),batch_size=128,num_workers=4,pin_memory=True):
                images.append(model.encode_image(batch.cuda(non_blocking=True),normalize=True).float().cpu().numpy())
            for start in range(0,len(prompts),512): texts.append(model.encode_text(tokenizer(prompts[start:start+512]).cuda(),normalize=True).float().cpu().numpy())
        np.savez(CACHE/'embeddings.npz',images=np.concatenate(images),texts=np.concatenate(texts).reshape(len(rows),4,-1),labels=[r['correct_choice_idx'] for r in rows])
        del model; torch.cuda.empty_cache()
        print('Embeddings ready',flush=True)
    a=np.load(CACHE/'embeddings.npz'); x=torch.tensor(a['images'],device='cuda'); t=torch.tensor(a['texts'],device='cuda'); y=torch.tensor(a['labels'],device='cuda')
    centered=t.double()-t.double().mean(1,keepdim=True)
    _,s,v=torch.linalg.svd(centered,full_matrices=False)
    g=t.double().mean(1)-x.double(); b=v*(s>s[:,:1]*1e-6).unsqueeze(-1)
    direction=(g-torch.einsum('nkd,nk->nd',b,torch.einsum('nd,nkd->nk',g,b))).float(); g=g.float()
    base=(x[:,None,:]*t).sum(-1)-.5*t.square().sum(-1); clean=base.argmax(-1); result=[]
    with torch.inference_mode():
        for sigma in (0.,.005,.0225,.04):
            torch.manual_seed(2026); noise=t[None]+sigma*torch.randn((100 if sigma else 1,*t.shape),device='cuda')
            score=(x[None,:,None,:]*noise).sum(-1)-.5*noise.square().sum(-1)
            change=(noise*direction[None,:,None,:]).sum(-1)
            for step in np.linspace(-2,1,13):
                pred=(score+float(step)*change).argmax(-1)
                if sigma==0: assert torch.equal(pred[0],clean)
                result.append(dict(sigma=sigma,step=step,gap=float((g-float(step)*direction).norm(dim=-1).mean()),accuracy=float((pred==y).float().mean()),robustness=float((pred==clean).float().mean())))
    frame=pd.DataFrame(result); frame.to_csv(OUT/'metrics.csv',index=False,float_format='%.4f')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for sigma,group in frame.groupby('sigma'):
        for ax,key in zip(axes,['accuracy','robustness']):
            ax.plot(group.gap,group[key],'.-',label=f'{sigma:.4f}'); ax.set(xlabel='Mean per-question gap norm',ylabel=key.title()); ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[0].legend(title='Noise std'); fig.savefig(OUT/'figure1.png',dpi=180); plt.close(fig)
    (OUT/'report.md').write_text('# A-OKVQA CLIP ViT-B/16\n\nValidation: 1,145 questions, 4 choices/question; frozen OpenAI CLIP ViT-B/16 (224 px). Prompt = question + space + answer. Initial embeddings are unit normalized. Candidate texts are translated, then evaluated by squared Euclidean distance.\n\nReproduction choices: PCA and gap are computed separately for each question using its four clean text candidates and its image. The paper specifies question–answer concatenation and the validation split; exact checkpoint identifier and PCA grouping are unspecified. Noise values are interpreted as standard deviations; 100 draws, seed 2026, shared noise across shift values. Clean prediction invariance is checked for every shift.\n\n'+markdown_table(frame[frame.step.isin([0.,1.])])+'\n\nCurrent result covers Gaussian noise (Fig. 8). Rephrasing (Fig. 10) remains pending.\n\nSource: https://arxiv.org/html/2603.29080v2\n',encoding='utf-8')
    print(frame[frame.step.isin([0.,1.])].to_string(index=False),flush=True)
    if siglip:
        path=OUT/'report.md'
        path.write_text(path.read_text(encoding='utf-8').replace('CLIP ViT-B/16','SigLIP ViT-B/16').replace('OpenAI SigLIP','WebLI SigLIP'),encoding='utf-8')

if __name__=='__main__': main()

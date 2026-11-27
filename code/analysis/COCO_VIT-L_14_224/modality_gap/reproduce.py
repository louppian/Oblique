from pathlib import Path
import sys,json
import numpy as np
import torch,open_clip
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'code/utils'))
from modality_gap_reproduction import gap_direction

class Images(Dataset):
    def __init__(self,ids,transform):self.ids,self.transform=ids,transform
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        with Image.open(ROOT/'data/coco/val2017'/f'{self.ids[i]:012d}.jpg') as im:return self.transform(im.convert('RGB'))

def main(siglip=False):
    torch.set_num_threads(8)
    name='SigLIP_VIT-B_16' if siglip else 'VIT-L_14_224'
    out=ROOT/'code/analysis'/f'COCO_{name}'/'modality_gap'; out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'cache/clip'/('coco_val_siglip_vitb16' if siglip else 'coco_val_vitl14_224'); cache.mkdir(parents=True,exist_ok=True)
    annotation=json.loads((ROOT/'data/coco/annotations/captions_val2017.json').read_text())
    ids=sorted(r['id'] for r in annotation['images']); lookup={v:k for k,v in enumerate(ids)}
    captions=sorted(annotation['annotations'],key=lambda r:r['id'])
    if not (cache/'embeddings.npz').exists():
        if siglip:
            from open_clip.tokenizer import HFTokenizer
            model,_,transform=open_clip.create_model_and_transforms('ViT-B-16-SigLIP',pretrained=str(ROOT/'ckpt/ViT-B-16-SigLIP/open_clip_pytorch_model.bin'),device='cuda',image_mean=(.5,.5,.5),image_std=(.5,.5,.5),image_resize_mode='squash')
            tokenizer=HFTokenizer(str(ROOT/'ckpt/ViT-B-16-SigLIP'),context_length=64,clean='canonicalize')
        else:
            model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(ROOT/'ckpt/ViT-L-14.pt'),device='cuda',force_quick_gelu=True,load_weights_only=False)
            tokenizer=open_clip.get_tokenizer('ViT-L-14')
        model.eval(); x=[];t=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            for batch in DataLoader(Images(ids,transform),batch_size=64,num_workers=4,pin_memory=True):x.append(model.encode_image(batch.cuda(non_blocking=True),normalize=True).float().cpu().numpy())
            for start in range(0,len(captions),256):t.append(model.encode_text(tokenizer([r['caption'] for r in captions[start:start+256]]).cuda(),normalize=True).float().cpu().numpy())
        np.savez(cache/'embeddings.npz',images=np.concatenate(x),texts=np.concatenate(t),image_ids=ids,caption_ids=[r['id'] for r in captions],caption_image=[lookup[r['image_id']] for r in captions])
        del model;torch.cuda.empty_cache();print('COCO embeddings ready',name,flush=True)
    a=np.load(cache/'embeddings.npz');x=torch.tensor(a['images'],device='cuda');t=torch.tensor(a['texts'],device='cuda');owner=torch.tensor(a['caption_image'],device='cuda');y=torch.arange(len(x),device='cuda')
    thresholds=(0.,1.,2.,5.,10.,15.,20.)
    directions=[gap_direction(x,t,e,singular_threshold=True)[0] for e in thresholds]
    configs=[(e,step,d) for e,d in zip(thresholds,directions) for step in (0.,1.)]
    configs += [(5.,float(step),directions[3]) for step in np.linspace(-1,1,9) if step not in (0.,1.)]
    noise_levels=(.005,.01,.025) if siglip else (.01,.02)
    stats=torch.zeros((len(noise_levels),len(configs),4),device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        clean=(x@t.T-.5*t.square().sum(1))
        clean_pred=[(clean+step*(t@d)).argmax(1) for e,step,d in configs]
        for j,p in enumerate(clean_pred):stats[:,j,0]=(owner[p]==y).double().mean()
        for k,sigma in enumerate(noise_levels):
            torch.manual_seed(2026)
            for draw in range(100):
                noisy=t+sigma*torch.randn_like(t); base=x@noisy.T-.5*noisy.square().sum(1)
                for j,(e,step,d) in enumerate(configs):
                    p=(base+step*(noisy@d)).argmax(1)
                    stats[k,j,1]+=(owner[p]==y).double().mean()/100
                    stats[k,j,2]+=(p==clean_pred[j]).double().mean()/100
                    stats[k,j,3]+=(owner[p]==owner[clean_pred[j]]).double().mean()/100
                if draw%20==19:print(name,'sigma',f'{sigma:.4f}','noise draws',draw+1,flush=True)
    gap=t.mean(0)-x.mean(0)
    frame=pd.DataFrame([dict(sigma=sigma,epsilon=e,step=s,gap=float((gap-s*d).norm()),clean_accuracy=v[0],noisy_accuracy=v[1],robustness_exact_caption=v[2],robustness_image_group=v[3]) for sigma,values in zip(noise_levels,stats.cpu().numpy()) for (e,s,d),v in zip(configs,values)])
    frame.to_csv(out/'metrics.csv',index=False,float_format='%.4f')
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    shifted=frame[(frame.step==1)&(frame.sigma==.01)];ax.plot(shifted.clean_accuracy,shifted.robustness_exact_caption,'o-')
    for r in shifted.itertuples():ax.annotate(f'{r.epsilon:.4f}',(r.clean_accuracy,r.robustness_exact_caption))
    ax.set(xlabel='Clean I2T recall@1',ylabel='Exact-caption robustness');ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'));fig.savefig(out/'figure1.png',dpi=180);plt.close(fig)
    gap=t.mean(0)-x.mean(0);fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for z,label in ((x,'image'),(t,'text')):
        z=z-z.mean(0);cos=((z@gap)/(z.norm(dim=1)*gap.norm())).cpu().numpy();ax.hist(cos,bins=70,density=True,histtype='step',label=label)
    ax.legend();ax.set(xlabel='Cosine(centered embedding, gap)',ylabel='Density');ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'));fig.savefig(out/'figure2.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for sigma,group in frame[frame.epsilon==5].groupby('sigma'):
        group=group.sort_values('gap');ax.plot(group.gap,group.robustness_exact_caption,label=f'Noise std {sigma:.4f}')
    ax.plot(group.gap,group.clean_accuracy,'k--',label='Clean R@1');ax.axvline(float(gap.norm()),color='gray',ls=':');ax.legend();ax.set(xlabel='Gap norm',ylabel='Accuracy / robustness');ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'));fig.savefig(out/'figure3.png',dpi=180);plt.close(fig)
    (out/'report.md').write_text(f'# COCO {name}: appendix reproduction\n\nCOCO val2017, {len(x)} images, {len(t)} captions. I2T recall@1 counts any caption belonging to the query image as correct. Frozen 224 px model; normalized initial embeddings; shifted squared Euclidean distance. Gaussian standard deviation 0.0100 (variance 0.0001), 100 shared draws, seed 2026.\n\nAlgorithm 1 threshold is applied to raw singular values of centered text embeddings, matching the Fig. 19 horizontal-axis label. The paper labels its noise parameter as variance; this implementation treats the plotted values as standard deviations, as in the main experiments. This convention is explicit and remains a discrepancy requiring author-code confirmation. Exact-caption robustness and image-group robustness are both recorded. Model checkpoint choice and the val2017 split are explicit reproduction choices.\n\n```csv\n'+frame.to_csv(index=False,float_format='%.4f')+'```\n\nfigure1: threshold versus accuracy/robustness (Fig. 19 protocol). figure2: centered embedding/gap orthogonality (Fig. 13 protocol).\n\nSource: https://arxiv.org/html/2603.29080v2\n',encoding='utf-8')
    print('Complete',out,flush=True)

if __name__=='__main__':main()

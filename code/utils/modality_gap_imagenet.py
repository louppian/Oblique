from pathlib import Path
import json
import numpy as np
import torch,open_clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
from modality_gap_reproduction import evaluate
ROOT=Path(__file__).resolve().parents[2]

def run(model_name):
    folder=ROOT/'data/imagenet/val'
    if not folder.exists():
        print('Pending ImageNet validation data:',folder,flush=True)
        return
    torch.set_num_threads(8)
    siglip='SigLIP' in model_name
    tag='SigLIP_VIT-B_16' if siglip else model_name.replace('ViT','VIT').replace('-14','_14').replace('-16','_16')
    cache=ROOT/'cache/clip'/f'imagenet_{tag.lower()}';cache.mkdir(parents=True,exist_ok=True)
    out=ROOT/'code/analysis'/f'IMAGENET_{tag}'/'modality_gap';out.mkdir(parents=True,exist_ok=True)
    if not (cache/'embeddings.npz').exists():
        if siglip:
            from open_clip.tokenizer import HFTokenizer
            model,_,transform=open_clip.create_model_and_transforms(model_name,pretrained=str(ROOT/'ckpt/ViT-B-16-SigLIP/open_clip_pytorch_model.bin'),device='cuda',image_mean=(.5,.5,.5),image_std=(.5,.5,.5),image_resize_mode='squash')
            tokenizer=HFTokenizer(str(ROOT/'ckpt/ViT-B-16-SigLIP'),context_length=64,clean='canonicalize')
        else:
            model,_,transform=open_clip.create_model_and_transforms(model_name,pretrained=str(ROOT/'ckpt'/f'{model_name}.pt'),force_quick_gelu=True,load_weights_only=False,device='cuda')
            tokenizer=open_clip.get_tokenizer(model_name)
        data=ImageFolder(folder,transform=transform)
        assert len(data)==50000 and len(data.classes)==1000 and (all(c.startswith('n') and c[1:].isdigit() for c in data.classes) or data.classes==[f'{i:04d}' for i in range(1000)]), 'Expected 50,000 validation images in 1,000 WNID or official numeric label folders'
        model.eval();images=[];texts=[];labels=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            for batch,y in DataLoader(data,batch_size=64,num_workers=4,pin_memory=True):
                images.append(model.encode_image(batch.cuda(non_blocking=True),normalize=True).float().cpu().numpy());labels.append(y.numpy())
            for start in range(0,1000,256):texts.append(model.encode_text(tokenizer([f'a photo of a {c}.' for c in IMAGENET_CLASSNAMES[start:start+256]]).cuda(),normalize=True).float().cpu().numpy())
        np.savez(cache/'embeddings.npz',images=np.concatenate(images),texts=np.concatenate(texts),labels=np.concatenate(labels))
        (cache/'metadata.json').write_text(json.dumps(dict(model=model_name,split='ImageNet-1K validation',classes=data.classes,prompt='a photo of a {class}.',label_names=IMAGENET_CLASSNAMES),indent=2),encoding='utf-8')
        del model;torch.cuda.empty_cache()
    a=np.load(cache/'embeddings.npz');evaluate(a['images'],a['texts'],a['labels'],out,threshold=.05,quantization=False)

if __name__=='__main__':
    for model in ('ViT-B-16','ViT-L-14','ViT-B-16-SigLIP'):run(model)

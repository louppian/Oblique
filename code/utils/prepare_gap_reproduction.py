from pathlib import Path
import subprocess,tarfile,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
ITEMS=[('https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt','ckpt/ViT-L-14.pt'),('https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt','ckpt/ViT-B-16.pt'),('https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet','data/cifar/cifar10_test.parquet'),('https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet','data/cifar/cifar100_test.parquet'),('https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz','data/aokvqa/aokvqa_v1p0.tar.gz')]
ITEMS += [(f'https://huggingface.co/timm/ViT-B-16-SigLIP/resolve/41f575766f40e752fdd1383e9565b7f02388c1c4/{name}',f'ckpt/ViT-B-16-SigLIP/{name}') for name in ('open_clip_pytorch_model.bin','open_clip_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')]
def fetch(item):
 url,relative=item;p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  print('Downloading',relative,flush=True)
  subprocess.run(['curl.exe','-L','--fail','--retry','3','--connect-timeout','20','--speed-limit','1000','--speed-time','60','-o',str(p.with_suffix(p.suffix+'.partial')),url],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  p.with_suffix(p.suffix+'.partial').replace(p)
 if p.suffix=='.pt':assert hashlib.sha256(p.read_bytes()).hexdigest()==url.split('/')[-2]
 if p.name.endswith('.tar.gz'):
  with tarfile.open(p) as tar:
   for member in tar.getmembers():
    target=(p.parent/member.name).resolve();assert target.is_relative_to(p.parent.resolve()) and not member.issym() and not member.islnk()
   tar.extractall(p.parent)
 print('Ready',relative,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(3) as pool:list(pool.map(fetch,ITEMS))
 print('All requested public assets ready',flush=True)

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'utils'))
from modality_gap_imagenet import run
if __name__=='__main__':run('ViT-B-16-SigLIP')

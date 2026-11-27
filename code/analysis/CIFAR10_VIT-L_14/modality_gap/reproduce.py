from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'utils'))
from modality_gap_reproduction import cifar

if __name__ == '__main__':
    cifar('CIFAR10', 'ViT-L-14')

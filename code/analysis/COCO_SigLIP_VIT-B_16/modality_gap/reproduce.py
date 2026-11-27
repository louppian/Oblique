from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'COCO_VIT-L_14_224/modality_gap'))
from reproduce import main
if __name__=='__main__':
    main(siglip=True)

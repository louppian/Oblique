"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from cached_analysis import run_magnitude
CONFIG = dict(pairs=(('clip','coco'),('clip','nih'),('carzero','coco'),('carzero','nih')),seeds=(0,1,2),n_reference=2000)

def main():
    return run_magnitude('02_component_magnitude',**CONFIG)


if __name__ == '__main__':
    main()

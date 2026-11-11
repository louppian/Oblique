"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from cached_analysis import run_medical
CONFIG = dict(models=('clip','carzero'),seeds=(0,1,2),n_reference=2000)

def main():
    return run_medical('04_medical_negation',**CONFIG)


if __name__ == '__main__':
    main()

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from supplementary_analysis import run

if __name__ == '__main__':
    run('domain','01_domain_shift')


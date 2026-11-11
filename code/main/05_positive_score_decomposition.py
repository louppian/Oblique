"""Positive-score analysis using alpha*rhat + beta*bhat + e + delta."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'/'utils'))
from positive_analysis import run
if __name__=='__main__':run()

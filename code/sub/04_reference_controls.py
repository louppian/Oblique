"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import residual_core
from cached_analysis import run_reference_controls
CONFIG = residual_core.CONFIG
MODELS = ('clip','carzero')

def main():
    return run_reference_controls('04_reference_controls',MODELS,CONFIG)


if __name__ == '__main__':
    main()

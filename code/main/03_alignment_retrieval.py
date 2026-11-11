"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import residual_core
from experiment_runner import run_retrieval
CONFIG = residual_core.CONFIG

def main():
    return run_retrieval('main','03_alignment_retrieval',CONFIG)


if __name__ == '__main__':
    main()

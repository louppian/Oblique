"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
from dataclasses import replace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import residual_core
from experiment_runner import run_retrieval
CONFIG = replace(residual_core.CONFIG, batch_size=512, encode_batch_size=512)

def main():
    return run_retrieval('main','03_alignment_retrieval',CONFIG)


if __name__ == '__main__':
    main()

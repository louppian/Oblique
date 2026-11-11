"""Manuscript 5.4. Run directly; one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import importlib
from experiment_runner import run_existing
source = importlib.import_module('3_alignment')
CONFIG = source.CONFIG


def main():
    return run_existing('sub', '04_mean_shift', '3_alignment', ('exp10_axis_alignment', 'exp11_mean_shift'), CONFIG)


if __name__ == '__main__':
    main()

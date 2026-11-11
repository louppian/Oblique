"""Manuscript 5.3. Run directly; one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import importlib
from experiment_runner import run_existing
source = importlib.import_module('1_analysis')
CONFIG = source.CONFIG


def main():
    return run_existing('sub', '03_domain_shift', '1_analysis', ('exp06_domain_shift',), CONFIG)


if __name__ == '__main__':
    main()

"""Manuscript 5.5. Run directly; one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import importlib
from experiment_runner import run_existing
source = importlib.import_module('2_application')
CONFIG = source.APP


def main():
    return run_existing('sub', '05_attack_detection', '2_application', ('exp09_attack_detection',), CONFIG)


if __name__ == '__main__':
    main()

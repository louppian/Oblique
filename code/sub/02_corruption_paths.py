"""Manuscript 5.2. Run directly; one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import importlib
from experiment_runner import run_existing
source = importlib.import_module('1_analysis')
CONFIG = source.CONFIG


def main():
    return run_existing('sub', '02_corruption_paths', '1_analysis', ('exp04_noise_trajectories', 'exp05_random_reference_controls'), CONFIG)


if __name__ == '__main__':
    main()

"""Run directly. Edit CONFIG; outputs are one CSV and one PNG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
import train_negation_alignment
from experiment_runner import run_medical_training
CONFIG = train_negation_alignment.TrainConfig()

def main():
    return run_medical_training('sub','06_medical_alignment_training',CONFIG)


if __name__ == '__main__':
    main()

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import build_arg_parser, run
from backbone import load_model, classify, classify_feature

DATA_PATH = "./../data/Chestpert/chexlocalize/CheXpert"
LABEL_DIR = "Chexpert"

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'No Finding']

POSITIVE_PROMPTS = {
    "0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Consolidation"],
    "3": ["There is Edema"], "4": ["There is Pleural Effusion"], "5": ["There is no Finding"],
}
NEGATIVE_PROMPTS = {
    "0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Consolidation"],
    "3": ["There is no Edema"], "4": ["There is no Pleural Effusion"],
}
PROMPTS = [v[0] for v in POSITIVE_PROMPTS.values()] + [v[0] for v in NEGATIVE_PROMPTS.values()]


def load_test_data(data_path):
    """Build the CheXpert5 test split with resolved image paths and multi-hot labels."""
    image_csv = pd.read_csv(os.path.join(LABEL_DIR, 'chexpert5_test_image.csv'))
    image_csv['Path'] = image_csv['Path'].apply(lambda x: os.path.join(data_path, '/'.join(x.split('/')[3:])))

    label_csv = pd.read_csv(os.path.join(LABEL_DIR, 'test_labels.csv'))
    true_labels = label_csv[CLASS_NAMES].values

    return image_csv, true_labels


def main(args):
    run(
        args, CLASS_NAMES, PROMPTS, load_model, load_test_data, classify, prefix="CheXpert",
        positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS, classify_fn_feature=classify_feature,
    )


if __name__ == "__main__":
    args = build_arg_parser(default_data_path=DATA_PATH).parse_args()
    main(args)

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import (
    NIH14_CLASS_NAMES as CLASS_NAMES,
    NIH14_POSITIVE_PROMPTS as POSITIVE_PROMPTS,
    NIH14_NEGATIVE_PROMPTS as NEGATIVE_PROMPTS,
    NIH14_PROMPTS as PROMPTS,
    build_arg_parser,
    run,
)
from backbone import load_model, classify, classify_feature

DATA_PATH = "./../data/OPEN-I"

# OPEN-I's free-text "labels_automatic" column is matched against the NIH14 pathology names
# (plus a few synonyms) to build the same 15-class ground truth used for NIH14.
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia", "No_Finding",
]
PATHOLOGY_SYNONYMS = {
    "Pleural_Thickening": ["pleural thickening"],
    "Infiltration": ["Infiltrate"],
    "Atelectasis": ["Atelectases"],
    "No_Finding": ["-1"],
}


def load_test_data(data_path):
    """Build the OPEN-I test split with resolved image paths and multi-hot labels."""
    csv = pd.read_csv(os.path.join(data_path, 'custom.csv')).iloc[2:, :].reset_index(drop=True)
    csv = csv.replace(np.nan, "-1")

    true_labels = []
    for pathology in PATHOLOGIES:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        for synonym in PATHOLOGY_SYNONYMS.get(pathology, []):
            mask |= csv["labels_automatic"].str.contains(synonym.lower())
        true_labels.append(mask.values)

    true_labels = np.asarray(true_labels).T.astype(np.float32)
    true_labels[:, 14] = (true_labels[:, :14].sum(axis=1) == 0).astype(np.float32)

    image_csv = pd.read_csv(os.path.join(data_path, 'openi_multi_label_image.csv')).iloc[2:, :].reset_index(drop=True)
    image_csv['Path'] = image_csv['Path'].apply(lambda x: os.path.join(data_path, x.split('/')[-1]))

    return image_csv, true_labels


def main(args):
    run(
        args, CLASS_NAMES, PROMPTS, load_model, load_test_data, classify, prefix="OPEN_I",
        positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS, classify_fn_feature=classify_feature,
    )


if __name__ == "__main__":
    args = build_arg_parser(default_data_path=DATA_PATH).parse_args()
    main(args)
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import build_arg_parser, run
from backbone import load_model, classify, classify_feature

DATA_PATH = "./../data/PadChest/images"
LABEL_DIR = "PadChest"

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pneumonia', 'No Finding']
TEST_QUERY = ['atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema', 'pneumonia', 'normal']

POSITIVE_PROMPTS = {
    "0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Consolidation"],
    "3": ["There is Edema"], "4": ["There is Pneumonia"], "5": ["There is no Finding"],
}
NEGATIVE_PROMPTS = {
    "0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Consolidation"],
    "3": ["There is no Edema"], "4": ["There is no Pneumonia"],
}
PROMPTS = [v[0] for v in POSITIVE_PROMPTS.values()] + [v[0] for v in NEGATIVE_PROMPTS.values()]


def load_test_data(data_path):
    """Build the PadChest test split with resolved image paths and multi-hot labels."""
    image_csv = pd.read_csv(os.path.join(LABEL_DIR, 'padchest_multi_label_image.csv'))
    image_csv['Path'] = image_csv['Path'].apply(lambda x: os.path.join(data_path, x.split('/')[-1]))

    with open(os.path.join(LABEL_DIR, "manual_image.json"), 'r') as f:
        label_json = json.load(f)

    labels_per_image = list(label_json.values())
    all_labels = [label for labels in labels_per_image for label in labels]
    sorted_labels = sorted(set(all_labels))
    normal_index = sorted_labels.index('normal')

    mlb = MultiLabelBinarizer(classes=sorted_labels)
    encoded_labels = mlb.fit_transform(labels_per_image)
    encoded_labels = np.delete(encoded_labels, normal_index, axis=1)

    query_indices = [sorted_labels.index(q) for q in TEST_QUERY]
    true_labels = encoded_labels[:, query_indices]

    return image_csv, true_labels


def main(args):
    run(
        args, CLASS_NAMES, PROMPTS, load_model, load_test_data, classify, prefix="PadChest",
        positive_prompts=POSITIVE_PROMPTS, negative_prompts=NEGATIVE_PROMPTS, classify_fn_feature=classify_feature,
    )


if __name__ == "__main__":
    args = build_arg_parser(default_data_path=DATA_PATH).parse_args()
    main(args)

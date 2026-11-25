import os
import sys
from glob import glob

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import (
    NIH14_CLASS_NAMES as CLASS_NAMES,
    NIH14_PROMPTS as PROMPTS,
    build_arg_parser,
    run,
)
from utils import process_images_generic
from backbone import load_model, classify

DATA_PATH = './../data/NIH'
LABEL_DIR = 'ChestXray-14'
DEFAULT_CFG_PATH = 'configs/chest14_finetuning_KAD.yaml'
DEFAULT_CKPT_PATH = 'checkpoints/KAD_BiMCQ_best_model.ckpt'


def load_test_data(data_path):
    """Build the ChestX-ray14 official test split with resolved image paths and multi-hot labels."""
    disease_columns = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Lung Mass', 'Lung Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural Thickening', 'Hernia',
    ]
    label_file = os.path.join(LABEL_DIR, 'test_list.txt')
    df_test = pd.read_csv(label_file, sep=' ', names=['path'] + disease_columns)
    df_test['No Finding'] = (df_test[disease_columns].sum(axis=1) == 0).astype(int)
    df_test['Image Index'] = df_test['path'].apply(os.path.basename)

    image_paths = {os.path.basename(p): p for p in glob(os.path.join(data_path, 'images*', '*', '*.png'))}
    df_test['Path'] = df_test['Image Index'].map(image_paths)

    rename_map = {'Lung Mass': 'Mass', 'Lung Nodule': 'Nodule'}
    ordered_columns = ['Image Index', 'Path'] + [rename_map.get(c, c) for c in disease_columns] + ['No Finding']
    test_df = df_test.rename(columns=rename_map)[ordered_columns]

    true_labels = test_df.iloc[:, 2:].values
    return test_df, true_labels


def main(args):
    run(args, CLASS_NAMES, PROMPTS, load_model, load_test_data, classify, prefix="NIH", process_img_fn=process_images_generic)


if __name__ == "__main__":
    args = build_arg_parser(
        default_data_path=DATA_PATH, default_cfg_path=DEFAULT_CFG_PATH, default_ckpt_path=DEFAULT_CKPT_PATH,
        support_directional=False,
    ).parse_args()
    main(args)

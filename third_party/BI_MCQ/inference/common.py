"""Shared evaluation/plotting logic for the per-dataset Inference_*.py scripts, usable
across backbones (CARZero, MedKLIP, KAD). Each backbone provides its own model-loading
and classification callables via a small `backbone.py` module (see inference/CARZero,
inference/MedKLIP, inference/KAD)."""
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import obtain_logit, obtain_simr, calculate_pnc_logit, calculate_metric

pd.options.display.float_format = '{:.3f}'.format
plt.style.use('default')

# The 14 NIH ChestX-ray14 pathologies + "No Finding", shared by the NIH14 and OPEN-I scripts
# since OPEN-I's ground truth is mapped onto the same label set.
NIH14_CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening',
    'Hernia', 'No Finding',
]
NIH14_POSITIVE_PROMPTS = {
    "0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"],
    "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"],
    "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"],
    "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"],
    "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14": ["There is no Finding"],
}
NIH14_NEGATIVE_PROMPTS = {
    "0": ["There is no Atelectasis"], "1": ["There is no Cardiomegaly"], "2": ["There is no Pleural Effusion"],
    "3": ["There is no Pulmonary Infiltration"], "4": ["There is no Pulmonary Mass"], "5": ["There is no Lung Nodule"],
    "6": ["There is no Pneumonia"], "7": ["There is no Pneumothorax"], "8": ["There is no Pulmonary Consolidation"],
    "9": ["There is no Pulmonary Edema"], "10": ["There is no Pulmonary Emphysema"], "11": ["There is no Fibrosis"],
    "12": ["There is no Pleural Thickening"], "13": ["There is no Hernia"],
}
# obtain_logit takes a flat prompt list: all positive prompts (incl. "No Finding") followed by all negative prompts.
NIH14_PROMPTS = [v[0] for v in NIH14_POSITIVE_PROMPTS.values()] + [v[0] for v in NIH14_NEGATIVE_PROMPTS.values()]


def build_arg_parser(default_data_path, default_cfg_path=None, default_ckpt_path=None, support_directional=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=default_cfg_path, required=default_cfg_path is None,
                         help="Path to the model's config.yaml")
    parser.add_argument("--ckpt_path", type=str, default=default_ckpt_path, help="Checkpoint path")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Root directory of the dataset images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to write results/plots to")
    if support_directional:
        parser.add_argument("--tsne", type=bool, default=False)
        parser.add_argument("--directional", type=bool, default=False, help="Also compute i2t/t2i directional metrics")
    return parser


def evaluate_prompts(test_df, true_labels, model, prompts, class_names, save_dir, device, batch_size, prefix, classify_fn, process_img_fn=None):
    logit = obtain_logit(test_df, prompts, model, device=device, batch_size=batch_size,
                          classify_fn=classify_fn, process_img_fn=process_img_fn)
    np.save(os.path.join(save_dir, f"{prefix}_logit.npy"), logit)

    p_logit = logit[:, :len(class_names)]  # positive logits (incl. "No Finding")
    n_logit = logit[:, len(class_names):]  # negative logits (excl. "No Finding")
    pnc_logit = calculate_pnc_logit(p_logit[:, :-1], n_logit)  # positive-negative combined (excl. "No Finding")

    pos_results = calculate_metric(p_logit, true_labels, class_names, nf=True)
    neg_results = calculate_metric(n_logit, true_labels, class_names, neg=True)
    pnc_results = calculate_metric(pnc_logit, true_labels, class_names)

    print("Positive Results:")
    print(pos_results)
    print("Negative Results:")
    print(neg_results)
    print("Positive-Negative Combined Results:")
    print(pnc_results)

    pos_results.to_csv(os.path.join(save_dir, f"{prefix}_pos_results.csv"))
    neg_results.to_csv(os.path.join(save_dir, f"{prefix}_neg_results.csv"))
    pnc_results.to_csv(os.path.join(save_dir, f"{prefix}_pnc_results.csv"))

    return pos_results, neg_results, pnc_results


def evaluate_directional(
    test_df, true_labels, model, positive_prompts, negative_prompts, class_names,
    save_dir, device, batch_size, prefix, classify_fn_feature, process_img_fn=None,
):
    """Compute i2t / t2i attention-direction metrics, needed for the optional t-SNE plots."""
    common_kwargs = dict(classify_fn_feature=classify_fn_feature, process_img_fn=process_img_fn, batch_size=batch_size)

    p_logit_i2t, p_i2t_feat, _ = obtain_simr(test_df, positive_prompts, model, device=device, mode='i2t', **common_kwargs)
    n_logit_i2t, n_i2t_feat, _ = obtain_simr(test_df, negative_prompts, model, device=device, mode='i2t', **common_kwargs)
    p_logit_t2i, _, p_t2i_feat = obtain_simr(test_df, positive_prompts, model, device=device, mode='t2i', **common_kwargs)
    n_logit_t2i, _, n_t2i_feat = obtain_simr(test_df, negative_prompts, model, device=device, mode='t2i', **common_kwargs)

    pnc_i2t_logit = calculate_pnc_logit(p_logit_i2t[:, :-1], n_logit_i2t)
    pnc_t2i_logit = calculate_pnc_logit(p_logit_t2i[:, :-1], n_logit_t2i)

    results = {
        'i2t': {
            'pos': calculate_metric(p_logit_i2t, true_labels, class_names, nf=True),
            'neg': calculate_metric(n_logit_i2t, true_labels, class_names, neg=True),
            'pnc': calculate_metric(pnc_i2t_logit, true_labels, class_names),
        },
        't2i': {
            'pos': calculate_metric(p_logit_t2i, true_labels, class_names, nf=True),
            'neg': calculate_metric(n_logit_t2i, true_labels, class_names, neg=True),
            'pnc': calculate_metric(pnc_t2i_logit, true_labels, class_names),
        },
    }

    for direction, res in results.items():
        print(f"Positive {direction.upper()} Results:")
        print(res['pos'])
        print(f"Negative {direction.upper()} Results:")
        print(res['neg'])
        print(f"Positive-Negative Combined {direction.upper()} Results:")
        print(res['pnc'])

        res['pos'].to_csv(os.path.join(save_dir, f"{prefix}_pos_{direction}_results.csv"))
        res['neg'].to_csv(os.path.join(save_dir, f"{prefix}_neg_{direction}_results.csv"))
        res['pnc'].to_csv(os.path.join(save_dir, f"{prefix}_pnc_{direction}_results.csv"))

    features = {
        'i2t': (p_i2t_feat, n_i2t_feat),
        't2i': (p_t2i_feat, n_t2i_feat),
    }
    return results, features


def plot_tsne(true_labels, class_names, directional_results, features, save_dir, prefix):
    ncols = 4
    nrows = -(-len(class_names) // ncols)  # ceil division

    for direction in ('i2t', 't2i'):
        p_feat, n_feat = features[direction]
        pos_results = directional_results[direction]['pos']
        neg_results = directional_results[direction]['neg']
        num_classes = p_feat.shape[1]

        pn_features_tsne = np.array([
            TSNE(n_components=2, random_state=42).fit_transform(
                np.concatenate((p_feat[:, i, :], n_feat[:, i, :]), axis=0)
            )
            for i in tqdm(range(num_classes), desc=f"t-SNE ({direction})")
        ])
        p_feat_tsne = pn_features_tsne[:, :p_feat.shape[0], :]
        n_feat_tsne = pn_features_tsne[:, p_feat.shape[0]:, :]

        plt.figure(figsize=(20, 20))
        for i in range(num_classes):
            true_sample = true_labels[:, i] == 1
            false_sample = true_labels[:, i] == 0
            plt.subplot(nrows, ncols, i + 1)
            plt.scatter(p_feat_tsne[i][false_sample, 0], p_feat_tsne[i][false_sample, 1], s=0.5, c='red', label='Positive - False', alpha=0.5)
            plt.scatter(n_feat_tsne[i][false_sample, 0], n_feat_tsne[i][false_sample, 1], s=0.5, c='green', label='Negative - False', alpha=0.5)
            plt.scatter(p_feat_tsne[i][true_sample, 0], p_feat_tsne[i][true_sample, 1], s=1, c='blue', label='Positive - True', alpha=0.5)
            plt.scatter(n_feat_tsne[i][true_sample, 0], n_feat_tsne[i][true_sample, 1], s=1, c='orange', label='Negative - True', alpha=0.5)
            plt.xlabel('t-SNE Component 1')
            plt.title(
                f"Class - {class_names[i]}, PosAUC - {pos_results.loc[class_names[i], 'auc']:.3f}, "
                f"NegAUC - {neg_results.loc[class_names[i], 'auc']:.3f}"
            )
            plt.legend()
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{direction}_tsne_plot.png"))
        plt.close()


def run(
    args, class_names, prompts, load_model_fn, load_test_data, classify_fn, prefix,
    process_img_fn=None, positive_prompts=None, negative_prompts=None, classify_fn_feature=None,
):
    model, cfg = load_model_fn(args)
    print(args.cfg_path)

    test_df, true_labels = load_test_data(args.data_path)
    # test_df = test_df.iloc[:1000].reset_index(drop=True)  # For testing purposes, limit to first 1000 samples
    # true_labels = true_labels[:1000]

    # save_dir/<model>/<dataset>/<run timestamp>/, so results from different models,
    # datasets, and runs never collide or overwrite each other. The model name is taken
    # from the directory that defines load_model_fn (inference/<Model>/backbone.py),
    # so callers don't need to pass it explicitly.
    model_name = os.path.basename(os.path.dirname(os.path.abspath(load_model_fn.__code__.co_filename)))
    # Callers like inference.sh export RUN_TIMESTAMP so the log file they tee alongside
    # this run lands in the exact same directory as the results below, rather than a
    # separately-timestamped sibling.
    timestamp = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, model_name, prefix, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    evaluate_prompts(test_df, true_labels, model, prompts, class_names, save_dir, args.device, args.batch_size,
                      prefix, classify_fn, process_img_fn)

    if getattr(args, "directional", False):
        if classify_fn_feature is None:
            print("--directional is not supported by this backbone; skipping.")
            return
        directional_results, features = evaluate_directional(
            test_df, true_labels, model, positive_prompts, negative_prompts, class_names,
            save_dir, args.device, args.batch_size, prefix, classify_fn_feature, process_img_fn,
        )
        if getattr(args, "tsne", False):
            plot_tsne(true_labels, class_names, directional_results, features, save_dir, prefix)
    elif getattr(args, "tsne", False):
        print("--tsne requires --directional (i2t/t2i features are only computed there); skipping t-SNE plots.")

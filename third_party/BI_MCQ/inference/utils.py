"""Backbone-agnostic batching, classification-loop, and metric helpers shared by the
CARZero/MedKLIP/KAD inference pipelines. Each backbone supplies its own `classify_fn`
(and, for backbones without a `process_img` method on the model, `process_img_fn`)."""
from tqdm import tqdm
import numpy as np
from typing import Literal
from sklearn.metrics import f1_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score, precision_recall_curve
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result


def process_images_generic(paths, device):
    """Generic PIL-based image loader (Resize 224 + ToTensor + ImageNet normalize),
    for backbones whose model object doesn't provide its own process_img method
    (e.g. MedKLIP, KAD)."""
    imgs = [IMAGENET_TRANSFORM(Image.open(p).convert('RGB')) for p in paths]
    return torch.stack(imgs, dim=0).to(device)


def obtain_logit(df, texts, model, device, classify_fn, process_img_fn=None, batch_size=256):
    image_list = split_list(df['Path'].tolist(), batch_size)
    processed_txt = model.process_text(texts, device)
    process_img = process_img_fn or model.process_img

    sim = []
    for img in tqdm(image_list, desc="Processing images"):
        processed_imgs = process_img(img, device)
        cls = classify_fn(model, processed_imgs, processed_txt)
        sim.append(cls)

    sim = np.concatenate(sim, axis=0)
    return sim


def obtain_simr(df, texts, model, device, mode, classify_fn_feature, process_img_fn=None, batch_size=256):
    """Like obtain_logit, but also returns the i2t/t2i features needed for t-SNE plots.
    Only supported by backbones that provide a classify_fn_feature."""
    image_list = split_list(df['Path'].tolist(), batch_size)
    prompt_list = [v[0] for v in texts.values()]
    processed_txt = model.process_text(prompt_list, device)
    process_img = process_img_fn or model.process_img

    sim = []
    i2t_feats = []
    t2i_feats = []
    for img in tqdm(image_list, desc="Processing images"):
        processed_imgs = process_img(img, device)
        similarities, i2t_feat, t2i_feat = classify_fn_feature(model, processed_imgs, processed_txt, mode)
        sim.append(similarities)
        i2t_feats.append(i2t_feat)
        t2i_feats.append(t2i_feat)

    sim = np.concatenate(sim, axis=0)
    i2t_feats = np.concatenate(i2t_feats, axis=0)
    t2i_feats = np.concatenate(t2i_feats, axis=0)

    return sim, i2t_feats, t2i_feats


def calculate_pnc_logit(
    pos_logits: np.ndarray,
    neg_logits: np.ndarray,
    reduction: Literal["none", "mean"] = "none",
) -> np.ndarray:
    if pos_logits.shape != neg_logits.shape:
        raise ValueError(
            f"Shape mismatch: pos_logits {pos_logits.shape}, "
            f"neg_logits {neg_logits.shape}"
        )

    # (..., 2)축으로 결합: 마지막 축 [-1] = positive, [0] = negative
    logits = np.stack([neg_logits, pos_logits], axis=-1)

    # --- Softmax 계산 ---
    # 안정성을 위해 log-sum-exp 사용
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # positive class 확률 = 마지막 축의 index 1
    prob_pos = probs[..., 1]

    if reduction == "mean":
        return prob_pos.mean()
    elif reduction == "none":
        return prob_pos
    else:
        raise ValueError("reduction must be 'none' or 'mean'")


def calculate_metric(logit, true_labels, class_name, nf=False, neg=False) :
    dic = {"accuracy": [], "f1_score": [], "recall": [], "precision": [], "auc": [], "mcc": []}
    sigmoid_output = torch.sigmoid(torch.tensor(logit)).numpy()
    for i, class_label in enumerate(tqdm(class_name)):
        if not nf :
            if i == len(class_name) - 1:  # 'No Finding' 클래스는 제외
                continue
        # 실제값과 예측값 비교
        true_label = true_labels[:,i]
        if neg :
            true_label = 1 - true_label

        precision, recall, thresholds = precision_recall_curve(true_label, sigmoid_output[:, i])
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        predicted_labels = (sigmoid_output[:, i] > max_f1_thresh).astype(int)  # 예측값 (threshold 사용)

        # 정확도 계산
        accuracy = np.mean(true_label == predicted_labels)
        # F1-score, Recall, Precision, and AUC 계산
        f1 = f1_score(true_label, predicted_labels)
        recall = recall_score(true_label, predicted_labels)
        precision = precision_score(true_label, predicted_labels)
        # roc_auc_score is undefined when true_label has only one class (e.g. a rare
        # pathology with no positive samples in a small test set) - skip it instead of crashing.
        auc = roc_auc_score(true_label, sigmoid_output[:, i]) if len(np.unique(true_label)) > 1 else np.nan
        mcc = matthews_corrcoef(true_label, predicted_labels)

        dic["accuracy"].append(accuracy)
        dic["f1_score"].append(f1)
        dic["recall"].append(recall)
        dic["precision"].append(precision)
        dic["auc"].append(auc)
        dic["mcc"].append(mcc)

    results = pd.DataFrame(dic, index=class_name[:-1]) if not nf else pd.DataFrame(dic, index=class_name)
    # 마지막 행을 제외한 나머지 행들의 평균을 'Mean' 행으로 추가
    if nf:
        mean_values = results.iloc[:-1].mean(numeric_only=True)
    else:
        mean_values = results.mean(numeric_only=True)
    results.loc['Mean'] = mean_values

    return results

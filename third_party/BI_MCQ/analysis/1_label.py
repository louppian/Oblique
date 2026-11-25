from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ── 수동 설정 ──────────────────────────────────────────
BASE = Path("/shared/home/mai/Taehun/Uncertainty")
PROJECT = BASE / "BI_MCQ"

CONFIG = PROJECT / "configs/chest14_finetuning_Bi-MCQ.yaml"
WEIGHT_ROOT = BASE / "MICCAI_2025/CARZero"
PRETRAIN = WEIGHT_ROOT / "pretrain_model/CARZero_best_model.ckpt"
M3AE = WEIGHT_ROOT / "pretrain_model/VITB-16-M3AE_last.ckpt"
CHECKPOINT = WEIGHT_ROOT / "checkpoints/BiMCQ_CARZero_best_model.ckpt"

IMAGE_ROOT = BASE / "data/NIH"
LABEL_CSV = IMAGE_ROOT / "Data_Entry_2017.csv"
TEST_CSV = IMAGE_ROOT / "NIH_test_data.csv"

OUTPUT = PROJECT / "analysis/label_frequency_analysis.png"
RESULT_CSV = PROJECT / "analysis/label_predictions.csv"

DEVICE = "cuda:0"
BATCH_SIZE = 512  # CUDA OOM이면 자동으로 절반으로 줄임
# ────────────────────────────────────────────────────────

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]


def load_data():
    source = pd.read_csv(
        LABEL_CSV, usecols=["Image Index", "Finding Labels"]
    )
    test = pd.read_csv(TEST_CSV, usecols=["Image Index"])

    for name, df in [("원본", source), ("테스트", test)]:
        if df.empty or df.isna().any().any():
            raise ValueError(f"{name} CSV가 비어 있거나 결측값이 있습니다.")
        df["Image Index"] = df["Image Index"].astype(str).str.strip()
        if df["Image Index"].duplicated().any():
            raise ValueError(f"{name} CSV에 중복 이미지가 있습니다.")

    source = source.set_index("Image Index")
    names = test["Image Index"].tolist()

    missing = sorted(set(names) - set(source.index))
    if missing:
        raise ValueError(f"원본 라벨 누락: {missing[:10]}")

    findings = (
        source.loc[names, "Finding Labels"]
        .str.replace("_", " ", regex=False)
        .str.split("|")
        .map(lambda values: {v.strip() for v in values})
    )

    unknown = set().union(*findings) - set(CLASSES) - {"No Finding"}
    if unknown:
        raise ValueError(f"알 수 없는 라벨: {sorted(unknown)}")

    labels = np.column_stack([
        findings.map(lambda values: int(c in values)).to_numpy()
        for c in CLASSES
    ])

    image_map = {}
    for path in IMAGE_ROOT.glob("images*/*/*.png"):
        if path.name in image_map:
            raise ValueError(f"중복 이미지 파일명: {path.name}")
        image_map[path.name] = path

    missing = [name for name in names if name not in image_map]
    if missing:
        raise FileNotFoundError(f"이미지 누락: {missing[:10]}")

    paths = [str(image_map[name]) for name in names]
    print(f"[DATA] 테스트 이미지 {len(paths):,}개 / 질환 14개")
    return paths, labels


def load_model():
    # 실제 Bi-MCQ 소스 위치 선택
    candidates = [PROJECT, PROJECT / "Bi-MCQ"]
    repo = next(
        (p for p in candidates
         if (p / "CARZero/CARZero.py").is_file()),
        None,
    )
    if repo is None:
        raise FileNotFoundError(
            "BI_MCQ 또는 BI_MCQ/Bi-MCQ 아래에 CARZero 소스가 없습니다."
        )

    sys.path.insert(0, str(repo))
    import CARZero

    cfg = OmegaConf.load(CONFIG)
    if not cfg.model.CARZero.multi:
        raise ValueError("Bi-MCQ 추론에는 model.CARZero.multi=true가 필요합니다.")

    # GitHub 내부의 ./pretrain_model/VITB-16-M3AE_last.ckpt를
    # 확인된 가중치 폴더에서 읽도록 모델 생성 중에만 작업 경로 변경
    previous_dir = Path.cwd()
    try:
        os.chdir(WEIGHT_ROOT)
        model = CARZero.load_CARZero(
            name=str(PRETRAIN),
            device="cpu",
            multi=True,
            cfg=cfg,
        )
    finally:
        os.chdir(previous_dir)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    if "state_dict" not in checkpoint:
        raise ValueError("체크포인트에 state_dict가 없습니다.")

    # GitHub inference/CARZero/backbone.py와 동일한 키 변환
    expected = model.state_dict()
    state = {}
    for key, value in checkpoint["state_dict"].items():
        key = key.split("CARZero_model.")[-1]
        if key in expected:
            if key in state:
                raise ValueError(f"체크포인트 키 중복: {key}")
            state[key] = value

    # 누락·shape 불일치가 있으면 즉시 중단
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print("[MODEL] Bi-MCQ 체크포인트 strict 로딩 완료")

    return model, CARZero


def cache_id():
    h = hashlib.sha256(b"bimcq-frequency-v1-mode-both-fp32")
    for path in [CONFIG, PRETRAIN, M3AE, CHECKPOINT]:
        stat = path.stat()
        h.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    h.update(CONFIG.read_bytes())
    # 로컬 소스 변경 시 이전 결과를 섞지 않음
    repo = next(p for p in [PROJECT, PROJECT / "Bi-MCQ"]
                if (p / "CARZero/CARZero.py").is_file())
    for path in sorted((repo / "CARZero").rglob("*.py")):
        h.update(str(path.relative_to(repo)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def infer(paths, labels):
    names = [Path(p).name for p in paths]
    ycols = [f"y_{c}" for c in CLASSES]
    pcols = [f"p_{c}" for c in CLASSES]
    ncols = [f"n_{c}" for c in CLASSES]
    columns = ["run_id", "Image Index"] + ycols + pcols + ncols
    signature = cache_id()
    completed = pd.DataFrame(columns=columns)
    if RESULT_CSV.exists():
        completed = pd.read_csv(RESULT_CSV)
        if list(completed.columns) != columns:
            raise ValueError("기존 결과 CSV 열이 다릅니다. RESULT_CSV를 다른 이름으로 지정하세요.")
        if len(completed) and not completed["run_id"].eq(signature).all():
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = RESULT_CSV.with_name(f"{RESULT_CSV.stem}_previous_{stamp}.csv")
            os.replace(RESULT_CSV, backup)
            print(f"[CACHE] 가중치/설정 변경: 이전 CSV 보존 → {backup}")
            completed = pd.DataFrame(columns=columns)
        count = len(completed)
        if (count > len(names) or
            completed["Image Index"].tolist() != names[:count] or
            not completed["run_id"].eq(signature).all() or
            not np.array_equal(completed[ycols].to_numpy(), labels[:count]) or
            not np.isfinite(completed[pcols + ncols].to_numpy(dtype=float)).all()):
            raise ValueError("기존 CSV의 모델/이미지 순서/라벨/점수가 현재 설정과 다릅니다.")
        print(f"[CACHE] {count:,}/{len(names):,}개 재사용")
    start = len(completed)
    if start == len(paths):
        return completed[pcols].to_numpy(float), completed[ncols].to_numpy(float)

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    model, carzero = load_model()
    terms = [
        "Atelectasis", "Cardiomegaly", "Pleural Effusion",
        "Pulmonary Infiltration", "Pulmonary Mass", "Lung Nodule",
        "Pneumonia", "Pneumothorax", "Pulmonary Consolidation",
        "Pulmonary Edema", "Pulmonary Emphysema", "Fibrosis",
        "Pleural Thickening", "Hernia",
    ]
    prompts = ([f"There is {t}" for t in terms] + ["There is no Finding"]
               + [f"There is no {t}" for t in terms])
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    batch_size = BATCH_SIZE
    if batch_size < 1:
        raise ValueError("BATCH_SIZE는 1 이상이어야 합니다.")
    if DEVICE.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(torch.device(DEVICE))
        print(f"[GPU] free={free / 2**30:.1f} GiB / total={total / 2**30:.1f} GiB")
    print(f"[INFERENCE] batch={batch_size}, mode=both (기존과 동일)")

    def forward_batch(batch_paths, text):
        images = model.process_img(batch_paths, DEVICE)
        return np.asarray(carzero.bimcq_classification(model, images, text, mode="both"))

    with torch.inference_mode(), tqdm(total=len(paths), initial=start, desc="Images") as bar:
        text = model.process_text(prompts, DEVICE)
        while start < len(paths):
            end = min(start + batch_size, len(paths))
            oom = False
            try:
                logits = forward_batch(paths[start:end], text)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                oom = True
            if oom:
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(f"[OOM] batch={batch_size}로 재시도")
                continue
            if logits.shape != (end - start, 29) or not np.isfinite(logits).all():
                raise ValueError(f"잘못된 logits: {logits.shape}")
            block = pd.DataFrame(
                np.column_stack([labels[start:end], logits[:, :14], logits[:, 15:29]]),
                columns=ycols + pcols + ncols,
            )
            block.insert(0, "Image Index", names[start:end])
            block.insert(0, "run_id", signature)
            completed = pd.concat([completed, block], ignore_index=True)
            # 배치마다 원자적으로 저장. 중단 후 같은 CSV에서 이어서 추론.
            temporary = RESULT_CSV.with_suffix(".csv.tmp")
            completed.to_csv(temporary, index=False)
            os.replace(temporary, RESULT_CSV)
            bar.update(end - start)
            start = end
    return completed[pcols].to_numpy(float), completed[ncols].to_numpy(float)


def frequency_table(labels, p_logit, n_logit):
    rows = []
    for i, disease in enumerate(CLASSES):
        y = labels[:, i]
        present, absent = int((y == 1).sum()), int((y == 0).sum())
        valid = present > 0 and absent > 0
        rows.append({
            "disease": disease,
            "POS_true_n": present, "POS_true_pct": 100 * present / len(y),
            "POS_false_n": absent, "POS_false_pct": 100 * absent / len(y),
            "POS_AUC": roc_auc_score(y, p_logit[:, i]) if valid else np.nan,
            "NEG_true_n": absent, "NEG_true_pct": 100 * absent / len(y),
            "NEG_false_n": present, "NEG_false_pct": 100 * present / len(y),
            "NEG_AUC": roc_auc_score(1 - y, n_logit[:, i]) if valid else np.nan,
        })
    return pd.DataFrame(rows)


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, len(x)
    r, p = pearsonr(x, y)
    return float(r), float(p), len(x)


def plot(labels, p_logit, n_logit):
    table = frequency_table(labels, p_logit, n_logit)
    order = np.argsort(-table.POS_true_pct.to_numpy())
    ids = {i: rank + 1 for rank, i in enumerate(order)}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(21, 15))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.09,
                        wspace=0.42, hspace=0.48)
    for col, prefix in enumerate(["POS", "NEG"]):
        color = "#2878B5" if prefix == "POS" else "#D7812E"
        ax = axes[0, col]
        true = table[f"{prefix}_true_pct"].to_numpy()[order]
        false = table[f"{prefix}_false_pct"].to_numpy()[order]
        auc = table[f"{prefix}_AUC"].to_numpy()[order]
        ax.barh(range(14), true, color=color, label="Prompt true")
        ax.barh(range(14), false, left=true, color="#DFE5EB", label="Prompt false")
        ax.set_yticks(range(14), [f"{ids[i]:02d} {CLASSES[i]}" for i in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 145)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Share of all test images")
        for j in range(14):
            ax.text(102, j, f"{true[j]:.2f} / {false[j]:.2f}", va="center", fontsize=9)
            ax.text(133, j, f"{auc[j]:.4f}", va="center", fontsize=9)
        ax.text(102, -0.85, "True% / False%", fontweight="bold", fontsize=9)
        ax.text(133, -0.85, "AUC", fontweight="bold", fontsize=9)
        meaning = "True = disease present" if prefix == "POS" else "True = disease absent"
        ax.set_title(f"{prefix}: label frequency and AUC | {meaning}", pad=30)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)

        ax = axes[1, col]
        x = table[f"{prefix}_true_pct"].to_numpy()
        y = table[f"{prefix}_AUC"].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        r, p, count = correlation(x, y)
        ax.scatter(x[valid], y[valid], s=80, color=color, edgecolors="white", zorder=3)
        for i in np.flatnonzero(valid):
            ax.annotate(f"{ids[i]:02d}", (x[i], y[i]), xytext=(5, 8 if i % 2 == 0 else -13),
                        textcoords="offset points", fontsize=10)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        if valid.any():
            low, high = x[valid].min(), x[valid].max()
            pad = max((high - low) * 0.13, 1)
            ax.set_xlim(max(0, low - pad), min(100, high + pad))
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{prefix} true-label frequency (%)")
        ax.set_ylabel(f"{prefix} AUROC")
        ax.set_title(f"{prefix}: frequency vs performance\nPearson r={r:.4f}, p={p:.6g}, n={count}", pad=12)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        print(f"[Pearson {prefix}] r={r:.6f}, p={p:.6g}, n={count}")

    fig.suptitle(f"NIH test set | N={len(labels):,} | 14 diseases | POS / NEG", fontsize=19, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, f"Macro POS AUC = {table.POS_AUC.mean():.4f}    |    Macro NEG AUC = {table.NEG_AUC.mean():.4f}", ha="center", fontsize=13)
    fig.text(0.5, 0.025, "POS: AUC(y, positive logit). NEG: AUC(1-y, negative logit). "
             "Each AUC uses both true and false groups.\nPoint IDs match the disease rows above. "
             "Correlations use full-precision AUCs across diseases.", ha="center", fontsize=11)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Macro POS AUC: {table.POS_AUC.mean():.4f}")
    print(f"Macro NEG AUC: {table.NEG_AUC.mean():.4f}")
    print(f"[SAVED] {OUTPUT}\n[CSV] {RESULT_CSV}")


def main():
    for path in [CONFIG, PRETRAIN, M3AE, CHECKPOINT, LABEL_CSV, TEST_CSV]:
        if not path.is_file():
            raise FileNotFoundError(path)

    paths, labels = load_data()
    p_logit, n_logit = infer(paths, labels)
    plot(labels, p_logit, n_logit)


if __name__ == "__main__":
    main()
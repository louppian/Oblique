#!/bin/bash
# Runs inference on all 4 datasets (NIH14, OPEN-I, CheXpert, PadChest) back to back,
# for a chosen backbone (CARZero, MedKLIP, or KAD).
set -e
set -o pipefail  # so a failing `python ... | tee ...` still aborts the script under set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_PWD="$(pwd)"

# Defaults — override via CLI args (model [cfg_path] [ckpt_path] [save_dir] [batch_size]) or these env vars.
DEFAULT_MODEL="${MODEL:-CARZero}"
DEFAULT_CFG_PATH="${CFG_PATH:-}"
DEFAULT_CKPT_PATH="${CKPT_PATH:-}"
DEFAULT_SAVE_DIR="${SAVE_DIR:-results}"
DEFAULT_BATCH_SIZE="${BATCH_SIZE:-64}"

model="${1:-$DEFAULT_MODEL}"
cfg_path="${2:-$DEFAULT_CFG_PATH}"
ckpt_path="${3:-$DEFAULT_CKPT_PATH}"
save_dir="${4:-$DEFAULT_SAVE_DIR}"
batch_size="${5:-$DEFAULT_BATCH_SIZE}"

case "$model" in
    CARZero|MedKLIP|KAD) ;;
    *)
        echo "Unknown model '$model'. Expected one of: CARZero, MedKLIP, KAD"
        exit 1
        ;;
esac

# Per-model defaults, used only when cfg_path/ckpt_path aren't given explicitly.
case "$model" in
    CARZero)
        [ -z "$cfg_path" ] && cfg_path="configs/chest14_finetuning_Bi-MCQ.yaml"
        [ -z "$ckpt_path" ] && ckpt_path="checkpoints/BiMCQ_CARZero_best_model.ckpt"
        ;;
    MedKLIP)
        [ -z "$cfg_path" ] && cfg_path="configs/chest14_finetuning_MedKLIP.yaml"
        [ -z "$ckpt_path" ] && ckpt_path="checkpoints/BiMCQ_MedKLIP_best_model.ckpt"
        ;;
    KAD)
        [ -z "$cfg_path" ] && cfg_path="configs/chest14_finetuning_KAD.yaml"
        [ -z "$ckpt_path" ] && ckpt_path="checkpoints/BiMCQ_KAD_best_model.ckpt"
        ;;
esac

# Resolve to absolute paths against the caller's cwd, since we cd into the repo root below
# and the Inference_*.py scripts resolve relative dataset paths (e.g. ChestXray-14/,
# pretrain_model/) against the repo root, not against wherever this script was invoked from.
to_abs_path() {
    local p="$1"
    if [ -z "$p" ]; then
        return
    fi
    case "$p" in
        /*) echo "$p" ;;
        *) echo "$ORIG_PWD/$p" ;;
    esac
}

cfg_path="$(to_abs_path "$cfg_path")"
ckpt_path="$(to_abs_path "$ckpt_path")"
save_dir="$(to_abs_path "$save_dir")"

COMMON_ARGS="--cfg_path $cfg_path --batch_size $batch_size"
if [ -n "$ckpt_path" ]; then
    COMMON_ARGS="$COMMON_ARGS --ckpt_path $ckpt_path"
fi
# --directional/--tsne only exist for the CARZero backbone (MedKLIP/KAD don't support them).
if [ "$model" = "CARZero" ]; then
    COMMON_ARGS="$COMMON_ARGS --directional=False"
fi

cd "$SCRIPT_DIR"

# Each script's run() nests save_dir as <save_dir>/<model>/<dataset>/<timestamp>/. We
# export RUN_TIMESTAMP so run() reuses this exact timestamp instead of generating its
# own, and tee the console output into that same directory - so the log file always
# lands right alongside the CSV/npy results it describes, not off in a sibling file.
export RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

run_dataset() {
    local script="$1"
    local dataset="$2"
    local log_dir="$save_dir/$model/$dataset/$RUN_TIMESTAMP"
    mkdir -p "$log_dir"
    echo "Inference $dataset Dataset ($model)"
    python "inference/$model/$script" $COMMON_ARGS --save_dir "$save_dir" 2>&1 | tee "$log_dir/inference.log"
}

run_dataset Inference_NIH14.py NIH
run_dataset Inference_OPENI.py OPEN_I
run_dataset Inference_CheXpert.py CheXpert
run_dataset Inference_PadChest.py PadChest

# To run a single dataset instead of all four, call its script directly from the repo root
# (dataset label paths like ChestXray-14/ are resolved relative to the repo root). Each
# script's run() automatically nests --save_dir as <save_dir>/<model>/<dataset>/<timestamp>/
# (e.g. results/CARZero/NIH/20260703-165423/), so results never collide across models,
# datasets, or repeated runs - just pass the same base --save_dir every time.
#
# ===== CARZero =====
#
# bash inference.sh CARZero configs/chest14_finetuning_Bi-MCQ.yaml checkpoints/BiMCQ_CARZero_best_model.ckpt results 64
#
# python inference/CARZero/Inference_NIH14.py \
#     --cfg_path configs/chest14_finetuning_Bi-MCQ.yaml \
#     --ckpt_path checkpoints/BiMCQ_CARZero_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64 \
#     --directional=True
#
# python inference/CARZero/Inference_OPENI.py \
#     --cfg_path configs/chest14_finetuning_Bi-MCQ.yaml \
#     --ckpt_path checkpoints/BiMCQ_CARZero_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64 \
#     --directional=True
#
# python inference/CARZero/Inference_CheXpert.py \
#     --cfg_path configs/chest14_finetuning_Bi-MCQ.yaml \
#     --ckpt_path checkpoints/BiMCQ_CARZero_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64 \
#     --directional=True
#
# python inference/CARZero/Inference_PadChest.py \
#     --cfg_path configs/chest14_finetuning_Bi-MCQ.yaml \
#     --ckpt_path checkpoints/BiMCQ_CARZero_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64 \
#     --directional=True
#
# Omit --ckpt_path for zero-shot; add --tsne=True (requires --directional=True) for t-SNE plots.
#
# ===== MedKLIP =====
#
# bash inference.sh MedKLIP configs/chest14_finetuning_MedKLIP.yaml checkpoints/BiMCQ_MedKLIP_best_model.ckpt results 64
#
# python inference/MedKLIP/Inference_NIH14.py \
#     --cfg_path configs/chest14_finetuning_MedKLIP.yaml \
#     --ckpt_path checkpoints/BiMCQ_MedKLIP_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/MedKLIP/Inference_OPENI.py \
#     --cfg_path configs/chest14_finetuning_MedKLIP.yaml \
#     --ckpt_path checkpoints/BiMCQ_MedKLIP_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/MedKLIP/Inference_CheXpert.py \
#     --cfg_path configs/chest14_finetuning_MedKLIP.yaml \
#     --ckpt_path checkpoints/BiMCQ_MedKLIP_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/MedKLIP/Inference_PadChest.py \
#     --cfg_path configs/chest14_finetuning_MedKLIP.yaml \
#     --ckpt_path checkpoints/BiMCQ_MedKLIP_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# ===== KAD =====
#
# bash inference.sh KAD configs/chest14_finetuning_KAD.yaml checkpoints/BiMCQ_KAD_best_model.ckpt results 64
#
# python inference/KAD/Inference_NIH14.py \
#     --cfg_path configs/chest14_finetuning_KAD.yaml \
#     --ckpt_path checkpoints/BiMCQ_KAD_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/KAD/Inference_OPENI.py \
#     --cfg_path configs/chest14_finetuning_KAD.yaml \
#     --ckpt_path checkpoints/BiMCQ_KAD_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/KAD/Inference_CheXpert.py \
#     --cfg_path configs/chest14_finetuning_KAD.yaml \
#     --ckpt_path checkpoints/BiMCQ_KAD_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# python inference/KAD/Inference_PadChest.py \
#     --cfg_path configs/chest14_finetuning_KAD.yaml \
#     --ckpt_path checkpoints/BiMCQ_KAD_best_model.ckpt \
#     --save_dir results \
#     --batch_size 64
#
# MedKLIP/KAD don't support --directional/--tsne (not available on these backbones).

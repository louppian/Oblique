#!/bin/bash
# Runs finetuning for a chosen backbone (CARZero, MedKLIP, or KAD).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model selection — override via CLI arg or this env var.
DEFAULT_MODEL="${MODEL:-CARZero}"
model="${1:-$DEFAULT_MODEL}"

case "$model" in
    CARZero|MedKLIP|KAD) ;;
    *)
        echo "Unknown model '$model'. Expected one of: CARZero, MedKLIP, KAD"
        exit 1
        ;;
esac

model_lower="$(echo "$model" | tr '[:upper:]' '[:lower:]')"
train_script="finetune/train_${model_lower}.py"

cd "$SCRIPT_DIR"

echo "Training $model"
python "$train_script"

# Examples:
#
# ./train.sh              # CARZero (default)
# ./train.sh MedKLIP
# ./train.sh KAD
#
# Each backbone's config is hardcoded inside its own finetune/train_<model>.py
# (configs/chest14_finetuning_{Bi-MCQ,MedKLIP,KAD}.yaml) rather than passed via CLI.

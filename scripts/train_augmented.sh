#!/bin/bash
#SBATCH --job-name=train_augmented_esrf
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/train_augmented_%J.out
#SBATCH --error=logs/train_augmented_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRAINING_PRECISION=32
export CUDA_VISIBLE_DEVICES=0
CUDNN_LIB="$HOME/miniconda3/envs/scroll/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
CFG="${PROJ}/configs/ft_esrf.py"
TIMESFORMER_CKPT="${HOME}/scroll_prize/villa/ink-detection/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"
AUGMENTED_DATA_DIR="${HOME}/scroll_prize/data/esrf/augmented"
CKPT_DIR="${PROJ}/checkpoints/ft_esrf_augmented_$(date +%Y%m%d_%H%M%S)"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== MiniUNETR Training on Augmented ESRF Data ==="
echo "Config: ${CFG}"
echo "Augmented data dir: ${AUGMENTED_DATA_DIR}"
echo "Transfer learning: Yes (using Epoch 15 baseline or transfer weights)"
echo "Epochs: 20"
echo "Checkpoint dir: ${CKPT_DIR}"
echo ""

# First, create augmented data if not present
if [ ! -d "${AUGMENTED_DATA_DIR}" ]; then
    echo "Creating augmented dataset..."
    python -u scripts/prepare_esrf_augmented.py \
        "$HOME/scroll_prize/data/esrf/scroll0" \
        --output-dir "${AUGMENTED_DATA_DIR}" \
        --num-augmentations 3
else
    echo "Using existing augmented dataset at ${AUGMENTED_DATA_DIR}"
fi

echo ""
echo "Starting training on augmented data..."
python -u scripts/train_full_transfer.py \
    "${CFG}" \
    --epochs 20 \
    --checkpoint-dir "${CKPT_DIR}" \
    --timesformer-ckpt "${TIMESFORMER_CKPT}" \
    --transfer-lr 5e-5 \
    --warmup-epochs 1

echo ""
echo "=== Training on augmented data complete ==="
echo "Final checkpoints: ${CKPT_DIR}"
echo ""
echo "Next: Run final inference on Scroll 3 with best augmented model"

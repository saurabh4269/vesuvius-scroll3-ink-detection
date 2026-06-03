#!/bin/bash
#SBATCH --job-name=b1_validate
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Validate B1 model on villa Scroll 1/2 labeled segment
# Edit SEGMENT_ID and SCROLL below to test different segments

# 20230827161847 is the only Scroll 1 segment in villa's scrolls.yaml — safest choice
SEGMENT_ID="20230827161847"
SCROLL=1

CKPT=~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt
CONFIG=~/scroll_prize/vesuvius_first_title_prize/configs/ft_esrf_b1.py
LABEL_DIR=~/scroll_prize/villa/ink-detection/all_labels/
OUT_DIR=~/scroll_prize/results/b1_validation/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

echo "[JOB] Validating B1 on Scroll ${SCROLL} segment ${SEGMENT_ID}"
echo "[JOB] Checkpoint: $(basename ${CKPT})"
echo "[JOB] Started: $(date)"

mkdir -p "${OUT_DIR}"
mkdir -p ~/logs
cd ~/scroll_prize/vesuvius_first_title_prize

python ~/scroll_prize/scripts/validate_b1_villa.py \
    --checkpoint "${CKPT}" \
    --config "${CONFIG}" \
    --segment-id "${SEGMENT_ID}" \
    --scroll "${SCROLL}" \
    --label-dir "${LABEL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --zarr-level 0 \
    --patch-size 128 \
    --stride 64

echo "[JOB] Done: $(date)"

#!/bin/bash
#SBATCH --job-name=villa_infer
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/villa_infer_%j.out
#SBATCH --error=logs/villa_infer_%j.err

# Run villa's own pre-trained checkpoint on a labeled Scroll 1 segment.
# Use the same segment as validate_b1_villa.sh so results are directly comparable.

SEGMENT_ID="20230827161847"
ZARR_PATH="https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/${SEGMENT_ID}.zarr/"
CKPT=~/scroll_prize/villa/ink-detection/wild14_deduped_64_pretrained2_20231210121321_0_fr_i3depoch=3-v2_256.ckpt
METADATA=~/scroll_prize/villa/ink-detection/metadata.json
OUT_PNG=~/scroll_prize/results/villa_validation/${SEGMENT_ID}_villa_pred.png
OUT_NPY=~/scroll_prize/results/villa_validation/${SEGMENT_ID}_villa_pred.npy

export PATH=$PATH:/opt/slurm/bin

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

echo "[JOB] villa pre-trained inference on segment ${SEGMENT_ID}"
echo "[JOB] Checkpoint: $(basename ${CKPT})"
echo "[JOB] Started: $(date)"

mkdir -p ~/scroll_prize/results/villa_validation/
mkdir -p ~/logs

cd ~/scroll_prize/villa/ink-detection/

python infer_resnet3d_vesuvius.py \
    --metadata_json "${METADATA}" \
    --ckpt_path "${CKPT}" \
    --segment_id "${SEGMENT_ID}" \
    --zarr_path "${ZARR_PATH}" \
    --output_path "${OUT_PNG}" \
    --output_npy "${OUT_NPY}" \
    --chunk_size 2048

echo "[JOB] Done: $(date)"
echo "[JOB] Output: ${OUT_PNG}"

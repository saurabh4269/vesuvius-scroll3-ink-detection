#!/bin/bash
#SBATCH --job-name=scroll_infer_s3
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/scroll_infer_s3_%J.out
#SBATCH --error=logs/scroll_infer_s3_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

SEG_ID="20240618142020"
PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
DATA_DIR="$HOME/scroll_prize/data"
CKPT="${PROJ}/checkpoints/scroll5/warm-planet-193-unetr-sf-b3-250417-171532/best-checkpoint-epoch=13-val_iou=0.71.ckpt"
RESULTS_DIR="$HOME/scroll_prize/results/s3_inference_$(date +%Y%m%d_%H%M%S)"
PARTS_DIR="${DATA_DIR}/scroll3/fragments/${SEG_ID}/parts_contrasted"

mkdir -p "${RESULTS_DIR}"

cd "${PROJ}"

echo "=== Scroll 3 Zero-Shot Inference ==="
echo "Segment: ${SEG_ID}"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Run inference on each chunk.
# Model was trained with layers 5-20 (16-layer window) on 21-layer fragments.
# Our segment has 65 layers. Try start layers 24-40 (surface-adjacent depth range).
# Each run: 17 windows of 16 layers, TTA level 1, stride 2.
for CHUNK_DIR in $(ls -d ${PARTS_DIR}/${SEG_ID}_*/); do
    CHUNK_NAME=$(basename "${CHUNK_DIR}")
    echo ""
    echo ">>> Processing chunk: ${CHUNK_NAME}"
    echo "    Path: ${CHUNK_DIR}"

    python scripts/inference.py \
        "${CHUNK_DIR}" \
        "${CKPT}" \
        --start_layer 24 \
        --end_layer 40 \
        --stride 2 \
        --tta_level 1 \
        --batch_size 32 \
        --output_dir "${RESULTS_DIR}" \
        --num_workers 8

    echo "    Chunk ${CHUNK_NAME} done."
done

echo ""
echo "=== All chunks processed ==="
echo "Results saved to: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}/"

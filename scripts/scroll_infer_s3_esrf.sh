#!/bin/bash
#SBATCH --job-name=scroll_infer_s3e
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/scroll_infer_s3e_%J.out
#SBATCH --error=logs/scroll_infer_s3e_%J.err

# Run MiniUNETR inference on Scroll 3 segment 20240618142020
# using the ESRF-trained checkpoint.
# Set CKPT_PATH before submitting, or it auto-detects latest checkpoint.

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=offline

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
SEG_DIR="$HOME/scroll_prize/data/scroll3/fragments/20240618142020"

# Auto-detect latest ESRF checkpoint if not set
if [ -z "${CKPT_PATH}" ]; then
    CKPT_PATH=$(find "${PROJ}/checkpoints/scroll0" -name "best-checkpoint*.ckpt" \
        2>/dev/null | sort | tail -1)
    if [ -z "${CKPT_PATH}" ]; then
        echo "ERROR: No checkpoint found in ${PROJ}/checkpoints/scroll0/"; exit 1
    fi
fi
echo "Using checkpoint: ${CKPT_PATH}"

RESULTS_DIR="$HOME/scroll_prize/results/s3_esrf_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# Create a "parts" symlink so inference uses 16-layer windows (matching training)
# inference.py uses `in_chans = 16 if "parts" in fragment_path else 12`
PARTS_DIR="${PROJ}/data/scroll3_parts"
PARTS_LINK="${PARTS_DIR}/20240618142020_parts_01"
if [ ! -d "${PARTS_LINK}" ]; then
    mkdir -p "${PARTS_LINK}/layers"
    for i in $(seq 0 64); do
        layer=$(printf "%02d" $i)
        ln -sf "${SEG_DIR}/layers/${layer}.tif" "${PARTS_LINK}/layers/${layer}.tif"
    done
    ln -sf "${SEG_DIR}/20240618142020_mask.png" \
        "${PARTS_LINK}/20240618142020_parts_01_mask.png"
    echo "Created parts symlink: ${PARTS_LINK}"
fi

cd "${PROJ}"

echo "=== Scroll 3 MiniUNETR Inference (ESRF-trained) ==="
echo "Segment: ${PARTS_LINK}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Results: ${RESULTS_DIR}"

# Run inference across the depth range (start_layer 24-40 covers surface)
python scripts/inference.py \
    "${PARTS_LINK}" \
    "${CKPT_PATH}" \
    --start_layer 24 \
    --end_layer 40 \
    --stride 2 \
    --tta_level 1 \
    --batch_size 32 \
    --output_dir "${RESULTS_DIR}" \
    --num_workers 8

echo ""
echo "=== Done. Results: ==="
ls -lh "${RESULTS_DIR}/"
echo ""
echo "Copy to local: scp -r prajna:${RESULTS_DIR} ./s3_esrf_results/"

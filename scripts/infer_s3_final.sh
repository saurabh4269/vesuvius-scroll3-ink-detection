#!/bin/bash
#SBATCH --job-name=final_inference_s3
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --output=logs/infer_final_%J.out
#SBATCH --error=logs/infer_final_%J.err

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

# Find the best checkpoint from Phase 2 (transfer learning)
# Look for newest best_epoch checkpoint in transfer directory
TRANSFER_CKPT=$(find ${PROJ}/checkpoints/ft_esrf_transfer_* -name "best_epoch_*.pt" -type f 2>/dev/null | sort -V | tail -1)

if [ -z "$TRANSFER_CKPT" ]; then
    echo "ERROR: No transfer learning checkpoint found!"
    echo "Falling back to Phase 1 baseline..."
    TRANSFER_CKPT="${PROJ}/checkpoints/ft_esrf_manual_20260531_003040/best_epoch_015_val_loss_0.6041.pt"
fi

SEGMENT_DIR="$HOME/scroll_prize/data/scroll3/fragments/20240618142020"
OUTPUT_DIR="${PROJ}/results/scroll3_final_$(date +%Y%m%d_%H%M%S)"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== Final Inference: Best MiniUNETR Model on Scroll 3 ==="
echo "Using checkpoint: ${TRANSFER_CKPT}"
echo "Segment: ${SEGMENT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python -u scripts/infer_s3_esrf.py \
    "${CFG}" \
    "${TRANSFER_CKPT}" \
    --segment-dir "${SEGMENT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Final inference complete ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "SUCCESS! Check results:"
echo "  PNG: ${OUTPUT_DIR}/scroll3_20240618142020_prediction.png"
echo "  NPY: ${OUTPUT_DIR}/scroll3_20240618142020_prediction.npy"

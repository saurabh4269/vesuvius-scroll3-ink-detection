#!/bin/bash
#SBATCH --job-name=infer_s3_esrf
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/infer_s3_%J.out
#SBATCH --error=logs/infer_s3_%J.err

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
CKPT="${PROJ}/checkpoints/ft_esrf_manual_20260531_003040/best_epoch_015_val_loss_0.6041.pt"
SEGMENT_DIR="$HOME/scroll_prize/data/scroll3/fragments/20240618142020"
OUTPUT_DIR="${PROJ}/results/scroll3_esrf_inference_$(date +%Y%m%d_%H%M%S)"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== MiniUNETR Inference on Scroll 3 ==="
echo "Config: ${CFG}"
echo "Checkpoint: ${CKPT}"
echo "Segment: ${SEGMENT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python -u scripts/infer_s3_esrf.py "${CFG}" "${CKPT}" --segment-dir "${SEGMENT_DIR}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Inference complete ==="
echo "Results saved to: ${OUTPUT_DIR}"

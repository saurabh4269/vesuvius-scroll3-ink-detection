#!/bin/bash
#SBATCH --job-name=train_full_esrf
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/train_full_%J.out
#SBATCH --error=logs/train_full_%J.err

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
CKPT_DIR="${PROJ}/checkpoints/ft_esrf_manual_$(date +%Y%m%d_%H%M%S)"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== MiniUNETR Full Training (No Lightning) ==="
echo "Config: ${CFG}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Epochs: 20"
echo ""

python -u scripts/train_full.py "${CFG}" --epochs 20 --checkpoint-dir "${CKPT_DIR}"

echo ""
echo "=== Training complete ==="
echo "Final checkpoints: ${CKPT_DIR}"

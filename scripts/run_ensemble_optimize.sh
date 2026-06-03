#!/bin/bash
#SBATCH --job-name=ensemble_opt
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/ensemble_opt_%J.out
#SBATCH --error=logs/ensemble_opt_%J.err

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
mkdir -p ~/scroll_prize/logs

cd "${PROJ}"

echo "=== Smart Ensemble Optimization — Phase 4 Option B ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Models:"
echo "  baseline  → val_loss 0.6041"
echo "  transfer  → val_loss 0.6122"
echo "  augmented → val_loss 0.6126"
echo ""

python -u scripts/ensemble_optimize.py

echo ""
echo "=== Ensemble optimization complete ==="
echo "End time: $(date)"

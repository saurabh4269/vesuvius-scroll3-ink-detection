#!/bin/bash
#SBATCH --job-name=scroll_manual_train
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/scroll_manual_train_%J.out
#SBATCH --error=logs/scroll_manual_train_%J.err

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

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== Manual Training Loop Test ==="
echo "Config: ${CFG}"
echo ""

python -u scripts/train_manual.py "${CFG}" --max-batches 1

echo "=== Manual training complete ==="

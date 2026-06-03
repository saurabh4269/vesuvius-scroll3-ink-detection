#!/bin/bash
#SBATCH --job-name=train_minimal
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_minimal_%J.out
#SBATCH --error=logs/train_minimal_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
CUDNN_LIB="$HOME/miniconda3/envs/scroll/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
CFG="${PROJ}/configs/ft_esrf.py"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== Minimal Training Loop (No Lightning) ==="
python -u scripts/train_minimal.py "${CFG}"
echo "=== Test Complete ==="

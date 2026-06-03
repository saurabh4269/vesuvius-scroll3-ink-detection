#!/bin/bash
#SBATCH --job-name=test_dataloader
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=logs/test_dataloader_%J.out
#SBATCH --error=logs/test_dataloader_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
CFG="${PROJ}/configs/ft_esrf.py"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== Dataloader Test (No Lightning) ==="
python -u scripts/test_dataloader.py "${CFG}"
echo "=== Test Complete ==="

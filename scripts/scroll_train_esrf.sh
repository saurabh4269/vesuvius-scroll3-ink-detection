#!/bin/bash
#SBATCH --job-name=scroll_esrf_train
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
# a40 (Ampere A40) instead of l40 (Ada Lovelace L40S) — l40S has broken cuDNN for Conv3d
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/scroll_esrf_train_%J.out
#SBATCH --error=logs/scroll_esrf_train_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=offline
export WANDB_API_KEY=local-offline-run
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRAINING_PRECISION=16-mixed
export CUDA_VISIBLE_DEVICES=0
# cuDNN 9.1 sub-libraries (modular cuDNN) need to be on LD_LIBRARY_PATH for compute nodes
CUDNN_LIB="$HOME/miniconda3/envs/scroll/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
CFG="${PROJ}/configs/ft_esrf.py"
RESULTS="$HOME/scroll_prize/results"

mkdir -p "${RESULTS}"
cd "${PROJ}"

echo "=== MiniUNETR Training on ESRF Fragments 500P2 + 343P ==="
echo "Config: ${CFG}"
echo "Output checkpoint: ${PROJ}/checkpoints/"
echo ""

# Step 1: create dataset patches (skip if dataset already exists)
DATASET_DIR="../data/esrf/datasets/ft_esrf_500P2_343P"
if [ ! -f "${DATASET_DIR}/label_infos.csv" ]; then
    echo "--- Creating dataset patches ---"
    python scripts/create_dataset.py "${CFG}" --ide_is_closed
    echo "--- Dataset creation done ---"
else
    echo "--- Dataset already exists at ${DATASET_DIR}, skipping creation ---"
fi
echo ""

# Step 2: train
echo "--- Starting training ---"
python scripts/train.py "${CFG}"
echo "--- Training done ---"

# Copy best checkpoint to results
cp -r "${PROJ}/checkpoints/" "${RESULTS}/esrf_checkpoints_$(date +%Y%m%d_%H%M%S)/"
echo "Checkpoint saved to results."

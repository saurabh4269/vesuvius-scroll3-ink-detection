#!/bin/bash
#SBATCH --job-name=train_transfer_esrf
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/train_transfer_%J.out
#SBATCH --error=logs/train_transfer_%J.err

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
TIMESFORMER_CKPT="${HOME}/scroll_prize/villa/ink-detection/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"
CKPT_DIR="${PROJ}/checkpoints/ft_esrf_transfer_$(date +%Y%m%d_%H%M%S)"

mkdir -p ~/scroll_prize/logs
cd "${PROJ}"

echo "=== MiniUNETR Transfer Learning from TimeSormer ==="
echo "Config: ${CFG}"
echo "TimeSormer checkpoint: ${TIMESFORMER_CKPT}"
echo "Transfer learning rate: 5e-5"
echo "Warmup epochs: 1"
echo "Total epochs: 20"
echo "Checkpoint dir: ${CKPT_DIR}"
echo ""

python -u scripts/train_full_transfer.py \
    "${CFG}" \
    --epochs 20 \
    --checkpoint-dir "${CKPT_DIR}" \
    --timesformer-ckpt "${TIMESFORMER_CKPT}" \
    --transfer-lr 5e-5 \
    --warmup-epochs 1

echo ""
echo "=== Transfer learning complete ==="
echo "Final checkpoints: ${CKPT_DIR}"
echo ""
echo "To run inference with transferred model:"
echo "  python scripts/infer_s3_esrf.py ${CFG} ${CKPT_DIR}/best_epoch_*.pt"

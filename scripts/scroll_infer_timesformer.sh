#!/bin/bash
#SBATCH --job-name=scroll_infer_ts
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/scroll_infer_ts_%J.out
#SBATCH --error=logs/scroll_infer_ts_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

export WANDB_MODE=disabled   # no wandb auth needed

INK_DIR="$HOME/scroll_prize/villa/ink-detection"
SEG_PATH="$HOME/scroll_prize/data/scroll3/fragments"
SEG_ID="20240618142020"
# The "canonical" TimeSformer checkpoint from the Grand Prize solution
CKPT="${INK_DIR}/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"
OUT_DIR="$HOME/scroll_prize/results/s3_timesformer_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}"

echo "=== Scroll 3 TimeSformer Zero-Shot Inference ==="
echo "Segment: ${SEG_ID}"
echo "Checkpoint: $(ls -lh ${CKPT})"
echo "Results: ${OUT_DIR}"
echo ""

# Default: start_idx=17 → uses layers 17-42 (26 layers from depth center)
# Also try start_idx=24 → layers 24-49 (slightly deeper)
for START in 17 24; do
    echo "--- Running with start_layer=${START} (layers ${START}-$((START+25))) ---"
    python "${INK_DIR}/inference_timesformer.py" \
        --segment_path "${SEG_PATH}" \
        --segment_id "${SEG_ID}" \
        --model_path "${CKPT}" \
        --out_path "${OUT_DIR}/start${START}" \
        --start_idx ${START} \
        --stride 32 \
        --batch_size 64 \
        --workers 8 \
        --gpus 1
    echo "--- start_layer=${START} done. Output: ${OUT_DIR}/start${START}/ ---"
done

echo ""
echo "=== Inference complete. Results: ==="
ls -lh "${OUT_DIR}/start17/"
echo ""
echo "To view results, copy to local machine:"
echo "  scp -r prajna:${OUT_DIR} ./s3_results/"

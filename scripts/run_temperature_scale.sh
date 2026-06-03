#!/bin/bash
#SBATCH --job-name=temp_scale
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=logs/temp_scale_%J.out
#SBATCH --error=logs/temp_scale_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"

echo "=== Temperature Scaling on Existing Predictions ==="
echo "Start: $(date)"
echo ""

# Scale the baseline prediction (best single model, val_loss 0.6041)
BASELINE_PRED="${PROJ}/results/scroll3_final_20260531_042222/scroll3_20240618142020_prediction.npy"
echo "--- Baseline prediction ---"
python -u "${PROJ}/scripts/temperature_scale.py" \
    --input "${BASELINE_PRED}" \
    --temperatures 0.2 0.3 0.5 0.7 \
    --save-all

# Scale the equal-weight ensemble prediction
ENSEMBLE_PRED="${PROJ}/results/ensemble_opt_20260531_172449/scroll3_20240618142020_ensemble_equal.npy"
echo ""
echo "--- Equal-weight ensemble prediction ---"
python -u "${PROJ}/scripts/temperature_scale.py" \
    --input "${ENSEMBLE_PRED}" \
    --temperatures 0.2 0.3 0.5 0.7 \
    --save-all

# Scale the optimal-weight ensemble (100% augmented)
OPTIMAL_PRED="${PROJ}/results/ensemble_opt_20260531_172449/scroll3_20240618142020_ensemble_optimal.npy"
echo ""
echo "--- Optimal-weight ensemble prediction ---"
python -u "${PROJ}/scripts/temperature_scale.py" \
    --input "${OPTIMAL_PRED}" \
    --temperatures 0.2 0.3 0.5 0.7 \
    --save-all

echo ""
echo "=== Temperature scaling complete ==="
echo "End: $(date)"

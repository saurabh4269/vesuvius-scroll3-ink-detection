#!/bin/bash
# Scroll Prize — SLURM job template for Prajna HPC
# Usage:  cd $HOME && sbatch scroll_prize/scroll_train.sh
# Docs:   PRAJNA_HPC.md §4 (partitions/limits), PRAJNA_RUNBOOK.md §6 (pitfalls)
#
# Partition guide:
#   l40   — default, 7 nodes, 8×L40S 48 GB, max 4 GPUs/job, max 4 running, 2-day wall
#   a40   — 19 nodes, 4×A40 48 GB, max 2 GPUs/job, max 3 running, 4-day wall
#   dgx   — 9 nodes, 8×A100 80 GB, max 4 GPUs/job, max 4 running, 6-day wall
#   debug — 1 A40 node, 30-min wall, no GPU cap, near-zero queue — use for quick tests

# ── Partition / QOS — MUST match exactly ─────────────────────────────────────
#SBATCH --partition=l40
#SBATCH --qos=l40

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # CPU cores (l40 max=32, a40 max=64, dgx max=256)
#SBATCH --gres=gpu:1                 # GPUs per job (l40≤4, a40≤2, dgx≤4)
#SBATCH --mem=64G                    # RAM per node (be reasonable — no hard cap)
#SBATCH --time=06:00:00              # HH:MM:SS — accurate estimate = better backfill

# ── Job identity ──────────────────────────────────────────────────────────────
#SBATCH --job-name=scroll_train
#SBATCH --output=logs/scroll_%J.out  # relative to $HOME — always run sbatch from $HOME
#SBATCH --error=logs/scroll_%J.err
# Uncomment for email notifications:
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=22b3907@iitb.ac.in

# ── SLURM tools (not in PATH in non-interactive shells) ───────────────────────
export PATH=$PATH:/opt/slurm/bin

# ── Shell options — NOTE: -u breaks conda, keep it as -eo ─────────────────────
set -eo pipefail

# ── Conda — MUST be sourced explicitly in SLURM jobs ─────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:     $(date)"
echo "GPUs:      $SLURM_JOB_GPUS"
echo ""
python -c "import torch; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count(), '|', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# ── Project dir ───────────────────────────────────────────────────────────────
PROJ=$HOME/scroll_prize
cd $PROJ

# ── Scratch — write heavy I/O here, copy results to home afterwards ───────────
# NOTE: /lustre-scratch/<user> requires admin to create — use home until provisioned
# When provisioned: SCRATCH=/lustre-scratch/shiwani.mishra/scroll_prize
SCRATCH=$HOME/scroll_prize/results
mkdir -p $SCRATCH

# ── Your training command ─────────────────────────────────────────────────────
# Replace the line below with your actual command, e.g.:
#   python villa/ink-detection/train.py \
#     --output $SCRATCH \
#     --config configs/timesformer_scroll1.yaml
echo "TODO: replace this placeholder with your training command"
# python train.py --output $SCRATCH

# ── Always copy final outputs to home (scratch purged after 3 months) ─────────
# (Already in home here, but if using LUSTRE_SCRATCH uncomment below)
# cp -r $SCRATCH $HOME/scroll_prize/results/

echo "=== Done: $(date) ==="

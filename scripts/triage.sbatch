#!/bin/bash
#SBATCH --job-name=m7triage
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/triage_%J.out
#SBATCH --error=logs/triage_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

cd ~/scroll_prize/vesuvius_first_title_prize
python scripts/triage_scan.py ~/scroll_prize/data/m7_triage
echo "TRIAGE JOB DONE"

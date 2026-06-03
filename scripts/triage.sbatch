#!/bin/bash
#SBATCH -J m7triage
#SBATCH -p l40
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /home/medal/shiwani.mishra/scroll_prize/logs/triage_%j.out
#SBATCH -e /home/medal/shiwani.mishra/scroll_prize/logs/triage_%j.err
source ~/.bashrc
conda activate scroll
cd ~/scroll_prize/vesuvius_first_title_prize
python scripts/triage_scan.py ~/scroll_prize/data/m7_triage
echo "TRIAGE JOB DONE"

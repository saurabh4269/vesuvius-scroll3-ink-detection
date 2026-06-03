#!/bin/bash
#SBATCH --job-name=scroll_deps
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:25:00
#SBATCH --mem=16G
#SBATCH --output=logs/scroll_deps_%J.out
#SBATCH --error=logs/scroll_deps_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

# Install missing deps (no zarr downgrade — test with 2.18.3 first)
pip install --quiet gdown dask numcodecs imagecodecs \
    segmentation_models_pytorch torchmetrics \
    beautifulsoup4 termcolor future h5py

# Install the phoenix package from the first title winner repo
cd ~/scroll_prize/vesuvius_first_title_prize
pip install --quiet -e .

echo "=== Dep install complete ==="
python -c "
import dask, numcodecs, imagecodecs, segmentation_models_pytorch
import phoenix
print('All deps OK')
print('phoenix location:', phoenix.__file__)
"

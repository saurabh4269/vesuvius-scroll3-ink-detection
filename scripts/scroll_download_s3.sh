#!/bin/bash
#SBATCH --job-name=scroll_dl_s3
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:25:00
#SBATCH --mem=8G
#SBATCH --output=logs/scroll_dl_s3_%J.out
#SBATCH --error=logs/scroll_dl_s3_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

SEG_ID="20240618142020"
BASE_URL="https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/${SEG_ID}"
DATA_DIR="$HOME/scroll_prize/data/scroll3/fragments/${SEG_ID}"
CKPT_DIR="$HOME/scroll_prize/vesuvius_first_title_prize/checkpoints/scroll5/warm-planet-193-unetr-sf-b3-250417-171532"

mkdir -p "${DATA_DIR}/layers"
mkdir -p "${CKPT_DIR}"

echo "=== Downloading checkpoint from Google Drive ==="
CKPT_FILE="${CKPT_DIR}/best-checkpoint-epoch=13-val_iou=0.71.ckpt"
if [ ! -f "${CKPT_FILE}" ]; then
    gdown "https://drive.google.com/uc?id=1OTMnO7bgPQRUlzQZ2m7dd924FEwFDdQz" \
        -O "${CKPT_FILE}"
    echo "Checkpoint downloaded: $(du -sh ${CKPT_FILE})"
else
    echo "Checkpoint already exists, skipping."
fi

echo "=== Downloading segment mask ==="
MASK_FILE="${DATA_DIR}/${SEG_ID}_mask.png"
if [ ! -f "${MASK_FILE}" ]; then
    wget -q "${BASE_URL}/${SEG_ID}_mask.png" -O "${MASK_FILE}"
    echo "Mask downloaded."
else
    echo "Mask already exists, skipping."
fi

echo "=== Downloading 65 segment layers (00-64) ==="
# Download in parallel with 8 concurrent connections
download_layer() {
    local i=$1
    local layer=$(printf "%02d" $i)
    local dest="${DATA_DIR}/layers/${layer}.tif"
    if [ ! -f "${dest}" ] || [ ! -s "${dest}" ]; then
        wget -q "${BASE_URL}/layers/${layer}.tif" -O "${dest}"
    fi
}
export -f download_layer
export DATA_DIR BASE_URL

# GNU parallel if available, otherwise sequential
if command -v parallel &>/dev/null; then
    seq 0 64 | parallel -j8 download_layer {}
else
    for i in $(seq 0 64); do
        download_layer $i
        echo -ne "\r  Layer $i/64 downloaded..."
    done
    echo ""
fi

echo "=== Verifying download ==="
COUNT=$(ls "${DATA_DIR}/layers/"*.tif 2>/dev/null | wc -l)
echo "Layers downloaded: ${COUNT}/65"
echo "Mask: $(ls -lh ${MASK_FILE} 2>/dev/null || echo MISSING)"
echo "Checkpoint: $(ls -lh ${CKPT_FILE} 2>/dev/null || echo MISSING)"

if [ "${COUNT}" -lt 65 ]; then
    echo "ERROR: Not all layers downloaded. Missing $((65 - COUNT)) layers."
    exit 1
fi

echo "=== Download complete ==="
du -sh "${DATA_DIR}"

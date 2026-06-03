#!/bin/bash
#SBATCH --job-name=scroll_prep_s3
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/scroll_prep_s3_%J.out
#SBATCH --error=logs/scroll_prep_s3_%J.err

export PATH=$PATH:/opt/slurm/bin
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scroll

SEG_ID="20240618142020"
PROJ="$HOME/scroll_prize/vesuvius_first_title_prize"
DATA_DIR="$HOME/scroll_prize/data"

cd "${PROJ}"

echo "=== Running fragment_splitter with contrast (scroll 3) ==="
# This splits the 25706px-wide segment into 3 chunks (~10000px each)
# and applies the LUT contrast transform. Creates:
#   data/scroll3/fragments/{SEG_ID}/parts_contrasted/{SEG_ID}_01/layers/
#   data/scroll3/fragments/{SEG_ID}/parts_contrasted/{SEG_ID}_02/layers/
#   data/scroll3/fragments/{SEG_ID}/parts_contrasted/{SEG_ID}_03/layers/
python scripts/fragment_splitter.py "${SEG_ID}" \
    --scroll-id 3 \
    --base-path "${DATA_DIR}" \
    --contrasted \
    --verbose

echo "=== Splitter done. Checking output ==="
ls "${DATA_DIR}/scroll3/fragments/${SEG_ID}/parts_contrasted/"

echo "=== Copying cropped masks to each chunk dir ==="
python3 - <<'PYEOF'
import os
from PIL import Image
import numpy as np

seg_id = "20240618142020"
data_dir = os.path.expanduser("~/scroll_prize/data")
parts_dir = f"{data_dir}/scroll3/fragments/{seg_id}/parts_contrasted"
mask_src = f"{data_dir}/scroll3/fragments/{seg_id}/{seg_id}_mask.png"

# Load original mask
orig_mask = np.array(Image.open(mask_src).convert("L"))
orig_w = orig_mask.shape[1]
print(f"Original mask shape: {orig_mask.shape}")

chunk_width = 10000
chunks = sorted(os.listdir(parts_dir))
print(f"Chunks found: {chunks}")

for chunk_name in chunks:
    chunk_dir = os.path.join(parts_dir, chunk_name)
    if not os.path.isdir(chunk_dir):
        continue
    # Parse chunk index from name (e.g. 20240618142020_01 -> idx 0)
    idx = int(chunk_name.split("_")[-1]) - 1
    x_start = idx * chunk_width
    x_end = min(x_start + chunk_width, orig_w)
    cropped = orig_mask[:, x_start:x_end]
    mask_out = os.path.join(chunk_dir, f"{chunk_name}_mask.png")
    Image.fromarray(cropped).save(mask_out)
    print(f"  {chunk_name}: x=[{x_start}:{x_end}] -> mask {cropped.shape} -> {mask_out}")

print("Mask creation done.")
PYEOF

echo "=== Prep complete. Chunk structure ==="
find "${DATA_DIR}/scroll3/fragments/${SEG_ID}/parts_contrasted" -name "*mask.png" | sort
find "${DATA_DIR}/scroll3/fragments/${SEG_ID}/parts_contrasted" -name "layers" | sort

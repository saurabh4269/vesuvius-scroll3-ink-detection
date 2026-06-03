#!/bin/bash
# Run this on Prajna to dump everything needed to write the training adapter.
# Usage: bash explore_villa.sh 2>&1 | tee villa_report.txt
# Then paste villa_report.txt back into the chat.

VILLA=~/scroll_prize/villa

echo "===== 1. TOP-LEVEL STRUCTURE ====="
ls -1 $VILLA

echo ""
echo "===== 2. INK-DETECTION DIRECTORY ====="
ls -1 $VILLA/ink-detection/

echo ""
echo "===== 3. ALL_LABELS CONTENTS ====="
ls -1 $VILLA/ink-detection/all_labels/

echo ""
echo "===== 4. train_resnet3d.py (first 200 lines) ====="
head -200 $VILLA/ink-detection/train_resnet3d.py

echo ""
echo "===== 5. CONFIG FILES ====="
ls -1 $VILLA/ink-detection/train_resnet3d_lib/ 2>/dev/null || echo "(no train_resnet3d_lib dir)"
find $VILLA/ink-detection -name "*.yaml" -o -name "*.yml" 2>/dev/null | head -20

echo ""
echo "===== 6. FIRST YAML/CONFIG FOUND ====="
CONFIG=$(find $VILLA/ink-detection -name "*.yaml" | head -1)
if [ -n "$CONFIG" ]; then
    echo "File: $CONFIG"
    cat "$CONFIG"
else
    echo "(no yaml config found)"
fi

echo ""
echo "===== 7. scrolls.yaml (vesuvius package) ====="
find $VILLA/vesuvius -name "scrolls.yaml" 2>/dev/null | while read f; do
    echo "File: $f"
    cat "$f"
done

echo ""
echo "===== 8. vesuvius Volume class (first 100 lines) ====="
find $VILLA/vesuvius -name "volume.py" 2>/dev/null | head -1 | xargs head -100 2>/dev/null || echo "(not found)"

echo ""
echo "===== 9. IS BCE BUG IN VILLA? ====="
echo "--- Searching for BCE / binary_cross_entropy in villa training code ---"
grep -rn "binary_cross_entropy\|BCEWithLogits\|BCELoss" $VILLA/ink-detection/ 2>/dev/null | head -30

echo ""
echo "===== 10. LABEL LOADING IN VILLA ====="
grep -rn "label\|inklabel\|channel\|[:,0\|[:,1" $VILLA/ink-detection/train_resnet3d.py 2>/dev/null | head -30

echo ""
echo "===== 11. S3 URL FORMAT FOR SCROLL 1 SEGMENTS ====="
grep -rn "vesuvius-challenge-open-data\|Scroll1\|PHercParis\|paths/" $VILLA/vesuvius/ 2>/dev/null | head -20

echo ""
echo "===== 12. ESRF FRAGMENTS ON PRAJNA ====="
ls ~/scroll_prize/data/esrf/ 2>/dev/null || echo "(not found)"

echo ""
echo "===== 13. CURRENT BEST CHECKPOINT ====="
find ~/scroll_prize/vesuvius_first_title_prize/checkpoints -name "*.pt" 2>/dev/null | sort | tail -5

echo ""
echo "===== 14. CONDA ENV PACKAGES (key ones) ====="
conda activate scroll 2>/dev/null
pip list 2>/dev/null | grep -iE "torch|vesuvius|zarr|s3fs|segformer|transformers|batchgenerators" || true

echo ""
echo "===== DONE — paste this output back into the chat ====="

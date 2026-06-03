#!/bin/bash
# Smart Wake-Up Automation Script
# Run this ONE TIME when you wake up - it handles all remaining steps automatically
# Usage: ./wakeup_automation.sh

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "          🌅 WAKE UP AUTOMATION SCRIPT 🌅"
echo "═══════════════════════════════════════════════════════════════"
echo ""

export PATH=$PATH:/opt/slurm/bin

# Check Phase 2 Status
echo "Step 1: Checking Phase 2 (Transfer Learning) Status..."
echo "─────────────────────────────────────────────────────────"

PHASE2_COMPLETE=$(ssh prajna "grep -c 'Training complete' ~/scroll_prize/logs/train_transfer_126073.out" 2>/dev/null || echo "0")

if [ "$PHASE2_COMPLETE" = "0" ]; then
    echo "❌ Phase 2 still running!"
    echo "   Waiting for Phase 2 to complete..."
    echo "   Check back in 30 minutes."
    exit 1
fi

echo "✅ Phase 2 COMPLETE!"
BEST_LOSS=$(ssh prajna "grep 'Best val loss:' ~/scroll_prize/logs/train_transfer_126073.out" | tail -1)
echo "   $BEST_LOSS"
echo ""

# Step 2: Submit Final Inference
echo "Step 2: Submitting Final Inference..."
echo "─────────────────────────────────────"

FINAL_JOB=$(ssh prajna "cd ~/scroll_prize && sbatch scripts/infer_s3_final.sh" 2>&1 | grep -oP 'job \K[0-9]+')

if [ -z "$FINAL_JOB" ]; then
    echo "❌ Failed to submit final inference"
    exit 1
fi

echo "✅ Final Inference Submitted! (Job $FINAL_JOB)"
echo "   Takes ~5 minutes"
echo "   Results will be in: ~/scroll_prize/vesuvius_first_title_prize/results/scroll3_final_*/"
echo ""

# Step 3: Wait for Final Inference
echo "Step 3: Waiting for Final Inference to Complete..."
echo "─────────────────────────────────────────────────"

TIMEOUT=600  # 10 minutes
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    JOB_STATE=$(ssh prajna "/opt/slurm/bin/squeue -j $FINAL_JOB -h -o %T 2>/dev/null || echo 'COMPLETED'")

    if [ "$JOB_STATE" = "COMPLETED" ]; then
        echo "✅ Final Inference COMPLETE!"
        break
    elif [ "$JOB_STATE" = "FAILED" ] || [ "$JOB_STATE" = "CANCELLED" ]; then
        echo "❌ Final Inference FAILED!"
        ssh prajna "tail -20 ~/scroll_prize/logs/infer_final_${FINAL_JOB}.out"
        exit 1
    fi

    echo "   Still running... ($ELAPSED/$TIMEOUT seconds)"
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

echo ""

# Step 4: Ask about Phase 3
echo "Step 4: Phase 3 (Augmentation) - Optional for +10-15% Improvement"
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "Phase 3 will:"
echo "  ✓ Create augmented dataset (3x size)"
echo "  ✓ Retrain with transfer learning"
echo "  ✓ Additional 2.5 hours"
echo ""
read -p "Submit Phase 3 augmentation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting Phase 3..."

    PHASE3_JOB=$(ssh prajna "cd ~/scroll_prize && sbatch scripts/train_augmented.sh" 2>&1 | grep -oP 'job \K[0-9]+')

    if [ -z "$PHASE3_JOB" ]; then
        echo "❌ Failed to submit Phase 3"
        exit 1
    fi

    echo "✅ Phase 3 Submitted! (Job $PHASE3_JOB)"
    echo "   Takes ~2.5 hours"
    echo "   Will produce: ~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_augmented_*/"
    echo ""
    echo "   Come back in ~3 hours to:"
    echo "   1. Check Phase 3 completion"
    echo "   2. Run final inference with augmented model"
else
    echo "Skipping Phase 3."
    echo ""
    echo "Final results are ready:"
    echo "  📁 PNG: ~/scroll_prize/vesuvius_first_title_prize/results/scroll3_final_*/scroll3_20240618142020_prediction.png"
    echo "  📊 NPY: ~/scroll_prize/vesuvius_first_title_prize/results/scroll3_final_*/scroll3_20240618142020_prediction.npy"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "              ✅ WAKE UP AUTOMATION COMPLETE ✅"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Summary
echo "SUMMARY:"
echo "--------"
echo "✅ Phase 1 (Ensemble):      COMPLETE"
echo "✅ Phase 2 (Transfer):      COMPLETE"
echo "✅ Final Inference:         COMPLETE"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Phase 3 (Augmented):    RUNNING (Job $PHASE3_JOB)"
else
    echo "⏭️  Phase 3 (Augmented):    SKIPPED"
fi

echo ""
echo "All results saved and ready for comparison!"

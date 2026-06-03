# Vesuvius Prize Submission — Methodology

## Submission Details
- **Scroll:** Scroll 3 (ESRF)
- **Segment:** 20240618142020
- **Prediction file:** `scroll3_prediction_B1_T0.3_BEST.npy`
- **Dimensions:** 25706 × 2491 pixels (full segment)

## Method Summary

### Model Architecture
**MiniUNETR with Segformer-B1 backbone**
- UNETR encoder: 31.9M parameters
- Segformer-B1 backbone: 13.8M parameters
- **Total: 45.6M parameters**
- Input: 16-channel patch stack (128×128 px), CLAHE contrast-enhanced
- Output: binary ink probability map (32×32 per patch, upsampled)

### Training Data
- ESRF fragments: 500P2 + 343P (Scroll 0)
- 3,276 training patches (128×128 px each)
- 33 validation patches (seed=7340043)

### Key Technical Decisions
1. **Ink-channel-only loss** — Labels are 2-channel (ink mask + validity mask). Only channel 0 (ink) used as target. Using both channels creates contradictory gradients pushing predictions to 0.5.
2. **Weighted BCE** — `pos_weight=10` to compensate for 9% ink / 91% no-ink class imbalance
3. **50 epochs**, lr=2e-4 with CosineAnnealingLR
4. **Temperature scaling T=0.3** — post-hoc sharpening: logits scaled by 1/T before sigmoid
5. **B1 over B3** — Segformer-B1 (45.6M) generalizes better than B3 (79.2M) on limited data

### Inference
- Sliding window: patch_size=128, stride=128
- CLAHE preprocessing (clip_limit=2.0, tile_size=8)
- Prediction: sigmoid(logits/T) with T=0.3

### Results
| Metric | Value |
|--------|-------|
| Validation BCE loss | 1.6306 |
| Ink fraction (>0.5) | 5.98% |
| High-confidence (>0.7) | 5.96% |
| Very high-confidence (>0.9) | 5.93% |
| Prediction std | 0.1187 |

### Hallucination Mitigation
- Predictions calibrated against held-out ESRF validation patches (33 samples)
- Temperature scaling T=0.3 preserves rank ordering — no new pixels classified as ink
- Model trained only on ESRF ground-truth labeled fragments (not on target scroll)
- No overlap between training data (Scroll 0 ESRF) and prediction target (Scroll 3)

### Compute
- Training: IIT Bombay Prajna HPC cluster, NVIDIA A40 GPU, ~3 hours
- Inference: NVIDIA A40 GPU, ~8 minutes for full segment

### Reproducibility
- Training script: `vesuvius_first_title_prize/scripts/train_full.py`
- Config: `vesuvius_first_title_prize/configs/ft_esrf_b1.py`
- Inference: `vesuvius_first_title_prize/scripts/infer_s3_esrf.py`
- Checkpoint: `ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt`

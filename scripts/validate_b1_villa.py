#!/usr/bin/env python
"""
Validate B1 ink-detection model on villa's Scroll 1/2 labeled segments.
Answers: does the model detect ink, or just papyrus fibers?

Run on Prajna (scroll conda env, vesuvius package installed):
  python validate_b1_villa.py \
    --checkpoint ~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt \
    --config ~/scroll_prize/vesuvius_first_title_prize/configs/ft_esrf_b1.py \
    --segment-id 20231007101615 \
    --scroll 1 \
    --label-dir ~/scroll_prize/villa/ink-detection/all_labels/ \
    --output-dir ~/scroll_prize/results/b1_validation/

Validation segments (Scroll 1/2, confirmed ink labels in villa):
  20231007101615, 20231012085431, 20231012173610, 20231016151000
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import tifffile
import s3fs

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.utility.configs import Config


# S3 layer paths for Scroll 1 and 2 segments (Kaggle 2023 format)
SCROLL_S3 = {
    1: "vesuvius-challenge-open-data/Scroll1/PHercParis4.volpkg/paths",
    2: "vesuvius-challenge-open-data/Scroll2/PHercParis3v1.volpkg/paths",
}


def download_layers(seg_id, scroll_id, n_layers=65, cache_dir="/tmp/b1_val_layers"):
    """Download TIFF layers for a Scroll 1/2 segment from public S3 bucket."""
    fs = s3fs.S3FileSystem(anon=True)
    base = f"{SCROLL_S3[scroll_id]}/{seg_id}/layers"
    local_dir = Path(cache_dir) / str(seg_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    for i in range(n_layers):
        local = local_dir / f"{i:02d}.tif"
        if not local.exists():
            s3_path = f"{base}/{i:02d}.tif"
            try:
                fs.get(s3_path, str(local))
            except Exception:
                break  # no more layers at this index
        if local.exists():
            layers.append(tifffile.imread(str(local)).astype(np.float32))

    if not layers:
        return None
    volume = np.stack(layers, axis=0)
    print(f"[VAL] Downloaded {len(layers)} layers → volume {volume.shape}", flush=True)
    return volume


def apply_clahe(volume, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = volume.copy()
    for z in range(volume.shape[0]):
        layer = volume[z]
        normed = ((layer - layer.min()) / (layer.max() - layer.min() + 1e-8) * 255).astype(np.uint8)
        out[z] = clahe.apply(normed).astype(np.float32) / 255.0
    return out


def infer_volume(model, volume, device, patch_size=128, stride=64, in_chans=16):
    model.eval()
    z, h, w = volume.shape
    if z < in_chans:
        volume = np.pad(volume, ((0, in_chans - z), (0, 0), (0, 0)), mode="reflect")

    prediction = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                z_start = max(0, volume.shape[0] - in_chans)
                patch = volume[z_start:z_start + in_chans, y0:y0 + patch_size, x0:x0 + patch_size]
                if patch.shape != (in_chans, patch_size, patch_size):
                    continue
                patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                t = torch.from_numpy(patch[np.newaxis, np.newaxis]).to(device)
                logits = model(t).squeeze().cpu().numpy()
                pred = 1.0 / (1.0 + np.exp(-logits))
                ls = logits.shape[0]
                prediction[y0:y0 + ls, x0:x0 + ls] += pred
                count[y0:y0 + ls, x0:x0 + ls] += 1.0

    np.divide(prediction, count, out=prediction, where=count > 0)
    prediction[count == 0] = 0.5
    return prediction


def load_label(label_dir, seg_id):
    """Load ink label PNG from villa's all_labels directory."""
    label_dir = Path(label_dir)
    for name in [
        f"{seg_id}_inklabels.png",
        f"{seg_id}.png",
        f"{seg_id}_labels.png",
    ]:
        p = label_dir / name
        if p.exists():
            lbl = np.array(Image.open(p).convert("L"))
            return (lbl > 127).astype(np.uint8)
    # try subdirectory layout
    p = label_dir / str(seg_id) / f"{seg_id}_inklabels.png"
    if p.exists():
        lbl = np.array(Image.open(p).convert("L"))
        return (lbl > 127).astype(np.uint8)
    return None


def compute_metrics(pred_prob, label, threshold):
    pred = (pred_prob >= threshold).astype(np.uint8)
    if pred.shape != label.shape:
        label = cv2.resize(label.astype(np.float32),
                           (pred.shape[1], pred.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        label = (label > 0.5).astype(np.uint8)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return dict(threshold=threshold, precision=precision, recall=recall,
                f1=f1, dice=dice, pred_frac=pred.mean(), label_frac=label.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to B1 .pt checkpoint")
    p.add_argument("--config", required=True, help="Path to ft_esrf_b1.py config")
    p.add_argument("--segment-id", required=True, help="Segment ID, e.g. 20231007101615")
    p.add_argument("--scroll", type=int, default=1, choices=[1, 2])
    p.add_argument("--label-dir", required=True, help="Path to villa/ink-detection/all_labels/")
    p.add_argument("--output-dir", default="~/scroll_prize/results/b1_validation")
    p.add_argument("--n-layers", type=int, default=65)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    args = p.parse_args()

    for attr in ("checkpoint", "config", "label_dir", "output_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VAL] Device: {device}", flush=True)

    # Load model
    config = Config.load_from_file(args.config)
    lm = UNETR_SF_Module(**vars(config))
    model = lm.model
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[VAL] Loaded checkpoint: {Path(args.checkpoint).name}", flush=True)

    # Download segment layers from S3
    print(f"[VAL] Fetching Scroll {args.scroll} segment {args.segment_id} from S3...", flush=True)
    volume = download_layers(args.segment_id, args.scroll, args.n_layers)
    if volume is None:
        print("[VAL] ERROR: no layers downloaded — check S3 path and network", flush=True)
        print(f"[VAL] Expected: {SCROLL_S3[args.scroll]}/{args.segment_id}/layers/00.tif", flush=True)
        return 1

    volume = apply_clahe(volume)

    # Run inference
    print(f"[VAL] Running inference (patch={args.patch_size}, stride={args.stride})...", flush=True)
    pred = infer_volume(model, volume, device,
                        patch_size=args.patch_size,
                        stride=args.stride,
                        in_chans=getattr(config, "in_chans", 16))
    print(f"[VAL] Prediction: shape={pred.shape}  range=[{pred.min():.3f}, {pred.max():.3f}]", flush=True)

    # Load ground-truth ink label
    label = load_label(args.label_dir, args.segment_id)
    if label is None:
        print(f"[VAL] No label found for {args.segment_id} in {args.label_dir}", flush=True)
        np.save(str(out_dir / f"{args.segment_id}_pred.npy"), pred)
        print("[VAL] Saved prediction (no metrics — label missing)", flush=True)
        return 1

    print(f"[VAL] Label: shape={label.shape}  ink fraction={label.mean():.4f}", flush=True)

    # Metrics at multiple thresholds
    print("\n[VAL] === VALIDATION METRICS ===", flush=True)
    print(f"{'Thresh':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Dice':>6}  {'PredFrac':>8}", flush=True)
    print("-" * 58, flush=True)
    metrics_rows = []
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        m = compute_metrics(pred, label, thresh)
        metrics_rows.append(m)
        print(f"{thresh:>6.1f}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  "
              f"{m['f1']:>6.4f}  {m['dice']:>6.4f}  {m['pred_frac']:>8.4f}", flush=True)

    best = max(metrics_rows, key=lambda m: m["f1"])
    print(f"\n[VAL] Best F1={best['f1']:.4f} at threshold={best['threshold']}", flush=True)

    if best["f1"] < 0.05:
        print("[VAL] VERDICT: model is NOT detecting ink on this segment (F1 < 0.05)", flush=True)
    elif best["f1"] < 0.20:
        print("[VAL] VERDICT: model shows weak ink signal — may be detecting some fiber patterns", flush=True)
    else:
        print("[VAL] VERDICT: model shows meaningful ink detection signal", flush=True)

    # Save side-by-side visualization
    pred_u8 = (pred * 255).astype(np.uint8)
    label_resized = cv2.resize(label.astype(np.float32),
                               (pred.shape[1], pred.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    label_u8 = (label_resized * 255).astype(np.uint8)
    side = np.hstack([label_u8, pred_u8])
    vis_path = out_dir / f"{args.segment_id}_label_vs_pred.png"
    Image.fromarray(side).save(str(vis_path))

    np.save(str(out_dir / f"{args.segment_id}_pred.npy"), pred)
    print(f"[VAL] Saved: {vis_path.name} (left=ground_truth, right=prediction)", flush=True)
    print(f"[VAL] Output dir: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

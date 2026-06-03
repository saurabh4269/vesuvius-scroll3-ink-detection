#!/usr/bin/env python
"""
Validate B1 ink-detection model on villa's Scroll 1/2 labeled segments.
Answers: does the model detect ink, or just papyrus fibers?

Segment data comes from dl.ash2txt.org zarr (level 0 = Z×H×W).
Labels come from villa/ink-detection/all_labels/{seg_id}_inklabels.png

Run on Prajna (scroll conda env):
  python validate_b1_villa.py \
    --checkpoint ~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt \
    --config ~/scroll_prize/vesuvius_first_title_prize/configs/ft_esrf_b1.py \
    --segment-id 20230827161847 \
    --scroll 1 \
    --label-dir ~/scroll_prize/villa/ink-detection/all_labels/ \
    --output-dir ~/scroll_prize/results/b1_validation/

All available labeled segment IDs (45 total in all_labels/):
  20230520175435, 20230522181603, 20230522215721, 20230530164535,
  20230530172803, 20230530212931, 20230531121653, 20230531193658,
  20230601193301, 20230611014200, 20230620230617, 20230620230619,
  20230701020044, 20230702185753, 20230820203112, 20230826170124,
  20230827161847, 20230901184804, 20230902141231, 20230903193206,
  20230904020426, 20230904135535, 20230905134255, 20230909121925,
  20230929220924, 20230929220926, 20231001164029, 20231004222109,
  20231005123333, 20231005123336, 20231007101615, 20231012085431,
  20231012173610, 20231012184420, 20231012184421, 20231012184423,
  20231016151000, 20231022170900, 20231022170901, 20231031143850,
  20231106155350, 20231106155351, 20231210121321, recto, verso
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import zarr

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.utility.configs import Config


# Scroll 1/2 segment zarr base URLs (dl.ash2txt.org, anon access)
SEG_ZARR_BASE = {
    1: "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/{seg_id}.zarr/",
    2: "https://dl.ash2txt.org/other/dev/scrolls/2/segments/54keV_7.91um/{seg_id}.zarr/",
}


def load_segment_zarr(seg_id, scroll_id, zarr_level=0):
    """Load a Scroll 1/2 segment from dl.ash2txt.org zarr store. Returns (Z, H, W) float32."""
    url = SEG_ZARR_BASE[scroll_id].format(seg_id=seg_id)
    print(f"[VAL] Opening zarr: {url}", flush=True)
    z = zarr.open(url, mode="r")
    arr = np.array(z[str(zarr_level)]).astype(np.float32)
    print(f"[VAL] Loaded level {zarr_level}: shape={arr.shape}  dtype=float32", flush=True)
    return arr


def apply_clahe(volume, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = volume.copy()
    for z in range(volume.shape[0]):
        layer = volume[z]
        lo, hi = layer.min(), layer.max()
        normed = ((layer - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
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
    for name in [f"{seg_id}_inklabels.png", f"{seg_id}_inklabels.tiff",
                 f"{seg_id}.png", f"{seg_id}_labels.png"]:
        p = label_dir / name
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--segment-id", required=True)
    p.add_argument("--scroll", type=int, default=1, choices=[1, 2])
    p.add_argument("--label-dir", required=True)
    p.add_argument("--output-dir", default="~/scroll_prize/results/b1_validation")
    p.add_argument("--zarr-level", type=int, default=0, help="Zarr pyramid level (0=finest)")
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
    print(f"[VAL] Loaded: {Path(args.checkpoint).name}", flush=True)

    # Load segment from zarr
    volume = load_segment_zarr(args.segment_id, args.scroll, args.zarr_level)
    volume = apply_clahe(volume)

    # Run inference
    in_chans = getattr(config, "in_chans", 16)
    print(f"[VAL] Inference: patch={args.patch_size} stride={args.stride} in_chans={in_chans}", flush=True)
    pred = infer_volume(model, volume, device,
                        patch_size=args.patch_size, stride=args.stride, in_chans=in_chans)
    print(f"[VAL] Prediction: shape={pred.shape}  range=[{pred.min():.3f},{pred.max():.3f}]", flush=True)

    # Load label
    label = load_label(args.label_dir, args.segment_id)
    if label is None:
        np.save(str(out_dir / f"{args.segment_id}_pred.npy"), pred)
        Image.fromarray((pred * 255).astype(np.uint8)).save(
            str(out_dir / f"{args.segment_id}_pred.png"))
        print(f"[VAL] No label found — saved prediction only to {out_dir}", flush=True)
        return 1

    print(f"[VAL] Label: shape={label.shape}  ink_frac={label.mean():.4f}", flush=True)

    # Metrics
    print("\n[VAL] === METRICS ===", flush=True)
    print(f"{'Thresh':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Dice':>6}  {'PredFrac':>8}", flush=True)
    print("-" * 58, flush=True)
    rows = []
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        m = compute_metrics(pred, label, thresh)
        rows.append(m)
        print(f"{thresh:>6.1f}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  "
              f"{m['f1']:>6.4f}  {m['dice']:>6.4f}  {m['pred_frac']:>8.4f}", flush=True)

    best = max(rows, key=lambda m: m["f1"])
    print(f"\n[VAL] Best F1={best['f1']:.4f} at threshold={best['threshold']}", flush=True)
    if best["f1"] < 0.05:
        print("[VAL] VERDICT: model is NOT detecting ink on this segment", flush=True)
    elif best["f1"] < 0.20:
        print("[VAL] VERDICT: weak ink signal — possible domain gap or fiber contamination", flush=True)
    else:
        print("[VAL] VERDICT: meaningful ink detection signal", flush=True)

    # Save
    pred_u8 = (pred * 255).astype(np.uint8)
    label_r = cv2.resize(label.astype(np.float32), (pred.shape[1], pred.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    side = np.hstack([(label_r * 255).astype(np.uint8), pred_u8])
    Image.fromarray(side).save(str(out_dir / f"{args.segment_id}_label_vs_pred.png"))
    np.save(str(out_dir / f"{args.segment_id}_pred.npy"), pred)
    print(f"[VAL] Saved: {out_dir}  (left=ground_truth, right=prediction)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

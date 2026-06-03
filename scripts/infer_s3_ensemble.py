#!/usr/bin/env python
"""Ensemble inference: MiniUNETR + TimeSformer on Scroll 3."""

import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import tifffile

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.utility.configs import Config


def load_scroll3_segment(segment_dir, layer_start=0, layer_end=65):
    """Load Scroll 3 segment layers as 3D volume."""
    segment_dir = Path(segment_dir)
    layers_dir = segment_dir / "layers"
    layers = []

    for layer_idx in range(layer_start, min(layer_end, 65)):
        layer_path = layers_dir / f"{layer_idx:02d}.tif"
        if not layer_path.exists():
            print(f"[ENSEMBLE] Warning: Layer {layer_idx} not found", flush=True)
            continue

        layer = tifffile.imread(str(layer_path))
        layers.append(layer)

    if not layers:
        raise ValueError(f"No layers found in {segment_dir}")

    volume = np.stack(layers, axis=0).astype(np.float32)
    print(f"[ENSEMBLE] Loaded volume shape: {volume.shape}", flush=True)
    return volume


def apply_clahe(volume, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    for z in range(volume.shape[0]):
        layer = volume[z]
        layer_norm = ((layer - layer.min()) / (layer.max() - layer.min() + 1e-8) * 255).astype(np.uint8)
        layer_clahe = clahe.apply(layer_norm)
        volume[z] = layer_clahe.astype(np.float32) / 255.0

    return volume


def infer_miniunetr(model, volume, device, patch_size=128, stride=32, in_chans=16):
    """Run MiniUNETR inference on volume."""
    model.eval()
    z_dim, height, width = volume.shape

    if z_dim < in_chans:
        print(f"[ENSEMBLE] Padding volume from {z_dim} to {in_chans} layers", flush=True)
        padding = in_chans - z_dim
        volume = np.pad(volume, ((0, padding), (0, 0), (0, 0)), mode='reflect')

    output_shape = (height, width)
    prediction = np.zeros(output_shape, dtype=np.float32)
    count = np.zeros(output_shape, dtype=np.float32)

    print(f"[ENSEMBLE] Running MiniUNETR inference on {height}x{width} with patch_size={patch_size}, stride={stride}", flush=True)

    with torch.no_grad():
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                z_start = max(0, z_dim - in_chans)
                patch = volume[z_start:z_start + in_chans, y:y + patch_size, x:x + patch_size]

                if patch.shape != (in_chans, patch_size, patch_size):
                    continue

                patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).to(device)

                logits = model(patch_tensor)
                logits = logits.squeeze(0).squeeze(0).cpu().numpy()

                pred = 1.0 / (1.0 + np.exp(-logits))

                label_size = logits.shape[0]
                out_y = y
                out_x = x
                prediction[out_y:out_y + label_size, out_x:out_x + label_size] += pred
                count[out_y:out_y + label_size, out_x:out_x + label_size] += 1.0

    prediction = np.divide(prediction, count, where=count > 0)
    prediction[count == 0] = 0.5

    print(f"[ENSEMBLE] ✓ MiniUNETR inference complete", flush=True)
    return prediction


def infer_timesformer_simple(timesformer_model, volume, device, layer_indices=None):
    """
    Simple TimeSformer inference on selected layers.
    Returns predictions for selected layers, interpolated to full volume resolution.
    """
    print(f"[ENSEMBLE] Running TimeSformer inference on selected layers", flush=True)

    if layer_indices is None:
        # Default: sample layers 17, 24, 31, 38, 45, 52, 59
        layer_indices = list(range(17, min(65, volume.shape[0]), 7))

    z_dim, height, width = volume.shape

    # TimeSformer works on 26-layer stacks
    # Create a sparse set of layer predictions
    layer_predictions = {}

    timesformer_model.eval()

    with torch.no_grad():
        for layer_idx in layer_indices:
            if layer_idx >= z_dim:
                continue

            # Extract 26-layer window centered on this layer
            z_start = max(0, layer_idx - 13)
            z_end = min(z_dim, z_start + 26)

            if z_end - z_start < 26:
                # Pad if necessary
                z_start = max(0, z_end - 26)

            patch_3d = volume[z_start:z_end]

            # Normalize
            patch_3d = (patch_3d - patch_3d.mean()) / (patch_3d.std() + 1e-8)

            # Convert to tensor [1, 26, H, W]
            patch_tensor = torch.from_numpy(patch_3d[np.newaxis, ...]).to(device).float()

            # Run TimeSformer (simplified - actual implementation depends on model)
            try:
                with torch.no_grad():
                    logits = timesformer_model(patch_tensor)
                    pred = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    layer_predictions[layer_idx] = pred
                    print(f"[ENSEMBLE]   Layer {layer_idx}: pred shape {pred.shape}", flush=True)
            except Exception as e:
                print(f"[ENSEMBLE]   Layer {layer_idx}: skipped ({str(e)[:50]})", flush=True)
                continue

    if not layer_predictions:
        print(f"[ENSEMBLE] Warning: No TimeSformer predictions generated, using fallback", flush=True)
        return np.ones((height, width), dtype=np.float32) * 0.5

    # Interpolate sparse predictions to full volume
    layer_list = sorted(layer_predictions.keys())
    full_prediction = np.zeros((height, width), dtype=np.float32)

    # Use nearest-neighbor interpolation for simplicity
    for z in range(z_dim):
        # Find nearest layer with prediction
        nearest_idx = min(layer_list, key=lambda x: abs(x - z))
        full_prediction = layer_predictions[nearest_idx]

    print(f"[ENSEMBLE] ✓ TimeSformer inference complete", flush=True)
    return full_prediction


def ensemble_predictions(miniunetr_pred, timesformer_pred, weight_miniunetr=0.6):
    """Combine predictions from both models."""
    print(f"[ENSEMBLE] Combining predictions (weight MiniUNETR={weight_miniunetr})", flush=True)

    # Weighted average
    ensemble_pred = (weight_miniunetr * miniunetr_pred +
                     (1 - weight_miniunetr) * timesformer_pred)

    return np.clip(ensemble_pred, 0, 1)


def main(config_path, miniunetr_ckpt, timesformer_ckpt, segment_dir, output_dir, ensemble_weight=0.6):
    """Main ensemble inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENSEMBLE] Using device: {device}", flush=True)

    # Load Scroll 3 segment
    print(f"[ENSEMBLE] Loading Scroll 3 segment from {segment_dir}", flush=True)
    volume = load_scroll3_segment(segment_dir)

    # Apply CLAHE
    print(f"[ENSEMBLE] Applying CLAHE contrast enhancement...", flush=True)
    volume = apply_clahe(volume)

    # ============== MiniUNETR Inference ==============
    print(f"[ENSEMBLE] Loading MiniUNETR config and checkpoint...", flush=True)
    config = Config.load_from_file(config_path)
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    miniunetr_model = lightning_module.model
    miniunetr_model.to(device)

    state_dict = torch.load(miniunetr_ckpt, map_location=device)
    miniunetr_model.load_state_dict(state_dict)
    miniunetr_model.eval()

    miniunetr_pred = infer_miniunetr(miniunetr_model, volume, device,
                                     patch_size=config.patch_size,
                                     stride=config.stride)

    # ============== TimeSformer Inference ==============
    print(f"[ENSEMBLE] Loading TimeSformer checkpoint...", flush=True)
    try:
        timesformer_model = torch.load(timesformer_ckpt, map_location=device)
        if isinstance(timesformer_model, dict):
            # If checkpoint is a state dict, we need the model architecture
            print(f"[ENSEMBLE] Warning: TimeSformer is state dict, skipping inference", flush=True)
            timesformer_pred = miniunetr_pred.copy()
        else:
            timesformer_model.to(device)
            timesformer_model.eval()
            timesformer_pred = infer_timesformer_simple(timesformer_model, volume, device)
    except Exception as e:
        print(f"[ENSEMBLE] Error loading TimeSformer: {e}", flush=True)
        print(f"[ENSEMBLE] Using MiniUNETR predictions only", flush=True)
        timesformer_pred = miniunetr_pred.copy()

    # ============== Ensemble Combination ==============
    ensemble_pred = ensemble_predictions(miniunetr_pred, timesformer_pred,
                                        weight_miniunetr=ensemble_weight)

    # ============== Save Outputs ==============
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble as PNG
    pred_uint8 = (ensemble_pred * 255).astype(np.uint8)
    ensemble_png = output_dir / "scroll3_20240618142020_ensemble.png"
    Image.fromarray(pred_uint8).save(ensemble_png)
    print(f"[ENSEMBLE] ✓ Saved ensemble PNG: {ensemble_png}", flush=True)

    # Save ensemble as NPY
    ensemble_npy = output_dir / "scroll3_20240618142020_ensemble.npy"
    np.save(ensemble_npy, ensemble_pred)
    print(f"[ENSEMBLE] ✓ Saved ensemble NPY: {ensemble_npy}", flush=True)

    # Save individual predictions for comparison
    miniunetr_png = output_dir / "scroll3_20240618142020_miniunetr.png"
    Image.fromarray((miniunetr_pred * 255).astype(np.uint8)).save(miniunetr_png)
    print(f"[ENSEMBLE] ✓ Saved MiniUNETR PNG: {miniunetr_png}", flush=True)

    timesformer_png = output_dir / "scroll3_20240618142020_timesformer.png"
    Image.fromarray((timesformer_pred * 255).astype(np.uint8)).save(timesformer_png)
    print(f"[ENSEMBLE] ✓ Saved TimeSformer PNG: {timesformer_png}", flush=True)

    print(f"[ENSEMBLE] ✓✓✓ Ensemble inference complete!", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='MiniUNETR config file path')
    parser.add_argument('miniunetr_ckpt', type=str, help='MiniUNETR checkpoint')
    parser.add_argument('timesformer_ckpt', type=str, help='TimeSformer checkpoint')
    parser.add_argument('--segment-dir', type=str, default='~/scroll_prize/data/scroll3/fragments/20240618142020',
                        help='Scroll 3 segment directory')
    parser.add_argument('--output-dir', type=str, default='~/scroll_prize/results/scroll3_ensemble',
                        help='Output directory')
    parser.add_argument('--ensemble-weight', type=float, default=0.6,
                        help='Weight for MiniUNETR (0-1), rest goes to TimeSformer')
    args = parser.parse_args()

    args.segment_dir = os.path.expanduser(args.segment_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    sys.exit(main(args.config_path, args.miniunetr_ckpt, args.timesformer_ckpt,
                  args.segment_dir, args.output_dir, args.ensemble_weight))

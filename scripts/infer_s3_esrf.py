#!/usr/bin/env python
"""Inference on Scroll 3 with trained MiniUNETR model."""

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
            print(f"[INFER] Warning: Layer {layer_idx} not found", flush=True)
            continue

        layer = tifffile.imread(str(layer_path))
        layers.append(layer)

    if not layers:
        raise ValueError(f"No layers found in {segment_dir}")

    # Stack into 3D volume: [Z, H, W]
    volume = np.stack(layers, axis=0).astype(np.float32)
    print(f"[INFER] Loaded volume shape: {volume.shape}", flush=True)

    return volume


def apply_clahe(volume, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    for z in range(volume.shape[0]):
        # Normalize to 0-255
        layer = volume[z]
        layer_norm = ((layer - layer.min()) / (layer.max() - layer.min() + 1e-8) * 255).astype(np.uint8)

        # Apply CLAHE
        layer_clahe = clahe.apply(layer_norm)

        # Normalize back
        volume[z] = layer_clahe.astype(np.float32) / 255.0

    return volume


def infer_volume(model, volume, device, patch_size=128, stride=32, in_chans=16):
    """Run inference on volume, patching in 3D."""
    model.eval()
    z_dim, height, width = volume.shape

    # Ensure volume has enough layers for patch
    if z_dim < in_chans:
        print(f"[INFER] Volume has {z_dim} layers, padding to {in_chans}", flush=True)
        padding = in_chans - z_dim
        volume = np.pad(volume, ((0, padding), (0, 0), (0, 0)), mode='reflect')

    # Initialize output
    output_shape = (height, width)
    prediction = np.zeros(output_shape, dtype=np.float32)
    count = np.zeros(output_shape, dtype=np.float32)

    print(f"[INFER] Running inference on {height}x{width} with patch_size={patch_size}, stride={stride}", flush=True)

    with torch.no_grad():
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Extract patch: [in_chans, patch_size, patch_size]
                z_start = max(0, z_dim - in_chans)
                patch = volume[z_start:z_start + in_chans, y:y + patch_size, x:x + patch_size]

                # Ensure patch is right size
                if patch.shape != (in_chans, patch_size, patch_size):
                    continue

                # Normalize patch
                patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                # Prepare batch: [1, 1, in_chans, patch_size, patch_size]
                patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).to(device)

                # Forward pass
                logits = model(patch_tensor)  # [1, out_channels, label_size, label_size]
                logits = logits.squeeze(0).squeeze(0).cpu().numpy()  # [label_size, label_size]

                # Sigmoid for binary prediction
                pred = 1.0 / (1.0 + np.exp(-logits))

                # Average into output
                label_size = logits.shape[0]
                out_y = y
                out_x = x
                prediction[out_y:out_y + label_size, out_x:out_x + label_size] += pred
                count[out_y:out_y + label_size, out_x:out_x + label_size] += 1.0

    # Average overlapping regions
    prediction = np.divide(prediction, count, where=count > 0)
    prediction[count == 0] = 0.5  # Default to 0.5 for uncovered regions

    return prediction


def main(config_path, checkpoint_path, segment_dir, output_dir):
    """Main inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] Using device: {device}", flush=True)

    # Load config
    config = Config.load_from_file(config_path)

    # Create model
    print("[INFER] Creating model...", flush=True)
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    model = lightning_module.model
    model.to(device)

    # Load checkpoint
    print(f"[INFER] Loading checkpoint: {checkpoint_path}", flush=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Scroll 3 segment
    print(f"[INFER] Loading Scroll 3 segment from {segment_dir}", flush=True)
    volume = load_scroll3_segment(segment_dir)

    # Apply CLAHE
    print("[INFER] Applying CLAHE contrast enhancement...", flush=True)
    volume = apply_clahe(volume)

    # Run inference
    print("[INFER] Running inference...", flush=True)
    prediction = infer_volume(model, volume, device, patch_size=config.patch_size, stride=config.stride)

    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as PNG
    pred_uint8 = (prediction * 255).astype(np.uint8)
    output_path = output_dir / "scroll3_20240618142020_prediction.png"
    Image.fromarray(pred_uint8).save(output_path)
    print(f"[INFER] ✓ Saved prediction: {output_path}", flush=True)

    # Save as NPY
    npy_path = output_dir / "scroll3_20240618142020_prediction.npy"
    np.save(npy_path, prediction)
    print(f"[INFER] ✓ Saved NPY: {npy_path}", flush=True)

    print("[INFER] ✓✓✓ Inference complete!", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Config file path')
    parser.add_argument('checkpoint_path', type=str, help='Trained model checkpoint')
    parser.add_argument('--segment-dir', type=str, default='~/scroll_prize/data/scroll3/fragments/20240618142020',
                        help='Scroll 3 segment directory')
    parser.add_argument('--output-dir', type=str, default='~/scroll_prize/results/scroll3_esrf_inference',
                        help='Output directory for predictions')
    args = parser.parse_args()

    # Expand paths
    args.segment_dir = os.path.expanduser(args.segment_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    sys.exit(main(args.config_path, args.checkpoint_path, args.segment_dir, args.output_dir))

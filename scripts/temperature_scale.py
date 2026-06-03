#!/usr/bin/env python3
"""
Temperature scaling for MiniUNETR predictions.

Applies logit scaling (T < 1 → sharper, T > 1 → softer) to existing NPY
predictions to push borderline pixels away from the 0.5 decision boundary.

Usage: python temperature_scale.py --input <pred.npy> --temperatures 0.3 0.5 0.7 1.0
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(pred, T):
    """Scale logits by 1/T then re-apply sigmoid."""
    logits = logit(pred)
    return sigmoid(logits / T)


def analyze(pred, name):
    high_05 = (pred > 0.5).sum() / pred.size * 100
    high_07 = (pred > 0.7).sum() / pred.size * 100
    high_09 = (pred > 0.9).sum() / pred.size * 100
    print(f"  {name}: mean={pred.mean():.4f}, std={pred.std():.4f}, "
          f">0.5={high_05:.2f}%, >0.7={high_07:.2f}%, >0.9={high_09:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input NPY prediction file')
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[0.2, 0.3, 0.5, 0.7, 1.0],
                        help='Temperature values to try (T<1 sharpens, T>1 softens)')
    parser.add_argument('--save-all', action='store_true',
                        help='Save PNG/NPY for all temperatures')
    args = parser.parse_args()

    input_path = Path(args.input)
    pred = np.load(input_path)
    print(f"Loaded: {input_path} shape={pred.shape}")
    print()

    analyze(pred, "original (T=1.0)")

    output_dir = input_path.parent

    for T in args.temperatures:
        scaled = apply_temperature(pred, T)
        analyze(scaled, f"T={T:.1f}")

        if args.save_all or T != 1.0:
            stem = input_path.stem
            npy_out = output_dir / f"{stem}_T{T:.1f}.npy"
            png_out = output_dir / f"{stem}_T{T:.1f}.png"
            np.save(npy_out, scaled)
            Image.fromarray((scaled * 255).astype(np.uint8)).save(png_out)
            print(f"    → Saved: {npy_out}")

    print("\nDone. Lower T (e.g. 0.3) → sharper, more confident predictions.")
    print("Higher T → softer, more uncertain predictions.")


if __name__ == "__main__":
    main()

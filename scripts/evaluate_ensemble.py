#!/usr/bin/env python
"""Evaluate ensemble predictions vs. individual models."""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def analyze_prediction(pred_array, name="Predictions"):
    """Analyze prediction statistics."""
    stats = {
        'name': name,
        'shape': pred_array.shape,
        'dtype': pred_array.dtype,
        'min': float(pred_array.min()),
        'max': float(pred_array.max()),
        'mean': float(pred_array.mean()),
        'median': float(np.median(pred_array)),
        'std': float(pred_array.std()),
        'p5': float(np.percentile(pred_array, 5)),
        'p25': float(np.percentile(pred_array, 25)),
        'p75': float(np.percentile(pred_array, 75)),
        'p95': float(np.percentile(pred_array, 95)),
    }

    # Ink fraction (p > 0.5)
    ink_pixels = (pred_array > 0.5).sum()
    total_pixels = pred_array.size
    stats['ink_fraction'] = float(ink_pixels / total_pixels * 100)
    stats['ink_pixels'] = int(ink_pixels)

    # Confidence distribution
    low_conf = (pred_array < 0.3).sum()
    med_conf = ((pred_array >= 0.3) & (pred_array <= 0.7)).sum()
    high_conf = (pred_array > 0.7).sum()

    stats['low_conf_pct'] = float(low_conf / total_pixels * 100)
    stats['med_conf_pct'] = float(med_conf / total_pixels * 100)
    stats['high_conf_pct'] = float(high_conf / total_pixels * 100)

    return stats


def compare_predictions(pred1, pred2, name1="Model 1", name2="Model 2"):
    """Compare two prediction arrays."""
    # Element-wise differences
    diff = np.abs(pred1 - pred2)

    comparison = {
        'comparison': f"{name1} vs {name2}",
        'mean_diff': float(diff.mean()),
        'max_diff': float(diff.max()),
        'std_diff': float(diff.std()),
        'agree_within_0.1': float(((diff < 0.1).sum() / diff.size * 100)),
        'agree_within_0.2': float(((diff < 0.2).sum() / diff.size * 100)),
        'disagree_over_0.3': float(((diff >= 0.3).sum() / diff.size * 100)),
    }

    return comparison


def main(result_dir):
    """Evaluate ensemble results."""
    result_dir = Path(result_dir)

    print("=" * 80)
    print("ENSEMBLE PREDICTION EVALUATION")
    print("=" * 80)
    print(f"Results directory: {result_dir}\n")

    # Load prediction files
    files = {
        'ensemble': result_dir / 'scroll3_20240618142020_ensemble.npy',
        'miniunetr': result_dir / 'scroll3_20240618142020_miniunetr.npy',
        'timesformer': result_dir / 'scroll3_20240618142020_timesformer.npy',
    }

    predictions = {}
    for key, path in files.items():
        if path.exists():
            predictions[key] = np.load(path)
            print(f"✓ Loaded {key}: {path}")
        else:
            print(f"⚠ Missing {key}: {path}")

    if not predictions:
        print("No predictions found!")
        return 1

    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL STATISTICS")
    print("=" * 80 + "\n")

    # Analyze each prediction
    all_stats = {}
    for name, pred in predictions.items():
        stats = analyze_prediction(pred, name)
        all_stats[name] = stats

        print(f"{name.upper()}")
        print("-" * 40)
        print(f"  Shape: {stats['shape']}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Median: {stats['median']:.6f}")
        print(f"  Percentiles: 5%={stats['p5']:.4f}, 25%={stats['p25']:.4f}, 75%={stats['p75']:.4f}, 95%={stats['p95']:.4f}")
        print(f"  Ink fraction (p>0.5): {stats['ink_fraction']:.2f}%")
        print(f"  Confidence distribution:")
        print(f"    Low (<0.3): {stats['low_conf_pct']:.1f}%")
        print(f"    Medium (0.3-0.7): {stats['med_conf_pct']:.1f}%")
        print(f"    High (>0.7): {stats['high_conf_pct']:.1f}%")
        print()

    # Compare predictions
    if 'ensemble' in predictions and 'miniunetr' in predictions:
        print("=" * 80)
        print("ENSEMBLE vs. INDIVIDUAL MODELS")
        print("=" * 80 + "\n")

        if 'miniunetr' in predictions:
            comp = compare_predictions(predictions['ensemble'], predictions['miniunetr'],
                                      "Ensemble", "MiniUNETR")
            print(f"Ensemble vs. MiniUNETR:")
            print(f"  Mean difference: {comp['mean_diff']:.6f}")
            print(f"  Max difference: {comp['max_diff']:.6f}")
            print(f"  Std difference: {comp['std_diff']:.6f}")
            print(f"  Agreement within 0.1: {comp['agree_within_0.1']:.1f}%")
            print(f"  Agreement within 0.2: {comp['agree_within_0.2']:.1f}%")
            print(f"  Disagreement >0.3: {comp['disagree_over_0.3']:.1f}%")
            print()

        if 'timesformer' in predictions:
            comp = compare_predictions(predictions['ensemble'], predictions['timesformer'],
                                      "Ensemble", "TimeSformer")
            print(f"Ensemble vs. TimeSformer:")
            print(f"  Mean difference: {comp['mean_diff']:.6f}")
            print(f"  Max difference: {comp['max_diff']:.6f}")
            print(f"  Std difference: {comp['std_diff']:.6f}")
            print(f"  Agreement within 0.1: {comp['agree_within_0.1']:.1f}%")
            print(f"  Agreement within 0.2: {comp['agree_within_0.2']:.1f}%")
            print(f"  Disagreement >0.3: {comp['disagree_over_0.3']:.1f}%")
            print()

        # Compare MiniUNETR vs TimeSformer
        if 'miniunetr' in predictions and 'timesformer' in predictions:
            comp = compare_predictions(predictions['miniunetr'], predictions['timesformer'],
                                      "MiniUNETR", "TimeSformer")
            print(f"MiniUNETR vs. TimeSformer:")
            print(f"  Mean difference: {comp['mean_diff']:.6f}")
            print(f"  Max difference: {comp['max_diff']:.6f}")
            print(f"  Std difference: {comp['std_diff']:.6f}")
            print(f"  Agreement within 0.1: {comp['agree_within_0.1']:.1f}%")
            print(f"  Agreement within 0.2: {comp['agree_within_0.2']:.1f}%")
            print(f"  Disagreement >0.3: {comp['disagree_over_0.3']:.1f}%")
            print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Model':<15} {'Mean':<12} {'Std':<12} {'Ink%':<10} {'Low%':<10} {'Med%':<10} {'High%':<10}")
    print("-" * 80)

    for name, stats in all_stats.items():
        print(f"{name:<15} {stats['mean']:>10.4f}  {stats['std']:>10.4f}  "
              f"{stats['ink_fraction']:>8.2f}  {stats['low_conf_pct']:>8.1f}  "
              f"{stats['med_conf_pct']:>8.1f}  {stats['high_conf_pct']:>8.1f}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if 'ensemble' in all_stats and 'miniunetr' in all_stats:
        ens_ink = all_stats['ensemble']['ink_fraction']
        mini_ink = all_stats['miniunetr']['ink_fraction']
        print(f"\n✓ Ensemble ink fraction: {ens_ink:.2f}%")
        print(f"✓ MiniUNETR ink fraction: {mini_ink:.2f}%")
        print(f"  Change: {ens_ink - mini_ink:+.2f}%")

    if 'ensemble' in all_stats:
        ens_high_conf = all_stats['ensemble']['high_conf_pct']
        print(f"\n✓ Ensemble high confidence (>0.7): {ens_high_conf:.1f}%")
        if ens_high_conf > 20:
            print("  → Good! Model is more confident in predictions")
        else:
            print("  → Consider retraining with transfer learning to increase confidence")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
If confidence is still low:
1. Implement Phase 2: Transfer learning from TimeSformer weights
2. Fine-tune MiniUNETR on ESRF data with TimeSformer initialization
3. Expected: 20-40% improvement in prediction confidence

If ensemble results are good:
1. Use ensemble predictions as baseline
2. Proceed with transfer learning for further improvements
3. Document both models' contributions to final predictions
""")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str, help='Directory containing prediction files')
    args = parser.parse_args()

    sys.exit(main(args.result_dir))

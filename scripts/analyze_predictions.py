#!/usr/bin/env python3
"""
Analyze and compare predictions from different phases.
Usage: python scripts/analyze_predictions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Configuration
RESULTS_DIR = Path.home() / "scroll_prize" / "vesuvius_first_title_prize" / "results"
OUTPUT_DIR = Path.home() / "scroll_prize" / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_prediction(result_dir, name="scroll3_20240618142020_prediction"):
    """Load NPY prediction from result directory."""
    npy_file = Path(result_dir) / f"{name}.npy"
    if not npy_file.exists():
        print(f"❌ Not found: {npy_file}")
        return None

    print(f"✅ Loading: {npy_file}")
    return np.load(npy_file)

def compute_metrics(pred, phase_name):
    """Compute key metrics for prediction."""
    if pred is None:
        return None

    metrics = {
        'phase': phase_name,
        'shape': pred.shape,
        'mean': float(pred.mean()),
        'std': float(pred.std()),
        'min': float(pred.min()),
        'max': float(pred.max()),
        'median': float(np.median(pred)),
        'high_conf_05_pct': float((pred > 0.5).sum() / pred.size * 100),
        'high_conf_08_pct': float((pred > 0.8).sum() / pred.size * 100),
        'high_conf_09_pct': float((pred > 0.9).sum() / pred.size * 100),
        'ink_fraction': float((pred > 0.5).sum() / pred.size * 100),
    }

    # Confidence percentiles
    metrics['p25'] = float(np.percentile(pred, 25))
    metrics['p50'] = float(np.percentile(pred, 50))
    metrics['p75'] = float(np.percentile(pred, 75))
    metrics['p90'] = float(np.percentile(pred, 90))
    metrics['p95'] = float(np.percentile(pred, 95))

    return metrics

def print_metrics(metrics):
    """Pretty print metrics."""
    if metrics is None:
        print("  (No data)")
        return

    phase = metrics['phase']
    print(f"\n{'='*60}")
    print(f"PHASE: {phase}")
    print(f"{'='*60}")
    print(f"Shape:             {metrics['shape']}")
    print(f"Mean confidence:   {metrics['mean']:.4f}")
    print(f"Std dev:           {metrics['std']:.4f}")
    print(f"Range:             [{metrics['min']:.4f}, {metrics['max']:.4f}]")
    print(f"Median:            {metrics['median']:.4f}")
    print(f"\nPercentiles:")
    print(f"  p25: {metrics['p25']:.4f}")
    print(f"  p50: {metrics['p50']:.4f}")
    print(f"  p75: {metrics['p75']:.4f}")
    print(f"  p90: {metrics['p90']:.4f}")
    print(f"  p95: {metrics['p95']:.4f}")
    print(f"\nHigh Confidence Pixels:")
    print(f"  > 0.5: {metrics['high_conf_05_pct']:.2f}%")
    print(f"  > 0.8: {metrics['high_conf_08_pct']:.2f}%")
    print(f"  > 0.9: {metrics['high_conf_09_pct']:.2f}%")
    print(f"\nInk Fraction (>0.5): {metrics['ink_fraction']:.2f}%")

def find_latest_result(pattern):
    """Find latest result directory matching pattern."""
    results = sorted([d for d in RESULTS_DIR.glob(pattern)])
    if results:
        return results[-1]
    return None

def plot_comparison(metrics_list, output_file):
    """Create comparison plots."""
    if not metrics_list or all(m is None for m in metrics_list):
        print("No data to plot")
        return

    valid_metrics = [m for m in metrics_list if m is not None]
    phases = [m['phase'] for m in valid_metrics]
    means = [m['mean'] for m in valid_metrics]
    high_conf_08 = [m['high_conf_08_pct'] for m in valid_metrics]
    ink_fractions = [m['ink_fraction'] for m in valid_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean confidence
    axes[0].bar(phases, means, color=['blue', 'orange', 'green'])
    axes[0].set_title('Mean Confidence by Phase')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].set_ylim([0.4, 0.7])
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # High confidence pixels (>0.8)
    axes[1].bar(phases, high_conf_08, color=['blue', 'orange', 'green'])
    axes[1].set_title('High Confidence Pixels (>0.8)')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_ylim([0, max(high_conf_08) * 1.2])
    for i, v in enumerate(high_conf_08):
        axes[1].text(i, v + 0.1, f'{v:.2f}%', ha='center')

    # Ink fraction
    axes[2].bar(phases, ink_fractions, color=['blue', 'orange', 'green'])
    axes[2].set_title('Ink Fraction (>0.5)')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_ylim([0, 20])
    for i, v in enumerate(ink_fractions):
        axes[2].text(i, v + 0.3, f'{v:.2f}%', ha='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")
    plt.close()

def main():
    print("="*60)
    print("PREDICTION ANALYSIS: Comparing All Phases")
    print("="*60)

    # Find result directories
    print("\n🔍 Searching for result directories...")

    baseline_dir = find_latest_result("scroll3_*")  # Find first baseline
    phase2_dir = find_latest_result("scroll3_final_*")  # Find transfer final

    # Also search by timestamp pattern if available
    all_results = sorted(RESULTS_DIR.glob("scroll3_*"))
    print(f"Found {len(all_results)} result directories")
    for r in all_results[-5:]:
        print(f"  {r.name}")

    # Load predictions
    print("\n📊 Loading predictions...")

    predictions = {}

    # Try to load Phase 2 (Final Inference with Transfer Learning)
    if phase2_dir:
        phase2_pred = load_prediction(phase2_dir, "scroll3_20240618142020_prediction")
        if phase2_pred is not None:
            predictions['Phase 2 (Transfer)'] = phase2_pred

    # Try to load baseline (Phase 1 inference results if available)
    # Note: Baseline would be from initial MiniUNETR training
    baseline_results = sorted([d for d in RESULTS_DIR.glob("scroll3_20240618*") if "final" not in d.name])
    if baseline_results:
        baseline_pred = load_prediction(baseline_results[0], "scroll3_20240618142020_prediction")
        if baseline_pred is not None:
            predictions['Phase 1 (Baseline)'] = baseline_pred

    # Compute metrics
    print("\n📈 Computing metrics...")
    metrics_list = []
    for phase, pred in predictions.items():
        metrics = compute_metrics(pred, phase)
        metrics_list.append(metrics)
        print_metrics(metrics)

    # Save analysis
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'predictions_analyzed': list(predictions.keys()),
        'metrics': metrics_list
    }

    analysis_file = OUTPUT_DIR / "phase_comparison.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ Analysis saved: {analysis_file}")

    # Create comparison plots
    if len(metrics_list) >= 2:
        plot_file = OUTPUT_DIR / "phase_comparison.png"
        plot_comparison(metrics_list, plot_file)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    if len(metrics_list) >= 2:
        for i, m in enumerate(metrics_list):
            print(f"\n{m['phase']}:")
            print(f"  Mean confidence: {m['mean']:.4f}")
            print(f"  High conf (>0.8): {m['high_conf_08_pct']:.2f}%")
            print(f"  Ink fraction: {m['ink_fraction']:.2f}%")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()

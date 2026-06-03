#!/usr/bin/env python3
"""
Smart Ensemble Weighting for MiniUNETR — Phase 4 Option B.

Learns optimal weights across 3 trained models using the validation set,
then generates a weighted ensemble prediction on Scroll 3.

Models:
  baseline  — ft_esrf_manual   — val_loss 0.6041 (best single model)
  transfer  — ft_esrf_transfer — val_loss 0.6122
  augmented — ft_esrf_augmented — val_loss 0.6126
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
import json
from datetime import datetime
import cv2
import tifffile
from PIL import Image

PROJ_DIR = Path.home() / "scroll_prize" / "vesuvius_first_title_prize"
DATA_DIR = Path.home() / "scroll_prize" / "data"
RESULTS_DIR = PROJ_DIR / "results"

sys.path.insert(0, str(PROJ_DIR / "src"))

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.utility.configs import Config
from phoenix.model.datamodule import UNETR_SF_DataModule


CHECKPOINTS = {
    "baseline": PROJ_DIR / "checkpoints/ft_esrf_manual_20260531_003040/best_epoch_015_val_loss_0.6041.pt",
    "transfer": PROJ_DIR / "checkpoints/ft_esrf_transfer_20260531_031817/best_epoch_019_val_loss_0.6122.pt",
    "augmented": PROJ_DIR / "checkpoints/ft_esrf_augmented_20260531_042222/best_epoch_012_val_loss_0.6126.pt",
}

CONFIG_PATH = PROJ_DIR / "configs/ft_esrf.py"
SEGMENT_DIR = DATA_DIR / "scroll3/fragments/20240618142020"


def load_model(config, checkpoint_path, device):
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    model = lightning_module.model
    model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_val_inference(model, val_loader, device):
    """Returns (all_preds, all_labels) as flattened numpy arrays."""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            # Labels may be 2-channel (background, foreground) — take foreground channel
            labels_np = y.cpu().numpy()
            if labels_np.ndim == 4 and labels_np.shape[1] == 2:
                labels_np = labels_np[:, 1, :, :]  # foreground/ink channel
            labels = labels_np.flatten()
            all_preds.append(preds)
            all_labels.append(labels)
    return np.concatenate(all_preds), np.concatenate(all_labels)


def bce_loss(preds, labels, eps=1e-7):
    p = np.clip(preds, eps, 1 - eps)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def ensemble_bce(weights_raw, preds_list, labels):
    """BCE of softmax-weighted ensemble (unconstrained optimization)."""
    w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))  # softmax → sums to 1
    combined = sum(w[i] * preds_list[i] for i in range(len(preds_list)))
    return bce_loss(combined, labels)


def optimize_weights(preds_list, labels):
    """Find optimal softmax weights minimizing ensemble BCE on val set."""
    print(f"\n[ENSEMBLE] Optimizing weights on {len(labels)} validation predictions...")

    # Baseline: equal weights
    equal_w = np.ones(len(preds_list)) / len(preds_list)
    equal_pred = sum(equal_w[i] * preds_list[i] for i in range(len(preds_list)))
    equal_loss = bce_loss(equal_pred, labels)
    print(f"[ENSEMBLE] Equal weights (1/3,1/3,1/3) loss: {equal_loss:.6f}")

    # Individual losses
    for i, (name, pred) in enumerate(zip(CHECKPOINTS.keys(), preds_list)):
        loss = bce_loss(pred, labels)
        print(f"[ENSEMBLE] {name} individual loss: {loss:.6f}")

    # Grid search over weights (course-grained first)
    best_loss = float('inf')
    best_weights = equal_w.copy()

    steps = 11  # 0.0, 0.1, ..., 1.0
    w_range = np.linspace(0, 1, steps)

    print(f"[ENSEMBLE] Grid search ({steps}^2 = {steps**2} combinations)...")
    for w1 in w_range:
        for w2 in w_range:
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            combined = w1 * preds_list[0] + w2 * preds_list[1] + w3 * preds_list[2]
            loss = bce_loss(combined, labels)
            if loss < best_loss:
                best_loss = loss
                best_weights = np.array([w1, w2, w3])

    print(f"[ENSEMBLE] Grid search best: w={best_weights} → loss={best_loss:.6f}")

    # Fine-tune with scipy around the grid search result
    def objective(w_raw):
        return ensemble_bce(w_raw, preds_list, labels)

    # Start from best grid weights (convert to log space for unconstrained opt)
    w0_safe = np.clip(best_weights, 1e-6, 1.0)
    w0_log = np.log(w0_safe)

    result = minimize(objective, w0_log, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})

    opt_w = np.exp(result.x) / np.sum(np.exp(result.x))
    opt_loss = bce_loss(
        sum(opt_w[i] * preds_list[i] for i in range(len(preds_list))),
        labels
    )
    print(f"[ENSEMBLE] Scipy refined: w={opt_w} → loss={opt_loss:.6f}")

    if opt_loss < best_loss:
        best_weights = opt_w
        best_loss = opt_loss

    print(f"\n[ENSEMBLE] FINAL WEIGHTS: {dict(zip(CHECKPOINTS.keys(), best_weights))}")
    print(f"[ENSEMBLE] FINAL VAL LOSS: {best_loss:.6f} (vs best single=0.6041)")

    return best_weights, best_loss


def apply_clahe(volume, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    for z in range(volume.shape[0]):
        layer = volume[z]
        layer_norm = ((layer - layer.min()) / (layer.max() - layer.min() + 1e-8) * 255).astype(np.uint8)
        volume[z] = clahe.apply(layer_norm).astype(np.float32) / 255.0
    return volume


def load_scroll3(segment_dir, layer_end=65):
    layers_dir = Path(segment_dir) / "layers"
    layers = []
    for i in range(layer_end):
        p = layers_dir / f"{i:02d}.tif"
        if p.exists():
            layers.append(tifffile.imread(str(p)))
    volume = np.stack(layers, axis=0).astype(np.float32)
    print(f"[INFER] Loaded volume: {volume.shape}", flush=True)
    return volume


def infer_volume(model, volume, device, patch_size=128, stride=128, in_chans=16):
    """Patch-based inference returning full 2D prediction map."""
    model.eval()
    z_dim, H, W = volume.shape
    prediction = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    print(f"[INFER] Running patch inference {H}x{W}, patch={patch_size}, stride={stride}", flush=True)

    with torch.no_grad():
        total_patches = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
        done = 0
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                z_start = max(0, z_dim - in_chans)
                patch = volume[z_start:z_start + in_chans, y:y + patch_size, x:x + patch_size]
                if patch.shape != (in_chans, patch_size, patch_size):
                    continue
                patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                pt = torch.from_numpy(patch[np.newaxis, np.newaxis]).to(device)
                logits = model(pt).squeeze(0).squeeze(0).cpu().numpy()
                pred = 1.0 / (1.0 + np.exp(-logits))
                label_size = logits.shape[0]
                prediction[y:y + label_size, x:x + label_size] += pred
                count[y:y + label_size, x:x + label_size] += 1.0
                done += 1
                if done % 500 == 0:
                    print(f"[INFER]   {done}/{total_patches} patches done", flush=True)

    prediction = np.divide(prediction, count, where=count > 0)
    prediction[count == 0] = 0.5
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-scroll3', action='store_true',
                        help='Skip Scroll 3 inference (val weight optimization only)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENSEMBLE] Device: {device}", flush=True)

    # Load config
    config = Config.load_from_file(str(CONFIG_PATH))
    print(f"[ENSEMBLE] Config loaded. Seed={config.seed}, val_frac={config.val_frac}", flush=True)

    # Create DataModule to get the same validation split
    print("[ENSEMBLE] Creating DataModule for validation split...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    val_loader = dm.val_dataloader()
    print(f"[ENSEMBLE] Validation samples: {len(dm.v_img_paths)}", flush=True)

    # Load all 3 models
    models = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n[ENSEMBLE] Loading model: {name} from {ckpt_path}", flush=True)
        models[name] = load_model(config, ckpt_path, device)

    # Run validation inference for each model
    print("\n[ENSEMBLE] Running validation inference for all 3 models...", flush=True)
    val_preds = {}
    val_labels = None
    for name, model in models.items():
        print(f"[ENSEMBLE] Val inference: {name}", flush=True)
        preds, labels = run_val_inference(model, val_loader, device)
        val_preds[name] = preds
        if val_labels is None:
            val_labels = labels
        print(f"[ENSEMBLE]   {name}: {len(preds)} predictions, mean={preds.mean():.4f}", flush=True)

    # Optimize weights
    preds_list = [val_preds[k] for k in CHECKPOINTS.keys()]
    best_weights, best_val_loss = optimize_weights(preds_list, val_labels)

    # Save weight optimization results
    results_dir = RESULTS_DIR / f"ensemble_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    weight_results = {
        "timestamp": datetime.now().isoformat(),
        "weights": dict(zip(CHECKPOINTS.keys(), best_weights.tolist())),
        "val_loss_ensemble": float(best_val_loss),
        "val_loss_baseline": 0.6041,
        "val_loss_transfer": 0.6122,
        "val_loss_augmented": 0.6126,
        "improvement_vs_baseline": float(0.6041 - best_val_loss),
    }

    with open(results_dir / "weights.json", "w") as f:
        json.dump(weight_results, f, indent=2)
    print(f"\n[ENSEMBLE] Weights saved: {results_dir}/weights.json", flush=True)

    if args.skip_scroll3:
        print("[ENSEMBLE] Skipping Scroll 3 inference (--skip-scroll3 flag set)", flush=True)
        return

    # Run Scroll 3 inference for all 3 models
    print(f"\n[ENSEMBLE] Loading Scroll 3 segment from {SEGMENT_DIR}", flush=True)
    volume = load_scroll3(SEGMENT_DIR)
    volume = apply_clahe(volume)

    scroll3_preds = []
    for i, (name, model) in enumerate(models.items()):
        print(f"\n[ENSEMBLE] Scroll 3 inference: {name} (model {i+1}/3)", flush=True)
        pred = infer_volume(model, volume, device,
                            patch_size=config.patch_size,
                            stride=config.stride,
                            in_chans=config.in_chans)
        scroll3_preds.append(pred)
        # Free GPU memory between models
        model.cpu()
        torch.cuda.empty_cache()
        print(f"[ENSEMBLE]   {name} prediction: mean={pred.mean():.4f}, "
              f"high_conf={((pred > 0.5).sum() / pred.size * 100):.2f}%", flush=True)

    # Combine with optimal weights
    print(f"\n[ENSEMBLE] Combining with weights: {best_weights}", flush=True)
    ensemble_pred = sum(best_weights[i] * scroll3_preds[i] for i in range(3))
    print(f"[ENSEMBLE] Ensemble prediction: mean={ensemble_pred.mean():.4f}, "
          f"high_conf={((ensemble_pred > 0.5).sum() / ensemble_pred.size * 100):.2f}%", flush=True)

    # Also compute equal-weight ensemble for comparison
    equal_pred = sum(scroll3_preds[i] / 3 for i in range(3))
    print(f"[ENSEMBLE] Equal-weight ensemble: mean={equal_pred.mean():.4f}, "
          f"high_conf={((equal_pred > 0.5).sum() / equal_pred.size * 100):.2f}%", flush=True)

    # Save predictions
    for pred_name, pred in [("ensemble_optimal", ensemble_pred), ("ensemble_equal", equal_pred)]:
        pred_uint8 = (pred * 255).astype(np.uint8)
        png_path = results_dir / f"scroll3_20240618142020_{pred_name}.png"
        npy_path = results_dir / f"scroll3_20240618142020_{pred_name}.npy"
        Image.fromarray(pred_uint8).save(png_path)
        np.save(npy_path, pred)
        print(f"[ENSEMBLE] Saved: {png_path}", flush=True)

    weight_results["scroll3_ensemble_mean"] = float(ensemble_pred.mean())
    weight_results["scroll3_ensemble_high_conf_pct"] = float(
        (ensemble_pred > 0.5).sum() / ensemble_pred.size * 100
    )
    with open(results_dir / "weights.json", "w") as f:
        json.dump(weight_results, f, indent=2)

    print(f"\n[ENSEMBLE] ✓✓✓ Complete! Results in {results_dir}", flush=True)
    print(f"[ENSEMBLE] Val loss improvement: {0.6041:.4f} → {best_val_loss:.4f} "
          f"({'better' if best_val_loss < 0.6041 else 'worse'})", flush=True)


if __name__ == "__main__":
    main()

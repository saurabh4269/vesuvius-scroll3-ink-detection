#!/usr/bin/env python
"""Transfer learning: Fine-tune MiniUNETR with TimeSformer-initialized weights."""

import sys
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.model.datamodule import UNETR_SF_DataModule
from phoenix.utility.configs import Config


def validate(model, val_loader, device, num_batches=None):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    num_batches_processed = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Labels are 2-channel (ink_mask, validity_mask).
            # Use only channel 0 (ink labels). Model outputs (batch, 32, 32).
            y_ink = y[:, 0, :, :].float()
            loss = F.binary_cross_entropy_with_logits(logits, y_ink)
            total_loss += loss.item()
            num_batches_processed += 1

            if num_batches is not None and num_batches_processed >= num_batches:
                break

    avg_loss = total_loss / max(num_batches_processed, 1)
    return avg_loss


def train_epoch(model, train_loader, optimizer, device, epoch, max_batches=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        # Labels are 2-channel (ink_mask, validity_mask).
        # Use only channel 0 (ink labels). Model outputs (batch, 32, 32).
        y_ink = y[:, 0, :, :].float()
        loss = F.binary_cross_entropy_with_logits(logits, y_ink)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}: loss={loss.item():.6f}", flush=True)

        if max_batches is not None and num_batches >= max_batches:
            break

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, num_batches


def load_timesformer_weights(timesformer_ckpt):
    """Load TimeSformer checkpoint and extract encoder weights."""
    print("[TRANSFER] Loading TimeSformer checkpoint...", flush=True)

    try:
        # Try loading as state dict first
        if timesformer_ckpt.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            ckpt = torch.load(timesformer_ckpt, map_location='cpu')

            # Extract state_dict from PyTorch Lightning format
            if 'state_dict' in ckpt:
                ts_state = ckpt['state_dict']
                print(f"[TRANSFER] ✓ Loaded PyTorch Lightning checkpoint", flush=True)
            else:
                ts_state = ckpt
                print(f"[TRANSFER] ✓ Loaded raw state dict", flush=True)
        else:
            ts_state = torch.load(timesformer_ckpt, map_location='cpu')
            print(f"[TRANSFER] ✓ Loaded checkpoint", flush=True)

        return ts_state
    except Exception as e:
        print(f"[TRANSFER] Error loading TimeSformer weights: {e}", flush=True)
        return None


def initialize_with_transfer_weights(model, ts_state):
    """Initialize model with compatible TimeSformer weights."""
    if ts_state is None:
        print("[TRANSFER] No TimeSformer weights to transfer", flush=True)
        return

    print("[TRANSFER] Attempting to transfer weights...", flush=True)

    model_state = model.state_dict()
    loaded_count = 0
    skipped_count = 0

    # Try to match weights between TimeSformer and MiniUNETR
    for name, param in ts_state.items():
        # Clean up naming (remove 'model.' prefix if present)
        clean_name = name.replace('model.', '')

        # Look for compatible layers
        # TimeSformer encoder → MiniUNETR encoder
        if 'encoder' in clean_name and clean_name in model_state:
            try:
                # Attempt to load weight
                if model_state[clean_name].shape == param.shape:
                    model_state[clean_name] = param
                    loaded_count += 1
                else:
                    print(f"[TRANSFER] Shape mismatch for {clean_name}: {param.shape} vs {model_state[clean_name].shape}", flush=True)
                    skipped_count += 1
            except Exception as e:
                print(f"[TRANSFER] Error transferring {clean_name}: {e}", flush=True)
                skipped_count += 1

        # Also try segment_former layers (Segformer components)
        elif 'segment' in clean_name or 'segformer' in clean_name.lower():
            if clean_name in model_state:
                try:
                    if model_state[clean_name].shape == param.shape:
                        model_state[clean_name] = param
                        loaded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    skipped_count += 1

    model.load_state_dict(model_state)
    print(f"[TRANSFER] ✓ Transferred {loaded_count} weight layers ({skipped_count} skipped)", flush=True)


def main(config_path, epochs=20, checkpoint_dir=None, timesformer_ckpt=None, transfer_lr=5e-5, warmup_epochs=1):
    """Main transfer learning training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRANSFER] Using device: {device}", flush=True)

    # Load config
    print("[TRANSFER] Loading config...", flush=True)
    config = Config.load_from_file(config_path)

    # Create checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd() / "checkpoints" / "ft_esrf_transfer"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRANSFER] Checkpoints will be saved to: {checkpoint_dir}", flush=True)

    # Create model
    print("[TRANSFER] Creating model...", flush=True)
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    model = lightning_module.model
    model.to(device)
    print(f"[TRANSFER] Model created: {type(model).__name__}", flush=True)

    # Initialize with transfer learning weights if provided
    if timesformer_ckpt and os.path.exists(timesformer_ckpt):
        ts_state = load_timesformer_weights(timesformer_ckpt)
        initialize_with_transfer_weights(model, ts_state)
    else:
        print(f"[TRANSFER] TimeSformer checkpoint not found: {timesformer_ckpt}", flush=True)

    # Create datamodule
    print("[TRANSFER] Creating DataModule...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"[TRANSFER] DataModule ready with {len(dm.t_img_paths)} training and {len(dm.v_img_paths)} validation samples", flush=True)

    # Create optimizer with lower learning rate for transfer learning
    print(f"[TRANSFER] Using transfer learning rate: {transfer_lr:.6f}", flush=True)
    optimizer = Adam(model.parameters(), lr=transfer_lr)

    # Scheduler with warmup for stable transfer learning
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs * len(train_loader))
        print(f"[TRANSFER] Warmup for {warmup_epochs} epochs", flush=True)
    else:
        warmup_scheduler = None

    print("[TRANSFER] Optimizer and scheduler created", flush=True)

    # Training loop
    print(f"[TRANSFER] Starting transfer learning for {epochs} epochs...", flush=True)
    print("=" * 80, flush=True)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"\n[TRANSFER] ===== Epoch {epoch}/{epochs} =====", flush=True)

        # Train
        train_loss, train_batches = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"[TRANSFER] Epoch {epoch} - Train Loss: {train_loss:.6f} ({train_batches} batches)", flush=True)

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"[TRANSFER] Epoch {epoch} - Val Loss: {val_loss:.6f}", flush=True)

        # Step scheduler
        if warmup_scheduler and epoch <= warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[TRANSFER] Epoch {epoch} - Warmup Learning rate: {current_lr:.6e}", flush=True)
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[TRANSFER] Epoch {epoch} - Learning rate: {current_lr:.6e}", flush=True)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / f"best_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[TRANSFER] ✓ Saved best checkpoint: {ckpt_path.name}", flush=True)

        # Always save latest
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(model.state_dict(), latest_path)

    print("\n" + "=" * 80, flush=True)
    print(f"[TRANSFER] ✓✓✓ Transfer learning complete! Best val loss: {best_val_loss:.6f}", flush=True)
    print(f"[TRANSFER] Checkpoints saved to: {checkpoint_dir}", flush=True)
    print(f"\n[TRANSFER] COMPARISON:")
    print(f"  Baseline (no transfer) best loss: 0.6041")
    print(f"  Transfer learning best loss: {best_val_loss:.4f}")
    print(f"  Improvement: {(0.6041 - best_val_loss) / 0.6041 * 100:+.1f}%")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Config file path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--timesformer-ckpt', type=str, default=None, help='TimeSformer checkpoint for transfer learning')
    parser.add_argument('--transfer-lr', type=float, default=5e-5, help='Transfer learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Warmup epochs')
    args = parser.parse_args()

    sys.exit(main(args.config_path, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
                  timesformer_ckpt=args.timesformer_ckpt, transfer_lr=args.transfer_lr,
                  warmup_epochs=args.warmup_epochs))

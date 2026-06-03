#!/usr/bin/env python
"""Full training loop without PyTorch Lightning."""

import sys
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.model.datamodule import UNETR_SF_DataModule
from phoenix.utility.configs import Config


def validate(model, val_loader, device, num_batches=None, pos_weight=None):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    num_batches_processed = 0
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Labels are 2-channel (ink_mask, validity_mask).
            # Use only channel 0 (ink labels). Model outputs (batch, 32, 32) —
            # no channel dim — so select without keeping the dim.
            y_ink = y[:, 0, :, :].float()
            loss = F.binary_cross_entropy_with_logits(logits, y_ink, pos_weight=pw)
            total_loss += loss.item()
            num_batches_processed += 1

            if num_batches is not None and num_batches_processed >= num_batches:
                break

    avg_loss = total_loss / max(num_batches_processed, 1)
    return avg_loss


def train_epoch(model, train_loader, optimizer, device, epoch, max_batches=None, pos_weight=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        # Labels are 2-channel (ink_mask, validity_mask).
        # Use only channel 0 (ink labels). Model outputs (batch, 32, 32).
        y_ink = y[:, 0, :, :].float()
        loss = F.binary_cross_entropy_with_logits(logits, y_ink, pos_weight=pw)

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


def main(config_path, epochs=5, checkpoint_dir=None, pos_weight=None, data_dir=None):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}", flush=True)
    if pos_weight:
        print(f"[TRAIN] pos_weight={pos_weight:.1f} (amplifying ink pixel gradients)", flush=True)

    # Load config
    print("[TRAIN] Loading config...", flush=True)
    config = Config.load_from_file(config_path)

    # Override dataset directory if provided
    if data_dir:
        config.dataset_target_dir = data_dir
        print(f"[TRAIN] Dataset overridden: {data_dir}", flush=True)

    # Create checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd() / "checkpoints" / "ft_esrf_manual"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved to: {checkpoint_dir}", flush=True)

    # Create model
    print("[TRAIN] Creating model...", flush=True)
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    model = lightning_module.model
    model.to(device)
    print(f"[TRAIN] Model created: {type(model).__name__}", flush=True)

    # Create datamodule
    print("[TRAIN] Creating DataModule...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"[TRAIN] DataModule ready with {len(dm.t_img_paths)} training and {len(dm.v_img_paths)} validation samples", flush=True)

    # Create optimizer — use lr from config (default 2e-4)
    lr = getattr(config, 'lr', 2e-4)
    optimizer = Adam(model.parameters(), lr=lr)
    eta_min = getattr(config, 'cos_eta_min', 1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    print("[TRAIN] Optimizer and scheduler created", flush=True)

    # Training loop
    print(f"[TRAIN] Starting training for {epochs} epochs...", flush=True)
    print("=" * 80, flush=True)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"\n[TRAIN] ===== Epoch {epoch}/{epochs} =====", flush=True)

        # Train
        train_loss, train_batches = train_epoch(model, train_loader, optimizer, device, epoch, pos_weight=pos_weight)
        print(f"[TRAIN] Epoch {epoch} - Train Loss: {train_loss:.6f} ({train_batches} batches)", flush=True)

        # Validate
        val_loss = validate(model, val_loader, device, pos_weight=pos_weight)
        print(f"[TRAIN] Epoch {epoch} - Val Loss: {val_loss:.6f}", flush=True)

        # Step scheduler
        scheduler.step()
        print(f"[TRAIN] Epoch {epoch} - Learning rate: {optimizer.param_groups[0]['lr']:.6e}", flush=True)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / f"best_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[TRAIN] ✓ Saved best checkpoint: {ckpt_path.name}", flush=True)

        # Always save latest
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(model.state_dict(), latest_path)

    print("\n" + "=" * 80, flush=True)
    print(f"[TRAIN] ✓✓✓ Training complete! Best val loss: {best_val_loss:.6f}", flush=True)
    print(f"[TRAIN] Checkpoints saved to: {checkpoint_dir}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Weight for positive (ink) class. Use ~10 for 9%% ink ratio.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override dataset_target_dir from config (for augmented data).')
    args = parser.parse_args()

    sys.exit(main(args.config_path, epochs=args.epochs,
                  checkpoint_dir=args.checkpoint_dir, pos_weight=args.pos_weight,
                  data_dir=args.data_dir))

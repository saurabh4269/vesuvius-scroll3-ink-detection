#!/usr/bin/env python
"""Manual training loop - bypass PyTorch Lightning to diagnose hang."""

import sys
import argparse
import torch
from torch.optim import Adam
from torch.nn import functional as F

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.model.datamodule import UNETR_SF_DataModule
from phoenix.utility.configs import Config


def train_one_epoch_manual(model, train_loader, optimizer, device, max_batches=None):
    """Manual training loop."""
    model.train()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    print("[MANUAL] Starting epoch iteration...")
    sys.stdout.flush()

    for batch_idx, batch in enumerate(train_loader):
        print(f"[MANUAL] Got batch {batch_idx}", flush=True)

        # Move batch to device
        x = batch[0].to(device)
        y = batch[1].to(device)

        print(f"[MANUAL] Batch shape: x={x.shape}, y={y.shape}", flush=True)

        # Forward pass
        print(f"[MANUAL] Forward pass...", flush=True)
        logits = model(x)
        print(f"[MANUAL] Got logits: {logits.shape}", flush=True)

        # Loss
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        print(f"[MANUAL] Loss: {loss.item():.4f}", flush=True)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if max_batches is not None and num_batches >= max_batches:
            break

    avg_loss = total_loss / max(num_batches, 1)
    print(f"[MANUAL] Epoch complete. Avg loss: {avg_loss:.4f}", flush=True)
    return avg_loss


def main(config_path, max_batches=1):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MANUAL] Using device: {device}", flush=True)

    # Load config
    config = Config(config_path)
    print(f"[MANUAL] Loaded config from {config_path}", flush=True)

    # Create model (using Lightning module's inner model)
    print("[MANUAL] Creating model...", flush=True)
    lightning_module = UNETR_SF_Module(cfg=config)
    model = lightning_module.model
    print(f"[MANUAL] Model created: {type(model).__name__}", flush=True)

    # Create datamodule
    print("[MANUAL] Creating DataModule...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    dm.setup(stage="fit")
    print(f"[MANUAL] DataModule setup complete", flush=True)

    # Get train dataloader
    print("[MANUAL] Getting train dataloader...", flush=True)
    train_loader = dm.train_dataloader()
    print(f"[MANUAL] Train dataloader ready", flush=True)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    print("[MANUAL] Optimizer created", flush=True)

    # Train one epoch
    print("[MANUAL] Starting manual training...", flush=True)
    sys.stdout.flush()

    loss = train_one_epoch_manual(model, train_loader, optimizer, device, max_batches=max_batches)

    print(f"[MANUAL] Training complete! Final loss: {loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--max-batches', type=int, default=1)
    args = parser.parse_args()

    sys.exit(main(args.config_path, max_batches=args.max_batches))

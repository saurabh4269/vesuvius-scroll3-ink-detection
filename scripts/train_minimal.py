#!/usr/bin/env python
"""Minimal training loop - test if we can do a forward/backward pass."""

import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.model.datamodule import UNETR_SF_DataModule
from phoenix.utility.configs import Config


def main(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}", flush=True)

    # Load config
    print("[TRAIN] Loading config...", flush=True)
    config = Config.load_from_file(config_path)

    # Create model from Lightning module
    print("[TRAIN] Creating model...", flush=True)
    config_dict = vars(config)
    lightning_module = UNETR_SF_Module(**config_dict)
    model = lightning_module.model
    model.to(device)
    model.train()
    print(f"[TRAIN] Model created: {type(model).__name__}", flush=True)

    # Create datamodule
    print("[TRAIN] Creating DataModule...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    print("[TRAIN] DataModule ready", flush=True)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    print("[TRAIN] Optimizer created", flush=True)

    # Get first batch
    print("[TRAIN] Getting first batch from dataloader...", flush=True)
    sys.stdout.flush()

    batch_iter = iter(train_loader)
    x, y = next(batch_iter)
    print(f"[TRAIN] Got batch: x shape={x.shape}, y shape={y.shape}", flush=True)

    # Move to device
    x = x.to(device)
    y = y.to(device)
    print("[TRAIN] Batch moved to device", flush=True)

    # Forward pass
    print("[TRAIN] Doing forward pass...", flush=True)
    sys.stdout.flush()

    with torch.cuda.device(device):
        logits = model(x)
    print(f"[TRAIN] Forward pass complete. Logits shape: {logits.shape}", flush=True)

    # Compute loss
    print("[TRAIN] Computing loss...", flush=True)
    print(f"[TRAIN] Logits shape: {logits.shape}, Target shape: {y.shape}", flush=True)

    # Model outputs single channel, but targets are 2-channel
    # Sum across channel dimension if needed
    if logits.dim() == 3 and y.dim() == 4:
        # Expand logits to match target channels
        logits = logits.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        logits = logits.expand_as(y)   # Expand to [B, 2, H, W]

    loss = F.binary_cross_entropy_with_logits(logits, y.float())
    print(f"[TRAIN] Loss computed: {loss.item():.6f}", flush=True)

    # Backward pass
    print("[TRAIN] Doing backward pass...", flush=True)
    sys.stdout.flush()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("[TRAIN] Backward pass complete", flush=True)

    print("[TRAIN] ✓✓✓ SUCCESS - Training step completed! ✓✓✓", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    sys.exit(main(args.config_path))

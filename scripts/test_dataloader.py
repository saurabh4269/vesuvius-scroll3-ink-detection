#!/usr/bin/env python
"""Test if dataloader works without Lightning."""

import sys
import argparse

from phoenix.model.datamodule import UNETR_SF_DataModule
from phoenix.utility.configs import Config


def main(config_path):
    print("[TEST] Loading config...", flush=True)
    config = Config.load_from_file(config_path)

    print("[TEST] Creating DataModule...", flush=True)
    dm = UNETR_SF_DataModule(cfg=config)
    dm.setup(stage="fit")

    print("[TEST] Getting train dataloader...", flush=True)
    train_loader = dm.train_dataloader()

    print("[TEST] Starting dataloader iteration...", flush=True)
    sys.stdout.flush()

    for batch_idx, batch in enumerate(train_loader):
        print(f"[TEST] Got batch {batch_idx}: x shape={batch[0].shape}, y shape={batch[1].shape}", flush=True)
        if batch_idx == 0:
            break

    print("[TEST] SUCCESS - First batch loaded!", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    sys.exit(main(args.config_path))

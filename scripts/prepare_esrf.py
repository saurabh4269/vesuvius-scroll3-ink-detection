"""
Convert raw ESRF fragment downloads into the format expected by create_dataset.py:
  {frag_dir}/layers/{00..65}.png  (already there)
  {frag_dir}/label.png            (binary grayscale 0/255, derived from inklabels.png)
  {frag_dir}/mask.png             (binary grayscale 0/255, from {ID}_mask.png)

Run from ~/scroll_prize/vesuvius_first_title_prize/:
  python ~/scroll_prize/scripts/prepare_esrf.py
"""

import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = os.path.expanduser("~/scroll_prize/data/esrf/scroll0/fragments")


def convert_500P2(frag_dir):
    """500P2: inklabels is RGBA, alpha=255 everywhere (ignore it).
    Ink is encoded in R channel (=G=B): 0=no ink, >0=ink (confidence map).
    Actual ink fraction: ~4.1% of pixels."""
    label_src = os.path.join(frag_dir, "500P2_inklabels.png")
    mask_src  = os.path.join(frag_dir, "500P2_mask.png")
    label_dst = os.path.join(frag_dir, "label.png")
    mask_dst  = os.path.join(frag_dir, "mask.png")

    print(f"[500P2] Loading inklabels from {label_src}...")
    rgba = np.array(Image.open(label_src).convert("RGBA"))
    # R channel (=G=B): 0=no ink, 1-255=ink with varying confidence
    r_channel = rgba[:, :, 0]
    binary = (r_channel > 0).astype(np.uint8) * 255
    ink_frac = (binary > 0).mean()
    print(f"[500P2] Ink fraction (R>0): {ink_frac:.1%} ({(binary>0).sum():,} pixels)")
    Image.fromarray(binary).save(label_dst)
    print(f"[500P2] label.png saved → {label_dst}")

    mask = np.array(Image.open(mask_src).convert("L"))
    mask_binary = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_binary).save(mask_dst)
    print(f"[500P2] mask.png saved  → {mask_dst}")


def convert_343P(frag_dir):
    """343P: inklabels is RGBA, ink = R channel > 0 (values 0-9, where 0=no ink)."""
    label_src = os.path.join(frag_dir, "343P_inklabels.png")
    mask_src  = os.path.join(frag_dir, "343P_mask.png")
    label_dst = os.path.join(frag_dir, "label.png")
    mask_dst  = os.path.join(frag_dir, "mask.png")

    print(f"[343P] Loading inklabels from {label_src}...")
    rgba = np.array(Image.open(label_src).convert("RGBA"))
    r_channel = rgba[:, :, 0]
    # R values 1-9 = ink (any annotation), 0 = no ink
    unique_vals = np.unique(r_channel)
    print(f"[343P] R channel unique values: {unique_vals}")
    binary = (r_channel > 0).astype(np.uint8) * 255
    ink_frac = (binary > 0).mean()
    print(f"[343P] Ink fraction (R>0): {ink_frac:.1%} ({(binary>0).sum():,} pixels)")
    Image.fromarray(binary).save(label_dst)
    print(f"[343P] label.png saved → {label_dst}")

    mask = np.array(Image.open(mask_src).convert("L"))
    mask_binary = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_binary).save(mask_dst)
    print(f"[343P] mask.png saved  → {mask_dst}")


def verify_fragment(frag_dir, frag_id):
    """Check all expected files are present and have correct format."""
    issues = []

    # Check layers
    layer_count = len([f for f in os.listdir(os.path.join(frag_dir, "layers"))
                       if f.endswith(".png")])
    if layer_count != 66:
        issues.append(f"Expected 66 layers, found {layer_count}")

    # Check label + mask
    for fname in ["label.png", "mask.png"]:
        fpath = os.path.join(frag_dir, fname)
        if not os.path.exists(fpath):
            issues.append(f"Missing {fname}")
        else:
            arr = np.array(Image.open(fpath).convert("L"))
            vals = set(np.unique(arr))
            if not vals.issubset({0, 255}):
                issues.append(f"{fname} has non-binary values: {vals}")

    if issues:
        print(f"[{frag_id}] ISSUES: {issues}")
    else:
        label = np.array(Image.open(os.path.join(frag_dir, "label.png")).convert("L"))
        mask  = np.array(Image.open(os.path.join(frag_dir, "mask.png")).convert("L"))
        print(f"[{frag_id}] OK — label: {label.shape}, ink={label.mean()/255:.1%}, "
              f"mask coverage={mask.mean()/255:.1%}")


if __name__ == "__main__":
    print(f"Data root: {DATA_ROOT}\n")

    for frag_id, convert_fn in [("500P2", convert_500P2), ("343P", convert_343P)]:
        frag_dir = os.path.join(DATA_ROOT, frag_id)
        if not os.path.isdir(frag_dir):
            print(f"[{frag_id}] Directory not found: {frag_dir} — skipping")
            continue
        convert_fn(frag_dir)
        verify_fragment(frag_dir, frag_id)
        print()

    print("Done. Run create_dataset.py next with configs/ft_esrf.py.")

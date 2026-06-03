"""
Analyze PHerc0009B:
1. CLAHE-enhance the flat segment PNG and search for letters
2. Check m7 ink prediction zarr for isolated letter-like structures
3. Generate comparison: raw CT vs ink prediction
"""
import cv2, zarr
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME    = Path.home()
SEG_DIR = HOME / "scroll_prize/data/pherc0009b/segments"
OUT_DIR = HOME / "scroll_prize/data/pherc0009b/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZARR_S3 = "s3://vesuvius-challenge-open-data/PHerc0009B/representations/predictions/surfaces/20260319104112-surface-20260413222639-surface-m7-L2-th0.2.zarr"

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))

# ─── 1. Enhance and analyse the raw CT segment PNG ──────────────────────────
print("1. Loading raw CT segment PNG...")
seg_file = SEG_DIR / "20250910185200_6e391d8bc886b7eaf7ed.png"
img_raw  = np.array(Image.open(seg_file))
print(f"   Shape: {img_raw.shape}  min={img_raw.min()} max={img_raw.max()}")

# Normalise to uint8
if img_raw.dtype != np.uint8:
    lo, hi = np.percentile(img_raw, [2, 98])
    img_u8 = np.clip((img_raw.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
else:
    img_u8 = img_raw

# CLAHE
print("   Applying CLAHE...")
sm   = gaussian_filter(img_u8.astype(float), sigma=0.5)
enh  = clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# Save full-width thumbnail (1500px wide, preserve aspect)
H, W = enh.shape
scale = min(1500/W, 4000/H)
thumb = Image.fromarray(enh).resize((int(W*scale), int(H*scale)), Image.LANCZOS)
thumb.save(str(OUT_DIR / "seg_clahe_thumb.png"))
print(f"   Saved seg_clahe_thumb.png ({thumb.size})")

# Save top 20% (most likely to contain text near scroll start)
top_h = H // 5
top   = Image.fromarray(enh[:top_h, :]).resize((int(W*scale), int(top_h*scale)), Image.LANCZOS)
top.save(str(OUT_DIR / "seg_clahe_top20pct.png"))

# Inverted (dark ink on white background)
inv = Image.fromarray(255 - enh).resize((int(W*scale), int(H*scale)), Image.LANCZOS)
inv.save(str(OUT_DIR / "seg_clahe_inverted.png"))
print(f"   Saved inverted thumbnail")

# ─── 2. Download and analyse m7 ink prediction zarr ─────────────────────────
print("\n2. Opening m7 ink prediction zarr (level-2, already on Prajna)...")
L2_LOCAL = HOME / "scroll_prize/data/pherc0009b/ink_pred_l2"

import s3fs
import importlib.util

# Try local first, fall back to s3
if L2_LOCAL.exists() and any(L2_LOCAL.iterdir()):
    arr = zarr.open_array(str(L2_LOCAL), mode="r")
    print(f"   Loaded from local: {arr.shape}")
else:
    print("   Opening from S3 (login node)...")
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=ZARR_S3 + "/2", s3=s3)
    arr = zarr.open(store, mode="r")
    print(f"   Shape: {arr.shape}")

NZ, NY, NX = arr.shape
print(f"   Physical: {NZ*4.8/1000:.1f}mm × {NY*4.8/1000:.1f}mm × {NX*4.8/1000:.1f}mm")

# Quick per-z ink fraction scan to find where ink is concentrated
print("   Scanning ink distribution by z-level...")
# Load entire sparse zarr (1820, 1767, 1767 ≈ 5.7GB uncompressed; should be sparse)
# Just sample the diagonal to get ink distribution
STEP = 5   # every 5th z-slice
z_fracs = []
for zi in range(0, NZ, STEP):
    # Sample a circle at center of the volume
    cy, cx = NY//2, NX//2
    r = min(NY, NX) // 3
    angles = np.linspace(0, 2*np.pi, 360)
    ys = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    row = arr[zi, ys, xs].astype(float)
    z_fracs.append((zi, float((row>0).mean())))

z_fracs.sort(key=lambda x: x[1], reverse=True)
print("   Top 10 z-levels by ink fraction:")
for zi, f in z_fracs[:10]:
    print(f"     z={zi} ({zi*4.8/1000:.2f}mm): {f:.3f}")

# Show the top ink slice
if z_fracs:
    best_z, best_frac = z_fracs[0]
    if best_frac > 0.01:
        print(f"\n   Best z={best_z} ({best_frac:.3f}): generating cross-section...")
        slice_z = arr[best_z, :, :]
        if hasattr(slice_z, 'compute'):
            slice_z = slice_z.compute()
        sl = np.array(slice_z, dtype=np.uint8)
        sl_enh = clahe.apply(sl)
        Image.fromarray(255 - sl_enh).save(str(OUT_DIR / f"ink_slice_z{best_z}.png"))
        print(f"   Saved ink_slice_z{best_z}.png")

print("\nDONE.")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix in ('.png',):
        print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

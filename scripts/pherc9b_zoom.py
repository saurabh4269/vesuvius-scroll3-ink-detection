"""
Zoom into the most promising regions of PHerc0009B segment.
The inverted CT image shows structured dark marks in lower third.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME    = Path.home()
SEG_DIR = HOME / "scroll_prize/data/pherc0009b/segments"
OUT_DIR = HOME / "scroll_prize/data/pherc0009b/analysis"

clahe_hard = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
clahe_soft = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))

seg_file = SEG_DIR / "20250910185200_6e391d8bc886b7eaf7ed.png"
img_raw  = np.array(Image.open(seg_file).convert("L"), dtype=np.uint8)
H, W     = img_raw.shape
print(f"Shape: {H}×{W}  (~{H*2.4/1000:.1f}mm × {W*2.4/1000:.1f}mm)")

def process(arr, sigma=0.5, clip=4.0, tile=8):
    sm = gaussian_filter(arr.astype(float), sigma=sigma)
    c  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return c.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── 1. Three horizontal strips with aggressive CLAHE ────────────────────────
print("1. Horizontal strips with CLAHE enhancement...")
N_STRIPS = 5
for i in range(N_STRIPS):
    y0 = i * H // N_STRIPS
    y1 = (i+1) * H // N_STRIPS
    strip = img_raw[y0:y1, :]

    enh  = process(strip, sigma=0.3, clip=5.0, tile=8)
    inv  = 255 - enh

    # 2× zoom, cap at 1500px wide
    scale = min(2.0, 1500/W)
    out_img = Image.fromarray(inv).resize((int(W*scale), int((y1-y0)*scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(out_img)
    phys_y0 = y0 * 2.4 / 1000
    phys_y1 = y1 * 2.4 / 1000
    draw.text((5, 5), f"Strip {i}: y={y0}-{y1} ({phys_y0:.1f}-{phys_y1:.1f}mm)", fill=128)
    # 1mm scale bar
    sb = int(1000/2.4 * scale)
    draw.line([(10, out_img.size[1]-20), (10+sb, out_img.size[1]-20)], fill=80, width=3)
    draw.text((10, out_img.size[1]-35), "1mm", fill=80)
    out_img.save(str(OUT_DIR / f"strip_{i:02d}_clahe.png"))

print("   Saved 5 horizontal strips")

# ─── 2. Focus on bottom third (highest-contrast ink region) ─────────────────
print("2. Bottom third high-zoom (most ink structure)...")
y0 = H * 2 // 3
strip_b = img_raw[y0:, :]
enh_b   = process(strip_b, sigma=0.2, clip=6.0, tile=6)
inv_b   = 255 - enh_b

# 3× zoom
scale3 = min(3.0, 1500/W)
out_b = Image.fromarray(inv_b).resize((int(W*scale3), int(strip_b.shape[0]*scale3)), Image.LANCZOS)
out_b.save(str(OUT_DIR / "bottom_third_3x.png"))
print(f"   Saved bottom_third_3x.png ({out_b.size})")

# ─── 3. If ink prediction zarr downloaded — load and overlay ────────────────
ink_path = HOME / "scroll_prize/data/pherc0009b/ink_pred_l2"
if ink_path.exists() and any(ink_path.glob("*/*")):
    print("3. Ink prediction zarr found — checking...")
    import zarr
    arr = zarr.open_array(str(ink_path), mode="r")
    print(f"   Shape: {arr.shape}")
    nz, ny, nx = arr.shape
    # Find z-slices with most ink
    print("   Loading sparse zarr...")
    data = arr[:][:]
    z_ink = (data>0).mean(axis=(1,2))
    top10 = np.argsort(z_ink)[::-1][:10]
    print("   Top ink z-levels:")
    for zi in top10:
        print(f"     z={zi} ({zi*4.8/1000:.2f}mm): {z_ink[zi]:.4f}")
    # Save best z-slice
    best_z = top10[0]
    sl = data[best_z, :, :].astype(np.uint8)
    enh_sl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16)).apply(sl)
    Image.fromarray(255-enh_sl).save(str(OUT_DIR / f"ink_best_z{best_z}.png"))
    print(f"   Saved ink_best_z{best_z}.png")
else:
    print("3. Ink zarr not ready yet (still downloading)")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

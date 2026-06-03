"""
Apply CLAHE to unrolled level-2 views and hunt for text-line patterns.
r=298 at level-2 is the most promising (12% ink, separated blobs).
"""
import numpy as np
from pathlib import Path
from PIL import Image
import zarr
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: cv2 not available, using manual CLAHE-like processing")

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/highres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Scroll parameters ────────────────────────────────────────────────────────
cy, cx = 496.0, 534.4   # center at level-2
r_target = 298           # inner readable radius
N_ANGLES = 1800
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
z_lo_l2 = 1400

# ─── Load level-2 zarr ───────────────────────────────────────────────────────
print("Loading level-2 zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
data = arr2[z_lo_l2:, :, :]
nz, ny, nx = data.shape
print(f"  Loaded: {data.shape}")

# ─── Build ADAPTIVE RADIUS unroll: for each z-slice, find best radius ────────
print("Building adaptive-radius unroll...")
# Test a range of radii and take the one with maximum ink at each z
RADII = list(range(220, 360, 5))
N_RADII = len(RADII)

# Sample all radii at once for efficiency
print(f"  Sampling {N_RADII} radii × {nz} z-slices × {N_ANGLES} angles...")
adaptive = np.zeros((nz, N_ANGLES), dtype=np.uint8)
per_radius_ink = {}

for r_px in RADII:
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx-1).astype(int)
    sampled = data[:, ys, xs]  # (nz, N_ANGLES)
    adaptive = np.maximum(adaptive, sampled)
    per_radius_ink[r_px] = (sampled > 0).mean()
    print(f"  r={r_px}: ink={(sampled>0).mean()*100:.1f}%")

print(f"  Adaptive max ink: {(adaptive>0).mean()*100:.2f}%")

# ─── Load the r=298 single-radius unroll ─────────────────────────────────────
print("\nSampling r=298 for detailed analysis...")
ys_298 = np.clip(cy + r_target * np.sin(angles), 0, ny-1).astype(int)
xs_298 = np.clip(cx + r_target * np.cos(angles), 0, nx-1).astype(int)
unrolled_298 = data[:, ys_298, xs_298].astype(np.uint8)

# ─── CLAHE-like enhancement ──────────────────────────────────────────────────
def apply_clahe_manual(img, clip=3.0, tile=64):
    """Manual CLAHE approximation: local histogram equalization"""
    result = np.zeros_like(img, dtype=np.uint8)
    H, W = img.shape
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            tile_data = img[y:y+tile, x:x+tile]
            if tile_data.max() == 0:
                continue
            # Compute and clip histogram
            counts, edges = np.histogram(tile_data[tile_data > 0].ravel(), bins=256, range=(0, 256))
            clip_val = max(1, int(clip * counts.mean()))
            excess = np.maximum(counts - clip_val, 0).sum()
            counts = np.minimum(counts, clip_val)
            counts += excess // 256
            # CDF
            cdf = counts.cumsum()
            cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 1
            cdf_max = max(cdf.max(), 1)
            # Map
            mapped = np.interp(tile_data.ravel(), edges[:-1],
                             (cdf - cdf_min) / max(cdf_max - cdf_min, 1) * 255).astype(np.uint8)
            result[y:y+tile, x:x+tile] = mapped.reshape(tile_data.shape)
    return result

print("\nApplying CLAHE enhancement to r=298...")
if HAS_CV2:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    unrolled_clahe = clahe.apply(unrolled_298)
else:
    unrolled_clahe = apply_clahe_manual(unrolled_298, clip=3.0, tile=64)

Image.fromarray(unrolled_clahe).save(str(OUT_DIR / "l2_r298_clahe.png"))
Image.fromarray(np.repeat(unrolled_clahe, 2, axis=0)).save(str(OUT_DIR / "l2_r298_clahe_2x.png"))
print("  Saved l2_r298_clahe.png")

# Also CLAHE on adaptive
print("Applying CLAHE to adaptive-radius unroll...")
if HAS_CV2:
    adaptive_clahe = clahe.apply(adaptive)
else:
    adaptive_clahe = apply_clahe_manual(adaptive, clip=3.0, tile=64)
Image.fromarray(adaptive_clahe).save(str(OUT_DIR / "l2_adaptive_clahe.png"))
Image.fromarray(np.repeat(adaptive_clahe, 2, axis=0)).save(str(OUT_DIR / "l2_adaptive_clahe_2x.png"))
print("  Saved l2_adaptive_clahe.png")

# ─── Text-line detection in r=298 CLAHE ─────────────────────────────────────
print("\nText-line detection in CLAHE-enhanced r=298...")
# Horizontal projection: mean ink per z-slice
z_proj = unrolled_clahe.astype(float).mean(axis=1)  # (nz,)

# Find peaks (text lines would create local maxima in horizontal projection)
# Expected line spacing: 2-3mm = 416-625 px at level-2 z-axis
peaks, _ = find_peaks(z_proj, height=z_proj.mean() * 1.2, distance=50)
print(f"  Found {len(peaks)} z-peaks (potential text lines): {peaks[:10].tolist()}")
if len(peaks) >= 2:
    spacings = np.diff(peaks)
    print(f"  Spacings (z-px): {spacings[:10].tolist()}")
    print(f"  Mean spacing: {spacings.mean():.1f} px = {spacings.mean() * 4.8 / 1000:.2f} mm")

# Also try on adaptive CLAHE
z_proj_adap = adaptive_clahe.astype(float).mean(axis=1)
peaks_a, _ = find_peaks(z_proj_adap, height=z_proj_adap.mean() * 1.2, distance=50)
print(f"\n  Adaptive: Found {len(peaks_a)} z-peaks: {peaks_a[:10].tolist()}")
if len(peaks_a) >= 2:
    sp_a = np.diff(peaks_a)
    print(f"  Mean adaptive spacing: {sp_a.mean():.1f} px = {sp_a.mean() * 4.8 / 1000:.2f} mm")

# ─── Save zoomed crops of the CLAHE r=298 at several specific z-ranges ──────
print("\nSaving zoomed strips from CLAHE r=298...")
# Look at 5 equal sections of the z-range
Z_RANGE = nz  # 700
STRIP_H = 140  # 140 z-px = 0.67 mm per strip

for strip_idx in range(5):
    z_start = strip_idx * STRIP_H
    z_end = min(z_start + STRIP_H, Z_RANGE)
    strip = unrolled_clahe[z_start:z_end, :]  # (140, 1800)
    strip_4x = np.repeat(strip, 4, axis=0)    # 4× vertical zoom
    Image.fromarray(strip_4x).save(str(OUT_DIR / f"l2_r298_clahe_strip{strip_idx:02d}.png"))
    print(f"  Strip {strip_idx}: z={z_start+z_lo_l2}-{z_end+z_lo_l2} ({(z_start+z_lo_l2)*4.8/1000:.2f}-{(z_end+z_lo_l2)*4.8/1000:.2f}mm)")

# ─── Save a dedicated 1200px-wide zoom of the most ink-rich angular region ──
print("\nFinding most ink-rich 600-px angular window...")
ink_per_angle = (unrolled_298 > 0).mean(axis=0)  # (N_ANGLES,)
best_a = 0
best_v = 0
for a_start in range(0, N_ANGLES - 600, 30):
    v = ink_per_angle[a_start:a_start+600].sum()
    if v > best_v:
        best_v = v
        best_a = a_start

print(f"  Best window: angle={best_a}-{best_a+600} (arc {best_a*4*4.8/1000:.1f}-{(best_a+600)*4*4.8/1000:.1f}mm)")
zoom_crop = unrolled_clahe[:, best_a:best_a+600]
Image.fromarray(np.repeat(zoom_crop, 3, axis=0)).save(str(OUT_DIR / "l2_r298_clahe_best_window.png"))
print("  Saved l2_r298_clahe_best_window.png")

# ─── Save radial comparison at the best angular window ───────────────────────
print("\nSaving radial comparison at best window...")
R_COMPARE = [220, 240, 260, 280, 298, 320, 340]
combined_radii = []
for r_cmp in R_COMPARE:
    ys_c = np.clip(cy + r_cmp * np.sin(angles), 0, ny-1).astype(int)
    xs_c = np.clip(cx + r_cmp * np.cos(angles), 0, nx-1).astype(int)
    slc = data[:, ys_c, xs_c].astype(np.uint8)
    crop = slc[:, best_a:best_a+600]
    if HAS_CV2:
        crop_clahe = clahe.apply(crop)
    else:
        crop_clahe = apply_clahe_manual(crop)
    combined_radii.append(crop_clahe)

# Stack vertically with 2-px separator
separator = np.ones((2, 600), dtype=np.uint8) * 128
stacked = []
for c in combined_radii:
    stacked.append(c)
    stacked.append(separator)
img_stacked = np.vstack(stacked[:-1])
Image.fromarray(img_stacked).save(str(OUT_DIR / "l2_radii_comparison.png"))
print(f"  Saved l2_radii_comparison.png ({img_stacked.shape}) showing r={R_COMPARE}")

print("\nDONE. Files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

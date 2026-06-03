"""
Unroll team's Scroll 3 3D ink predictions (m7_nnUNet zarr, level 3)
to a flat 2D surface view for letter hunting.

The scroll is roughly cylindrical. Strategy:
1. Find the scroll center in the Y-X cross-section
2. For each z-slice, extract the ink values along concentric rings at different radii
3. Produce a rectangular (z × angle) "unrolled" image at each radius
4. Look for letter-shaped ink patterns in the unrolled views
"""
import zarr
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

HOME = Path.home()
ZARR_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level3"
OUT_DIR   = HOME / "scroll_prize/data/scroll3_ink_pred/unrolled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Load zarr ────────────────────────────────────────────────────────────
print("Loading zarr level-3...")
arr = zarr.open_array(str(ZARR_PATH), mode="r")
print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")  # (1050, 493, 493)
data = arr[:]  # 12 MB — fits in RAM
nz, ny, nx = data.shape
print(f"  Global ink>0: {(data>0).mean()*100:.2f}%")

# ─── 2. Find scroll center and radius from a representative z-slice ──────────
print("\nFinding scroll center from z-slice average...")
# Average over z-range with most ink (z=800-1049)
avg_slice = (data[800:, :, :] > 0).astype(float).mean(axis=0)  # (ny, nx)

# The scroll body is a ring — center is where the max-projection density peaks
# Use centroid of bright ring
ys, xs = np.mgrid[0:ny, 0:nx]
mask = avg_slice > avg_slice.mean()
if mask.sum() == 0:
    mask = avg_slice > 0
cy_center = (ys * mask).sum() / mask.sum()
cx_center = (xs * mask).sum() / mask.sum()
print(f"  Estimated center: ({cy_center:.1f}, {cx_center:.1f})")

# Compute radial distance from center for all pixels
r_map = np.sqrt((ys - cy_center)**2 + (xs - cx_center)**2)

# Find the dominant radius (ring radius) via radial histogram of ink
r_bins = np.arange(0, min(ny, nx)//2 + 1, 1)
z_ink_map = (data[800:] > 0).astype(float).mean(axis=0)
r_hist, _ = np.histogram(r_map.ravel(), bins=r_bins, weights=z_ink_map.ravel())
r_hist_ct, _ = np.histogram(r_map.ravel(), bins=r_bins)
r_hist_norm = r_hist / np.maximum(r_hist_ct, 1)

# Find dominant radius (peak of radial ink distribution)
dominant_r = r_bins[np.argmax(r_hist_norm[10:])+10]
print(f"  Dominant ink radius: {dominant_r:.1f} px = {dominant_r * 19.2 / 1000:.1f} mm")
print(f"  (Scroll circumference at this radius: {2*np.pi*dominant_r:.0f} px = {2*np.pi*dominant_r*19.2/1000:.0f} mm)")

# Plot radial ink histogram
hist_img = np.zeros((200, len(r_hist_norm)), dtype=np.uint8)
max_h = r_hist_norm.max()
for i, v in enumerate(r_hist_norm):
    bar_h = int(v / max_h * 190)
    if bar_h > 0:
        hist_img[200-bar_h:200, i] = 200
hist_img[:, int(dominant_r)] = 255
Image.fromarray(hist_img).save(str(OUT_DIR / "radial_ink_histogram.png"))
print(f"  Saved radial_ink_histogram.png")

# ─── 3. Create unrolled views at multiple radii ──────────────────────────────
# We'll sample at: dominant_r-16, dominant_r, dominant_r+16, dominant_r+32 (different layers)
radii_to_sample = [
    max(10, dominant_r - 32),
    max(10, dominant_r - 16),
    dominant_r,
    dominant_r + 16,
    dominant_r + 32,
]

N_ANGLES = 800  # angular resolution for unrolled image
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

print(f"\nUnrolling scroll (z × theta) for radii: {[f'{r:.0f}' for r in radii_to_sample]}")

for r_target in radii_to_sample:
    # Sample coordinates for this radius
    ys_sample = cy_center + r_target * np.sin(angles)  # shape (N_ANGLES,)
    xs_sample = cx_center + r_target * np.cos(angles)  # shape (N_ANGLES,)

    # Clip to valid range
    ys_sample = np.clip(ys_sample, 0, ny-1).astype(int)
    xs_sample = np.clip(xs_sample, 0, nx-1).astype(int)

    # Build unrolled image: rows = z (0 to nz-1), cols = angle (0 to N_ANGLES-1)
    unrolled = data[:, ys_sample, xs_sample]  # (nz, N_ANGLES)
    print(f"  r={r_target:.0f}: unrolled shape={unrolled.shape}, ink>0={( unrolled>0).mean()*100:.2f}%")

    # Save full unrolled PNG
    Image.fromarray(unrolled).save(str(OUT_DIR / f"unrolled_r{int(r_target):03d}_full.png"))

    # Also save the bottom z-range (highest ink density: z=900-1049)
    bottom_strip = unrolled[900:, :]
    bottom_strip_smooth = gaussian_filter(bottom_strip.astype(float), sigma=1)
    bottom_8 = np.clip(bottom_strip_smooth / max(bottom_strip_smooth.max(), 1) * 255, 0, 255).astype(np.uint8)
    # Upscale 3× in y for visibility (150px → 450px)
    bottom_big = np.repeat(bottom_8, 3, axis=0)
    Image.fromarray(bottom_big).save(str(OUT_DIR / f"unrolled_r{int(r_target):03d}_bottom3x.png"))

    print(f"    Saved unrolled_r{int(r_target):03d}_full.png + _bottom3x.png")

# ─── 4. Max-radius unroll (take max over ±8 radius band) ─────────────────────
print(f"\nBuilding max-radius-band unrolled view (r={dominant_r:.0f}±8)...")
band_unrolled = np.zeros((nz, N_ANGLES), dtype=np.uint8)
for r_offset in range(-8, 9):
    r_band = dominant_r + r_offset
    if r_band < 5:
        continue
    ys_s = np.clip(cy_center + r_band * np.sin(angles), 0, ny-1).astype(int)
    xs_s = np.clip(cx_center + r_band * np.cos(angles), 0, nx-1).astype(int)
    band_unrolled = np.maximum(band_unrolled, data[:, ys_s, xs_s])

print(f"  Band unrolled ink>0: {(band_unrolled>0).mean()*100:.2f}%")
Image.fromarray(band_unrolled).save(str(OUT_DIR / f"unrolled_band_full.png"))

# Bottom section (z=800-1049 = highest ink)
bottom_band = band_unrolled[800:, :]
bottom_band_big = np.repeat(np.repeat(bottom_band, 3, axis=0), 1, axis=1)
Image.fromarray(bottom_band_big).save(str(OUT_DIR / f"unrolled_band_bottom3x.png"))

# ─── 5. Look for text-like structure in unrolled view ────────────────────────
print("\nAnalysing unrolled image for text-line patterns...")
# Count ink per z-row in band unrolled
ink_per_z = (band_unrolled > 0).mean(axis=1)  # fraction of angles with ink at each z

# Find z-ranges with most ink (text lines should show horizontal bands)
# Use top z-range where ink is concentrated
top_z_idx = np.argsort(ink_per_z)[::-1][:20]
print(f"  Top-20 z-slices by ink fraction:")
for z in sorted(top_z_idx[:20]):
    print(f"    z={z} (level-0 z≈{z*8}): ink={ink_per_z[z]*100:.1f}%")

# Check for text-line spacing (regular banding in z)
# Text lines create peaks in the z-profile; measure their spacing
from scipy.signal import find_peaks
peaks, props = find_peaks(ink_per_z[800:], height=0.05, distance=5)
print(f"\n  Peaks in z-profile (z=800+):")
if len(peaks) > 0:
    print(f"    Found {len(peaks)} peaks at z+800: {(peaks+800)[:20].tolist()}")
    if len(peaks) >= 2:
        spacings = np.diff(peaks)
        print(f"    Peak spacings: {spacings[:10].tolist()} (mean={spacings.mean():.1f} px = {spacings.mean()*19.2:.0f} µm = {spacings.mean()*19.2/1000:.2f} mm)")
else:
    print(f"    No clear peaks (ink too uniform or too sparse)")

# ─── 6. Also check the z-profile of a specific angular wedge ────────────────
print("\nZ-profile for specific angular wedges...")
wedge_size = N_ANGLES // 8  # 1/8 of circumference
for wedge_start in range(0, N_ANGLES, N_ANGLES // 4):
    wedge = band_unrolled[:, wedge_start:wedge_start+wedge_size]
    z_frac = (wedge > 0).mean(axis=1)
    top_zs = np.argsort(z_frac)[::-1][:3]
    print(f"  Wedge angle {wedge_start*360//N_ANGLES}°–{(wedge_start+wedge_size)*360//N_ANGLES}°: "
          f"mean_ink={z_frac.mean()*100:.1f}%, top_z={top_zs.tolist()}")

print("\nDONE. Output files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

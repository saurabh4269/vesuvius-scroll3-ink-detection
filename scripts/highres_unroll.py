"""
High-resolution unrolling: read zarr level-2 (4.8 µm/px) directly from S3
for the inner readable region of Scroll 3.

Level-2 shape: ~2100 × 985 × 985
Level-3 readable radius ≈ 149 px → level-2 ≈ 149*2 = 298 px
Level-3 center: (248, 267) → level-2 center: (496, 534)
"""
import zarr, s3fs
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

HOME = Path.home()
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/highres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZARR_S3 = "vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"

# ─── 1. Open zarr level-2 from S3 ────────────────────────────────────────────
print("Connecting to S3 zarr...")
fs = s3fs.S3FileSystem(anon=True)
store = s3fs.S3Map(ZARR_S3, s3=fs)
root = zarr.open(store, mode="r")
print(f"  Root keys: {list(root.keys())}")
arr2 = root["2"]
print(f"  Level-2 shape: {arr2.shape}, dtype: {arr2.dtype}")
# Level-2 shape should be ~(2100, 985, 985)

# ─── 2. Determine coordinates ─────────────────────────────────────────────────
# Level-3 center: (248.0, 267.2), readable radius: 149 px
# Level-2 is 2× higher res than level-3
SCALE = 2  # level-2 relative to level-3
cy_l3, cx_l3 = 248.0, 267.2
readable_r_l3 = 149.0  # px at level-3

cy = cy_l3 * SCALE
cx = cx_l3 * SCALE
r_inner = (readable_r_l3 - 30) * SCALE  # go a bit inside
r_outer = (readable_r_l3 + 30) * SCALE  # and outside
print(f"  Level-2 center: ({cy:.0f}, {cx:.0f})")
print(f"  Sampling radii: {r_inner:.0f} to {r_outer:.0f} px (level-2)")
print(f"  = {r_inner*4.8/1000:.2f} to {r_outer*4.8/1000:.2f} mm")

# ─── 3. Load only the bottom z-range (high-ink zone) ────────────────────────
# Level-3 high-ink zone: z=700-1049
# Level-2: z=700*2=1400 to 1049*2=2098
z_lo = max(0, 700 * SCALE)
z_hi = arr2.shape[0]
print(f"\nLoading level-2 z={z_lo}:{z_hi} (size: {z_hi-z_lo} slices)...")
# y and x: load full cross-section to allow polar sampling
data_l2 = arr2[z_lo:z_hi, :, :]
print(f"  Loaded shape: {data_l2.shape}, ink>0: {(data_l2>0).mean()*100:.2f}%")

nz, ny, nx = data_l2.shape

# ─── 4. Unroll at multiple radii at level-2 ──────────────────────────────────
N_ANGLES = 1200
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

radii_l2 = [
    int(readable_r_l3 * SCALE - 40),   # deep inner
    int(readable_r_l3 * SCALE - 20),   # inner
    int(readable_r_l3 * SCALE),        # readable surface
    int(readable_r_l3 * SCALE + 20),   # outer
]

print(f"\nUnrolling at radii: {radii_l2} px (level-2)")

for r_target in radii_l2:
    ys_s = np.clip(cy + r_target * np.sin(angles), 0, ny-1).astype(int)
    xs_s = np.clip(cx + r_target * np.cos(angles), 0, nx-1).astype(int)

    unrolled = data_l2[:, ys_s, xs_s]  # (nz, N_ANGLES)
    ink_frac = (unrolled > 0).mean() * 100

    # Apply light Gaussian smoothing
    unrolled_f = gaussian_filter(unrolled.astype(float), sigma=0.5)

    # Normalize
    lo_p, hi_p = np.percentile(unrolled_f[unrolled_f > 0], 5), unrolled_f.max()
    unrolled_8 = np.clip((unrolled_f - lo_p) / max(hi_p - lo_p, 1) * 255, 0, 255).astype(np.uint8)

    print(f"  r={r_target}: ink={ink_frac:.1f}%, range=[{unrolled.min()},{unrolled.max()}]")

    # Save full bottom strip
    Image.fromarray(unrolled_8).save(str(OUT_DIR / f"l2_unrolled_r{r_target:03d}_z{z_lo}.png"))

    # Save 2× zoom in z (rows) for visibility
    zoomed = np.repeat(unrolled_8, 2, axis=0)
    Image.fromarray(zoomed).save(str(OUT_DIR / f"l2_unrolled_r{r_target:03d}_z{z_lo}_2x.png"))

# ─── 5. Max-band unroll (inner surface ±20 px band) ─────────────────────────
print("\nBuilding max-band unrolled view...")
r_center = int(readable_r_l3 * SCALE)
band_data = np.zeros((nz, N_ANGLES), dtype=np.uint8)
for r_off in range(-20, 21, 2):
    r_b = r_center + r_off
    if r_b < 10:
        continue
    ys_b = np.clip(cy + r_b * np.sin(angles), 0, ny-1).astype(int)
    xs_b = np.clip(cx + r_b * np.cos(angles), 0, nx-1).astype(int)
    band_data = np.maximum(band_data, data_l2[:, ys_b, xs_b])

print(f"  Band ink>0: {(band_data>0).mean()*100:.2f}%")

# Normalize
band_f = gaussian_filter(band_data.astype(float), sigma=0.7)
lo_b, hi_b = np.percentile(band_f[band_f > 0], 5) if (band_f > 0).any() else (0, 1), band_f.max()
band_8 = np.clip((band_f - lo_b) / max(hi_b - lo_b, 1) * 255, 0, 255).astype(np.uint8)
Image.fromarray(band_8).save(str(OUT_DIR / f"l2_band_z{z_lo}.png"))
Image.fromarray(np.repeat(band_8, 2, axis=0)).save(str(OUT_DIR / f"l2_band_z{z_lo}_2x.png"))

# ─── 6. Find letter-candidate regions in the band ────────────────────────────
print("\nScanning for letter candidates (50×50 px windows) ...")
WIN = 50
threshold_count = 5  # at least 5 px > 20 in the window

letter_cands = []
for z in range(0, nz - WIN, WIN//2):
    for a in range(0, N_ANGLES - WIN, WIN//2):
        block = band_data[z:z+WIN, a:a+WIN]
        high = (block > 0).sum()
        if high < threshold_count:
            continue
        # Want spatial extent in both directions
        rows_with_ink = (block > 0).any(axis=1).sum()
        cols_with_ink = (block > 0).any(axis=0).sum()
        if rows_with_ink >= 5 and cols_with_ink >= 5:
            letter_cands.append((high, rows_with_ink, cols_with_ink, z, a))

letter_cands.sort(reverse=True)
print(f"  {len(letter_cands)} letter candidates")
print("  Top-30:")
for cnt, rw, cw, z, a in letter_cands[:30]:
    # Convert to physical coordinates
    z_phys_mm = (z + z_lo) * 4.8 / 1000
    arc_mm = a * (2 * np.pi * r_center) / N_ANGLES * 4.8 / 1000
    print(f"    z_slice={z+z_lo}(z={z_phys_mm:.1f}mm), angle_px={a}(arc={arc_mm:.1f}mm): "
          f"ink_px={cnt}, rows={rw}, cols={cw}")

# Save top-10 candidate crops
print("\nSaving letter candidate crops...")
CROP_SIZE = 200  # 200×200 px context at level-2
for idx, (cnt, rw, cw, z, a) in enumerate(letter_cands[:10]):
    z_lo_c = max(0, z - CROP_SIZE//2)
    z_hi_c = min(nz, z + CROP_SIZE//2)
    a_lo_c = max(0, a - CROP_SIZE//2)
    a_hi_c = min(N_ANGLES, a + CROP_SIZE//2)

    crop = band_data[z_lo_c:z_hi_c, a_lo_c:a_hi_c]
    crop_f = gaussian_filter(crop.astype(float), sigma=0.5)
    crop_8 = np.clip(crop_f / max(crop_f.max(), 1) * 255, 0, 255).astype(np.uint8)

    # 3× zoom for visibility
    crop_big = np.repeat(np.repeat(crop_8, 3, axis=0), 3, axis=1)
    img = Image.fromarray(crop_big)
    fname = OUT_DIR / f"lcand_{idx:02d}_z{z+z_lo}_a{a}.png"
    img.save(str(fname))
    print(f"  Saved {fname.name}")

print("\nDONE. Files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

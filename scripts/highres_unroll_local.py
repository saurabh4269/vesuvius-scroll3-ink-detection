"""
High-resolution unrolling using locally downloaded level-2 zarr chunks.
Also does fine-grained letter hunting in unrolled views.

Level-2 shape: (2100, 986, 986) @ 4.8 µm/px
z=1400-2100 = bottom high-ink zone (level-3 z=700-1049)
Readable radius at level-2 ≈ 298 px (2× level-3 value of 149)
Center at level-2 ≈ (496, 534)
"""
import zarr
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/highres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Open local level-2 zarr ──────────────────────────────────────────────
print("Opening level-2 zarr from local disk...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
print(f"  Shape: {arr2.shape}, dtype: {arr2.dtype}, chunks: {arr2.chunks}")

# ─── 2. Define parameters ────────────────────────────────────────────────────
# Level-3 values (from unroll_zarr.py results)
cy_l3, cx_l3 = 248.0, 267.2
r_readable_l3 = 149.0  # inner readable radius at level-3

SCALE = 2  # level-2 is 2× finer than level-3
cy = cy_l3 * SCALE      # 496.0
cx = cx_l3 * SCALE      # 534.4
r_center = r_readable_l3 * SCALE  # 298 px at level-2

N_ANGLES = 1800  # high angular resolution for level-2
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

# z range: level-3 z=700 → level-2 z=1400; level-3 z=1050 → level-2 z=2100
z_lo_l2 = 1400
z_hi_l2 = arr2.shape[0]  # 2100
print(f"  Using z={z_lo_l2}:{z_hi_l2}, radius={r_center:.0f} px")
print(f"  Physical: z = {z_lo_l2*4.8/1000:.1f} – {z_hi_l2*4.8/1000:.1f} mm")
print(f"  Circumference at r={r_center:.0f}: {2*np.pi*r_center*4.8/1000:.1f} mm")

# ─── 3. Load the z-range we need ─────────────────────────────────────────────
print(f"\nLoading level-2 z={z_lo_l2}:{z_hi_l2}...")
try:
    data = arr2[z_lo_l2:z_hi_l2, :, :]
    print(f"  Loaded: {data.shape}, ink>0: {(data>0).mean()*100:.2f}%")
except Exception as e:
    print(f"  Load failed: {e}")
    print("  Checking what chunks exist...")
    import os
    existing = sorted([d for d in Path(str(L2_PATH)).iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"  Z-chunk dirs: {[d.name for d in existing]}")
    raise

nz_l = data.shape[0]
ny_l = data.shape[1]
nx_l = data.shape[2]

# ─── 4. Unroll at multiple radii ─────────────────────────────────────────────
radii = [
    int(r_center - 60),  # deep inner
    int(r_center - 30),  # inner
    int(r_center),       # readable surface
    int(r_center + 30),  # slightly outer
]

print(f"\nUnrolling at radii: {radii} px")
print(f"Arc resolution: {2*np.pi*r_center/N_ANGLES*4.8:.1f} µm per angle pixel")

for r_px in radii:
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny_l-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx_l-1).astype(int)

    unrolled = data[:, ys, xs].astype(np.uint8)  # (nz_l, N_ANGLES)
    ink_frac = (unrolled > 0).mean() * 100
    print(f"  r={r_px}: ink={ink_frac:.1f}%")

    # Normalize and smooth
    sm = gaussian_filter(unrolled.astype(float), sigma=[0.5, 1.0])
    norm = np.clip(sm / max(sm.max(), 1) * 255, 0, 255).astype(np.uint8)

    fname = OUT_DIR / f"l2_r{r_px:03d}.png"
    Image.fromarray(norm).save(str(fname))

    # 2× zoom version
    Image.fromarray(np.repeat(norm, 2, axis=0)).save(str(OUT_DIR / f"l2_r{r_px:03d}_2x.png"))
    print(f"    Saved l2_r{r_px:03d}.png ({norm.shape})")

# ─── 5. Max-band unroll ──────────────────────────────────────────────────────
print("\nMax-band unroll (r±30)...")
band = np.zeros((nz_l, N_ANGLES), dtype=np.uint8)
for r_off in range(-30, 31, 3):
    r_b = int(r_center + r_off)
    ys_b = np.clip(cy + r_b * np.sin(angles), 0, ny_l-1).astype(int)
    xs_b = np.clip(cx + r_b * np.cos(angles), 0, nx_l-1).astype(int)
    band = np.maximum(band, data[:, ys_b, xs_b])

print(f"  Band ink>0: {(band>0).mean()*100:.2f}%")
sm_band = gaussian_filter(band.astype(float), sigma=[0.5, 1.0])
norm_band = np.clip(sm_band / max(sm_band.max(), 1) * 255, 0, 255).astype(np.uint8)
Image.fromarray(norm_band).save(str(OUT_DIR / "l2_band.png"))
Image.fromarray(np.repeat(norm_band, 2, axis=0)).save(str(OUT_DIR / "l2_band_2x.png"))

# ─── 6. Letter candidate detection ──────────────────────────────────────────
print("\nFinding letter candidates in band view (z × angle)...")
# At level-2: letter ~1mm wide = 208 px circumferential, 1.5mm tall = 312 px in z
# Search windows: 100×100 px at level-2
WIN_Z = 80
WIN_A = 100
STRIDE_Z = 40
STRIDE_A = 50

cands = []
for z in range(0, nz_l - WIN_Z, STRIDE_Z):
    for a in range(0, N_ANGLES - WIN_A, STRIDE_A):
        blk = band[z:z+WIN_Z, a:a+WIN_A]
        ink = (blk > 0).sum()
        if ink < 20:
            continue
        rows_ink = (blk > 0).any(axis=1).sum()
        cols_ink = (blk > 0).any(axis=0).sum()
        if rows_ink < 10 and cols_ink < 10:
            continue
        cands.append((ink, rows_ink, cols_ink, z, a))

cands.sort(reverse=True)
print(f"  Found {len(cands)} candidates")
print("  Top-20:")
for ink, rw, cw, z, a in cands[:20]:
    z_mm = (z + z_lo_l2) * 4.8 / 1000
    arc_mm = a * (2 * np.pi * r_center) / N_ANGLES * 4.8 / 1000
    print(f"    z={z+z_lo_l2}({z_mm:.2f}mm), a={a}({arc_mm:.1f}mm): ink_px={ink}, r={rw}, c={cw}")

# Save top-15 candidate crops (300×300 context)
CROP = 300
print("\nSaving letter candidate crops...")
for idx, (ink, rw, cw, z, a) in enumerate(cands[:15]):
    z0 = max(0, z - CROP//2)
    z1 = min(nz_l, z + CROP//2)
    a0 = max(0, a - CROP//2)
    a1 = min(N_ANGLES, a + CROP//2)

    crop = band[z0:z1, a0:a1]
    sm_crop = gaussian_filter(crop.astype(float), sigma=0.5)
    norm_c = np.clip(sm_crop / max(sm_crop.max(), 1) * 255, 0, 255).astype(np.uint8)

    # 3× zoom
    big = np.repeat(np.repeat(norm_c, 3, axis=0), 3, axis=1)
    Image.fromarray(big).save(str(OUT_DIR / f"l2_cand_{idx:02d}_z{z+z_lo_l2}_a{a}.png"))

print(f"  Saved {min(len(cands), 15)} crops")

# ─── 7. Z-profile of band ────────────────────────────────────────────────────
print("\nZ-profile (ink fraction per z-slice)...")
z_frac = (band > 0).mean(axis=1)
top_z = np.argsort(z_frac)[::-1][:10]
print("  Top-10 z-slices by ink fraction:")
for z in sorted(top_z):
    print(f"    z={z+z_lo_l2} ({(z+z_lo_l2)*4.8/1000:.2f} mm): {z_frac[z]*100:.1f}%")

print("\nDONE. Files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

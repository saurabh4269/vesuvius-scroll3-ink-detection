"""
Full scroll scan: all z-slabs 0-10 (z=0-2100, full 10mm height) at r=310.

Now that we have all level-2 z-slabs downloaded, do a comprehensive search:
1. Compute per-z ink fraction at r=310, angle=280-520 (known candidate zone)
2. Compute per-z ink fraction at r=310, FULL CIRCLE to find ink-dense z-levels
3. Find all z-zones with significant ink (>5%)
4. Generate full-scroll panorama strip at angle=280-520
5. Generate full-circle scan at key z-levels where ink fraction is highest
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

print("Opening full level-2 zarr (z=0-2100)...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
NZ, NY, NX = arr2.shape
print(f"  Shape: {arr2.shape}")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# Precompute angle indices for r=310
r_px = 310
ys310 = np.clip(cy + r_px * np.sin(angles), 0, NY-1).astype(int)
xs310 = np.clip(cx + r_px * np.cos(angles), 0, NX-1).astype(int)

# ─── 1. Per-z ink fraction scan ──────────────────────────────────────────────
print("\n1. Scanning all z-slices for ink fraction at r=310...")
CAND_A0, CAND_A1 = 280, 520   # known candidate angle range

# Process in z-slab chunks to avoid loading full 2100×986×986 into RAM
CHUNK = 192   # zarr chunk size
z_fracs_cand  = []   # ink fraction in candidate angle range
z_fracs_full  = []   # ink fraction in full circle (excl. outer wall a>1100)
z_fracs_inner = []   # ink fraction at r=280-320 (inner surface band)

for z_start in range(0, NZ, CHUNK):
    z_end = min(z_start + CHUNK, NZ)
    slab_raw = arr2[z_start:z_end, :, :][:]   # (z_chunk, 986, 986)
    slab = slab_raw[:, ys310, xs310].astype(np.float32)  # (z_chunk, 1800)

    frac_cand = (slab[:, CAND_A0:CAND_A1] > 0).mean(axis=1)
    frac_full = (slab[:, :1100] > 0).mean(axis=1)   # exclude outer wall

    z_fracs_cand.extend(frac_cand.tolist())
    z_fracs_full.extend(frac_full.tolist())

    print(f"  z={z_start}-{z_end}: cand_mean={frac_cand.mean():.3f}, full_mean={frac_full.mean():.3f}")

z_fracs_cand = np.array(z_fracs_cand)
z_fracs_full = np.array(z_fracs_full)
print(f"\n  Global: cand_max={float(z_fracs_cand.max()):.3f} at z={int(np.argmax(z_fracs_cand))}")
print(f"  Global: full_max={float(z_fracs_full.max()):.3f} at z={int(np.argmax(z_fracs_full))}")

# ─── 2. Find top-20 z-levels by full-circle ink fraction ─────────────────────
print("\n2. Top-20 z-levels by full-circle ink (excl. outer wall)...")
top_z = np.argsort(z_fracs_full)[::-1][:20]
top_z_info = []
for zi in top_z:
    zi = int(zi)
    print(f"  z={zi:4d} ({zi*4.8/1000:.3f}mm): full={float(z_fracs_full[zi]):.3f}, cand={float(z_fracs_cand[zi]):.3f}")
    top_z_info.append(zi)

# ─── 3. Find contiguous ink zones in candidate angle range ───────────────────
print("\n3. Ink zones at angle=280-520 across full scroll height...")
in_zone = False
zones = []
start = 0
THRESH = 0.05
for zi, f in enumerate(z_fracs_cand):
    if not in_zone and f > THRESH:
        in_zone = True
        start = zi
    elif in_zone and f <= THRESH * 0.4:
        zones.append((start, zi))
        in_zone = False
if in_zone:
    zones.append((start, NZ))

print(f"  Found {len(zones)} zones:")
for z0, z1 in zones:
    phys_z0 = z0 * 4.8 / 1000
    phys_z1 = z1 * 4.8 / 1000
    peak    = z_fracs_cand[z0:z1].max()
    print(f"    z={z0}-{z1} ({phys_z0:.2f}-{phys_z1:.2f}mm), span={z1-z0}px={( z1-z0)*4.8/1000:.2f}mm, peak_frac={peak:.2f}")

# ─── 4. Full-scroll strip at candidate angle range ───────────────────────────
print("\n4. Generating full-scroll strip (z=0-2100, angle=280-520, r=310)...")
# 1× zoom — just the raw CLAHE strip, 2100×240
strip_all = np.zeros((NZ, CAND_A1 - CAND_A0), dtype=np.uint8)

for z_start in range(0, NZ, CHUNK):
    z_end  = min(z_start + CHUNK, NZ)
    slab_raw = arr2[z_start:z_end, :, :][:]
    slab   = slab_raw[:, ys310, xs310].astype(np.uint8)
    strip_all[z_start:z_end] = slab[:, CAND_A0:CAND_A1]

# CLAHE on full strip
strip_enh = enh(strip_all, sigma=0.5)
strip_inv = 255 - strip_enh

# 3× zoom in z, 2× in angle
full_strip_zoom = np.repeat(np.repeat(strip_inv, 3, axis=0), 2, axis=1)

img_fs = Image.fromarray(np.stack([full_strip_zoom]*3, axis=2))
draw   = ImageDraw.Draw(img_fs)
W = full_strip_zoom.shape[1]

# Mark mm ticks on left edge
for mm in range(0, 11):
    z_px = int(mm * 1000 / 4.8) * 3
    if z_px < full_strip_zoom.shape[0]:
        draw.line([(0, z_px), (20, z_px)], fill=(200,0,0), width=2)
        draw.text((22, z_px - 8), f"{mm}mm", fill=(200,0,0))

# Mark known candidate zone
draw.rectangle([0, zones[0][0]*3 if zones else 0, W-1, zones[0][1]*3 if zones else 0],
               outline=(0,200,0), width=2)

# Scale bar: 1mm = 208px × 3 = 624 vertical px
img_fs.save(str(OUT_DIR / "v5_fullscroll_strip_280_520.png"))
print(f"  Saved v5_fullscroll_strip_280_520.png ({full_strip_zoom.shape})")

# ─── 5. Per-z-slab panoramas for the top ink-dense slabs ─────────────────────
print("\n5. Generating panoramas at top-5 z-levels...")
top5_unique_slabs = sorted(set([zi // CHUNK for zi in top_z[:10]]))[:5]

for z_slab_idx in top5_unique_slabs:
    z0 = z_slab_idx * CHUNK
    z1 = min(z0 + CHUNK, NZ)
    slab = arr2[z0:z1, :, :][:, ys310, xs310].astype(np.uint8)  # (192, 1800)
    # Take z-max projection in slab
    proj = slab.max(axis=0)   # (1800,)
    # Enhance
    proj_e = clahe.apply(proj)
    inv    = 255 - proj_e
    # Make into a thin panorama image (1800 wide, 80 high)
    pano = np.tile(inv.reshape(1, -1), (80, 1))
    img_p = Image.fromarray(np.stack([pano]*3, axis=2))
    draw_p = ImageDraw.Draw(img_p)
    draw_p.rectangle([CAND_A0, 0, CAND_A1, 79], outline=(0,200,0), width=2)
    draw_p.text((5, 5), f"z={z0}-{z1} ({z0*4.8/1000:.2f}-{z1*4.8/1000:.2f}mm) max-proj r=310", fill=(200,0,0))
    img_p.save(str(OUT_DIR / f"v5_pano_slab{z_slab_idx:02d}_z{z0}_{z1}.png"))
    print(f"  Saved v5_pano_slab{z_slab_idx:02d} (z={z0}-{z1}, {z0*4.8/1000:.2f}-{z1*4.8/1000:.2f}mm)")

# ─── 6. Save z-fraction profile as text ──────────────────────────────────────
with open(str(OUT_DIR / "v5_z_fraction_profile.txt"), "w") as f:
    f.write("z\tphys_mm\tcand_frac\tfull_frac\n")
    for zi in range(NZ):
        f.write(f"{zi}\t{zi*4.8/1000:.3f}\t{z_fracs_cand[zi]:.4f}\t{z_fracs_full[zi]:.4f}\n")
print("  Saved v5_z_fraction_profile.txt")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v5_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

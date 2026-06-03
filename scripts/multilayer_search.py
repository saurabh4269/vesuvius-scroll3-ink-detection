"""
Multi-layer letter search across all scroll layers.

The scroll has ~60 papyrus layers from r=310 (innermost) to r≈1000 (outer wall).
Each layer at radius r has a distinct ink surface. We systematically scan
radii 310-700 in steps of 15px (~72µm ≈ 1-2 scroll layers) at the
Zone 4 z-range where we know ink exists (z=1564-1751).

For each radius:
  1. Compute ink fraction at candidate angles (280-520) and full circle
  2. Run connected component analysis to find isolated letter-like structures
  3. Save panorama strip at that radius

Output: ranked list of (radius, angle, z-range) letter candidates.
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

HOME    = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/multilayer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

print("Loading level-2 zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
NY, NX = arr2.shape[1], arr2.shape[2]

# Zone 4 z-range (confirmed letter zone)
Z0_ABS, Z1_ABS = 1400, 1900   # wider window to catch full ink column
data = arr2[Z0_ABS:Z1_ABS, :, :][:]
nz   = data.shape[0]
print(f"  Slab: {data.shape}")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, NX-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# Radii to search: 310-700 in steps of 15
RADII = list(range(310, 701, 15))
print(f"\nSearching {len(RADII)} radii: {RADII[0]}-{RADII[-1]}")

# ─── Per-radius analysis ──────────────────────────────────────────────────────
results = []   # (radius, ink_frac, n_letter_cands, top_cand_info)

for r in RADII:
    u = sample_r(r)
    e = enh(u, sigma=0.3)

    # Ink fraction full circle (exclude outer wall saturated zone a>1100)
    frac_full = (u[:, :1100] > 0).mean()
    frac_inner = (u[:, 280:520] > 0).mean()

    # Connected components for letter detection
    thresh = (e > 100).astype(np.uint8)
    labeled_arr, n_comp = label(thresh)

    letter_cands = []
    for c in range(1, n_comp + 1):
        mask = (labeled_arr == c)
        pix  = mask.sum()
        if pix < 30 or pix > 50000:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        z_mm  = (rows.max() - rows.min() + 1) * 4.8 / 1000
        a_mm  = (cols.max() - cols.min() + 1) * ARC_UM / 1000
        a_ctr = int(cols.mean())
        # Letter criteria: 0.3-3mm in z, 0.2-2mm in arc, NOT outer wall
        if 0.3 <= z_mm <= 3.5 and 0.2 <= a_mm <= 2.5 and a_ctr < 1100:
            letter_cands.append({
                'a_center': a_ctr,
                'pixels': int(pix),
                'z_mm': round(z_mm, 3),
                'a_mm': round(a_mm, 3),
                'rows': (int(rows.min()), int(rows.max())),
                'cols': (int(cols.min()), int(cols.max())),
            })

    letter_cands.sort(key=lambda x: x['pixels'], reverse=True)
    results.append({
        'r': r,
        'r_mm': round(r * 4.8 / 1000, 3),
        'frac_full': round(frac_full, 4),
        'frac_inner': round(frac_inner, 4),
        'n_cands': len(letter_cands),
        'top_cands': letter_cands[:5],
    })

    status = f"r={r}px ({r*4.8/1000:.2f}mm): full_ink={frac_full:.3f}, cands={len(letter_cands)}"
    if letter_cands:
        top = letter_cands[0]
        status += f"  TOP: a={top['a_center']} {top['a_mm']:.2f}×{top['z_mm']:.2f}mm"
    print(f"  {status}")

# ─── Summary: best candidates ranked by size ─────────────────────────────────
print("\n=== TOP LETTER CANDIDATES ACROSS ALL LAYERS ===")
all_cands = []
for res in results:
    for c in res['top_cands']:
        all_cands.append({**c, 'r': res['r'], 'r_mm': res['r_mm']})
all_cands.sort(key=lambda x: x['pixels'], reverse=True)

print(f"Total candidates found: {len(all_cands)}")
print("\nTop 30 by pixel count:")
for c in all_cands[:30]:
    arc_pos = c['a_center'] * ARC_UM / 1000
    print(f"  r={c['r']}px({c['r_mm']:.2f}mm) a={c['a_center']}({arc_pos:.2f}mm) "
          f"size={c['a_mm']:.2f}×{c['z_mm']:.2f}mm px={c['pixels']}")

# ─── Panorama strip: all radii stacked ───────────────────────────────────────
print("\nGenerating multi-layer panorama strip...")
# For each radius, take z-max projection and stack as rows
PANO_ROW_H = 20
pano_rows = []
for res in results:
    r = res['r']
    u = sample_r(r)
    proj = u.max(axis=0).astype(np.uint8)
    proj_e = clahe.apply(proj)
    inv    = 255 - proj_e
    row    = np.tile(inv.reshape(1, -1), (PANO_ROW_H, 1))  # (20, 1800)
    pano_rows.append(row)

pano = np.vstack(pano_rows)  # (n_radii × 20, 1800)
img_pano = Image.fromarray(np.stack([pano]*3, axis=2))
draw     = ImageDraw.Draw(img_pano)

# Mark known candidate zone (a=280-520)
draw.rectangle([280, 0, 520, pano.shape[0]-1], outline=(0,200,0), width=1)

# Mark r=310 row (known letter)
r310_row = RADII.index(310) * PANO_ROW_H
draw.line([(0, r310_row), (1799, r310_row)], fill=(200,0,0), width=1)
draw.text((5, r310_row+2), "r=310 ← KNOWN LETTER", fill=(200,0,0))

# Mark all top candidates with red dots
for c in all_cands[:20]:
    row_y = RADII.index(c['r']) * PANO_ROW_H + PANO_ROW_H // 2
    draw.ellipse([c['a_center']-3, row_y-3, c['a_center']+3, row_y+3], fill=(255,100,0))

# Scale bar
draw.line([(20, pano.shape[0]-4), (220, pano.shape[0]-4)], fill=(200,0,0), width=2)
draw.text((20, pano.shape[0]-14), "1mm arc", fill=(200,0,0))

img_pano.save(str(OUT_DIR / "multilayer_panorama.png"))
print(f"  Saved multilayer_panorama.png ({pano.shape})")

# ─── High-zoom strips for top-5 NEW candidates (not r=310, a=280-520) ────────
print("\nGenerating zoom strips for top new candidates...")
seen_r310 = False
new_cands = []
for c in all_cands[:50]:
    if c['r'] == 310 and 280 <= c['a_center'] <= 520:
        continue   # skip already-known candidate
    new_cands.append(c)
    if len(new_cands) >= 8:
        break

print(f"  Top new candidates (excluding r=310 known):")
for i, c in enumerate(new_cands):
    arc_pos = c['a_center'] * ARC_UM / 1000
    print(f"  [{i}] r={c['r']}px a={c['a_center']}({arc_pos:.2f}mm) {c['a_mm']:.2f}×{c['z_mm']:.2f}mm px={c['pixels']}")

for i, c in enumerate(new_cands[:5]):
    r = c['r']
    a0 = max(0, c['cols'][0] - 50)
    a1 = min(N_ANGLES, c['cols'][1] + 50)
    z0 = max(0, c['rows'][0] - 20)
    z1 = min(nz, c['rows'][1] + 20)

    u = sample_r(r)
    crop = u[z0:z1, a0:a1]
    e_crop = enh(crop, sigma=0.2)
    inv_crop = 255 - e_crop

    zoom = np.repeat(np.repeat(inv_crop, 6, axis=0), 4, axis=1)
    img_z = Image.fromarray(np.stack([zoom]*3, axis=2))
    d_z   = ImageDraw.Draw(img_z)
    arc_pos = c['a_center'] * ARC_UM / 1000
    z_abs0 = Z0_ABS + z0
    d_z.text((4, 4), f"r={r}px={r*4.8/1000:.2f}mm | a={c['a_center']}({arc_pos:.2f}mm) | z={z_abs0}-{Z0_ABS+z1} | {c['a_mm']:.2f}x{c['z_mm']:.2f}mm", fill=(200,0,0))
    # 1mm scale bar
    sb = int(1000/ARC_UM)*4
    H, W = zoom.shape
    if sb < W:
        d_z.line([(8, H-20),(8+sb, H-20)], fill=(200,0,0), width=3)
        d_z.text((8, H-36), "1mm arc", fill=(200,0,0))
    img_z.save(str(OUT_DIR / f"new_cand_{i:02d}_r{r}_a{c['a_center']}.png"))
    print(f"  Saved new_cand_{i:02d}_r{r}_a{c['a_center']}.png")

print("\nDONE.")

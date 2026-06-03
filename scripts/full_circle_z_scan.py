"""
Full-circle scan at the key z-level where our letter candidate appears.

Strategy:
- z=164-351 relative (abs 1564-1751) = main ink zone
- Sample full circle (1800 angles) at r=310
- Find all isolated ink features and produce annotated panorama
- Also check center thin-line feature at angle=600-700

Goal: Find whether there are more letter-like features on the same text "line"
(same z-range = same horizontal position on the unrolled scroll).
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
z_lo_l2  = 1400
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

print("Loading zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")

# Load just the main zone z=164-471 (zone1+zone2+some buffer)
Z0_REL, Z1_REL = 140, 500
data = arr2[z_lo_l2 + Z0_REL : z_lo_l2 + Z1_REL, :, :]
nz, ny, nx = data.shape
print(f"  Loaded {data.shape} (abs z={z_lo_l2+Z0_REL}-{z_lo_l2+Z1_REL})")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

print("\nSampling r=310 full circle...")
u310 = sample_r(310)   # (360, 1800)
e310 = enh(u310, sigma=0.3)

# ─── 1. Find connected components in the enhanced image ───────────────────────
print("\n1. Connected component analysis...")
# Threshold at 50% of max to find ink blobs
thresh = (e310 > 100).astype(np.uint8)
# Label connected components
labeled, n_comp = label(thresh)
print(f"  Found {n_comp} connected components")

# Measure each component
comp_stats = []
for c in range(1, n_comp + 1):
    mask = (labeled == c)
    pixels = mask.sum()
    if pixels < 20:
        continue
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    z_span  = rows.max() - rows.min() + 1
    a_span  = cols.max() - cols.min() + 1
    a_center = cols.mean()
    z_center = rows.mean()
    # Physical size
    z_mm = z_span * 4.8 / 1000
    a_mm = a_span * ARC_UM / 1000
    comp_stats.append({
        'id': c,
        'pixels': pixels,
        'z_span': z_span, 'a_span': a_span,
        'z_mm': z_mm, 'a_mm': a_mm,
        'z_center': z_center, 'a_center': a_center,
        'cols': (cols.min(), cols.max()),
        'rows': (rows.min(), rows.max()),
        'aspect': z_mm / max(a_mm, 0.001),
    })

# Sort by pixel count (largest first)
comp_stats.sort(key=lambda x: x['pixels'], reverse=True)
print(f"  {len(comp_stats)} components with ≥20px")
print(f"  Top 20:")
for s in comp_stats[:20]:
    arc_pos_mm = s['a_center'] * ARC_UM / 1000
    print(f"    a={int(s['a_center'])} ({arc_pos_mm:.2f}mm), pixels={s['pixels']}, "
          f"size={s['a_mm']:.2f}×{s['z_mm']:.2f}mm, aspect={s['aspect']:.2f}")

# ─── 2. Identify letter-like components ──────────────────────────────────────
# Letter criteria: 0.3-3mm in both dimensions, not too elongated in angle
print("\n2. Letter candidates (0.3-3mm in z, 0.2-2mm in arc, away from edges)...")
letter_cands = []
for s in comp_stats:
    # Filter: reasonable size
    if s['z_mm'] < 0.3 or s['z_mm'] > 4.0:
        continue
    if s['a_mm'] < 0.2 or s['a_mm'] > 2.5:
        continue
    # Not at outer wall (angle>1100 often saturated)
    if s['a_center'] > 1100:
        continue
    letter_cands.append(s)

print(f"  {len(letter_cands)} letter-like candidates:")
for s in letter_cands:
    arc_mm = s['a_center'] * ARC_UM / 1000
    print(f"    a_center={int(s['a_center'])} ({arc_mm:.2f}mm), "
          f"size={s['a_mm']:.2f}×{s['z_mm']:.2f}mm, pixels={s['pixels']}")

# ─── 3. Full-circle panorama with all components highlighted ─────────────────
print("\n3. Generating annotated full-circle panorama...")
inv310 = 255 - e310
# 2× zoom in z, 1× angle → (720, 1800)
pano = np.repeat(inv310, 2, axis=0)
img_pano = Image.fromarray(np.stack([pano]*3, axis=2))
draw_pano = ImageDraw.Draw(img_pano)

H_p = pano.shape[0]

# Draw all components with color coding:
# GREEN = letter-like candidates
# RED = too large (likely saturation)
# GRAY = too small

letter_set = {s['id'] for s in letter_cands}
for s in comp_stats[:50]:  # top-50
    r0, r1 = s['rows']
    c0, c1 = s['cols']
    color = (0, 200, 0) if s['id'] in letter_set else (180, 0, 0)
    if s['pixels'] < 100:
        color = (128, 128, 128)
    # Rectangle on panorama (z×2 zoom)
    draw_pano.rectangle([c0, r0*2, c1, r1*2], outline=color, width=1)

# Mark our known candidate (a=280-520)
draw_pano.rectangle([280, 0, 520, H_p], outline=(0, 255, 255), width=3)
draw_pano.text((280, 2), "MAIN CAND", fill=(0, 200, 200))

# Scale bar
draw_pano.line([(20, H_p-10), (220, H_p-10)], fill=(200,0,0), width=3)
draw_pano.text((20, H_p-25), "1mm", fill=(200,0,0))

img_pano.save(str(OUT_DIR / "v4_fullcircle_annotated.png"))
print(f"  Saved v4_fullcircle_annotated.png ({pano.shape})")

# ─── 4. High-zoom view of center thin-line feature ────────────────────────────
print("\n4. Center thin-line zoom (angle=620-720)...")
CTR_A0, CTR_A1 = 580, 760
ctr_block = u310[:, CTR_A0:CTR_A1]
ctr_enh   = enh(ctr_block, sigma=0.2)
ctr_inv   = 255 - ctr_enh

# 8× zoom in z, 5× in angle
ctr_big = np.repeat(np.repeat(ctr_inv, 8, axis=0), 5, axis=1)
img_ctr = Image.fromarray(np.stack([ctr_big]*3, axis=2))
draw_ctr = ImageDraw.Draw(img_ctr)
# Physical annotation
phys_z = nz * 4.8 / 1000
draw_ctr.text((5, 5), f"PHerc.332 | angle={CTR_A0}-{CTR_A1} | r=310 | z={z_lo_l2+Z0_REL}-{z_lo_l2+Z1_REL}", fill=(200,0,0))
# Scale bar: 1mm in z = 208px × 8 = 1664 display px (vertical)
sb_z = int(1000/4.8) * 8
H_ctr = ctr_big.shape[0]
if sb_z < H_ctr:
    draw_ctr.line([(5, H_ctr-40-sb_z), (5, H_ctr-40)], fill=(200,0,0), width=4)
    draw_ctr.text((12, H_ctr-40-sb_z//2), "1mm (z)", fill=(200,0,0))
img_ctr.save(str(OUT_DIR / "v4_center_thinline_zoom.png"))
print(f"  Saved v4_center_thinline_zoom.png ({ctr_big.shape})")

# ─── 5. Side-by-side: main candidate vs center thin-line ─────────────────────
print("\n5. Side-by-side: main candidate vs center thin line (same z range)...")
MAIN_A0, MAIN_A1 = 280, 520

def make_panel_6x3(a0, a1, label_text, sigma=0.25):
    block = u310[:, a0:a1]
    e     = enh(block, sigma=sigma)
    inv   = 255 - e
    zoomed = np.repeat(np.repeat(inv, 6, axis=0), 3, axis=1)
    rgb = np.stack([zoomed]*3, axis=2)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), label_text, fill=(200, 0, 0))
    # 1mm horizontal scale bar
    sb = int(1000/ARC_UM) * 3
    W = zoomed.shape[1]
    H = zoomed.shape[0]
    draw.line([(4, H-20), (4+sb, H-20)], fill=(200,0,0), width=3)
    draw.text((4, H-38), "1mm arc", fill=(200,0,0))
    return np.array(img)[:, :, 0]

p1 = make_panel_6x3(MAIN_A0, MAIN_A1, f"a={MAIN_A0}-{MAIN_A1} MAIN CANDIDATE")
p2 = make_panel_6x3(CTR_A0,  CTR_A1,  f"a={CTR_A0}-{CTR_A1} CENTER REGION")
sep = np.ones((p1.shape[0], 8), dtype=np.uint8) * 180
sbs = np.hstack([p1, sep, p2])
Image.fromarray(sbs).save(str(OUT_DIR / "v4_sbs_main_vs_center.png"))
print(f"  Saved v4_sbs_main_vs_center.png ({sbs.shape})")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v4_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

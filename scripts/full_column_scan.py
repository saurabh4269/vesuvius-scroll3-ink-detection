"""
Full-column scan: extend z-range to see complete ink column structure.

From search_context_r310.png we saw that the ink structure extends below CAND_Z1=330.
This script:
1. Scans full z=0-700 (level-2 z=1400-2100) at r=310 to find the complete extent
2. Generates a tall vertical strip at angle=280-520 (known candidate region)
3. Also generates high-zoom views of the center parallel-line region (angle=600-750)
4. Saves everything to letter_report/v3_*

Physical reminder:
  z-axis:  4.8 µm/px → 700px = 3.36mm total scan height
  angle:   5.0 µm/px at r=310
  r=310:   1.488mm physical radius = innermost ink surface
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
z_lo_l2  = 1400
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

print("Loading zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
data = arr2[z_lo_l2:, :, :]   # (700, 986, 986)
nz, ny, nx = data.shape
print(f"  {data.shape}")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

print("Sampling r=310 full z-range...")
u310 = sample_r(310)   # (700, 1800)
print(f"  ink>0: {(u310>0).mean()*100:.1f}%")

# ─── 1. Full vertical column at known candidate angle range (280-520) ──────────
print("\n1. Full column at angle=280-520 (known candidate)...")
CAND_A0, CAND_A1 = 280, 520
col_block = u310[:, CAND_A0:CAND_A1]   # (700, 240)
col_enh   = enh(col_block, sigma=0.3)
col_inv   = 255 - col_enh

# 4× zoom in z, 3× in angle → (2800, 720)
col_zoom = np.repeat(np.repeat(col_inv, 4, axis=0), 3, axis=1)

# Add horizontal scale bar (1mm = 200 arc-px at 5µm/px; at 3× zoom = 600px)
# Add vertical scale bar (1mm = 208 z-px at 4.8µm/px; at 4× zoom = 832px)
img_col = Image.fromarray(np.stack([col_zoom]*3, axis=2))
draw    = ImageDraw.Draw(img_col)
H, W    = col_zoom.shape

# Mark the original candidate box (CAND_Z0=100, CAND_Z1=330)
CAND_Z0, CAND_Z1 = 100, 330
draw.rectangle([0, CAND_Z0*4, W-1, CAND_Z1*4], outline=(255,0,0), width=3)
draw.text((5, CAND_Z0*4 + 5), "ORIGINAL CANDIDATE", fill=(255,0,0))

# z-axis annotation
for z_mm in range(0, 4):
    z_px = int(z_mm * 1000 / 4.8) * 4
    if z_px < H:
        draw.line([(0, z_px), (15, z_px)], fill=(200,0,0), width=2)
        draw.text((18, z_px - 8), f"{z_mm:.0f}mm", fill=(200,0,0))

# Scale bar bottom-right: 1mm horizontal
sb_h = int(1000/ARC_UM) * 3   # 600px
draw.line([(W-sb_h-10, H-20), (W-10, H-20)], fill=(200,0,0), width=4)
draw.text((W-sb_h-10, H-40), "1 mm (arc)", fill=(200,0,0))

img_col.save(str(OUT_DIR / "v3_full_column_280_520.png"))
print(f"  Saved v3_full_column_280_520.png  ({col_zoom.shape})")

# ─── 2. z-profile to find full extent of the ink column ───────────────────────
print("\n2. z-profile analysis...")
z_ink_fracs = (u310[:, CAND_A0:CAND_A1] > 0).mean(axis=1)
# Find contiguous zones with ink > 5%
in_zone = False
zones = []
start = 0
for zi, f in enumerate(z_ink_fracs):
    if not in_zone and f > 0.05:
        in_zone = True
        start = zi
    elif in_zone and f <= 0.02:
        zones.append((start, zi, zi-start))
        in_zone = False
if in_zone:
    zones.append((start, len(z_ink_fracs), len(z_ink_fracs)-start))

print(f"  Ink zones at angle 280-520:")
for z0, z1, length in zones:
    phys_z0 = (z_lo_l2 + z0) * 4.8/1000
    phys_z1 = (z_lo_l2 + z1) * 4.8/1000
    print(f"    z={z0}-{z1} (abs {z_lo_l2+z0}-{z_lo_l2+z1}) = {phys_z0:.2f}-{phys_z1:.2f}mm, len={length}px={length*4.8/1000:.2f}mm, frac_max={z_ink_fracs[z0:z1].max():.2f}")

# ─── 3. Center parallel-line region (angle=600-750) at high zoom ──────────────
print("\n3. Center parallel-line region (angle=600-750)...")
CTR_A0, CTR_A1 = 550, 800   # wider window

# z=0-280 only (strips 0+1, where we saw features)
ctr_block = u310[:280, CTR_A0:CTR_A1]   # (280, 250)
ctr_enh   = enh(ctr_block, sigma=0.3)
ctr_inv   = 255 - ctr_enh

# 6× zoom in z, 4× in angle
ctr_zoom = np.repeat(np.repeat(ctr_inv, 6, axis=0), 4, axis=1)  # (1680, 1000)

img_ctr = Image.fromarray(np.stack([ctr_zoom]*3, axis=2))
draw_ctr = ImageDraw.Draw(img_ctr)
# Scale bar
sb_arc = int(1000/ARC_UM) * 4  # 1mm at 4× zoom = 800px
draw_ctr.line([(20, ctr_zoom.shape[0]-30), (20+sb_arc, ctr_zoom.shape[0]-30)], fill=(200,0,0), width=4)
draw_ctr.text((20, ctr_zoom.shape[0]-55), "1 mm (arc)", fill=(200,0,0))
draw_ctr.text((5, 5), f"PHerc.332 | angle={CTR_A0}-{CTR_A1} | r=310 | z={z_lo_l2}-{z_lo_l2+280}", fill=(200,0,0))

img_ctr.save(str(OUT_DIR / "v3_center_region_600_750.png"))
print(f"  Saved v3_center_region_600_750.png  ({ctr_zoom.shape})")

# ─── 4. Also look at the center region for full z-range ───────────────────────
print("\n4. Center region full z-range (z=0-700, angle=550-800)...")
ctr_full  = u310[:, CTR_A0:CTR_A1]   # (700, 250)
ctr_f_enh = enh(ctr_full, sigma=0.3)
ctr_f_inv = 255 - ctr_f_enh

# 3× in z, 4× in angle → (2100, 1000)
ctr_f_zoom = np.repeat(np.repeat(ctr_f_inv, 3, axis=0), 4, axis=1)

img_cf = Image.fromarray(np.stack([ctr_f_zoom]*3, axis=2))
draw_cf = ImageDraw.Draw(img_cf)
H2, W2 = ctr_f_zoom.shape
for z_mm in range(0, 4):
    z_px = int(z_mm * 1000 / 4.8) * 3
    if z_px < H2:
        draw_cf.line([(0, z_px), (15, z_px)], fill=(200,0,0), width=2)
        draw_cf.text((18, z_px - 8), f"{z_mm:.0f}mm", fill=(200,0,0))
img_cf.save(str(OUT_DIR / "v3_center_full_z.png"))
print(f"  Saved v3_center_full_z.png  ({ctr_f_zoom.shape})")

# ─── 5. Side-by-side: candidate region vs center region (z=0-280) ─────────────
print("\n5. Side-by-side comparison at z=0-280...")
# Both at same zoom: 6× z, 3× angle
def make_panel(a0, a1, label, sigma=0.25):
    block = u310[:280, a0:a1]
    e     = enh(block, sigma=sigma)
    inv   = 255 - e
    zoomed = np.repeat(np.repeat(inv, 6, axis=0), 3, axis=1)
    rgb = np.stack([zoomed]*3, axis=2)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), label, fill=(200, 0, 0))
    return np.array(img)[:, :, 0]

p1 = make_panel(CAND_A0, CAND_A1, f"a={CAND_A0}-{CAND_A1} CANDIDATE")
p2 = make_panel(CTR_A0, CTR_A1, f"a={CTR_A0}-{CTR_A1} CENTER")
sep = np.ones((p1.shape[0], 8), dtype=np.uint8) * 180
sbs = np.hstack([p1, sep, p2])
Image.fromarray(sbs).save(str(OUT_DIR / "v3_sidebyside_z0_280.png"))
print(f"  Saved v3_sidebyside_z0_280.png  ({sbs.shape})")

# ─── 6. Summary ───────────────────────────────────────────────────────────────
print("\nDONE.")
for f in sorted(OUT_DIR.glob("v3_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

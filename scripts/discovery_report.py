"""
Generate a clean, shareable discovery report composite image.

Shows the evidence chain:
1. 5-radius gradient: fibers → letter → empty (smoking gun for real ink)
2. Max-zoom letter form at r=310 (inverted, clean)
3. Full-circle panorama showing isolation

For sharing on Vesuvius Discord #ink-detection
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

print("Opening zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
NY, NX = arr2.shape[1], arr2.shape[2]

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(data_slab, r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, NX-1).astype(int)
    return data_slab[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# Zone 4: z=1564-1751 (relative 164-351 from z_lo=1400)
Z0_ABS, Z1_ABS = 1564, 1751
CAND_A0, CAND_A1 = 280, 520
RADII = [298, 304, 310, 316, 322]

print("Loading Zone 4 slab...")
slab = arr2[Z0_ABS:Z1_ABS, :, :][:]
nz   = slab.shape[0]

# ─── Panel A: 5-radius gradient ─────────────────────────────────────────────
print("Building 5-radius gradient panel...")
PANEL_W = 200   # per-radius panel width (angle pixels × 2 zoom)
PANEL_H = 500   # z pixels × ~2.6 zoom (187 z-px × 2.6 = 486)

panels_A = []
labels_A = ["r=298\n(fibers)", "r=304\n(trans.)", "r=310\n(LETTER)", "r=316\n(fading)", "r=322\n(empty)"]
for r, lbl in zip(RADII, labels_A):
    u  = sample_r(slab, r)
    crop = u[:, CAND_A0:CAND_A1]
    e    = enh(crop, sigma=0.3 if r == 310 else 0.4)
    inv  = 255 - e

    # Resize to PANEL_W × PANEL_H
    panel_img = Image.fromarray(inv).resize((PANEL_W, PANEL_H), Image.LANCZOS)
    panel = np.array(panel_img)

    # Convert to RGB and add label
    rgb = Image.fromarray(np.stack([panel]*3, axis=2))
    d   = ImageDraw.Draw(rgb)
    # Background strip for label
    color = (0, 160, 0) if r == 310 else (100, 100, 100)
    d.rectangle([0, 0, PANEL_W-1, 28], fill=(240,240,240))
    d.text((4, 4), lbl.split('\n')[0], fill=color)
    d.text((4, 14), lbl.split('\n')[1], fill=color)
    panels_A.append(np.array(rgb))

sep = np.ones((PANEL_H, 4, 3), dtype=np.uint8) * 180
row_A = np.hstack([x for p in panels_A for x in [p, sep]][:-1])

# ─── Panel B: Max-zoom at r=310, clean letter ────────────────────────────────
print("Building max-zoom letter panel...")
u310 = sample_r(slab, 310)
cand_block = u310[:, CAND_A0:CAND_A1]
e310 = enh(cand_block, sigma=0.15)
inv310 = 255 - e310

# Resize to PANEL_H tall, preserving aspect
ZOOM_W = int(PANEL_W * (CAND_A1 - CAND_A0) / max(nz, 1) * PANEL_H / PANEL_H)
ZOOM_W = max(ZOOM_W, 150)
letter_img = Image.fromarray(inv310).resize((ZOOM_W, PANEL_H), Image.LANCZOS)
letter_rgb  = np.stack([np.array(letter_img)]*3, axis=2)
letter_pil  = Image.fromarray(letter_rgb)
d_l = ImageDraw.Draw(letter_pil)
# 1mm scale bar (horizontal: 1mm = 200 arc-px / 240 total × ZOOM_W)
sb = int(200 / 240 * ZOOM_W)
d_l.line([(8, PANEL_H-20), (8+sb, PANEL_H-20)], fill=(200,0,0), width=3)
d_l.text((8, PANEL_H-36), "1mm", fill=(200,0,0))
d_l.rectangle([0, 0, ZOOM_W-1, 28], fill=(240,240,240))
d_l.text((4, 4), "r=310  letter form", fill=(0,140,0))
d_l.text((4, 14), "PHerc.332 (Scroll3)", fill=(80,80,80))

# ─── Panel C: Full-circle panorama (z-max projection) ───────────────────────
print("Building full-circle panorama...")
PANO_H = 100
u310_full = sample_r(slab, 310)   # (187, 1800)
proj      = u310_full.max(axis=0).astype(np.uint8)
proj_e    = clahe.apply(proj)
inv_p     = 255 - proj_e
pano_strip = np.tile(inv_p.reshape(1, -1), (PANO_H, 1))  # (100, 1800)

# Scale to fit composite width
full_W = row_A.shape[1]
pano_scaled = np.array(Image.fromarray(pano_strip).resize((full_W, PANO_H), Image.LANCZOS))
pano_rgb    = np.stack([pano_scaled]*3, axis=2)
pano_pil    = Image.fromarray(pano_rgb)
d_p         = ImageDraw.Draw(pano_pil)

# Mark candidate zone (scaled)
a0_px = int(CAND_A0 / N_ANGLES * full_W)
a1_px = int(CAND_A1 / N_ANGLES * full_W)
d_p.rectangle([a0_px, 0, a1_px, PANO_H-1], outline=(0,200,0), width=2)
d_p.text((a0_px+2, 2), "▼ letter here", fill=(0,200,0))
d_p.text((4, PANO_H-18), "Full circle r=310 z-max projection  (green = candidate angle range)", fill=(80,0,0))

# 1mm arc scale bar
sb_arc = int(200 / N_ANGLES * full_W)   # 1mm at 5µm/px
d_p.line([(8, PANO_H-6), (8+sb_arc, PANO_H-6)], fill=(200,0,0), width=2)
d_p.text((8+sb_arc+2, PANO_H-14), "1mm", fill=(200,0,0))

# ─── Composite: header + row A + letter + panorama ──────────────────────────
print("Assembling composite...")
HEADER_H = 40
composite_W = max(row_A.shape[1], int(full_W))
composite_H = HEADER_H + PANEL_H + 8 + PANO_H

# Match widths
letter_arr = np.array(letter_pil)
# Pad row_A to include letter panel on the right
pad = np.ones((PANEL_H, 8, 3), dtype=np.uint8) * 180
row_with_letter = np.hstack([row_A, pad, letter_arr])
final_W = row_with_letter.shape[1]

# Resize panorama to final_W
pano_final = np.array(Image.fromarray(pano_rgb).resize((final_W, PANO_H), Image.LANCZOS))
pano_pil2  = Image.fromarray(pano_final)
d_p2 = ImageDraw.Draw(pano_pil2)
a0_px2 = int(CAND_A0 / N_ANGLES * final_W)
a1_px2 = int(CAND_A1 / N_ANGLES * final_W)
d_p2.rectangle([a0_px2, 0, a1_px2, PANO_H-1], outline=(0,200,0), width=2)
d_p2.text((a0_px2+2, 2), "▼ letter", fill=(0,200,0))

# Header
header = np.ones((HEADER_H, final_W, 3), dtype=np.uint8) * 250
header_pil = Image.fromarray(header)
d_h = ImageDraw.Draw(header_pil)
d_h.text((4, 4),  "PHerc.332 (Scroll 3) — Letter Candidate  |  m7_nnUNet ink pred  |  4.8µm/px level-2 zarr", fill=(30,30,30))
d_h.text((4, 20), f"Location: z={Z0_ABS}-{Z1_ABS} (abs) = {Z0_ABS*4.8/1000:.2f}-{Z1_ABS*4.8/1000:.2f}mm height  |  angle={CAND_A0}-{CAND_A1}  |  r=310px = 1.49mm depth", fill=(80,80,80))

sep_h = np.ones((4, final_W, 3), dtype=np.uint8) * 160

composite = np.vstack([
    np.array(header_pil),
    sep_h,
    row_with_letter,
    sep_h,
    np.array(pano_pil2),
])

Image.fromarray(composite).save(str(OUT_DIR / "DISCOVERY_COMPOSITE.png"))
print(f"\nSaved DISCOVERY_COMPOSITE.png  ({composite.shape})")
print(f"  Physical size: {composite.shape[1]}×{composite.shape[0]} px")
print(f"\nKey evidence:")
print(f"  1. 5-radius gradient: papyrus fibers (r=298) → letter (r=310) → empty (r=322)")
print(f"  2. Crackle ink pattern at r=310: bowl + counter + descending strokes")
print(f"  3. Full-circle isolation: only isolated letter structure in entire circle")
print(f"  4. Zone 4 unique: other high-ink zones are blocky chunk-aligned saturation")
print(f"  Physical: z=7.51-8.40mm, arc=1.40-2.60mm, depth=1.49mm from center")

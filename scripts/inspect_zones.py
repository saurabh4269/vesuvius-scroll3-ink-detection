"""
High-zoom inspection of Zones 1-3 at angle=280-520.

Zones from full_scroll_scan:
  Zone 1: z=192-310   (0.92-1.49mm) peak=0.98  — solid black in strip
  Zone 2: z=692-883   (3.32-4.24mm) peak=0.55
  Zone 3: z=1146-1312 (5.50-6.30mm) peak=0.72
  Zone 4: z=1564-1751 (7.51-8.40mm) peak=0.24  — our known letter candidate

Strategy:
  - CLAHE at clipLimit=2 (gentler) to prevent whiteout on dense regions
  - Try multiple radii: 298, 304, 310, 316, 322
  - Side-by-side 5-radius comparison for each zone
  - Also try r=350-420 (outer layers) for zones 1-3
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

print("Opening zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
NY, NX = arr2.shape[1], arr2.shape[2]

# CLAHE variants
clahe_soft = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_hard = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(data_slab, r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, NX-1).astype(int)
    return data_slab[:, ys, xs].astype(np.uint8)

def enh(img, soft=False):
    sm = gaussian_filter(img.astype(float), sigma=0.3)
    c  = clahe_soft if soft else clahe_hard
    return c.apply(np.clip(sm, 0, 255).astype(np.uint8))

# Zone definitions
ZONES = [
    ("z1", 192,  310, "Zone1 0.92-1.49mm peak=0.98"),
    ("z2", 692,  883, "Zone2 3.32-4.24mm peak=0.55"),
    ("z3", 1146, 1312,"Zone3 5.50-6.30mm peak=0.72"),
    ("z4", 1564, 1751,"Zone4 7.51-8.40mm peak=0.24 [KNOWN CAND]"),
]

CAND_A0, CAND_A1 = 280, 520
RADII_INNER = [298, 304, 310, 316, 322]
RADII_OUTER = [340, 360, 380, 400]   # check outer layers for zones 1-3

def make_strip_row(data_slab, r_px, a0, a1, soft=False):
    u = sample_r(data_slab, r_px)
    crop = u[:, a0:a1]
    e = enh(crop, soft=soft)
    inv = 255 - e
    # 6× z-zoom, 2× angle-zoom
    return np.repeat(np.repeat(inv, 6, axis=0), 2, axis=1)

for zone_id, z0, z1, label in ZONES:
    print(f"\n--- {label} ---")
    slab = arr2[z0:z1, :, :][:]   # full spatial slab

    # ── Inner radii comparison (5 panels) ────────────────────────────────────
    panels = []
    for r in RADII_INNER:
        row = make_strip_row(slab, r, CAND_A0, CAND_A1, soft=(zone_id in ("z1","z2","z3")))
        H, W = row.shape
        rgb = np.stack([row]*3, axis=2)
        img = Image.fromarray(rgb)
        d   = ImageDraw.Draw(img)
        d.text((3, 3), f"r={r}", fill=(200,0,0))
        panels.append(np.array(img)[:, :, 0])

    sep = np.ones((panels[0].shape[0], 6), dtype=np.uint8) * 160
    combo = np.hstack([x for p in panels for x in [p, sep]][:-1])
    phys_label = label.split()[0]
    Image.fromarray(combo).save(str(OUT_DIR / f"v6_{zone_id}_inner_radii.png"))
    print(f"  Saved v6_{zone_id}_inner_radii.png  ({combo.shape})")

    # ── Outer radii (only for zones 1-3, to check outer layers) ─────────────
    if zone_id != "z4":
        panels_out = []
        for r in RADII_OUTER:
            row = make_strip_row(slab, r, CAND_A0, CAND_A1, soft=True)
            rgb = np.stack([row]*3, axis=2)
            img = Image.fromarray(rgb)
            d   = ImageDraw.Draw(img)
            d.text((3, 3), f"r={r}", fill=(200,0,0))
            panels_out.append(np.array(img)[:, :, 0])
        sep = np.ones((panels_out[0].shape[0], 6), dtype=np.uint8) * 160
        combo_out = np.hstack([x for p in panels_out for x in [p, sep]][:-1])
        Image.fromarray(combo_out).save(str(OUT_DIR / f"v6_{zone_id}_outer_radii.png"))
        print(f"  Saved v6_{zone_id}_outer_radii.png  ({combo_out.shape})")

    # ── Full-circle panorama (z-max projection at r=310) ─────────────────────
    u310 = sample_r(slab, 310)
    proj = u310.max(axis=0).astype(np.uint8)
    proj_e = clahe_hard.apply(proj)
    inv_p  = 255 - proj_e
    pano   = np.tile(inv_p.reshape(1, -1), (40, 1))
    img_p  = Image.fromarray(np.stack([pano]*3, axis=2))
    d_p    = ImageDraw.Draw(img_p)
    d_p.rectangle([CAND_A0, 0, CAND_A1, 39], outline=(0,200,0), width=2)
    d_p.text((5, 5), f"{label} | r=310 max-proj", fill=(200,0,0))
    # 1mm arc scale bar
    d_p.line([(20, 32), (220, 32)], fill=(200,0,0), width=2)
    d_p.text((20, 22), "1mm", fill=(200,0,0))
    img_p.save(str(OUT_DIR / f"v6_{zone_id}_panorama.png"))
    print(f"  Saved v6_{zone_id}_panorama.png")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v6_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

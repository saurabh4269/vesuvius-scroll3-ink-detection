"""
Generate a shareable report image for the letter candidate found in strip1A.

Location: z=1540-1680 (level-2), angle ~350-450, radius ~298-315 px
Physical: z=7.4-8.1mm height, arc=1.75-2.25mm from angle=0, depth~1.4-1.6mm from center

Goals:
1. Create clean inverted-color image (dark letters on light background)
2. Add scale bar (1mm)
3. Search ±200px in z and ±200px in angle for more letter candidates
4. Show 3-radius comparison at best candidate location
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
z_lo_l2  = 1400
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0   # µm per angle pixel at r=298

print("Loading zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
data = arr2[z_lo_l2:, :, :]
nz, ny, nx = data.shape
print(f"  {data.shape}")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.3):
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── 1. Characterize the candidate ───────────────────────────────────────────
# From previous analysis: the structure appears at angle~350-450 in strip1 (z=140-280)
CAND_Z0, CAND_Z1 = 100, 330   # relative z (level-2 z = z_lo + this)
CAND_A0, CAND_A1 = 280, 520   # angle range (the right edge of strip1A crop was 100-450 and structure was at ~350-450)
CAND_R = 310                   # best radius

print(f"\nCandidate region: z={z_lo_l2+CAND_Z0}-{z_lo_l2+CAND_Z1}, angle={CAND_A0}-{CAND_A1}, r={CAND_R}")
phys_z0 = (z_lo_l2 + CAND_Z0) * 4.8 / 1000
phys_z1 = (z_lo_l2 + CAND_Z1) * 4.8 / 1000
phys_a0 = CAND_A0 * ARC_UM / 1000
phys_a1 = CAND_A1 * ARC_UM / 1000
print(f"  Physical: z={phys_z0:.2f}-{phys_z1:.2f}mm, arc={phys_a0:.2f}-{phys_a1:.2f}mm")

# ─── 2. High-quality crop at 3 radii ─────────────────────────────────────────
print("\nGenerating 3-radius comparison at candidate...")
radii_cmp = [298, 305, 310]
panels = []
for r in radii_cmp:
    ys = np.clip(cy + r * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r * np.cos(angles), 0, nx-1).astype(int)
    block = data[CAND_Z0:CAND_Z1, ys[CAND_A0:CAND_A1], 0]   # This won't work — fix below
    # Correct approach: sample the full unrolled, then slice
    unrolled_r = data[CAND_Z0:CAND_Z1, ys, xs]  # (nz_sub, N_ANGLES)
    crop = unrolled_r[:, CAND_A0:CAND_A1].astype(np.uint8)

    e = enh(crop, sigma=0.2)
    # Invert: white background, dark letters
    inv = 255 - e

    # 6× zoom in z, 3× in angle
    zoomed = np.repeat(np.repeat(inv, 6, axis=0), 3, axis=1)
    panels.append(zoomed)

# Add labels and separators
H = panels[0].shape[0]
sep = np.ones((H, 8), dtype=np.uint8) * 200
labeled = []
for i, (panel, r) in enumerate(zip(panels, radii_cmp)):
    # Convert to RGB to add colored label
    rgb = np.stack([panel, panel, panel], axis=2)
    # Draw radius label at top
    img_pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img_pil)
    draw.text((5, 5), f"r={r}px ({r*4.8/1000:.2f}mm)", fill=(200, 0, 0))
    labeled.append(np.array(img_pil)[:,:,0])  # back to grayscale for now

combined = np.hstack([labeled[0], sep, labeled[1], sep, labeled[2]])
Image.fromarray(combined).save(str(OUT_DIR / "candidate_3radius.png"))
print(f"  Saved candidate_3radius.png ({combined.shape})")

# ─── 3. Full-context search: wider region around candidate ───────────────────
print("\nSearching for more letter candidates in wider region...")
# z = CAND_Z0-200 to CAND_Z1+200 (±1mm around candidate z-center)
SEARCH_Z0 = max(0, CAND_Z0 - 100)
SEARCH_Z1 = min(nz, CAND_Z1 + 100)
SEARCH_A0 = max(0, CAND_A0 - 200)
SEARCH_A1 = min(N_ANGLES, CAND_A1 + 400)

u310_search = sample_r(310)[SEARCH_Z0:SEARCH_Z1, SEARCH_A0:SEARCH_A1]
e_search = enh(u310_search.astype(np.uint8), sigma=0.3)
inv_search = 255 - e_search

# 4× zoom both axes
zoom4 = np.repeat(np.repeat(inv_search, 4, axis=0), 4, axis=1)

# Mark the known candidate region on this image
img_ctx = Image.fromarray(np.stack([zoom4]*3, axis=2))
draw = ImageDraw.Draw(img_ctx)
# Candidate bounds in the search crop (in zoomed space)
box_y0 = (CAND_Z0 - SEARCH_Z0) * 4
box_y1 = (CAND_Z1 - SEARCH_Z0) * 4
box_x0 = (CAND_A0 - SEARCH_A0) * 4
box_x1 = (CAND_A1 - SEARCH_A0) * 4
draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline=(255, 0, 0), width=3)
draw.text((box_x0, box_y0 - 20), "KNOWN CANDIDATE", fill=(255, 0, 0))

# Add scale bar: 1mm = 1000µm / 4.8µm/px = 208px → at 4× zoom = 833px
scale_bar_px = int(1000 / ARC_UM * 4)
draw.line([(20, zoom4.shape[0]-30), (20+scale_bar_px, zoom4.shape[0]-30)], fill=(255,0,0), width=4)
draw.text((20, zoom4.shape[0]-50), "1 mm", fill=(255,0,0))

img_ctx.save(str(OUT_DIR / "search_context_r310.png"))
print(f"  Saved search_context_r310.png ({zoom4.shape}) — wide context with 1mm scale bar")
phys_search_z = (SEARCH_Z1 - SEARCH_Z0) * 4.8 / 1000
phys_search_a = (SEARCH_A1 - SEARCH_A0) * ARC_UM / 1000
print(f"  Coverage: {phys_search_z:.2f}mm × {phys_search_a:.2f}mm physical")

# ─── 4. Inverted clean image of candidate at r=310, maximum zoom ─────────────
print("\nGenerating maximum-zoom inverted candidate...")
u310_cand = sample_r(310)[CAND_Z0:CAND_Z1, CAND_A0:CAND_A1]
e_cand = enh(u310_cand.astype(np.uint8), sigma=0.15)
inv_cand = 255 - e_cand

# 10× vertical, 6× horizontal zoom
zoom_big = np.repeat(np.repeat(inv_cand, 10, axis=0), 6, axis=1)

# Add scale bar (1mm = 208px arc, at 6× = 1248px)
img_big = Image.fromarray(np.stack([zoom_big]*3, axis=2))
draw_big = ImageDraw.Draw(img_big)
sb = int(1000 / ARC_UM * 6)
draw_big.line([(20, zoom_big.shape[0]-40), (20+sb, zoom_big.shape[0]-40)], fill=(200,0,0), width=5)
draw_big.text((20, zoom_big.shape[0]-70), "1 mm", fill=(200,0,0))

# Physical size annotation
phys_w = (CAND_A1 - CAND_A0) * ARC_UM / 1000
phys_h = (CAND_Z1 - CAND_Z0) * 4.8 / 1000
draw_big.text((5, 5), f"PHerc.332 (Scroll3) | r≈{CAND_R*4.8/1000:.2f}mm depth | {phys_w:.1f}×{phys_h:.1f}mm", fill=(200,0,0))
draw_big.text((5, 25), f"m7_nnUNet ink pred | z={z_lo_l2+CAND_Z0}-{z_lo_l2+CAND_Z1} | arc={phys_a0:.1f}-{phys_a1:.1f}mm", fill=(180,0,0))

img_big.save(str(OUT_DIR / "candidate_maxzoom_inverted.png"))
print(f"  Saved candidate_maxzoom_inverted.png ({zoom_big.shape})")

# ─── 5. Also search the FULL angular range at z=140-280 for more features ────
print("\nFull-circle scan at z=140-280 for all letter candidates...")
u310_full = sample_r(310)[CAND_Z0:CAND_Z1, :]  # (z-range, all 1800 angles)
e_full = enh(u310_full.astype(np.uint8), sigma=0.5)
inv_full = 255 - e_full

# Find non-empty 60-angle windows (letters ~12-60 angle px at 5µm/px = 60-300µm)
WIN = 60
threshold = 200  # in inverted image, < 200 means ink was present
found_regions = []
for a in range(0, N_ANGLES - WIN, 10):
    block = inv_full[:, a:a+WIN]
    # "ink pixels" = pixels below threshold (dark on inverted)
    ink_count = (block < threshold).sum()
    if ink_count > 50:  # at least 50 ink pixels
        # Check spatial extent
        ink_rows = (block < threshold).any(axis=1).sum()
        ink_cols = (block < threshold).any(axis=0).sum()
        if ink_rows >= 5 and ink_cols >= 5:
            found_regions.append((ink_count, ink_rows, ink_cols, a))

found_regions.sort(reverse=True)
print(f"  Found {len(found_regions)} candidate windows")
print("  Top-15:")
for cnt, rw, cw, a in found_regions[:15]:
    arc_mm = a * ARC_UM / 1000
    print(f"    a={a} ({arc_mm:.2f}mm): ink_px={cnt}, rows={rw}, cols={cw}")

# Panorama strip: full circle at z=140-280, r=310, inverted
# Downscale to 2× zoom (1800 × 2*(Z1-Z0) display)
pano = np.repeat(inv_full, 2, axis=0)
img_pano = Image.fromarray(np.stack([pano]*3, axis=2))
draw_pano = ImageDraw.Draw(img_pano)
# Mark all found regions
for cnt, rw, cw, a in found_regions[:20]:
    draw_pano.rectangle([a, 0, a+WIN, pano.shape[0]], outline=(255,0,0), width=1)
# Mark our known candidate
draw_pano.rectangle([CAND_A0, 0, CAND_A1, pano.shape[0]], outline=(0,200,0), width=3)
# Scale bar: 1mm = 200px at 1× (no zoom horizontally)
draw_pano.line([(20, pano.shape[0]-10), (20+200, pano.shape[0]-10)], fill=(255,0,0), width=3)
draw_pano.text((20, pano.shape[0]-25), "1mm", fill=(255,0,0))
img_pano.save(str(OUT_DIR / "panorama_z140_280_r310.png"))
print(f"  Saved panorama_z140_280_r310.png — full circle scan")

print("\nDONE.")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

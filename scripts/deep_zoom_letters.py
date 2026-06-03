"""
Deep zoom into letter-candidate regions found in clahe_text_hunt.py.

Target regions:
  Strip 0 (z=1400-1540): parallel-line structure at angle ~650-850 px
  Strip 1 (z=1540-1680): two tall structures at angle ~150-450 and ~600-800 px
  Also tries r=305, 310 (minimum ink radius = inner surface boundary)
"""
import zarr
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import cv2

HOME = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR  = HOME / "scroll_prize/data/scroll3_ink_pred/deepzoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scroll geometry (confirmed from previous runs)
cy, cx   = 496.0, 534.4
z_lo_l2  = 1400
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

# Load zarr
print("Loading level-2 zarr...")
arr2 = zarr.open_array(str(L2_PATH), mode="r")
data = arr2[z_lo_l2:, :, :]  # (700, 986, 986)
nz, ny, nx = data.shape
print(f"  {data.shape}, ink>0: {(data>0).mean()*100:.2f}%")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_radius(r_px):
    ys = np.clip(cy + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx + r_px * np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enhance(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── 1. Sample 5 radii around minimum ink ────────────────────────────────────
radii = [298, 305, 310, 315, 320]
print(f"\nSampling radii {radii}...")
unrolled = {}
for r in radii:
    u = sample_radius(r)
    unrolled[r] = u
    print(f"  r={r}: ink={(u>0).mean()*100:.1f}%")

# ─── 2. Deep zoom — strip 0 (z=0-140 relative, abs z=1400-1540) ──────────────
print("\nDeep zoom: Strip 0 (z=1400-1540)")
Z0, Z1 = 0, 140  # relative indices

# The parallel-line candidate was at angle ~650-850 px (in 1800-angle image)
# Let's also search a wider window 300-1000 to not miss it
WINDOW_A = [300, 1000]

for r in radii:
    strip = unrolled[r][Z0:Z1, WINDOW_A[0]:WINDOW_A[1]]  # (140, 700)
    enh = enhance(strip, sigma=0.3)

    # Save at 6× vertical zoom, 2× horizontal
    zoomed = np.repeat(np.repeat(enh, 6, axis=0), 2, axis=1)  # (840, 1400)
    img = Image.fromarray(zoomed)
    draw = ImageDraw.Draw(img)
    # Mark the candidate zone (~650-850 global → 350-550 in this crop)
    local_lo = (650 - WINDOW_A[0]) * 2
    local_hi = (850 - WINDOW_A[0]) * 2
    draw.rectangle([local_lo, 0, local_hi, zoomed.shape[0]], outline=128, width=2)
    img.save(str(OUT_DIR / f"strip0_r{r}_6x.png"))

print(f"  Saved strip0_r*_6x.png (6× vertical zoom, 140×700 px source)")

# Ultra-zoom on just the candidate window (angle 600-900 = 300px wide)
print("\nUltra-zoom: strip0 angle 600-900 at all radii")
CAND_A0, CAND_A1 = 600, 900  # 300 angle pixels = ~1.5mm arc

for r in radii:
    strip = unrolled[r][Z0:Z1, CAND_A0:CAND_A1]  # (140, 300)
    enh   = enhance(strip, sigma=0.2)
    # 8× vertical, 4× horizontal
    zoomed = np.repeat(np.repeat(enh, 8, axis=0), 4, axis=1)  # (1120, 1200)
    Image.fromarray(zoomed).save(str(OUT_DIR / f"cand_strip0_r{r}_8x4x.png"))

print("  Saved cand_strip0_r*_8x4x.png")

# ─── 3. Deep zoom — strip 1 (z=140-280 relative, abs z=1540-1680) ────────────
print("\nDeep zoom: Strip 1 (z=1540-1680)")
Z2, Z3 = 140, 280

# Two structures found at ~150-450 and ~600-800 global angle
for r in radii:
    strip = unrolled[r][Z2:Z3, 100:900]  # (140, 800)
    enh   = enhance(strip, sigma=0.3)
    zoomed = np.repeat(np.repeat(enh, 6, axis=0), 2, axis=1)
    img = Image.fromarray(zoomed)
    draw = ImageDraw.Draw(img)
    # Mark the two candidate structures
    for lo, hi in [(50*2, 350*2), (500*2, 700*2)]:
        draw.rectangle([lo, 0, hi, zoomed.shape[0]], outline=180, width=2)
    img.save(str(OUT_DIR / f"strip1_r{r}_6x.png"))

print("  Saved strip1_r*_6x.png")

# Ultra-zoom on strip1 structure A (angle 100-450)
print("\nUltra-zoom: strip1 structure A (angle 100-450)")
for r in radii:
    strip = unrolled[r][Z2:Z3, 100:450]  # (140, 350)
    enh   = enhance(strip, sigma=0.2)
    zoomed = np.repeat(np.repeat(enh, 8, axis=0), 3, axis=1)  # (1120, 1050)
    Image.fromarray(zoomed).save(str(OUT_DIR / f"cand_strip1A_r{r}_8x3x.png"))

# Ultra-zoom on strip1 structure B (angle 500-800)
print("Ultra-zoom: strip1 structure B (angle 500-800)")
for r in radii:
    strip = unrolled[r][Z2:Z3, 500:800]  # (140, 300)
    enh   = enhance(strip, sigma=0.2)
    zoomed = np.repeat(np.repeat(enh, 8, axis=0), 4, axis=1)  # (1120, 1200)
    Image.fromarray(zoomed).save(str(OUT_DIR / f"cand_strip1B_r{r}_8x4x.png"))

print("  Saved cand_strip1A/B_r*_8x.png")

# ─── 4. Multi-radius composite: best window, all 5 radii side-by-side ────────
print("\nBuilding multi-radius composite for best window...")
# z=0-280 (full strip0+strip1), angle=600-900
COMP_Z0, COMP_Z1 = 0, 280
COMP_A0, COMP_A1 = 600, 900

panels = []
for r in radii:
    block = unrolled[r][COMP_Z0:COMP_Z1, COMP_A0:COMP_A1]  # (280, 300)
    enh   = enhance(block, sigma=0.3)
    # 4× zoom each direction
    zoomed = np.repeat(np.repeat(enh, 4, axis=0), 4, axis=1)  # (1120, 1200)
    # Add 4px gray separator
    sep = np.ones((zoomed.shape[0], 4), dtype=np.uint8) * 80
    panels.extend([zoomed, sep])

composite = np.hstack(panels[:-1])  # remove trailing separator
Image.fromarray(composite).save(str(OUT_DIR / "multi_radius_composite.png"))
print(f"  Saved multi_radius_composite.png ({composite.shape}) — radii {radii} side-by-side")

# ─── 5. Look for text-line regularity in r=310 ───────────────────────────────
print("\nText-line regularity analysis at r=310...")
u310 = unrolled[310]
enh310 = enhance(u310, sigma=0.5)

# For each 100-angle-px window, compute vertical autocorrelation
# (text lines → peaks in z-autocorrelation at ~2-3mm spacing = ~416-625 z-px)
best_ac_score = 0
best_window = 0
for a_start in range(0, N_ANGLES - 100, 50):
    col = enh310[:, a_start:a_start+100].mean(axis=1).astype(float)
    col -= col.mean()
    if col.std() < 1:
        continue
    ac = np.correlate(col, col, mode='full')
    ac = ac[len(ac)//2:]
    ac /= ac[0]
    # Look for peaks at 200-700 px lag (1-3.4mm = plausible line spacing)
    peak_score = ac[200:700].max() if len(ac) > 700 else 0
    if peak_score > best_ac_score:
        best_ac_score = peak_score
        best_window = a_start

print(f"  Best autocorrelation window: angle={best_window}-{best_window+100}")
print(f"  Score: {best_ac_score:.4f} (>0.3 suggests regular spacing)")

# Save the z-profile of that window
col_best = enh310[:, best_window:best_window+100].mean(axis=1)
profile_img = np.zeros((256, nz), dtype=np.uint8)
for zi, v in enumerate(col_best):
    bar = int(v / 255 * 250)
    profile_img[256-bar:256, zi] = 200
Image.fromarray(profile_img).save(str(OUT_DIR / "r310_best_zprofile.png"))

# Save the strip at best window, 8× zoom
strip_best = enh310[:, best_window:best_window+200]
Image.fromarray(np.repeat(np.repeat(strip_best, 8, axis=0), 2, axis=1)).save(
    str(OUT_DIR / f"r310_best_window_a{best_window}_8x.png"))
print(f"  Saved r310_best_window_a{best_window}_8x.png")

# ─── 6. Summary ──────────────────────────────────────────────────────────────
print("\nDONE. Key output files:")
key_files = [
    "multi_radius_composite.png",
    "cand_strip0_r310_8x4x.png",
    "cand_strip1A_r310_8x3x.png",
    "cand_strip1B_r310_8x4x.png",
    f"r310_best_window_a{best_window}_8x.png",
]
for f in key_files:
    p = OUT_DIR / f
    if p.exists():
        print(f"  {f}  ({p.stat().st_size//1024}KB)")
    else:
        print(f"  {f}  MISSING")
print("\nAll files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

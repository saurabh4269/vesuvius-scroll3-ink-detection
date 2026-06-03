"""
Comprehensive search at r=310 (the only layer where letters are detectable):
  - Full z=0-2100 (all 11 z-slabs now downloaded)
  - Full 1800 angles (but exclude outer-wall saturation a>1100)
  - Sliding window connected component analysis
  - 5-radius gradient validation on every candidate found

Output:
  - r310_all_candidates.txt — ranked list
  - r310_panorama_full.png  — full z × angle heat map
  - r310_candidate_XX.png   — zoom + gradient for each passing candidate
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

HOME    = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/r310_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0
R_MAIN   = 310

arr2 = zarr.open_array(str(L2_PATH), mode="r")
NZ, NY, NX = arr2.shape
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

# Precompute y,x indices for r=310 and validation radii
def get_yx(r):
    ys = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    return ys, xs

ys310, xs310 = get_yx(R_MAIN)
GRAD_RADII   = [298, 304, 310, 316, 322]

def enh(img, sigma=0.3):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── Pass 1: full z scan, chunk by chunk ─────────────────────────────────────
print(f"Pass 1: Full z=0-{NZ} scan at r=310...")
CHUNK = 192
Z_FRACS = []   # per-z ink fraction at a=0-1100 (exclude outer wall)
UNROLLED_STRIPS = []   # store CLAHE-enhanced strips for panorama

all_candidates = []

for z0 in range(0, NZ, CHUNK):
    z1   = min(z0 + CHUNK, NZ)
    slab = arr2[z0:z1, :, :][:]
    u    = slab[:, ys310, xs310].astype(np.uint8)   # (chunk, 1800)

    frac_per_z = (u[:, :1100] > 0).mean(axis=1)
    Z_FRACS.extend(frac_per_z.tolist())

    # Enhance the strip for panorama
    e   = enh(u[:, :1100], sigma=0.5)
    inv = 255 - e
    UNROLLED_STRIPS.append(inv)

    # Connected component search on the enhanced strip (exclude a>1100)
    e_full = enh(u, sigma=0.3)
    thresh = (e_full[:, :1100] > 100).astype(np.uint8)
    labeled_arr, n_comp = label(thresh)

    for c in range(1, n_comp + 1):
        mask = (labeled_arr == c)
        pix  = int(mask.sum())
        if pix < 25 or pix > 20000:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        z_span = rows.max() - rows.min() + 1
        a_span = cols.max() - cols.min() + 1
        z_mm   = z_span * 4.8 / 1000
        a_mm   = a_span * ARC_UM / 1000
        a_ctr  = int(cols.mean())
        if 0.3 <= z_mm <= 3.0 and 0.2 <= a_mm <= 2.5:
            all_candidates.append({
                'z_abs0': z0 + int(rows.min()),
                'z_abs1': z0 + int(rows.max()),
                'a0':     int(cols.min()),
                'a1':     int(cols.max()),
                'a_ctr':  a_ctr,
                'pixels': pix,
                'z_mm':   round(z_mm, 3),
                'a_mm':   round(a_mm, 3),
            })

    print(f"  z={z0}-{z1}: ink={frac_per_z.mean():.3f}, new_cands={len([c for c in all_candidates if c['z_abs0'] >= z0])}")

Z_FRACS = np.array(Z_FRACS)
print(f"\nTotal raw candidates: {len(all_candidates)}")

# ─── Deduplicate overlapping candidates ──────────────────────────────────────
all_candidates.sort(key=lambda x: x['pixels'], reverse=True)
deduped = []
for c in all_candidates:
    overlap = False
    for d in deduped:
        if (abs(c['a_ctr'] - d['a_ctr']) < 30 and
            abs(c['z_abs0'] - d['z_abs0']) < 50):
            overlap = True; break
    if not overlap:
        deduped.append(c)

print(f"After dedup: {len(deduped)} candidates")
print("\nTop 20:")
for i, c in enumerate(deduped[:20]):
    arc_mm = c['a_ctr'] * ARC_UM / 1000
    z_phys = c['z_abs0'] * 4.8 / 1000
    print(f"  [{i:2d}] z={c['z_abs0']}-{c['z_abs1']} ({z_phys:.2f}mm), "
          f"a={c['a_ctr']}({arc_mm:.2f}mm), {c['a_mm']:.2f}×{c['z_mm']:.2f}mm, px={c['pixels']}")

# ─── Pass 2: gradient validation for top candidates ──────────────────────────
print("\nPass 2: 5-radius gradient validation for top candidates...")
validated = []

for i, c in enumerate(deduped[:15]):
    # Load a generous z-window around candidate
    Z_BUF = 50
    za = max(0, c['z_abs0'] - Z_BUF)
    zb = min(NZ, c['z_abs1'] + Z_BUF)
    slab = arr2[za:zb, :, :][:]

    # Sample 5 radii
    ink_peaks = []
    for r in GRAD_RADII:
        ys_r, xs_r = get_yx(r)
        u_r   = slab[:, ys_r, xs_r].astype(np.uint8)
        crop  = u_r[:, c['a0']:c['a1']+1]
        score = (crop > 0).mean()
        ink_peaks.append(score)

    # Is the gradient peaked at r=310?
    center_idx = GRAD_RADII.index(310)
    peak_is_310 = (
        ink_peaks[center_idx] == max(ink_peaks) and
        ink_peaks[0] < ink_peaks[center_idx] * 0.7 and   # smaller at r=298
        ink_peaks[-1] < ink_peaks[center_idx] * 0.5      # much smaller at r=322
    )

    arc_mm = c['a_ctr'] * ARC_UM / 1000
    z_phys = c['z_abs0'] * 4.8 / 1000
    print(f"  [{i:2d}] z={c['z_abs0']}({z_phys:.2f}mm) a={c['a_ctr']}({arc_mm:.2f}mm): "
          f"grad={[f'{v:.2f}' for v in ink_peaks]}  PASS={'✓' if peak_is_310 else '✗'}")

    if peak_is_310:
        validated.append({**c, 'gradient': ink_peaks})

print(f"\nGradient-validated candidates: {len(validated)}")
for c in validated:
    arc_mm = c['a_ctr'] * ARC_UM / 1000
    z_phys = c['z_abs0'] * 4.8 / 1000
    print(f"  z={c['z_abs0']}({z_phys:.2f}mm) a={c['a_ctr']}({arc_mm:.2f}mm) "
          f"{c['a_mm']:.2f}×{c['z_mm']:.2f}mm  grad={[f'{v:.2f}' for v in c['gradient']]}")

# ─── Generate panorama ────────────────────────────────────────────────────────
print("\nGenerating full-scroll panorama at r=310 (a=0-1100)...")
full_strip = np.vstack(UNROLLED_STRIPS)   # (NZ, 1100)
# Downsample z 4× for display
DS = 4
strip_ds = full_strip[::DS, :]
pano = np.stack([strip_ds]*3, axis=2)
img_pano = Image.fromarray(pano)
draw     = ImageDraw.Draw(img_pano)

# Mark all validated candidates
for c in validated:
    row  = c['z_abs0'] // DS
    a0_p = c['a0']
    a1_p = c['a1']
    draw.rectangle([a0_p, row, a1_p, row + c['z_span']//DS if 'z_span' in c else row+10],
                   outline=(0, 220, 0), width=2)

# Mark mm ticks
H_p = strip_ds.shape[0]
for mm in range(0, 11):
    y_px = int(mm * 1000 / 4.8 / DS)
    if y_px < H_p:
        draw.line([(0, y_px), (15, y_px)], fill=(200,0,0), width=1)
        draw.text((16, y_px-7), f"{mm}mm", fill=(200,0,0))

# Scale bar 1mm arc
draw.line([(20, H_p-6), (220, H_p-6)], fill=(200,0,0), width=2)
draw.text((20, H_p-18), "1mm arc", fill=(200,0,0))
draw.text((5, 5), "PHerc.332 r=310 full scroll panorama (a=0-1100, z-downsampled 4×)", fill=(200,0,0))

img_pano.save(str(OUT_DIR / "r310_panorama_full.png"))
print(f"  Saved r310_panorama_full.png ({pano.shape})")

# ─── Save candidate report ───────────────────────────────────────────────────
with open(str(OUT_DIR / "r310_candidates.txt"), "w") as f:
    f.write(f"Total raw: {len(all_candidates)}, deduped: {len(deduped)}, validated: {len(validated)}\n\n")
    f.write("=== GRADIENT-VALIDATED CANDIDATES ===\n")
    for c in validated:
        f.write(f"z={c['z_abs0']}-{c['z_abs1']} ({c['z_abs0']*4.8/1000:.2f}mm)  "
                f"a={c['a_ctr']} ({c['a_ctr']*ARC_UM/1000:.2f}mm arc)  "
                f"{c['a_mm']:.2f}×{c['z_mm']:.2f}mm  px={c['pixels']}\n"
                f"  gradient r298-322: {[f'{v:.3f}' for v in c['gradient']]}\n")
    f.write("\n=== ALL DEDUPED CANDIDATES (TOP 30) ===\n")
    for c in deduped[:30]:
        f.write(f"z={c['z_abs0']} ({c['z_abs0']*4.8/1000:.2f}mm)  a={c['a_ctr']}  {c['a_mm']:.2f}×{c['z_mm']:.2f}mm  px={c['pixels']}\n")

print("  Saved r310_candidates.txt")
print("\nDONE.")

"""
5-radius gradient test on two new letter candidates:

A) r=400, a=988, z=1400-1520 — isolated oval at top of candidate [1]
   Arc position: 4.94mm, nearby z range only (before chunk saturation kicks in)

B) r=385, a=335, z=1400-1900 — candidate [3], arc=1.68mm
   Close to known letter arc=2.01mm, different scroll layer (r=385 vs r=310)

For each: sample radii ±30px in steps of 6 and show the gradient.
Real ink = peaks sharply at one radius.
Chunk artifact = flat across all radii (either all-black or all-white).
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME    = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 5.0

arr2 = zarr.open_array(str(L2_PATH), mode="r")
NY, NX = arr2.shape[1], arr2.shape[2]
clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r_slab(slab, r_px):
    ys = np.clip(cy + r_px*np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r_px*np.cos(angles), 0, NX-1).astype(int)
    return slab[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.25):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

def make_gradient_panel(slab, r_center, a0, a1, radii_offsets, label_prefix, outname):
    radii = [r_center + d for d in radii_offsets]
    panels = []
    for r in radii:
        u    = sample_r_slab(slab, r)
        crop = u[:, a0:a1]
        e    = enh(crop, sigma=0.2)
        inv  = 255 - e
        W_out, H_out = 180, 500
        panel = np.array(Image.fromarray(inv).resize((W_out, H_out), Image.LANCZOS))
        rgb   = np.stack([panel]*3, axis=2)
        img   = Image.fromarray(rgb)
        d     = ImageDraw.Draw(img)
        is_center = (r == r_center)
        color = (0, 140, 0) if is_center else (100, 100, 100)
        d.rectangle([0, 0, W_out-1, 26], fill=(245,245,245))
        d.text((3, 4),  f"r={r}px", fill=color)
        d.text((3, 14), f"{'← TARGET' if is_center else ''}", fill=color)
        panels.append(np.array(img))

    sep  = np.ones((panels[0].shape[0], 5, 3), dtype=np.uint8) * 160
    row  = np.hstack([x for p in panels for x in [p, sep]][:-1])
    img_row = Image.fromarray(row)
    draw    = ImageDraw.Draw(img_row)
    # Header
    phys_r = r_center * 4.8 / 1000
    phys_a = ((a0+a1)//2) * ARC_UM / 1000
    draw.text((4, 4), f"{label_prefix}  r={r_center}px={phys_r:.2f}mm  a={a0}-{a1} ({phys_a:.2f}mm)", fill=(200,0,0))
    img_row.save(str(OUT_DIR / outname))
    print(f"  Saved {outname}  ({row.shape})")
    return row

# ─── Candidate A: r=400, a=960-1020, z=1400-1520 (just the isolated oval) ───
print("=== Candidate A: r=400, a=960-1020, z=1400-1520 ===")
Z0_A, Z1_A = 1400, 1520
slab_A = arr2[Z0_A:Z1_A, :, :][:]
make_gradient_panel(
    slab_A, r_center=400, a0=960, a1=1020,
    radii_offsets=[-18, -12, -6, 0, 6, 12, 18],
    label_prefix="CandA isolated oval",
    outname="v8_candA_gradient.png"
)

# ─── Candidate B: r=385, a=280-400, z=1400-1900 ─────────────────────────────
print("\n=== Candidate B: r=385, a=280-400, z=1400-1900 ===")
Z0_B, Z1_B = 1400, 1900
slab_B = arr2[Z0_B:Z1_B, :, :][:]
make_gradient_panel(
    slab_B, r_center=385, a0=280, a1=400,
    radii_offsets=[-18, -12, -6, 0, 6, 12, 18],
    label_prefix="CandB near-known arc",
    outname="v8_candB_gradient.png"
)

# Also compare B directly with our known letter at r=310, a=280-400
print("\n=== Known letter at r=310, a=280-400 for comparison ===")
make_gradient_panel(
    slab_B, r_center=310, a0=280, a1=400,
    radii_offsets=[-18, -12, -6, 0, 6, 12, 18],
    label_prefix="KNOWN LETTER r=310",
    outname="v8_known_gradient_comparison.png"
)

# ─── Side-by-side: r=310 known vs r=385 candidate B at same angle range ──────
print("\n=== Side-by-side comparison r=310 vs r=385 at a=280-400 ===")
def make_strip(slab, r_px, a0, a1):
    u = sample_r_slab(slab, r_px)
    c = u[:, a0:a1]
    e = enh(c, sigma=0.2)
    i = 255 - e
    return np.array(Image.fromarray(i).resize((250, 600), Image.LANCZOS))

p310  = make_strip(slab_B, 310, 280, 400)
p385  = make_strip(slab_B, 385, 280, 400)
sep   = np.ones((600, 8), dtype=np.uint8) * 160
sbs   = np.hstack([p310, sep, p385])
img_sbs = Image.fromarray(sbs)
d_sbs   = ImageDraw.Draw(img_sbs)
d_sbs.text((4, 4),   "r=310 (known letter)", fill=(0,140,0))
d_sbs.text((262, 4), "r=385 (candidate B)", fill=(100,100,100))
img_sbs.save(str(OUT_DIR / "v8_310vs385_sidebyside.png"))
print(f"  Saved v8_310vs385_sidebyside.png")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v8_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

"""
Full-resolution (1.2 µm/px) analysis of the confirmed letter candidate.

Level-0 zarr: 4× finer than level-2 (4.8µm → 1.2µm/px)
Downloaded chunks: z=32-36 × y=15-16 × x=9-14
  covers z=6144-7104 (letter at z=6256-7004)
  spatial: y=2880-3264, x=1728-2880

Scroll center at level-0: cy=1984.0, cx=2136.8 (= 4 × level-2)
Letter at r_l0=1240px (= 4 × 310), z_l0=6256-7004
Arc resolution: ~1.25 µm/angle-px at r=1240
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME    = Path.home()
L0_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level0"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cy_l0, cx_l0 = 1984.0, 2136.8
N_ANGLES = 1800
angles   = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 1.25   # µm per angle-pixel at r=1240

R_L0     = 1240   # px  (310 × 4)
Z0_L0    = 6256   # abs z in level-0  (1564 × 4)
Z1_L0    = 7004   # abs z in level-0  (1751 × 4)
CAND_A0  = 280
CAND_A1  = 520

print("Opening level-0 zarr (partial: z-chunks 32-36)...")
arr0 = zarr.open_array(str(L0_PATH), mode="r")
print(f"  Full shape declared: {arr0.shape}")

# Load just the letter slab
print(f"  Loading z={Z0_L0}-{Z1_L0} (letter region)...")
data = arr0[Z0_L0:Z1_L0, :, :][:]
nz, ny, nx = data.shape
print(f"  Loaded: {data.shape}  ({nz*1.2/1000:.2f}mm z-height)")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy_l0 + r_px*np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx_l0 + r_px*np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.15):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── 1. 5-radius at level-0 ──────────────────────────────────────────────────
print("\n1. 5-radius comparison at level-0 (1.2µm/px)...")
RADII_L0 = [1192, 1216, 1240, 1264, 1288]  # 298,304,310,316,322 × 4

panels = []
labels = ["r≈298 (fibers)", "r≈304 (trans.)", "r≈310 (LETTER)", "r≈316 (fading)", "r≈322 (empty)"]
for r, lbl in zip(RADII_L0, labels):
    u    = sample_r(r)
    crop = u[:, CAND_A0:CAND_A1]
    e    = enh(crop, sigma=0.15)
    inv  = 255 - e
    W_out, H_out = 200, 600
    panel = np.array(Image.fromarray(inv).resize((W_out, H_out), Image.LANCZOS))
    rgb   = np.stack([panel]*3, axis=2)
    img   = Image.fromarray(rgb)
    d     = ImageDraw.Draw(img)
    color = (0, 140, 0) if r == R_L0 else (100, 100, 100)
    d.rectangle([0, 0, W_out-1, 26], fill=(245,245,245))
    d.text((3, 4),  lbl, fill=color)
    d.text((3, 14), "1.2µm/px", fill=(120,120,120))
    panels.append(np.array(img))

sep  = np.ones((panels[0].shape[0], 5, 3), dtype=np.uint8) * 160
row5 = np.hstack([x for p in panels for x in [p, sep]][:-1])
Image.fromarray(row5).save(str(OUT_DIR / "v9_l0_5radius.png"))
print(f"  Saved v9_l0_5radius.png  ({row5.shape})")

# ─── 2. Max-zoom at r=1240 (level-0) ─────────────────────────────────────────
print("\n2. Max-zoom letter at r=1240 level-0...")
u1240  = sample_r(R_L0)
crop   = u1240[:, CAND_A0:CAND_A1]
e1240  = enh(crop, sigma=0.1)
inv1240 = 255 - e1240

# 6× zoom in z, 4× in angle
zoomed = np.repeat(np.repeat(inv1240, 6, axis=0), 4, axis=1)
img_big = Image.fromarray(np.stack([zoomed]*3, axis=2))
d_big   = ImageDraw.Draw(img_big)

sb_arc = int(1000 / ARC_UM) * 4   # 1mm at 4× zoom
H_big, W_big = zoomed.shape
if sb_arc < W_big:
    d_big.line([(10, H_big-30), (10+sb_arc, H_big-30)], fill=(200,0,0), width=4)
    d_big.text((10, H_big-52), "1 mm (arc)", fill=(200,0,0))
d_big.text((5, 5), f"PHerc.332 | level-0 1.2µm/px | r=1.49mm | z=7.51-8.40mm", fill=(200,0,0))

img_big.save(str(OUT_DIR / "v9_l0_maxzoom.png"))
print(f"  Saved v9_l0_maxzoom.png  ({zoomed.shape})")

# ─── 3. Three-level comparison: level-2 / level-1 / level-0 ──────────────────
print("\n3. Three-level comparison (4.8 / 2.4 / 1.2 µm/px)...")

def load_at_res(path, z0, z1, cy, cx, r, a0, a1, res_label):
    arr  = zarr.open_array(str(path), mode="r")
    ny_  = arr.shape[1]
    nx_  = arr.shape[2]
    slab = arr[z0:z1, :, :][:]
    ys   = np.clip(cy + r*np.sin(angles), 0, ny_-1).astype(int)
    xs   = np.clip(cx + r*np.cos(angles), 0, nx_-1).astype(int)
    u    = slab[:, ys, xs][:, a0:a1].astype(np.uint8)
    e    = enh(u, sigma=0.1)
    return 255 - e

L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
L1_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level1"

p2 = load_at_res(L2_PATH, 1564, 1751, 496.0, 534.4, 310, CAND_A0, CAND_A1, "4.8µm")
p1 = load_at_res(L1_PATH, 3128, 3502, 992.0, 1068.8, 620, CAND_A0, CAND_A1, "2.4µm")
p0 = inv1240

TARGET_H = 700
def resize_panel(arr, lbl):
    W = int(TARGET_H * arr.shape[1] / arr.shape[0])
    resized = np.array(Image.fromarray(arr).resize((W, TARGET_H), Image.LANCZOS))
    rgb = np.stack([resized]*3, axis=2)
    img = Image.fromarray(rgb)
    d   = ImageDraw.Draw(img)
    d.rectangle([0, 0, W-1, 28], fill=(240,240,240))
    d.text((4, 5),  lbl, fill=(30,30,30))
    d.text((4, 17), "r=1.49mm  CLAHE  inverted", fill=(100,100,100))
    return np.array(img)[:, :, 0]

lp2 = resize_panel(p2, "4.8 µm/px (level-2)")
lp1 = resize_panel(p1, "2.4 µm/px (level-1)")
lp0 = resize_panel(p0, "1.2 µm/px (level-0) ◀ FULL RES")

W_max = max(lp2.shape[1], lp1.shape[1], lp0.shape[1])
def pad_to(arr, W):
    if arr.shape[1] < W:
        pad = np.ones((arr.shape[0], W-arr.shape[1]), dtype=np.uint8)*240
        arr = np.hstack([arr, pad])
    return arr

lp2 = pad_to(lp2, W_max); lp1 = pad_to(lp1, W_max); lp0 = pad_to(lp0, W_max)
sep = np.ones((TARGET_H, 10), dtype=np.uint8)*160
tri = np.hstack([lp2, sep, lp1, sep, lp0])
Image.fromarray(tri).save(str(OUT_DIR / "v9_three_levels.png"))
print(f"  Saved v9_three_levels.png  ({tri.shape})")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v9_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

"""
High-resolution (2.4 µm/px) analysis of the confirmed letter candidate.

Level-1 zarr: 2× finer than level-2 (4.8µm → 2.4µm/px)
Shape: (4199, 1971, 1971), chunks (192,192,192)
Downloaded chunks: z=16-18 × y=7-8 × x=4-7
  covers z=3072-3647 (candidate at z=3128-3502), spatial arc region

Scroll center at level-1: cy=992.0, cx=1068.8 (= 2 × level-2 values)
Letter at r_l1=620px (= 2 × 310), z_l1=3128-3502
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HOME    = Path.home()
L1_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level1"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/letter_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Level-1 geometry (2× level-2)
cy_l1, cx_l1 = 992.0, 1068.8
N_ANGLES = 1800
angles   = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
ARC_UM   = 2.5   # µm per angle-pixel at r=620 (5µm at r=310 → 2.5µm at 2.4µm/px)

# Level-2 candidate params scaled to level-1
R_L1     = 620   # px  (310 × 2)
Z0_L1    = 3128  # abs z in level-1  (1564 × 2)
Z1_L1    = 3502  # abs z in level-1  (1751 × 2)
CAND_A0  = 280
CAND_A1  = 520

print("Opening level-1 zarr (partial: z-chunks 16-18)...")
arr1 = zarr.open_array(str(L1_PATH), mode="r")
print(f"  Full shape: {arr1.shape}")

# Load the letter region
data = arr1[Z0_L1:Z1_L1, :, :][:]
nz, ny, nx = data.shape
print(f"  Letter slab: {data.shape}  ({nz * 2.4/1000:.2f}mm z-height)")

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def sample_r(r_px):
    ys = np.clip(cy_l1 + r_px * np.sin(angles), 0, ny-1).astype(int)
    xs = np.clip(cx_l1 + r_px * np.cos(angles), 0, nx-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def enh(img, sigma=0.2):
    sm = gaussian_filter(img.astype(float), sigma=sigma)
    return clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

# ─── 1. 5-radius comparison at level-1 ───────────────────────────────────────
print("\n1. 5-radius comparison at level-1 (2.4µm/px)...")
RADII_L1 = [596, 608, 620, 632, 644]   # ≈ 298,304,310,316,322 × 2

panels = []
labels = ["r≈298 (fibers)", "r≈304 (trans.)", "r≈310 (LETTER)", "r≈316 (fading)", "r≈322 (empty)"]
for r, lbl in zip(RADII_L1, labels):
    u    = sample_r(r)
    crop = u[:, CAND_A0:CAND_A1]
    e    = enh(crop, sigma=0.2)
    inv  = 255 - e
    # Resize to 200px wide, preserve height
    W_out = 200
    H_out = int(nz * W_out / (CAND_A1 - CAND_A0))
    panel = np.array(Image.fromarray(inv).resize((W_out, H_out), Image.LANCZOS))
    rgb   = np.stack([panel]*3, axis=2)
    img   = Image.fromarray(rgb)
    d     = ImageDraw.Draw(img)
    color = (0, 150, 0) if r == R_L1 else (100, 100, 100)
    d.rectangle([0, 0, W_out-1, 26], fill=(245,245,245))
    d.text((3, 3),  lbl, fill=color)
    d.text((3, 14), f"2.4µm/px", fill=(120,120,120))
    panels.append(np.array(img))

sep  = np.ones((panels[0].shape[0], 5, 3), dtype=np.uint8) * 170
row5 = np.hstack([x for p in panels for x in [p, sep]][:-1])
Image.fromarray(row5).save(str(OUT_DIR / "v7_l1_5radius.png"))
print(f"  Saved v7_l1_5radius.png  ({row5.shape})")

# ─── 2. Max-zoom at r=620 (level-1) ──────────────────────────────────────────
print("\n2. Max-zoom letter at r=620 level-1...")
u620  = sample_r(R_L1)
crop  = u620[:, CAND_A0:CAND_A1]
e620  = enh(crop, sigma=0.1)
inv620 = 255 - e620

# 8× zoom in z, 4× in angle
zoomed = np.repeat(np.repeat(inv620, 8, axis=0), 4, axis=1)
img_big = Image.fromarray(np.stack([zoomed]*3, axis=2))
d_big   = ImageDraw.Draw(img_big)

# Scale bar: 1mm = 1000/2.4 = 417 z-px × 8 zoom = 3333 px (vertical)
# and 1mm = 1000/2.5 = 400 angle-px × 4 = 1600 px (horizontal)
sb_arc = int(1000 / ARC_UM) * 4
H_big, W_big = zoomed.shape
d_big.line([(10, H_big-30), (10+sb_arc, H_big-30)], fill=(200,0,0), width=4)
d_big.text((10, H_big-52), "1 mm (arc)", fill=(200,0,0))
d_big.text((5, 5), f"PHerc.332 | level-1 2.4µm/px | r=1.49mm | z=7.51-8.40mm", fill=(200,0,0))
d_big.text((5, 18), f"Candidate: angle={CAND_A0}-{CAND_A1}, size={( CAND_A1-CAND_A0)*ARC_UM/1000:.1f}×{nz*2.4/1000:.2f}mm", fill=(160,0,0))

img_big.save(str(OUT_DIR / "v7_l1_maxzoom.png"))
print(f"  Saved v7_l1_maxzoom.png  ({zoomed.shape})")

# ─── 3. Side-by-side: level-1 vs level-2 at r=310 ───────────────────────────
print("\n3. Level-1 vs level-2 comparison at r=310...")
# level-2 (reload for comparison)
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
arr2    = zarr.open_array(str(L2_PATH), mode="r")
ny2, nx2 = arr2.shape[1], arr2.shape[2]
Z0_L2, Z1_L2 = 1564, 1751
data2   = arr2[Z0_L2:Z1_L2, :, :][:]
ys2 = np.clip(496.0 + 310 * np.sin(angles), 0, ny2-1).astype(int)
xs2 = np.clip(534.4 + 310 * np.cos(angles), 0, nx2-1).astype(int)
crop2   = data2[:, ys2, xs2][:, CAND_A0:CAND_A1].astype(np.uint8)
e2      = enh(crop2, sigma=0.15)
inv2    = 255 - e2

TARGET_H = 600
p_l2 = np.array(Image.fromarray(inv2).resize(
    (int(TARGET_H * inv2.shape[1] / inv2.shape[0]), TARGET_H), Image.LANCZOS))
p_l1 = np.array(Image.fromarray(inv620).resize(
    (int(TARGET_H * inv620.shape[1] / inv620.shape[0]), TARGET_H), Image.LANCZOS))

# Pad to same width
W_max = max(p_l2.shape[1], p_l1.shape[1])
def pad_w(arr, W):
    pad = np.ones((arr.shape[0], W - arr.shape[1]), dtype=np.uint8) * 240
    return np.hstack([arr, pad])
p_l2 = pad_w(p_l2, W_max)
p_l1 = pad_w(p_l1, W_max)

# Labels
def label_panel(arr, text):
    rgb = np.stack([arr]*3, axis=2)
    img = Image.fromarray(rgb)
    d   = ImageDraw.Draw(img)
    d.rectangle([0, 0, W_max-1, 28], fill=(240,240,240))
    d.text((5, 5),  text,              fill=(30, 30, 30))
    d.text((5, 16), "r=1.49mm  CLAHE  inverted", fill=(100,100,100))
    return np.array(img)[:, :, 0]

lp2 = label_panel(p_l2, "Level-2: 4.8 µm/px")
lp1 = label_panel(p_l1, "Level-1: 2.4 µm/px  (2× finer)")
sep = np.ones((TARGET_H, 10), dtype=np.uint8) * 160
sbs = np.hstack([lp2, sep, lp1])
Image.fromarray(sbs).save(str(OUT_DIR / "v7_l1_vs_l2.png"))
print(f"  Saved v7_l1_vs_l2.png  ({sbs.shape})")

print("\nDONE.")
for f in sorted(OUT_DIR.glob("v7_*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")

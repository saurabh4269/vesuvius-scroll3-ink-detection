"""
Letter hunting: zoom into high-confidence ink regions from B1 prediction
and cross-reference with raw scroll layers.
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import tifffile

HOME = Path.home()
PROJ = HOME / "scroll_prize/vesuvius_first_title_prize"
B1_PRED = PROJ / "results/infer_b1_20260603_181612/scroll3_20240618142020_prediction_T0.3.npy"
SEG_DIR = HOME / "scroll_prize/data/scroll3/fragments/20240618142020/layers"
OUT_DIR = HOME / "scroll_prize/data/scroll3_ink_pred/letter_hunt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Load B1 prediction ───────────────────────────────────────────────────
print("Loading B1 prediction...")
pred = np.load(str(B1_PRED))
H, W = pred.shape
print(f"  Shape: {H}x{W}, mean={pred.mean():.4f}, >0.9={(pred>0.9).mean()*100:.2f}%")

# ─── 2. Find high-conf regions with 2D spatial extent ───────────────────────
print("Finding 2D-extent ink regions (128px stride 64)...")
PATCH = 128
STRIDE = 64
threshold = 0.85

candidates = []
for y in range(0, H - PATCH, STRIDE):
    for x in range(0, W - PATCH, STRIDE):
        block = pred[y:y+PATCH, x:x+PATCH]
        high = block > threshold
        frac = high.mean()
        if frac < 0.001:
            continue
        row_maxima = int(high.any(axis=1).sum())
        col_maxima = int(high.any(axis=0).sum())
        if row_maxima >= 5 and col_maxima >= 5:
            candidates.append((frac, row_maxima, col_maxima, y, x))

candidates.sort(reverse=True)
print(f"  {len(candidates)} 2D-extent candidates")
print("  Top-20:")
for frac, rm, cm, y, x in candidates[:20]:
    print(f"    y={y}-{y+PATCH}, x={x}-{x+PATCH}: frac={frac*100:.2f}%, rows={rm}, cols={cm}")

# ─── 3. Save zoomed prediction crops ────────────────────────────────────────
print("\nSaving 512px context crops for top 20 candidates...")
CONTEXT = 512

for idx, (frac, rm, cm, y0, x0) in enumerate(candidates[:20]):
    cy = y0 + PATCH // 2
    cx = x0 + PATCH // 2
    y_lo = max(0, cy - CONTEXT//2)
    y_hi = min(H, cy + CONTEXT//2)
    x_lo = max(0, cx - CONTEXT//2)
    x_hi = min(W, cx + CONTEXT//2)

    crop = pred[y_lo:y_hi, x_lo:x_hi]

    vis = np.zeros((*crop.shape, 3), dtype=np.uint8)
    vis[crop > 0.9, 0] = 255
    vis[crop > 0.9, 1] = 200
    mask_mid = (crop > 0.7) & (crop <= 0.9)
    vis[mask_mid, 0] = 200
    mask_lo = (crop > 0.5) & (crop <= 0.7)
    vis[mask_lo, 0] = 120

    img = Image.fromarray(vis)
    draw = ImageDraw.Draw(img)
    r_y0 = y0 - y_lo
    r_x0 = x0 - x_lo
    draw.rectangle([r_x0, r_y0, r_x0+PATCH, r_y0+PATCH], outline=(0, 255, 0), width=2)

    fname = OUT_DIR / f"cand_{idx:02d}_y{y0}_x{x0}_f{int(frac*1000)}.png"
    img.save(str(fname))

print(f"  Saved {min(len(candidates), 20)} prediction crops")

# ─── 4. Cross-reference with scroll raw layers ───────────────────────────────
print("\nLoading scroll layers for cross-reference...")
layer_files = sorted(SEG_DIR.glob("*.tif"))
if not layer_files:
    layer_files = sorted(SEG_DIR.glob("*.png"))
print(f"  {len(layer_files)} layer files found")

# Load 3 representative layers: 20 (near-surface), 32 (middle), 44 (deep)
target_indices = [20, 32, 44]

for li in target_indices:
    if li >= len(layer_files):
        continue
    lf = layer_files[li]
    print(f"  Layer {li}: {lf.name}")
    try:
        raw = tifffile.imread(str(lf))
    except Exception:
        raw = np.array(Image.open(str(lf)))

    lo_p, hi_p = np.percentile(raw.astype(float), 1), np.percentile(raw.astype(float), 99)
    raw_8 = np.clip((raw.astype(float) - lo_p) / max(hi_p - lo_p, 1) * 255, 0, 255).astype(np.uint8)

    for idx, (frac, rm, cm, y0, x0) in enumerate(candidates[:10]):
        cy = y0 + PATCH // 2
        cx = x0 + PATCH // 2
        y_lo = max(0, cy - CONTEXT//2)
        y_hi = min(H, cy + CONTEXT//2)
        x_lo = max(0, cx - CONTEXT//2)
        x_hi = min(W, cx + CONTEXT//2)

        raw_crop = raw_8[y_lo:y_hi, x_lo:x_hi]
        pred_crop = pred[y_lo:y_hi, x_lo:x_hi]
        ch, cw = raw_crop.shape

        combined = np.zeros((ch, cw*2, 3), dtype=np.uint8)
        combined[:, :cw, :] = np.stack([raw_crop]*3, axis=-1)

        overlay = np.stack([raw_crop]*3, axis=-1).copy()
        high_mask = pred_crop > threshold
        r_ch = overlay[:,:,0].copy()
        g_ch = overlay[:,:,1].copy()
        b_ch = overlay[:,:,2].copy()
        r_ch[high_mask] = np.clip(r_ch[high_mask].astype(int) + 150, 0, 255)
        g_ch[high_mask] = np.clip(g_ch[high_mask].astype(int) - 50, 0, 255)
        b_ch[high_mask] = np.clip(b_ch[high_mask].astype(int) - 50, 0, 255)
        overlay[:,:,0] = r_ch
        overlay[:,:,1] = g_ch
        overlay[:,:,2] = b_ch
        combined[:, cw:, :] = overlay

        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"Layer {li}", fill=(255, 255, 0))
        draw.text((cw+5, 5), f"+ B1 ink (>{threshold})", fill=(255, 255, 0))

        fname = OUT_DIR / f"overlay_l{li:02d}_cand_{idx:02d}_y{y0}_x{x0}.png"
        img.save(str(fname))

    print(f"    Saved 10 overlay crops for layer {li}")

# ─── 5. Global ink density heatmap ──────────────────────────────────────────
print("\nBuilding global heatmap (256px blocks)...")
HEAT_STRIDE = 256
heat_h = H // HEAT_STRIDE
heat_w = W // HEAT_STRIDE
heatmap = np.zeros((heat_h, heat_w), dtype=float)
for i in range(heat_h):
    for j in range(heat_w):
        blk = pred[i*HEAT_STRIDE:(i+1)*HEAT_STRIDE, j*HEAT_STRIDE:(j+1)*HEAT_STRIDE]
        heatmap[i, j] = (blk > 0.85).mean()

hm_norm = (heatmap / max(heatmap.max(), 1e-7) * 255).astype(np.uint8)
hm_big = np.repeat(np.repeat(hm_norm, 20, axis=0), 20, axis=1)
Image.fromarray(hm_big).save(str(OUT_DIR / "global_heatmap_20x.png"))

print(f"  Heatmap shape: {heatmap.shape}, max block density: {heatmap.max()*100:.2f}%")
row_sums = heatmap.sum(axis=1)
top_rows = np.argsort(row_sums)[::-1][:5]
print("  Ink-hottest y-bands (each band = 256px):")
for r in top_rows:
    yp = r * HEAT_STRIDE
    print(f"    row {r} → y={yp}-{yp+HEAT_STRIDE}: total_density={heatmap[r].sum():.4f}")

# ─── 6. Also do a high-res zoom of a 4096-wide strip around the hottest row ──
best_row = top_rows[0]
y_center = best_row * HEAT_STRIDE + HEAT_STRIDE // 2
print(f"\nExtracting 256-high strip at hottest row y={y_center}...")
strip_y0 = max(0, y_center - 128)
strip_y1 = min(H, y_center + 128)

# Take the strip across full width and find highest-density 2048-wide window
strip = pred[strip_y0:strip_y1, :]
step = 512
best_xw = 0
best_xval = 0.0
for xw in range(0, W - 2048, step):
    v = (strip[:, xw:xw+2048] > 0.85).mean()
    if v > best_xval:
        best_xval = v
        best_xw = xw

print(f"  Best 2048-wide window: x={best_xw}-{best_xw+2048}, density={best_xval*100:.2f}%")
strip_crop = pred[strip_y0:strip_y1, best_xw:best_xw+2048]
vis_strip = np.zeros((*strip_crop.shape, 3), dtype=np.uint8)
vis_strip[strip_crop > 0.9, 0] = 255
vis_strip[strip_crop > 0.9, 1] = 220
mid_m = (strip_crop > 0.7) & (strip_crop <= 0.9)
vis_strip[mid_m, 0] = 200
lo_m = (strip_crop > 0.5) & (strip_crop <= 0.7)
vis_strip[lo_m, 0] = 100
Image.fromarray(vis_strip).save(str(OUT_DIR / "hottest_strip_pred.png"))

print("\nDONE. Output files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == '.png':
        print(f"  {f.name}")

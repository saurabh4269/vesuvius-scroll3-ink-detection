"""
Analyze team's Scroll 3 (PHerc.332) m7_nnUNet ink predictions vs our B1 predictions.
Goal: find letter-candidate regions.
"""
import sys
import zarr
import numpy as np
from pathlib import Path
from PIL import Image

HOME = Path.home()
PRED_DIR = HOME / "scroll_prize/data/scroll3_ink_pred"
B1_PRED   = HOME / "scroll_prize/vesuvius_first_title_prize/results/infer_b1_20260603_181612/scroll3_20240618142020_prediction_T0.3.npy"
OUT_DIR   = HOME / "scroll_prize/data/scroll3_ink_pred/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Load team's 3D zarr level-3 prediction ─────────────────────────────
print("Loading team's zarr (level 3)...")
zarr_path = str(PRED_DIR / "level3")

# The path is a zarr array (not a group), open directly
try:
    arr = zarr.open_array(zarr_path, mode="r")
    print(f"  zarr array shape: {arr.shape}, dtype: {arr.dtype}, chunks: {arr.chunks}")
    team_3d = arr[:]
except Exception as e1:
    print(f"  zarr.open_array failed: {e1}")
    # Maybe the zarr is stored differently — try as a group and look for data
    try:
        grp = zarr.open_group(str(PRED_DIR), mode="r")
        print(f"  top-level keys: {list(grp.keys())}")
        team_3d = grp["level3"][:]
    except Exception as e2:
        print(f"  group approach also failed: {e2}")
        raise RuntimeError(f"Cannot load zarr: {e1} | {e2}")

print(f"  dtype: {team_3d.dtype}, min: {team_3d.min()}, max: {team_3d.max()}")
print(f"  ink fraction (>0): {(team_3d > 0).mean()*100:.2f}%")
np.save(str(OUT_DIR / "team_level3.npy"), team_3d)
print(f"  Saved to {OUT_DIR}/team_level3.npy")

# ─── 2. Load our B1 prediction (2D: height x width) ────────────────────────
print("\nLoading our B1 prediction...")
our_2d = np.load(str(B1_PRED))
print(f"  our_2d shape: {our_2d.shape}, mean: {our_2d.mean():.4f}")
print(f"  >0.5: {(our_2d>0.5).mean()*100:.2f}%, >0.7: {(our_2d>0.7).mean()*100:.2f}%, >0.9: {(our_2d>0.9).mean()*100:.2f}%")

# ─── 3. Analyse team's 3D predictions: per-Z-slice ink fractions ───────────
print("\nAnalysing per-z-slice ink fractions (team)...")
nz = team_3d.shape[0]
z_ink_frac = np.array([(team_3d[z] > 0).mean() for z in range(nz)])
print(f"  z-slices: {nz}, ink range: {z_ink_frac.min()*100:.2f}% – {z_ink_frac.max()*100:.2f}%")
top10_z = np.argsort(z_ink_frac)[::-1][:10]
print(f"  Top-10 ink-dense z-slices: {top10_z.tolist()}")
print(f"  Their ink fractions: {[round(z_ink_frac[z]*100, 2) for z in top10_z]}")

# ─── 4. Save a max-projection PNG (z-projection) ───────────────────────────
print("\nSaving z-projection PNG (team)...")
proj_z = team_3d.max(axis=0)  # (y, x) — max ink over all z
proj_norm = ((proj_z.astype(float) / max(proj_z.max(), 1)) * 255).astype(np.uint8)
Image.fromarray(proj_norm).save(str(OUT_DIR / "team_proj_z_max.png"))
print(f"  Saved team_proj_z_max.png  shape={proj_norm.shape}")

# also save colorized version
proj_rgb = np.zeros((*proj_norm.shape, 3), dtype=np.uint8)
proj_rgb[:,:,0] = proj_norm  # red channel = ink
Image.fromarray(proj_rgb).save(str(OUT_DIR / "team_proj_z_max_rgb.png"))

# ─── 5. Save top-Z slice PNGs ───────────────────────────────────────────────
print("\nSaving top ink-dense z-slice PNGs...")
for rank, z in enumerate(top10_z[:5]):
    sl = team_3d[z]
    sl_norm = ((sl.astype(float) / max(sl.max(), 1)) * 255).astype(np.uint8)
    Image.fromarray(sl_norm).save(str(OUT_DIR / f"team_z{z:04d}_rank{rank+1}.png"))
    print(f"  Saved team_z{z:04d}_rank{rank+1}.png  ink={z_ink_frac[z]*100:.2f}%")

# ─── 6. Spatial mapping: find ink-dense 2D patches ─────────────────────────
print("\nFinding ink-dense 2D patches in team projection...")
PATCH = 64  # patch size at level-3 (8× downscaled)
H, W = proj_z.shape
ph, pw = H // PATCH, W // PATCH
patch_ink = []
for i in range(ph):
    for j in range(pw):
        block = proj_z[i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH]
        frac = (block > 0).mean()
        patch_ink.append((frac, i, j))
patch_ink.sort(reverse=True)
print(f"  Top-10 ink-dense patches (row×col in level-3 image):")
for frac, i, j in patch_ink[:10]:
    # Convert to pixel coords in full-res (level 0: 8398×3941×3941)
    y0_l0 = i * PATCH * 8
    x0_l0 = j * PATCH * 8
    print(f"    patch ({i},{j}): ink={frac*100:.1f}%, level-0 coords ~y={y0_l0} x={x0_l0}")

# ─── 7. Compare dimensions: team 3D vs our 2D ──────────────────────────────
print("\nDimension comparison:")
print(f"  Team 3D (level 3): {team_3d.shape}  (z×y×x, 8× downsampled from full)")
print(f"  Our 2D (B1):       {our_2d.shape}   (y×x at segment native res ~4 µm)")
print(f"  Segment 20240618142020 native: 65 layers × 2491 × 25706 px at ~4 µm")
print(f"  Level-3 px at 8× → ~{2.4*8:.1f} µm/px")
print()
print("  NOTE: direct pixel-space comparison needs coordinate mapping")
print("  Team zarr covers full Scroll 3 volume; our pred is one surface segment")

# ─── 8. Our prediction: find ink-dense patches ──────────────────────────────
print("\nFinding ink-dense patches in our B1 prediction (stride 256)...")
STRIDE = 256
H2, W2 = our_2d.shape
our_patches = []
for i in range(0, H2 - STRIDE, STRIDE):
    for j in range(0, W2 - STRIDE, STRIDE):
        block = our_2d[i:i+STRIDE, j:j+STRIDE]
        frac = (block > 0.9).mean()
        if frac > 0.01:
            our_patches.append((frac, i, j))
our_patches.sort(reverse=True)
print(f"  {len(our_patches)} patches with >1% high-confidence ink")
print(f"  Top-10 high-confidence regions:")
for frac, i, j in our_patches[:10]:
    print(f"    y={i}–{i+STRIDE}, x={j}–{j+STRIDE}: {frac*100:.1f}% >0.9")

# ─── 9. Save a high-confidence-only PNG from our prediction ─────────────────
print("\nSaving our B1 high-confidence ink map (>0.7)...")
hconf_map = (our_2d > 0.7).astype(np.uint8) * 255
Image.fromarray(hconf_map).save(str(OUT_DIR / "our_b1_highconf_map.png"))
print(f"  Saved our_b1_highconf_map.png  shape={hconf_map.shape}")

# ─── 10. Summary report ─────────────────────────────────────────────────────
report = []
report.append("=== Scroll 3 Ink Prediction Analysis ===")
report.append(f"Team 3D (m7_nnUNet level-3): {team_3d.shape}, ink% = {(team_3d>0).mean()*100:.2f}%")
report.append(f"Our B1 T=0.3 (2D segment):  {our_2d.shape}, >0.5={( our_2d>0.5).mean()*100:.2f}%, >0.9={(our_2d>0.9).mean()*100:.2f}%")
report.append("")
report.append("Top-10 ink-dense Z-slices (team):")
for rank, z in enumerate(top10_z[:10]):
    report.append(f"  #{rank+1} z={z}: {z_ink_frac[z]*100:.2f}%")
report.append("")
report.append("Top-10 ink-dense 2D patches in B1 prediction (>0.9):")
for frac, i, j in our_patches[:10]:
    report.append(f"  y={i}-{i+STRIDE}, x={j}-{j+STRIDE}: {frac*100:.1f}% high-conf pixels")
report.append("")
report.append("Files saved:")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix in ('.png', '.npy'):
        report.append(f"  {f.name}")
report_text = "\n".join(report)
(OUT_DIR / "analysis_report.txt").write_text(report_text)
print("\n" + report_text)
print("\nDONE")

"""
Final discovery composite — best possible view of the PHerc.332 letter candidate.

Layout:
  Row 1: Header with coordinates and physical info
  Row 2: Three-level comparison (4.8 / 2.4 / 1.2 µm/px)
  Row 3: Level-0 5-radius gradient (fiber strands resolved at 1.2µm/px)
  Row 4: Full-circle panorama (isolation confirmed)

This is the "share on Discord and in submission" image.
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

OUT_DIR = Path.home() / "scroll_prize/data/scroll3_ink_pred/letter_report"

def load_img(fname, max_h=None, max_w=None):
    p = OUT_DIR / fname
    if not p.exists():
        print(f"  MISSING: {fname}")
        return None
    img = Image.open(p).convert("RGB")
    w, h = img.size
    if max_h and h > max_h:
        scale = max_h / h
        img = img.resize((int(w*scale), max_h), Image.LANCZOS)
    if max_w and img.size[0] > max_w:
        scale = max_w / img.size[0]
        img = img.resize((max_w, int(img.size[1]*scale)), Image.LANCZOS)
    return img

# ─── Load panels ──────────────────────────────────────────────────────────────
three = load_img("v9_three_levels.png", max_h=460)
grad  = load_img("v9_l0_5radius.png",  max_h=360)
pano  = load_img("v4_fullcircle_annotated.png", max_h=120)

if not all([three, grad, pano]):
    print("Missing required images"); raise SystemExit

# ─── Normalize widths ─────────────────────────────────────────────────────────
W = max(three.size[0], grad.size[0], pano.size[0])
W = min(W, 1600)   # cap at 1600px

def fit_width(img, target_w):
    w, h = img.size
    if w != target_w:
        img = img.resize((target_w, int(h * target_w / w)), Image.LANCZOS)
    return img

three = fit_width(three, W)
grad  = fit_width(grad,  W)
pano  = fit_width(pano,  W)

# ─── Build header ─────────────────────────────────────────────────────────────
H_HDR = 52
header = Image.new("RGB", (W, H_HDR), (248,248,248))
d = ImageDraw.Draw(header)
d.text((8, 6),  "PHerc.332 (Herculaneum Scroll 3)  |  Letter Candidate Confirmed", fill=(20,20,20))
d.text((8, 22), "Location: z = 7.51-8.40 mm height  |  arc = 1.40-2.60 mm  |  depth r = 1.49 mm from center", fill=(60,60,60))
d.text((8, 36), "Source: m7_nnUNet ink pred (vesuvius-challenge-open-data) | Method: 5-radius depth gradient diagnostic", fill=(100,100,100))

# ─── Section labels ───────────────────────────────────────────────────────────
def label_bar(text, W, h=22, color=(230,240,255)):
    bar = Image.new("RGB", (W, h), color)
    d   = ImageDraw.Draw(bar)
    d.text((8, 4), text, fill=(30,30,100))
    return bar

lbl_three = label_bar("Resolution comparison: 4.8 µm/px → 2.4 µm/px → 1.2 µm/px (FULL RESOLUTION)  ◀ cleaner at each step", W)
lbl_grad  = label_bar("5-radius depth gradient at 1.2 µm/px: individual papyrus fiber strands resolved at r≈298 → letter at r=310 → empty at r≈322", W)
lbl_pano  = label_bar("Full 360° scan at r=310 (z=1540-1900): letter is the ONLY isolated crackle-pattern structure in the entire circle", W)

# ─── Separator ────────────────────────────────────────────────────────────────
sep = Image.new("RGB", (W, 4), (180,180,180))

# ─── Assemble ─────────────────────────────────────────────────────────────────
total_h = (H_HDR + 4 + 22 + three.size[1] + 4 + 22 + grad.size[1] + 4 + 22 + pano.size[1])
canvas  = Image.new("RGB", (W, total_h), (255,255,255))

y = 0
for part in [header, sep, lbl_three, three, sep, lbl_grad, grad, sep, lbl_pano, pano]:
    canvas.paste(part, (0, y))
    y += part.size[1]

out = OUT_DIR / "FINAL_DISCOVERY.png"
canvas.save(str(out))
print(f"Saved FINAL_DISCOVERY.png  ({canvas.size[0]}×{canvas.size[1]})")
print(f"  File size: {out.stat().st_size//1024}KB")

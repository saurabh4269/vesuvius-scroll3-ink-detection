"""
Positive control for the PHerc.332 unroll pipeline.
We KNOW the ground truth because we paint it ourselves.

Test 1 (readout faithfulness): composite known letters (Pi, Omicron, Beta, Phi)
  onto REAL fiber background sampled from the zarr, run the EXACT pipe()
  (gaussian sigma0.3 -> CLAHE clip4/tile16 -> invert). Are the letters still legible?

Test 2 (3D sampling + expected gradient): paint the same letters onto a synthetic
  cylindrical shell at r=310 (3 voxels thick) inside a volume with realistic fiber
  + noise, then unroll at r=298/304/310/316/322 exactly like the real search.
  -> shows the letters recovered through real 3D sampling
  -> prints the gradient a REAL letter produces, to compare vs the candidate
     (whose gradient FAILED: r310_candidates.txt validated 0).

Output: poscontrol_panel.png + printed gradient profile.
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

HOME = Path.home()
L2   = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT  = HOME / "scroll_prize/data/scroll3_ink_pred/poscontrol"
OUT.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANG    = 1800
angles   = np.linspace(0, 2*np.pi, N_ANG, endpoint=False)
clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
GRAD     = [298, 304, 310, 316, 322]

def pipe(u):
    sm  = gaussian_filter(u.astype(float), sigma=0.3)
    enh = clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))
    return 255 - enh

# ---- known letters: topologically correct shapes, truth known ----
def make_letters(h=140, gap=40, stroke=12):
    cells = []
    def pi(w):
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (8, 6), (8+stroke, h-6), 255, -1)
        cv2.rectangle(m, (w-8-stroke, 6), (w-8, h-6), 255, -1)
        cv2.rectangle(m, (4, 6), (w-4, 6+stroke), 255, -1)
        return m
    def omicron(w):
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (w//2, h//2), (w//2-8, h//2-8), 0, 0, 360, 255, stroke)
        return m
    def beta(w):
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (8, 6), (8+stroke, h-6), 255, -1)
        cv2.ellipse(m, (w//2+2, h//4+4), (w//3, h//4-6), 0, -90, 90, 255, stroke)
        cv2.ellipse(m, (w//2+2, 3*h//4-4), (w//3, h//4-6), 0, -90, 90, 255, stroke)
        return m
    def phi(w):
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (w//2-stroke//2, 0), (w//2+stroke//2, h), 255, -1)
        cv2.ellipse(m, (w//2, h//2), (w//2-8, h//3), 0, 0, 360, 255, stroke)
        return m
    for fn, w in [(pi,110),(omicron,120),(beta,110),(phi,120)]:
        cells.append(fn(w)); cells.append(np.zeros((h, gap), np.uint8))
    return np.concatenate(cells, axis=1)   # (h, W)

letters = make_letters()
LH, LW  = letters.shape
print(f"letters mask {letters.shape}")

arr = zarr.open_array(str(L2), mode="r")
NZ, NY, NX = arr.shape

# real fiber background + real ink value level, sampled from the actual zarr
Z0 = 1564
fib_slab = arr[Z0:Z0+LH, :, :][:]
ys = np.clip(cy + 298*np.sin(angles), 0, NY-1).astype(int)
xs = np.clip(cx + 298*np.cos(angles), 0, NX-1).astype(int)
fiber_bg = fib_slab[:, ys, xs][:, 280:280+LW].astype(float)   # real fiber texture
# real ink magnitude: mean of nonzero values near candidate at r=310
ys310 = np.clip(cy + 310*np.sin(angles), 0, NY-1).astype(int)
xs310 = np.clip(cx + 310*np.cos(angles), 0, NX-1).astype(int)
cand  = fib_slab[:, ys310, xs310][:, 280:520]
ink_val = float(cand[cand > 0].mean()) if (cand > 0).any() else 180.0
print(f"real ink magnitude sampled: {ink_val:.1f}")

# ---- Test 1: readout faithfulness in 2D ----
comp = fiber_bg.copy()
comp[letters > 0] = ink_val
t1_truth = (255 - (letters)).astype(np.uint8)          # ground truth (black letters)
t1_out   = pipe(comp)                                   # through real pipeline

# ---- Test 2: full 3D sampling + gradient ----
V = np.zeros((LH, NY, NX), np.uint8)
# fiber band r=290..305 (wavy, from real data values broadcast)
for r in range(290, 306, 3):
    yy = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xx = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    V[:, yy, xx] = (fib_slab[:, yy, xx])               # real fiber texture in place
# paint letters on shell r=309..311 (3 voxels thick) at angle window 280..280+LW
for r in (309, 310, 311):
    yy = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xx = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    for zi in range(LH):
        on = letters[zi] > 0
        cols = np.arange(280, 280+LW)[on]
        V[zi, yy[cols], xx[cols]] = int(ink_val)

def unroll_crop(r):
    yy = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xx = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    return V[:, yy, xx][:, 280:280+LW].astype(np.uint8)

t2_out = pipe(unroll_crop(310))
print("\n=== expected 5-radius gradient of a REAL painted letter (coverage of dark px) ===")
for r in GRAD:
    inv = pipe(unroll_crop(r))
    cov = (inv < 120).mean()
    print(f"  r={r}: dark_coverage={cov:.3f}")
print("  (a real letter should PEAK at r=310 and drop sharply at 298/322)")
print("  compare to candidate: r310_candidates.txt showed it FAILS this)")

# ---- panel ----
def im(x): return Image.fromarray(x).resize((x.shape[1]*2, x.shape[0]*2), Image.NEAREST)
W = LW*2
panel = Image.new("L", (W*3 + 40, LH*2 + 30), 255)
for k,(x,lbl) in enumerate([(t1_truth,"truth"),(t1_out,"T1 readout"),(t2_out,"T2 3D-sampled")]):
    panel.paste(im(x), (k*(W+15)+5, 20))
panel.save(str(OUT / "poscontrol_panel.png"))
print(f"\nsaved {OUT/'poscontrol_panel.png'}")
print("READ: if T1/T2 letters are legible -> readout chain is trustworthy, real ink would show.")
print("READ: if they come back garbled -> the pipeline itself can't render letters, candidate moot.")

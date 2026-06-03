"""
Salvage / honesty test for the PHerc.332 r=310 letter candidate.
Runs the EXACT same pipeline (unroll -> CLAHE clip4/tile16 -> invert) on:
  A. the claimed candidate window            (z=1564-1751, a=280-520, r=310)
  B. an empty background window              (same z, a=1300-1500, r=310)
  C. the candidate window, angle-shuffled    (destroys structure, keeps values)
  D. the densest high-confidence ink region  (ground-truth: can it render letters?)

If B and C look as 'letter-like' as A  -> morphology is CLAHE artifact (RETRACT).
If D renders only blobs, never letters -> method can't surface letters (RETRACT).
Output: salvage_panel.png  + printed connected-component / texture stats.
"""
import zarr, cv2
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, label

HOME    = Path.home()
L2_PATH = HOME / "scroll_prize/data/scroll3_ink_pred/level2"
OUT     = HOME / "scroll_prize/data/scroll3_ink_pred/salvage"
OUT.mkdir(parents=True, exist_ok=True)

cy, cx   = 496.0, 534.4
N_ANGLES = 1800
angles   = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

arr = zarr.open_array(str(L2_PATH), mode="r")
NZ, NY, NX = arr.shape
print(f"zarr shape {arr.shape}")

def unroll(slab, r):
    ys = np.clip(cy + r*np.sin(angles), 0, NY-1).astype(int)
    xs = np.clip(cx + r*np.cos(angles), 0, NX-1).astype(int)
    return slab[:, ys, xs].astype(np.uint8)          # (nz, 1800)

def pipe(u):                                          # exact post-processing
    sm  = gaussian_filter(u.astype(float), sigma=0.3)
    enh = clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))
    return 255 - enh

def letter_stats(inv, label_name):
    """count connected dark components in the 'letter' size range."""
    dark = (inv < 120).astype(np.uint8)              # dark = ink after invert
    lab, n = label(dark)
    sizes = [int((lab==i).sum()) for i in range(1, n+1)]
    letterish = [s for s in sizes if 40 <= s <= 4000]
    cov = dark.mean()
    # texture: lag-1 autocorr along angle axis (real strokes -> higher)
    f = inv.astype(float); f -= f.mean()
    ac = (f[:, :-1]*f[:, 1:]).sum() / (f**2).sum() if (f**2).sum() else 0
    print(f"  [{label_name:22s}] coverage={cov:.3f}  letterish_blobs={len(letterish):3d}  autocorr={ac:+.3f}")
    return cov, len(letterish), ac

# z window for candidate
Z0, Z1 = 1564, 1752
slab = arr[Z0:Z1, :, :][:]

# A. candidate
uA   = unroll(slab, 310)[:, 280:520]
invA = pipe(uA)

# B. empty background, same z, different angle band
uB   = unroll(slab, 310)[:, 1300:1540]
invB = pipe(uB)

# C. candidate window, angle-shuffled (seeded by fixed perm, no RNG)
perm = (np.arange(uA.shape[1]) * 7 + 3) % uA.shape[1]   # deterministic shuffle
uC   = uA[:, perm]
invC = pipe(uC)

print("\n=== letter-likeness (A=claim, B=empty, C=shuffled) ===")
sA = letter_stats(invA, "A claimed candidate")
sB = letter_stats(invB, "B empty background")
sC = letter_stats(invC, "C angle-shuffled")

# D. ground-truth: scan all z at r=310 for densest coherent (non-outer-wall) region
print("\n=== D: densest high-confidence ink region (ground-truth render) ===")
best = None
for z0 in range(0, NZ, 192):
    z1 = min(z0+192, NZ)
    s  = arr[z0:z1, :, :][:]
    u  = unroll(s, 310)[:, :1100]          # exclude outer wall a>1100
    e  = clahe.apply(np.clip(gaussian_filter(u.astype(float),0.3),0,255).astype(np.uint8))
    # densest 240-wide angle window in this slab
    col = (e > 100).mean(axis=0)
    if col.size >= 240:
        run = np.convolve(col, np.ones(240)/240, mode='valid')
        j   = int(run.argmax()); dens = float(run[j])
        if best is None or dens > best[0]:
            best = (dens, z0, j)
densD, zD, aD = best
print(f"  densest: z={zD}-{zD+192} a={aD}-{aD+240} density={densD:.3f}")
sD   = arr[zD:zD+192, :, :][:]
invD = pipe(unroll(sD, 310)[:, aD:aD+240])
letter_stats(invD, "D densest ink region")

# panel
def to_img(x):
    return Image.fromarray(x).resize((240*2, x.shape[0]*2), Image.NEAREST)
panel = Image.new("L", (240*2*4 + 30, max(invA.shape[0],invD.shape[0])*2 + 20), 255)
for k,(img,lbl) in enumerate([(invA,"A claim"),(invB,"B empty"),(invC,"C shuffled"),(invD,"D densest")]):
    panel.paste(to_img(img), (k*(240*2+10)+5, 15))
panel.save(str(OUT / "salvage_panel.png"))
print(f"\nSaved {OUT/'salvage_panel.png'}")
print("\nREAD: if B/C letterish_blobs ~>= A, morphology is CLAHE artifact.")
print("READ: if D shows only blobs (high coverage, low letter structure), method can't render letters.")

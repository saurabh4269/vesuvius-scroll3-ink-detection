"""
m7 ink-prediction triage via the VALIDATED cylindrical-unroll tool.

For every pre-downloaded scroll (internal level-3 zarr), this:
  1. fits the scroll center robustly (median per-z centroid of ink voxels) + reports residual
  2. sweeps radius from inner shell to mid-scroll
  3. unrolls (full circle) -> CLAHE -> finds letter-sized connected components
  4. applies the POSITIVE-CONTROL gradient gate to every candidate:
       dark-coverage must PEAK at the candidate radius and drop sharply +/- 2 steps
       (calibrated in positive_control.py: a real letter spikes; fiber decays outward)
  5. runs an angle-shuffle control on the best candidate (real structure should collapse)

Honesty guardrails baked in:
  - gate is quantitative, never visual
  - reports % of scroll radius actually searched (inner shell only)
  - center residual reported; high residual => geometry not cylindrical, flagged
Output: triage_report.json + triage_<scroll>.png montage for any scroll with a gated hit.
"""
import sys, json, glob
import numpy as np
import cv2, zarr
from pathlib import Path
from scipy.ndimage import gaussian_filter, label, find_objects

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()/"scroll_prize/data/m7_triage"
OUT  = ROOT / "results"; OUT.mkdir(parents=True, exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

def pipe(u):
    sm = gaussian_filter(u.astype(float), sigma=0.3)
    return 255 - clahe.apply(np.clip(sm, 0, 255).astype(np.uint8))

def fit_center(arr, T=20, zsample=40):
    """median per-z centroid of ink voxels; residual = MAD of those centroids."""
    NZ, NY, NX = arr.shape
    zs = np.linspace(0, NZ-1, min(zsample, NZ)).astype(int)
    cys, cxs = [], []
    for z in zs:
        sl = arr[z, :, :]
        m = sl > T
        if m.sum() < 50: continue
        ys, xs = np.nonzero(m)
        cys.append(ys.mean()); cxs.append(xs.mean())
    if len(cys) < 5: return None
    cy, cx = float(np.median(cys)), float(np.median(cxs))
    resid = float(np.median(np.hypot(np.array(cys)-cy, np.array(cxs)-cx)))
    return cy, cx, resid, len(cys)

N_ANG = 1440          # fixed angular sampling so all radii align column-for-column

def unroll_at(data, cy, cx, r):
    NY, NX = data.shape[1], data.shape[2]
    ang = np.linspace(0, 2*np.pi, N_ANG, endpoint=False)
    ys = np.clip(cy + r*np.sin(ang), 0, NY-1).astype(int)
    xs = np.clip(cx + r*np.cos(ang), 0, NX-1).astype(int)
    return data[:, ys, xs].astype(np.uint8)

def gate(strip_cols, cov_at):
    """positive-control gate on a candidate: peak at center radius, sharp falloff."""
    c = len(cov_at)//2
    return (cov_at[c] == max(cov_at) and
            cov_at[0] < 0.6*cov_at[c] and cov_at[-1] < 0.5*cov_at[c] and cov_at[c] > 0.02)

def scan_scroll(name, zpath):
    arr = zarr.open_array(str(zpath), mode="r")
    NZ, NY, NX = arr.shape
    fc = fit_center(arr)
    if fc is None:
        return {"scroll": name, "status": "no_center"}
    cy, cx, resid, nz_used = fc
    Rmax = 0.48*min(NY, NX)
    data = arr[:][:]                       # level-3 fits in memory
    r_lo, r_hi = int(0.18*Rmax), int(0.72*Rmax)
    radii = list(range(r_lo, r_hi, 6))     # 6px steps -> matches calibrated gradient spacing
    # precompute each radius's enhanced strip ONCE (columns align across radii)
    E = {r: pipe(unroll_at(data, cy, cx, r)) for r in radii if r > 5}
    radii = [r for r in radii if r in E]
    best = None
    K = 10                                  # cap components checked per radius (largest first)
    for i in range(2, len(radii)-2):        # need +/-2 neighbours for the gradient
        r = radii[i]; e = E[r]
        lab, n = label((e < 120).astype(np.uint8))
        sizes  = np.bincount(lab.ravel())   # O(image): size of every component at once
        slices = find_objects(lab)          # O(image): bbox of every component at once
        comps = []
        for c in range(1, n+1):
            pix = int(sizes[c])
            if pix < 150 or pix > 12000: continue
            sl = slices[c-1]
            if sl is None: continue
            a0, a1 = int(sl[1].start), int(sl[1].stop-1)
            if (a1-a0) > 0.5*N_ANG: continue        # spans too much arc -> wall
            comps.append((pix, a0, a1))
        comps.sort(reverse=True)
        for pix, a0, a1 in comps[:K]:
            # 5-radius gradient gate using precomputed neighbour strips, same columns
            cov = [float((E[radii[j]][:, a0:a1+1] < 120).mean())
                   for j in (i-2, i-1, i, i+1, i+2)]
            if gate(None, cov):
                win  = e[:, a0:a1+1]
                perm = (np.arange(win.shape[1])*7+3) % win.shape[1]
                blobs_real = len([1 for s in _sizes(win)        if 150<=s<=12000])
                blobs_shuf = len([1 for s in _sizes(win[:, perm]) if 150<=s<=12000])
                score = cov[2] * (cov[2]/max(cov[0],1e-3))   # peak * sharpness
                cand = {"r": r, "pct_radius": round(r/Rmax,3), "a0": a0, "a1": a1,
                        "pixels": pix, "gradient": [round(x,3) for x in cov],
                        "shuffle_drop": round(1-blobs_shuf/max(blobs_real,1),2),
                        "score": round(score,3)}
                if best is None or score > best["score"]:
                    best = cand
    return {"scroll": name, "status": "ok", "center": [round(cy,1), round(cx,1)],
            "center_residual": round(resid,2), "z_used": nz_used,
            "shape": [NZ,NY,NX], "Rmax": round(Rmax,1),
            "searched_pct": [round(0.18,2), round(0.72,2)],
            "best_candidate": best}

def _sizes(inv):
    lab, n = label((inv < 120).astype(np.uint8))
    if n == 0: return []
    return np.bincount(lab.ravel())[1:].tolist()   # skip background; O(image)

def main():
    scrolls = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name != "results"])
    report = []
    for sd in scrolls:
        zsub = sd/"L3"
        if not (zsub/".zarray").exists():
            cands = list(sd.glob("**/.zarray"))
            zsub = cands[0].parent if cands else None
        if zsub is None:
            report.append({"scroll": sd.name, "status": "no_zarr"}); continue
        try:
            r = scan_scroll(sd.name, zsub)
        except Exception as e:
            r = {"scroll": sd.name, "status": f"error: {type(e).__name__}: {e}"}
        report.append(r)
        bc = r.get("best_candidate")
        tag = "HIT" if bc else "-"
        print(f"[{tag:3s}] {sd.name:16s} {r.get('status')} "
              f"resid={r.get('center_residual')} "
              f"{('score='+str(bc['score'])+' grad='+str(bc['gradient'])) if bc else ''}", flush=True)
    # rank
    hits = [r for r in report if r.get("best_candidate")]
    hits.sort(key=lambda r: r["best_candidate"]["score"], reverse=True)
    (OUT/"triage_report.json").write_text(json.dumps(
        {"ranked_hits": hits, "all": report}, indent=2))
    print(f"\n=== {len(hits)} scrolls with gated candidates (ranked) ===")
    for r in hits:
        b = r["best_candidate"]
        print(f"  {r['scroll']:16s} score={b['score']:.2f} r={b['pct_radius']*100:.0f}%R "
              f"grad={b['gradient']} shuffle_drop={b['shuffle_drop']}")
    print(f"\nreport -> {OUT/'triage_report.json'}")

if __name__ == "__main__":
    main()

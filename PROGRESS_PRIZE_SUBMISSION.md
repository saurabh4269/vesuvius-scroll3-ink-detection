# Vesuvius Challenge — June 2026 Progress Prize Submission

**GitHub:** https://github.com/saurabh4269/vesuvius-scroll3-ink-detection  
**Submission form:** https://forms.gle/LrpQmSAqdwGpTczLA  
**Target tier:** Denarius ($10k) — or Gold Aureus ($20k) if adopted  
**Deadline:** June 30, 2026

---

## One-line summary

A validated, open-source tool that reads 3D ink-prediction zarrs directly via cylindrical polar unrolling — no Volume Cartographer required — with a calibrated gradient gate and built-in pareidolia controls, applied systematically to all 35 scrolls with m7 predictions in the open-data bucket.

---

## The problem this solves

The standard workflow to examine ink predictions requires a complete VolumeCartographer surface mesh — a slow, geometry-dependent step that can take hours per segment and fails on severely deformed scrolls. The m7_nnUNet team has published **3D ink-prediction zarrs for ~35 scrolls**, but most contributors look at them only through 2D projections, missing the radial (depth) structure that can distinguish real ink from artifacts.

This tool bypasses segmentation entirely: it samples the 3D prediction volume along concentric circular arcs, unrolling the scroll directly in 3D — applicable to any roughly-cylindrical scroll within seconds of zarr download.

---

## What was built

### 1. 3D cylindrical polar unrolling

Sample the 3D zarr along circles of radius `r` centered on the scroll axis. Varying `r` sweeps through the layers like peeling an onion:

```python
angles = np.linspace(0, 2*pi, 1440, endpoint=False)
ys = clip(cy + r*sin(angles), 0, NY-1).astype(int)
xs = clip(cx + r*cos(angles), 0, NX-1).astype(int)
strip = data[:, ys, xs]   # (NZ, 1440) — 2D view of one radial shell
```

Works at any zarr resolution level. Level-3 (~9.6 µm/px) is cheap enough to scan a full scroll in seconds; level-0 (1.2 µm/px) resolves individual papyrus fibers.

### 2. Calibrated depth gradient gate

Real carbonized ink sits in a thin layer (~15–50 µm) on the papyrus surface. Sampling at the ink radius should produce a **sharp spike** — high signal at one depth, near-empty on both sides. Two confounders produce completely different profiles:

- **Papyrus fiber:** peaks at the inner edge and fades outward (signal at `r-2` > `r`)
- **Chunk-aligned saturation:** flat block, coarse spatial resolution, not radius-specific

The gate checks: does coverage peak at the candidate radius and drop significantly on both sides?

**Calibration via positive control** (`scripts/positive_control.py`): Greek letters (Π Ο Β Φ) were painted onto a synthetic cylindrical shell at a known radius, embedded in real zarr fiber/noise values, then processed through the identical pipeline. This established:

- What the pipeline looks like when it works (letters render legibly — the method is sound)
- The numerical gradient signature of a real ink shell:
  `r-2: 0.002 | r-1: 0.002 | r: 0.218 | r+1: 0.000 | r+2: 0.000`

The gate threshold was set from this calibration, not from a visual inspection of the candidate.

![Positive control panel](results/report/poscontrol_panel.png)
*Left: ground truth. Centre: through 2D CLAHE readout. Right: through full 3D cylindrical sampling. Letters recover legibly — the readout chain is trustworthy.*

### 3. Built-in pareidolia controls

CLAHE (the contrast-enhancement step) can manufacture letter-like patterns from noise. Two controls are run automatically on any candidate that passes the gate:

- **Angle-shuffle control:** randomise the angular axis, preserving brightness values but destroying spatial coherence. Real structure (ink strokes) disappears; CLAHE texture survives. The `shuffle_drop` metric measures how much blob count falls.
- **Salvage controls** (`scripts/salvage_test.py`): run the same pipeline on an empty region (B) and on the model's densest ink zone (D). B should produce no letter blobs; D reveals whether the pipeline renders known dense ink as legible letters.

### 4. 35-scroll systematic triage (`scripts/triage_scan.py`)

Applied the validated tool to every scroll in `s3://vesuvius-challenge-open-data/` that has an m7 surface-prediction zarr — 35 scrolls total. For each:

1. Fit the scroll center robustly (median per-z ink centroid, report residual)
2. Precompute enhanced strips at all radii once (O(image), not O(candidates×radii))
3. Find connected components sized for letter-scale structures (150–12,000 px at ~9.6 µm/px)
4. Apply the gradient gate and shuffle control to every candidate
5. Output a ranked JSON report with all metrics

Scripts: `triage_prep.sh` (downloads level-3 of all zarrs), `triage_scan.py` (the scan), `triage.sbatch` (SLURM job for Prajna/similar HPC). Runtime: ~2 hours for 35 scrolls on a single L40 GPU node.

---

## Results: 35-scroll triage

```
Scrolls scanned:    35
Gated candidates:    8   (passed gradient gate)
After shuffle filter: 3   (shuffle_drop > 0 — some spatial coherence)
Approaching ink signature: 0  (none match the calibrated thin-shell profile)
```

**Full ranked output:**

| Scroll | Score | Resid | Gradient [r-2..r+2] | Shuffle drop | Verdict |
|--------|-------|-------|---------------------|--------------|---------|
| PHerc0846B | 1.24 | 33.8 | [0.17, 0.45, **0.46**, 0.13, 0.13] | +0.17 | Weak |
| PHerc1451  | 0.77 | 19.8 | [0.22, 0.22, **0.41**, 0.18, 0.20] | +0.33 | Candidate |
| PHerc0125  | 0.76 | 33.8 | [0.22, 0.37, **0.41**, 0.41, 0.20] | **−0.60** | Artifact |
| PHercMANB  | 0.49 | 53.9 | [0.08, 0.07, **0.20**, 0.06, 0.07] | +0.40 | Candidate |
| PHerc0139  | 0.40 | 32.9 | [0.12, 0.10, **0.22**, 0.21, 0.11] | +0.20 | Weak |
| PHercMAN5  | 0.37 | 22.0 | [0.11, 0.11, **0.20**, 0.18, 0.09] | 0.00 | Artifact |
| PHerc1447  | 0.33 | 57.8 | [0.10, 0.14, **0.18**, 0.11, 0.05] | **−0.50** | Artifact |
| PHerc0826  | 0.32 | 92.3 | [0.09, 0.13, **0.17**, 0.07, 0.02] | 0.00 | Artifact |

*Residual = median deviation of per-z scroll centers from the fitted global center (px). Higher = less cylindrical = unrolling less reliable. Shuffle drop ≤ 0 = structure survives randomisation = CLAHE artifact.*

**Compared to the calibrated real-letter signature:**

| | r-2 | r-1 | **r (peak)** | r+1 | r+2 | inner/peak | outer/peak |
|---|---|---|---|---|---|---|---|
| **Known real letter** | 0.002 | 0.002 | **0.218** | 0.000 | 0.000 | 0.01× | 0.00× |
| PHerc.332 fiber (retracted) | 0.164 | 0.137 | 0.098 | 0.050 | 0.000 | 1.67× | — |
| PHercMANB (best candidate) | 0.078 | 0.074 | **0.195** | 0.062 | 0.070 | 0.40× | 0.36× |
| PHerc1451 | 0.217 | 0.215 | **0.408** | 0.176 | 0.198 | 0.53× | 0.49× |

No candidate approaches the thin-shell spike of a real ink layer. The three surviving candidates show broad radial profiles consistent with thick density variations or overlapping layers, not a single inked sheet.

**Honest bottom line:** the tool scanned every available m7 prediction, applied calibrated filters, and found no confirmed ink candidates. That is a clean, reproducible negative result — itself informative (the inner shell of these scrolls does not contain easily-readable isolated letters under cylindrical unrolling at 9.6 µm/px).

---

## Secondary contribution: BCE loss fix

Identified a silent bug in 2D ink-detection training. BCE loss was computed against both label channels (ink mask + validity mask), producing contradictory gradients that caused predictions to saturate at 0.5:

```python
# wrong — both channels, contradictory gradients
loss = BCE(logits, y.float())          # y is (B, 2, H, W)

# correct — ink channel only, weighted for class imbalance
loss = BCE(logits, y[:,0,:,:].float(), pos_weight=tensor([10.0]).to(device))
```

Effect: improved Segformer-B1 high-confidence ink fraction on Scroll 3 from 0% → 5.93% (`>0.9` threshold). Best model: B1 backbone, 50 epochs, pos_weight=10.

---

## What makes this reusable

| Property | Details |
|----------|---------|
| **No new data required** | Works on the public m7 zarrs already in the open-data bucket |
| **No segmentation required** | Bypasses Volume Cartographer / ThaumatoAnakalyptor entirely |
| **Calibrated, not visual** | Gradient gate set from positive control, not by eye |
| **Controls built in** | Shuffle drop and salvage panel prevent pareidolia claims |
| **Any scroll** | `triage_scan.py` runs on any roughly-cylindrical scroll with an m7 zarr |
| **Reproducible** | Public data + public code + exact zarr coordinates in JSON report |
| **Honest** | Negative result reported as clearly as a positive one would be |

Anyone can run `triage_prep.sh` + `triage.sbatch` on their own cluster (or locally on the L0 subset) to get the same ranked output.

---

## Repository structure

```
scripts/
├── positive_control.py      ← calibration: validates the readout chain
├── salvage_test.py          ← pareidolia controls (shuffle, empty, densest)
├── triage_prep.sh           ← downloads level-3 of all 35 zarrs
├── triage_scan.py           ← 35-scroll scan with gradient gate + controls
├── triage.sbatch            ← SLURM job for HPC
├── full_scroll_scan.py      ← z-profile sweep for a single scroll
├── inspect_zones.py         ← per-zone gradient comparison
├── r310_fullsearch.py       ← full r=310 sweep on PHerc.332 (validates 0)
├── level0_letter_zoom.py    ← 1.2 µm/px full-resolution analysis
└── train_full.py / ft_esrf_b1.py  ← BCE-fix training pipeline

results/report/
├── poscontrol_panel.png     ← positive control: truth | 2D | 3D-sampled
├── salvage_panel.png        ← controls: claim | empty | shuffled | densest
├── v9_l0_5radius.png        ← 5-radius gradient at 1.2 µm/px
└── v9_l0_maxzoom.png        ← PHerc.332 candidate at full resolution
```

---

## Known limitations (stated honestly)

1. **Cylindrical assumption** — works best on scrolls with low center residual. High-residual scrolls (resid > 50) have unreliable unrolling. 10 of 35 scrolls had resid > 50; their results are less trustworthy.
2. **Inner shell only** — at level-3, only the innermost accessible shell (~18–72% of scroll radius) is searchable. Outer layers saturate. A result here is "not in the accessible shell," not "not in the scroll."
3. **Level-3 resolution** — 9.6 µm/px is enough for letter-scale structures (~0.5–2 mm) but not fine brush strokes. Some real ink might be sub-threshold.
4. **Cylindrical unrolling smears deformed text** — if the papyrus sheet wanders significantly from the assumed circle, strokes get smeared into illegibility. The method works best on the innermost, most-cylindrical layers.

---

## Zarr coordinates for independent verification

```python
# PHerc.332 — the scroll we examined most thoroughly (run r310_fullsearch.py)
zarr_path = "s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr/2/"
# center cy=496.0, cx=534.4 | r_ink=310px | result: 0 gated candidates

# Triage report (all 35 scrolls)
# ~/scroll_prize/data/m7_triage/results/triage_report.json  (on HPC)
# or run triage_prep.sh + triage_scan.py to reproduce from scratch
```

---

## What would move this from Denarius to Gold Aureus

The tool reaches Gold Aureus tier if adopted by the community — i.e. another contributor runs it, finds something, or extends it. Possible extension points:
- Run on **Scroll 2 (PHercParis4)** once its m7 zarr is published
- Adapt the center-fit to handle non-circular scroll geometry (ellipse or spline fit)
- Add a Level-0 zoom step that auto-runs when a candidate survives triage, so the full pipeline from S3 download to full-resolution render is one command

All code is MIT licensed and ready to use.

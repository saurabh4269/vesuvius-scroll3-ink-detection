# Vesuvius Challenge — June 2026 Progress Prize Submission

**GitHub:** https://github.com/saurabh4269/vesuvius-scroll3-ink-detection  
**Submission form:** https://forms.gle/LrpQmSAqdwGpTczLA  
**Target tier:** Sestertius ($2,500) – Denarius ($10k)  
**Deadline:** June 30, 2026

---

## One-line summary

A BCE loss fix that raises ink-detection confidence from 0% to 5.93% on Scroll 3, plus a documented methodology for validating 3D zarr readout pipelines with positive controls and pareidolia tests.

### Why this is timely

Per the Vesuvius Challenge master plan, the 2023/24 ink models amplify signals that correspond to morphological cracks or metal-rich bright spots, and these **do not generalize** to the newer scrolls — "ink remains elusive in all our new data." In that environment, the prize criteria for First Letters/Title explicitly require **hallucination mitigation** and warn against window sizes larger than 0.5×0.5 mm. Both contributions below are aimed squarely at that reality: a correctness fix in the training loss, and a reusable methodology for proving a readout pipeline isn't manufacturing letters from noise. We are not claiming a reading; we are contributing tooling that makes claims trustworthy.

---

## Contribution 1 — BCE loss fix (primary)

### The bug

When training a 2D ink-detection model on ESRF fragments, BCE loss was computed against both label channels of a `(B, 2, H, W)` label tensor. Channel 0 is the ink mask; channel 1 is an all-ones validity mask. Training against both produces contradictory gradients — the model receives equal signal to predict ink and to predict no-ink at every pixel — causing predictions to saturate near 0.5.

```python
# wrong — both channels, contradictory gradients
loss = BCE(logits, y.float())                      # y shape: (B, 2, H, W)

# correct — ink channel only, weighted for class imbalance
loss = BCE(logits, y[:, 0, :, :].float(),
           pos_weight=tensor([10.0]).to(device))
```

### The fix

Slice to ink channel only (`y[:, 0, :, :]`) and add `pos_weight=10` to compensate for class imbalance (~9% ink pixels → ~10× more background than ink).

### Measured effect

| Model | Epochs | Val loss | >0.9 confidence |
|-------|--------|----------|----------------|
| B3, no pos_weight, 30ep | 30 | 0.634 | **0.00%** |
| B3, pos_weight=10, 30ep | 30 | 1.736 | 0.43% |
| B1, pos_weight=10, 50ep | 50 | 1.631 | **5.93%** |

Segment: Scroll 3 fragment 20240618142020. Hardware: Prajna HPC, NVIDIA A40.

### Who this affects

Any contributor training a 2D ink-detection model on ESRF-format fragments where labels are stored as multi-channel tensors with a validity mask in channel 1. The bug silently produces a trained model that looks converged (loss decreases) but outputs near-0.5 everywhere.

---

## Contribution 2 — Documented methodology: positive control + pareidolia testing

### The problem

CLAHE (contrast-limited adaptive histogram equalization) is widely used to enhance faint ink marks in zarr data. It can also manufacture convincing letter-like patterns from noise. Without systematic controls, any claimed finding from a CLAHE-enhanced zarr is vulnerable to being a visual artifact.

### What we built

**Positive control** (`scripts/positive_control.py`): paints known Greek letters (Π Ο Β Φ) onto a synthetic cylindrical shell at a known radius, embeds them in real zarr fiber/noise values, runs the identical processing pipeline. Confirms the readout chain is trustworthy before applying it to real data.

**Pareidolia controls** (`scripts/salvage_test.py`): for any candidate structure, runs the same pipeline on (a) an empty region, (b) the candidate window with its angle axis shuffled (destroys spatial coherence, keeps brightness values). Quantifies `shuffle_drop` — how much structure disappears when spatial coherence is removed. If shuffle_drop ≤ 0, the structure is CLAHE texture, not signal.

**Calibrated gradient gate** (`scripts/r310_fullsearch.py`, `scripts/inspect_zones.py`): samples at 5 radii spaced 6px apart. A real thin ink layer spikes sharply at one radius; papyrus fiber peaks at the inner edge and decays. The gate threshold is set numerically from the positive control, not visually.

### Why this matters

We applied these controls to our own claimed findings and they failed. The gradient gate validated 0 of 10 candidates (including the one we had posted publicly). The salvage controls confirmed: angle-shuffling retained ~half the "letter-like" blobs (CLAHE texture) and the densest zarr region rendered as blocks, not letters. This disproved our own claim before anyone else had to. The controls are the contribution — they can be applied to any future zarr-based finding.

### Correction documented in this repo

In the process of this work we also discovered and documented that the m7 zarrs (`representations/predictions/surfaces/...surface-m7...`) are **papyrus surface/sheet localization predictions**, not ink predictions. They predict which voxels are part of the physical papyrus sheet in 3D space — used as input to ThaumatoAnakalyptor and VolumeCartographer. We had been calling them "ink predictions" throughout. This correction is documented in the README and knowledge base for the community.

---

## What this is not

- Not a letter finding — all candidates retracted
- Not a working ink-detection tool for arbitrary scrolls — the BCE fix applies to 2D models trained on ESRF fragments
- Not a replacement for segmentation — the cylindrical unrolling methodology only works on roughly-cylindrical scroll geometry and scans surface predictions, not ink

---

## Code

All scripts are in `scripts/`. Key files:

| Script | Purpose |
|---|---|
| `train_full.py` + `ft_esrf_b1.py` | BCE-fix training pipeline (Segformer-B1, ESRF data) |
| `positive_control.py` | Validates readout chain on any zarr |
| `salvage_test.py` | Pareidolia controls for any candidate |
| `inspect_zones.py` | 5-radius gradient comparison |
| `r310_fullsearch.py` | Full gradient-gate scan (returns 0 — the honest result) |

MIT licensed.

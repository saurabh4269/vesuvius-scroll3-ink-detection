# Scroll Prize — Data Inventory & Contribution Strategy

> Data explored: 2026-05-29 on Prajna via `scroll` conda env.
> Strategy research: 2026-05-29. Sources: scrollprize.org, substack, GitHub issues, Kaggle writeups.

---

## Part 1 — Data Inventory

### Scroll Volumes

All scrolls: OME-Zarr, 6 multi-resolution levels (level 0 = full res, each level = 2× downsampled),
128³ chunks, uint8, voxel size 7.91 µm.

| Scroll | ID | Physical size (z×y×x mm) | Shape at level 0 | Level 3 shape | Notes |
|--------|----|--------------------------|------------------|---------------|-------|
| Scroll 1 | PHerc. Paris 4 | 114×62×64 | 14376×7888×8096 | 1797×986×1012 | Grand Prize scroll; reference |
| Scroll 2 | PHerc. Paris 3 | 114×80×95 | 14428×10112×11984 | 1804×1264×1498 | **Prize target.** Scan artifact in centre. |
| Scroll 3 | PHerc. 332 | **77×28×27** | 9778×3550×3400 | 1223×444×425 | **Prize target. Smallest — best to start.** |
| Scroll 4 | PHerc. 1667 | — | ~same range | — | Letters found Dec 2025 @ 2.4 µm. Not a prize target. |
| Scroll 5 | PHerc. 172 | — | — | — | Title found May 2025. Not a prize target. |

#### Intensity Statistics (centre slab sample)

| Scroll | Mean | Max | Non-zero fraction | Comment |
|--------|------|-----|-------------------|---------|
| Scroll 1 | 63.8 | 248 | 92% | Strong signal, bimodal histogram |
| Scroll 2 | 47.3 | 248 | 87% | Weaker signal; more air pockets |
| Scroll 3 | 44.9 | 248 | 95% | Low-contrast, peaks ≤159 — **needs strong CLAHE** |

Access pattern for scrolls not in vesuvius.Volume() config (e.g. Scroll 3):
```python
import fsspec, zarr
url = "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231027191953.zarr"
store = zarr.open_group(fsspec.get_mapper(url), mode='r')
level3 = store['3']   # shape: (1223, 444, 425)
```

---

### Scroll Segments (surface already rendered — ready for ink detection)

#### Scroll 2 — 59 segments

| Segment ID | Area | Layers | Layer shape | Notes |
|------------|------|--------|-------------|-------|
| 20240516205750 | **26.0 cm²** | 65 | 3079×19931 | Most recent (2024), largest — start here |
| (58 others) | varies | 65 | varies | All 2023 exploration, mostly tiny |

Only the 2024 segment is practically useful. 4 cm² needed for prize → 26 cm² gives 6× headroom.

#### Scroll 3 — 14 segments

| Segment ID | Area | Layers | Layer shape | Notes |
|------------|------|--------|-------------|-------|
| 20240618142020 | **33.5 cm²** | 65 | 2491×25706 | **Best starting point — largest, 2024** |
| 20231030220150 | 2.58 cm² | 157 | 9417×10745 | 157 layers (unusual) — high depth sampling |
| 20240712064330 | 1.59 cm² | 65 | 995×2755 | Small |
| 20240702133100_thaumato_20231117143551 | ? | ? | ? | Thaumato autoseg — may cover more area |
| (10 others) | — | — | — | |

**Scroll 3 is the better First Letters target:** physically smallest (27 mm × 28 mm cross-section),
already has 33.5 cm² (6× prize requirement), fewer wraps = simpler geometry.

Base URL: `https://dl.ash2txt.org/full-scrolls/Scroll{2,3}/`

---

### ESRF Training Fragments (ground truth for ink detection)

Three fragments scanned at ESRF Grenoble in May 2025 at 2.2 µm.

| Fragment | Volume res | Surface layers | Layer shape | Ink label | Ink fraction | Usable? |
|----------|------------|----------------|-------------|-----------|-------------|---------|
| PHerc.500P2 | 2.215 µm | 66 PNG (00–65) | 27160×14990 | `500P2_inklabels.png` RGBA, binary alpha | **28%** | **YES** |
| PHerc.343P | 2.215 µm | 66 PNG (00–65) | 13924×8416 | `343P_inklabels.png` RGBA, R channel 0–9 | 3.4% | YES (caveat) |
| PHerc.9B | 4.32 µm only | none | — | none | — | **NO** |

**343P caveat:** R channel values are 0–9, not standard 0/255 binary. May be confidence scores or
multi-class. Investigate before using as training ground truth.

**9B gap:** No surface segment, no ink labels publicly available. Literature claims "trained on 500P2+343P+9B"
are imprecise or reference internal labels. **Effective public training data = 500P2 + 343P only.**

Base URL: `https://dl.ash2txt.org/fragments/PHerc{0500P2,0343P,0009B}/`

**Accessing 500P2 layers (PNG, not TIFF):**
```python
from PIL import Image
import requests, io
Image.MAX_IMAGE_PIXELS = None   # required — label is 407M pixels
url = "https://dl.ash2txt.org/fragments/PHerc0500P2/layers/00.png"
img = Image.open(io.BytesIO(requests.get(url).content))
```

---

### DLS Fragments (Legacy Training Data — Still Valid)

Six fragments from Diamond Light Source (Oxford) at 3.24 µm with hand-labeled ink masks.
Fragments 1–6 at `dl.ash2txt.org/fragments/Frag{1-6}/`.

---

### First Title Winner Code

Repo at `~/scroll_prize/vesuvius_first_title_prize/` (MiniUNETR + SegFormer-B3).

```
in_chans=16, patch_size=128, label_size=32
encoder: nvidia/mit-b3
lr=2e-4 cosine, batch=32, epochs~14, ink_ratio=5, contrasted=True (CLAHE)
```

Entry: `train_title.sh`. Inference: `scripts/inference.py`.
`create_dataset.py` **currently targets Scroll 5 auto-segmentations** — must be adapted for
Scroll 3's `layers/` format before use.

**Dependency gaps (not yet installed):**
```bash
conda activate scroll
pip install vesuvius-phalanx zarr==2.16.1 numcodecs==0.12.1 \
    batchgenerators nnunetv2 python-dotenv rich imagecodecs imageio
```
Note: we have zarr==2.18.3. Test before pinning to 2.16.1.

---

## Part 2 — State of Play

### What Has Been Read

| Scroll | Status |
|--------|--------|
| Scroll 1 (PHerc. Paris 4) | ~15 columns read. Philodemus, *On Pleasure*. Grand Prize awarded Feb 2024. |
| Scroll 5 (PHerc. 172) | Title read: *On Vices* by Philodemus. First Title Prize awarded May 2025. |
| Scroll 4 (PHerc. 1667) | Letters found Dec 2025 using 2.4 µm ESRF scan + generalist model. Prize retired. |
| **Scrolls 2 & 3** | **No letters found. Both First Letters prizes ($60k each) OPEN.** |

### The Two Unsolved Bottlenecks

1. **Ink detection generalization** — models trained on Scroll 1/5 don't transfer to most others.
   The Scroll 4 breakthrough required 2.4 µm (3× finer). A generalist model trained on ESRF fragments
   (500P2+343P) did generalize without scroll-specific fine-tuning. This is the current frontier.

2. **Unwrapping at scale** — still costs $1–5M per scroll. $200k prize for automating ≥70% of two
   scrolls. Kaggle Surface Detection competition (Feb 2026, $100k) just ended — nnUNet ensemble won
   and solutions are now public.

---

## Part 3 — Open Prizes

| Prize | Amount | Target | Status |
|-------|--------|--------|--------|
| First Letters (Scroll 2) | $60,000 | 10 letters in 4 cm² | **OPEN** |
| First Letters (Scroll 3) | $60,000 | 10 letters in 4 cm² | **OPEN** |
| First Title (Scroll 2) | $60,000 | Title of Scroll 2 | **OPEN** |
| First Title (Scroll 3) | $60,000 | Title of Scroll 3 | **OPEN** |
| Unwrapping at Scale | $200,000 | Automate ≥70% of 2 scrolls | **OPEN** |
| Progress Prizes (monthly) | $1k–$20k | Open-source tools, models, data | **OPEN** (last day of each month) |

---

## Part 4 — Prize Strategy

```
EASY ─────────────────────────────────────────────────────── HARD
[Progress Prizes] → [Ink Detection] → [First Letters] → [Unwrapping at Scale]
$1k–$10k/month      $2.5k–$20k        $60k×2             $200k
4–6 weeks           2–3 months         3–6 months          6–12 months
```

Run all tracks in parallel — progress prizes build reputation and income while working toward First Letters.

---

## Track A: Progress Prizes (Start Immediately)

**Tiers:** Papyrus ($1k), Sestertius ($2.5k), Denarius ($10k), Gold Aureus ($20k).
**What wins:** released early, well documented, actually used by community.
**Deadline:** 11:59pm Pacific, last day of each month. Submit: `forms.gle/LrpQmSAqdwGpTczLA`

### A1 — Good First Issues (Villa GitHub)

**Issue #201: 3D scroll augmentations**
- Augmentation functions tailored to scroll geometry (curvature, fiber direction, layer topology)
- Extend torchvision/albumentations with scroll-aware transforms + ablation study
- Expected: Sestertius–Denarius ($2.5k–$10k). Compute: l40, 1 GPU, short runs.

**Issue #192: Accurate 3D ink labels**
- Methodology for volumetric ink labels in 3D (current labels are 2D projections)
- Use ESRF 2.2 µm data; auto-label propagation from 2D IR ground truth
- Expected: Sestertius–Denarius ($2.5k–$10k).

**Issue #199: Update out-of-date documentation**
- Pure documentation PR. Expected: Papyrus ($1k). Time: 1–2 days.

**Issue #193: Methods for generating surface, fiber, and ink labels**
- Automated pipelines for training labels at scale.
- Could combine with #192. Expected: Denarius ($10k).

### A2 — Medium-Effort

**Knowledge distillation: 2 µm → 9 µm model transfer**
- Teacher on ESRF 2.2 µm → student on 9 µm with soft targets + feature distillation
- If it works: unlocks Scrolls 2/3. Expected: Gold Aureus ($20k). Compute: dgx (A100), multi-GPU.

**VC3D sheet-switch detection**
- Automated detection + correction of mesh jumping between adjacent papyrus layers
- Directly addresses the Unwrapping at Scale bottleneck. Expected: Denarius ($10k).

---

## Track B: First Letters Prize — Scroll 3 ($60k)

### The Winning Pipeline

```
Existing segment (20240618142020, 33.5 cm²)
    ↓
Surface volume layers already rendered (layers/ exists, 65 layers, uint16)
    ↓
CLAHE contrast enhancement (contrasted=True in MiniUNETR, or manual)
    ↓
Generalist ink detection model (MiniUNETR trained on 500P2+343P)
    ↓
If signal found: iterative labeling + retrain
    ↓
Papyrologist review via Discord → submission
```

### Phase B1: Get Data (Week 1)
- Install missing deps (see Part 1 §First Title Winner Code)
- Download Scroll 3 segment 20240618142020 layers to Prajna: ~65 layers × 2491×25706 × uint16 ≈ **21 GB**
- Adapt `create_dataset.py` for Scroll 3's `layers/` format

### Phase B2: Zero-Shot Inference (Weeks 2–3)
- Run MiniUNETR pretrained on fragment data → look for any faint letterforms
- Also try Grand Prize TimeSformer model (`villa/ink-detection/`)
- Apply strong CLAHE (clipLimit≥2.0, tileGridSize 8×8)

### Phase B3: Iterative Labeling (Weeks 3–8)
- If signal visible: annotate with crackle-viewer or khartes
- Use "ignore mask strategy" — don't force uncertain pixels
- MiniUNETR: ~1h per training run on A100 → fast iteration

### Phase B4: Verify and Submit
- Letters confirmed → methodology doc, Docker image, scale bar, 3D position
- Discord papyrologist channel → formal submission

### Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Base model | MiniUNETR (First Title winner) | 1h training, 10× faster than TimeSformer |
| Input | 16–21 layers surface volume | Same as Grand Prize winners |
| Training data | ESRF fragments 500P2+343P | Only usable public ground truth |
| Loss | Masked loss (ignore mask strategy) | Prevents label noise from hurting training |

### Compute Allocation (Prajna)

| Task | Partition | GPUs | Wall |
|------|-----------|------|------|
| Ink detection training | dgx (A100) | 1 | 2h |
| Hyperparameter sweep | l40 (L40S) | job array of 4 | 2h each |
| Full training run | dgx | 1–2 | 6h |
| Segmentation inference | a40 (A40) | 1 | 4h |
| Debug/sanity check | debug | 1 | 25 min |

---

## Track C: Unwrapping at Scale ($200k)

**Requirement:** Automate virtual unwrapping of ≥70% of two different scrolls.

### The Pipeline (Kaggle winner now public)

```
CT volume
  ↓ [Stage 1] Surface detection — nnUNet ensemble (Kaggle winner, NOW PUBLIC)
  ↓ [Stage 2] Instance segmentation — Mask3D or sliding-window 3D
  ↓ [Stage 3] Winding angle assignment — Thaumato graph or spiral fitting
  ↓ [Stage 4] Mesh + flattening — VC3D tracer + SLIM parametrization
Flat surface volumes → ink detection
```

### Contributions by Stage

**Stage 1 (tractable now):** Improve surface detection using Kaggle data — TTA, better post-processing,
ensemble more architectures. Expected: Sestertius–Gold Aureus ($2.5k–$20k).

**Stage 2:** Better instance segmentation in compressed regions (structure tensor analysis showing
"remarkable early results" per April 2026 progress prizes).

**Stage 3:** Winding angle as differentiable optimization or GNN.

---

## Part 5 — Execution Timeline

### Month 1 (June 2026) — Foundation + Quick Wins

| Week | Action |
|------|--------|
| 1 | Submit documentation fix PR (issue #199) → Papyrus prize ($1k) |
| 1 | Install missing deps, adapt `create_dataset.py`, download S3 segment layers |
| 2 | Implement 3D scroll augmentations (issue #201), ablation on fragment data |
| 3 | Set up MiniUNETR training on Prajna, reproduce First Title result on fragments |
| 4 | Submit augmentations + ablation → Sestertius/Denarius prize |

### Month 2 (July 2026) — Ink Detection Depth

| Week | Action |
|------|--------|
| 1–2 | Run zero-shot inference on Scroll 3 segment — document all signals seen |
| 3 | Knowledge distillation experiment: 2.2 µm teacher → 9 µm student |
| 4 | Submit distillation results as progress prize |

### Month 3 (August 2026) — First Letters Push

| Week | Action |
|------|--------|
| 1–2 | Iterative labeling on any faint Scroll 3 signals |
| 2–3 | MiniUNETR hyperparameter sweep (job arrays on Prajna) |
| 3–4 | If letters visible: papyrologist review via Discord, prepare submission |

### Month 4+ — Scale Up or Pivot

- If letters found: full pipeline to find title → second $60k prize
- If not: pivot to VC3D/segmentation contributions + Unwrapping at Scale
- Continue monthly progress prize submissions

---

## Part 6 — What Makes a Winning Contribution

1. **Release early** — tools released while the community still needs them win more than polished-but-late work.
2. **Community adoption** — judges look at whether others use and build on your work. Announce in Discord.
3. **Infrastructure over results** — most winners built tools/datasets, not just ink-reading results.
4. **Document everything** — walkthroughs, notebooks, videos. Undocumented tools rarely win.
5. **Modular design** — outputs in standard formats (OME-Zarr, quadmesh) that plug into existing pipeline.
6. **Iterate in public** — post interim results to Discord even before they're good.

---

## Part 7 — Key Risks

| Risk | Mitigation |
|------|-----------|
| Scroll 3 ink signal too faint at 9 µm | Knowledge distillation from 2 µm; more ESRF labels |
| Scroll 2 too compressed to segment | Work Scroll 3 in parallel; surface detection model |
| Someone else finds letters first | Progress prizes continue — ship tools regardless |
| Prajna scratch not provisioned | Use ~/scroll_prize/results/ until admin creates it |
| Monthly deadline missed | Always have a docs PR or small tool ready to submit |
| OOM during training | MiniUNETR designed for 24GB; A100 80GB has headroom |

---

## Part 8 — Key Resources

| Resource | Location | Use for |
|----------|----------|---------|
| First Title winner | `~/scroll_prize/vesuvius_first_title_prize/` | Base architecture (MiniUNETR) |
| Grand Prize winner | `~/scroll_prize/villa/ink-detection/` | TimeSformer reference |
| ESRF fragments | `dl.ash2txt.org/fragments/PHerc{0500P2,0343P}/` | Training data with ink |
| Scroll 3 best segment | `dl.ash2txt.org/.../Scroll3/.../20240618142020/layers/` | Target surface volumes |
| Kaggle Surface Det. solutions | kaggle.com/competitions/vesuvius-challenge-surface-detection | Segmentation pipeline |
| Khartes | `github.com/KhartesViewer/khartes` | Manual annotation |
| Crackle Viewer | `~/scroll_prize/villa/crackle-viewer/` | Ink labeling |
| Progress Prize form | `forms.gle/LrpQmSAqdwGpTczLA` | Monthly submission |
| Discord | `discord.gg/V4fJhvtaQn` | Community + papyrologist review |

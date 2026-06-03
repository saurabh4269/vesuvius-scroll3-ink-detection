# Knowledge Base — Scroll Prize

Clean reference as of 2026-06-04. No evolutionary log — only verified facts.

---

## 1. The Competition

**Vesuvius Challenge** — read carbonized 2,000-year-old Herculaneum papyrus scrolls using X-ray CT + ML.

| Prize | Amount | Requirement | Status |
|-------|--------|-------------|--------|
| First Letters — Scroll 3 (PHerc.332) | $60,000 | 10 readable letters in 4 cm² | Open, no winner |
| First Title — Scroll 3 | $60,000 | Read the scroll's title | Open, no winner |
| June Progress Prize (Gold Aureus) | $20,000 | Major tool/method adopted by community | **Jun 30, 2026 deadline** |
| Denarius | $10,000 | Significant contribution | Jun 30 |
| Sestertius | $2,500 | Notable contribution | Jun 30 |

Submission form: https://forms.gle/LrpQmSAqdwGpTczLA  
Only Scroll 3 (PHerc.332) is prize-eligible for First Letters/Title — other scrolls don't qualify.

---

## 2. The Data — What Each Type Actually Is

### CT Volumes
Raw X-ray scans. Stored as OME-Zarr in S3 (`s3://vesuvius-challenge-open-data/`). Multi-resolution pyramid (level 0 = finest, each level = 2× downsampled). Voxel size ~7.91 µm at level 0. Access: `aws s3 ... --no-sign-request` (public, no login).

### Surface Predictions (m7)
**CRITICAL: m7 is a papyrus sheet/surface localization model — NOT an ink detector.**

Path: `<scroll>/representations/predictions/surfaces/...surface-m7-L2-th0.2.zarr`

m7 predicts which voxels in the 3D CT volume are part of the physical papyrus sheet (medial surface). This is used as INPUT to ThaumatoAnakalyptor and VolumeCartographer to find and trace the sheet. The `th0.2` means voxels where the model is ≥20% confident the sheet passes through. ~14% nonzero at level-2 is consistent with a curved sheet occupying that fraction of the bounding box — not ink density.

**There are no publicly available 3D volumetric ink-prediction zarrs in the open-data bucket.**

### Segments (flat surface volumes)
Output of segmentation (ThaumatoAnakalyptor / VolumeCartographer). A flattened 2D+depth representation of a traced sheet section. These are what ink detection models actually run on.

### Ink Labels / 2D Ink Predictions
2D outputs from ink detection models applied to segments. Our B1 model produces a 2D probability map for a given segment. These are not 3D volumes.

### ESRF Fragments
Ground-truth ink-labeled fragments from ESRF synchrotron scans. Used as training data. Labels stored as `(H, W, 2)` tensors — channel 0 = ink mask, channel 1 = all-ones validity mask. **Only use channel 0 for training.**

---

## 3. Villa (ScrollPrize/villa) — Foundation Codebase

**villa is already cloned at `~/scroll_prize/villa/` on Prajna.** Use it as the foundation — don't build from scratch.

### What villa gives us

| Component | What it is | Value to us |
|-----------|-----------|-------------|
| `vesuvius/` package | Data loading: `Volume` class, zarr access via `scrolls.yaml` | Standard API for scroll/segment access |
| `ink-detection/train_resnet3d.py` | ResNet3D + 3D decoder, GroupDRO, Lightning | Better architecture than our `train_full.py` |
| `ink-detection/all_labels/` | **45** labeled segment ink PNGs (Scroll 1/2) | More diverse training data than ESRF alone |
| `ink-detection/infer_resnet3d_vesuvius.py` | Inference on any zarr segment — takes `--zarr_path` directly | Run on Scroll 3 without config gymnastics |
| `wild14_deduped_64_pretrained2_...ckpt` | Pre-trained ResNet3D checkpoint (Scroll 1/2 trained) | Use as starting point or compare against our B1 |

### vesuvius package — how to use it

Data is served from `dl.ash2txt.org`, **not** from `s3://vesuvius-challenge-open-data`. The `scrolls.yaml` config resolves scroll/segment IDs to their actual zarr URLs.

```python
# vesuvius already installed in scroll conda env on Prajna
from vesuvius import Volume

# load a Scroll 1 segment by ID (segment in scrolls.yaml)
seg = Volume(type="segment", scroll_id=1, segment_id=20230827161847, anon=True)

# load directly by zarr URL (for segments not in scrolls.yaml)
import zarr
z = zarr.open("https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/")
volume = z['0'][:]   # level 0 → shape (65, H, W) for Scroll 1/2 segments
```

**Segment zarr URL pattern:**
- Scroll 1: `https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/{seg_id}.zarr/`
- Scroll 2: `https://dl.ash2txt.org/other/dev/scrolls/2/segments/54keV_7.91um/{seg_id}.zarr/`
- Level `'0'` = finest resolution, shape `(65, H, W)`, Z=65 layers

**Note:** Scroll 3 volume is at `https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/`

### Labeled segments available for training

`~/scroll_prize/villa/ink-detection/all_labels/` — **45** PNG ink label files (format: `{seg_id}_inklabels.png`). These are 2023 Kaggle competition Scroll 1/2 segments with confirmed ink annotations.

All segment IDs:
`20230520175435`, `20230522181603`, `20230522215721`, `20230530164535`, `20230530172803`,
`20230530212931`, `20230531121653`, `20230531193658`, `20230601193301`, `20230611014200`,
`20230620230617`, `20230620230619`, `20230701020044`, `20230702185753`, `20230813_real_1`,
`20230820203112`, `20230826170124`, `20230827161847`, `20230901184804`, `20230902141231`,
`20230903193206`, `20230904020426`, `20230904135535`, `20230905134255`, `20230909121925`,
`20230929220924`, `20230929220926`, `20231001164029`, `20231004222109`, `20231005123333`,
`20231005123336`, `20231007101615`, `20231012085431`, `20231012173610`, `20231012184420`,
`20231012184421`, `20231012184423`, `20231016151000`, `20231022170900`, `20231022170901`,
`20231031143850`, `20231106155350`, `20231106155351`, `20231210121321`, `recto`, `verso`

### villa training pipeline vs ours

| Aspect | Our `train_full.py` | Villa `train_resnet3d.py` |
|--------|--------------------|-----------------------------|
| Architecture | Segformer-B1 (2D) | ResNet3D + 3D decoder |
| Loss | BCE (fixed: ink channel only) | 0.5×Dice + 0.5×BCE (SoftBCE, smooth_factor=0.25) |
| Training strategy | Basic ERM | ERM or GroupDRO, per-sample loss |
| Framework | Manual loop (no Lightning) | PyTorch Lightning |
| Config | Python file | JSON metadata + YAML sweeps |
| Data loading | Custom ESRF zarr | ZarrSegmentVolume + MONAI sliding window |
| Pre-trained ckpt | No | Yes — `wild14_deduped_64_pretrained2_...ckpt` |

**Important:** villa uses PyTorch Lightning → use `l40` partition, NOT `a40` (Lightning hangs on A40 + CUDA 12.8).

### villa inference (use this for Scroll 3)

```bash
# Run villa's own pre-trained checkpoint on any segment:
cd ~/scroll_prize/villa/ink-detection/
python infer_resnet3d_vesuvius.py \
    --metadata_json metadata.json \
    --ckpt_path wild14_deduped_64_pretrained2_20231210121321_0_fr_i3depoch=3-v2_256.ckpt \
    --segment_id 20230827161847 \
    --zarr_path "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/" \
    --output_path ~/scroll_prize/results/villa_pred.png \
    --output_npy ~/scroll_prize/results/villa_pred.npy
```

### Key villa paths on Prajna
```
~/scroll_prize/villa/
├── vesuvius/                    ← Python package (already installed in scroll env)
├── ink-detection/
│   ├── train_resnet3d.py        ← main training script (uses Lightning → l40 only)
│   ├── train_resnet3d_lib/      ← config, data ops, model, orchestration
│   ├── infer_resnet3d_vesuvius.py ← inference on any zarr
│   ├── all_labels/              ← 45 Scroll 1/2 ink label PNGs
│   ├── metadata.json            ← training config (segments, hyperparams)
│   └── wild14_deduped_64_pretrained2_...ckpt  ← pre-trained checkpoint
├── thaumato-anakalyptor/        ← auto-segmentation (for future work)
└── scrollprize.org/docs/        ← competition documentation
```

---

## 4. Infrastructure (Prajna HPC)

| Item | Value |
|------|-------|
| User | `shiwani.mishra` |
| Group | `medal` (GrpSubmit=20 shared across group) |
| Home | `/home/medal/shiwani.mishra/` |
| Project root | `~/scroll_prize/` |
| Conda env | `scroll` |
| Good partitions | `a40` (training), `l40` (analysis/inference) |
| Avoid | `debug` — rejects `medal` account |

SSH: requires TOTP + password (2FA). ControlMaster: `~/.ssh/ctl/shiwani.mishra@prajna.iitb.ac.in:22` — re-auth after laptop sleep. VPN required off-campus.

**Key data paths on Prajna:**
```
~/scroll_prize/data/scroll3_ink_pred/level2/   # PHerc.332 m7 zarr, level-2, all z-slabs (71MB)
~/scroll_prize/data/esrf/                      # ESRF training fragments (500P2 + 343P)
~/scroll_prize/data/m7_triage/                 # level-3 of all 35 m7 zarrs (triage data)
```

**PyTorch Lightning hangs on A40 + CUDA 12.8** — use manual training loop (`train_full.py`), not Lightning.

---

## 5. Our Model — BCE Fix and Results

### The bug (fixed 2026-05-31)
BCE loss computed against both label channels on ESRF `(B, 2, H, W)` labels. Channel 1 is all-ones — training against it produces contradictory gradients, predictions saturate at 0.5.

```python
# wrong
loss = BCE(logits, y.float())
# correct
loss = BCE(logits, y[:, 0, :, :].float(), pos_weight=tensor([10.0]).to(device))
```

### Best model
**Segformer-B1, pos_weight=10, 50 epochs** (SLURM job 127266)

| Metric | Value |
|--------|-------|
| Val loss | 1.631 |
| >0.9 confidence on Scroll 3 segment 20240618142020 | **5.93%** |
| vs. pre-fix (B3, no pos_weight) | 0.00% |

Checkpoint: `~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt`  
Config: `vesuvius_first_title_prize/configs/ft_esrf_b1.py`  
Local prediction: `results/scroll3_prediction_B1_T0.3_BEST.npy`

**B1 domain gap:** The B1 model produces a uniform 32px-period dot grid on segment 20240618142020 — these are papyrus fibers, not ink. The model has not been validated as a true ink detector on Scroll 3.

---

## 6. What We Tried and What Happened

### PHerc.332 — cylindrical unrolling of m7 zarr

**Method:** sample the m7 zarr in circles at different radii (cylindrical polar unrolling), apply CLAHE, look for letter-shaped structures.

**What we found:** an isolated ~1 mm structure at r=310 px (z=7.51–8.40 mm) that looked visually like a letter.

**Why it was retracted:**
1. Automated gradient filter (`r310_fullsearch.py`) validated 0 of 10 candidates including this one
2. Positive control: a real ink layer spikes sharply at one radius (inner=0.002, peak=0.218, outer=0.000); this candidate peaks at r=298 and decays outward (0.164 → 0.098 → 0.000) — the papyrus fiber signature
3. Fundamental error: m7 is a surface predictor, not ink. We were never looking at ink data

**Posted to Discord as "confirmed letter" on 2026-06-04. Corrected publicly same day.**

### PHerc0009B — CLAHE on flat segment

**Method:** download ThaumatoAnakalyptor flat segment, apply CLAHE, identify letter-shaped marks.

**Why it was retracted:**
1. Pareidolia — the same CLAHE pipeline produces identical "letter-like" marks from empty regions
2. Data type error: segment was likely the ink-detection render, not raw CT as stated
3. PHerc0009B is not prize-eligible (not Scroll 2 or 3)

**Also posted to Discord 2026-06-04. Corrected publicly same day.**

### 35-scroll m7 triage (SLURM job 127851)

Scanned all 35 scrolls with m7 surface-prediction zarrs at level-3 (~9.6 µm/px) using the cylindrical unrolling + gradient gate. 8 gated candidates, 3 survived shuffle filter, 0 matched the calibrated thin-shell profile. Since the data is surface predictions (not ink), these results describe scroll sheet geometry, not ink distribution.

---

## 7. Active Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `train_full.py` | Manual training loop (no Lightning) | Active |
| `ft_esrf_b1.py` | B1 fine-tuning on ESRF | Active |
| `infer_s3_esrf.py` | B1 inference on scroll segment | Active |
| `prepare_esrf.py` | ESRF data prep | Active |
| `prajna_lib.py` | SSH helper library | Active |
| `positive_control.py` | Validates 3D readout chain on any zarr | Active |
| `salvage_test.py` | Pareidolia controls for any candidate | Active |
| `train_full.sh` | SLURM job script for training | Active |
| `full_scroll_scan.py` | z-profile sweep of PHerc.332 m7 zarr (surface data, not ink) | Retracted-analysis reference |
| `inspect_zones.py` | Per-zone gradient on PHerc.332 m7 zarr | Retracted-analysis reference |
| `r310_fullsearch.py` | Full gradient-gate scan — produced the "0 validated" result | Retracted-analysis reference |
| `level0_letter_zoom.py` | 1.2 µm/px zoom of the retracted candidate | Retracted-analysis reference |

---

## 8. Progress Prize Submission

**Target:** Sestertius ($2,500) – Denarius ($10k)  
**Primary contribution:** BCE loss fix (0% → 5.93% ink confidence)  
**Secondary:** positive control + pareidolia control methodology  
**File:** `PROGRESS_PRIZE_SUBMISSION.md`  
**Submit by:** June 30, 2026 at https://forms.gle/LrpQmSAqdwGpTczLA

---

## 9. Open Questions

### Priority 1 — Before next experiment

1. **Does the B1 model actually detect ink or just papyrus fibers?**
   The 32px dot grid on segment 20240618142020 is suspicious. Need to validate on a villa labeled segment (Scroll 1/2) where ground truth is known. If it fails there, the model is wrong; if it passes, Scroll 3 is just a harder case.

2. **Can we retrain using villa's Scroll 1/2 labels + ESRF fragments combined?**
   Villa has 15 labeled Scroll 1/2 segments in `all_labels/`. Combined with our ESRF fragments, this is 3× more diverse training data. This is the clearest path to a better model and a stronger Progress Prize submission.
   - Use villa's `train_resnet3d.py` with ResNet3D-3D-decoder (better than our B1)
   - Apply our BCE channel fix (`y[:,0,:,:]`) if the label format is 2-channel
   - Compare val loss and >0.9 confidence against our B1 baseline

3. **What is the path to getting Scroll 3 ink labels?**
   No public ink labels exist for Scroll 3 (PHerc.332). Options:
   - Community annotation of our segment 20240618142020 (we'd need to post predictions and ask for feedback)
   - Use ThaumatoAnakalyptor to generate a flat segment, then apply our B1 model, then submit for annotation

### Priority 2 — Architecture

4. **villa's ResNet3D vs our Segformer-B1 — which is better for this data?**
   villa's ResNet3D takes 3D surface volumes as input (20 layers × H × W). Our B1 takes the same format. On Scroll 1/2 segments (where ground truth exists) we can do a direct comparison.

5. **Can we use villa's pre-trained checkpoints as a starting point?**
   The community has released checkpoints — check the villa repo and Kaggle for Scroll 1/2 trained models. Fine-tuning from a good starting point is faster than training from scratch.

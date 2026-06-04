# Knowledge Base — Scroll Prize

Clean reference as of 2026-06-04. Only verified facts. §5–6 (environment, errors) restored from the original operational log — they are the anti-repeat reference; keep appending to the §6 error table.

---

## 1. The Competition

**Vesuvius Challenge** — read carbonized 2,000-year-old Herculaneum papyrus scrolls using X-ray CT + ML.

| Prize | Amount | Requirement | Status |
|-------|--------|-------------|--------|
| First Letters | $60,000 | 10 legible letters in a single 4 cm² area of **any of Scrolls 2-3** | Open, no winner |
| First Title | $60,000 | Produce a readable image of the title in **any of Scrolls 2-3** | Open, no winner |
| Progress — Gold Aureus | $20,000 | Major contribution (tool/method adopted by community) | Monthly, rolling |
| Progress — Denarius | $10,000 | Significant contribution | Monthly |
| Progress — Sestertius | $2,500 | Notable contribution | Monthly |
| Progress — Papyrus | $1,000 | Useful contribution | Monthly |

Submission forms: First Letters/Title → Google Form on the prizes page; Progress → https://forms.gle/Sy6mW5cfJS2U7E9F7  
**Correction (verified from official docs):** First Letters/Title are eligible on **Scrolls 2 AND 3** (PHerc.332 is Scroll 3; Scroll 2 = PHercParis3). Earlier notes saying "only Scroll 3" were wrong. Progress Prizes are evaluated **monthly** (rolling), not a single June 30 deadline — total awarded to date: **$1,781,500**.

**First Letters/Title submission rules that constrain us (from `34_prizes.md`):**
- Image must be a **programmatic output of CT data** — no manual character annotation, and the ink-model output region must NOT overlap any training data (prevents memorization).
- **Hallucination mitigation is required** — they explicitly discourage window sizes > 0.5×0.5 mm (= 64×64 px at 8 µm). Larger windows may be rejected. Our positive-control + pareidolia methodology directly addresses this.
- Include a 1 cm scale bar + the 3D position (segmentation ID). Do **not** go public before winning.

---

## 1b. Research Landscape — what's been done, what people use, what's wanted

> **Full reference already in-repo: [`docs/vesuvius_challenge_reference.md`](docs/vesuvius_challenge_reference.md)** — a 1088-line contributor compendium (history, milestones, every open problem, the complete prize structure + awarded-winner tables, all community tools, key people, master plan, formats, citations). This §1b is the *distilled, project-relevant* takeaway; go there for the full landscape.
>
> Both are cross-checked against villa's bundled official docs (`~/scroll_prize/villa/scrollprize.org/docs/`: `34_prizes.md`, `27_master_plan.md`, `15_winners.md`, `22_firstletters.md`). Authoritative outside context — verify against these, don't trust memory.

### The pipeline everyone works within
CT scan → **segmentation/unwrapping** (trace the papyrus sheet in 3D, flatten to 2D) → **ink detection** (ML on the flattened surface volume) → papyrologist reads. The two unsolved problems are *unwrapping at scale* and *ink identification that generalizes*.

### What's been done (milestones)
- **2023 Grand Prize ($850k):** first text from an unopened scroll (Scroll 1), by Youssef Nader, Luke Farritor, Julian Schilliger.
- **The ink lineage:** Casey Handmer found the **"crackle pattern"** (ink looks like crackle in the CT) → Luke Farritor trained a model on crackle and read **ΠΟΡΦΥΡΑϹ** ("porphyras"/purple), winning First Letters → Youssef Nader added **domain transfer** (unsupervised pretraining on scroll data + fine-tune on Kaggle fragment labels, then iterative pseudo-labeling).
- **2024:** words extracted from a *second* scroll; First Automated Segmentation ($60k).
- **2025:** First Title ($60k) — author identified as **Philodemus of Gadara**. Focus shifted to scanning (ESRF BM18, 9.2 µm in <2 hr; 30+ scrolls scanned).
- **March 2026:** Kaggle Surface Detection ($200k).

### What people use (methods / tools)
- **Ink detection (2023 GP winning solution, = our `villa/ink-detection/`):** ensemble of **TimeSformer-small** (divided space-time attention, the canonical model) + **ResNet3D-101** (pretrained) + **I3D** (non-local block + maxpooling); ~15 rounds of label cleaning. Stack: torch + lightning + timesformer-pytorch + 3D-ResNets-PyTorch.
- **Current villa ink model:** ResNet3D + 3D decoder with **GroupDRO** (`train_resnet3d.py`) — the newer pipeline we should build on.
- **Segmentation/unwrapping:** **VC3D** (volume-cartographer fork, Schilling & Johnson) — the team's approach as of Sept 2025; **spiral-fitting** (Henderson, fully automatic); **ThaumatoAnakalyptor** (Schilliger).

> **Latest news / "what's happening now":** [`docs/news_and_status.md`](docs/news_and_status.md) — a dated snapshot from scrollprize.org + Substack. Headlines as of mid-2026: ink is now appearing in multiple new scrolls via a **generalist model + ~2 µm scans + curriculum learning**; **Scroll 139** is the key training set (an autoresearch agent ~2×'d Scroll 4 perf training only on 139); **3D ink detection** on unflattened volumes now matches 2.5D; ~70% of Scroll 5 is auto-unwrapped. This directly reshapes our next move (see that doc's "what this means for our project").

### What's wanted (the open frontier — and why our results look the way they do)
From the Stage Two master plan: the 2023/24 ink models amplify signals that turn out to be **morphological cracks** or **metal-rich bright spots**. **These do not generalize** — "ink remains elusive in all our new data." The team is now hunting for *different* ink characteristics, and suspects that when ink is neither metal-rich nor cracked, **higher-resolution scans** are needed.

**This is the exact wall we hit:** our B1 (trained on ESRF crack/metal signal) produces only fiber texture on Scroll 3 (§7). It's not a bug in our model alone — it's the field-wide generalization gap. Implications for us:
- Don't expect a crack/metal-trained model to "just work" on Scroll 3. Reproducing it on Scroll 1/2 (where that signal exists) is the honest validation (§11 Q1).
- The genuinely valued contributions right now are **hallucination-safe methodology** and **tools that get used** (Progress Prize criteria) — which is where our positive-control/pareidolia work fits — not another retracted "letter" claim.

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
Ground-truth ink-labeled fragments from ESRF synchrotron scans. Used as training data. Only **500P2 + 343P** are usable public fragments (PHerc.9B has no surface/labels; PHerc1667Cr1Fr3 is only a JPEG photo, no scan).

- Surface layers are **PNG not TIFF** (`tifffile` raises `TiffFileError: not a TIFF file b'\x89PNG'`). Use `PIL.Image.open`.
- Ink labels are large (500P2 label = 27160×14990 ≈ 407M px). Set `Image.MAX_IMAGE_PIXELS = None` before opening or PIL raises `DecompressionBombError`.
- **Raw label format on disk:** RGBA. Alpha is always 255 (useless). The **R channel (=G=B)** holds the signal: 0 = no ink, >0 = ink. Convert with `(R > 0)`. (An old DATA_EXPLORATION.md "28% ink" figure was wrong — real fractions: 500P2 = 4.1%, 343P = 3.4%.)
- **As loaded by our DataModule:** labels become `(H, W, 2)` tensors — channel 0 = ink mask, channel 1 = all-ones validity mask. **Only use channel 0 for training** (see BCE fix in §7).
- Dataset built: 13,873 patches (500P2) + 3,849 (343P) ≈ 3,276 training patches after sampling.

### Scroll volume shapes (confirmed, level 0 / level 3)

| Scroll | Level 0 (z×y×x) | Level 3 | Notes |
|--------|------------------|---------|-------|
| Scroll 1 | 14376×7888×8096 | 1797×986×1012 | Grand Prize scroll |
| Scroll 2 | 14428×10112×11984 | 1804×1264×1498 | Scan artifact in centre |
| Scroll 3 (PHerc.332) | 9778×3550×3400 | 1223×444×425 | First Letters target |

Scroll 3 is low-contrast (mean intensity 44.9/255, peaks ≤159) — **CLAHE is essential** before any inference (clipLimit ≥ 2.0, tileGridSize 8×8). Scroll 1/2 **segments** are `(65, H, W)` at level 0 (see §3 for zarr URLs).

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

**PyTorch Lightning hangs on A40 + CUDA 12.8** — use manual training loop (`train_full.py`), not Lightning. (Villa's pipeline uses Lightning and runs fine on `l40` — the hang is A40-specific.)

---

## 5. Environment & Dependencies

### Conda env `scroll` — verified working stack
| Package | Version | Note |
|---------|---------|------|
| torch | **2.5.1+cu121** | Works on l40 (L40S, CUDA 12.8 driver 570.x) AND a40. Do NOT use cu130/cu124 builds (see below). |
| transformers | 5.9.0 | Needs HF model cache pre-downloaded (no internet on compute nodes). |
| vesuvius | 0.2.4 | Installed editable from `~/scroll_prize/villa/vesuvius` (NOT on PyPI). |
| phoenix | 1.0 | Our training package, editable from `vesuvius_first_title_prize/`. |
| zarr | 2.18.3 | Use `zarr.open_group(fsspec.get_mapper(url))` — stable across 2.x. |
| segmentation_models_pytorch | 0.5.0 | villa loss functions (Dice, SoftBCE). |
| s3fs | 2026.4.0 | For S3 — but most data is on `dl.ash2txt.org` (HTTP), not S3. |

### CUDA / torch version trap (cost us a full day in May)
- `torch 2.12.0+cu130` → `RuntimeError: NVIDIA driver too old (found 12080)`. Cluster driver is CUDA 12.8, not 13.0.
- `pip install torch --index-url .../cu128` silently installs cu130 anyway → must pin: `pip install 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121`.
- cu124 builds have broken cuDNN for `Conv3d` on l40 → do not use. **Stick to 2.5.1+cu121.**
- `torchaudio` with mismatched torch → import error. `pip uninstall torchaudio`.

### Compute nodes have NO internet — pre-download everything on the login node
- HuggingFace `from_pretrained('nvidia/mit-b1')` fails on compute nodes: `OSError: Can't load the model... [Errno -2] Name or service not known`.
- **Fix:** pre-download the model on the login node once, then export in every job script:
  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export WANDB_MODE=disabled
  ```
- Same applies to any `requests`/`gdown`/`pip` call inside a SLURM job — they will all DNS-fail.

### SLURM group quota
- Group `medal` has `GrpSubmit=20` shared across **all** group users → `AssocGrpSubmitJobsLimit` even when your own queue is empty. Wait ~10 min and retry, or do lightweight work (pip, wget) on the login node.

### SLURM job-script must-haves (learned the hard way)
- `source ~/miniconda3/etc/profile.d/conda.sh` before `conda activate` (`.bashrc` is NOT sourced in jobs).
- `set -eo pipefail` (NOT `-u` — it breaks conda activate).
- `mkdir -p ~/logs` before sbatch (missing log dir = silent immediate failure, empty log).
- `python -u` (unbuffered) or logs never flush before timeout.
- `--qos` must exactly equal `--partition` or the job is rejected.

---

## 6. Errors Encountered & Fixes

The clean-rewrite dropped this table; it is the single most useful anti-repeat reference. Keep appending.

| Error | Cause | Fix |
|-------|-------|-----|
| `Name or service not known` (prajna.iitb.ac.in) | VPN not active | Connect IITB VPN |
| `OSError: Can't load nvidia/mit-bN` in a job | Compute node has no internet; HF download fails | Pre-download on login node + `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` |
| `Local config not found!` / model init aborts in inference | Same HF-offline issue — `from_pretrained` of the Segformer backbone hits the network | Set offline env vars; verify backbone is in `~/.cache/huggingface` |
| `RuntimeError: NVIDIA driver too old (12080)` | torch cu130 needs CUDA 13 driver | `pip install 'torch==2.5.1+cu121'` |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` (Conv3d) | cu124 cuDNN broken on l40 | Use 2.5.1+cu121 |
| Lightning `trainer.fit()` hangs after preload | Lightning bug on a40+CUDA12.8; also `num_sanity_val_steps` deadlock on tiny val sets | Manual training loop (`train_full.py`); or `num_sanity_val_steps=0` |
| Training "stuck" — no batch logs | stdout buffering | `python -u` |
| `pip install vesuvius` → no distribution | Not on PyPI | `pip install -e ~/scroll_prize/villa/vesuvius` |
| `TypeError: Volume.__init__() ... 'normalize'` | API has no `normalize` kwarg | Remove it |
| `ValueError: URL not found in config for scroll=3` | Scroll 3 not in old scrolls.yaml | Use direct zarr URL (see §3) |
| `TiffFileError: not a TIFF file b'\x89PNG'` | ESRF layers are PNG | `PIL.Image.open` not `tifffile` |
| `PIL DecompressionBombError` | 407M-px ESRF label | `Image.MAX_IMAGE_PIXELS = None` |
| `AssocGrpSubmitJobsLimit` | Group quota (20) shared | Wait + retry, or work on login node |
| `Permission denied` on `/lustre-scratch/` | Not provisioned | Use `~/scroll_prize/results/` |
| villa infer `KeyError: segment_id ... not found in metadata_json.segments` | Segment not in `metadata.json` | Pass `--layer_range start:end` (e.g. `1:63`) for external zarrs |

---

## 7. Our Model — BCE Fix and Results

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

### Full experiment trajectory (how we got to B1)

| Model | Val loss | Scroll 3 >0.9 conf | Verdict |
|-------|----------|--------------------|---------|
| Baseline (pre-fix, broken 2-ch BCE) | 0.604* | ~0% | Predictions saturate at 0.5 |
| TimeSformer zero-shot (Scroll 1 model) | — | noise | Domain shift — detects S3 fiber everywhere |
| Transfer from TimeSformer weights | 0.612 | — | **Worse** — cross-domain transfer fails |
| Offline augmentation (3×) | 0.613 | — | **Worse** — adds no new info (online aug already covers it) |
| Fixed BCE, no pos_weight, 30ep | 0.634 | 0.00% | Class imbalance (91% no-ink) → defaults to "no ink" |
| Fixed BCE + pos_weight=10, B3, 50ep | 1.635 | 5.86% | pos_weight makes it actually learn ink |
| **Fixed BCE + pos_weight=10, B1, 50ep** | **1.631** | **5.93%** | **Best — smaller model overfits less** |

*measured under the broken loss, not comparable to post-fix numbers (the fix changes the loss scale).

**What did NOT work (don't retry these):**
- **Transfer learning from TimeSformer** — it learned Scroll 1 ink (high-contrast crackle); ESRF is low-contrast surface topography. Different domains → degrades.
- **Offline pre-augmentation** — `train_aug` already applies rotations/flips/noise online. Offline files just duplicate patches. The bottleneck is *new fragment data*, not more aug passes.
- **More epochs (70) / bigger backbone (B3→79M)** — overfits 3,276 patches after ~50 epochs. B1 (45.6M) is near-optimal for this dataset size.
- **Temperature scaling (T=0.2–0.3)** — sharpens the histogram for visualization but does NOT change which pixels are classified; not a real improvement.

**The real bottleneck is data, not architecture.** 3,276 ESRF patches is too few. The clearest path forward is villa's 45 Scroll 1/2 labeled segments (see §3, §11).

**B1 domain gap:** The B1 model produces a uniform 32px-period dot grid on segment 20240618142020 — these are papyrus fibers (~128 µm weave period = 32 px × 4 µm/px), not ink. The `>0.9 = 5.93%` metric reflects uniform texture detection, not ink. The model has NOT been validated as a true ink detector — that's exactly what `validate_b1_villa.py` is for (§11 Q1).

---

## 8. What We Tried and What Happened

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

### What worked — validated tools & methods (keep using these)

These were proven out over the project and are the durable, reusable wins:

- **The BCE channel-0 fix + `pos_weight=10`** — the actual modeling win (§7). Slice labels to `y[:,0]`, weight the positive class for the 9% ink ratio.
- **`positive_control.py`** — painted Π/Ο/Β/Φ on a synthetic cylindrical shell and recovered them through the full unroll→CLAHE pipeline. Proves a readout chain before trusting it on real data. Reusable on any zarr.
- **Manual PyTorch training loop** (no Lightning) — sidesteps the a40 Lightning hang; full control over loss/checkpointing.
- **CLAHE preprocessing** (clip 2.0, tile 8×8) — essential for low-contrast Scroll 3; the First Title winner's `contrasted=True` does the same.
- **Patch-based inference with overlapping averaging** (patch 128, stride 128) — robust full-volume predictions, no OOM on A40.
- **SLURM dependency chaining** (`--dependency=afterok:JOBID`) — ran a 3-phase pipeline overnight with zero intervention.
- **ControlMaster + pyotp TOTP automation** — auth once, reuse the socket (now `prajna_lib.py`).
- **Direct `zarr.open_group(fsspec.get_mapper(url))`** — more reliable than `vesuvius.Volume()` for anything not in `scrolls.yaml`; needs no auth.
- **`gdown --folder`** for Google-Drive model weights; **nohup wget loop** on the login node for large multi-file segment downloads (65 TIFs, 7.7 GB); **BeautifulSoup** for `dl.ash2txt.org` directory listings.
- **`PIL.Image.open` with `MAX_IMAGE_PIXELS = None`** for the large ESRF labels.

### Timeline — what we did, dated

- **2026-05-29 (Setup):** Prajna env + villa/First-Title repos cloned. Downloaded Scroll 3 segment 20240618142020 (65 TIFs, 7.7 GB). TimeSformer **zero-shot on Scroll 3 → uniform noise** (Scroll-1 model, domain shift). Decided to train MiniUNETR on ESRF 500P2+343P. Survived the torch-CUDA version trap (→ 2.5.1+cu121).
- **2026-05-30 (Lightning hang):** Diagnosed `trainer.fit()` deadlock on a40 (not cuDNN/data/precision). Confirmed a manual loop runs cleanly.
- **2026-05-31 (Baseline + the bug):** MiniUNETR baseline (20ep, val 0.604). **Found & fixed the 2-channel BCE bug.** Overnight SLURM chain ran Phase 1 ensemble, Phase 2 transfer (**failed**, 0.612), Phase 3 augmentation (**failed**, 0.613).
- **2026-06-02:** Confirmed `pos_weight=10` fixes class imbalance. 50-ep B3 → val 1.635, Scroll 3 >0.9 = 5.86%.
- **2026-06-03:** B1 backbone → val 1.631, >0.9 = **5.93% (best)**. Found the **B1 domain gap** (32px dot grid on Scroll 3 = fibers). Ran level-2/3 m7 zarr CLAHE "candidate" hunting (later retracted).
- **2026-06-04 (Self-audit):** PHerc.332 candidate + PHerc0009B **both RETRACTED** (positive control + gradient gate; pareidolia). 35-scroll m7 triage → 0 calibrated hits. Discord posts corrected. **Discovered m7 = surface predictor, not ink.** Adopted villa as the research foundation.

---

## 8b. Lessons & Failure Modes — claims, communication, PRs (don't repeat)

The retractions (§8) were **not** a data hoax — the data and code were always real and public. The failure was **wording that outran the evidence**, and it cost credibility. These are the durable rules so it doesn't recur. This is the behavioral counterpart to the finding-evaluation toolkit in `GUIDE.md` Part 8.

**Rule 1 — A public claim may never exceed what your own code outputs.** `r310_fullsearch.py` printed `Total raw: 10, deduped: 10, validated: 0` — the automated gradient filter rejected the very candidate we called "confirmed." We had leaned on hand-picked CLAHE images while our own quantitative test said no. Read your own script's output before posting; the claim is capped at what it reports.

**Rule 2 — No "renders ink as letters" claim without a positive control.** Build `positive_control.py`-style proof that the method renders *known* ink as legible letters first. (Ours validated the readout chain — and showed real ink spikes at r=310 while our candidate peaked at r=298 and decayed outward = fiber signature.) The positive control is the line between "candidate" and "discovery."

**Rule 3 — Always run the empty/shuffle control before claiming letters.** CLAHE manufactures letter-like marks from noise. PHerc0009B's Π/Ο were pareidolia (same pipeline → equally letter-like marks on empty regions); angle-shuffling the PHerc.332 window kept ~half the "blobs." If empty/shuffled data produces the same thing, you have nothing.

**Rule 4 — Scope claims to what you actually searched.** "the only letter in the entire scroll" was really "the only one in the thin accessible shell at r=310" — saturated outer layers (r≥340) were never searchable. Never say "entire scroll / whole dataset" when you searched a slice.

**Rule 5 — Use "candidate," not "confirmed." Letter IDs need a papyrologist.** β/φ and "carbonized ink" were interpretations. Default wording: "a candidate structure I find visually suggestive; could be a crack, fold, or fiber."

### The overclaim → honest-version table (reference)
| What we posted | What the evidence supported |
|----------------|------------------------------|
| "diagnostic signature of real carbonized ink" | "a structure with an ink-like radial profile I find visually suggestive" |
| "the only letter-form structure across the entire scroll" | "the only one in the thin accessible shell at r=310 (outer layers saturated/unreadable)" |
| "confirmed letter — β or φ" | "a candidate ~1 mm structure; reads as easily as a crack or fiber" |
| "Π and Ο in raw CT" (PHerc0009B) | retracted — pareidolia; also likely an ink-render not raw CT, and not a prize-eligible scroll |

### If you've already posted something wrong — the correction protocol
1. **Self-correct fast, in the same thread.** Reaching the `validated: 0` honesty *before* someone else reruns your script is a credibility **gain**, not a loss.
2. **Do NOT delete the original** — deletion looks worse than correcting.
3. **Do NOT double down** or add new confident claims.
4. **Don't post the next finding** until it has been through the same controls.
5. Lead with "candidate, not confirmation; data + code public; happy to share coords for independent verification."

**Rule 6 — Terminology must be verified against villa BEFORE any PR or public output.** We mislabeled the m7 zarrs "ink predictions" for weeks; they are **surface/sheet localization** predictions (§2). If a PR reaches `ScrollPrize/villa` (or a public post) with wrong terminology, the organizers flag it immediately and it looks worse than not posting. Before any PR or public claim, verify every technical term against villa's actual code/docs (`scrollprize.org/docs/`, the model READMEs). **Never open a PR without showing the full diff and getting explicit sign-off** (standing instruction).

---

## 9. Active Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `train_full.py` | Manual training loop (no Lightning) | Active |
| `ft_esrf_b1.py` | B1 fine-tuning on ESRF | Active |
| `infer_s3_esrf.py` | B1 inference on scroll segment | Active |
| `validate_b1_villa.py` | Run B1 on a villa Scroll 1/2 labeled segment, report P/R/F1 vs ground truth | Active |
| `validate_b1_villa.sh` | SLURM job (l40) for the above | Active |
| `infer_villa_pretrained.sh` | Run villa's own pre-trained checkpoint on same segment for comparison | Active |
| `prepare_esrf.py` | ESRF data prep | Active |
| `prajna_lib.py` | SSH helper library (copy of the prajna-hpc skill lib) | Active |
| `positive_control.py` | Validates 3D readout chain on any zarr | Active |
| `salvage_test.py` | Pareidolia controls for any candidate | Active |
| `train_full.sh` | SLURM job script for training | Active |
| `full_scroll_scan.py` | z-profile sweep of PHerc.332 m7 zarr (surface data, not ink) | Retracted-analysis reference |
| `inspect_zones.py` | Per-zone gradient on PHerc.332 m7 zarr | Retracted-analysis reference |
| `r310_fullsearch.py` | Full gradient-gate scan — produced the "0 validated" result | Retracted-analysis reference |
| `level0_letter_zoom.py` | 1.2 µm/px zoom of the retracted candidate | Retracted-analysis reference |

---

## 10. Progress Prize Submission

**Realistic tier:** Papyrus ($1,000) – Sestertius ($2,500). *Honest read: not higher — the prize weights community adoption and the repo has no users yet. (Earlier "Sestertius–Denarius" was optimistic.)*  
**Primary contribution:** BCE loss fix (0% → 5.93% ink confidence)  
**Secondary:** positive control + pareidolia control methodology  
**File:** `PROGRESS_PRIZE_SUBMISSION.md` — full field-by-field draft answers + the required `awesome-scroll-tools` PR entry + tier assessment are there.  
**Submitting account:** saurabhgupta0342@gmail.com  
**Submit by:** June 30, 2026, 11:59pm Pacific at https://forms.gle/Sy6mW5cfJS2U7E9F7  
**Status:** form answers drafted, **not submitted**; `awesome-scroll-tools` PR drafted, **not opened** (awaiting sign-off per the no-PR-without-approval rule, §8b Rule 5).

---

## 11. Open Questions

### Priority 1 — Before next experiment

1. **Does the B1 model actually detect ink or just papyrus fibers?**
   The 32px dot grid on segment 20240618142020 is suspicious. Validate on a villa labeled segment (Scroll 1/2) where ground truth is known via `validate_b1_villa.py`. If F1 < 0.05 there, the model is wrong; if F1 > 0.2, Scroll 3 is just a harder domain-gap case. **In progress** — first job failed on the HF-offline error (now documented §6); needs `TRANSFORMERS_OFFLINE=1` and resubmit.

2. **Can we retrain using villa's Scroll 1/2 labels + ESRF fragments combined?**
   Villa has **45** labeled Scroll 1/2 segments in `all_labels/`. Combined with our ESRF fragments, this is far more diverse training data — the clearest path to a better model and a stronger Progress Prize submission.
   - Use villa's `train_resnet3d.py` (ResNet3D + 3D decoder, Lightning → l40 partition)
   - Note villa's loss is `0.5·Dice + 0.5·SoftBCE` (different from our BCE-only fix) — already handles class balance via Dice
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

### Priority 3 — Contribution options (preserved from the original strategy doc `docs/PLAN.md`)

These are forward-looking ideas from the first strategy pass; the full prize wishlist is in `docs/vesuvius_challenge_reference.md` §18 (and the live list: villa GitHub issues labeled `help wanted` / `good first issue`).

6. **Knowledge distillation 2 µm → 9 µm** (potential Gold Aureus). Train a teacher on ESRF 2.2 µm fragments, distill into a student that runs on 9 µm scroll data (soft targets + feature distillation). Directly targets the documented generalization gap — the reason ink doesn't transfer to the newer lower-res scans. Compute: dgx (A100).
7. **3D scroll-aware augmentations** (villa wishlist) — curvature/fiber/layer-topology-aware transforms + ablation. Sestertius–Denarius.
8. **3D ink-label methodology** (villa wishlist) — volumetric labels vs today's 2D projections, using ESRF 2.2 µm. Could combine with distillation.
9. **VC3D sheet-switch detection** — auto-detect/correct mesh jumping between adjacent layers; targets the unwrapping-at-scale bottleneck.

**Data sources not yet used:** DLS fragments (Frag1–6, 3.24 µm, hand-labeled) and the EduceLab legacy fragments — additional ink ground truth beyond ESRF 500P2+343P. **Useful segment sizes:** Scroll 3 `20240618142020` = 33.5 cm² (6× the 4 cm² prize area); Scroll 2 `20240516205750` = 26 cm².

**What wins a Progress Prize** (from the official criteria): released early, *actually gets used* by the community, well documented, modular (standard formats: OME-Zarr, meshes). Infrastructure/tools tend to win over one-off results.

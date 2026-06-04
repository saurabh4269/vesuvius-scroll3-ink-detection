# Vesuvius Challenge (Scroll Prize) — Complete Contributor Reference

> Last synced: May 2026  
> Audience: coding agent / contributor onboarding  
> Primary repo: https://github.com/ScrollPrize/villa  
> Website: https://scrollprize.org  
> Mailing list: https://scrollprize.substack.com  
> Discord: https://discord.com/invite/uTfNwwecCQ  
>
> **This is the full research landscape — what's been done, what people use, what's wanted.** For the latest *verified* prize scope and our project-specific takeaways, see `knowledge_base.md` §1, §1b (re-checked June 2026 against villa's bundled official docs). Where this doc and `knowledge_base.md` differ on a fact, the knowledge base is the corrected one — e.g. First Letters **and** First Title are currently scoped to **Scrolls 2-3** per `34_prizes.md`.

---

## Table of Contents

1. [What Is the Vesuvius Challenge?](#1-what-is-the-vesuvius-challenge)
2. [History and Milestones](#2-history-and-milestones)
3. [The Science: How Scrolls Are Read](#3-the-science-how-scrolls-are-read)
4. [Open Problems (Current Focus Areas)](#4-open-problems-current-focus-areas)
5. [Prize Structure](#5-prize-structure)
6. [The Data](#6-the-data)
7. [Monorepo Structure (`villa`)](#7-monorepo-structure-villa)
8. [Core Tools and Libraries](#8-core-tools-and-libraries)
9. [Segmentation Pipeline Deep Dive](#9-segmentation-pipeline-deep-dive)
10. [Ink Detection Deep Dive](#10-ink-detection-deep-dive)
11. [Virtual Unwrapping: State of the Art](#11-virtual-unwrapping-state-of-the-art)
12. [Community Projects and Ecosystem](#12-community-projects-and-ecosystem)
13. [Key People](#13-key-people)
14. [Sponsors and Funding](#14-sponsors-and-funding)
15. [Master Plan: Stages 1–4](#15-master-plan-stages-14)
16. [How to Contribute (Step by Step)](#16-how-to-contribute-step-by-step)
17. [AGENTS.md: Rules for Coding Agents in This Repo](#17-agentsmd-rules-for-coding-agents-in-this-repo)
18. [Wish List / Good First Issues](#18-wish-list--good-first-issues)
19. [Formats, Conventions, and File Specs](#19-formats-conventions-and-file-specs)
20. [Citations and Licensing](#20-citations-and-licensing)

---

## 1. What Is the Vesuvius Challenge?

The Vesuvius Challenge (also called "Scroll Prize") is a machine learning, computer vision, and geometry competition to read the **Herculaneum Papyri** — ancient scrolls buried when Mount Vesuvius erupted in 79 AD, carbonizing but preserving an entire library at the Villa dei Papiri near Herculaneum, Italy.

The scrolls cannot be physically unrolled without destroying them. The challenge: read them virtually using X-ray CT scans, machine learning, and geometry processing.

As of May 2026, over **$1,700,000 in prizes** have been awarded. The challenge is ongoing.

**Organizers:** Nat Friedman (former GitHub CEO), Daniel Gross (entrepreneur/investor), Dr. Brent Seales (Professor of Computer Science, University of Kentucky).

**Operating entity:** Curious Cases, Inc.

---

## 2. History and Milestones

| Year | Event |
|------|-------|
| 79 AD | Mount Vesuvius erupts; Herculaneum and its library are buried under pyroclastic flow |
| 1750 | Italian farmworker discovers the buried Villa dei Papiri while digging a well; excavations unearth hundreds of carbonized scrolls |
| 18th c. | Monk Padre Antonio Piaggio physically unrolls a few scrolls over decades; many are destroyed in the process |
| 2015 | Dr. Brent Seales virtually unwraps the **En-Gedi scroll** (Dead Sea region) using X-ray tomography and computer vision — reading Leviticus without opening it |
| 2019 | EduceLab-Scrolls dataset created; first scans of Herculaneum scrolls at Diamond Light Source |
| March 2023 | **Vesuvius Challenge launched** by Nat Friedman, Daniel Gross, and Brent Seales. Scrolls scanned at Diamond Light Source (Oxford). $1M+ in prizes announced |
| August 2023 | **Casey Handmer** discovers the "crackle pattern" — first directly visible evidence of ink inside a complete scroll |
| October 2023 | **Luke Farritor** wins First Letters Prize: first word read from inside a Herculaneum scroll ("ΠΟΡΦΥΡΑϹ" = "purple"). **Youssef Nader** wins second place |
| February 2024 | **2023 Grand Prize ($700,000)** awarded to Youssef Nader, Luke Farritor, and Julian Schilliger — 15 columns of never-before-seen text extracted from Scroll 1 (PHerc. Paris. 4) |
| 2024 | Community builds on Grand Prize success; autosegmentation pipelines refined; text found in **Scroll 5 (PHerc 172)** — a second scroll |
| 2024 | **First Automated Segmentation Prize ($60,000)** awarded to Sean Johnson et al. |
| May 2025 | **First Title Prize ($60,000)** awarded to Marcel Roth and Micha Nowak — title of PHerc. 172 read as "Philodemus, *On Vices*, Book 1(?)". First time a still-rolled scroll's title has been read noninvasively |
| July 2025 | Master Plan updated; ESRF BM18 beamline scanning begins; 30+ scrolls scanned; ink still elusive in new scans |
| 2025–2026 | Focus: unwrapping at scale + generalizable ink detection across full collection |

**What the scroll says (Grand Prize content):** Scroll 1 (PHerc. Paris. 4) contains an Epicurean philosophical text attributed to **Philodemus of Gadara**. The subject is pleasure as the highest good. The scroll discusses whether scarce goods provide more pleasure than abundant ones, and closes with a critique of adversaries who "have nothing to say about pleasure." The first word found was "ΠΟΡΦΥΡΑϹ" (purple).

---

## 3. The Science: How Scrolls Are Read

The virtual unwrapping pipeline has five steps:

### Step 1: Scanning

- Physical scroll is placed in a **synchrotron particle accelerator** (Diamond Light Source near Oxford; ESRF BM18 in Grenoble, France)
- Hundreds to thousands of X-ray photographs captured from different angles as scroll rotates 360°
- Images are reconstructed via **tomographic reconstruction** into a 3D volumetric image
- Output: a stack of `.tif` files where each file = one horizontal slice ("image stack")
- Each voxel (3D pixel) stores the **radiodensity** at that location
- Current best resolution: **7.91–9.2 µm voxel size** (scans at ESRF achievable in under 2 hours with helical acquisitions)
- Scanning protocols explored: tetra-, hexa-, octa-helical acquisitions on BM18

### Step 2: Representation

- Raw voxel data is mathematically transformed into more useful representations
- Examples: point clouds (.ply), surface normals, fiber direction fields
- Key challenge: the scroll is a mess — carbonized, crushed, wrapped tightly
- **nnUNet**-based semantic segmentation used to predict: surface sheets (S), vertical papyrus fibers (Fv), horizontal papyrus fibers (Fh)
- These intermediate representations are noisy but far more tractable than raw data

### Step 3: Segmentation

The goal is to isolate the 2-manifold (the rolled papyrus surface) and flatten it.

Sub-steps:
1. **Map:** trace the surface in 3D (using VC3D, Khartes, or automated pipelines)
2. **Mesh:** triangulate the surface into an `.obj` file
3. **Subvolume:** sample voxels around the mesh to create a surface volume
4. **Flatten:** apply isometric parametrization to unroll the surface into 2D — like making a map of the earth on flat paper

Output: a flat `.tif` image stack ("surface volume") where papyrus sits roughly in the middle

### Step 4: Ink Detection

- Input: surface volume (flat TIFF stack) + optional hand-labeled binary masks
- ML models detect ink by identifying characteristic patterns
- Key discovery: **crackle pattern** — ink appears as faint bright/crackled regions in the surface volume
- Current best model: **TimeSformer-based** (from Grand Prize winning submission)
- Models use **small input/output windows** (max 64×64 px at 8 µm = 0.5×0.5 mm) to avoid hallucination
- Two ink signals identified: (a) morphological cracks, (b) brighter X-ray spots from metal-rich ink
- These signals have NOT generalized well to most of the new (non-PHerc. Paris. 4/PHerc. 172) scrolls yet

### Step 5: Read

- Papyrologists decode the ink prediction images
- Current state: ~5% of Scroll 1 read; title found in Scroll 5
- Papyrology team led by Prof. Federica Nicolardi (Univ. Naples Federico II)

---

## 4. Open Problems (Current Focus Areas)

### 4.1 Representation (Semantic Segmentation)

The raw 3D scans are chaotic. Mapping the papyrus sheets in 3D requires better semantic segmentation of the volumes into structured representations.

- **Skills needed:** image annotation, computer vision, ML, medical imaging (3D segmentation)
- **Current approach:** nnUNet trained to predict surface sheets, vertical fibers, horizontal fibers
- **Problem:** predictions contain fake mergers, holes, false positives

### 4.2 Geometric Reconstruction (Unwrapping at Scale)

Even with good representations, constructing the actual mesh and flattening it is unsolved for large areas automatically.

- **Current cost:** $1–5M to manually unwrap a full scroll
- **Target cost:** $5,000 or below (100–1000× reduction needed)
- **Skills needed:** geometry processing, computer vision, ML, optimization
- **Three current state-of-the-art methods (none fully solved):**
  - Spiral Fitting (top-down, global)
  - Surface Tracer (bottom-up, local patch growth)
  - Thaumato Anakalyptor (point cloud + instance segmentation + graph stitching)

### 4.3 Ink Detection Generalization

- Ink has been found in only 2 of 5+ scrolls
- Models trained on Scroll 1 do not generalize to most other scrolls
- **Hypothesis 1:** ink in new scrolls is neither metal-rich nor cracked → higher-resolution scans may help
- **Hypothesis 2:** different papyrus surface properties
- **Active exploration:** analysis of fragment scans, varying scan protocols
- **Skills needed:** ML, pattern recognition, image annotation, domain adaptation

---

## 5. Prize Structure

### Currently Open Prizes

#### First Letters Prize — 7 × $60,000
- **Scrolls 2–3**: $60,000 to the first team uncovering 10 letters within a single 4 cm² area
- Submission requires: image, methodology, hallucination mitigation, 3D position of text
- Ink model output must NOT overlap training data
- Window size: max 0.5×0.5 mm (64×64 px for 8 µm scans)

#### First Title Prize — 7 × $60,000
- $60,000 to the first team discovering the title of any of Scrolls 1–3 (Scroll 5 title already won)
- Papyrologists must be able to read it; you don't need to read it yourself
- Submit image showing ink predictions in spatial context of title search

#### Monthly Progress Prizes — $350,000 pool
- Awarded monthly, open-ended, for open source contributions
- Tiers:
  - **Gold Aureus:** $20,000 (4–8/year) — major contributions
  - **Denarius:** $10,000 (10–15/year)
  - **Sestertius:** $2,500 (~25/year)
  - **Papyrus:** $1,000 (~50/year)
- Favored: released early, actually used by community, well documented
- Submission deadline: 11:59pm Pacific, last day of each month
- Submission form: https://forms.gle/Sy6mW5cfJS2U7E9F7

#### Unwrapping at Scale Prize — $200,000
- Automate virtual unwrapping of entire scrolls

### Already Awarded Prizes

| Prize | Amount | Winners |
|-------|--------|---------|
| 2023 Grand Prize | $850,000 | Youssef Nader, Luke Farritor, Julian Schilliger (+ 3 runner-up teams at $50k each) |
| First Automated Segmentation Prize | $60,000 | Sean Johnson et al. |
| First Title Prize | $60,000 | Marcel Roth, Micha Nowak |
| First Letters & First Ink | $60,000 | Luke Farritor, Youssef Nader, Casey Handmer |
| Open Source Prizes | $200,000+ | 50+ winners including Giorgio Angelotti, Yao Hsiao, Brett Olsen |
| Ink Detection Prizes | $112,000 | 16 winners including Yannick Kirchhoff, tattaka, Ryan Chesler, Felix Yu |

### Prize Terms
- Winners must open source their method
- Prizes awarded at sole discretion of Curious Cases, Inc.
- Technical + annotation + papyrological review required for milestone prizes
- Payment info must be provided within 30 days

---

## 6. The Data

### Data Access

- **S3 bucket (public, free):** `s3://vesuvius-challenge-open-data/`
  - Browse: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html
- **Web-browsable samples:** https://data.aws.ash2txt.org/samples/
- **Data Browser:** https://scrollprize.org/data_browser
- **Segments page:** https://scrollprize.org/data_segments
- **Curated Datasets:** https://scrollprize.org/data_datasets
- **Quick start notebook:** https://github.com/ScrollPrize/open-data/blob/main/examples/get-to-know-a-dataset.ipynb

### Datasets

**EduceLab-Scrolls (legacy)**
- Scrolls 1–4 and Fragments 1–6
- Citation: Parsons et al. 2023, arXiv:2304.02084
- Copyright: EduceLab / University of Kentucky
- License: CC-BY-NC 4.0

**Vesuvius Challenge - CT Scans of Herculaneum Papyri (current)**
- Newer scans released directly by Vesuvius Challenge
- Citation: Giorgio Angelotti, Stephen Parsons, Sean Johnson et al., *Vesuvius Challenge - CT Scans of Herculaneum Papyri*, Vesuvius Challenge

### Scroll Inventory

| ID | Name | Notes |
|----|------|-------|
| Scroll 1 | PHerc. Paris. 4 | Grand Prize scroll; 15 columns read; Philodemus text |
| Scroll 2 | PHerc. Paris. 3 | First Letters Prize target |
| Scroll 3 | — | First Letters Prize target |
| Scroll 4 | — | Scanned; no readable ink yet |
| Scroll 5 | PHerc. 172 | Second readable scroll; title found ("On Vices") |

Beyond these 5, 30+ additional scrolls have been scanned at ESRF (as of July 2025). Ink remains elusive in all new scans. Total extant collection: ~300 scrolls mostly in Naples (Biblioteca Nazionale di Napoli).

### Data Formats

| Type | Format |
|------|--------|
| Volumetric scans | OME-Zarr (primary, cloud-optimized, multi-resolution), sometimes TIFF stacks |
| Segment surface volumes | OME-Zarr and/or TIFF stacks (`00.tif`, `01.tif`, ...) |
| Surface geometry (meshes) | OBJ meshes; TIFXYZ (x/y/z TIFF triplet + metadata) |
| Model outputs | OME-Zarr (volumetric), TIFF (image) |
| Metadata | JSON |

**Why OME-Zarr:** Cloud-optimized, chunked, multi-resolution. Supports streaming/partial reads — you don't need to download entire terabyte-scale volumes.

### Directory Structure (per sample)

```
{SAMPLE_ID}/
├── volumes/            # 3D reconstructed volumes (OME-Zarr, sometimes TIFF)
├── segments/           # Extracted surfaces: meshes, surface volumes, ink results
└── representations/    # Derived artifacts (e.g., predictions, normals, fiber fields)
```

### Accessing Data with Python

```python
# Install: pip install vesuvius
import vesuvius

# Access scroll data without downloading terabytes
scroll = vesuvius.Volume("Scroll1")
```

### Accessing Data with C

```c
// Single-header library: vesuvius-c
#include "vesuvius-c.h"
```

---

## 7. Monorepo Structure (`villa`)

**Repo:** https://github.com/ScrollPrize/villa  
**License:** MIT  
**Language breakdown:** Python 42.3%, C++ 25.7%, Jupyter Notebook 25.2%, TypeScript 2.3%, CUDA 1.3%, C 0.8%

```
villa/
├── volume-cartographer/     # VC3D: semi-automatic segmentation pipeline (C++, CMake)
├── thaumato-anakalyptor/    # Thaumato: automatic segmentation (Python)
├── vesuvius/                # Python library for accessing scroll CT data
├── vesuvius-c/              # C single-header library for scroll data access
├── ink-detection/           # Grand Prize winning ink detection model
├── crackle-viewer/          # GUI tool for labeling ink on segments
├── foundation/              # Dataset management and cloud infrastructure tools
├── segmentation/            # Additional segmentation utilities
├── sam2-photogrammetry/     # SAM2-based photogrammetry tools
├── scripts/                 # Utility scripts
├── discord_chatbot/         # Discord bot
├── scrollprize.org/         # Website source (Docusaurus)
├── AGENTS.md                # Rules for AI coding agents (IMPORTANT — read this)
├── README.md
└── .gitmodules              # Submodules
```

---

## 8. Core Tools and Libraries

### vesuvius (Python)

```bash
pip install vesuvius
```

```python
import vesuvius

# Browse and stream scroll data — no local storage of full volumes needed
vol = vesuvius.Volume("Scroll1")
slice_data = vol[1000]  # Get slice 1000

# Access segments
seg = vesuvius.Segment("Scroll1", "20230702185753")
```

- GitHub: https://github.com/ScrollPrize/villa/tree/main/vesuvius
- Allows direct access to CT data without downloading multi-TB volumes
- Streams from S3

### vesuvius-c (C)

- Single-header C library: `vesuvius-c.h`
- GitHub: https://github.com/ScrollPrize/villa/tree/main/vesuvius-c
- Use for low-level or high-performance applications

### VC3D / Volume Cartographer (C++)

**The primary segmentation tool used by the Vesuvius Challenge team (as of September 2025).**

- GitHub: https://github.com/ScrollPrize/villa/tree/main/volume-cartographer
- Originally: https://github.com/educelab/volume-cartographer (EduceLab)
- Fork with community contributions by Philip Allgaier
- **Current fork used by team:** Hendrik Schilling and Sean Johnson's fork with OME-Zarr + tracer support

**Quick start with Docker (recommended):**
```bash
docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge

xhost +local:docker
sudo docker run -it --rm \
  -v "/path/to/data/:/path/to/data/" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
  ghcr.io/scrollprize/villa/volume-cartographer:edge
```

**Build from source:**
```bash
# Ubuntu/macOS, amd64/arm64
cd volume-cartographer
bash scripts/build_dependencies.sh
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

**Data requirements for VC3D:**
- All volumes must be OME-Zarr, dtype uint8
- Must include `meta.json`:
  ```json
  {"height":3550,"max":65535.0,"min":0.0,"name":"scroll name","slices":9778,
   "type":"vol","uuid":"20231117143551-zarr-s3_raw","voxelsize":7.91,"width":3400,"format":"zarr"}
  ```
- Must compute normal grids: `vc_gen_normalgrids`
- Pre-computed resources for Scroll 5 (PHerc172):
  - Normal grids: https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/normal_grids/
  - Fiber directions: https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/representations/direction_fields/

**Directory structure:**
```
scroll1.volpkg/
├── volumes/
│   ├── s1_uint8_ome.zarr/    # volume data
│   │   └── meta.json         # REQUIRED
│   └── 050_entire_scroll_ome.zarr/  # surface prediction
│       └── meta.json         # REQUIRED
├── paths/
├── normal_grids/             # REQUIRED
└── config.json               # REQUIRED
```

**Key CLI tools:**
```bash
vc_grow_seg_from_seed -v <volume.zarr> -t <paths_dir> -p <seed.json> -s <x y z>
vc_gen_normalgrids
vc_render_tifxyz
vc_tifxyz2obj
vc_fill_quadmesh
vc_tifxyz_winding
```

**VC3D UI key bindings:**
- `Ctrl + Left click`: center focus point
- `F` / `G`: push/pull surface along normal
- `1`–`5`: grow left/up/down/right/all
- `T`: new correction set
- `Shift + drag + E`: erase mesh region
- `Ctrl+Z`: undo
- `A` / `D`: alpha refinement along normals
- `S + Left click`: pull mesh along drawn path
- `Spacebar`: toggle overlay
- `C`: composite view

### Thaumato Anakalyptor

- GitHub: https://github.com/ScrollPrize/villa/tree/main/thaumato-anakalyptor (also: https://github.com/schillij95/ThaumatoAnakalyptor)
- Author: Julian Schilliger
- Approach: gradient-based point cloud extraction → Mask3D instance segmentation → graph-based winding angle assignment
- Included as additional material in 2023 Grand Prize submission

### Crackle Viewer / Vesuvius Kintsugi

- GUI for inspecting and labeling ink on virtually unwrapped segments
- GitHub: https://github.com/ScrollPrize/villa/tree/main/crackle-viewer
- Also: https://github.com/schillij95/Crackle-Viewer

### Ink Detection (Grand Prize Winner Model)

- GitHub: https://github.com/ScrollPrize/villa/tree/main/ink-detection
- Also: https://github.com/younader/Vesuvius-Grandprize-Winner
- Authors: Youssef Nader, Luke Farritor, Julian Schilliger
- Architecture: TimeSformer (primary), multiple architectures for cross-validation
- Key features: small input windows (64×64), label smoothing, domain adaptation, multiple validation folds

### Khartes

- Tool for manual segment creation and visualization with real-time flattened preview
- GitHub: https://github.com/KhartesViewer/khartes
- Author: Chuck
- Highly recommended for manual annotation work

### Spiral Fitting

- GitHub: https://github.com/pmh47/spiral-fitting
- Author: Paul Henderson (Prof, PhD)
- Top-down approach: fits deformed Archimedean spiral to scroll data
- Submitted for First Automated Segmentation Prize 2024; most elegant but didn't meet quality criteria

---

## 9. Segmentation Pipeline Deep Dive

### Goal

Isolate the 2-manifold (papyrus sheet surface) inside the 3D CT volume and flatten it into a readable 2D image.

### Current Semi-Automated Workflow (VC3D)

1. **Seed:** place initial seed point(s) on surface predictions via GUI or CLI
2. **Grow:** semi-automatically expand the mesh using the tracer optimizer
3. **Correct:** manually fix errors (sheet switches, mesh deformations)
4. **Repeat** until satisfactory coverage

**Sheet switching** is the most common error — the mesh jumps from one papyrus layer to an adjacent one.

### Three State-of-the-Art Automated Approaches

#### A) Spiral Fitting (Top-Down, Global)

**Pipeline:**
1. nnUNet predicts: surface sheets S, vertical fibers Fv, horizontal fibers Fh
2. Fit a canonical Archimedean spiral S₀ to these predictions via a 3D diffeomorphism (integrable, parametrized flow field u(x))
3. Minimize: `||S₀(x) + u(x) - S(x)||² + λR(u, Fv, Fh)` where R enforces fiber alignment

**Status:** submitted for First Automated Segmentation Prize 2024; didn't meet quality criteria but judged most elegant  
**Code:** https://github.com/pmh47/spiral-fitting

#### B) Surface Tracer (Bottom-Up, Local) — used in VC3D

**Pipeline:**
1. nnUNet predicts surface sheet voxels
2. Start from seed patches
3. Iteratively expand by minimizing: `λ_data * d(x_new, S) + λ_dist * ||x_new - x_fringe|| + λ_bend * B(x_new, M_current)`
4. Periodically smooth mesh to enforce developable surface
5. Multiple patches grown in parallel, merged via consensus

**Status:** very close to winning First Automated Segmentation Prize with ~4 hours human input; currently the active method used by the team  
**Code:** https://github.com/hendrikschilling/FASP, also within VC3D

#### C) Thaumato Anakalyptor (Point Cloud + Instance Segmentation)

**Pipeline:**
1. Apply box blur + Sobel-like gradient thresholding → surface point cloud (loosely 3D Canny edge)
2. Divide into overlapping chunks; apply Mask3D instance segmentation on each chunk
3. Build global connectivity graph where nodes = patch instances, edges = adjacency relationships with weights
4. Solve for consistent winding angles: `f* = argmax_f Σ w(e) * c(e)` where c(e) = 1 if f(n₁) - f(n₀) = observed angle difference
5. Stitch into single coherent mesh

**Status:** used in Grand Prize submission; shown working on Scroll 5; struggles with compressed regions  
**Code:** https://github.com/schillij95/ThaumatoAnakalyptor

### Key Challenge: All Methods Fail in Compressed Regions

All three methods struggle where papyrus layers are tightly compressed or damaged. The most common failure mode is **sheet switching** — incorrectly jumping between adjacent wraps of the scroll.

---

## 10. Ink Detection Deep Dive

### The Challenge

The Herculaneum ink is **radiolucent** (mostly transparent to X-rays). It's made from carbon-based materials (unlike iron-rich inks that are clearly visible in CT). This makes it extraordinarily hard to detect.

### What Was Discovered

**Two detectable ink signals (in Scrolls 1 and 5):**
1. **Crackle pattern** — ink causes microscopic cracking/deformation of papyrus fibers visible in surface volumes. Discovered by Casey Handmer in August 2023.
2. **Brighter X-ray spots** — slightly elevated radiodensity, possibly from trace metal content in the ink.

**Problem:** models trained on these signals don't generalize to most other scrolls. The ink in other scrolls may lack these characteristics, or require higher-resolution scans.

### Grand Prize Winning Approach

- **Architecture:** TimeSformer (primary) + multiple models for cross-validation
- **Input:** small windows (max 64×64 px = 0.5×0.5 mm at 8 µm resolution) from surface volumes
- **Key technique:** domain adaptation from fragment ground truth → scroll surface volumes
- **Anti-hallucination measures:** multiple architectures, no overlap between training and prediction regions, small window sizes, label smoothing, multiple validation folds
- **Open source:** https://github.com/younader/Vesuvius-Grandprize-Winner

### Fragment-Based Training

Detached fragments have **exposed ink visible to naked eye** (photographed + IR). These provide ground truth for training ink detection models. Strategy:

1. Train on fragment IR/CT-aligned binary masks
2. Apply domain adaptation to transfer learned representations to full scroll surface volumes

### 3D (Volumetric) Ink Detection

More recent approach: detect ink directly in the 3D volume without needing segmentation.
- **3D Ink Detection model:** https://github.com/ryanchesler/3d-ink-detection (Ryan Chesler)
- **Large Scroll Model:** 3D UNet pretrained on scroll data (https://github.com/ryanchesler/LSM)
- Predictions for Scrolls 1–4: https://dl.ash2txt.org/community-uploads/bruniss/3d%20Ink%20/

### Current Status (2025–2026)

- Ink in Scrolls 1 (PHerc. Paris. 4) and 5 (PHerc. 172): readable
- Ink in Scrolls 2, 3, 4, and all 30+ new ESRF-scanned scrolls: not yet detected
- Team exploring: higher-resolution scanning protocols, different scan energies, new ML approaches on fragment analysis

---

## 11. Virtual Unwrapping: State of the Art

### The Open Question

> Can we find an automated way to isolate the 2-manifold representing the rolled scroll in 3D images?

This question has been the core unsolved problem for 25+ years (Dr. Seales' research) and over 2 years of the Vesuvius Challenge. The $100K Automated Segmentation Prize offered in 2024 was **not claimed**.

### Current Cost vs Target

| Metric | Current | Target |
|--------|---------|--------|
| Cost to unwrap 1 full scroll | $1–5M | $5,000 |
| Total cost for 300 scrolls | $300M–$1.5B+ | $1.5M |
| Human input for reasonable result | 4+ hours | 0 hours (full automation) |

### What "Unwrapping at Scale Prize" ($200,000) Requires

Automate the unwrapping of an entire scroll to a quality sufficient for ink detection. No prize has been claimed yet.

---

## 12. Community Projects and Ecosystem

> **Canonical, always-current list:** `awesome-scroll-tools` = `~/scroll_prize/villa/scrollprize.org/docs/20_community_projects.md` (GitHub: `ScrollPrize/villa/.../20_community_projects.md`). The Progress Prize form **requires a PR adding your tool there.** The tables below are a curated snapshot — check the canonical list before assuming a tool doesn't exist.
>
> **Recent / strategically-relevant additions (as of 2026-06-04):**
> - **Scroll-specific augmentations** (pscamillo, PR #999) — Squeeze/Decohesion/Warp; **closes wishlist issue #201**, so that idea is taken.
> - **ScrollMAE** (jgcarrasco) — 3D ResNet pretrain→finetune for ink; **DINO ink detection** (jgcarrasco, unsupervised, Colab); **DINOv2 models** (Pnev, pretrained on Scrolls 1-5).
> - **3D ink detection** (Ryan Chesler) + **LSM** 3D U-Net; Sean Johnson's volumetric ink predictions (Scrolls 1-4).
> - **Iterative labeling** (Youssef Nader) — the documented lever for improving ink predictions; **Inkalyzer** (Nader) — XAI + volumetric labels.
> - **Vesuvius GP+** (Jared Landau) — GP ink script + extras; **VesuviusPretraining** (Nader) — the First-Letters pretrain→finetune recipe.
> - **Ayush Mishra** — ink detection with rescaled fragments, Gabor-filter surface prediction, affinity-Unet segmentation.
> - See `knowledge_base.md` §11 Priority 3 for which of our ideas are already done vs still open, and where our work is still differentiated.

### Data Access / Visualization

| Tool | Author | Description |
|------|--------|-------------|
| [vesuvius](https://github.com/scrollprize/vesuvius) | ScrollPrize | Python library, streaming S3 access |
| [vesuvius-c](https://github.com/ScrollPrize/villa/tree/main/vesuvius-c) | ScrollPrize | C single-header library |
| [vesuvius-gui](https://github.com/jrudolph/vesuvius-gui) | Johannes Rudolph | Single binary GUI for volumes and segments |
| [Segment browser](https://github.com/jrudolph/vesuvius-browser) | Johannes Rudolph | Web-based browser for segments and ink detection results |
| [vesuvius-phalanx](https://github.com/mvrcii/phalanx) | Marcel Roth | Python/CLI flexible data access |
| [llfio-chunkloader](https://github.com/climbmax123/LLFIOCunkloadingTestingAndBenching) | — | Faster chunk loading than Zarr in C++ |
| [Scroll Viewer](https://github.com/lukeboi/scroll-viewer) | Luke Farritor | Lightweight browser-based volumetric viewer |
| [Scroll Sleuth](https://github.com/Paul-G2/ScrollSleuth) | Paul Geiger | Web app for visual ink search in segment volumes |
| [Neuroglancer Mini](https://github.com/tomhsiao1260/neuroglancer-mini) | Yao Hsiao | Trimmed Neuroglancer for scroll data |
| [vesuvius-blender](https://github.com/spelufo/vesuvius-blender) | Santiago Pelufo | Explore X-ray scans in Blender |

### Segmentation Tools

| Tool | Author | Description |
|------|--------|-------------|
| [Volume Cartographer / VC3D](https://github.com/ScrollPrize/villa/tree/main/volume-cartographer) | EduceLab → Hendrik Schilling/Sean Johnson | Primary semi-automatic segmentation tool |
| [Khartes](https://github.com/KhartesViewer/khartes) | Chuck | Manual mesh creation with live flattened preview |
| [Thaumato Anakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) | Julian Schilliger | Automatic segmentation via point clouds |
| [Volumetric Vesuvius Labelling](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling) | James Darby | Napari 3D viewer for 3D annotation |
| [Hraun](https://github.com/SuperOptimizer/Hraun) | Forrest McDonald | Python tools for volumetric scroll data |
| [Volume Annotate](https://github.com/MosheLevy20/VolumeAnnotate) | Moshe Levy | Python reimplementation of Volume Cartographer |
| [scrollreading](https://github.com/WillStevens/scrollreading) | Will Stevens | Flood-fill based surface extraction |
| [Spiral Fitting](https://github.com/pmh47/spiral-fitting) | Paul Henderson | Fully automatic top-down global spiral fitting |
| [Segment Flattening (SLIM)](https://github.com/giorgioangel/slim-flatboi) | Giorgio Angelotti | SLIM algorithm for isometric flattening |

### Ink Detection Tools

| Tool | Author | Description |
|------|--------|-------------|
| [Grand Prize Winner](https://github.com/younader/Vesuvius-Grandprize-Winner) | Nader/Farritor/Schilliger | TimeSformer-based, production ink detection |
| [Ink-ID](https://github.com/educelab/ink-id) | Stephen Parsons | EduceLab ink detection baseline |
| [3D Ink Detection Model](https://github.com/ryanchesler/3d-ink-detection) | Ryan Chesler | Volumetric ink detection without segmentation |
| [Large Scroll Model (LSM)](https://github.com/ryanchesler/LSM) | Ryan Chesler | 3D UNet pretrained on scroll data |
| [Inkalyzer](https://github.com/younader/Inkalyzer) | Youssef Nader | XAI package for explaining ink model predictions |
| [Vesuvius Kintsugi](https://github.com/giorgioangel/vesuvius-kintsugi) | Giorgio Angelotti | Tool for labeling surface volumes |
| [ScrollMAE](https://github.com/jgcarrasco/ScrollMAE) | Jorge García | 3D ResNet pretraining + finetuning for ink |
| [DINOv2 for scrolls](https://github.com/SergeyPnev/dinov2-vesuvius) | Sergei Pnev | Self-supervised pretraining on scrolls 1–5 |

### Materials / Datasets Available

| Resource | Description |
|----------|-------------|
| [Denoised volumes](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_denoised_ce/) | Scroll 1 denoised and contrast-enhanced |
| [Surface predictions S1/S3](https://dl.ash2txt.org/community-uploads/bruniss/p2_submission/) | Sean Johnson's nnUNet predictions |
| [Surface predictions S4](https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/Predictions/s4/) | Sean Johnson |
| [3D Ink predictions](https://dl.ash2txt.org/community-uploads/ryan/) | Ryan Chesler, all scrolls in Zarr format |
| [Volumetric instance labels](https://github.com/JamesDarby345/Vesuvius_3D_datasets) | James Darby |
| [Campfire scroll CT](https://dl.ash2txt.org/community-uploads/waynewaynehello/) | Ahron Wayne — replicated carbonization process |

---

## 13. Key People

### Vesuvius Challenge Core Team

| Name | Role |
|------|------|
| Nat Friedman | Instigator & Founding Sponsor |
| Daniel Gross | Founding Sponsor |
| Giorgio Angelotti | Research Project Lead, PhD |
| Sean Johnson | Research Assistant (segmentation/surface predictions) |
| Youssef Nader | Machine Learning Researcher |
| Hendrik Schilling | Computer Vision & AI Expert, PhD (VC3D lead) |
| Paul Henderson | Computer Vision & AI Expert, PhD (Spiral Fitting) |
| Elian Rafael Dal Prá | ML Intern & Annotation Specialist |
| Johannes Rudolph | Platform Engineer |
| Forrest McDonald | Software Engineer |
| David Josey | ML Annotation Team Lead, PhD |
| Eric Thvedt | Annotation Specialist |
| Kendra Brown | Annotation Specialist |
| Sarah Morejohn | Annotation Specialist |

### EduceLab Team (University of Kentucky)

| Name | Role |
|------|------|
| W. Brent Seales | Principal Investigator, Professor of CS |
| Seth Parker | Research Manager |
| Christy Chapman | Research & Partnership Manager |
| Mami Hayashida | Research Staff |
| James Brusuelas | Associate Professor of Classics |
| Roger Macfarlane | Professor of Classical Studies |

### Advisors / Alumni

| Name | Notable Contribution |
|------|---------------------|
| JP Posma | Project Lead |
| Stephen Parsons | Project Lead, PhD; Ink-ID; Hard-Hearted Scrolls dissertation |
| Ben Kyles | Segmentation Team Lead |
| Julian Schilliger | Grand Prize winner; ThaumatoAnakalyptor; SLIM flattening |
| Daniel Havíř | Machine Learning |

### Key Community Contributors

| Name | Contributions |
|------|--------------|
| Youssef Nader | Grand Prize winner; First Letters Prize; domain adaptation techniques |
| Luke Farritor | Grand Prize winner; First Letters Prize (first word ever: ΠΟΡΦΥΡΑϹ) |
| Casey Handmer | Discovered crackle pattern (First Ink Prize) |
| Ryan Chesler | Kaggle 1st place; 3D ink detection models |
| Marcel Roth | First Title Prize winner |
| Micha Nowak | First Title Prize winner |
| Philip Allgaier | Volume Cartographer active fork maintainer |
| James Darby | Volumetric labels and segmentation tools |
| Giorgio Angelotti | Autosegmentation preprocessing; SLIM flattening; visualization |
| Yao Hsiao | Browser-based visualization tools |
| Santiago Pelufo | Meshing, chunking, Blender tools |
| Chuck | Khartes development |
| Paul Geiger | Scroll Sleuth, Scroll Slab Viewer, alignment tools |
| Jared Landau | Ink detection tutorials and Vesuvius GP+ |

### Papyrology Team

| Name | Institution |
|------|-------------|
| Federica Nicolardi (Lead) | Univ. of Naples Federico II |
| Marzia D'Angelo | Univ. of Naples Federico II |
| Kilian Fleischer | Univ. Tübingen & CNR |
| Alessia Lavorante | Univ. of Naples Federico II |
| Michael McOsker | UCL |
| Maria Chiara Robustelli | Univ. of Naples Federico II |
| Claudio Vergara | Univ. of Naples Federico II |
| Rossella Villa | Univ. of Salerno |

### Papyrology Advisors

- Daniel Delattre (CNRS/IRHT)
- Gianluca Del Mastro (Univ. Campania)
- Robert Fowler (British Academy / Bristol)
- Richard Janko (Univ. Michigan)
- Tobias Reinhardt (Oxford)

---

## 14. Sponsors and Funding

### Major Sponsors

| Sponsor | Amount |
|---------|--------|
| Musk Foundation | $2,084,000 |
| Alex Gerko (XTX Markets) | $450,000 |
| Joseph Jacks | $250,000 |
| Nat Friedman | $225,000 |
| Daniel Gross | $225,000 |
| Matt Mullenweg | $150,000 |
| John & Patrick Collison (Stripe) | $125,000 |
| Emergent Ventures | $100,000 |
| Julia DeWahl & Dan Romero | $100,000 |
| Eugene Jhong | $100,000 |

### Partners

- EduceLab (University of Kentucky)
- Institut de France (scroll custodian)
- Biblioteca Nazionale di Napoli (scroll custodian)
- Getty (museum partner)
- Kaggle (competition platform)
- ESRF (European Synchrotron Radiation Facility, Grenoble — BM18 beamline)
- Diamond Light Source (Oxford — original scanning facility)

### EduceLab Funders

- National Science Foundation
- National Endowment for the Humanities
- Andrew W. Mellon Foundation
- Digital Restoration Initiative
- Arts & Humanities Research Council (UKRI)

---

## 15. Master Plan: Stages 1–4

### Stage 1 (Complete): First Proof of Concept

- Proved virtual unwrapping + ML ink detection works on carbonized scrolls
- Extracted 15 columns from Scroll 1 (PHerc. Paris. 4)
- Grand Prize awarded February 2024

### Stage 2 (Current): Reading Multiple Scrolls

**Goal:** Read entire scrolls, not just passages.

**Two key bottlenecks:**
1. **Unwrapping at Scale** — current methods too slow/expensive for 300 scrolls
2. **Ink Identification** — doesn't generalize beyond 2 scrolls yet

**Cost estimate:** $5–6M for Stage 2 (partially funded via Musk Foundation donation)

**Scanning achievements (as of July 2025):**
- 30+ scrolls scanned at ESRF BM18 with tetra/hexa/octa-helical acquisitions
- 9.2 µm resolution, full scroll scanned in under 2 hours
- Ink remains elusive in all new data despite resolution improvements
- Title of PHerc. 172 confirmed: "On Vices, Book 1" by Philodemus

### Stage 3 (Future): Industrialize Reading All 300 Scrolls

- After full automation of unwrapping and generalizable ink detection
- Scan + read all 300 extant scrolls (mostly in Naples)
- Estimated timeline: 2–3 years after Stage 2
- **Cost estimate:** $4–8M if efficient accelerator protocol works; $15M+ if not

### Stage 4 (Long-Term): Excavate the Villa

- The Villa dei Papiri has two unexcavated levels — almost certainly containing more scrolls
- Main library (which would contain Greek and Latin literature, not just philosophy) has never been found
- Potentially tens of thousands of scrolls still buried
- Largely a political/funding effort — the breakthrough discoveries from Stages 2–3 are expected to catalyze political will
- If Stage 3 results don't inspire excavation, team will push directly

### The Payoff

- Overfit stories of history get rewritten
- Never-before-seen ancient literature revealed
- Possibly: Aristotle dialogues, lost books of Livy, Homeric epics, Sappho poems
- "The greatest revolution in the classics since the Renaissance" (team's papyrologists)

---

## 16. How to Contribute (Step by Step)

### Step 1: Join the Community

- Discord: https://discord.com/invite/uTfNwwecCQ (primary collaboration channel)
- Substack (mailing list): https://scrollprize.substack.com
- Twitter/X: https://x.com/scrollprize
- Community survey: https://forms.gle/mtA3B4uQusVFTEDu9

### Step 2: Get Familiar with the Data

```python
# Option A: View in browser
# https://dl.ash2txt.org/view/Scroll1

# Option B: Python
pip install vesuvius
python -c "import vesuvius; v = vesuvius.Volume('Scroll1'); print(v[1000].shape)"

# Option C: Example notebook
# https://github.com/ScrollPrize/open-data/blob/main/examples/get-to-know-a-dataset.ipynb
```

### Step 3: Choose a Contribution Track

**Track A: Segmentation (finding papyrus surface in 3D)**
- Learn VC3D (see segmentation tutorial: https://scrollprize.org/segmentation)
- Read Virtual Unwrapping doc: https://scrollprize.org/unwrapping
- Check open issues labeled `VC3D` or `help wanted`: https://github.com/ScrollPrize/villa/issues

**Track B: Ink Detection (finding ink in flattened segments)**
- Read ink detection tutorial: https://scrollprize.org/tutorial5
- Look at Grand Prize model: https://github.com/younader/Vesuvius-Grandprize-Winner
- Key targets: Scrolls 2–3 (first letters), Scrolls 1–4 (first title for Scroll 1/2/3/4)
- Prize: 7 × $60,000

**Track C: Open Source Tools (progress prizes)**
- Check wishlist: https://github.com/ScrollPrize/villa/issues?q=is%3Aissue+state%3Aopen+label%3A%22help+wanted%22
- Check good first issues: https://github.com/ScrollPrize/villa/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22
- Submit by last day of month

**Track D: Annotation / Data Labeling**
- Help annotate volumetric labels for 3D segmentation: https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/cubes/
- Produce ink labels for new segments

### Step 4: Submit

- **Progress Prizes:** https://forms.gle/Sy6mW5cfJS2U7E9F7
- **First Letters/Title:** https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform

### Step 5: Open Source

All prize winners must release code under a permissive license.

---

## 17. AGENTS.md: Rules for Coding Agents in This Repo

The repo has an `AGENTS.md` at the root (https://github.com/ScrollPrize/villa/blob/main/AGENTS.md) that all coding agents (Codex, Claude, etc.) must follow.

### Critical Rules

1. **Identify the target subproject first** — each top-level folder is an independent product
2. **Do NOT run install/bootstrap commands by default** — treat discovery as read-only unless explicitly asked
3. **Scoped installs only** — set `AGENTS_AGENT_MODE=1` and `AGENTS_ALLOW_INSTALL=1` for automated installs
4. **Smallest change that solves the task** — avoid large refactors unless explicitly requested
5. **Read subproject docs first** — look for `README`, `CMakeLists.txt`, `pyproject.toml`, `Dockerfile`, etc.
6. **Portability required** — target Ubuntu + macOS, amd64 + arm64
7. **Tests are not optional** — run existing tests or add minimal regression tests
8. **Performance work must be measured** — baseline → profiler → before/after stats

### Subproject Playbooks

#### `volume-cartographer/` (C++, CMake)
- Build: `bash scripts/build_dependencies.sh` then CMake
- Performance constraint: **no numeric changes** (no `-ffast-math`, no precision reduction)
- Profiling: `RelWithDebInfo`; final perf: `Release`
- Export compile commands: `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`

#### `vesuvius/` (Python, ML)
- Detect environment from: `pyproject.toml`, `requirements*.txt`, `environment.yml`, `poetry.lock`, `uv.lock`, `Dockerfile`
- Preserve: fixed seeds, stable evaluation protocols, exact preprocessing
- Safe speedups: data loading (caching, prefetching), redundant preprocessing elimination, I/O improvements
- Do NOT change: model precision, kernels, quantization, batch sizing

### Agent Output Format

When completing a task, provide:
1. What you changed (files + rationale)
2. How to build and run (exact commands)
3. How you verified (tests + dataset/inputs)
4. Perf results (if applicable) with before/after
5. Risks/limitations (OS/arch edge cases)

---

## 18. Wish List / Good First Issues

Check current list at: https://github.com/ScrollPrize/villa/issues?q=is%3Aissue+state%3Aopen+label%3A%22help+wanted%22

General wishlist areas as of early 2026:

### High Impact (likely Gold Aureus / Denarius level)
- Better automated ink detection generalizing across scrolls (especially new ESRF scans)
- Fully automated segmentation that handles compressed scroll regions
- Improved nnUNet-based surface/fiber predictions with higher accuracy and fewer false mergers
- Methods to distinguish between adjacent papyrus layers in very compressed areas
- New scanning analysis: characterize ink in fragments vs full scrolls at different resolutions

### Medium Impact (Sestertius / Denarius)
- Improvements to VC3D's tracer method (speed, accuracy, sheet-switch detection)
- Better mesh smoothing and flattening algorithms (less distortion)
- Automatic sheet-switch detection and correction
- Tools for comparing segmentation quality across different methods
- 3D ink detection improvements for Scrolls 2–4

### Good First Issues
- VC3D UI improvements and bug fixes
- Documentation improvements
- New visualization tools for segment inspection
- Data format converters (Zarr ↔ TIFF, OBJ processing utilities)
- Python wrappers for VC3D CLI tools
- Performance optimizations in vesuvius library

---

## 19. Formats, Conventions, and File Specs

### Volume Files

```
# OME-Zarr structure
{volume.zarr}/
├── 0/          # highest resolution
│   ├── .zarray
│   └── {chunks}
├── 1/          # 2x downsampled
├── 2/          # 4x downsampled
└── .zattrs     # OME metadata

# dtype: uint8 (preferred) or uint16
# check dtype: look at .zarray → "dtype" field
# "|u1" = uint8, "|u2" = uint16
```

### Segment / Mesh Files

```
# Standard segment in .volpkg
{segment_id}/
├── {segment_id}.obj          # triangular mesh (3D coordinates)
├── {segment_id}_normals.tif  # surface normals
└── layers/
    ├── 00.tif                # surface volume layer 0
    ├── 01.tif                # surface volume layer 1
    ...
    └── 64.tif                # typically 32 layers above/below surface
```

### TIFXYZ Format

```
# Three TIFF files encoding x, y, z coordinates + metadata JSON
{name}_x.tif   # x coordinates at each pixel
{name}_y.tif   # y coordinates at each pixel  
{name}_z.tif   # z coordinates at each pixel
{name}.json    # metadata: dimensions, voxel size, etc.
```

### Segment Submission Format

```python
# For prize submissions
submission = {
    "scroll_id": "Scroll2",           # which scroll
    "segment_id": "20230702185753",   # segment ID or file
    "position_3d": (x, y, z),         # 3D location in scroll
    "method": "...",                   # methodology description
}
```

### Volpkg Config

```json
{
  "version": "2.0",
  "name": "PHerc0172",
  "materialthickness": 150
}
```

### Meta.json (required for VC3D)

```json
{
  "height": 3550,
  "max": 65535.0,
  "min": 0.0,
  "name": "descriptive name",
  "slices": 9778,
  "type": "vol",
  "uuid": "unique-id",
  "voxelsize": 7.91,
  "width": 3400,
  "format": "zarr"
}
```

### Ink Model Input Constraints

- **Maximum window size for ink detection submissions:** 0.5×0.5 mm = 64×64 px at 8 µm voxel size
- **No overlap** between training data regions and prediction regions
- Models should be reproducible (Docker preferred for automated pipelines)

---

## 20. Citations and Licensing

### Dataset Citations

**EduceLab-Scrolls (Scrolls 1–4, Fragments 1–6):**
```
Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023).
EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT.
ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084
```
Include in methods: *"Data used in the preparation of this article were obtained from the EduceLab-Scrolls dataset."*

**Vesuvius Challenge - CT Scans of Herculaneum Papyri (newer scans):**
```
Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph,
Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald,
David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales.
Vesuvius Challenge - CT Scans of Herculaneum Papyri. Vesuvius Challenge.
```

### Licenses

- Data: **CC-BY-NC 4.0** (unless otherwise noted)
- `villa` repository code: **MIT**

### Key Papers

- Parsons et al. 2023 — EduceLab-Scrolls dataset: https://arxiv.org/abs/2304.02084
- Parsons (dissertation) — Hard-Hearted Scrolls (ink detection in CT): https://uknowledge.uky.edu/cs_etds/138/
- Rabinovich et al. 2017 — Scalable Locally Injective Mappings (flattening): ACM TOG 36(4)
- Isensee et al. 2024 — nnU-Net Revisited (segmentation backbone): MICCAI 2024
- Schult et al. 2023 — Mask3D (instance segmentation for Thaumato): ICRA 2023
- Canny 1986 — Computational Approach to Edge Detection: IEEE TPAMI 8(6)

---

## Quick Reference Card

```
Data access:
  s3://vesuvius-challenge-open-data/
  pip install vesuvius

Primary tool (segmentation):
  docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge

Main repo:
  git clone https://github.com/ScrollPrize/villa

Submit progress prize:
  https://forms.gle/Sy6mW5cfJS2U7E9F7  (deadline: last day of each month)

Submit First Letters/Title:
  https://docs.google.com/forms/d/e/1FAIpQLSdw43FX.../viewform

Current open problems in priority order:
  1. Ink detection generalizing to new scrolls
  2. Automated segmentation / unwrapping at scale
  3. High-quality annotation of volumetric labels

Current prize targets:
  - $60k × 6: First Letters in Scrolls 2-3
  - $60k × 6: First Title in Scrolls 1-4
  - $200k: Unwrapping at Scale
  - $1k–$20k/month: Progress Prizes

Community: Discord https://discord.com/invite/uTfNwwecCQ
Updates: https://scrollprize.substack.com
```

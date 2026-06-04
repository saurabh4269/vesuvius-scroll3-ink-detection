# Understanding This Project — A Plain-Language Master Guide

*Written for you, the project owner, to understand everything happening here without needing a background in machine learning, papyrology, or high-performance computing. No prior knowledge assumed. Read it top to bottom once; after that it's a reference.*

*Last updated: 2026-06-04*

---

## 0. The 60-second version (read this first)

- We are trying to **read 2,000-year-old burned scrolls** that can't be physically unrolled, using X-ray scans and software. This is a real, famous research competition called the **Vesuvius Challenge**, with large cash prizes.
- We built a **software tool** that takes the scan data and tries to make hidden letters visible.
- We thought we'd **found a letter** in one scroll (PHerc.332) and posted it publicly. Then we **carefully checked our own work** and discovered the "letter" is almost certainly just a **papyrus fiber**, not ink. We **corrected the public posts** — honestly and quickly.
- The **good news**: while checking, we *proved our tool actually works* (it can render real letters when they exist). So the tool itself is a genuine, useful contribution.
- **Right now**: we're pointing that proven tool at **all ~30 scrolls** at once (not just the one), to see if a real letter shows up anywhere.

That's the whole story. The rest of this document explains every piece of it so you can follow along and make good calls.

---

## Part 1 — The Big Picture: What is the Vesuvius Challenge?

### The scrolls

In 79 AD, Mount Vesuvius erupted and buried the Roman town of Herculaneum. A library of papyrus scrolls was **carbonized** — turned to charcoal — by the heat. They survived, but they're now fragile lumps of carbon. If you try to physically unroll one, it crumbles to dust. For 250 years they've been essentially unreadable.

These are the **only surviving library from the ancient world**. Reading them could recover lost works of philosophy, science, and literature. That's why this matters and why there's serious money behind it.

### The modern approach: don't touch them, scan them

Instead of unrolling, researchers put the scrolls in a **particle accelerator** and take an extremely high-resolution **X-ray CT scan** — the same idea as a hospital CT scan, but far finer. This produces a **3D digital model** of the scroll: a block of data where every tiny point in space has a brightness value (how dense the material is there).

Now the challenge becomes a **software problem**, broken into three hard steps:

1. **Segmentation** — find the rolled-up sheet of papyrus inside the 3D block and "flatten" it digitally. Imagine digitally peeling and laying flat a tightly-wound spiral. This is genuinely difficult; whole tools exist just for this (one is called **Volume Cartographer**, another **ThaumatoAnakalyptor**).
2. **Ink detection** — the carbon ink looks almost identical to the carbon papyrus in the scan. You can barely see the difference by eye. So people train **AI models** to spot the subtle texture differences that mark where ink is.
3. **Reading** — once ink is detected and laid flat, papyrologists (ancient-text experts) read the letters.

### The prizes (why we care which scroll)

| Prize | Amount | What it requires | Status |
|-------|--------|------------------|--------|
| **First Letters** | **$60,000** | 10 readable letters in one 4 cm² area of **any of Scrolls 2-3** | Open, no winner |
| **First Title** | **$60,000** | A readable image of the title in **any of Scrolls 2-3** | Open, no winner |
| **Progress Prizes** | **$1k–$20k** | A *tool or method* the community adopts/uses (Papyrus → Gold Aureus tiers) | Monthly, rolling |

**Key point that affects our decisions:** the big $60k prizes are tied to **specific scrolls** (Scroll 2 **and** Scroll 3 — both qualify, not just Scroll 3). A finding in some *other* scroll doesn't qualify for those. But the **Progress Prizes reward useful tools** regardless of scroll, are judged **every month**, and explicitly value work that is open-sourced early, actually gets used, and is well documented — that's the realistic target for what we've built.

Also important: First Letters/Title submissions must be **programmatic outputs** (no hand-drawn letters), must **not** reuse training regions, and the judges **explicitly warn against window sizes bigger than 0.5×0.5 mm** because large windows let models hallucinate letters. Our pareidolia/positive-control checks exist precisely to satisfy this.

### What others have already done, and what's still wanted

(Grounded in the official Vesuvius docs that ship inside villa — not guesswork.)

- **2023:** the $850k Grand Prize — the first text ever read from an unopened scroll (Scroll 1). The breakthrough chain: Casey Handmer spotted a **"crackle pattern"** in the scan that looks like ink → Luke Farritor trained an AI on it and read the word **ΠΟΡΦΥΡΑϹ** ("purple") → Youssef Nader improved it with *domain transfer* (pretrain on scroll, fine-tune on labeled fragments). Total prizes awarded across the project so far: ~$1.78M.
- **2024–2025:** words found in a *second* scroll; the **title** of one scroll read (author: **Philodemus of Gadara**); focus moved to faster scanning (new ESRF beamline runs).
- **The methods people use:** for ink, an ensemble of **TimeSformer** (a video-style transformer) + **3D ResNet** + **I3D** models; for unwrapping, tools called **VC3D**, **spiral-fitting**, and **ThaumatoAnakalyptor**. Villa (which we now build on) contains all of these.
- **What's still wanted (the hard open problem):** the 2023/24 ink models learned to amplify **cracks** and **metal-rich bright spots** — and those signals **don't show up in the newer scrolls**. In the team's own words, "ink remains elusive in all our new data." They're now searching for *different* ink characteristics, possibly needing higher-resolution scans.

**Why this matters for us:** our model was trained on that same crack/metal kind of signal, so when it sees Scroll 3 it mostly lights up on papyrus *fiber texture*, not ink. That's not just our bug — it's the wall the whole field is currently stuck at. So the valuable thing we can offer right now isn't "another letter claim," it's **honest, hallucination-proof tooling** — which is exactly what the Progress Prize rewards.

---

## Part 2 — The Data (what we're actually working with)

### A CT scan is a 3D grid of brightness

Picture a stack of thousands of grayscale photographs, one on top of another. Together they form a solid 3D block. Each tiny cube in that block is called a **voxel** (a 3D pixel). Its value is how much X-ray it absorbed — roughly, how dense the material is there.

- **Resolution** is measured in **µm/px** (micrometers per pixel). "1.2 µm/px" means each voxel is 1.2 millionths of a meter wide — incredibly fine. Smaller number = more detail = much bigger files.
- A single scroll scan can be **hundreds of gigabytes**.

### Multiscale "levels" — same data, different zoom

Because the full-resolution data is enormous, it's stored as a **pyramid** of versions:

- **Level 0** = full resolution (e.g. 1.2 µm/px) — huge, sharp.
- **Level 1** = half resolution (2.4 µm/px) — 8× smaller file.
- **Level 2** = quarter (4.8 µm/px), **Level 3** = eighth (9.6 µm/px), and so on.

You use a **coarse level** (small, fast) to scan around and find interesting spots, then drop to a **fine level** (level 0) only at the exact spot you care about. We do exactly this.

### The file format: "zarr"

The 3D data is stored in a format called **zarr**. You don't need to know the internals — just that it's a way to store a giant 3D array as many small **chunk** files (each a 192×192×192 cube here), so software can load only the pieces it needs instead of the whole 200 GB.

### Where the data lives: a public cloud bucket

All the official data sits in an **Amazon S3 bucket** (cloud storage) called `vesuvius-challenge-open-data`. It's **public** — the `--no-sign-request` flag in our commands just means "I'm accessing public data, no login needed." Anyone, including anyone verifying our work, can download exactly what we used.

### Three different *kinds* of data you'll hear about

This trips people up, so be clear on it:

1. **Raw CT volume** — the original scan. Just density. Ink and papyrus look nearly the same.
2. **Segments** — a sheet of the scroll that's been found and flattened. Sometimes saved as a flat image.
3. **Surface predictions** — the output of an AI model (nicknamed **m7**) that looks at the raw 3D scan and predicts *where the physical papyrus sheet is* in 3D space. These are used as input to segmentation tools (ThaumatoAnakalyptor, VolumeCartographer) so they can trace the sheet. **This is not ink detection** — it tells you where the sheet is, not where ink is on it.
4. **Ink predictions** — a separate step that runs *after* segmentation, on a flattened segment. Takes the surface volume and outputs a 2D map of where ink probably is. This is what our B1 model produces.

**Important correction:** early in this project we mistakenly called the m7 surface-prediction zarrs "ink predictions." They are not. We were sampling sheet location data and looking for letters in it, which is why nothing was found. Actual ink predictions only exist as 2D outputs after a segment has been created.

---

## Part 3 — Our Idea: "Cylindrical Unrolling"

### The problem in one picture

A scroll is a long sheet wound into a tight spiral, like a rolled-up newspaper or a roll of paper towels. The text sits **on the sheet**, which means in the 3D scan the text is wound around and around in rings. To read it, you have to "unwind" the spiral.

The normal way (Volume Cartographer / ThaumatoAnakalyptor) is to **carefully trace the actual sheet** through all its bends and deformations. That's accurate but slow and hard — the scrolls are crushed and warped.

### Our shortcut

We tried a simpler idea: **pretend the scroll is a perfect cylinder.** Then, instead of tracing the warped sheet, we just **sample the data in circles** at a fixed distance from the center.

Analogy: imagine the roll of paper towels. Pick a distance from the cardboard tube — say "3 cm out." Walk all the way around the roll at exactly 3 cm out, reading off what's there. You've just "unrolled" one layer into a straight strip. Do it at 3.1 cm, 3.2 cm, etc., and you sweep through the layers like **peeling an onion**.

In our numbers:
- **r** ("radius") = how far out from the center we sample, in pixels. `r=310` is one specific layer.
- **angle** = where around the circle we are (0° to 360°, which we split into 1800 steps).
- The result is an **"unrolled strip"** — a flat rectangle (height = scroll height, width = going around the circle) that *might* show letters lying flat, if the cylinder assumption holds well enough at that spot.

### Why this could be valuable

If applied to real 3D ink data, it could read letters without the slow segmentation step. As a quick way to understand scroll geometry from surface predictions, it also has some value — but it cannot find ink in surface prediction data because surface predictions don't contain ink information.

### Why this could fail

The scrolls are **not** perfect cylinders — they're crushed. If the assumed center is off, or the sheet bends, the circles cut across the sheet at an angle and **smear** any letters into garbage. So the method needs proof that it actually renders real letters, not just "something."

### CLAHE — the powerful, dangerous enhancement

After unrolling, the strip is faint. We run a contrast-booster called **CLAHE** (Contrast Limited Adaptive Histogram Equalization). It dramatically sharpens local contrast and makes faint patterns pop.

**This is the single most dangerous tool in the whole pipeline, and you must understand why:** CLAHE will happily turn **random noise into convincing-looking patterns**. Crank up contrast on a blurry nothing and your eye starts seeing structure — like staring at TV static until you "see" shapes. This is called **pareidolia** (the same reason people see faces in clouds or in burnt toast). A huge fraction of false "discoveries" in this field come from over-enhanced noise. Keep this word in mind: **pareidolia**.

---

## Part 4 — What We Claimed (and Why It Was Shaky)

We ran the pipeline on **PHerc.332 (Scroll 3)** and found, at `r=310`, a small (~1 mm) shape that looked like it had loops — possibly a Greek letter like β (beta) or φ (phi). We also looked at another scroll, **PHerc0009B**, and thought we saw letters Π (pi) and Ο (omicron).

We posted both to the **Vesuvius Challenge Discord** (the community chat where researchers share findings), and we used strong words: *"confirmed,"* *"diagnostic signature of real carbonized ink,"* *"the only letter in the entire scroll."*

**The problem:** that confidence was not backed by the evidence. We were reading enhanced images by eye and trusting what we wanted to see. Three specific overstatements:

1. Calling it **"confirmed ink"** when we hadn't ruled out it being a fiber or a crack.
2. Saying **"the only letter in the whole scroll"** when we'd actually only searched a thin accessible layer — most of the scroll was unreadable with our method, so we hadn't really looked at "the whole scroll."
3. Identifying it as a **specific Greek letter** — that's a job for a papyrologist, and even they'd want much better evidence.

---

## Part 5 — The Self-Audit (the most important part)

When you make a public scientific claim, the right thing — and the thing that protects your reputation — is to **try as hard as you can to prove yourself wrong** before others do. If your finding survives your own brutal attack, it's probably real. If it doesn't, you want to be the one who caught it. We did this. Here's each test in plain language.

### Test 1: "Does our own automatic filter agree?"

We had earlier written a script (`r310_fullsearch.py`) that **automatically scores** candidates using a rule (described below). When we actually read its saved output, it said:

> **0 of 10 candidates passed. Including the one we posted.**

So our *own* automated, unbiased filter had **rejected** the very thing we called "confirmed." We'd been relying on hand-picked pretty pictures instead of our own numbers. That's the first red flag, and a decisive one.

### The "gradient test" — explained simply

This is the core idea, so here's the plain version. Real ink sits in a **thin layer** on the sheet's surface. So if you sample just inside it, right at it, and just outside it, you should see: **nothing → ink → nothing.** A sharp, isolated spike at one depth. Like a single sheet of printed paper: the ink is on the surface, not in the air above or in the table below.

We sample at 5 depths (radii) around the candidate: `r=298, 304, 310, 316, 322`. A real letter should **peak sharply at one** and be near-empty on both sides.

### Test 2: The "salvage" controls (`salvage_test.py`)

We ran the exact same pipeline on three control cases to see if our candidate was special:

- **Empty region** — run the pipeline where there's nothing. (If it produces "letters" too → it's an artifact.)
- **Shuffled** — take our candidate and scramble it, destroying any real structure but keeping the same brightness values. (If it *still* looks letter-like → the look comes from CLAHE, not from real structure.) Result: scrambling removed about **half** the "letter-ness" — meaning roughly half of what we saw was just CLAHE texture.
- **Densest region** — point the pipeline at the spot where the m7 model is most confident the sheet is. Result: it rendered **solid blocks, not letters** — consistent with dense sheet predictions, not with ink.

### Test 3: The "positive control" — the decisive experiment (`positive_control.py`)

This is the gold-standard test, and it's worth understanding because it's how you should think about *any* claim.

**The idea:** if you have a detector and you don't know whether to trust it, **feed it something you already know the answer to.** A scale you don't trust? Put a known 1 kg weight on it. If it reads 1 kg, trust it; if it reads 5 kg, the scale is broken.

So we **painted Greek letters ourselves** (Π Ο Β Φ) onto a fake scroll layer — letters we *know* are there — added realistic fake fiber and noise, and ran the **identical** pipeline. Two things came out:

1. **The letters came back clearly readable.** So the pipeline *can* render letters from a 3D volume when they exist in it. The readout chain itself works.

2. **We measured the gradient of a *known* letter** and compared it to our candidate:

   | Depth (radius) | A known real letter | Our PHerc.332 candidate |
   |---|---|---|
   | r=298 | almost nothing (0.002) | **the most stuff (0.164)** |
   | r=310 | **sharp spike (0.218)** | less (0.098) |
   | r=322 | nothing (0.000) | nothing (0.000) |

   A real letter **spikes** at r=310. Our candidate does the **opposite** — it's heaviest at the *inner* edge and **fades outward**. That fade-outward pattern is the fingerprint of **papyrus fiber**, not ink on a surface.

### The verdict

- ✅ **The readout pipeline works** — it renders letters legibly from 3D data when they're actually in it.
- ❌ **The m7 data is not ink data** — we were sampling surface/sheet location predictions, not ink. No ink was ever in the data we scanned.
- ❌ **This particular candidate is almost certainly papyrus fiber** — the gradient profile matches fiber, not a thin ink layer.
- ❌ **The PHerc0009B "letters" were pareidolia** — the same pipeline invents similar shapes from empty regions.

### What we did about it

We **publicly corrected** both Discord posts, rewrote the README, and documented the m7 misidentification clearly so the community has the right information. **In this community, self-correcting fast earns more respect than a flashy claim that gets debunked by someone else.** That was the right call and it's done.

---

## Part 6 — Where Things Stand Right Now

### Settled
- The PHerc.332 candidate: **retracted** — fiber, and wrong data type (surface predictions, not ink).
- PHerc0009B letters: **retracted** — pareidolia.
- The repository and Discord posts: **honest and consistent.**
- The 35-scroll triage: **complete** — scanned all m7 surface-prediction zarrs, found nothing matching a real thin-shell ink signature (and couldn't have, since the data doesn't contain ink).

### What we actually have
The genuine contribution is the **BCE loss fix** — a bug that caused ink-detection model predictions to saturate at 0.5, fixed by using only the ink label channel with pos_weight. This raised high-confidence ink predictions from 0% to 5.93% on a Scroll 3 segment. That's documented, reproducible, and verifiable.

The positive control and pareidolia controls are also a real methodology contribution — a documented way to validate any zarr-based finding before claiming it.

---

## Part 7 — The Infrastructure (so the commands stop looking scary)

### Prajna = a shared supercomputer

**Prajna** is a **High-Performance Computing (HPC) cluster** at IIT Bombay — basically a large collection of powerful computers (with strong **GPUs**, the chips that run AI) shared by many researchers. We use it because the data is huge and AI needs GPUs. (We connect using an account belonging to a collaborator, `shiwani.mishra`.)

Two kinds of machines on it:
- **Login node** — the "front desk." You land here when you connect. It has internet, so we download data here. You don't run heavy work here.
- **Compute nodes** — the powerful workhorses. You don't use them directly; you **submit a job** and a scheduler runs it for you when a machine is free.

### SLURM = the scheduler, and the commands you'll see

**SLURM** is the software that manages the queue of jobs. The commands:
- `sbatch somejob.sh` — "please run this job when a machine is free." (sbatch = "submit batch")
- `squeue` — "show me the queue / is my job running?"
- `srun` — run something interactively (Prajna restricts this).
- A **partition** (like `l40` or `a40`) is a group of machines with a certain GPU type. We use `l40` for analysis.

So when you saw me "submit the triage job to l40," it means: *hand the 30-scroll scan to the supercomputer's queue to run on an L40-GPU machine.*

### villa — the official codebase we build on

**villa** (`github.com/ScrollPrize/villa`) is the official ScrollPrize code repository. It contains:
- The **vesuvius Python package** — the proper way to load scroll volumes and segments. Instead of writing raw S3/zarr access code ourselves, we use `from vesuvius import Volume`.
- **ink-detection training pipeline** — a more sophisticated model (ResNet3D + 3D decoder) and training setup than we built from scratch.
- **Labeled training segments** — 15 Scroll 1/2 segments with confirmed ink annotations, far more data than we had before.

villa is already cloned on Prajna at `~/scroll_prize/villa/`. Going forward, we use it as the foundation instead of reinventing things.

### Git, GitHub, and Discord

- **Git** tracks every change to our files, with a full history (so nothing is ever truly lost and we can prove what we did when).
- **GitHub** is the public website hosting our repository: `github.com/saurabh4269/vesuvius-scroll3-ink-detection`. "Pushing" = uploading our latest changes there. The Discord posts link to it, so it must always reflect the honest truth.
- **Discord** is the community chat where the Vesuvius researchers gather and where we posted (and corrected) our findings.

---

## Part 8 — How to Make Good Decisions From Here

You don't need to do the math; you need a **reliable gut for what's solid vs. shaky.** Here's the toolkit.

### The one principle
**Real data, calibrated confidence.** Our data and code were always real and public — that part was never in question. The mistake was the *confidence dial* set too high. Always separate "is the data real?" (usually yes) from "is my *interpretation* proven?" (usually not yet).

### Questions to ask about ANY finding (ours or someone else's)
1. **Does an automatic, unbiased test agree?** — or is it just someone reading a pretty enhanced picture by eye?
2. **Is there a positive control?** — has the method been shown to work on something where the answer is already known?
3. **What do the controls say?** — does the same method produce "findings" from empty or scrambled data?
4. **How much was actually searched?** — "the only one in the whole scroll" vs. "the only one in the 5% we could look at" are very different claims.
5. **Is enhancement (CLAHE) doing the heavy lifting?** — if the structure only appears after aggressive enhancement, suspect pareidolia.

### Red flags (be more skeptical when you see these)
- The words **"confirmed," "proven," "clearly"** on a single faint image.
- A **specific letter identification** without a papyrologist.
- Heavy **contrast enhancement** as the main evidence.
- **No control experiments** mentioned.
- A claim that conveniently can't be checked.

### Green flags (more trustworthy)
- A **positive control** is shown.
- The author **states what could be wrong** and what they ruled out.
- **Raw, un-enhanced** data is shown alongside the enhanced version.
- Results are **reproducible** — public data + public code + exact coordinates.

### "Candidate" vs. "Discovery" — use the right word
- A **candidate** = "here's something interesting, not yet proven, please help check." Low risk, honest, community-friendly.
- A **discovery** = "this is real, I've ruled out the alternatives." High bar. Only claim this after the controls pass.
- Our whole correction was essentially **downgrading a "discovery" back to a "candidate."** That's normal science, not failure.

### Realistic odds (so expectations are calibrated)
- **First Letters / First Title ($60k each):** very hard. Requires 10 clear letters in a small area of a specific scroll. We have **zero** proven letters. This is a long shot and shouldn't drive day-to-day decisions.
- **June Progress Prize ($20k, tool/method):** **the realistic target.** A validated, open-source tool that scans all scrolls for letter candidates — with honest controls built in — is exactly the kind of contribution this prize exists for. Even a clean *negative* result plus a good tool counts.

The honest, valuable position to aim for: *"I built and openly validated a tool, applied it systematically to every scroll, reported what it found with full controls, and shared everything."* That's good science and a credible submission — regardless of whether a dramatic letter ever turns up.

---

## Glossary (every term, one line each)

- **Vesuvius Challenge** — the competition to read the burned Herculaneum scrolls via software.
- **Herculaneum / PHerc.** — the Roman town and the naming prefix for its scrolls (e.g. PHerc.332).
- **Scroll 2 / Scroll 3** — the two scrolls with active $60k prizes (Scroll 3 = PHerc.332).
- **Carbonized** — turned to charcoal by heat; why ink and paper look alike in scans.
- **CT scan** — X-ray imaging that produces a 3D density model.
- **Voxel** — a 3D pixel; one cube of the scan with a brightness value.
- **Resolution (µm/px)** — physical size of one voxel; smaller = finer detail = bigger files.
- **Level (0,1,2,3…)** — zoom versions of the same data; 0 = finest, higher = coarser/smaller.
- **zarr** — the file format storing the giant 3D array as many small chunk files.
- **Chunk** — one small cube (192³) of a zarr array, loaded on demand.
- **S3 / bucket** — Amazon cloud storage; ours is public (`--no-sign-request` = no login needed).
- **Segmentation** — finding and flattening the wound-up sheet inside the 3D scan.
- **Volume Cartographer / ThaumatoAnakalyptor** — tools that do segmentation.
- **Ink detection** — using AI to guess where ink is, since it's nearly invisible in raw scans.
- **m7 / m7_nnUNet** — the organizers' papyrus **surface/sheet localization** model; predicts where the sheet is in 3D, used as input to segmentation tools. NOT an ink detector.
- **Ink prediction** — the AI's per-point "how likely is this ink" output (a zarr volume).
- **Cylindrical unrolling** — our shortcut: treat the scroll as a cylinder and sample in circles.
- **Radius (r)** — how far out from the center we sample; one r = one layer.
- **Angle** — position around the circle (0–360°, split into 1800 steps).
- **Unrolled strip** — the flat rectangle produced by sampling a full circle at one radius.
- **CLAHE** — aggressive local contrast enhancement; powerful but can manufacture fake patterns.
- **Pareidolia** — the brain seeing structure (faces, letters) in random noise.
- **Gradient test** — checking that signal spikes at one depth (ink) vs. fading (fiber).
- **Positive control** — testing a method on a known answer to see if it's trustworthy.
- **Negative / shuffle / empty control** — testing on nothing/scrambled data to expose false positives.
- **Candidate vs. discovery** — "interesting, unproven" vs. "proven, alternatives ruled out."
- **Triage** — quickly scanning everything to rank where to look closely.
- **HPC / cluster** — a shared supercomputer (Prajna = IIT Bombay's).
- **GPU** — the chip that runs AI fast; the reason we need the cluster.
- **Login node / compute node** — the front desk (has internet) vs. the workhorses (run jobs).
- **SLURM** — the job scheduler; `sbatch` submits, `squeue` shows the queue.
- **Partition (l40, a40)** — a group of cluster machines with a given GPU type.
- **Git / GitHub** — change tracking / the public website hosting our code.
- **Push** — upload latest changes to GitHub.
- **Discord** — the community chat where findings (and our corrections) are posted.
- **villa** — the official ScrollPrize codebase (`github.com/ScrollPrize/villa`); contains the vesuvius Python package, ink-detection training pipeline, and labeled training segments. Already cloned on Prajna.
- **vesuvius (package)** — the Python library inside villa for loading scroll data; use `from vesuvius import Volume` instead of raw zarr/s3fs.
- **First Letters / First Title / Progress Prize** — the three prizes; the Progress Prize (tool) is our realistic target.

---

*If any single section here is still fuzzy, tell me which one and I'll go deeper on just that — with more analogies, or by walking through the actual images and numbers from our own files.*

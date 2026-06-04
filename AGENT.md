# AGENT.md — Start Here (full-context handoff for any coding agent)

If you are a coding agent picking this project up cold, **read this file top to bottom first.** It gives you the whole picture and points you to the deeper docs. Last updated 2026-06-04.

---

## 0. TL;DR — the 60-second version

- **Project:** Vesuvius Challenge — read carbonized Herculaneum scrolls from X-ray CT using ML. We work on **ink detection** for **Scroll 3 (PHerc.332)**, plus tooling.
- **Honest status:** We have **no confirmed letters.** Two earlier "letter" Discord posts were **retracted** (one was papyrus fiber, one was pareidolia). Our real contribution is a **training-loss bug fix** + a **hallucination-control methodology**, aimed at a **Progress Prize** ($2.5k–$20k), not the $60k letter prizes.
- **The thing that matters most now:** the field has moved to **generalist model + ~2 µm scans + curriculum learning + iterative labeling**, trained on **Scroll 139** data (already in villa). Our ESRF-only 2D model is the approach the field moved past. See `docs/news_and_status.md`.
- **Before you claim anything or open a PR:** read §6 (standing rules). We got burned by overclaiming; don't repeat it.

---

## 1. Document map — where the full context lives

Read in this order depending on what you need. **Don't duplicate these — update them.**

| Doc | Use it for |
|-----|-----------|
| **`AGENT.md`** (this file) | Orientation + handoff. Start here. |
| **`GUIDE.md`** | Plain-language walkthrough of the whole project (no background assumed) + a decision toolkit for judging findings. |
| **`knowledge_base.md`** | The working record: data types, villa, Prajna infra, env/error fixes, the model + BCE fix, what worked/didn't, **§8b failure-mode lessons**, open questions, contribution options. **Primary source of truth.** |
| **`docs/news_and_status.md`** | What's happening *now* (scraped scrollprize.org + Substack) and what it means for our next move. |
| **`docs/vesuvius_challenge_reference.md`** | The full evergreen research landscape (history, every prize + winners, all community tools, people, formats, citations — 1088 lines). |
| **`PROGRESS_PRIZE_SUBMISSION.md`** | Our Progress Prize write-up. |
| `docs/PRAJNA_HPC.md`, `docs/PRAJNA_RUNBOOK.md` | Prajna cluster reference + runbook. **Do not edit these.** |

---

## 2. What this project is

The Vesuvius Challenge reads ~2,000-year-old carbonized scrolls via CT + ML. Pipeline: **scan → segment/unwrap the papyrus sheet → flatten → detect ink (ML) → papyrologist reads.** We focus on **ink detection** for **Scroll 3 (PHerc.332)** and on reusable tooling.

Open prizes (verified, see `knowledge_base.md` §1): **First Letters / First Title — 7 × $60,000** (Scrolls 2-3), **Unwrapping at Scale — $200,000**, **Monthly Progress Prizes — $1k–$20k** (our realistic target).

---

## 3. Current honest state (2026-06-04)

### Our model
- **Best:** Segformer-B1 (MiniUNETR, 45.6M params), `pos_weight=10`, 50 epochs, trained on ESRF fragments 500P2+343P.
- Val loss **1.631**; on Scroll 3 segment `20240618142020`, **>0.9 confidence = 5.93%** (vs **0.00%** pre-fix).
- Checkpoint (Prajna): `~/scroll_prize/vesuvius_first_title_prize/checkpoints/ft_esrf_b1_20260603_045037/best_epoch_046_val_loss_1.6306.pt`
- Config: `vesuvius_first_title_prize/configs/ft_esrf_b1.py`
- **Caveat:** on Scroll 3 it produces a uniform 32px-period dot grid = papyrus **fiber texture, not ink.** Not validated as a true ink detector. (This is the documented field-wide generalization wall — see §5.)

### The BCE bug fix (our real modeling contribution)
ESRF labels are `(B, 2, H, W)`: channel 0 = ink mask, channel 1 = all-ones validity mask. Training BCE against **both** gives contradictory gradients → predictions saturate at 0.5.
```python
# WRONG: loss = BCE(logits, y.float())
# RIGHT:
loss = F.binary_cross_entropy_with_logits(
    logits, y[:, 0, :, :].float(),
    pos_weight=torch.tensor([10.0], device=device))   # ~9% ink ratio
```

### What was retracted — do NOT resurrect as fact
> ⚠️ Earlier versions of this file claimed a "CONFIRMED letter candidate" in PHerc.332. **That is false and retracted.** If you see that framing anywhere, it is stale.
- **PHerc.332 "letter candidate"** — RETRACTED. Our own `r310_fullsearch.py` scored `validated: 0`; the gradient profile is the fiber signature (peaks r=298, decays outward), not ink (which spikes sharply at r=310). And m7 is the wrong data type (below).
- **PHerc0009B "Π/Ο letters"** — RETRACTED. Pareidolia (same CLAHE pipeline invents marks on empty regions); likely an ink-render not raw CT; not a prize-eligible scroll.

### Terminology correction (was wrong project-wide for weeks)
- **m7 zarrs** (`representations/predictions/surfaces/...surface-m7...`) are **papyrus surface/sheet localization** predictions — input to segmentation tools. They are **NOT ink predictions.** Never call them ink.

---

## 4. Validated tools & method (keep using these)
- `scripts/positive_control.py` — paints known Greek letters on a synthetic shell, runs the full pipeline; proves a readout chain before trusting it.
- `scripts/salvage_test.py` — empty/shuffle pareidolia controls.
- BCE channel-0 fix + `pos_weight=10`; manual training loop (Lightning hangs on a40); CLAHE (clip 2.0, tile 8×8); overlapping-patch inference; SLURM dependency chaining; ControlMaster+pyotp (`scripts/prajna_lib.py`); `zarr.open_group(fsspec.get_mapper(url))`.
- Script status (active vs retracted-analysis reference): `knowledge_base.md` §9.

---

## 5. Strategic direction (what to actually do next)

From `docs/news_and_status.md` — the team's reported winning recipe, which reshapes our plan:

1. **Ink that generalizes = generalist model + ~2 µm data + curriculum learning + iterative labeling.** Ink appeared in 4+ new scrolls (9B/814/841/139) this way. Our ESRF-only, single-resolution, crack/metal-signal 2D model is the approach the field moved past.
2. **Scroll 139 ("0139") is the key training set — already in villa** (`villa/ink-detection/metadata.json` 2µm segments have `base_path: "0139"`). An autoresearch agent ~2×'d Scroll 4 perf training only on 139.
3. **3D ink detection on unflattened volumes now matches 2.5D** — villa's `train_resnet3d.py` is exactly this (ResNet3D + 3D decoder, Lightning → **l40** partition).
4. **Higher resolution unlocks ink** (Scroll 4 letters appeared only at ~2.4 µm). Scroll 3 at 7.9 µm may be under-resolved.

**Recommended next move:** stop polishing the ESRF-only 2D B1; train on villa's Scroll-139 2µm data + the 45 labeled Scroll 1/2 segments (`villa/ink-detection/all_labels/`) via villa's 3D `train_resnet3d.py`, with curriculum + iterative labeling. First, validate the current B1 on a villa labeled segment to confirm it detects ink at all (`scripts/validate_b1_villa.py`).

Other contribution options + the prize wishlist: `knowledge_base.md` §11 Priority 3.

---

## 6. Standing rules — never skip (the user is firm on these)

1. **No claim may exceed what your own code outputs.** If a quantitative filter rejects a candidate, it's a "candidate," not "confirmed." (`r310_fullsearch.py` said `validated: 0` on the thing the old AGENT.md called confirmed.)
2. **Run positive + empty/shuffle controls before claiming letters.** CLAHE manufactures letter-like marks from noise.
3. **Use "candidate," not "confirmed."** Letter IDs need a papyrologist.
4. **Verify terminology against villa's code/docs before any public output** (e.g. m7 = surface, not ink).
5. **Never open a PR (especially to `ScrollPrize/villa`) without showing the full diff and getting explicit sign-off from the user.** Wrong terminology in a PR gets flagged by organizers and is worse than not posting.
6. **If you've posted something wrong:** self-correct fast in the same thread, don't delete, don't double down, check the next finding first.
7. **Do not edit** `docs/PRAJNA_HPC.md`, `docs/PRAJNA_RUNBOOK.md`, or the Prajna-specific content without being asked.

Full detail + the overclaim→honest-version table: `knowledge_base.md` §8b and `GUIDE.md` Part 8.

---

## 7. Environment & access (Prajna HPC)

- **Cluster:** IIT Bombay Prajna. User `shiwani.mishra`, group `medal`. Project root `~/scroll_prize/`. Conda env `scroll`.
- **Credentials:** in `.env` (PRAJNA_USER / PRAJNA_PASSWORD / PRAJNA_TOTP_SECRET). **Never print or commit these.**
- **Connect from automation** via `scripts/prajna_lib.py` (copy of the prajna-hpc skill lib):
  ```python
  import sys; sys.path.insert(0, "scripts")
  from prajna_lib import connect_prajna, run_paramiko as run, submit_job
  client = connect_prajna()           # reads .env, handles TOTP
  out, err, rc = run(client, "squeue --me")
  ```
- **Partitions:** `l40` (inference/analysis + villa Lightning training; default), `a40` (our manual-loop training; **Lightning hangs here**). `--qos` must equal `--partition`.
- **Compute nodes have NO internet.** Pre-download HF models on the login node and export `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled` in job scripts.
- **Job-script must-haves:** `source ~/miniconda3/etc/profile.d/conda.sh` before `conda activate`; `set -eo pipefail` (not `-u`); `mkdir -p ~/logs`; `python -u`; full path `/opt/slurm/bin/sbatch`.
- Full cluster ref: `docs/PRAJNA_HPC.md`, `docs/PRAJNA_RUNBOOK.md`. Versions/SLURM/error specifics: `knowledge_base.md` §4–6.

**Key Prajna paths:**
```
~/scroll_prize/villa/                              # official ScrollPrize codebase (vesuvius pkg, ink-detection, VC3D, Thaumato)
~/scroll_prize/villa/ink-detection/all_labels/     # 45 labeled Scroll 1/2 segments
~/scroll_prize/villa/ink-detection/train_resnet3d.py  # 3D ink pipeline (run on l40)
~/scroll_prize/villa/ink-detection/metadata.json   # villa training config; 2µm segments base_path "0139" (Scroll 139)
~/scroll_prize/vesuvius_first_title_prize/         # our training code (phoenix pkg) + checkpoints
~/scroll_prize/data/esrf/                          # ESRF training fragments (500P2 + 343P)
```

---

## 8. Immediate pending work (pick up here)

1. **Two SLURM jobs failed and need fixing + resubmit** (root causes in `knowledge_base.md` §6):
   - `scripts/validate_b1_villa.sh` (was job 127870) — failed: tried to download `nvidia/mit-b1` on a compute node. **Fix:** export `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1`, verify the backbone is in `~/.cache/huggingface`, resubmit.
   - `scripts/infer_villa_pretrained.sh` (was job 127871) — failed: segment not in `metadata.json`. **Fix:** add `--layer_range 1:63`.
   These answer the open question "does our B1 detect ink or just fiber?" — validate on a villa labeled Scroll 1/2 segment where ground truth exists.
2. **Then** decide on the strategic pivot in §5 (villa 3D pipeline + Scroll 139 + curriculum).
3. Progress Prize submission (`PROGRESS_PRIZE_SUBMISSION.md`) — monthly deadline, form `https://forms.gle/LrpQmSAqdwGpTczLA`.

---

## 9. Repo + git
- GitHub: `git@github.com:saurabh4269/vesuvius-scroll3-ink-detection.git`. Branch `master`.
- Commit messages end with the `Co-Authored-By` trailer. Push when the user asks; **show diffs before any PR** (§6 rule 5).
- Scripts in `scripts/`; active vs retracted-analysis status in `knowledge_base.md` §9.
- Memory (cross-session): `~/.claude/projects/.../memory/` — `project_scroll_prajna.md` (setup) and `feedback_claims_and_prs.md` (the claims/PR discipline).

# Vesuvius Challenge — News & Status (what's happening, what to do)

> **Point-in-time snapshot, compiled 2026-06-04** from `scrollprize.org` and `scrollprize.substack.com`.
> This is the time-sensitive "what's happening now" view. For the evergreen landscape (history, tools, people, formats) see [`vesuvius_challenge_reference.md`](vesuvius_challenge_reference.md); for our project takeaways see [`../knowledge_base.md`](../knowledge_base.md) §1b. Re-scrape periodically — dates/prizes move.

---

## Timeline — recent developments (newest first)

| Date | Headline | What actually happened |
|------|----------|------------------------|
| **2026-03** | "We are cooking" | **Autoresearch agents** running on ink detection — one **~doubled validation performance on Scroll 4 while training only on Scroll 139 data** (cross-scroll transfer works). **DINO-based volumetric foundation model** training on the large dataset. New **3D ink model on PHerc.172** runs on *unflattened* volumes and matches 2.5D SOTA. **VC3D** gained remote streaming, a **Neural Tracer** widget (automated mesh), and a **"Lasagna"** tool (~3× faster annotation). **Kaggle Surface Detection ($200K) concluded** — winning solutions were large ensembles of the baseline **ResEncUNet** trained 4,000+ epochs. **AWS Open Data sponsorship** (free hosting, 2 yrs); redesigned Data Browser. |
| **2026-01** | "~70% of PHerc. 172 is now digitally unwrapped" | Automatic unwrapping starting to work at scale — a large fraction of Scroll 5 unwrapped with much less manual tracing. Direct progress toward the $200K Unwrapping-at-Scale prize. |
| **2025-12** | "Finally — letters in Scroll 4!" | A **higher-resolution scan (~2.4 µm)** revealed ink in PHerc. 1667 where the standard ~7.9 µm scan showed nothing. Confirms the master-plan hypothesis: when ink isn't crack/metal-rich, **more resolution unlocks it**. |
| **2025-11** | "$100K Kaggle Surface Detection" | Surface-detection competition launched (later raised to $200K). Goal: topologically accurate papyrus-surface detection in 3D CT. |
| **2025-10** | "Multiple scrolls now show Greek letters" | **PHerc. 9B, 814, 841, 139** now show Greek letters — via a **generalist ink model trained on fragment surfaces + ~2 µm CT + curriculum learning** (visible fragment surfaces → hidden layers → closed scrolls). **Iterative labeling** improved predictions. $200K Unwrapping-at-Scale prize introduced. |
| **2025-09** | "Column of text from PHerc. 172" | A full column of text extracted from Scroll 5. |
| **2025-08** | "Unveiling the Mystery of Compressed Regions" | Sub-micron scanning explains why some compressed papyrus layers look blurred. |
| **2025-05** | "$60,000 First Title Prize Awarded" | Title of PHerc. 172 read: *On Vices* by **Philodemus** — first title from a still-rolled scroll. |

---

## Current open prizes — "what to do" (from scrollprize.org landing)

- **Unwrapping at Scale — $200,000** — automate virtual unwrapping of ≥70% of two different scrolls.
- **First Letters / First Title — 7 × $60,000** — 10 legible letters in a 4 cm² area / a readable title image, in **Scrolls 2-3**.
- **Monthly Progress Prizes — $350,000 pool** — $1k (Papyrus) → $20k (Gold Aureus), judged the last day of each month. Favors: released early, *actually used* by the community, well documented.
- **Three focus areas:** representation (segmentation/annotation), geometric reconstruction (unwrapping), ink detection.

---

## What this means for OUR project (the actionable part)

The news points clearly at what's working and what isn't — and our crack/metal-trained 2D B1 is on the wrong side of it:

1. **The winning ink recipe is now visible: generalist model + ~2 µm data + curriculum learning + iterative labeling.** Ink appeared in 4+ new scrolls (9B/814/841/139) via a generalist trained on *fragment surfaces*, stepped from visible → hidden → closed. Our B1 (one resolution, crack/metal signal, no curriculum) is exactly the approach the field has moved past.
2. **Scroll 139 ("0139") is the key dataset — and it's already in villa.** villa's `metadata.json` 2µm segments have `base_path: "0139"`. An autoresearch agent doubled Scroll 4 performance training *only* on 139 → cross-scroll transfer is achievable with this data. **This is the data to train on, not just ESRF 500P2/343P.**
3. **3D ink detection (unflattened volumes) now matches 2.5D** — and villa's `train_resnet3d.py` is exactly a 3D pipeline. This validates pursuing villa's ResNet3D over our 2D Segformer-B1.
4. **The levers are curriculum learning + iterative labeling, not bigger backbones.** Matches our own finding that B3→B1 and more epochs didn't help (data/strategy bound, not capacity bound).
5. **Higher resolution unlocks ink** (Scroll 4 at 2.4 µm). Scroll 3 work at 7.9 µm may simply be under-resolved for its ink — manage expectations accordingly.
6. **Segmentation baseline, if we go there:** big ResEncUNet ensembles, long training (Kaggle Surface Detection winners). VC3D now has remote streaming + Neural Tracer + Lasagna.
7. **Tooling that gets used wins Progress Prizes** — Paul Geiger won a Papyrus ($1k) for a 3D viewer that got 70+ upvotes from competitors. Reinforces our "hallucination-safe, reusable tooling" angle over one-off claims.

**Net recommendation:** the highest-leverage move is to stop polishing the ESRF-only 2D B1 and instead train on villa's Scroll-139 2 µm data (and the 45 labeled Scroll 1/2 segments) with a curriculum + iterative-labeling approach — ideally via villa's 3D `train_resnet3d.py`. That's aligned with exactly what the team reports is working.

---

## Live-site scrape — currency notes (2026-06-04)

Direct scrape of `scrollprize.org` pages caught changes vs the villa doc clone. **These have been propagated into our docs.**

- **⚠️ Progress Prize submission form CHANGED** → now **`https://forms.gle/Sy6mW5cfJS2U7E9F7`** (was `…/LrpQmSAqdwGpTczLA`). Verified verbatim from `/prizes` twice. Updated across AGENT.md, knowledge_base.md, PROGRESS_PRIZE_SUBMISSION.md, reference doc. **Submitting to the old form = missed submission.**
- **Next Progress Prize deadline: June 30, 2026, 11:59pm Pacific** (~26 days out as of this snapshot). Monthly/rolling, but this is the next concrete cutoff for our submission.
- **Milestone (First Letters/Title) form** (unchanged): `https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform`. Confirmed **First Letters AND First Title = Scrolls 2-3**.
- **Discord invite** is now **`https://discord.com/invite/uTfNwwecCQ`** (site nav still shows the old `discord.gg/V4fJhvtaQn`). Updated in the reference doc.
- **"Unwrapping at Scale $200k" no longer appears on the live `/prizes` page.** The $200k now reads as the **concluded** Kaggle Surface Detection competition (winners announced March 2026). Treat a *standing* Unwrapping-at-Scale prize as **unconfirmed** — verify before relying on it. (The landing page and older docs still mention it.)
- **Jobs:** no open roles right now; speculative applications to `jobs@scrollprize.org`.

### Onboarding / how-to quickstart (from get-started, segmentation & ink tutorials)

For any agent that needs to actually run the pipeline:

- **Look inside a scroll:** browser `https://dl.ash2txt.org/view/Scroll1`, or `pip install vesuvius` then `vesuvius.Volume('Scroll1')` (Python) / `vesuvius-c` (C).
- **Ink detection starting points:** the high-level tutorial (`/tutorial5`) → hands-on Kaggle notebook (`kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial`) and the current Colab `ScrollPrize/vesuvius/notebooks/example2_ink_detection.ipynb`. Train on fragments (IR-photo ground truth) → domain-adapt to scroll surface volumes.
- **Segmentation / VC3D:** `docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge`. Data must be **OME-Zarr, uint8, with a `meta.json` containing `"format":"zarr"`** (check `.zarray` dtype: `|u1`=uint8, `|u2`=uint16). Gotcha: raise the open-file limit first — `ulimit -Sn 750000`.
- **Four contribution tracks** (get-started): Surface Detection (Kaggle), Segmentation (→ `/unwrapping`, `/segmentation`), Ink Detection (→ `/tutorial5`; the 7×$60k prizes), Open-Source Tools (→ `/community_projects` + the Progress wishlist).

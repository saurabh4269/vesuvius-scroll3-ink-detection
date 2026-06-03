# Prajna (AI-ML) HPC — Complete Agent Reference

> **Authoritative source:** IIT Bombay-provided PDF user manual + live SLURM verification on 2026-04-29.
> The HPC website (hpcverse.iitb.ac.in) has known errors (wrong partition names, stale limits) and
> should NOT be trusted without cross-checking via live `sinfo`/`scontrol` commands.
> This file is self-sufficient — an agent given this file plus a populated `.env` file can work
> seamlessly on the cluster. Links in §14 allow the agent to re-verify and update stale sections.
> See `PRAJNA_RUNBOOK.md §0` for ControlMaster setup (required once per machine) and `.env` loading.

---

## Credentials — `.env` file (required)

Only 3 values live in `.env`. Everything else is derived or fetched from the server.

```
.env          ← actual credentials — KEEP SECRET, never commit
.env.example  ← template — safe to commit, fill in and save as .env
```

| Variable | Source | Notes |
|----------|--------|-------|
| `PRAJNA_USER` | User fills in `.env` | Required |
| `PRAJNA_PASSWORD` | User fills in `.env` | Required |
| `PRAJNA_TOTP_SECRET` | User fills in `.env` | Required — base32 key from GA setup |
| `PRAJNA_EMAIL` | User fills in `.env` | Optional — for SLURM job notifications |
| `PRAJNA_GROUP` | Auto-filled by `prajna_setup()` | `id -gn` on server |
| `PRAJNA_UID` | Auto-filled by `prajna_setup()` | `id -u` on server |
| `PRAJNA_GID` | Auto-filled by `prajna_setup()` | `id -g` on server |
| `PRAJNA_SCRATCH_1..5` | Auto-filled by `prajna_setup()` | From `~/.google_authenticator` |
| home dir | **Derived** — never stored | `/home/$PRAJNA_GROUP/$PRAJNA_USER` |

`prajna_setup()` runs once on first connect and writes the auto-filled values back into `.env`.
On subsequent runs it skips any field already set. Full code in `PRAJNA_RUNBOOK.md §0.2`.

---

## 1. Access

```
Cluster:   Prajna (AI-ML) — IIT Bombay High Performance & AI/ML Computing Facility
Host:      prajna.iitb.ac.in   (alias: login1.prajna.iitb.ac.in)
User:      $PRAJNA_USER          (from .env)
Home:      $HOME                 (set by shell; typically /home/<group>/<user>)
Scratch:   /lustre-scratch/$PRAJNA_USER/   ← real user scratch, 820 TB Lustre
SSH:       ssh $PRAJNA_USER@prajna.iitb.ac.in
SCP in:    scp -r ./local/ $PRAJNA_USER@prajna.iitb.ac.in:~/
SCP out:   scp -r $PRAJNA_USER@prajna.iitb.ac.in:~/results ./
SLURM bin: /opt/slurm/bin/   (in PATH on login node; add explicitly in job scripts)
Group/UID: run 'id' on Prajna after login
```

### 1.1 Two-Factor Authentication (enabled 2026-04-28)

Prajna now requires **TOTP + password** on every SSH login. Simple password auth no longer works.

**Interactive login sequence:**
```
($PRAJNA_USER@prajna.iitb.ac.in) Verification code:   ← enter 6-digit TOTP from app
($PRAJNA_USER@prajna.iitb.ac.in) Password:            ← then enter password
```

**TOTP details:**
```
Secret key:  $PRAJNA_TOTP_SECRET   (from .env — base32 key printed during GA setup)
Algorithm:   TOTP (time-based, 30-second window)
App:         Google Authenticator (or any TOTP app — Authy, 1Password, etc.)
Config file: ~/.google_authenticator  (on Prajna — first line is the secret)
Emergency scratch codes: cat ~/.google_authenticator on Prajna to see remaining ones.
                         The server removes each code when it's used — no manual tracking needed.
```

**TOTP settings (configured during setup):**
- `DISALLOW_REUSE` — same 6-digit code cannot be used twice in the same 30s window
- Extended window (17 codes, ±4 minutes) — tolerates local clock skew up to 4 minutes
- Rate limiting — max 3 failed attempts per 30s

**For agents / automated scripts** — generate TOTP in Python with `pyotp`:
```python
import pyotp, os
code = pyotp.TOTP(os.environ["PRAJNA_TOTP_SECRET"]).now()   # fresh 6-digit code
```
Full paramiko connection pattern with TOTP is in `PRAJNA_RUNBOOK.md §2`.

**SSH key auth:** Does NOT bypass TOTP (tested 2026-04-28). Server enforces TOTP for all
auth methods including public key.

**Recommended for agents/automation — SSH ControlMaster** (authenticate once, persists until reboot):
```bash
# In shell (substitute from .env):
source .env
ssh ${PRAJNA_USER}@prajna.iitb.ac.in "echo ready"   # TOTP + password once
# All subsequent ssh/scp/rsync reuse the socket — zero prompts
ssh -O check prajna       # verify master is alive (requires Host alias in ~/.ssh/config)
```
Full ControlMaster setup is in `PRAJNA_RUNBOOK.md §0`.

**Fallback for headless automation** — pyotp + paramiko keyboard-interactive.
Full pattern in `PRAJNA_RUNBOOK.md §2`.

---

## 2. Cardinal Rules (Enforced — Read Before Doing Anything)

| Rule | What happens if violated |
|------|--------------------------|
| **NEVER run computation on the login node** | Admin monitors login nodes. **1st offence: 1-month job-submission ban.** Repeat offence: PI must meet Computing Centre. Login node is ONLY for: editing files, writing scripts, scp/rsync, sbatch, squeue, conda env creation (lightweight). |
| **All computation must go through SLURM** | `sbatch` for batch jobs; `srun --pty bash` for interactive. Never run `python train.py` directly in the SSH session. |
| **`/lustre-scratch/` purge** | Files not accessed for **3 months are permanently deleted without warning.** Always `cp` final results to `/home/` after a job. |
| **`/scratch/` is NOT user space** | `/scratch` (250 GB, mount `slurm:/scratch`) is a SLURM system directory containing admin files. Writing user data there may corrupt cluster operations. Use `/lustre-scratch/` instead. |
| **Own jobs only** | `scancel` and `scontrol` only work on your own jobs. |
| **Password** | Change the temporary password immediately on first login with `passwd`. Required by institute security policy. |
| **No `module` command** | This cluster does not use environment modules. Use Spack or conda instead. |

---

## 3. System Information (Live-Verified 2026-04-29)

> PDF manual states Rocky 9 and SLURM 23.11.10 — both are **outdated**.

| Item | Value |
|------|-------|
| OS | Rocky Linux **8.10** (Green Obsidian) |
| SLURM | **24.05.6** |
| Interconnect | InfiniBand |
| Parallel filesystem | Lustre |
| Login node CPUs | 32 vCPUs, 94 GB RAM, no GPU |
| SLURM binaries | `/opt/slurm/bin/` |
| Conda | `~/miniconda3/` (v25.11.0, **already initialised in `~/.bashrc`**) |
| Spack | `/lustre-flash/apps/spack/share/spack/setup-env.sh` |

### Hardware Nodes (from PDF manual + live node detail)

| Type | Count in SLURM | Node names | CPUs/node | RAM/node | GPUs/node | GPU VRAM |
|------|---------------|-----------|-----------|----------|-----------|----------|
| DGX A100 (`dgx`) | 9 | cn11-dgx – cn19-dgx | 256 vCPU (128 cores × 2 HT) | ~2 TB | 8× A100 | 80 GB HBM2e |
| Exatron A40 (`a40`) | 19 live (20 physical) | cn21-a40 – cn39-a40 | 64 vCPU (32 cores × 2 HT) | ~503 GB | 4× A40 | 48 GB |
| Exatron L40S (`l40`) | 7 | cn40-l40 – cn46-l40 | 32 vCPU (32 cores × 1 HT) | ~503 GB | 8× L40S ADA | 48 GB |
| Tyron L4 | 10 (hardware only) | — | 24 cores | 128 GB | 2× L4 | 24 GB |

Special-purpose node assignments (live-verified 2026-04-29):
- `dgx-mpi`: cn11-dgx – cn13-dgx (3-node subset of dgx)
- `interactive`: cn11-dgx, cn22-a40, cn41-l40
- `debug`: cn39-a40 (single A40 node)

> L4 nodes are hardware-present but **have no SLURM partition configured** — not submittable.

### Storage Filesystems (Live-Verified)

| Mount | Size | Used | Purpose |
|-------|------|------|---------|
| `/home` | 1.3 PB | 267 TB (22%) | Home dirs — no quota set, persistent |
| `/lustre-scratch` | 820 TB | ~6 MB (empty) | **User scratch** — fast Lustre, 3-month purge |
| `/lustre-flash` | 399 TB | 1.9 TB | System software (Spack) — do not write here |
| `/scratch` | 250 GB | 39 GB | SLURM system dir — **NOT for user data** |

> First scratch use: `mkdir -p /lustre-scratch/$PRAJNA_USER/`

---

## 4. SLURM Partitions (Live-Verified 2026-04-29)

> ⚠️ Website uses wrong names (`gpu_a100`, `gpu_a40`, `gpu_l40s`). Always use the names below.

### Partition Table

| Partition | Default | Nodes | GPUs/node | GPU type | Max walltime | Nodes currently |
|-----------|---------|-------|-----------|----------|-------------|-----------------|
| `l40` | **YES** | 7 | 8× L40S ADA | L40S | **2-00:00:00** | 1 down, 5 mix, 1 alloc |
| `a40` | no | 19 | 4× A40 | A40 | **4-00:00:00** | 3 down, 14 mix, 2 alloc |
| `dgx` | no | 9 | 8× A100 | A100 | **6-00:00:00** | 1 drain, 8 mix |
| `dgx-mpi` | no | 3 (subset of dgx) | 8× A100 | A100 | **6-00:00:00** | 3 mix |
| `interactive` | no | 3 (1×dgx, 1×a40, 1×l40) | mixed | mixed | **4:00:00** | all mix |
| `debug` | no | 1 (cn39-a40) | 4× A40 | A40 | **0:30:00** | mix |

### Per-Job Limits from QOS (Live `sacctmgr` — most authoritative)

| Partition (`--qos`) | Max running jobs | Max total (run+queue) | **Max GPUs per job** | Max walltime |
|---------------------|-----------------|----------------------|---------------------|-------------|
| `a40` | **3** | 6 | **2 GPUs** | 4 days |
| `l40` | **4** | 5 | **4 GPUs** | 2 days |
| `dgx` | **4** | 5 | **4 GPUs** | 6 days |
| `dgx-mpi` | **1** | 2 | **8 GPUs** | 6 days |
| `interactive` | **2** | 2 | **8 GPUs** | 4 hours |
| `debug` | no limit | no limit | no limit | 30 minutes |

### Account-Level Limits (your group account)

| Limit | Value |
|-------|-------|
| Max total jobs running (all partitions) | **12** |
| Max total submitted (running + queued) | **20** |

### QOS Rules
- `--qos=<name>` must **exactly match** `--partition=<name>` in every job script
- `--account=<your-group>` should be specified (run `id -gn` on Prajna to find it)
- Max GPU cap is enforced by the QOS `MaxTRES` field — exceeding it causes immediate job rejection

---

## 5. Full Potential — What You Can Do

| Goal | How |
|------|-----|
| Run 3 parallel GPU jobs on A40 | Submit 3 `sbatch` jobs to `a40` partition (max 2 GPUs each) |
| Run 4 parallel jobs on L40S | Submit 4 jobs to `l40` partition (max 4 GPUs each) |
| Run 4 parallel jobs on A100 | Submit 4 jobs to `dgx` partition (max 4 GPUs each) |
| Use a full DGX node (8 GPUs) | Use `dgx-mpi` partition (max 8 GPUs, but only 1 job at a time) |
| Test a script quickly | Submit to `debug` partition (30 min, no GPU limit, instant queue) |
| Interactive GPU debugging | `srun --partition=interactive --qos=interactive --gres=gpu:1 --pty bash` |
| Run 10 experiments in parallel | Job arrays: `--array=0-9` with `$SLURM_ARRAY_TASK_ID` as experiment index |
| Pipeline (job B after job A) | `sbatch --dependency=afterok:<jobA-id> jobB.sh` |
| Check estimated start time | `sbatch --test-only job.sh` |
| Install Python packages | `conda activate myenv && pip install ...` (on login node, no compute) |
| Use system libraries (CUDA, MPI) | `source /lustre-flash/apps/spack/share/spack/setup-env.sh && spack load cuda@12` |

---

## 6. Software Environment

### Conda (use for Python — already configured)

Conda is already initialised in `~/.bashrc` via `conda initialize` block — no manual sourcing needed on login.

```bash
# Conda is active automatically. Just do:
conda activate myenv

# Create a new env (safe on login node):
conda create -n myenv python=3.10 -y
conda activate myenv
pip install "jax[cuda12]" flax optax scikit-learn pandas

# List existing envs:
conda env list
```

> In SLURM job scripts (non-interactive shells), source conda explicitly:
> `source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv`

### Spack (use for system libraries — CUDA, MPI, compilers)

```bash
# Initialise (required each session or add to ~/.bashrc)
source /lustre-flash/apps/spack/share/spack/setup-env.sh

spack find                      # all installed packages
spack find --loaded             # currently active
spack load cuda@12              # load CUDA 12
spack load openmpi              # load MPI
spack list                      # all installable packages
spack compilers                 # gcc 8.5/12.4/13.3/14.2, nvhpc 23.11/24.11, oneapi 2024.2.1/2025.0.1

# Install a package
spack install <pkg>@<ver> +<variant> %<compiler>
```

---

## 7. SLURM Command Reference

```bash
# Load env vars first (in shell):
set -a; source .env; set +a

# ── Submitting ─────────────────────────────────────────────────────────────
sbatch job.sh                                         # submit batch job
sbatch --test-only job.sh                             # dry-run: validate + estimated start (no submission)
sbatch --array=0-9 job.sh                             # job array; $SLURM_ARRAY_TASK_ID = 0..9 inside script
sbatch --array=0-9%4 job.sh                           # array, max 4 tasks running simultaneously
sbatch --dependency=afterok:<jobid> job.sh            # run only after another job succeeds
sbatch --dependency=singleton job.sh                  # run after all same-name jobs finish

# ── Monitoring ─────────────────────────────────────────────────────────────
squeue --me                                           # your jobs (running + pending)
squeue -j <jobid>                                     # one specific job
scontrol show job <jobid>                             # full job details (state, nodes, reason)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS  # post-completion accounting
sinfo                                                 # partition and node availability
scontrol show node <nodename>                         # e.g. scontrol show node cn21-a40
scontrol show partition <name>                        # full partition config

# ── Modifying ──────────────────────────────────────────────────────────────
scontrol hold <jobid>                                 # pause a pending job
scontrol release <jobid>                              # resume a held job
scontrol update jobid=<id> TimeLimit=<HH:MM:SS>       # change walltime of submitted job

# ── Cancelling ─────────────────────────────────────────────────────────────
scancel <jobid>                                       # cancel one job
scancel --me                                          # cancel ALL your jobs

# ── Interactive sessions ────────────────────────────────────────────────────
# Quick test (30 min, debug, nearly always free):
srun --partition=debug --qos=debug --gres=gpu:1 --ntasks=4 --time=00:25:00 --pty bash

# Longer interactive (up to 4h):
srun --partition=interactive --qos=interactive --gres=gpu:1 --ntasks=4 --time=02:00:00 --pty bash
```

**squeue state codes:** `R`=Running | `PD`=Pending | `CG`=Completing | `F`=Failed | `TO`=Timeout | `CA`=Cancelled
**Node states in sinfo:** `idle`=free | `mix`=partially used | `alloc`=fully busy | `down`=unavailable | `drain`=being taken offline

---

## 8. SLURM Batch Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=a40                  # l40 (default) | a40 | dgx | dgx-mpi | interactive | debug
#SBATCH --qos=a40                        # must exactly match --partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8             # CPU cores per node (max: a40=64, l40=32, dgx=256)
#SBATCH --gres=gpu:1                     # GPUs requested (max per job: a40=2, l40=4, dgx=4, dgx-mpi=8)
#SBATCH --time=04:00:00                  # HH:MM:SS — accurate estimate matters for backfill priority
#SBATCH --mem=32G                        # RAM per node (no hard cap on Prajna; be reasonable)
#SBATCH --output=logs/job_%J.out        # relative to submission dir (run sbatch from $HOME)
#SBATCH --error=logs/job_%J.err
# Optional: #SBATCH --mail-type=BEGIN,END,FAIL
# Optional: #SBATCH --mail-user=your@email.ac.in

# Ensure SLURM tools available in non-login shells
export PATH=$PATH:/opt/slurm/bin

# Fail fast on errors; -u omitted intentionally (breaks conda activate)
set -eo pipefail

# Activate conda (required in non-interactive job shells)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Write outputs to scratch; copy to home at end
SCRATCH=/lustre-scratch/$(whoami)
mkdir -p $SCRATCH/results

cd $HOME/my_project
python train.py --output $SCRATCH/results

# Archive results to home after job
cp -r $SCRATCH/results $HOME/my_project/results/
```

Submit from `$HOME` (the `logs/` path in `--output` is relative — must run from `$HOME`):
```bash
mkdir -p ~/logs            # create log dir if it doesn't exist yet
cd $HOME && sbatch job.sh
```
`$HOME` and `$(whoami)` inside the job script are resolved at runtime on the compute node.

---

## 9. Key SBATCH Parameters

| Parameter | Syntax | Notes |
|-----------|--------|-------|
| Partition | `--partition=a40` | `l40` (default), `a40`, `dgx`, `dgx-mpi`, `interactive`, `debug` |
| QOS | `--qos=a40` | Must exactly match partition |
| Account | `--account=$(id -gn)` | Your group — resolved at submit time |
| Nodes | `--nodes=1` | Number of nodes |
| CPU cores | `--ntasks-per-node=8` | Cores per node |
| GPUs | `--gres=gpu:2` | Respect QOS MaxTRES cap: a40≤2, l40≤4, dgx≤4, dgx-mpi≤8 |
| Walltime | `--time=08:00:00` | HH:MM:SS — be accurate for backfill scheduling |
| RAM | `--mem=64G` | Per node; no hard cap on Prajna but be fair |
| stdout | `--output=job.%J.out` | `%J`=jobid, `%A`=array-jobid, `%a`=array-taskid |
| stderr | `--error=job.%J.err` | |
| Job name | `--job-name=train` | Shown in squeue |
| Email events | `--mail-type=BEGIN,END,FAIL` | Also: ALL, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_50 |
| Email target | `--mail-user=your@email.ac.in` | Optional — for job notifications |
| Job array | `--array=0-9` | `$SLURM_ARRAY_TASK_ID` available in script |
| Array concurrency | `--array=0-9%4` | Max 4 simultaneous array tasks |
| Dependency | `--dependency=afterok:<jobid>` | After a job succeeds |
| Singleton | `--dependency=singleton` | After all same-name jobs complete |
| Pin to nodes | `--nodelist=cn20-a40,cn21-a40` | Specific nodes (use `sinfo` to find idle ones) |
| Exclusive | `--exclusive` | Full node, not shared — use sparingly |
| Dry run | `--test-only` | Validates script + estimates start time; does not submit |

---

## 10. End-to-End Workflow

```bash
# Load credentials from .env
set -a; source .env; set +a

# ── From your local machine ──────────────────────────────────────────────
scp -r ./my_project ${PRAJNA_USER}@prajna.iitb.ac.in:~/

# ── SSH in ───────────────────────────────────────────────────────────────
ssh ${PRAJNA_USER}@prajna.iitb.ac.in   # TOTP + password (or via ControlMaster: ssh prajna)

# Step 1: Create conda env if not already done (login node OK — lightweight)
conda create -n myenv python=3.10 -y
conda activate myenv
pip install <packages>

# Step 2: Create your scratch directory
mkdir -p /lustre-scratch/${PRAJNA_USER}/my_project/results
mkdir -p ~/logs

# Step 3: Test your script quickly on debug partition (30 min, no queue wait)
#   Set --partition=debug --qos=debug --time=00:25:00 in job.sh temporarily
/opt/slurm/bin/sbatch job.sh
/opt/slurm/bin/squeue --me          # watch it; check logs/job_<id>.out if it fails

# Step 4: Submit the real job with proper partition and time
#   Restore --partition=a40 --qos=a40 --time=<realistic estimate>
/opt/slurm/bin/sbatch job.sh        # returns: "Submitted batch job <jobid>"
/opt/slurm/bin/squeue --me          # monitor; PD=pending, R=running
/opt/slurm/bin/scontrol show job <jobid>    # see reason if stuck in PD

# Step 5: After completion — results already copied to home by the job script itself

# ── From your local machine ──────────────────────────────────────────────
scp -r ${PRAJNA_USER}@prajna.iitb.ac.in:~/my_project/results ./
```

---

## 11. Storage Rules

| Path | Size | Purpose | Purge policy |
|------|------|---------|-------------|
| `$HOME/` | 1.3 PB shared | Code, conda envs, final results | No quota, no auto-purge |
| `/lustre-scratch/$PRAJNA_USER/` | 820 TB shared | Job I/O, large temp data | **Deleted after 3 months of no access** |
| `/lustre-flash/` | 399 TB | Spack system software | Do NOT write here |
| `/scratch/` | 250 GB | SLURM system dir | Do NOT write here |

**Best practice:** Write job outputs to `/lustre-scratch/$PRAJNA_USER/`, then `cp` final results to `$HOME/` at the end of the job script. Back up important data to your local machine.

---

## 12. File Transfer

```bash
# Load credentials from .env first:
set -a; source .env; set +a

# Upload to cluster (from local machine)
scp -r ./my_project ${PRAJNA_USER}@prajna.iitb.ac.in:~/

# Download from cluster (to local machine)
scp -r ${PRAJNA_USER}@prajna.iitb.ac.in:~/results ./results

# rsync (faster for large or incremental transfers)
rsync -avz --progress ./my_project ${PRAJNA_USER}@prajna.iitb.ac.in:~/

# Copy between home and scratch (on cluster)
cp -r ~/my_project /lustre-scratch/${PRAJNA_USER}/
```

---

## 13. What Is Outdated / Known Discrepancies

These were found by cross-checking the PDF manual and website against live SLURM on 2026-04-29.

| Item | PDF Manual says | Website says | Live reality |
|------|----------------|--------------|-------------|
| OS | Rocky Linux 9 | — | **Rocky Linux 8.10** |
| SLURM version | 23.11.10 | — | **24.05.6** |
| A100 partition name | — | `gpu_a100` | **`dgx`** |
| A40 partition name | — | `gpu_a40` | **`a40`** |
| L40S partition name | — | `gpu_l40s` | **`l40`** |
| L4 partition | — | `gpu_l4` (listed) | **Does not exist in SLURM** |
| User scratch path | `/scratch` | `/scratch` | **`/lustre-scratch/`** |
| `/scratch` role | User temp storage | User temp storage | **SLURM system dir — not for users** |
| Max A40 jobs/user | — | 5 run / 1 queue | **3 running, 6 total (from sacctmgr)** |
| Max interactive jobs | — | 1 | **2 (from sacctmgr)** |
| Max GPUs/job (A40) | — | "2" | **2 (confirmed via MaxTRES in QOS)** |
| Max GPUs/job (L40S) | — | "8×N" | **4 (MaxTRES=gres/gpu=4 in QOS)** |
| Max GPUs/job (DGX) | — | "4×N" | **4 (MaxTRES=gres/gpu=4 in QOS)** |
| A40 node count | 20 | 20 | **19 in SLURM** (cn20-a40 absent) |
| `debug` partition | not listed | not listed | **Exists — 30 min, 1 A40 node** |
| Module command | implied | implied | **Not available** — use conda/Spack |
| Conda in ~/.bashrc | not mentioned | not mentioned | **Already initialised** (conda init block) |
| SSH auth | password only | password only | **TOTP + password required** (2FA since 2026-04-28) |
| SSH key bypasses 2FA | — | — | **No** — TOTP required even with pubkey |

---

## 14. How to Verify and Update This File

Run these commands **on the cluster** to get ground truth. Always trust live SLURM over website or PDF.

```bash
export PATH=$PATH:/opt/slurm/bin

# ── Partition names and availability ──────────────────────────────────
sinfo
scontrol show partition        # full config of all partitions

# ── GPU and job limits per QOS ────────────────────────────────────────
# NOTE: regular users get empty output — sacctmgr requires admin access on Prajna.
# Use scontrol show partition <name> to verify walltimes without admin access.
/opt/slurm/bin/sacctmgr show qos -p format=name,maxwall,maxjobspu,maxsubmitjobspu,maxnodes,maxtresperjob
# Look at MaxTRES column for gres/gpu=N — that's the real GPU cap per job

# Alternative (no admin needed):
scontrol show partition l40
scontrol show partition a40
scontrol show partition dgx

# ── Your account limits ───────────────────────────────────────────────
/opt/slurm/bin/sacctmgr show user $PRAJNA_USER withassoc format=user,account,maxjobs,maxsubmit

# ── Storage and quotas ────────────────────────────────────────────────
df -h | grep -vE 'tmpfs|devtmpfs'
lfs quota -u $PRAJNA_USER /home
lfs quota -u $PRAJNA_USER /lustre-scratch

# ── OS and SLURM versions ─────────────────────────────────────────────
cat /etc/os-release | grep PRETTY
sinfo --version

# ── Re-fetch PDF manual (SSL cert is self-signed — use -k) ───────────
curl -k -L -o /tmp/prajna_manual.pdf \
  "https://hpcverse.iitb.ac.in/st2manuals/manuals/Updated%20User%20Manual%20for%20Prajna%20(AiML)%20HPC.pdf"
# Then read it: agent can use Read tool on /tmp/prajna_manual.pdf

# ── Re-fetch queue policy page ────────────────────────────────────────
curl -k -s -L "https://hpcverse.iitb.ac.in/queue-policy/prajna" \
  | python3 -c "import sys,html,re; b=sys.stdin.read(); print(html.unescape(re.sub(r'<[^>]+>',' ',b)))"

# ── Re-fetch hardware specs ───────────────────────────────────────────
curl -k -s -L "https://hpcverse.iitb.ac.in/platform-aiml" \
  | python3 -c "import sys,html,re; b=sys.stdin.read(); print(html.unescape(re.sub(r'<[^>]+>',' ',b)))"
```

After running the above, update Sections 3, 4, and 13. Update the "Live-Verified" date.

---

## 15. Support & Reference Links

| Resource | URL / Contact |
|----------|--------------|
| HPC portal | https://hpcverse.iitb.ac.in |
| Prajna hardware overview | https://hpcverse.iitb.ac.in/platform-aiml |
| Prajna queue policy | https://hpcverse.iitb.ac.in/queue-policy/prajna |
| Prajna user manual (PDF) | https://hpcverse.iitb.ac.in/st2manuals/manuals/Updated%20User%20Manual%20for%20Prajna%20(AiML)%20HPC.pdf |
| Prajna manuals page | https://hpcverse.iitb.ac.in/prajna-manuals |
| Account request / management | https://hpcaccreq.iitb.ac.in |
| Submit a support ticket | https://help.cc.iitb.ac.in |
| Computer Centre website | https://cc.iitb.ac.in |
| Email support | hpc@iitb.ac.in |
| Phone helpline | Ext. **2678** — Mon–Fri, 9:30 AM–5:30 PM |
| Helpline location | CC Building, G-02, Ground Floor, IIT Bombay |

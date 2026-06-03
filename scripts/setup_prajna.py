#!/usr/bin/env python3
"""
Scroll Prize — Prajna HPC automation entry-point.

What this does:
  1. Loads credentials from .env
  2. Connects (handles TOTP 2FA automatically)
  3. Verifies the remote environment is intact
  4. Syncs local scroll_prize/ → ~/scroll_prize/ on Prajna (excludes data/, results/)
  5. Submits a job from a script string and polls until done

Usage:
  python setup_prajna.py                # verify + sync only
  python setup_prajna.py --submit       # sync + submit scroll_train.sh
  python setup_prajna.py --check        # print cluster queue and env status

Requires: pip install paramiko pyotp  (or: pip install --break-system-packages paramiko pyotp)
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prajna_lib import (
    _load_env, connect_prajna, run, run_paramiko, stream_command,
    upload_project, ensure_env, submit_job, active_jobs, wait_until_done
)

# ── Constants ──────────────────────────────────────────────────────────────────
LOCAL_ROOT   = Path(__file__).parent.parent  # project root (scroll_prize/)
SCRIPTS_DIR  = Path(__file__).parent         # scripts/
REMOTE_PROJ  = "scroll_prize"                # relative to $HOME on Prajna
CONDA_ENV    = "scroll"
SQUEUE       = "/opt/slurm/bin/squeue"
SBATCH       = "/opt/slurm/bin/sbatch"


def verify_env(client):
    out, _, _ = run(client,
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && "
        "python -c \"import torch, vesuvius, zarr, open3d; "
        "print('torch', torch.__version__, 'vesuvius OK, zarr', zarr.__version__)\"",
        timeout=120, check=False)
    print("[verify]", out if out else "(no output — check conda env)")


def sync_project(client):
    """Rsync local scroll_prize/ → remote ~/scroll_prize/ skipping large dirs."""
    user = os.environ["PRAJNA_USER"]
    # Use rsync over the ControlMaster socket
    skip = ["--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
            "--exclude=.env", "--exclude=results/", "--exclude=data/",
            "--exclude=checkpoints/", "--exclude=*.egg-info/"]
    import subprocess
    ctrl = str(Path.home() / f".ssh/ctl/{user}@prajna.iitb.ac.in:22")
    ssh_opts = f"ssh -o ControlPath={ctrl} -o ControlMaster=no -o BatchMode=yes"
    cmd = (["rsync", "-avz", "--progress"] + skip +
           ["-e", ssh_opts,
            str(LOCAL_ROOT) + "/",
            f"{user}@prajna.iitb.ac.in:~/{REMOTE_PROJ}/"])
    subprocess.run(cmd, check=True)


def submit(client):
    job_script = (SCRIPTS_DIR / "scroll_train.sh").read_text()
    # Ensure log dir exists first
    user_home = f"/home/{os.environ.get('PRAJNA_GROUP', 'medal')}/{os.environ['PRAJNA_USER']}"
    run(client, f"mkdir -p {user_home}/logs", timeout=15)
    jid = submit_job(client, job_script)
    print(f"[submit] Submitted job {jid}")
    return jid


def check_status(client):
    out, _, _ = run(client, f"{SQUEUE} --me", timeout=30, check=False)
    print("[queue]\n", out or "(empty — no running/pending jobs)")
    out2, _, _ = run(client,
        "source ~/miniconda3/etc/profile.d/conda.sh && conda env list | grep scroll",
        timeout=120, check=False)
    print("[envs]", out2)


def main():
    parser = argparse.ArgumentParser(description="Scroll Prize / Prajna HPC helper")
    parser.add_argument("--submit", action="store_true", help="Sync and submit scroll_train.sh")
    parser.add_argument("--check",  action="store_true", help="Print queue + env status only")
    parser.add_argument("--wait",   action="store_true", help="Poll until submitted jobs finish")
    args = parser.parse_args()

    _load_env()
    client = connect_prajna()

    if args.check:
        check_status(client)
        client.close()
        return

    print("[setup] Verifying remote conda env...")
    verify_env(client)

    print("[setup] Syncing project files...")
    sync_project(client)

    if args.submit:
        jid = submit(client)
        if args.wait:
            print("[setup] Waiting for jobs to finish...")
            wait_until_done(client, name_prefix="scroll")

    client.close()
    print("[setup] Done.")


if __name__ == "__main__":
    main()

# Prajna HPC — Automation Runbook

**This file covers how to automate work on Prajna via SSH ControlMaster + subprocess
(preferred) or paramiko with pyotp (fallback). Pitfalls, SSH helpers, SFTP upload,
job submission, wave logic, and complete scripts.**

Cluster specs, partition names, QOS limits, storage paths, SLURM commands, hardware,
known discrepancies, and support contacts are in **`PRAJNA_HPC.md`** — read that first.

All credentials come from `.env`. Never hardcode them. If `.env` is missing, the wizard
in §0.0 asks for the minimum and builds it automatically.

---

## 0. Setup — Credentials + ControlMaster

### 0.0 First-time wizard — no `.env` yet

When `.env` is absent, call `interactive_setup()`. It asks for three things, logs in,
reads the rest from the server, and writes a complete `.env`.

**Two paths depending on what the user has:**
- **Has TOTP secret key** (base32 string from Google Authenticator setup, or first line of
  `~/.google_authenticator` on Prajna) → enters it directly, full automation from that moment.
- **Has only a live 6-digit code** (opened Google Authenticator right now) → wizard logs in
  once with that code, reads `~/.google_authenticator` on the server to get the secret, stores
  it. Full automation from then on.

```python
import getpass, os, re, sys, subprocess
from pathlib import Path

HOST = "prajna.iitb.ac.in"

def _fetch_totp_secret_via_code(user, password, totp_code):
    """One-shot login with a live TOTP code; reads TOTP secret from server."""
    try:
        import paramiko
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", "paramiko"], check=False)
        import paramiko

    t = paramiko.Transport((HOST, 22))
    t.connect()
    def handler(title, instructions, prompt_list):
        responses = []
        for prompt, _ in prompt_list:
            p = prompt.lower()
            if "password" in p:
                responses.append(password)
            elif "verification" in p or "code" in p:
                responses.append(totp_code)
            else:
                responses.append("")
        return responses
    try:
        t.auth_interactive(user, handler)
    except paramiko.AuthenticationException:
        t.close()
        raise SystemExit(
            "[prajna] Login failed — check username, password, and the 6-digit TOTP code.\n"
            "TOTP codes expire every 30s — open your app and try again with a fresh code."
        )
    ch = t.open_session()
    ch.exec_command("head -1 ~/.google_authenticator")
    secret = ch.makefile().read().strip()
    t.close()
    return secret if len(secret) > 10 else ""

def interactive_setup(env_dir="."):
    """Ask minimum questions, log in, auto-fill .env. Call when .env is missing."""
    print("[prajna] .env not found — first-time setup")
    user     = input("  Username: ").strip()
    password = getpass.getpass("  Password: ").strip()   # hidden input

    secret_raw = input(
        "  TOTP secret key (base32 from Google Auth setup — leave blank if you don't have it): "
    ).strip()

    if secret_raw:
        secret = secret_raw
    else:
        code = input("  Open Google Authenticator now and enter the current 6-digit code: ").strip()
        print("  [prajna] Logging in once to read TOTP secret from server...")
        secret = _fetch_totp_secret_via_code(user, password, code)
        if not secret:
            raise SystemExit(
                "[prajna] Could not read TOTP secret from server.\n"
                "Log in interactively, run `head -1 ~/.google_authenticator`, "
                "and paste that string as PRAJNA_TOTP_SECRET in .env."
            )
        print("  [prajna] Got TOTP secret — storing in .env.")

    email = input("  Email for SLURM job notifications (Enter to skip): ").strip()

    env_path = Path(env_dir) / ".env"
    env_path.write_text(
        "# Prajna HPC credentials — managed by prajna_setup()\n"
        f"PRAJNA_USER={user}\n"
        f"PRAJNA_PASSWORD={password}\n"
        f"PRAJNA_TOTP_SECRET={secret}\n"
        f"PRAJNA_EMAIL={email}\n"
        "PRAJNA_GROUP=\nPRAJNA_UID=\nPRAJNA_GID=\n"
        "PRAJNA_SCRATCH_1=\nPRAJNA_SCRATCH_2=\nPRAJNA_SCRATCH_3=\n"
        "PRAJNA_SCRATCH_4=\nPRAJNA_SCRATCH_5=\n"
    )
    print(f"  [prajna] .env written → {env_path.resolve()}")
```

After `interactive_setup()`, call `_load_env()` then `connect_prajna()` then `prajna_setup(client)`
to finish filling group/uid/gid/scratch codes — all in one shot.

---

### 0.1 Load credentials (every script starts with this)

Three required fields, optional email. Everything else (`PRAJNA_GROUP`, `PRAJNA_UID`,
`PRAJNA_GID`, scratch codes) is auto-filled by `prajna_setup()` — see §0.2.

> `interactive_setup()` (§0.0) must be defined before `_load_env()` so the fallback call works.

```python
import os, re
from pathlib import Path

_ENV_PATH = None   # set by _load_env()

def _load_env(search=(".", "..")):
    global _ENV_PATH
    for d in search:
        p = Path(d) / ".env"
        if p.exists():
            _ENV_PATH = p.resolve()
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return
    # .env not found — run the interactive wizard (see §0.0)
    interactive_setup(search[0])
    _load_env(search)   # reload after wizard writes the file

_load_env()

HOST        = "prajna.iitb.ac.in"           # never changes
USER        = os.environ["PRAJNA_USER"]
PASSWORD    = os.environ["PRAJNA_PASSWORD"]
TOTP_SECRET = os.environ["PRAJNA_TOTP_SECRET"]
```

**Derived values — never stored, computed on use:**
```python
GROUP = os.environ.get("PRAJNA_GROUP", "")
HOME  = f"/home/{GROUP}/{USER}" if GROUP else None   # set after prajna_setup()
```

In bash scripts, load with:
```bash
set -a; source .env; set +a
# Home is always derivable:
PRAJNA_HOME=/home/${PRAJNA_GROUP}/${PRAJNA_USER}
```

### 0.2 First-run setup — auto-fill `.env` from server

Run `prajna_setup()` once after filling in the 3 required fields.
It connects, fetches group/uid/gid/scratch codes, and writes them back into `.env`.

> `prajna_setup()` calls `run_paramiko` — paste `run_paramiko` from §3 (or alias it) before calling this.

```python
def _write_env_key(key, value):
    """Update or append a key=value line in .env, preserving all comments."""
    if _ENV_PATH is None:
        return
    text = _ENV_PATH.read_text()
    pattern = re.compile(rf"^({re.escape(key)}=).*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(rf"\g<1>{value}", text)
    else:
        text = text.rstrip("\n") + f"\n{key}={value}\n"
    _ENV_PATH.write_text(text)
    os.environ[key] = value

def prajna_setup(client):
    """Fetch server info and write it into .env. Run once after first connect.

    Fills: PRAJNA_GROUP, PRAJNA_UID, PRAJNA_GID, PRAJNA_SCRATCH_1..5
    Skips any field already set (won't overwrite existing values).
    """
    def _r(cmd): return run_paramiko(client, cmd, timeout=15)[0]

    if not os.environ.get("PRAJNA_GROUP"):
        _write_env_key("PRAJNA_GROUP", _r("id -gn"))
    if not os.environ.get("PRAJNA_UID"):
        _write_env_key("PRAJNA_UID",   _r("id -u"))
    if not os.environ.get("PRAJNA_GID"):
        _write_env_key("PRAJNA_GID",   _r("id -g"))

    # Scratch codes live in ~/.google_authenticator — server removes each when used
    need_scratches = any(
        not os.environ.get(f"PRAJNA_SCRATCH_{i}") for i in range(1, 6)
    )
    if need_scratches:
        ga, _, rc = run_paramiko(client, "cat ~/.google_authenticator",
                                 timeout=15, check=False)
        codes = [l.strip() for l in ga.splitlines()
                 if l.strip().isdigit() and len(l.strip()) == 8]
        for i, code in enumerate(codes[:5], 1):
            if not os.environ.get(f"PRAJNA_SCRATCH_{i}"):
                _write_env_key(f"PRAJNA_SCRATCH_{i}", code)

    print(f"[prajna_setup] .env updated → group={os.environ.get('PRAJNA_GROUP')} "
          f"uid={os.environ.get('PRAJNA_UID')} "
          f"scratches_found={sum(bool(os.environ.get(f'PRAJNA_SCRATCH_{i}')) for i in range(1,6))}")
```

### 0.3 SSH ControlMaster — authenticate once, reuse forever

Prajna requires TOTP + password on every new SSH connection (2FA enforced even for key
auth). ControlMaster solves this: authenticate **once**, then all subsequent connections
— including from scripts — reuse the socket with no prompts, until you reboot or kill it.

#### One-time local setup (run once on each new machine)

```bash
# 1. Create socket directory
mkdir -p ~/.ssh/ctl
chmod 700 ~/.ssh/ctl

# 2. Add to ~/.ssh/config  (create it if it doesn't exist)
set -a; source .env; set +a   # load PRAJNA_USER from .env
cat >> ~/.ssh/config << EOF

Host prajna prajna.iitb.ac.in
    HostName prajna.iitb.ac.in
    User $PRAJNA_USER
    ControlMaster auto
    ControlPath ~/.ssh/ctl/%r@%h:%p
    ControlPersist yes
EOF
chmod 600 ~/.ssh/config
```

> The unquoted `<< EOF` expands `$PRAJNA_USER` from your shell, writing your actual username
> into the config. Note: `~` in `ControlPath` is intentionally kept — ssh expands it at runtime.

#### Establish the master connection (once — persists until you kill it or reboot)

```bash
set -a; source .env; set +a
ssh prajna "echo ready"   # enter TOTP + password once
```

After that, every `ssh prajna`, `scp`, `rsync`, and script using the socket works
instantly with zero auth prompts.

```bash
ssh -O check prajna        # prints "Master running (pid=…)" or error
ssh -O exit prajna         # kill master early
```

#### If SSH hangs after laptop sleep / lock

Socket goes stale on suspend. Fix manually:
```bash
ssh -O exit prajna 2>/dev/null; ssh prajna "echo ready"
```

Or fully automatically from Python (no interaction needed — uses TOTP secret from `.env`):
```python
import subprocess, time, pyotp, pexpect   # pip install pexpect

def reestablish_master():
    """Kill stale ControlMaster socket and re-authenticate automatically."""
    # Kill stale socket (ignore error if already dead)
    subprocess.run(["ssh", "-O", "exit", "-o", f"ControlPath={CTRL_SOCKET}",
                    f"{USER}@{HOST}"], capture_output=True)
    # Spawn new master, respond to TOTP + password prompts
    child = pexpect.spawn(
        f"ssh -M -N -f -o ControlPath={CTRL_SOCKET} -o ControlPersist=yes "
        f"-o StrictHostKeyChecking=no {USER}@{HOST}", timeout=30)
    child.expect("Verification code:")
    child.sendline(pyotp.TOTP(TOTP_SECRET).now())
    child.expect("Password:")
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF)
    time.sleep(2)   # let master start before using socket
    print("[prajna] ControlMaster re-established.")

# Uses USER, PASSWORD, TOTP_SECRET, CTRL_SOCKET from §0.3 — call _load_env() first
```

If Prajna itself is unreachable: `ping -c 3 prajna.iitb.ac.in`

#### Preferred agent pattern — subprocess over ControlMaster socket

```python
import subprocess, os
from pathlib import Path

_load_env()   # from §0.1

HOST        = "prajna.iitb.ac.in"
USER        = os.environ["PRAJNA_USER"]
CTRL_SOCKET = str(Path.home() / f".ssh/ctl/{USER}@{HOST}:22")
SSH_OPTS    = ["-o", f"ControlPath={CTRL_SOCKET}", "-o", "ControlMaster=no",
               "-o", "BatchMode=yes"]

def run(cmd, timeout=60, check=True):
    """Run cmd on Prajna via ControlMaster socket — no auth required if master is up."""
    r = subprocess.run(
        ["ssh"] + SSH_OPTS + [f"{USER}@{HOST}", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd}\n{r.stderr}")
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def upload(local_path, remote_path):
    """Upload file/dir via scp over ControlMaster socket."""
    subprocess.run(
        ["scp"] + SSH_OPTS + ["-r", str(local_path), f"{USER}@{HOST}:{remote_path}"],
        check=True
    )

def rsync_up(local_root, remote_root, excludes=("__pycache__", "*.pyc", "results/", "cache/")):
    """Incremental upload via rsync over ControlMaster socket."""
    excl = []
    for e in excludes:
        excl += ["--exclude", e]
    subprocess.run(
        ["rsync", "-avz", "--progress"]
        + excl                                          # excludes must come before src/dst
        + ["-e", f"ssh {' '.join(SSH_OPTS)}",
           str(local_root) + "/", f"{USER}@{HOST}:{remote_root}/"],
        check=True
    )
```

> If the master socket is not up, `BatchMode=yes` causes an immediate error instead of
> hanging at the TOTP prompt. Check status with `ssh -O check prajna`.

**TOTP with ControlMaster:** Only one real auth per session — avoids all DISALLOW_REUSE
and rate-limit issues. See §1.9 for the full failure mode reference.

### Fallback — pyotp + paramiko (when no ControlMaster available)

See Section 2 below. Use when running headlessly or the ControlMaster socket has expired.

---

## 1. Critical Pitfalls

### 1.0 Installing dependencies on the local machine (Kali / Debian / Ubuntu 23+)

Some distros (Kali, Debian 12+, Ubuntu 23.04+) block `pip install` with:
```
error: externally-managed-environment
```
This is PEP 668 — the system Python is protected. Use this helper at the top of every
agent script instead of a bare `pip install`:

```python
import subprocess, sys

def _ensure_deps(*pkgs):
    """Install packages if missing. Handles externally-managed-environment (Kali/Debian)."""
    import importlib
    for pkg in pkgs:
        mod = pkg.split("[")[0].replace("-", "_")   # e.g. "pyotp", "paramiko"
        try:
            importlib.import_module(mod)
            continue                                  # already installed
        except ImportError:
            pass
        installed = False
        for cmd in [
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg],
            ["conda", "install", "-y", "-q", pkg],
        ]:
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                installed = True
                break
        if not installed:
            raise SystemExit(
                f"Could not install '{pkg}'.\n"
                f"Try manually: pip install --break-system-packages {pkg}\n"
                f"Or in a venv: python3 -m venv .venv && source .venv/bin/activate"
            )

_ensure_deps("paramiko", "pyotp")
import paramiko, pyotp   # safe to import now
```

Put `_ensure_deps(...)` and the imports **after** it at the very top of every script,
before any other code that uses these packages.

### 1.1 SLURM binaries not in PATH in scripted SSH

Non-interactive SSH (`paramiko`, `ssh host "cmd"`) does NOT source `~/.bashrc`.
`/opt/slurm/bin/` is not in PATH. Always use full paths.

```python
# WRONG — "bash: squeue: command not found"
run("squeue -u $USER")

# CORRECT
SQUEUE   = "/opt/slurm/bin/squeue"
SBATCH   = "/opt/slurm/bin/sbatch"
SCANCEL  = "/opt/slurm/bin/scancel"
SACCT    = "/opt/slurm/bin/sacct"
SCONTROL = "/opt/slurm/bin/scontrol"
SACCTMGR = "/opt/slurm/bin/sacctmgr"
run(f"{SQUEUE} -u {USER}")
```

In bash scripts called over SSH, either use full paths or prepend:
```bash
export PATH=/opt/slurm/bin:$PATH
```

### 1.2 Conda not sourced in SLURM job scripts

SLURM jobs are non-interactive. `~/.bashrc` is NOT sourced. Conda is unavailable.

```bash
# WRONG — "conda: command not found"
conda activate myenv

# CORRECT — source conda.sh explicitly at top of every job script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
```

### 1.3 `set -u` breaks conda activate

`set -euo pipefail` causes conda to crash on its own internal unset variables.

```bash
# WRONG
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv   # aborts: unbound variable

# CORRECT — drop the -u flag
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
```

### 1.4 Log directory must exist before sbatch

If `--output` or `--error` points to a directory that doesn't exist, the job is accepted
but immediately fails with no visible error.

```bash
# Always create the log dir before submitting
mkdir -p $HOME/myproject/logs
/opt/slurm/bin/sbatch myjob.sh
```

### 1.5 CUDA errors on the login node are harmless

The login node has no GPU. JAX/PyTorch/TensorFlow print CUDA errors on import.
This is expected — they fall back to CPU. Compute nodes have full GPU access.

```
RuntimeError: operation cuInit(0) failed: Unknown CUDA error 303
# ↑ Normal on login node. Do not abort for this.
```

### 1.6 No module system

There is no `module load` command. Use:
- **conda** for Python packages and environments
- **Spack** at `/lustre-flash/apps/spack/share/spack/setup-env.sh` for compiled system libs

### 1.7 Do not run heavy compute on the login node

Policy: 1-month ban for abusing the login node.
Allowed on login node: `git`, file edits, `pip install`, small downloads (<1 GB), job submission.
Everything else goes in a SLURM job.

### 1.8 2FA enabled — simple password auth no longer works

Since **2026-04-28**, Prajna requires TOTP + password. Calling `paramiko.SSHClient.connect(password=…)`
without a custom keyboard-interactive handler will fail or send the password for the TOTP
prompt and be rejected.

Use `connect_prajna()` from Section 2 instead. It uses `pyotp` to generate the TOTP code
and a custom `auth_interactive` handler to respond to both prompts correctly.

```bash
# prerequisite (local machine only) — see §1.0 for systems where pip is blocked
pip install pyotp paramiko
# or on Kali/Debian:
pip install --break-system-packages pyotp paramiko
# or if using conda:
conda install -c conda-forge pyotp paramiko
```

### 1.9 TOTP edge cases

Server config: `DISALLOW_REUSE` (code valid once per 30s), ±4 min clock tolerance, 3 attempts/30s rate limit.

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Invalid verification code" | Same code used twice within 30s | Wait 31s, retry — `connect_prajna()` does this automatically |
| Every code rejected | Clock skew > 4 minutes | `date && ssh prajna "date"` to confirm; then `sudo timedatectl set-ntp true` or `sudo date -s "$(ssh prajna 'date -u')"` |
| Immediate rejection, no prompt | Rate limited (3 failures in 30s) | Wait 31s, don't retry before then |
| `ssh prajna` works but paramiko fails | ControlMaster dead | `ssh prajna "echo ready"` to re-establish |
| "Google Authenticator setup required" | 2FA not yet configured | Log in interactively, follow wizard, save secret + scratch codes to `.env` |

**Scratch codes (`PRAJNA_SCRATCH_1..5` in `.env`):** Single-use emergency codes — the server burns each one on use.
**⚠️ Agents must never use a scratch code automatically.** Stop and tell the user:
> TOTP auth failed and needs manual intervention. Check your TOTP app (wait for a fresh code if needed).
> If your app is unavailable, log in with `ssh $PRAJNA_USER@prajna.iitb.ac.in` and enter a scratch code from `.env` manually.
> Afterwards, update `PRAJNA_TOTP_SECRET` and `PRAJNA_SCRATCH_*` in `.env` with the new values.

Only use a scratch code programmatically if the user explicitly instructs it, then comment that code out in `.env`.

### 1.10 paramiko `exec_command` timeout causes silent failures

`client.exec_command(cmd, timeout=30)` sets a 30-second **channel read timeout**.
On Prajna's Lustre home filesystem, `conda env list` alone can take >30s.
`stdout.read()` then raises `socket.timeout` and the calling function crashes.

```python
# WRONG — crashes on slow commands like "conda env list"
_, stdout, stderr = client.exec_command("conda env list", timeout=30)
out = stdout.read()  # raises TimeoutError if command takes >30s

# CORRECT — set a generous timeout appropriate to the command,
# or use stream_command() (Section 7) for anything that takes >10s
_, stdout, stderr = client.exec_command(
    "source ~/miniconda3/etc/profile.d/conda.sh && conda env list",
    timeout=120   # 2 minutes for conda on Lustre
)
out = stdout.read().decode()
```

### 1.11 File upload — match only top-level directory names

When uploading via SFTP, skipping a path component named `"data"` silently drops
source subdirectories like `mypackage/data/loader.py`.

```python
# WRONG — drops src/data/, lib/data/, etc.
if any(part in {"data","results"} for part in path.parts):
    continue

# CORRECT — skip only if the FIRST component matches
if path.parts[0] in {"data", "results", ".git", "__pycache__"}:
    continue
```

---

## 2. SSH Access — paramiko fallback

> **Use ControlMaster (§0.3) when possible.** This section is the fallback for headless
> or automated contexts where no ControlMaster socket is available.
> `pip install pyotp paramiko` required on the local machine.

```python
import paramiko, pyotp, os
from pathlib import Path

_load_env()   # from §0.1

HOST        = "prajna.iitb.ac.in"
USER        = os.environ["PRAJNA_USER"]
TOTP_SECRET = os.environ["PRAJNA_TOTP_SECRET"]


def _do_connect(password, totp_secret):
    """Single connection attempt. Raises paramiko.AuthenticationException on auth failure."""
    transport = paramiko.Transport((HOST, 22))
    transport.connect()

    totp = pyotp.TOTP(totp_secret)

    def _handler(title, instructions, prompt_list):
        responses = []
        for prompt, echo in prompt_list:
            p = prompt.strip().lower()
            if "verification" in p or "code" in p or "authenticat" in p:
                responses.append(totp.now())
            else:
                responses.append(password)
        return responses

    transport.auth_interactive(USER, _handler)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client._transport = transport
    return client


def connect_prajna(password=None, totp_secret=None):
    """Connect with TOTP + password. Handles DISALLOW_REUSE with one automatic retry.

    TOTP failure modes handled automatically:
      - DISALLOW_REUSE (same code used twice in 30s): waits 31s, retries with fresh code
      - Rate limit (3 failures/30s): waits 31s before retry

    TOTP failure modes that require user intervention (raises with clear message):
      - Clock skew > 4 minutes: tells user to sync clock
      - Scratch codes needed: NEVER used automatically — user must act

    See PRAJNA_RUNBOOK.md §1.9 for full edge-case reference.
    """
    password    = password    or os.environ["PRAJNA_PASSWORD"]
    totp_secret = totp_secret or os.environ["PRAJNA_TOTP_SECRET"]

    try:
        return _do_connect(password, totp_secret)
    except paramiko.AuthenticationException as e:
        err = str(e).lower()

        # DISALLOW_REUSE or rate limit: wait 31s for the window to roll over, retry once
        if "keyboard-interactive" in err or "authentication" in err:
            import time
            print("[prajna] TOTP auth failed — likely DISALLOW_REUSE or rate limit. "
                  "Waiting 31s for window to roll over before retrying...")
            time.sleep(31)
            try:
                return _do_connect(password, totp_secret)
            except paramiko.AuthenticationException:
                pass   # fall through to diagnosis below

        # Diagnose: check clock skew
        import subprocess, datetime
        try:
            server_time_str = subprocess.run(
                ["ssh", "-o", "BatchMode=yes",
                 "-o", f"ControlPath={str(Path.home())}/.ssh/ctl/{USER}@{HOST}:22",
                 "-o", "ControlMaster=no",
                 f"{USER}@{HOST}", "date +%s"],
                capture_output=True, text=True, timeout=10
            ).stdout.strip()
            if server_time_str.isdigit():
                skew = abs(int(datetime.datetime.now().timestamp()) - int(server_time_str))
                if skew > 240:
                    raise RuntimeError(
                        f"[prajna] TOTP auth failed — clock skew is {skew}s (> 4 min limit).\n"
                        f"Fix: sudo timedatectl set-ntp true  (or sync manually)\n"
                        f"Then retry."
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        raise RuntimeError(
            "[prajna] TOTP authentication failed after retry.\n"
            "\n"
            "Checklist:\n"
            "  1. Is your TOTP app showing the current code? Open it and wait for a fresh one.\n"
            "  2. Is your local clock correct?  Run: date\n"
            "     Sync if needed: sudo timedatectl set-ntp true\n"
            "  3. Have you used this code already in the last 30s?\n"
            "     Wait 31s and try again.\n"
            "\n"
            "If your TOTP app is unavailable:\n"
            "  → Use a scratch code MANUALLY — log in interactively:\n"
            f"    ssh {USER}@{HOST}\n"
            "    Enter a PRAJNA_SCRATCH_* code from .env when prompted.\n"
            "    Then run google-authenticator to get a new TOTP secret.\n"
            "    Update .env with the new PRAJNA_TOTP_SECRET and PRAJNA_SCRATCH_* codes.\n"
            "\n"
            "⚠️  Scratch codes are NEVER used automatically — they are single-use and\n"
            "    irreplaceable. A human must approve using one."
        )


def reconnect():
    """Drop-in reconnect for autopilot loops after SSHException / channel timeout."""
    return connect_prajna()


```

---

## 3. Running Commands via paramiko

```python
def run_paramiko(client, cmd, timeout=60, check=True):
    """Run cmd, return (stdout_str, stderr_str, exit_code). Raises on non-zero if check=True."""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\n{err}")
    return out, err, rc
```

**Timeout rules of thumb:**

| Command type | Suggested timeout |
|---|---|
| Quick checks (`ls`, `echo`) | 15s |
| SLURM commands (`squeue`, `sbatch`) | 30s |
| `conda env list` on Lustre | 120s |
| `conda activate` + import check | 120s |
| `pip install` (small) | 600s |
| `pip install` (jax, torch) | 1800s |
| Data download | 1800s+ |

---

## 4. Uploading Code

**Option A — rsync over ControlMaster** (recommended):
```bash
set -a; source .env; set +a
rsync -avz --progress \
    --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" \
    --exclude="results/" --exclude="data/" --exclude="cache/" \
    /local/path/to/project/ \
    ${PRAJNA_USER}@prajna.iitb.ac.in:~/project/
```
Or use `rsync_up()` from §0.3 which routes through the ControlMaster socket.

**Option B — paramiko SFTP** (for automated agents without ControlMaster):
```python
def upload_project(sftp, local_root, remote_root,
                   skip_top=None, skip_ext=None):
    """Upload local_root → remote_root. Skips top-level dirs in skip_top."""
    from pathlib import Path
    skip_top = skip_top or {"data","results","analysis","cache",".git","__pycache__",".env"}
    skip_ext = skip_ext or {".pyc", ".log"}

    local_root = Path(local_root)
    try: sftp.stat(remote_root)
    except FileNotFoundError: sftp.mkdir(remote_root)   # create root if first upload
    for p in sorted(local_root.rglob("*")):
        rel = p.relative_to(local_root)
        if rel.parts[0] in skip_top: continue      # top-level only, not subdirs
        if p.suffix in skip_ext: continue
        remote = f"{remote_root}/{rel.as_posix()}"
        if p.is_dir():
            try: sftp.stat(remote)
            except FileNotFoundError: sftp.mkdir(remote)
        else:
            sftp.put(str(p), remote)

# Usage (client from connect_prajna(); call prajna_setup(client) first to populate PRAJNA_GROUP)
sftp = client.open_sftp()
remote_home = f"/home/{os.environ['PRAJNA_GROUP']}/{os.environ['PRAJNA_USER']}"
upload_project(sftp, "/local/tn_explainability", remote_home + "/tn_explainability")
sftp.close()
```

---

## 5. Conda Environment Setup

On the login node (safe — pip install is allowed):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n myenv python=3.10
conda activate myenv
pip install --upgrade pip
pip install package1 package2 ...
```

Via paramiko (use `stream_command` — see Section 7 — for installs that take >1 min):
```python
CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"   # ~ expands correctly in remote shells
stream_command(client,
    f"source {CONDA_SH} && "
    "conda create -y -n myenv python=3.10 && "
    "conda activate myenv && "
    "pip install torch torchvision",
    timeout=1800)
```

---

## 6. Writing SLURM Job Scripts

```bash
#!/bin/bash
#SBATCH --partition=l40                     # partition name — see PRAJNA_HPC.md §4
#SBATCH --qos=l40                           # must match partition
#SBATCH --gres=gpu:1                        # GPU request; omit entirely for CPU-only jobs
#SBATCH --cpus-per-task=8                   # CPU cores
#SBATCH --mem=32G                           # total RAM (or --mem-per-cpu=4G)
#SBATCH --time=04:00:00                     # wall time HH:MM:SS
#SBATCH --job-name=myjob
#SBATCH --output=logs/myjob_%j.out   # relative to $HOME (run sbatch from $HOME)
#SBATCH --error=logs/myjob_%j.err

# IMPORTANT: -u breaks conda; use -eo only
set -eo pipefail

# IMPORTANT: must source conda.sh — ~/.bashrc is NOT sourced in SLURM jobs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

echo "Running on $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

cd $HOME/myproject
python train.py --arg value

echo "Done at $(date)"
```

Submit:
```bash
set -a; source .env; set +a
mkdir -p $HOME/logs            # create log dir FIRST
cd $HOME && /opt/slurm/bin/sbatch myjob.sh   # run from $HOME so relative log path resolves correctly
```

---

## 7. Streaming Long-Running Commands (paramiko)

Use this for anything that takes >10 seconds (conda install, data download, training).
Never use `stdout.read()` with a short timeout on these.

```python
import re, time

def stream_command(client, cmd, timeout=3600):
    """Run cmd and stream stdout. Returns (exit_code, full_output_str)."""
    transport = client.get_transport()
    chan = transport.open_session()
    chan.get_pty(width=200)      # allocate PTY so programs don't buffer output
    chan.exec_command(cmd)
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.recv_ready():
            raw = chan.recv(8192).decode(errors="replace")
            # strip ANSI escape codes for clean logging
            clean = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r', '', raw)
            print(clean, end="", flush=True)
            buf += clean
        if chan.exit_status_ready():
            # drain remaining output
            while chan.recv_ready():
                raw = chan.recv(8192).decode(errors="replace")
                clean = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r', '', raw)
                print(clean, end="", flush=True)
                buf += clean
            break
        time.sleep(0.2)
    else:
        print(f"\n[TIMEOUT after {timeout}s]")
    rc = chan.recv_exit_status() if chan.exit_status_ready() else -1
    return rc, buf
```

---

## 8. Submitting Jobs Programmatically

```python
import time

_load_env()   # from §0.1
USER    = os.environ["PRAJNA_USER"]
SBATCH   = "/opt/slurm/bin/sbatch"
SQUEUE   = "/opt/slurm/bin/squeue"
SCANCEL  = "/opt/slurm/bin/scancel"
SACCT    = "/opt/slurm/bin/sacct"
SACCTMGR = "/opt/slurm/bin/sacctmgr"

def submit_job(client, script_content):
    """Upload script via SFTP and sbatch it. Returns job ID string."""
    tmp = f"/tmp/job_{int(time.time()*1000)}.sh"
    # Write via SFTP — do NOT use "cat > file" over exec_command (stdin handling is fragile)
    sftp = client.open_sftp()
    with sftp.file(tmp, "wb") as f:
        f.write(script_content.encode())   # paramiko SFTPFile.write() requires bytes
    sftp.close()
    out, _, _ = run_paramiko(client, f"{SBATCH} --parsable {tmp}", timeout=30)
    return out.strip()  # job ID

def active_jobs(client, name_prefix=None):
    """Return list of names of pending+running jobs (optionally filtered)."""
    out, _, _ = run_paramiko(client, f"{SQUEUE} -u {USER} -h -o '%j'",
                    timeout=30, check=False)
    jobs = [j.strip() for j in out.splitlines() if j.strip()]
    if name_prefix:
        jobs = [j for j in jobs if j.startswith(name_prefix)]
    return jobs

def wait_until_done(client, name_prefix, poll_sec=60, label="jobs"):
    """Block until all jobs with name_prefix finish."""
    while True:
        n = len(active_jobs(client, name_prefix))
        if n == 0: break
        print(f"  [{label}] {n} still running...", flush=True)
        time.sleep(poll_sec)
    print(f"  [{label}] all done.")

def check_job_exit(client, job_id):
    """Return exit code of a completed job (None if still running)."""
    out, _, _ = run_paramiko(client,
        f"{SACCT} -j {job_id} --format=ExitCode --noheader",
        timeout=30, check=False)
    for line in out.splitlines():
        line = line.strip()
        if line:
            return int(line.split(":")[0])
    return None
```

---

## 9. Wave-Based Submission

When jobs > `MaxSubmitPU`, submit in batches and wait for slots to free:

```python
def submit_wave(client, scripts, partition, max_submit, label=""):
    """Submit scripts one by one, blocking when partition is at capacity."""
    job_ids = []
    for i, script in enumerate(scripts):
        while True:
            out, _, _ = run_paramiko(client,
                f"{SQUEUE} -u {USER} -h -p {partition} -o '%j'",
                timeout=30, check=False)
            n = len([x for x in out.splitlines() if x.strip()])
            if n < max_submit:
                break
            print(f"  [{partition}] at capacity ({n}/{max_submit}), waiting...")
            time.sleep(60)
        jid = submit_job(client, script)
        job_ids.append(jid)
        print(f"  [{partition}] submitted job {jid} ({i+1}/{len(scripts)}) {label}")
    return job_ids
```

> **Note:** Whether a job array counts as 1 or N toward `MaxSubmitPU` depends on SLURM
> version and site config. On Prajna, test with a small array first. If you hit
> `QOSMaxSubmitJobPerUserLimit`, use this wave pattern instead.

---

## 10. Complete End-to-End Automation Script

Paste, fill in the config block at the top, and run. Handles upload, conda env,
log dir, job submission, and polling. Run ControlMaster setup from §0.3 first.

Paste **all of §0.0, §0.1, §0.2** above the `_load_env()` call — in this exact order:
`_fetch_totp_secret_via_code` → `interactive_setup` → `_ENV_PATH = None` → `_load_env` → `_write_env_key` → `prajna_setup`

```python
#!/usr/bin/env python3
"""
Generic Prajna HPC pipeline runner.
Reads all credentials from .env — see .env.example for required variables.
Dependencies are auto-installed if missing (handles Kali/Debian pip restrictions).
"""
import os, re, subprocess, sys, time
from pathlib import Path

def _ensure_deps(*pkgs):
    import importlib
    for pkg in pkgs:
        mod = pkg.split("[")[0].replace("-", "_")
        try:
            importlib.import_module(mod); continue
        except ImportError:
            pass
        for cmd in [
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg],
            ["conda", "install", "-y", "-q", pkg],
        ]:
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                break
        else:
            raise SystemExit(f"Could not install '{pkg}'. See PRAJNA_RUNBOOK.md §1.0")

_ensure_deps("paramiko", "pyotp")
import paramiko, pyotp

# ── Paste §0.0 + §0.1 + §0.2 here (order matters — see section header above) ─
_load_env()

# ── CONFIG ───────────────────────────────────────────────────────────────────
HOST        = "prajna.iitb.ac.in"
USER        = os.environ["PRAJNA_USER"]
PASSWORD    = os.environ["PRAJNA_PASSWORD"]
TOTP_SECRET = os.environ["PRAJNA_TOTP_SECRET"]
LOCAL_ROOT  = Path("/local/path/to/project")    # ← edit this
CONDA_ENV   = "myenv"
PYTHON_VER  = "3.10"
PIP_PKGS    = "numpy pandas scikit-learn torch"
SKIP_TOP    = {"data","results","cache",".git","__pycache__",".env"}
# ── END CONFIG ───────────────────────────────────────────────────────────────

SQUEUE   = "/opt/slurm/bin/squeue"
SBATCH   = "/opt/slurm/bin/sbatch"
CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"

def _remote_home():
    """Derive home from group+user (both known after prajna_setup())."""
    return f"/home/{os.environ['PRAJNA_GROUP']}/{USER}"


def connect():
    """Connect with TOTP + password. Retries once on DISALLOW_REUSE/rate-limit."""
    def _try():
        transport = paramiko.Transport((HOST, 22))
        transport.connect()
        totp = pyotp.TOTP(TOTP_SECRET)
        def _handler(title, instructions, prompt_list):
            responses = []
            for prompt, echo in prompt_list:
                p = prompt.strip().lower()
                if "verification" in p or "code" in p or "authenticat" in p:
                    responses.append(totp.now())
                else:
                    responses.append(PASSWORD)
            return responses
        transport.auth_interactive(USER, _handler)
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c._transport = transport
        return c

    try:
        return _try()
    except paramiko.AuthenticationException:
        print("[prajna] TOTP failed — waiting 31s for DISALLOW_REUSE window to roll over...")
        time.sleep(31)
        return _try()   # raises if still failing — check clock and TOTP app


def run(client, cmd, timeout=60, check=True):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"FAILED (rc={rc}): {cmd}\n{err}")
    return out, err, rc

run_paramiko = run   # prajna_setup (pasted from §0.2) calls run_paramiko


def stream(client, cmd, timeout=3600):
    transport = client.get_transport()
    chan = transport.open_session()
    chan.get_pty(width=200)
    chan.exec_command(cmd)
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.recv_ready():
            raw = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r','',
                         chan.recv(8192).decode(errors="replace"))
            print(raw, end="", flush=True); buf += raw
        if chan.exit_status_ready():
            while chan.recv_ready():
                raw = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r','',
                             chan.recv(8192).decode(errors="replace"))
                print(raw, end="", flush=True); buf += raw
            break
        time.sleep(0.2)
    rc = chan.recv_exit_status() if chan.exit_status_ready() else -1
    return rc, buf


def upload(client, remote_root):
    sftp = client.open_sftp()
    try: sftp.stat(remote_root)
    except FileNotFoundError: sftp.mkdir(remote_root)   # create root if first upload
    total = 0
    for p in sorted(LOCAL_ROOT.rglob("*")):
        rel = p.relative_to(LOCAL_ROOT)
        if rel.parts[0] in SKIP_TOP: continue
        if p.suffix in {".pyc",".log"}: continue
        remote = f"{remote_root}/{rel.as_posix()}"
        if p.is_dir():
            try: sftp.stat(remote)
            except FileNotFoundError: sftp.mkdir(remote)
        else:
            sftp.put(str(p), remote); total += 1
    sftp.close()
    print(f"Uploaded {total} files.")


def ensure_env(client):
    out, _, _ = run(client,
        f"source {CONDA_SH} && conda env list", timeout=120, check=False)
    if CONDA_ENV not in out:
        print(f"Creating conda env '{CONDA_ENV}'...")
        rc2, _ = stream(client,
            f"source {CONDA_SH} && conda create -y -n {CONDA_ENV} python={PYTHON_VER} "
            f"&& conda activate {CONDA_ENV} && pip install {PIP_PKGS}",
            timeout=1800)
        if rc2 != 0: raise RuntimeError("Conda env creation failed.")
    else:
        print(f"Conda env '{CONDA_ENV}' already exists.")


def submit_job(client, script_content):
    tmp = f"/tmp/job_{int(time.time()*1000)}.sh"
    sftp = client.open_sftp()
    with sftp.file(tmp, "wb") as f: f.write(script_content.encode())
    sftp.close()
    out, _, _ = run(client, f"{SBATCH} --parsable {tmp}", timeout=30)
    return out.strip()


def n_jobs(client, partition=None):
    cmd = f"{SQUEUE} -u {USER} -h -o '%j'"
    if partition: cmd += f" -p {partition}"
    out, _, _ = run(client, cmd, timeout=30, check=False)
    return len([x for x in out.splitlines() if x.strip()])


def main():
    # 1. Connect (TOTP + password — 2FA required since 2026-04-28)
    print("Connecting...")
    client = connect()

    # 2. First-run setup: fills PRAJNA_GROUP/UID/GID/SCRATCH into .env if missing
    prajna_setup(client)   # no-op if already populated

    # 3. Derive home from group + user (no .env redundancy)
    remote_home = _remote_home()
    remote_root = f"{remote_home}/project"   # ← edit project subdir as needed

    # 4. Upload
    print("Uploading project...")
    upload(client, remote_root)

    # 5. Conda env
    print("Checking conda env...")
    ensure_env(client)

    # 6. Create log dir
    run(client, f"mkdir -p {remote_home}/logs", timeout=15)

    # 7. Submit jobs (edit this section for your specific jobs)
    email_line = (f"#SBATCH --mail-type=BEGIN,END,FAIL\n"
                  f"#SBATCH --mail-user={os.environ['PRAJNA_EMAIL']}")  \
                 if os.environ.get("PRAJNA_EMAIL") else ""
    print("Submitting jobs...")
    job_script = f"""#!/bin/bash
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=myjob
#SBATCH --output={remote_home}/logs/myjob_%j.out
#SBATCH --error={remote_home}/logs/myjob_%j.err
{email_line}
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {CONDA_ENV}
cd {remote_root}
python train.py
"""
    jid = submit_job(client, job_script)
    print(f"Submitted job {jid}")

    # 8. Poll until done
    print("Waiting for jobs to complete...")
    while True:
        n = n_jobs(client)
        if n == 0: break
        print(f"  {n} job(s) running...")
        time.sleep(60)

    print("All done.")
    client.close()


if __name__ == "__main__":
    main()
```

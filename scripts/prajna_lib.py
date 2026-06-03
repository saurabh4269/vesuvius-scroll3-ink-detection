"""
prajna_lib.py — Prajna HPC automation library.

Bundled with the prajna-hpc skill. Copy this file into your project,
then: from prajna_lib import connect_prajna, run, stream, submit_job, ...

Handles:
  - .env loading + first-time interactive setup
  - SSH ControlMaster subprocess pattern (preferred)
  - paramiko + pyotp fallback (headless)
  - SLURM job submission, monitoring, wave scheduling
  - SFTP file upload
  - Conda env check/create

Requirements: paramiko, pyotp  (auto-installed if missing)
"""

import importlib, os, re, subprocess, sys, time
from pathlib import Path

# ── Dependency installer (handles Kali/Debian PEP 668 restriction) ──────────

def _ensure_deps(*pkgs):
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
            raise SystemExit(
                f"Could not install '{pkg}'.\n"
                f"Try: pip install --break-system-packages {pkg}\n"
                f"Or in a venv: python3 -m venv .venv && source .venv/bin/activate"
            )

_ensure_deps("paramiko", "pyotp")
import paramiko, pyotp

# ── Global state ─────────────────────────────────────────────────────────────

HOST = "prajna.iitb.ac.in"
_ENV_PATH = None

# ── First-time wizard ────────────────────────────────────────────────────────

def _fetch_totp_secret_via_code(user, password, totp_code):
    """One-shot login with a live 6-digit code; reads TOTP secret from server."""
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
            "TOTP codes expire every 30s — get a fresh code and retry."
        )
    ch = t.open_session()
    ch.exec_command("head -1 ~/.google_authenticator")
    secret = ch.makefile().read().strip()
    t.close()
    return secret if len(secret) > 10 else ""


def interactive_setup(env_dir="."):
    """Ask minimum questions, log in, write a complete .env. Call when .env is missing."""
    import getpass
    print("[prajna] .env not found — first-time setup")
    user     = input("  Username: ").strip()
    password = getpass.getpass("  Password: ").strip()
    secret_raw = input(
        "  TOTP secret key (base32 from GA setup — leave blank if you don't have it): "
    ).strip()

    if secret_raw:
        secret = secret_raw
    else:
        code = input("  Current 6-digit code from Google Authenticator: ").strip()
        print("  [prajna] Logging in once to read TOTP secret from server...")
        secret = _fetch_totp_secret_via_code(user, password, code)
        if not secret:
            raise SystemExit(
                "[prajna] Could not read TOTP secret from server.\n"
                "Log in interactively, run `head -1 ~/.google_authenticator`, "
                "and paste that string as PRAJNA_TOTP_SECRET in .env."
            )
        print("  [prajna] Got TOTP secret — storing.")

    email = input("  Email for SLURM job notifications (Enter to skip): ").strip()

    env_path = Path(env_dir) / ".env"
    env_path.write_text(
        "# Prajna HPC credentials — managed by prajna_lib.py\n"
        f"PRAJNA_USER={user}\n"
        f"PRAJNA_PASSWORD={password}\n"
        f"PRAJNA_TOTP_SECRET={secret}\n"
        f"PRAJNA_EMAIL={email}\n"
        "PRAJNA_GROUP=\nPRAJNA_UID=\nPRAJNA_GID=\n"
        "PRAJNA_SCRATCH_1=\nPRAJNA_SCRATCH_2=\nPRAJNA_SCRATCH_3=\n"
        "PRAJNA_SCRATCH_4=\nPRAJNA_SCRATCH_5=\n"
    )
    print(f"  [prajna] .env written → {env_path.resolve()}")


# ── Load credentials ─────────────────────────────────────────────────────────

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
    interactive_setup(search[0])
    _load_env(search)


_load_env()

USER        = os.environ["PRAJNA_USER"]
PASSWORD    = os.environ["PRAJNA_PASSWORD"]
TOTP_SECRET = os.environ["PRAJNA_TOTP_SECRET"]


def _home():
    """Derive home dir from group + user. Valid after prajna_setup() fills PRAJNA_GROUP."""
    g = os.environ.get("PRAJNA_GROUP", "")
    return f"/home/{g}/{USER}" if g else None


# ── Write key back to .env ───────────────────────────────────────────────────

def _write_env_key(key, value):
    if _ENV_PATH is None:
        return
    text = _ENV_PATH.read_text()
    pat = re.compile(rf"^({re.escape(key)}=).*$", re.MULTILINE)
    if pat.search(text):
        text = pat.sub(rf"\g<1>{value}", text)
    else:
        text = text.rstrip("\n") + f"\n{key}={value}\n"
    _ENV_PATH.write_text(text)
    os.environ[key] = value


# ── Server info — fill .env from server ─────────────────────────────────────

def prajna_setup(client):
    """Fetch group/uid/gid/scratch codes from server; write into .env. No-op for set fields."""
    def _r(cmd):
        return run_paramiko(client, cmd, timeout=15)[0]

    if not os.environ.get("PRAJNA_GROUP"): _write_env_key("PRAJNA_GROUP", _r("id -gn"))
    if not os.environ.get("PRAJNA_UID"):   _write_env_key("PRAJNA_UID",   _r("id -u"))
    if not os.environ.get("PRAJNA_GID"):   _write_env_key("PRAJNA_GID",   _r("id -g"))

    if any(not os.environ.get(f"PRAJNA_SCRATCH_{i}") for i in range(1, 6)):
        ga, _, _ = run_paramiko(client, "cat ~/.google_authenticator", timeout=15, check=False)
        codes = [l.strip() for l in ga.splitlines()
                 if l.strip().isdigit() and len(l.strip()) == 8]
        for i, code in enumerate(codes[:5], 1):
            if not os.environ.get(f"PRAJNA_SCRATCH_{i}"):
                _write_env_key(f"PRAJNA_SCRATCH_{i}", code)

    print(f"[prajna_setup] group={os.environ.get('PRAJNA_GROUP')} uid={os.environ.get('PRAJNA_UID')}")


# ── paramiko connection (fallback for headless / no ControlMaster) ───────────

def _do_connect(password, totp_secret):
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
    """Connect to Prajna with TOTP + password. Auto-retries once on DISALLOW_REUSE."""
    password    = password    or os.environ["PRAJNA_PASSWORD"]
    totp_secret = totp_secret or os.environ["PRAJNA_TOTP_SECRET"]
    try:
        return _do_connect(password, totp_secret)
    except paramiko.AuthenticationException:
        print("[prajna] TOTP failed — waiting 31s for DISALLOW_REUSE window to roll over...")
        time.sleep(31)
        try:
            return _do_connect(password, totp_secret)
        except paramiko.AuthenticationException:
            raise RuntimeError(
                "[prajna] Auth failed after retry.\n"
                "Checklist:\n"
                "  1. Is your TOTP app showing the current code? Wait for a fresh one.\n"
                "  2. Is your local clock correct? Run: date\n"
                "     Sync: sudo timedatectl set-ntp true\n"
                "  3. Have you used this code in the last 30s? Wait 31s.\n"
                "\nNEVER use scratch codes automatically — user must approve."
            )


def reconnect():
    """Drop-in reconnect after SSHException or channel timeout."""
    return connect_prajna()


# ── SSH ControlMaster subprocess helpers (preferred over paramiko) ────────────

CTRL_SOCKET = str(Path.home() / f".ssh/ctl/{USER}@{HOST}:22")
SSH_OPTS    = ["-o", f"ControlPath={CTRL_SOCKET}", "-o", "ControlMaster=no",
               "-o", "BatchMode=yes"]


def check_controlmaster():
    """Return True if ControlMaster socket is alive."""
    r = subprocess.run(["ssh", "-O", "check"] + SSH_OPTS + [f"{USER}@{HOST}"],
                       capture_output=True)
    return r.returncode == 0


def reestablish_master():
    """Kill stale ControlMaster socket and re-authenticate automatically.

    Call this after laptop sleep/lock/suspend — the socket goes stale on suspend
    and all subsequent ssh/scp/rsync calls hang silently.
    Requires pexpect: pip install pexpect
    """
    try:
        import pexpect
    except ImportError:
        _ensure_deps("pexpect")
        import pexpect

    # Kill stale socket — ignore error if already dead
    subprocess.run(
        ["ssh", "-O", "exit", "-o", f"ControlPath={CTRL_SOCKET}", f"{USER}@{HOST}"],
        capture_output=True
    )
    time.sleep(0.5)

    # Spawn new ControlMaster, respond to TOTP + password prompts
    child = pexpect.spawn(
        f"ssh -M -N -f "
        f"-o ControlPath={CTRL_SOCKET} "
        f"-o ControlPersist=yes "
        f"-o StrictHostKeyChecking=no "
        f"{USER}@{HOST}",
        timeout=30
    )
    child.expect("Verification code:")
    child.sendline(pyotp.TOTP(TOTP_SECRET).now())
    child.expect("Password:")
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF)
    time.sleep(2)   # let master start before using socket
    print("[prajna] ControlMaster re-established.")


def ensure_master():
    """Check ControlMaster; re-establish automatically if stale. Returns True if live."""
    if check_controlmaster():
        return True
    print("[prajna] ControlMaster socket stale — re-establishing...")
    reestablish_master()
    return check_controlmaster()


def run_ssh(cmd, timeout=60, check=True):
    """Run cmd on Prajna via ControlMaster socket. Zero auth if master is alive."""
    r = subprocess.run(
        ["ssh"] + SSH_OPTS + [f"{USER}@{HOST}", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd}\n{r.stderr}")
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def upload_scp(local_path, remote_path):
    """Upload file or directory via scp over ControlMaster socket."""
    subprocess.run(
        ["scp"] + SSH_OPTS + ["-r", str(local_path), f"{USER}@{HOST}:{remote_path}"],
        check=True
    )


def rsync_up(local_root, remote_root,
             excludes=("__pycache__", "*.pyc", "results/", "data/", "cache/")):
    """Incremental upload via rsync over ControlMaster socket."""
    excl = []
    for e in excludes:
        excl += ["--exclude", e]
    subprocess.run(
        ["rsync", "-avz", "--progress"] + excl
        + ["-e", f"ssh {' '.join(SSH_OPTS)}",
           str(local_root) + "/", f"{USER}@{HOST}:{remote_root}/"],
        check=True
    )


# ── paramiko helpers ─────────────────────────────────────────────────────────

def run_paramiko(client, cmd, timeout=60, check=True):
    """Run cmd via paramiko, return (stdout_str, stderr_str, exit_code). Raises on non-zero if check."""
    # Pass timeout=None to exec_command — lets the channel block without a
    # socket-level I/O timeout. The TOTP retry delay can exceed 30s and causes
    # PipeTimeout when a channel timeout is set. Overall command correctness
    # is guarded by recv_exit_status() which blocks until the process exits.
    _, stdout, stderr = client.exec_command(cmd, timeout=None)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\n{err}")
    return out, err, rc


def run(client, cmd, timeout=60, check=True):
    """Alias for run_paramiko — returns (stdout_str, stderr_str, exit_code)."""
    return run_paramiko(client, cmd, timeout=timeout, check=check)


def stream_command(client, cmd, timeout=3600):
    """Run cmd and stream stdout live. Use for anything that takes >10s. Returns (rc, output)."""
    chan = client.get_transport().open_session()
    chan.get_pty(width=200)
    chan.exec_command(cmd)
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.recv_ready():
            raw = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r', '',
                         chan.recv(8192).decode(errors="replace"))
            print(raw, end="", flush=True)
            buf += raw
        if chan.exit_status_ready():
            while chan.recv_ready():
                raw = re.sub(r'\x1b\[[0-9;]*[mK]|\x1b\(B|\r', '',
                             chan.recv(8192).decode(errors="replace"))
                print(raw, end="", flush=True)
                buf += raw
            break
        time.sleep(0.2)
    else:
        print(f"\n[TIMEOUT after {timeout}s]")
    rc = chan.recv_exit_status() if chan.exit_status_ready() else -1
    return rc, buf


# ── File upload (SFTP) ───────────────────────────────────────────────────────

def upload_project(sftp, local_root, remote_root, skip_top=None, skip_ext=None):
    """Upload local_root → remote_root via SFTP. Only skips top-level dirs in skip_top."""
    skip_top = skip_top or {"data", "results", "analysis", "cache", ".git",
                            "__pycache__", ".env"}
    skip_ext = skip_ext or {".pyc", ".log"}
    local_root = Path(local_root)
    try:
        sftp.stat(remote_root)
    except FileNotFoundError:
        sftp.mkdir(remote_root)
    total = 0
    for p in sorted(local_root.rglob("*")):
        rel = p.relative_to(local_root)
        if rel.parts[0] in skip_top:
            continue
        if p.suffix in skip_ext:
            continue
        remote = f"{remote_root}/{rel.as_posix()}"
        if p.is_dir():
            try:
                sftp.stat(remote)
            except FileNotFoundError:
                sftp.mkdir(remote)
        else:
            sftp.put(str(p), remote)
            total += 1
    print(f"Uploaded {total} files to {remote_root}")


# ── Conda env ────────────────────────────────────────────────────────────────

CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"


def ensure_env(client, env_name, python_ver="3.10", pip_pkgs=""):
    """Check if conda env exists; create and pip-install packages if not."""
    out, _, _ = run_paramiko(client,
                             f"source {CONDA_SH} && conda env list",
                             timeout=120, check=False)
    if env_name in out:
        print(f"Conda env '{env_name}' already exists.")
        return
    print(f"Creating conda env '{env_name}'...")
    cmd = f"source {CONDA_SH} && conda create -y -n {env_name} python={python_ver}"
    if pip_pkgs:
        cmd += f" && conda activate {env_name} && pip install {pip_pkgs}"
    rc, _ = stream_command(client, cmd, timeout=1800)
    if rc != 0:
        raise RuntimeError(f"Conda env creation failed (rc={rc}).")
    print(f"Conda env '{env_name}' ready.")


# ── SLURM helpers ────────────────────────────────────────────────────────────

SBATCH    = "/opt/slurm/bin/sbatch"
SQUEUE    = "/opt/slurm/bin/squeue"
SCANCEL   = "/opt/slurm/bin/scancel"
SACCT     = "/opt/slurm/bin/sacct"
SCONTROL  = "/opt/slurm/bin/scontrol"
SACCTMGR  = "/opt/slurm/bin/sacctmgr"


def submit_job(client, script_content):
    """Write script via SFTP and sbatch it. Returns job ID string."""
    tmp = f"/tmp/job_{int(time.time() * 1000)}.sh"
    sftp = client.open_sftp()
    with sftp.file(tmp, "wb") as f:
        f.write(script_content.encode())
    sftp.close()
    out, _, _ = run_paramiko(client, f"{SBATCH} --parsable {tmp}", timeout=30)
    return out.strip()


def active_jobs(client, partition=None, name_prefix=None):
    """Return names of pending + running jobs, optionally filtered."""
    cmd = f"{SQUEUE} -u {USER} -h -o '%j'"
    if partition:
        cmd += f" -p {partition}"
    out, _, _ = run_paramiko(client, cmd, timeout=30, check=False)
    jobs = [j.strip() for j in out.splitlines() if j.strip()]
    if name_prefix:
        jobs = [j for j in jobs if j.startswith(name_prefix)]
    return jobs


def wait_until_done(client, name_prefix=None, partition=None,
                    poll_sec=60, label="jobs"):
    """Block until all matching jobs complete, polling every poll_sec seconds."""
    while True:
        n = len(active_jobs(client, partition=partition, name_prefix=name_prefix))
        if n == 0:
            break
        print(f"  [{label}] {n} still running...", flush=True)
        time.sleep(poll_sec)
    print(f"  [{label}] all done.")


def check_job_exit(client, job_id):
    """Return exit code of a completed job (None if still running)."""
    out, _, _ = run_paramiko(
        client,
        f"{SACCT} -j {job_id} --format=ExitCode --noheader",
        timeout=30, check=False
    )
    for line in out.splitlines():
        line = line.strip()
        if line:
            return int(line.split(":")[0])
    return None


def submit_wave(client, scripts, partition, max_submit, label=""):
    """Submit scripts in waves, blocking when partition is at QOS capacity."""
    job_ids = []
    for i, script in enumerate(scripts):
        while True:
            n = len(active_jobs(client, partition=partition))
            if n < max_submit:
                break
            print(f"  [{partition}] at capacity ({n}/{max_submit}), waiting 60s...")
            time.sleep(60)
        jid = submit_job(client, script)
        job_ids.append(jid)
        print(f"  [{partition}] job {jid} submitted ({i + 1}/{len(scripts)}) {label}")
    return job_ids

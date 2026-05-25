from __future__ import annotations

import asyncio
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from http.client import HTTPException
import json
import os
import random
import signal
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import TaskOS

DEFAULT_SHARED_DIR = Path("/wbl-fast/usrs/trost/harbor-slurm-pyxis")
DEFAULT_SQSH_CACHE = Path("/wbl-fast/usrs/trost/tbench-sqsh-cache/images")
DEFAULT_TAR_CACHE = Path("/wbl-fast/usrs/ee/agent-collab/docker-image-cache")
DEFAULT_ENROOT_SYSCONF = Path("/etc/enroot")
DEFAULT_HOST_PYTHON_PREFIX = Path(sys.prefix).resolve()
HOST_PYTHON_MOUNT = "/opt/harbor-python"
DEFAULT_STARTUP_TIMEOUT_SEC = 12 * 60 * 60
DEFAULT_HEALTH_TIMEOUT_SEC = 20 * 60
DEFAULT_STARTUP_RETRIES = 3
DEFAULT_STARTUP_RETRY_DELAY_SEC = 60
DEFAULT_EXEC_REQUEST_GRACE_SEC = 120
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 5
DEFAULT_SCANCEL_TIMEOUT_SEC = 5
DEFAULT_POST_TIMEOUT_GRACE_SEC = 5
DEFAULT_VERIFIER_REWARD_SETTLE_SEC = 90.0
DEFAULT_IO_WORKERS = 512
_TRANSIENT_STARTUP_ERRORS = (
    "node failure",
    "still not ready",
    "something is wrong with the boot",
)
_IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("HARBOR_SLURM_PYXIS_IO_WORKERS", DEFAULT_IO_WORKERS)),
    thread_name_prefix="slurm-pyxis-io",
)


@dataclass(frozen=True)
class UnifiedExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int | None
    chunk_id: str
    wall_time_seconds: float
    session_id: int | None = None
    original_token_count: int | None = None


async def _run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_IO_EXECUTOR, partial(func, *args))


SLURM_BOOTSTRAP = """#!/bin/bash
export WORKDIR="${1:-/app}"; shift
export HARBOR_STAGING="/staging/env_files"
export DEBIAN_FRONTEND=noninteractive
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
mkdir -p "$WORKDIR"

_unmount_localtime() {
  if awk '$5 == "/etc/localtime" { found=1 } END { exit found ? 0 : 1 }' /proc/self/mountinfo 2>/dev/null; then
    umount /etc/localtime 2>/dev/null || true
  fi
}
_unmount_localtime

_PY=
for candidate in /usr/bin/python3 /usr/local/bin/python3 /opt/harbor-python/bin/python /opt/harbor-python/bin/python3; do
  if [ -x "$candidate" ]; then
    _PY="$candidate"
    break
  fi
done
if [ -z "$_PY" ]; then
  echo "[harbor] FATAL: no Python runtime available for the Slurm/Pyxis exec server" >&2
  exit 1
fi

if [ -f "$HARBOR_STAGING/setup.sh" ]; then
  echo "[harbor] Running task setup.sh..." >&2
  source "$HARBOR_STAGING/setup.sh"
fi

exec "$_PY" "$@"
"""
STDLIB_EXEC_SERVER = r"""import argparse
import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_SESSION_LOCK = threading.Lock()
_SESSIONS = {}
_NEXT_SESSION_ID = 1
_DEFAULT_YIELD_TIME_MS = 10000
_DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250
_MIN_EMPTY_WRITE_STDIN_YIELD_TIME_MS = 5000
_DEFAULT_MAX_OUTPUT_TOKENS = 10000
_VERIFY_SETTLE_SEC = float(os.environ.get("HARBOR_SLURM_PYXIS_VERIFY_SETTLE_SEC", "90"))
_VERIFY_SETTLE_MARKERS = ("/verifier/", "test-stdout", "reward.txt", "reward.json", "ctrf.json")


def _setup(workdir):
    os.makedirs(workdir, exist_ok=True)
    os.environ["SINGULARITY_WORKDIR"] = workdir
    for path in (
        "/etc/apt/apt.conf.d",
        "/var/lib/apt/lists/partial",
        "/var/cache/apt/archives/partial",
        "/tmp",
        "/var/tmp",
        "/root/.cache",
        "/root/.local/bin",
        "/usr/local/bin",
    ):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
    _setup_dpkg()
    _fake_sudo()


def _setup_dpkg():
    cfg = "/etc/dpkg/dpkg.cfg.d"
    if not os.path.isdir("/etc/dpkg"):
        return
    try:
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "harbor-pyxis"), "w", encoding="utf-8") as f:
            f.write("force-overwrite\nforce-overwrite-dir\nforce-unsafe-io\n")
    except OSError:
        pass


def _fake_sudo():
    path = "/usr/local/bin/sudo"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexec \"$@\"\n")
        os.chmod(path, 0o755)
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        data = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/shutdown":
            self._send(200, {"message": "shutdown initiated"})
            threading.Thread(target=_exit_later, daemon=True).start()
        elif self.path == "/exec":
            self._send(200, _exec(data))
        elif self.path == "/exec_command":
            self._send(200, _exec_command(data))
        elif self.path == "/write_stdin":
            self._send(200, _write_stdin(data))
        else:
            self._send(404, {"error": "not found"})


def _exec(data):
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/usr/local/bin:" + env.get("PATH", "/bin")
    env.update(data.get("env") or {})
    cwd = data.get("cwd") or os.environ.get("SINGULARITY_WORKDIR", "/app")
    try:
        result = subprocess.run(
            data["command"],
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=data.get("timeout_sec"),
        )
        _settle_verifier_mount(data["command"])
        return {"stdout": (result.stdout or "").strip(), "stderr": None, "return_code": result.returncode}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {
            "stdout": output.strip(),
            "stderr": None,
            "return_code": 124,
            "timeout": True,
            "timeout_sec": data.get("timeout_sec"),
        }


def _settle_verifier_mount(command):
    if _VERIFY_SETTLE_SEC <= 0:
        return
    if not any(marker in command for marker in _VERIFY_SETTLE_MARKERS):
        return
    try:
        os.sync()
    except Exception:
        pass
    time.sleep(_VERIFY_SETTLE_SEC)


def _exec_command(data):
    env = _exec_env(data)
    cwd = data.get("cwd") or os.environ.get("SINGULARITY_WORKDIR", "/app")
    command = data["command"]
    shell = data.get("shell") or "/bin/bash"
    login = bool(data.get("login"))
    yield_time_ms = int(data.get("yield_time_ms") or _DEFAULT_YIELD_TIME_MS)
    max_output_tokens = _optional_int(data.get("max_output_tokens"))
    timeout_sec = _optional_float(data.get("timeout_sec"))
    tty = bool(data.get("tty"))
    start = time.monotonic()
    try:
        session = _spawn_session(command, cwd, env, shell, login, tty, timeout_sec)
    except Exception as exc:
        return _response(
            output=f"exec_command failed: {exc}",
            wall_time=time.monotonic() - start,
            exit_code=1,
            max_output_tokens=max_output_tokens,
        )
    output = _collect_until(session, time.monotonic() + yield_time_ms / 1000.0)
    exit_code = _session_exit_code(session)
    session_id = None
    if exit_code is None:
        session_id = _store_session(session)
    else:
        _close_session(session)
    return _response(
        output=output,
        wall_time=time.monotonic() - start,
        exit_code=exit_code,
        session_id=session_id,
        max_output_tokens=max_output_tokens,
    )


def _write_stdin(data):
    session_id = int(data["session_id"])
    chars = data.get("chars") or ""
    yield_time_ms = int(data.get("yield_time_ms") or _DEFAULT_WRITE_STDIN_YIELD_TIME_MS)
    if not chars:
        yield_time_ms = max(yield_time_ms, _MIN_EMPTY_WRITE_STDIN_YIELD_TIME_MS)
    max_output_tokens = _optional_int(data.get("max_output_tokens"))
    start = time.monotonic()
    with _SESSION_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        return _response(
            output=f"write_stdin failed: unknown session_id {session_id}",
            wall_time=time.monotonic() - start,
            exit_code=1,
            max_output_tokens=max_output_tokens,
        )
    if chars:
        try:
            os.write(session.stdin_fd, chars.encode("utf-8"))
        except OSError as exc:
            return _response(
                output=f"write_stdin failed: {exc}",
                wall_time=time.monotonic() - start,
                exit_code=1,
                max_output_tokens=max_output_tokens,
            )
    output = _collect_until(session, time.monotonic() + yield_time_ms / 1000.0)
    exit_code = _session_exit_code(session)
    response_session_id = session_id if exit_code is None else None
    if exit_code is not None:
        with _SESSION_LOCK:
            _SESSIONS.pop(session_id, None)
        _close_session(session)
    return _response(
        output=output,
        wall_time=time.monotonic() - start,
        exit_code=exit_code,
        session_id=response_session_id,
        max_output_tokens=max_output_tokens,
    )


def _exec_env(data):
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/usr/local/bin:" + env.get("PATH", "/bin")
    env.update(data.get("env") or {})
    return env


class _Session:
    def __init__(self, process, output_fd, stdin_fd, timeout_at=None, stdout=None, stdin=None):
        self.process = process
        self.output_fd = output_fd
        self.stdin_fd = stdin_fd
        self.timeout_at = timeout_at
        self.stdout = stdout
        self.stdin = stdin


def _spawn_session(command, cwd, env, shell, login, tty, timeout_sec):
    timeout_at = time.monotonic() + timeout_sec if timeout_sec else None
    argv = [shell, "-lc" if login else "-c", command]
    if tty:
        master_fd, slave_fd = pty.openpty()
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        _set_nonblocking(master_fd)
        return _Session(process, master_fd, master_fd, timeout_at)
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    output_fd = process.stdout.fileno()
    stdin_fd = process.stdin.fileno()
    _set_nonblocking(output_fd)
    return _Session(process, output_fd, stdin_fd, timeout_at, process.stdout, process.stdin)


def _set_nonblocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _store_session(session):
    global _NEXT_SESSION_ID
    with _SESSION_LOCK:
        session_id = _NEXT_SESSION_ID
        _NEXT_SESSION_ID += 1
        _SESSIONS[session_id] = session
        return session_id


def _collect_until(session, deadline):
    if session.timeout_at is not None:
        deadline = min(deadline, session.timeout_at)
    chunks = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, _, _ = select.select([session.output_fd], [], [], min(remaining, 0.1))
        if readable:
            chunks.extend(_drain_fd(session.output_fd))
            continue
        if session.process.poll() is not None:
            chunks.extend(_drain_fd(session.output_fd))
            break
    return b"".join(chunks).decode("utf-8", "replace")


def _session_exit_code(session):
    exit_code = session.process.poll()
    if exit_code is not None:
        return exit_code
    if session.timeout_at is None or time.monotonic() < session.timeout_at:
        return None
    try:
        os.killpg(session.process.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        session.process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(session.process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            session.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    return 124


def _drain_fd(fd):
    chunks = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return chunks


def _close_session(session):
    if session.stdout is not None or session.stdin is not None:
        for stream in {session.stdout, session.stdin}:
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass
        return
    for fd in {session.output_fd, session.stdin_fd}:
        try:
            os.close(fd)
        except OSError:
            pass


def _terminate_sessions():
    with _SESSION_LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        if session.process.poll() is None:
            try:
                os.killpg(session.process.pid, signal.SIGTERM)
            except OSError:
                pass
        _close_session(session)


def _optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _response(output, wall_time, exit_code=None, session_id=None, max_output_tokens=None):
    original_token_count = _approx_token_count(output)
    truncated = _truncate_output(output, max_output_tokens)
    payload = {
        "chunk_id": _chunk_id(),
        "wall_time_seconds": wall_time,
        "output": truncated,
        "original_token_count": original_token_count,
        "stdout": truncated,
        "stderr": None,
        "return_code": exit_code if exit_code is not None else 0,
        "exit_code": exit_code,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def _truncate_output(output, max_output_tokens):
    token_limit = max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS
    char_limit = max(1, int(token_limit) * 4)
    if len(output) <= char_limit:
        return output
    omitted = len(output) - char_limit
    return f"<omitted {omitted} chars>\n" + output[-char_limit:]


def _approx_token_count(text):
    return len(re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE))


def _chunk_id():
    return f"{int(time.time() * 1000000) & 0xffffff:06x}"


def _exit_later():
    _terminate_sessions()
    time.sleep(0.1)
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workdir", default="/app")
    args = parser.parse_args()
    _setup(args.workdir)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    port = server.server_address[1]
    print(f"__HARBOR_PYXIS_READY__{socket.gethostname()}:{port}", flush=True)
    print(f"[harbor] stdlib exec server listening on {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
"""


class SlurmPyxisEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        sqsh_cache_dir: str | Path = DEFAULT_SQSH_CACHE,
        docker_tar_cache_dir: str | Path = DEFAULT_TAR_CACHE,
        shared_dir: str | Path = DEFAULT_SHARED_DIR,
        slurm_partition: str = "m7i-cpu",
        slurm_time: str = "02:00:00",
        startup_timeout_sec: int | str = DEFAULT_STARTUP_TIMEOUT_SEC,
        health_timeout_sec: int | str = DEFAULT_HEALTH_TIMEOUT_SEC,
        startup_retries: int | str = DEFAULT_STARTUP_RETRIES,
        startup_retry_delay_sec: int | str = DEFAULT_STARTUP_RETRY_DELAY_SEC,
        startup_parallelism: int | str | None = None,
        host_python_prefix: str | Path | None = DEFAULT_HOST_PYTHON_PREFIX,
        remap_root: bool = True,
        verifier_reward_settle_sec: int | float | str = DEFAULT_VERIFIER_REWARD_SETTLE_SEC,
        **kwargs,
    ):
        self._sqsh_cache_dir = Path(sqsh_cache_dir)
        self._docker_tar_cache_dir = Path(docker_tar_cache_dir)
        self._shared_dir = Path(shared_dir)
        self._slurm_partition = slurm_partition
        self._slurm_time = slurm_time
        self._startup_timeout_sec = int(startup_timeout_sec)
        self._health_timeout_sec = int(health_timeout_sec)
        self._startup_retries = max(1, int(startup_retries))
        self._startup_retry_delay_sec = max(0, int(startup_retry_delay_sec))
        self._exec_request_timeout_sec = (
            _slurm_time_to_seconds(self._slurm_time) + DEFAULT_EXEC_REQUEST_GRACE_SEC
        )
        # Kept for old Harbor configs; Slurm now owns startup queuing.
        _ = startup_parallelism
        self._host_python_prefix = (
            Path(host_python_prefix).resolve() if host_python_prefix else None
        )
        self._remap_root = remap_root
        self._verifier_reward_settle_sec = max(0.0, float(verifier_reward_settle_sec))
        self._process: asyncio.subprocess.Process | None = None
        self._stream_task: asyncio.Task | None = None
        self._node: str | None = None
        self._port = 0
        self._slurm_job_name = f"hb-{os.getpid()}-{random.randint(100000, 999999)}"
        self._staging_dir: Path | None = None
        self._enroot_sysconf_dir: Path | None = None
        self._recent_srun_output: list[str] = []
        self._workdir = "/app"
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "slurm-pyxis"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        if self.task_env_config.os == TaskOS.WINDOWS:
            raise RuntimeError("Slurm/Pyxis only supports Linux task containers")
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise RuntimeError("Slurm/Pyxis backend currently supports one-container tasks only")
        if not self.task_env_config.docker_image:
            raise RuntimeError("Slurm/Pyxis backend requires [environment].docker_image")

    @classmethod
    def preflight(cls) -> None:
        missing = [name for name in ("srun", "enroot") if shutil.which(name) is None]
        if missing:
            raise SystemExit(f"Missing Slurm/Pyxis dependency: {', '.join(missing)}")

    async def start(self, force_build: bool) -> None:
        image = await _run_blocking(self._resolve_sqsh, force_build)
        self._staging_dir = self._prepare_staging()
        self._enroot_sysconf_dir = _prepare_enroot_sysconf(
            self._shared_dir / self.session_id / "enroot-sysconf"
        )
        self._workdir = (
            self.task_env_config.workdir or self._image_workdir() or self._dockerfile_workdir()
        )
        await self._start_with_retries(image)
        await self.ensure_dirs(self._mount_targets(writable_only=True))

    async def _start_with_retries(self, image: Path) -> None:
        for attempt in range(1, self._startup_retries + 1):
            try:
                await self._start_srun(image)
                return
            except RuntimeError as exc:
                await self._cleanup_failed_start()
                if attempt >= self._startup_retries or not _is_transient_startup_error(str(exc)):
                    raise
                delay = self._startup_retry_delay_sec * attempt
                self.logger.warning(
                    "Retrying Slurm/Pyxis startup after transient failure "
                    f"({attempt}/{self._startup_retries}): {exc}"
                )
                if delay:
                    await asyncio.sleep(delay)

    async def _start_srun(self, image: Path) -> None:
        cmd = self._srun_command(image)
        self.logger.info("Starting Slurm/Pyxis environment")
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._srun_env(),
            start_new_session=True,
        )
        self._node = await self._read_startup_node()
        self._stream_task = asyncio.create_task(self._stream_output())
        await self._wait_for_health()

    async def _cleanup_failed_start(self) -> None:
        self._node = None
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        if self._process and self._process.returncode is None:
            await self._stop_srun()
        elif self._process:
            await self._process.wait()
        self._process = None

    async def stop(self, delete: bool) -> None:
        if self._process and self._process.returncode is None:
            if self._node:
                try:
                    await self._post("/shutdown", {}, timeout=DEFAULT_SHUTDOWN_TIMEOUT_SEC)
                except Exception:
                    pass
            await self._stop_srun()
        elif self._process:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=1)
        if self._stream_task:
            self._stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stream_task
            self._stream_task = None
        self._process = None
        self._node = None
        if delete and self._staging_dir:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        if delete and self._enroot_sysconf_dir:
            shutil.rmtree(self._enroot_sysconf_dir, ignore_errors=True)

    async def _stop_srun(self) -> None:
        assert self._process
        pid = self._process.pid
        for sig, timeout in ((signal.SIGINT, 3), (signal.SIGTERM, 3)):
            if self._process.returncode is not None:
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=timeout)
        await _run_blocking(self._cancel_slurm_job)
        if self._process.returncode is None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timed out waiting for Slurm/Pyxis srun {pid} to exit after SIGKILL"
                )

    def _cancel_slurm_job(self) -> None:
        if shutil.which("scancel") is None:
            return
        command = ["scancel", "--name", self._slurm_job_name]
        if user := os.environ.get("USER"):
            command.extend(["--user", user])
        try:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DEFAULT_SCANCEL_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            pass

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            command = f"su {shlex.quote(str(resolved_user))} -s /bin/bash -c {shlex.quote(command)}"
        payload = {
            "command": command,
            "cwd": cwd,
            "env": self._merge_env(env),
            "timeout_sec": timeout_sec,
        }
        request_timeout = (
            timeout_sec + 10 if timeout_sec is not None else self._exec_request_timeout_sec
        )
        data = await self._post_while_srun_lives("/exec", payload, timeout=request_timeout)
        await self._settle_verifier_reward(command)
        if data.get("timeout"):
            seconds = data.get("timeout_sec", timeout_sec)
            raise RuntimeError(f"Command timed out after {seconds} seconds")
        return ExecResult(
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            return_code=int(data.get("return_code", 1)),
        )

    async def exec_command(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str | None = None,
        login: bool = False,
        tty: bool = False,
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ) -> UnifiedExecResult:
        payload = {
            "command": command,
            "cwd": cwd,
            "env": self._merge_env(env),
            "timeout_sec": timeout_sec,
            "shell": shell,
            "login": login,
            "tty": tty,
            "yield_time_ms": yield_time_ms,
            "max_output_tokens": max_output_tokens,
        }
        wait_ms = yield_time_ms if yield_time_ms is not None else 10000
        request_timeout = max(30, int(wait_ms / 1000) + 30)
        data = await self._post_while_srun_lives("/exec_command", payload, timeout=request_timeout)
        return _unified_exec_result(data)

    async def write_stdin(
        self,
        session_id: int,
        chars: str = "",
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ) -> UnifiedExecResult:
        payload = {
            "session_id": session_id,
            "chars": chars,
            "yield_time_ms": yield_time_ms,
            "max_output_tokens": max_output_tokens,
        }
        wait_ms = yield_time_ms if yield_time_ms is not None else 250
        request_timeout = max(30, int(wait_ms / 1000) + 30)
        data = await self._post_while_srun_lives("/write_stdin", payload, timeout=request_timeout)
        return _unified_exec_result(data)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        staged = self._staging / source.name
        shutil.copy2(source, staged)
        result = await self.exec(
            f"cp {shlex.quote('/staging/' + source.name)} {shlex.quote(target_path)}"
        )
        staged.unlink(missing_ok=True)
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout or "upload_file failed")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        staged = self._staging / source.name
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(source, staged)
        result = await self.exec(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"cp -r {shlex.quote('/staging/' + source.name)}/. {shlex.quote(target_dir)}/"
        )
        shutil.rmtree(staged, ignore_errors=True)
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout or "upload_dir failed")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged_name = "download_" + Path(source_path).name
        result = await self.exec(
            f"cp {shlex.quote(source_path)} {shlex.quote('/staging/' + staged_name)}"
        )
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout or "download_file failed")
        shutil.copy2(self._staging / staged_name, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        staged_name = "download_" + Path(source_dir).name
        staged = self._staging / staged_name
        result = await self.exec(
            f"cp -r {shlex.quote(source_dir)} {shlex.quote('/staging/' + staged_name)}"
        )
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout or "download_dir failed")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staged, target)

    async def _settle_verifier_reward(self, command: str) -> None:
        if self._verifier_reward_settle_sec <= 0 or "/logs/verifier" not in command:
            return
        reward_paths = self._verifier_reward_host_paths()
        if not reward_paths:
            return
        deadline = time.monotonic() + self._verifier_reward_settle_sec
        while time.monotonic() < deadline:
            if any(path.exists() for path in reward_paths):
                return
            await asyncio.sleep(0.05)

    def _verifier_reward_host_paths(self) -> list[Path]:
        host_verifier = self._host_path_for_container("/logs/verifier")
        if host_verifier is None:
            trial_paths = getattr(self, "trial_paths", None)
            host_verifier = getattr(trial_paths, "verifier_dir", None)
        if host_verifier is None:
            return []
        host_verifier = Path(host_verifier)
        return [host_verifier / "reward.txt", host_verifier / "reward.json"]

    def _host_path_for_container(self, container_path: str) -> Path | None:
        best_target = ""
        best_source: Path | None = None
        for mount in self._mounts:
            if mount.get("type") != "bind":
                continue
            target = str(mount.get("target") or "").rstrip("/")
            source = mount.get("source")
            if not target or source is None:
                continue
            if container_path == target or container_path.startswith(target + "/"):
                if len(target) > len(best_target):
                    best_target = target
                    suffix = container_path[len(target) :].lstrip("/")
                    best_source = Path(source) / suffix if suffix else Path(source)
        return best_source

    @property
    def _staging(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError("Slurm/Pyxis environment is not started")
        return self._staging_dir

    def _prepare_staging(self) -> Path:
        staging = self._shared_dir / self.session_id / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        staging.chmod(0o755)
        (staging / "bootstrap.sh").write_text(SLURM_BOOTSTRAP, encoding="utf-8")
        (staging / "_hbexec.py").write_text(STDLIB_EXEC_SERVER, encoding="utf-8")
        (staging / "bootstrap.sh").chmod(0o755)
        return staging

    def _srun_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("TZ", "Etc/UTC")
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["HARBOR_SLURM_PYXIS_VERIFY_SETTLE_SEC"] = str(self._verifier_reward_settle_sec)
        if self._enroot_sysconf_dir:
            env["ENROOT_SYSCONF_PATH"] = str(self._enroot_sysconf_dir)
        return env

    def _srun_command(self, image: Path) -> list[str]:
        mounts = [f"{self._staging}:/staging"]
        for mount in self._mounts:
            if mount.get("type") == "bind":
                item = f"{mount['source']}:{mount['target']}"
                if mount.get("read_only"):
                    item += ":ro"
                mounts.append(item)
        env_files = self.environment_dir / "files"
        if env_files.exists():
            mounts.append(f"{env_files}:/staging/env_files")
        if self._host_python_prefix and (self._host_python_prefix / "bin" / "python").exists():
            mounts.append(f"{self._host_python_prefix}:{HOST_PYTHON_MOUNT}:ro")
        boot = (
            f"exec /staging/bootstrap.sh {shlex.quote(self._workdir)} "
            f"/staging/_hbexec.py --port {self._port} --workdir {shlex.quote(self._workdir)}"
        )
        cmd = [
            "srun",
            "--job-name",
            self._slurm_job_name,
            "--partition",
            self._slurm_partition,
            "--nodes",
            "1",
            "--ntasks",
            "1",
            "--cpus-per-task",
            str(self.task_env_config.cpus),
            "--mem",
            f"{self.task_env_config.memory_mb}M",
            "--time",
            self._slurm_time,
            "--kill-on-bad-exit=1",
            "--wait",
            "1",
            "--quit-on-interrupt",
            "--export",
            "ALL",
            "--container-image",
            str(image),
            "--container-mounts",
            ",".join(mounts),
            "--container-workdir",
            "/",
            "--container-writable",
        ]
        if self._remap_root:
            cmd.append("--container-remap-root")
        return [*cmd, "/bin/bash", "-lc", boot]

    async def _read_startup_node(self) -> str:
        assert self._process and self._process.stdout
        deadline = time.monotonic() + self._startup_timeout_sec
        startup_lines: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = await asyncio.wait_for(self._process.stdout.readline(), timeout=30)
            except asyncio.TimeoutError:
                if self._process.returncode is not None:
                    raise RuntimeError(
                        _startup_failure_message(self._process.returncode, startup_lines)
                    )
                continue
            text = line.decode(errors="replace").rstrip()
            if not line:
                if self._process.returncode is None:
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=1)
                    except asyncio.TimeoutError:
                        pass
                if self._process.returncode is not None:
                    raise RuntimeError(
                        _startup_failure_message(self._process.returncode, startup_lines)
                    )
                continue
            if text:
                self.logger.debug(text)
                self._record_srun_output(text)
                startup_lines = [*startup_lines[-19:], text]
            if text.startswith("__HARBOR_PYXIS_READY__"):
                node, port = text.removeprefix("__HARBOR_PYXIS_READY__").rsplit(":", 1)
                self._port = int(port)
                return node
            if text.startswith("__HARBOR_PYXIS_NODE__"):
                return text.removeprefix("__HARBOR_PYXIS_NODE__")
            if self._process.returncode is not None:
                raise RuntimeError(
                    _startup_failure_message(self._process.returncode, startup_lines)
                )
        raise RuntimeError(f"Timed out waiting {self._startup_timeout_sec}s for Slurm/Pyxis node")

    async def _stream_output(self) -> None:
        assert self._process and self._process.stdout
        async for line in self._process.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                self.logger.debug(text)
                self._record_srun_output(text)

    async def _wait_for_health(self) -> None:
        for _ in range(self._health_timeout_sec):
            if self._process and self._process.returncode is not None:
                raise RuntimeError(self._srun_exit_message("before health check"))
            try:
                data = await self._get("/health", timeout=5)
                if data.get("status") in {"ok", "healthy"}:
                    return
            except Exception:
                await asyncio.sleep(1)
        raise RuntimeError(
            f"Slurm/Pyxis exec server did not become healthy within {self._health_timeout_sec}s"
        )

    async def _get(self, path: str, timeout: int) -> dict[str, object]:
        return await _run_blocking(self._request, path, None, timeout)

    async def _post(
        self, path: str, payload: dict[str, object], timeout: int = 30
    ) -> dict[str, object]:
        return await _run_blocking(self._request, path, payload, timeout)

    async def _post_while_srun_lives(
        self, path: str, payload: dict[str, object], timeout: int
    ) -> dict[str, object]:
        self._raise_if_srun_exited("before request")
        request_task = asyncio.create_task(self._post(path, payload, timeout=timeout))
        wait_task = (
            asyncio.create_task(self._process.wait())
            if self._process and self._process.returncode is None
            else None
        )
        tasks = {request_task, *([wait_task] if wait_task else [])}
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout + DEFAULT_POST_TIMEOUT_GRACE_SEC,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if not done:
            request_task.cancel()
            with suppress(asyncio.CancelledError):
                await request_task
            raise RuntimeError(f"Slurm/Pyxis request to {path} timed out after {timeout}s")
        if request_task in done:
            return await request_task
        request_task.cancel()
        with suppress(asyncio.CancelledError):
            await request_task
        self._raise_if_srun_exited("during request")
        raise RuntimeError("Slurm/Pyxis srun exited during request")

    def _raise_if_srun_exited(self, when: str) -> None:
        if self._process and self._process.returncode is not None:
            raise RuntimeError(self._srun_exit_message(when))

    def _srun_exit_message(self, when: str) -> str:
        message = (
            f"Slurm/Pyxis srun exited {when}: {self._process.returncode if self._process else None}"
        )
        if self._recent_srun_output:
            message += "; output: " + " | ".join(self._recent_srun_output[-10:])
        return message

    def _record_srun_output(self, text: str) -> None:
        self._recent_srun_output = [*self._recent_srun_output[-19:], text]

    def _request(
        self, path: str, payload: dict[str, object] | None, timeout: int
    ) -> dict[str, object]:
        if not self._node:
            raise RuntimeError("Slurm/Pyxis node is unknown")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://{self._node}:{self._port}{path}",
            data=body,
            headers={"content-type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            raise RuntimeError(f"Slurm/Pyxis server request failed: {exc}") from exc

    def _resolve_sqsh(self, force_build: bool) -> Path:
        image = self.task_env_config.docker_image
        assert image
        image_path = Path(image)
        if image_path.suffix == ".sqsh" and image_path.exists():
            return image_path
        tar_path = image_path if image_path.suffix == ".tar" else self._tar_for_image(image)
        tag = _docker_archive_tag(tar_path)
        out = self._sqsh_cache_dir / f"{_safe_image_name(tag)}.sqsh"
        if out.exists() and not force_build:
            return out
        if shutil.which("docker") is None:
            raise RuntimeError(
                f"No cached SquashFS image found at {out}. Preconvert {tar_path} into "
                f"{self._sqsh_cache_dir}, or run on a host with Docker available for conversion."
            )
        _convert_archive(tar_path, tag, out, self._sqsh_cache_dir.parent / "enroot")
        return out

    def _tar_for_image(self, image: str) -> Path:
        candidates = [_safe_image_name(ref) for ref in _image_ref_candidates(image)]
        for safe in candidates:
            path = self._docker_tar_cache_dir / f"{safe}.tar"
            if path.exists():
                return path
        raise FileNotFoundError(f"No cached Docker archive found for image {image!r}")

    def _image_workdir(self) -> str | None:
        image = self.task_env_config.docker_image
        if not image:
            return None
        image_path = Path(image)
        try:
            tar_path = image_path if image_path.suffix == ".tar" else self._tar_for_image(image)
        except FileNotFoundError:
            return None
        try:
            with tarfile.open(tar_path) as tar:
                manifest_file = tar.extractfile("manifest.json")
                if manifest_file is None:
                    return None
                config_name = json.load(manifest_file)[0].get("Config")
                config_file = tar.extractfile(config_name) if config_name else None
                if config_file is None:
                    return None
                return json.load(config_file).get("config", {}).get("WorkingDir") or None
        except (OSError, KeyError, IndexError, json.JSONDecodeError, tarfile.TarError):
            return None

    def _dockerfile_workdir(self) -> str:
        dockerfile = self.environment_dir / "Dockerfile"
        workdir = PurePosixPath("/")
        if dockerfile.exists():
            for line in dockerfile.read_text(errors="ignore").splitlines():
                if line.strip().upper().startswith("WORKDIR "):
                    value = line.split(None, 1)[1].strip()
                    workdir = PurePosixPath(value) if value.startswith("/") else workdir / value
        return workdir.as_posix()


def _startup_failure_message(returncode: int | None, startup_lines: list[str]) -> str:
    message = f"srun exited before startup: {returncode}"
    if startup_lines:
        message += "; output: " + " | ".join(startup_lines)
    return message


def _is_transient_startup_error(message: str) -> bool:
    lower = message.lower()
    return any(pattern in lower for pattern in _TRANSIENT_STARTUP_ERRORS)


def _unified_exec_result(data: dict[str, object]) -> UnifiedExecResult:
    exit_code = data.get("exit_code") if "exit_code" in data else data.get("return_code")
    session_id = data.get("session_id")
    original_token_count = data.get("original_token_count")
    return UnifiedExecResult(
        stdout=str(data.get("stdout") or data.get("output") or ""),
        stderr=data.get("stderr") if isinstance(data.get("stderr"), str) else None,
        return_code=None if exit_code is None else int(exit_code),
        chunk_id=str(data.get("chunk_id") or ""),
        wall_time_seconds=float(data.get("wall_time_seconds") or 0.0),
        session_id=None if session_id is None else int(session_id),
        original_token_count=None if original_token_count is None else int(original_token_count),
    )


def _slurm_time_to_seconds(value: str) -> int:
    if "-" in value:
        days, value = value.split("-", 1)
        day_seconds = int(days) * 24 * 60 * 60
    else:
        day_seconds = 0
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    else:
        raise ValueError(f"invalid Slurm time: {value!r}")
    return day_seconds + hours * 60 * 60 + minutes * 60 + seconds


def _image_ref_candidates(image: str) -> list[str]:
    ref = image.removeprefix("docker://")
    ref = ref.split("#", 1)[-1]
    refs = [ref]
    for prefix in ("docker.io/", "registry-1.docker.io/"):
        if ref.startswith(prefix):
            refs.append(ref.removeprefix(prefix))
    if ":" not in refs[0].rsplit("/", 1)[-1]:
        refs.extend(f"{item}:latest" for item in list(refs))
    return refs


def _safe_image_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "_").replace("#", "_")


def _prepare_enroot_sysconf(target: Path, source: Path = DEFAULT_ENROOT_SYSCONF) -> Path | None:
    if not source.exists():
        return None
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True)
    for path in (target / "mounts.d").glob("*.fstab"):
        _remove_localtime_mount(path)
    return target


def _remove_localtime_mount(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not _is_localtime_mount(line)]
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _is_localtime_mount(line: str) -> bool:
    fields = line.split()
    return len(fields) >= 2 and fields[0] == "/etc/localtime" and fields[1] == "/etc/localtime"


def _docker_archive_tag(path: Path) -> str:
    with tarfile.open(path) as tar:
        manifest = tar.extractfile("manifest.json")
        if manifest is None:
            raise RuntimeError(f"{path} does not contain manifest.json")
        data = json.load(manifest)
    return data[0]["RepoTags"][0]


def _convert_archive(tar_path: Path, tag: str, out: Path, enroot_dir: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    enroot_dir.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(out.suffix + ".lock")
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if out.exists():
                return
            time.sleep(2)
    try:
        if out.exists():
            return
        env = os.environ.copy()
        for name in ("cache", "data", "runtime"):
            path = enroot_dir / name
            path.mkdir(parents=True, exist_ok=True)
            env[f"ENROOT_{name.upper()}_PATH"] = str(path)
        subprocess.run(["docker", "load", "-i", str(tar_path)], check=True, env=env)
        tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
        subprocess.run(
            ["enroot", "import", "-o", str(tmp), f"dockerd://{tag}"], check=True, env=env
        )
        tmp.replace(out)
    finally:
        lock.rmdir()

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
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
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def _exit_later():
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
        self._process: asyncio.subprocess.Process | None = None
        self._stream_task: asyncio.Task | None = None
        self._node: str | None = None
        self._port = 0
        self._slurm_job_name = f"hb-{os.getpid()}-{random.randint(100000, 999999)}"
        self._staging_dir: Path | None = None
        self._enroot_sysconf_dir: Path | None = None
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
                    await self._post("/shutdown", {})
                except Exception:
                    pass
            await self._stop_srun()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if delete and self._staging_dir:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        if delete and self._enroot_sysconf_dir:
            shutil.rmtree(self._enroot_sysconf_dir, ignore_errors=True)

    async def _stop_srun(self) -> None:
        assert self._process
        pid = self._process.pid
        await _run_blocking(self._cancel_slurm_job)
        for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            if self._process.returncode is not None:
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            try:
                await asyncio.wait_for(self._process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
        if self._process.returncode is None:
            self._process.kill()
            await self._process.wait()

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
                timeout=10,
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
        data = await self._post("/exec", payload, timeout=request_timeout)
        if data.get("timeout"):
            seconds = data.get("timeout_sec", timeout_sec)
            raise RuntimeError(f"Command timed out after {seconds} seconds")
        return ExecResult(
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            return_code=int(data.get("return_code", 1)),
        )

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

    async def _wait_for_health(self) -> None:
        for _ in range(self._health_timeout_sec):
            if self._process and self._process.returncode is not None:
                raise RuntimeError(f"srun exited before health check: {self._process.returncode}")
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

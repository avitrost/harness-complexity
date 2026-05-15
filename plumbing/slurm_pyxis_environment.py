from __future__ import annotations

import asyncio
import json
import os
import random
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import TaskOS

DEFAULT_SHARED_DIR = Path("/wbl-fast/usrs/trost/harbor-slurm-pyxis")
DEFAULT_SQSH_CACHE = Path("/wbl-fast/usrs/trost/tbench-sqsh-cache/images")
DEFAULT_TAR_CACHE = Path("/wbl-fast/usrs/ee/agent-collab/docker-image-cache")


class SlurmPyxisEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        sqsh_cache_dir: str | Path = DEFAULT_SQSH_CACHE,
        docker_tar_cache_dir: str | Path = DEFAULT_TAR_CACHE,
        shared_dir: str | Path = DEFAULT_SHARED_DIR,
        slurm_partition: str = "m7i-cpu2",
        slurm_time: str = "02:00:00",
        remap_root: bool = True,
        **kwargs,
    ):
        self._sqsh_cache_dir = Path(sqsh_cache_dir)
        self._docker_tar_cache_dir = Path(docker_tar_cache_dir)
        self._shared_dir = Path(shared_dir)
        self._slurm_partition = slurm_partition
        self._slurm_time = slurm_time
        self._remap_root = remap_root
        self._process: asyncio.subprocess.Process | None = None
        self._stream_task: asyncio.Task | None = None
        self._node: str | None = None
        self._port = random.randint(20000, 60000)
        self._staging_dir: Path | None = None
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
        image = await asyncio.to_thread(self._resolve_sqsh, force_build)
        self._staging_dir = self._prepare_staging()
        self._workdir = self.task_env_config.workdir or self._dockerfile_workdir()
        cmd = self._srun_command(image)
        self.logger.info("Starting Slurm/Pyxis environment")
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._node = await self._read_startup_node()
        self._stream_task = asyncio.create_task(self._stream_output())
        await self._wait_for_health()
        await self.ensure_dirs(self._mount_targets(writable_only=True))

    async def stop(self, delete: bool) -> None:
        if self._process and self._process.returncode is None:
            if self._node:
                try:
                    await self._post("/shutdown", {})
                except Exception:
                    pass
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if delete and self._staging_dir:
            shutil.rmtree(self._staging_dir, ignore_errors=True)

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
        data = await self._post("/exec", payload, timeout=(timeout_sec or 600) + 10)
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
        from harbor.environments.singularity import singularity

        src = Path(singularity.__file__).parent
        shutil.copy2(src / "bootstrap.sh", staging / "bootstrap.sh")
        server_text = (src / "server.py").read_text(encoding="utf-8")
        server_text = server_text.replace('host="127.0.0.1"', 'host="0.0.0.0"')
        (staging / "_hbexec.py").write_text(server_text, encoding="utf-8")
        (staging / "bootstrap.sh").chmod(0o755)
        return staging

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
        boot = (
            f"echo __HARBOR_PYXIS_NODE__$(hostname); "
            f"exec /staging/bootstrap.sh {shlex.quote(self._workdir)} "
            f"/staging/_hbexec.py --port {self._port} --workdir {shlex.quote(self._workdir)}"
        )
        cmd = [
            "srun",
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
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                line = await asyncio.wait_for(self._process.stdout.readline(), timeout=30)
            except asyncio.TimeoutError:
                if self._process.returncode is not None:
                    raise RuntimeError(f"srun exited before startup: {self._process.returncode}")
                continue
            text = line.decode(errors="replace").rstrip()
            if text:
                self.logger.debug(text)
            if text.startswith("__HARBOR_PYXIS_NODE__"):
                return text.removeprefix("__HARBOR_PYXIS_NODE__")
            if self._process.returncode is not None:
                raise RuntimeError(f"srun exited before startup: {self._process.returncode}")
        raise RuntimeError("Timed out waiting for Slurm/Pyxis node")

    async def _stream_output(self) -> None:
        assert self._process and self._process.stdout
        async for line in self._process.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                self.logger.debug(text)

    async def _wait_for_health(self) -> None:
        for _ in range(120):
            try:
                data = await self._get("/health", timeout=5)
                if data.get("status") in {"ok", "healthy"}:
                    return
            except Exception:
                await asyncio.sleep(1)
        raise RuntimeError("Slurm/Pyxis exec server did not become healthy")

    async def _get(self, path: str, timeout: int) -> dict[str, object]:
        return await asyncio.to_thread(self._request, path, None, timeout)

    async def _post(
        self, path: str, payload: dict[str, object], timeout: int = 30
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._request, path, payload, timeout)

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
        except URLError as exc:
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
        _convert_archive(tar_path, tag, out, self._sqsh_cache_dir.parent / "enroot")
        return out

    def _tar_for_image(self, image: str) -> Path:
        candidates = [_safe_image_name(ref) for ref in _image_ref_candidates(image)]
        for safe in candidates:
            path = self._docker_tar_cache_dir / f"{safe}.tar"
            if path.exists():
                return path
        raise FileNotFoundError(f"No cached Docker archive found for image {image!r}")

    def _dockerfile_workdir(self) -> str:
        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.exists():
            for line in dockerfile.read_text(errors="ignore").splitlines():
                if line.strip().upper().startswith("WORKDIR "):
                    return line.split(None, 1)[1].strip()
        return "/app"


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

import asyncio
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import plumbing.slurm_pyxis_environment as slurm_pyxis
from plumbing.slurm_pyxis_environment import (
    SLURM_BOOTSTRAP,
    SlurmPyxisEnvironment,
    STDLIB_EXEC_SERVER,
    _is_transient_startup_error,
    _prepare_enroot_sysconf,
    _slurm_time_to_seconds,
    _unified_exec_result,
)


def test_slurm_bootstrap_has_no_pip_or_asciinema_dependency() -> None:
    assert "get-pip" not in SLURM_BOOTSTRAP
    assert "uvicorn" not in SLURM_BOOTSTRAP
    assert "asciinema" not in SLURM_BOOTSTRAP
    assert "apt-get" not in SLURM_BOOTSTRAP
    assert "apk add" not in SLURM_BOOTSTRAP
    assert "dnf install" not in SLURM_BOOTSTRAP
    assert "yum install" not in SLURM_BOOTSTRAP
    assert "no Python runtime available" in SLURM_BOOTSTRAP


def test_stdlib_exec_server_has_required_routes_without_fastapi() -> None:
    assert "ThreadingHTTPServer" in STDLIB_EXEC_SERVER
    assert 'self.path == "/health"' in STDLIB_EXEC_SERVER
    assert 'self.path == "/exec"' in STDLIB_EXEC_SERVER
    assert 'self.path == "/exec_command"' in STDLIB_EXEC_SERVER
    assert 'self.path == "/write_stdin"' in STDLIB_EXEC_SERVER
    assert "__HARBOR_PYXIS_READY__" in STDLIB_EXEC_SERVER
    assert "server.server_address[1]" in STDLIB_EXEC_SERVER
    assert "fastapi" not in STDLIB_EXEC_SERVER.lower()


def test_prepare_enroot_sysconf_removes_localtime_mount(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mounts = source / "mounts.d"
    mounts.mkdir(parents=True)
    (source / "enroot.conf").write_text("# config\n", encoding="utf-8")
    (mounts / "20-config.fstab").write_text(
        "/etc/hosts /etc/hosts none bind 0 -1\n"
        "/etc/localtime /etc/localtime none bind,ro 0 -1\n",
        encoding="utf-8",
    )
    (mounts / "30-extra.fstab").write_text(
        "/etc/localtime /etc/localtime none bind,ro 0 -1\n"
        "/etc/resolv.conf /etc/resolv.conf none bind,ro 0 -1\n",
        encoding="utf-8",
    )

    target = _prepare_enroot_sysconf(tmp_path / "target", source)

    assert target == tmp_path / "target"
    patched = (target / "mounts.d" / "20-config.fstab").read_text(encoding="utf-8")
    assert "/etc/hosts /etc/hosts" in patched
    assert "/etc/localtime" not in patched
    extra = (target / "mounts.d" / "30-extra.fstab").read_text(encoding="utf-8")
    assert "/etc/resolv.conf /etc/resolv.conf" in extra
    assert "/etc/localtime" not in extra


def _make_env(tmp_path: Path) -> SlurmPyxisEnvironment:
    environment_dir = tmp_path / "env"
    environment_dir.mkdir()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    env = SlurmPyxisEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=EnvironmentConfig(docker_image="ubuntu:latest"),
        shared_dir=tmp_path / "shared",
        host_python_prefix=tmp_path / "host-python",
    )
    (tmp_path / "host-python" / "bin").mkdir(parents=True)
    (tmp_path / "host-python" / "bin" / "python").write_text("", encoding="utf-8")
    env._staging_dir = tmp_path / "staging"
    env._staging_dir.mkdir()
    return env


def test_srun_command_uses_unique_job_name(tmp_path: Path) -> None:
    env = _make_env(tmp_path)

    command = env._srun_command(tmp_path / "image.sqsh")

    job_name_index = command.index("--job-name") + 1
    assert command[job_name_index] == env._slurm_job_name
    assert command[command.index("--partition") + 1] == "m7i-cpu"
    assert env._slurm_job_name.startswith("hb-")
    assert "--kill-on-bad-exit=1" in command
    assert command[command.index("--wait") + 1] == "1"
    assert "--quit-on-interrupt" in command
    assert "--port 0" in command[-1]
    mounts = command[command.index("--container-mounts") + 1].split(",")
    assert f"{tmp_path / 'host-python'}:/opt/harbor-python:ro" in mounts


def test_default_exec_request_timeout_tracks_slurm_time(tmp_path: Path) -> None:
    env = _make_env(tmp_path)

    assert env._exec_request_timeout_sec == 2 * 60 * 60 + 120


def test_dockerfile_workdir_uses_final_workdir_with_relative_steps(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    (env.environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:latest\nWORKDIR /app\nWORKDIR project\n",
        encoding="utf-8",
    )

    assert env._dockerfile_workdir() == "/app/project"


def test_image_workdir_uses_cached_docker_config(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    env._docker_tar_cache_dir = tmp_path / "docker-cache"
    env._docker_tar_cache_dir.mkdir()
    tar_path = env._docker_tar_cache_dir / "ubuntu_latest.tar"
    _write_image_tar(tar_path, "/image/workdir")

    assert env._image_workdir() == "/image/workdir"


def test_exec_raises_like_harbor_on_command_timeout(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)

    async def fake_post(path, payload, timeout):
        return {"timeout": True, "timeout_sec": payload["timeout_sec"]}

    monkeypatch.setattr(env, "_post", fake_post)

    with pytest.raises(RuntimeError, match="Command timed out after 3 seconds"):
        asyncio.run(env.exec("sleep 10", timeout_sec=3))


def test_exec_stops_waiting_when_srun_exits(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)
    env._process = _ExitedProcess()
    env._recent_srun_output = ["srun: error: Node failure on m7i-cpu"]

    async def slow_post(path, payload, timeout):
        await asyncio.sleep(30)

    monkeypatch.setattr(env, "_post", slow_post)

    with pytest.raises(RuntimeError, match="Node failure"):
        asyncio.run(env.exec("sleep 10"))


def test_post_while_srun_lives_has_hard_timeout(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)
    env._node = "node"
    env._process = _RunningProcess()
    monkeypatch.setattr(slurm_pyxis, "DEFAULT_POST_TIMEOUT_GRACE_SEC", 0)

    async def slow_post(path, payload, timeout):
        await asyncio.sleep(30)

    monkeypatch.setattr(env, "_post", slow_post)

    with pytest.raises(RuntimeError, match=r"request to /write_stdin timed out"):
        asyncio.run(env._post_while_srun_lives("/write_stdin", {}, timeout=0))


def test_exec_command_uses_unified_session_route(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)
    call = {}

    async def fake_post(path, payload, timeout):
        call.update({"path": path, "payload": payload, "timeout": timeout})
        return {
            "stdout": "ready\n",
            "stderr": None,
            "exit_code": None,
            "chunk_id": "abc123",
            "wall_time_seconds": 0.25,
            "session_id": 7,
            "original_token_count": 3,
        }

    monkeypatch.setattr(env, "_post_while_srun_lives", fake_post)

    result = asyncio.run(
        env.exec_command(
            "python -i",
            cwd="/app/src",
            shell="/bin/sh",
            login=True,
            tty=True,
            yield_time_ms=250,
            max_output_tokens=100,
        )
    )

    assert call["path"] == "/exec_command"
    assert call["timeout"] == 30
    assert call["payload"]["command"] == "python -i"
    assert call["payload"]["cwd"] == "/app/src"
    assert call["payload"]["shell"] == "/bin/sh"
    assert call["payload"]["login"] is True
    assert call["payload"]["tty"] is True
    assert result.return_code is None
    assert result.session_id == 7
    assert result.chunk_id == "abc123"


def test_write_stdin_uses_unified_session_route(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)
    call = {}

    async def fake_post(path, payload, timeout):
        call.update({"path": path, "payload": payload, "timeout": timeout})
        return {
            "stdout": "done\n",
            "stderr": None,
            "exit_code": 0,
            "chunk_id": "def456",
            "wall_time_seconds": 0.1,
            "original_token_count": 2,
        }

    monkeypatch.setattr(env, "_post_while_srun_lives", fake_post)

    result = asyncio.run(env.write_stdin(7, "exit()\n", yield_time_ms=100))

    assert call["path"] == "/write_stdin"
    assert call["payload"]["session_id"] == 7
    assert call["payload"]["chars"] == "exit()\n"
    assert call["timeout"] == 30
    assert result.return_code == 0
    assert result.session_id is None
    assert result.stdout == "done\n"


def test_unified_exec_result_prefers_exit_code_over_return_code() -> None:
    result = _unified_exec_result(
        {
            "stdout": "partial",
            "return_code": 0,
            "exit_code": None,
            "chunk_id": "abc",
            "wall_time_seconds": 1.0,
            "session_id": 3,
        }
    )

    assert result.return_code is None
    assert result.session_id == 3


def test_stdlib_exec_server_runs_unified_session_smoke(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    server.write_text(STDLIB_EXEC_SERVER, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(server), "--port", "0", "--workdir", str(tmp_path / "work")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    endpoint = None
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line.startswith("__HARBOR_PYXIS_READY__"):
                endpoint = line.strip().removeprefix("__HARBOR_PYXIS_READY__")
                break
        assert endpoint is not None

        first = _post_json(
            endpoint,
            "/exec_command",
            {"command": "read line; echo got:$line", "yield_time_ms": 100},
        )
        second = _post_json(
            endpoint,
            "/write_stdin",
            {
                "session_id": first["session_id"],
                "chars": "hello\n",
                "yield_time_ms": 500,
                "max_output_tokens": 1,
            },
        )

        assert second["exit_code"] == 0
        assert "got:hello" not in second["output"]
        assert second["output"].startswith("<omitted ")
        assert second["output"].endswith("llo\n")
    finally:
        if endpoint is not None:
            try:
                _post_json(endpoint, "/shutdown", {})
            except Exception:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_http_requests_use_slurm_executor(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)

    def fake_request(path, payload, timeout):
        return {"path": path, "payload": payload, "timeout": timeout}

    async def fail_to_thread(*args, **kwargs):
        raise AssertionError("Slurm/Pyxis requests should not use asyncio.to_thread")

    monkeypatch.setattr(env, "_request", fake_request)
    monkeypatch.setattr(slurm_pyxis.asyncio, "to_thread", fail_to_thread)

    assert asyncio.run(env._get("/health", timeout=5)) == {
        "path": "/health",
        "payload": None,
        "timeout": 5,
    }
    assert asyncio.run(env._post("/shutdown", {}, timeout=5)) == {
        "path": "/shutdown",
        "payload": {},
        "timeout": 5,
    }


def test_slurm_time_to_seconds() -> None:
    assert _slurm_time_to_seconds("02:00:00") == 7200
    assert _slurm_time_to_seconds("15:30") == 930
    assert _slurm_time_to_seconds("2-01:00:00") == 176400


def test_transient_startup_error_detects_cloud_node_boot_failures() -> None:
    assert _is_transient_startup_error("srun: error: Node failure on m7i-cpu2-dy-0")
    assert _is_transient_startup_error("Nodes m7i-cpu2-dy-0 are still not ready")
    assert _is_transient_startup_error("Something is wrong with the boot of the nodes.")
    assert not _is_transient_startup_error("FATAL: no Python runtime available")


def test_cancel_slurm_job_is_scoped_to_current_user(tmp_path: Path, monkeypatch) -> None:
    env = _make_env(tmp_path)
    calls = []

    monkeypatch.setenv("USER", "trost")
    monkeypatch.setattr(slurm_pyxis.shutil, "which", lambda name: "/usr/bin/scancel")
    monkeypatch.setattr(
        slurm_pyxis.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    env._cancel_slurm_job()

    assert calls
    assert calls[0][0] == ["scancel", "--name", env._slurm_job_name, "--user", "trost"]


def _write_image_tar(path: Path, workdir: str) -> None:
    config = json.dumps({"config": {"WorkingDir": workdir}}).encode()
    manifest = json.dumps([{"Config": "config.json", "RepoTags": ["ubuntu:latest"]}]).encode()
    with tarfile.open(path, "w") as tar:
        for name, payload in (("manifest.json", manifest), ("config.json", config)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


class _ExitedProcess:
    pid = 1234
    returncode = None

    async def wait(self) -> int:
        self.returncode = 1
        return self.returncode


class _RunningProcess:
    pid = 1234
    returncode = None

    async def wait(self) -> int:
        await asyncio.sleep(30)
        return 0


def _post_json(endpoint: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"http://{endpoint}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))

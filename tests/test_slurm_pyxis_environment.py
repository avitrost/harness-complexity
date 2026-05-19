from pathlib import Path

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import plumbing.slurm_pyxis_environment as slurm_pyxis
from plumbing.slurm_pyxis_environment import (
    SLURM_BOOTSTRAP,
    SlurmPyxisEnvironment,
    STDLIB_EXEC_SERVER,
    _is_transient_startup_error,
    _prepare_enroot_sysconf,
)


def test_slurm_bootstrap_has_no_pip_or_asciinema_dependency() -> None:
    assert "get-pip" not in SLURM_BOOTSTRAP
    assert "uvicorn" not in SLURM_BOOTSTRAP
    assert "asciinema" not in SLURM_BOOTSTRAP


def test_stdlib_exec_server_has_required_routes_without_fastapi() -> None:
    assert "ThreadingHTTPServer" in STDLIB_EXEC_SERVER
    assert 'self.path == "/health"' in STDLIB_EXEC_SERVER
    assert 'self.path == "/exec"' in STDLIB_EXEC_SERVER
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

    target = _prepare_enroot_sysconf(tmp_path / "target", source)

    assert target == tmp_path / "target"
    patched = (target / "mounts.d" / "20-config.fstab").read_text(encoding="utf-8")
    assert "/etc/hosts /etc/hosts" in patched
    assert "/etc/localtime" not in patched


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
    )
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
    assert "--port 0" in command[-1]


def test_transient_startup_error_detects_cloud_node_boot_failures() -> None:
    assert _is_transient_startup_error("srun: error: Node failure on m7i-cpu2-dy-0")
    assert _is_transient_startup_error("Nodes m7i-cpu2-dy-0 are still not ready")
    assert _is_transient_startup_error("Something is wrong with the boot of the nodes.")
    assert not _is_transient_startup_error("FATAL: cannot install /usr/bin/python3")


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

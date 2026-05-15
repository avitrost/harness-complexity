from pathlib import Path

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import plumbing.slurm_pyxis_environment as slurm_pyxis
from plumbing.slurm_pyxis_environment import (
    SLURM_BOOTSTRAP,
    SlurmPyxisEnvironment,
    STDLIB_EXEC_SERVER,
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
    assert env._slurm_job_name.startswith("hb-")


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

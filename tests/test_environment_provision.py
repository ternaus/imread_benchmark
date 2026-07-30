from __future__ import annotations

import json
import sys
from pathlib import Path

from imread_benchmark.environments import EnvironmentDescriptor, load_environment_descriptor
from imread_benchmark.environments.provision import EnvironmentRequest, provision_environment


def test_frozen_environment_is_built_atomically_and_reused(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("fixture lock\n")
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    log_path = tmp_path / "uv-calls.json"
    monkeypatch.setenv("FAKE_UV_LOG", str(log_path))
    request = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
        native_backends=(("fixture", "1.0"),),
        uv_command=(sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uv.py")),
    )

    def probe(_python: Path, key: str) -> EnvironmentDescriptor:
        return EnvironmentDescriptor.build(
            dependency_group=request.dependency_group,
            lock_sha256=request.lock_sha256,
            project_sha256=request.project_sha256,
            runner_revision=request.runner_revision,
            python=request.python_identity,
            platform_tags=request.platform_tags,
            distributions=(("fixture", "1.0"),),
            native_backends=dict(request.native_backends),
        )

    first = provision_environment(request, probe=probe)
    second = provision_environment(request, probe=probe)

    assert first == second
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.python_executable.is_file()
    assert load_environment_descriptor(first.descriptor_path).environment_id == first.environment_id
    marker = json.loads((first.root / ".READY.json").read_text())
    assert marker["environment_key"] == request.environment_key
    assert marker["environment_id"] == first.environment_id
    calls = json.loads(log_path.read_text())
    assert len(calls) == 1
    assert "--frozen" in calls[0]["argv"]
    assert "--no-editable" in calls[0]["argv"]
    assert calls[0]["argv"][-2:] == ["--extra", "mainstream"]


def test_environment_key_changes_with_lock_python_group_revision_or_native_backend(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("first\n")
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    base = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
    )
    (project / "uv.lock").write_text("second\n")
    lock_changed = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
    )
    group_changed = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="tensorflow",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
    )
    revision_changed = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="2" * 40,
        python_executable=Path(sys.executable),
    )
    backend_changed = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
        native_backends=(("libjpeg-turbo", "3.2.0"),),
    )

    keys = {
        base.environment_key,
        lock_changed.environment_key,
        group_changed.environment_key,
        revision_changed.environment_key,
        backend_changed.environment_key,
    }
    assert len(keys) == 5

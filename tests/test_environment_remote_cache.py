from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from imread_benchmark.datasets.materializer import LocalObjectStore, ObjectMetadata
from imread_benchmark.environments import EnvironmentDescriptor
from imread_benchmark.environments.cache import materialize_environment_cache, publish_environment_cache
from imread_benchmark.environments.provision import EnvironmentRequest, provision_environment


@dataclass
class _RacingMarkerStore:
    base: LocalObjectStore
    raced: bool = False

    def put_create_only(self, source: Path, key: str) -> None:
        if "/keys/" in key and not self.raced:
            self.raced = True
            document = json.loads(source.read_text())
            document["race_winner"] = True
            with tempfile.TemporaryDirectory() as directory:
                winner = Path(directory) / "marker.json"
                winner.write_text(json.dumps(document, sort_keys=True))
                self.base.put_create_only(winner, key)
        self.base.put_create_only(source, key)

    def download(self, key: str, destination: Path) -> None:
        self.base.download(key, destination)

    def metadata(self, key: str) -> ObjectMetadata:
        return self.base.metadata(key)


def test_remote_environment_cache_is_content_addressed_verified_and_reusable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("fixture lock\n")
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    monkeypatch.setenv("FAKE_UV_LOG", str(tmp_path / "uv-calls.json"))
    fake_uv = (sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uv.py"))
    first_request = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache-one",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
        uv_command=fake_uv,
    )

    def probe(_python: Path, _key: str) -> EnvironmentDescriptor:
        return EnvironmentDescriptor.build(
            dependency_group=first_request.dependency_group,
            lock_sha256=first_request.lock_sha256,
            project_sha256=first_request.project_sha256,
            runner_revision=first_request.runner_revision,
            python=first_request.python_identity,
            platform_tags=first_request.platform_tags,
            distributions=(("fixture", "1.0"),),
            native_backends={},
        )

    built = provision_environment(first_request, probe=probe)
    store = LocalObjectStore(tmp_path / "object-store")
    marker_key = publish_environment_cache(built.root, store=store, prefix="environments")
    assert publish_environment_cache(built.root, store=store, prefix="environments") == marker_key

    second_request = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache-two",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
        uv_command=fake_uv,
    )
    restored = materialize_environment_cache(second_request, store=store, prefix="environments")

    assert restored is not None
    assert restored.environment_id == built.environment_id
    assert restored.environment_key == built.environment_key
    assert restored.python_executable.is_file()
    assert restored.cache_hit is True


def test_concurrent_environment_marker_accepts_the_first_matching_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("fixture lock\n")
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    monkeypatch.setenv("FAKE_UV_LOG", str(tmp_path / "uv-calls.json"))
    request = EnvironmentRequest(
        project_root=project,
        cache_root=tmp_path / "cache",
        dependency_group="mainstream",
        runner_revision="1" * 40,
        python_executable=Path(sys.executable),
        uv_command=(sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uv.py")),
    )

    def probe(_python: Path, _key: str) -> EnvironmentDescriptor:
        return EnvironmentDescriptor.build(
            dependency_group=request.dependency_group,
            lock_sha256=request.lock_sha256,
            project_sha256=request.project_sha256,
            runner_revision=request.runner_revision,
            python=request.python_identity,
            platform_tags=request.platform_tags,
            distributions=(("fixture", "1.0"),),
            native_backends={},
        )

    built = provision_environment(request, probe=probe)
    store = _RacingMarkerStore(LocalObjectStore(tmp_path / "object-store"))

    marker_key = publish_environment_cache(built.root, store=store, prefix="environments")

    assert marker_key.endswith(f"keys/{request.environment_key}.json")
    assert store.raced is True

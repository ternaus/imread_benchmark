from __future__ import annotations

import json
from pathlib import Path

import pytest

from imread_benchmark.artifacts import (
    RemoteArtifactError,
    publish_run_bundle,
    pull_committed_run,
    validate_run_bundle,
    write_run_bundle,
)
from imread_benchmark.datasets.materializer import LocalObjectStore
from tests.factories import valid_bundle_data


def _bundle(root: Path) -> tuple[str, Path]:
    run_key, data = valid_bundle_data()
    return run_key, write_run_bundle(
        root=root,
        run_key=run_key,
        data=data,
    )


def test_remote_commit_is_last_and_fresh_machine_resumes_without_rewriting_bundle(tmp_path: Path) -> None:
    run_key, source = _bundle(tmp_path / "source-runs")
    store = LocalObjectStore(tmp_path / "object-store")

    publish_run_bundle(source, store=store)
    marker = tmp_path / "object-store" / "artifacts" / "runs" / run_key / "COMMITTED.json"
    assert marker.is_file()

    pulled = pull_committed_run(run_key, store=store, artifact_root=tmp_path / "fresh-machine")
    assert pulled is not None
    validate_run_bundle(pulled, expected_run_key=run_key)
    mtime = (pulled / "COMMITTED.json").stat().st_mtime_ns
    assert pull_committed_run(run_key, store=store, artifact_root=tmp_path / "fresh-machine") == pulled
    assert (pulled / "COMMITTED.json").stat().st_mtime_ns == mtime


def test_incomplete_remote_prefix_is_ignored_but_broken_commit_is_terminal(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "object-store")
    incomplete_key = "1" * 64
    incomplete = tmp_path / "object-store" / "artifacts" / "bundles" / ("2" * 64)
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text("{}")
    assert pull_committed_run(incomplete_key, store=store, artifact_root=tmp_path / "fresh") is None

    run_key, source = _bundle(tmp_path / "source-runs")
    publish_run_bundle(source, store=store)
    marker = tmp_path / "object-store" / "artifacts" / "runs" / run_key / "COMMITTED.json"
    document = json.loads(marker.read_text())
    document["bundle_id"] = "9" * 64
    marker.write_text(json.dumps(document))

    with pytest.raises(RemoteArtifactError, match="missing"):
        pull_committed_run(run_key, store=store, artifact_root=tmp_path / "broken-fresh")

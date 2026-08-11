from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from imread_benchmark.artifacts.bundle import REMOTE_BUNDLE_FILES, validate_run_bundle
from imread_benchmark.datasets.materializer import LocalObjectStore, ObjectNotFoundError, ObjectStore


class RemoteArtifactError(RuntimeError):
    pass


def publish_run_bundle(
    bundle: str | Path,
    *,
    store: ObjectStore,
    prefix: str = "artifacts",
) -> None:
    bundle_path = Path(bundle).resolve()
    commit = _read_object(bundle_path / "COMMITTED.json")
    run_key = _required_string(commit, "run_key")
    bundle_id = _required_string(commit, "bundle_id")
    validate_run_bundle(bundle_path, expected_run_key=run_key)
    for name in REMOTE_BUNDLE_FILES:
        store.put_create_only(bundle_path / name, f"{prefix}/bundles/{bundle_id}/{name}")
    store.put_create_only(bundle_path / "COMMITTED.json", f"{prefix}/runs/{run_key}/COMMITTED.json")


def pull_committed_run(
    run_key: str,
    *,
    store: ObjectStore,
    artifact_root: str | Path,
    prefix: str = "artifacts",
) -> Path | None:
    root = Path(artifact_root).resolve()
    destination = root / "runs" / run_key
    if destination.exists():
        validate_run_bundle(destination, expected_run_key=run_key)
        return destination
    root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".pull-{run_key}.", dir=root))
    bundle_staging = staging_root / run_key
    bundle_staging.mkdir()
    try:
        return _pull_into_staging(
            run_key,
            store=store,
            prefix=prefix,
            staging=bundle_staging,
            destination=destination,
        )
    except RemoteArtifactError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise RemoteArtifactError(f"cannot materialize committed run {run_key}: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def hydrate_committed_runs(
    *,
    source_artifact_root: str | Path,
    destination_artifact_root: str | Path,
) -> tuple[Path, ...]:
    """Materialize a downloaded remote artifact layout into local run bundles."""
    source_root = Path(source_artifact_root).resolve()
    destination_root = Path(destination_artifact_root).resolve()
    markers_root = source_root / "runs"
    if not markers_root.is_dir():
        raise RemoteArtifactError(f"downloaded artifact root has no runs directory: {source_root}")
    if source_root == destination_root:
        raise RemoteArtifactError("source and destination artifact roots must differ")

    store = LocalObjectStore(source_root.parent)
    hydrated: list[Path] = []
    for marker in sorted(markers_root.glob("*/COMMITTED.json")):
        run_key = marker.parent.name
        bundle = pull_committed_run(
            run_key,
            store=store,
            artifact_root=destination_root,
            prefix=source_root.name,
        )
        if bundle is None:
            raise RemoteArtifactError(f"downloaded commit marker disappeared for run {run_key}")
        hydrated.append(bundle)
    if not hydrated:
        raise RemoteArtifactError(f"downloaded artifact root has no committed runs: {source_root}")
    return tuple(hydrated)


def _pull_into_staging(
    run_key: str,
    *,
    store: ObjectStore,
    prefix: str,
    staging: Path,
    destination: Path,
) -> Path | None:
    marker_key = f"{prefix}/runs/{run_key}/COMMITTED.json"
    try:
        store.download(marker_key, staging / "COMMITTED.json")
    except ObjectNotFoundError:
        return None
    commit = _read_object(staging / "COMMITTED.json")
    if _required_string(commit, "run_key") != run_key:
        raise RemoteArtifactError("remote commit marker run_key mismatch")
    bundle_id = _required_string(commit, "bundle_id")
    for name in REMOTE_BUNDLE_FILES:
        try:
            store.download(f"{prefix}/bundles/{bundle_id}/{name}", staging / name)
        except ObjectNotFoundError as exc:
            raise RemoteArtifactError(f"committed remote bundle is missing {name}") from exc
    validate_run_bundle(staging, expected_run_key=run_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(destination)
    except FileExistsError:
        validate_run_bundle(destination, expected_run_key=run_key)
    return destination


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RemoteArtifactError(f"remote artifact field {key!r} must be a non-empty string")
    return value

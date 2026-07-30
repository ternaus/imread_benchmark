from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard

from imread_benchmark.datasets.materializer import ObjectConflictError, ObjectNotFoundError, ObjectStore
from imread_benchmark.environments.descriptor import load_environment_descriptor
from imread_benchmark.environments.provision import (
    EnvironmentRequest,
    ProvisionedEnvironment,
    load_provisioned_environment,
)

ENVIRONMENT_CACHE_SCHEMA_VERSION = "2.0"


def publish_environment_cache(
    environment_root: str | Path,
    *,
    store: ObjectStore,
    prefix: str,
) -> str:
    root = Path(environment_root).resolve()
    ready = _read_object(root / ".READY.json")
    environment_key = _required_string(ready, "environment_key")
    environment_id = _required_string(ready, "environment_id")
    descriptor = load_environment_descriptor(root / "environment.json")
    if descriptor.environment_id != environment_id:
        raise ValueError("environment cache READY marker does not match descriptor")
    marker_key = _key(prefix, "keys", f"{environment_key}.json")
    existing = _download_optional_json(store, marker_key)
    if existing is not None:
        _validate_marker(existing, environment_key=environment_key, environment_id=environment_id)
        return marker_key

    with tempfile.TemporaryDirectory(prefix="environment-cache-publish.") as temporary_directory:
        archive = Path(temporary_directory) / "environment.tar.zst"
        _write_deterministic_archive(root, archive)
        archive_sha256 = _sha256_file(archive)
        archive_key = _key(prefix, "blobs", f"{archive_sha256}.tar.zst")
        store.put_create_only(archive, archive_key)
        marker_path = Path(temporary_directory) / "marker.json"
        marker = {
            "archive": {
                "bytes": archive.stat().st_size,
                "key": archive_key,
                "sha256": archive_sha256,
            },
            "environment_id": environment_id,
            "environment_key": environment_key,
            "schema_version": ENVIRONMENT_CACHE_SCHEMA_VERSION,
            "status": "committed",
        }
        marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        try:
            store.put_create_only(marker_path, marker_key)
        except ObjectConflictError:
            winner = _download_optional_json(store, marker_key)
            if winner is None:
                raise
            _validate_marker(winner, environment_key=environment_key, environment_id=environment_id)
    return marker_key


def materialize_environment_cache(
    request: EnvironmentRequest,
    *,
    store: ObjectStore,
    prefix: str,
) -> ProvisionedEnvironment | None:
    destination = request.cache_root / request.environment_key
    if destination.exists():
        return load_provisioned_environment(destination, request, cache_hit=True)
    marker_key = _key(prefix, "keys", f"{request.environment_key}.json")
    marker = _download_optional_json(store, marker_key)
    if marker is None:
        return None
    environment_id = _required_string(marker, "environment_id")
    _validate_marker(marker, environment_key=request.environment_key, environment_id=environment_id)
    archive_metadata = marker.get("archive")
    if not isinstance(archive_metadata, dict):
        raise TypeError("environment cache marker archive must be an object")
    archive_key = _required_string(archive_metadata, "key")
    archive_sha256 = _required_string(archive_metadata, "sha256")
    archive_bytes = archive_metadata.get("bytes")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise ValueError("environment cache marker archive bytes must be positive")

    request.cache_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{request.environment_key}.", dir=request.cache_root))
    try:
        compressed = staging_parent / "environment.tar.zst"
        store.download(archive_key, compressed)
        if compressed.stat().st_size != archive_bytes or _sha256_file(compressed) != archive_sha256:
            raise ValueError("environment cache archive checksum mismatch")
        tar_path = staging_parent / "environment.tar"
        _decompress_archive(compressed, tar_path)
        staging = staging_parent / "environment"
        staging.mkdir()
        _extract_safe_tar(tar_path, staging)
        restored = load_provisioned_environment(staging, request, cache_hit=True)
        if restored.environment_id != environment_id:
            raise ValueError("restored environment_id does not match remote marker")
        with suppress(FileExistsError):
            staging.rename(destination)
        return load_provisioned_environment(destination, request, cache_hit=True)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _write_deterministic_archive(root: Path, destination: Path) -> None:
    compressor = zstandard.ZstdCompressor(level=3, threads=0)
    with (
        destination.open("wb") as compressed,
        compressor.stream_writer(compressed, closefd=False) as stream,
        tarfile.open(fileobj=stream, mode="w|", format=tarfile.GNU_FORMAT, dereference=True) as archive,
    ):
        for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
            archive.add(
                path,
                arcname=path.relative_to(root).as_posix(),
                recursive=False,
                filter=_normalized_tar_info,
            )


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _decompress_archive(source: Path, destination: Path) -> None:
    decompressor = zstandard.ZstdDecompressor()
    with source.open("rb") as compressed, destination.open("wb") as target:
        decompressor.copy_stream(compressed, target)


def _extract_safe_tar(source: Path, destination: Path) -> None:
    with tarfile.open(source, mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError(f"unsafe environment cache member: {member.name!r}")
            if not (member.isdir() or member.isfile() or member.issym()):
                raise ValueError(f"unsupported environment cache member type: {member.name!r}")
        archive.extractall(destination, filter="data")


def _download_optional_json(store: ObjectStore, key: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix="environment-cache-marker.") as directory:
        path = Path(directory) / "marker.json"
        try:
            store.download(key, path)
        except ObjectNotFoundError:
            return None
        return _read_object(path)


def _validate_marker(document: dict[str, Any], *, environment_key: str, environment_id: str) -> None:
    if (
        document.get("schema_version") != ENVIRONMENT_CACHE_SCHEMA_VERSION
        or document.get("status") != "committed"
        or document.get("environment_key") != environment_key
        or document.get("environment_id") != environment_id
    ):
        raise ValueError("environment cache marker identity mismatch")


def _key(*parts: str) -> str:
    path = PurePosixPath(*parts)
    if path.is_absolute() or not path.parts or "." in path.parts or ".." in path.parts:
        raise ValueError(f"unsafe environment cache object key: {path}")
    return str(path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read environment cache JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("environment cache JSON must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"environment cache field {key!r} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

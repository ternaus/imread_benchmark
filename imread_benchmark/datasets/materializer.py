from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from imread_benchmark.datasets.package import DatasetPackageError, open_dataset_package


class ObjectStoreError(RuntimeError):
    pass


class ObjectNotFoundError(ObjectStoreError):
    pass


class ObjectConflictError(ObjectStoreError):
    pass


class MaterializationError(ObjectStoreError):
    pass


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    size: int
    generation: str


class ObjectStore(Protocol):
    def put_create_only(self, source: Path, key: str) -> None: ...

    def download(self, key: str, destination: Path) -> None: ...

    def metadata(self, key: str) -> ObjectMetadata: ...


@dataclass(frozen=True, slots=True)
class LocalObjectStore:
    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root).resolve())

    def put_create_only(self, source: Path, key: str) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256_file(destination) != _sha256_file(source):
                raise ObjectConflictError(f"remote object already exists with different content: {key}")
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if _sha256_file(destination) != _sha256_file(source):
                    raise ObjectConflictError(f"concurrent writer published different content: {key}") from None
        finally:
            temporary.unlink(missing_ok=True)

    def download(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise ObjectNotFoundError(f"remote object does not exist: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def metadata(self, key: str) -> ObjectMetadata:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"remote object does not exist: {key}")
        stat = path.stat()
        return ObjectMetadata(size=stat.st_size, generation=f"{stat.st_ino}:{stat.st_mtime_ns}")

    def _path(self, key: str) -> Path:
        relative = _safe_key(key)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise MaterializationError(f"object key escapes store root: {key!r}")
        return path


def publish_dataset_package(
    descriptor_path: str | Path,
    *,
    store: ObjectStore,
    prefix: str,
) -> str:
    descriptor_source = Path(descriptor_path).resolve()
    try:
        package = open_dataset_package(descriptor_source)
    except DatasetPackageError as exc:
        raise MaterializationError(f"cannot publish invalid dataset package: {exc}") from exc
    package_id = _required_string(package.descriptor, "package_id")
    remote_root = _join_key(prefix, package_id)
    relative_files = _package_files(package.descriptor)
    for relative in relative_files:
        source = package.root / relative
        key = _join_key(remote_root, relative)
        store.put_create_only(source, key)
        metadata = store.metadata(key)
        if metadata.size != source.stat().st_size:
            raise MaterializationError(f"remote object size mismatch after upload: {key}")
    return _join_key(remote_root, "package.json")


def materialize_dataset_package(
    remote_descriptor: str,
    *,
    store: ObjectStore,
    cache_root: str | Path,
) -> Path:
    cache = Path(cache_root).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".materialize.", dir=cache))
    try:
        bootstrap_descriptor = staging_root / "package.json"
        store.download(remote_descriptor, bootstrap_descriptor)
        descriptor = _read_object(bootstrap_descriptor)
        package_id = _required_string(descriptor, "package_id")
        destination = cache / package_id
        with _FileLock(cache / f".{package_id}.lock"):
            if destination.exists():
                _open_ready(destination / "package.json")
                return destination / "package.json"
            package_staging = staging_root / package_id
            package_staging.mkdir()
            bootstrap_descriptor.rename(package_staging / "package.json")
            remote_root = str(PurePosixPath(_safe_key(remote_descriptor)).parent)
            for relative in _package_files(descriptor):
                if relative == "package.json":
                    continue
                store.download(_join_key(remote_root, relative), package_staging / relative)
            _verify_and_mark(package_staging / "package.json")
            package_staging.rename(destination)
            return destination / "package.json"
    except (DatasetPackageError, OSError, TypeError, ValueError) as exc:
        raise MaterializationError(f"dataset materialization checksum or identity failure: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _verify_and_mark(descriptor_path: Path) -> None:
    package = open_dataset_package(descriptor_path)
    package_id = _required_string(package.descriptor, "package_id")
    workloads = package.descriptor.get("workloads")
    if not isinstance(workloads, dict):
        raise MaterializationError("dataset descriptor has no workloads object")
    marker_root = package.root / ".CONTENT_VERIFIED"
    marker_root.mkdir()
    for workload_id, workload in workloads.items():
        if not isinstance(workload_id, str) or not isinstance(workload, dict):
            raise MaterializationError("dataset descriptor has an invalid workload")
        items = package.read_workload_items(workload_id)
        _write_json(
            marker_root / f"{workload_id}.json",
            {
                "item_count": len(items),
                "manifest_id": workload.get("manifest_id"),
                "package_id": package_id,
                "schema_version": "2.0",
            },
        )
    archive = package.descriptor.get("archive")
    if not isinstance(archive, dict):
        raise MaterializationError("dataset descriptor has no archive object")
    archive_path = package.root / _required_string(archive, "file")
    archive_stat = archive_path.stat()
    _write_json(
        package.root / ".READY.json",
        {
            "archive": {
                "bytes": archive_stat.st_size,
                "file": archive_path.name,
                "inode": archive_stat.st_ino,
                "mtime_ns": archive_stat.st_mtime_ns,
                "sha256": archive.get("sha256"),
            },
            "descriptor_sha256": _sha256_file(descriptor_path),
            "package_id": package_id,
            "schema_version": "2.0",
        },
    )
    for path in package.root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)


def _open_ready(descriptor_path: Path) -> None:
    try:
        open_dataset_package(descriptor_path, trust_ready=True)
    except DatasetPackageError as exc:
        raise MaterializationError(f"existing materialized package is invalid: {exc}") from exc


def _package_files(descriptor: dict[str, Any]) -> tuple[str, ...]:
    archive = descriptor.get("archive")
    archive_index = descriptor.get("archive_index")
    workloads = descriptor.get("workloads")
    if not isinstance(archive, dict) or not isinstance(archive_index, dict) or not isinstance(workloads, dict):
        raise MaterializationError("dataset descriptor is missing package file references")
    files = {
        "package.json",
        _required_string(archive, "file"),
        _required_string(archive_index, "file"),
    }
    for workload in workloads.values():
        if not isinstance(workload, dict):
            raise MaterializationError("dataset descriptor has an invalid workload")
        files.add(_required_string(workload, "manifest"))
    for relative in files:
        _safe_key(relative)
    return tuple(sorted(files))


class _FileLock(AbstractContextManager[None]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: Any = None

    def __enter__(self) -> None:
        self.file = self.path.open("a+b")
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.file is None:
            return
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


def _safe_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise MaterializationError(f"unsafe object key: {key!r}")
    return str(path)


def _join_key(*parts: str) -> str:
    return _safe_key(str(PurePosixPath(*parts)))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"field {key!r} must be a non-empty string")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

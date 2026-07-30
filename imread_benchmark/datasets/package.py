from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from imread_benchmark.datasets.manifest import DatasetManifest

if TYPE_CHECKING:
    from collections.abc import Mapping

PACKAGE_SCHEMA_VERSION = "2.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetPackageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResidentItem:
    item_id: str
    relative_path: str
    sha256: str
    compressed_bytes: int
    data: bytes


@dataclass(frozen=True, slots=True)
class DatasetPackageReader:
    root: Path
    descriptor: dict[str, Any]
    archive_index: dict[str, dict[str, int | str]]
    verify_item_hashes: bool = True

    def read_workload(self, workload_id: str) -> tuple[bytes, ...]:
        return tuple(item.data for item in self.read_workload_items(workload_id))

    def read_workload_items(self, workload_id: str) -> tuple[ResidentItem, ...]:
        workload = _workload(self.descriptor, workload_id)
        manifest_path = _resolve_member(self.root, _required_string(workload, "manifest"))
        manifest = _read_object(manifest_path)
        items = manifest.get("items")
        if not isinstance(items, list):
            raise DatasetPackageError(f"manifest for {workload_id!r} has no items list")

        archive = _archive_path(self.root, self.descriptor)
        blobs: dict[str, bytes] = {}
        with archive.open("rb") as file:
            for raw_item in items:
                if not isinstance(raw_item, dict):
                    raise DatasetPackageError(f"manifest for {workload_id!r} contains a non-object item")
                digest = _required_string(raw_item, "sha256")
                if digest in blobs:
                    continue
                entry = self.archive_index.get(digest)
                if entry is None:
                    raise DatasetPackageError(f"archive index has no blob {digest}")
                offset = entry.get("offset")
                size = entry.get("size")
                if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size <= 0:
                    raise DatasetPackageError(f"archive index entry for {digest} has invalid offset or size")
                file.seek(offset)
                data = file.read(size)
                if len(data) != size or (self.verify_item_hashes and hashlib.sha256(data).hexdigest() != digest):
                    raise DatasetPackageError(f"archive blob {digest} failed content verification")
                blobs[digest] = data
        return tuple(
            ResidentItem(
                item_id=_required_string(item, "item_id"),
                relative_path=_required_string(item, "relative_path"),
                sha256=_required_string(item, "sha256"),
                compressed_bytes=_required_positive_int(item, "compressed_bytes"),
                data=blobs[_required_string(item, "sha256")],
            )
            for item in items
            if isinstance(item, dict)
        )


def build_dataset_package(
    *,
    package_name: str,
    workloads: Mapping[str, str | Path],
    output_root: str | Path,
    provenance: Mapping[str, object],
) -> Path:
    _validate_id(package_name, field="package_name")
    if not workloads:
        raise DatasetPackageError("at least one workload is required")
    resolved_workloads = _resolve_workloads(workloads)

    destination_root = Path(output_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-package.", dir=destination_root))
    try:
        manifests, manifest_files, manifest_ids, blob_sources = _build_manifests(resolved_workloads, staging)

        temporary_archive = staging / "payload.tar"
        with tarfile.open(temporary_archive, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for digest, source in sorted(blob_sources.items()):
                _add_file(archive, source, f"blobs/{digest}.jpg")
            for manifest_file in sorted(manifest_files.values()):
                _add_bytes(archive, (staging / manifest_file).read_bytes(), manifest_file)

        archive_index = _build_archive_index(temporary_archive)
        index_path = staging / "archive-index.json"
        _write_json(
            index_path,
            {
                "blobs": archive_index,
                "schema_version": PACKAGE_SCHEMA_VERSION,
            },
        )
        archive_sha256 = _sha256_file(temporary_archive)
        archive_name = f"{archive_sha256}.tar"
        temporary_archive.rename(staging / archive_name)
        index_sha256 = _sha256_file(index_path)

        descriptor_without_id: dict[str, object] = {
            "archive": {
                "bytes": (staging / archive_name).stat().st_size,
                "file": archive_name,
                "format": "tar",
                "sha256": archive_sha256,
            },
            "archive_index": {
                "file": index_path.name,
                "sha256": index_sha256,
            },
            "package_name": package_name,
            "provenance": dict(provenance),
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "workloads": {
                workload_id: {
                    "item_count": len(manifest.items),
                    "manifest": manifest_files[workload_id],
                    "manifest_id": manifest_ids[workload_id],
                }
                for workload_id, manifest in sorted(manifests.items())
            },
        }
        package_id = _digest_json(descriptor_without_id)
        descriptor = {**descriptor_without_id, "package_id": package_id}
        _write_json(staging / "package.json", descriptor)

        destination = destination_root / package_id
        if destination.exists():
            existing = _read_object(destination / "package.json")
            if existing != descriptor:
                raise DatasetPackageError(f"existing package {package_id} has different content")
            return destination / "package.json"
        staging.rename(destination)
        return destination / "package.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _resolve_workloads(workloads: Mapping[str, str | Path]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for workload_id, root in workloads.items():
        _validate_id(workload_id, field="workload_id")
        path = Path(root).resolve()
        if not path.is_dir():
            raise DatasetPackageError(f"workload directory does not exist: {path}")
        resolved[workload_id] = path
    return resolved


def _build_manifests(
    workloads: Mapping[str, Path],
    staging: Path,
) -> tuple[dict[str, DatasetManifest], dict[str, str], dict[str, str], dict[str, Path]]:
    manifests: dict[str, DatasetManifest] = {}
    manifest_files: dict[str, str] = {}
    manifest_ids: dict[str, str] = {}
    blob_sources: dict[str, Path] = {}
    for workload_id, root in sorted(workloads.items()):
        manifest = DatasetManifest.build(root, dataset_name=workload_id)
        manifests[workload_id] = manifest
        manifest_file = f"manifests/{workload_id}.json"
        manifest_files[workload_id] = manifest_file
        manifest_payload = _package_manifest(manifest)
        manifest_ids[workload_id] = _required_string(manifest_payload, "manifest_id")
        _write_json(staging / manifest_file, manifest_payload)
        for item in manifest.items:
            blob_sources.setdefault(item.sha256, manifest.resolve(item))
    return manifests, manifest_files, manifest_ids, blob_sources


def _package_manifest(manifest: DatasetManifest) -> dict[str, object]:
    payload = {
        "dataset_name": manifest.dataset_name,
        "items": [item.to_dict() for item in manifest.items],
        "schema_version": PACKAGE_SCHEMA_VERSION,
    }
    return {**payload, "manifest_id": _digest_json(payload)}


def open_dataset_package(
    descriptor_path: str | Path,
    *,
    trust_ready: bool = False,
) -> DatasetPackageReader:
    path = Path(descriptor_path).resolve()
    descriptor = _read_object(path)
    if descriptor.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise DatasetPackageError(f"unsupported dataset package schema: {descriptor.get('schema_version')!r}")
    package_id = descriptor.get("package_id")
    if not isinstance(package_id, str):
        raise DatasetPackageError("dataset package has no package_id")
    without_id = {key: value for key, value in descriptor.items() if key != "package_id"}
    if _digest_json(without_id) != package_id or path.parent.name != package_id:
        raise DatasetPackageError("dataset package identity mismatch")

    archive_path = _archive_path(path.parent, descriptor)
    archive = _required_object(descriptor, "archive")
    if trust_ready:
        _validate_ready_package(path, descriptor, archive_path)
    elif archive_path.stat().st_size != archive.get("bytes") or _sha256_file(archive_path) != archive.get("sha256"):
        raise DatasetPackageError("dataset package archive checksum mismatch")

    index_descriptor = _required_object(descriptor, "archive_index")
    index_path = _resolve_member(path.parent, _required_string(index_descriptor, "file"))
    if _sha256_file(index_path) != index_descriptor.get("sha256"):
        raise DatasetPackageError("dataset package archive index checksum mismatch")
    index_document = _read_object(index_path)
    blobs = index_document.get("blobs")
    if not isinstance(blobs, dict):
        raise DatasetPackageError("dataset package archive index has no blobs object")
    if not trust_ready:
        _audit_tar_members(archive_path)
        if _build_archive_index(archive_path) != blobs:
            raise DatasetPackageError("dataset package archive index does not match tar members")
    _validate_workload_manifests(path.parent, descriptor)
    return DatasetPackageReader(
        root=path.parent,
        descriptor=descriptor,
        archive_index=blobs,
        verify_item_hashes=not trust_ready,
    )


def _validate_ready_package(path: Path, descriptor: dict[str, Any], archive_path: Path) -> None:
    ready = _read_object(path.parent / ".READY.json")
    archive_stat = archive_path.stat()
    archive_ready = ready.get("archive")
    workloads = _required_object(descriptor, "workloads")
    if (
        ready.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or ready.get("package_id") != descriptor.get("package_id")
        or ready.get("descriptor_sha256") != _sha256_file(path)
        or not isinstance(archive_ready, dict)
        or archive_ready.get("file") != archive_path.name
        or archive_ready.get("bytes") != archive_stat.st_size
        or archive_ready.get("mtime_ns") != archive_stat.st_mtime_ns
        or archive_ready.get("inode") != archive_stat.st_ino
    ):
        raise DatasetPackageError("dataset package .READY identity mismatch")
    for workload_id, workload in workloads.items():
        if not isinstance(workload, dict):
            raise DatasetPackageError(f"dataset package workload {workload_id!r} must be an object")
        marker = _read_object(path.parent / ".CONTENT_VERIFIED" / f"{workload_id}.json")
        if (
            marker.get("package_id") != descriptor.get("package_id")
            or marker.get("manifest_id") != workload.get("manifest_id")
            or marker.get("item_count") != workload.get("item_count")
        ):
            raise DatasetPackageError(f"workload {workload_id!r} content verification marker mismatch")


def _audit_tar_members(path: Path) -> None:
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                raise DatasetPackageError(f"unsafe tar member: {member.name!r}")


def _validate_workload_manifests(root: Path, descriptor: dict[str, Any]) -> None:
    workloads = _required_object(descriptor, "workloads")
    for workload_id, raw_workload in workloads.items():
        if not isinstance(raw_workload, dict):
            raise DatasetPackageError(f"dataset package workload {workload_id!r} must be an object")
        manifest = _read_object(_resolve_member(root, _required_string(raw_workload, "manifest")))
        stored_manifest_id = manifest.get("manifest_id")
        manifest_without_id = {key: value for key, value in manifest.items() if key != "manifest_id"}
        expected_manifest_id = raw_workload.get("manifest_id")
        items = manifest.get("items")
        if (
            manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
            or not isinstance(stored_manifest_id, str)
            or _digest_json(manifest_without_id) != stored_manifest_id
            or stored_manifest_id != expected_manifest_id
            or not isinstance(items, list)
            or len(items) != raw_workload.get("item_count")
        ):
            raise DatasetPackageError(f"workload {workload_id!r} manifest identity mismatch")


def _build_archive_index(path: Path) -> dict[str, dict[str, int | str]]:
    index: dict[str, dict[str, int | str]] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith("blobs/"):
                continue
            digest = Path(member.name).stem
            index[digest] = {
                "member": member.name,
                "offset": member.offset_data,
                "size": member.size,
            }
    return index


def _add_file(archive: tarfile.TarFile, source: Path, member_name: str) -> None:
    info = _tar_info(member_name, source.stat().st_size)
    with source.open("rb") as file:
        archive.addfile(info, file)


def _add_bytes(archive: tarfile.TarFile, data: bytes, member_name: str) -> None:
    archive.addfile(_tar_info(member_name, len(data)), io.BytesIO(data))


def _tar_info(member_name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(member_name)
    info.size = size
    info.mode = 0o444
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _archive_path(root: Path, descriptor: dict[str, Any]) -> Path:
    archive = _required_object(descriptor, "archive")
    return _resolve_member(root, _required_string(archive, "file"))


def _workload(descriptor: dict[str, Any], workload_id: str) -> dict[str, Any]:
    workloads = _required_object(descriptor, "workloads")
    workload = workloads.get(workload_id)
    if not isinstance(workload, dict):
        raise DatasetPackageError(f"dataset package has no workload {workload_id!r}")
    return workload


def _resolve_member(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise DatasetPackageError(f"package path escapes root: {relative_path!r}")
    if not candidate.is_file():
        raise DatasetPackageError(f"package file does not exist: {relative_path!r}")
    return candidate


def _validate_id(value: str, *, field: str) -> None:
    if _SAFE_ID.fullmatch(value) is None or value in {".", ".."}:
        raise DatasetPackageError(f"{field} must be a safe identifier, got {value!r}")


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise DatasetPackageError(f"dataset package field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetPackageError(f"dataset package field {key!r} must be a non-empty string")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetPackageError(f"dataset package field {key!r} must be a positive integer")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPackageError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetPackageError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload, pretty=True))


def _canonical_json(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _digest_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

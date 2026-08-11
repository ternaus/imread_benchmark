from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

from imread_benchmark.datasets.package import DatasetPackageError, build_dataset_package, open_dataset_package


def test_package_reads_manifest_order_directly_from_content_addressed_tar(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )

    descriptor = json.loads(descriptor_path.read_text())
    package_root = descriptor_path.parent
    archive_path = package_root / descriptor["archive"]["file"]
    reader = open_dataset_package(descriptor_path)

    assert descriptor["schema_version"] == "2.0"
    assert descriptor_path == package_root / "package.json"
    assert package_root.name == descriptor["package_id"]
    assert archive_path.suffix == ".tar"
    assert archive_path.is_file()
    assert not list(package_root.rglob("*.jpg"))
    workload = descriptor["workloads"]["fixture"]
    manifest = json.loads((package_root / workload["manifest"]).read_text())
    assert manifest["schema_version"] == "2.0"
    assert manifest["manifest_id"] == workload["manifest_id"]
    resident_items = reader.read_workload_items("fixture")
    assert tuple(item.item_id for item in resident_items) == tuple(item["item_id"] for item in manifest["items"])
    assert reader.read_workload("fixture") == tuple(path.read_bytes() for path in sorted(jpeg_dir.glob("*.jpg")))


def test_open_package_rejects_manifest_that_no_longer_matches_descriptor(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    manifest_path = descriptor_path.parent / descriptor["workloads"]["fixture"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["items"][0]["relative_path"] = "tampered.jpg"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DatasetPackageError, match="manifest identity mismatch"):
        open_dataset_package(descriptor_path)


def test_open_package_audits_archive_index_against_tar_members(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    package_root = descriptor_path.parent
    descriptor = json.loads(descriptor_path.read_text())
    index_path = package_root / descriptor["archive_index"]["file"]
    archive_index = json.loads(index_path.read_text())
    first_entry = next(iter(archive_index["blobs"].values()))
    first_entry["offset"] += 1
    index_path.write_text(json.dumps(archive_index, sort_keys=True))
    descriptor["archive_index"]["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    moved_root = _rewrite_package_identity(descriptor_path, descriptor)

    with pytest.raises(DatasetPackageError, match="archive index does not match tar members"):
        open_dataset_package(moved_root / "package.json")


def test_open_package_rejects_unsafe_tar_members(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    package_root = descriptor_path.parent
    descriptor = json.loads(descriptor_path.read_text())
    archive_path = package_root / descriptor["archive"]["file"]
    with tarfile.open(archive_path, mode="a") as archive:
        unsafe = tarfile.TarInfo("../escape")
        unsafe.type = tarfile.SYMTYPE
        unsafe.linkname = "../../outside"
        archive.addfile(unsafe)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    renamed_archive = archive_path.with_name(f"{archive_sha256}.tar")
    archive_path.rename(renamed_archive)
    descriptor["archive"] = {
        **descriptor["archive"],
        "bytes": renamed_archive.stat().st_size,
        "file": renamed_archive.name,
        "sha256": archive_sha256,
    }
    moved_root = _rewrite_package_identity(descriptor_path, descriptor)

    with pytest.raises(DatasetPackageError, match="unsafe tar member"):
        open_dataset_package(moved_root / "package.json")


def test_package_stores_shared_workload_content_once(tmp_path: Path, jpeg_bytes: bytes) -> None:
    corpus = tmp_path / "corpus"
    native = tmp_path / "native"
    mixed = tmp_path / "mixed"
    corpus.mkdir()
    native.mkdir()
    mixed.mkdir()
    source = corpus / "source.jpg"
    source.write_bytes(jpeg_bytes)
    os.link(source, native / "native.jpg")
    os.link(source, mixed / "first.jpg")
    os.link(source, mixed / "second.jpg")

    descriptor_path = build_dataset_package(
        package_name="hardlinked-views",
        workloads={"native": native, "mixed": mixed},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    descriptor = json.loads(descriptor_path.read_text())
    archive_path = descriptor_path.parent / descriptor["archive"]["file"]
    with tarfile.open(archive_path, mode="r:") as archive:
        blob_members = [member for member in archive if member.name.startswith("blobs/")]
    mixed_bytes = open_dataset_package(descriptor_path).read_workload("mixed")

    assert len(blob_members) == 1
    assert len(mixed_bytes) == 2
    assert mixed_bytes[0] is mixed_bytes[1]


def _rewrite_package_identity(descriptor_path: Path, descriptor: dict[str, object]) -> Path:
    package_root = descriptor_path.parent
    descriptor_without_id = {key: value for key, value in descriptor.items() if key != "package_id"}
    descriptor["package_id"] = hashlib.sha256(
        json.dumps(descriptor_without_id, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(),
    ).hexdigest()
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True))
    moved_root = package_root.parent / str(descriptor["package_id"])
    package_root.rename(moved_root)
    return moved_root

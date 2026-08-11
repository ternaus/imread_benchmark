from __future__ import annotations

import sys
from pathlib import Path

import pytest

from imread_benchmark.datasets.gcs import GcloudObjectStore
from imread_benchmark.datasets.materializer import (
    LocalObjectStore,
    MaterializationError,
    materialize_dataset_package,
    publish_dataset_package,
)
from imread_benchmark.datasets.package import build_dataset_package, open_dataset_package


def test_package_publish_and_materialize_is_verified_atomic_and_reusable(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="materializer-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "source-packages",
        provenance={"source": "pytest"},
    )
    store = LocalObjectStore(tmp_path / "object-store")
    remote_descriptor = publish_dataset_package(descriptor_path, store=store, prefix="datasets")

    materialized_descriptor = materialize_dataset_package(
        remote_descriptor,
        store=store,
        cache_root=tmp_path / "cache",
    )
    package = open_dataset_package(materialized_descriptor, trust_ready=True)
    archive_path = package.root / package.descriptor["archive"]["file"]
    archive_mtime = archive_path.stat().st_mtime_ns

    assert (package.root / ".READY.json").is_file()
    assert (package.root / ".CONTENT_VERIFIED" / "fixture.json").is_file()
    assert not list(package.root.rglob("*.jpg"))
    assert len(package.read_workload_items("fixture")) == 4

    reused = materialize_dataset_package(remote_descriptor, store=store, cache_root=tmp_path / "cache")
    assert reused == materialized_descriptor
    assert archive_path.stat().st_mtime_ns == archive_mtime


def test_materializer_rejects_corrupt_remote_archive_without_publishing_partial_package(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="corrupt-materializer-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "source-packages",
        provenance={"source": "pytest"},
    )
    store = LocalObjectStore(tmp_path / "object-store")
    remote_descriptor = publish_dataset_package(descriptor_path, store=store, prefix="datasets")
    descriptor = open_dataset_package(descriptor_path).descriptor
    package_id = descriptor["package_id"]
    archive_name = descriptor["archive"]["file"]
    assert isinstance(package_id, str)
    assert isinstance(archive_name, str)
    remote_archive = tmp_path / "object-store" / "datasets" / package_id / archive_name
    remote_archive.write_bytes(remote_archive.read_bytes()[:-1] + b"x")

    with pytest.raises(MaterializationError, match="checksum"):
        materialize_dataset_package(remote_descriptor, store=store, cache_root=tmp_path / "cache")
    assert not (tmp_path / "cache" / package_id).exists()


def test_gcloud_adapter_exercises_create_only_upload_and_download_via_real_subprocess(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="gcloud-adapter-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "source-packages",
        provenance={"source": "pytest"},
    )
    fake_gcloud = Path(__file__).parent / "fixtures" / "fake_gcloud.py"
    store = GcloudObjectStore(
        "gs://test-bucket/benchmark",
        command_prefix=(sys.executable, str(fake_gcloud)),
        environment={"FAKE_GCS_ROOT": str(tmp_path / "fake-gcs")},
    )

    remote_descriptor = publish_dataset_package(descriptor_path, store=store, prefix="datasets")
    publish_dataset_package(descriptor_path, store=store, prefix="datasets")
    materialized = materialize_dataset_package(
        remote_descriptor,
        store=store,
        cache_root=tmp_path / "cache",
    )

    package = open_dataset_package(materialized, trust_ready=True)
    assert len(package.read_workload_items("fixture")) == 4

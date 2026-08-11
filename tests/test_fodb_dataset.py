from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from imread_benchmark.datasets.fodb import (
    FODB_PROVENANCES,
    _describe_jpeg,
    complete_scene_ids,
    prepare_fodb,
    read_fodb_catalog,
    select_scene_ids,
)
from imread_benchmark.datasets.package import open_dataset_package


def _write_archive(
    path: Path,
    jpeg_bytes: bytes,
    *,
    devices: tuple[str, ...],
    scenes: tuple[str, ...],
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for device_id in devices:
            device_dir = f"{device_id}_Test_Camera_1"
            for provenance in FODB_PROVENANCES:
                for scene_id in scenes:
                    archive.writestr(
                        f"{device_dir}/{provenance}/{device_id}_img_{provenance}_{scene_id}.jpg",
                        jpeg_bytes,
                    )
        archive.writestr("inspection/check_devices/not-a-corpus-image.jpg", jpeg_bytes)


def test_catalog_ignores_inspection_and_selects_complete_scenes(tmp_path: Path, jpeg_bytes: bytes) -> None:
    archive = tmp_path / "part01.zip"
    devices = ("D01", "D02")
    _write_archive(archive, jpeg_bytes, devices=devices, scenes=("0001", "0002"))

    members = read_fodb_catalog([archive])
    complete = complete_scene_ids(members, expected_device_count=len(devices))

    assert len(members) == 2 * len(devices) * len(FODB_PROVENANCES)
    assert complete == ("0001", "0002")
    assert select_scene_ids(complete, count=1, seed=7) == select_scene_ids(complete, count=1, seed=7)


def test_prepare_fodb_rejects_a_resident_workload_above_the_byte_budget(
    tmp_path: Path,
    jpeg_bytes: bytes,
) -> None:
    archive = tmp_path / "part01.zip"
    devices = ("D01", "D02")
    _write_archive(archive, jpeg_bytes, devices=devices, scenes=("0001",))
    mixed_bytes = len(jpeg_bytes) * len(devices) * len(FODB_PROVENANCES)

    with pytest.raises(ValueError, match="resident compressed-byte budget"):
        prepare_fodb(
            [archive],
            tmp_path / "selected",
            scene_count=1,
            compressed_byte_limit=mixed_bytes - 1,
            expected_device_count=len(devices),
        )


def test_jpeg_descriptor_stops_at_end_of_image(jpeg_bytes: bytes) -> None:
    descriptor = _describe_jpeg(jpeg_bytes + b"trailing-data\xff\x02\x00\x00")

    assert descriptor.parse_error is None
    assert descriptor.scan_count == 1


def test_prepare_fodb_extracts_once_and_builds_native_and_mixed_workloads(
    tmp_path: Path,
    jpeg_bytes: bytes,
) -> None:
    archive = tmp_path / "part01.zip"
    output = tmp_path / "selected"
    devices = ("D01", "D02")
    _write_archive(archive, jpeg_bytes, devices=devices, scenes=("0001", "0002"))

    descriptor_path = prepare_fodb(
        [archive],
        output,
        scene_count=1,
        seed=11,
        compressed_byte_limit=1024**2,
        expected_device_count=len(devices),
    )
    package = open_dataset_package(descriptor_path)
    payload = package.descriptor["provenance"]

    assert payload["selection"]["requested_scene_count"] == 1
    assert len(payload["selection"]["selected_scene_ids"]) == 1
    assert payload["workloads"]["FODB-native"]["num_items"] == len(devices)
    assert payload["workloads"]["FODB-mixed"]["num_items"] == len(devices) * len(FODB_PROVENANCES)
    assert len(payload["items"]) == len(devices) * len(FODB_PROVENANCES)
    assert all(item["jpeg"]["width"] == 64 for item in payload["items"])
    assert all(item["crc32"] == item["archive_member_crc32"] for item in payload["items"])

    selection_root = output / "selections" / payload["selection"]["selection_id"]
    mixed_item = next((selection_root / "workloads" / "fodb-mixed").rglob("*.jpg"))
    workload_relative = mixed_item.relative_to(selection_root / "workloads" / "fodb-mixed")
    corpus_item = output / "corpus" / Path(*workload_relative.parts[1:])
    assert mixed_item.samefile(corpus_item)
    assert len(package.read_workload_items("fodb-native")) == len(devices)
    assert len(package.read_workload_items("fodb-mixed")) == len(devices) * len(FODB_PROVENANCES)


def test_prepare_fodb_selections_do_not_contaminate_each_other(
    tmp_path: Path,
    jpeg_bytes: bytes,
) -> None:
    archive = tmp_path / "part01.zip"
    output = tmp_path / "selected"
    devices = ("D01", "D02")
    _write_archive(archive, jpeg_bytes, devices=devices, scenes=("0001", "0002"))

    first = prepare_fodb(
        [archive],
        output,
        scene_count=2,
        seed=11,
        compressed_byte_limit=1024**2,
        expected_device_count=len(devices),
    )
    second = prepare_fodb(
        [archive],
        output,
        scene_count=1,
        seed=11,
        compressed_byte_limit=1024**2,
        expected_device_count=len(devices),
    )

    assert first != second
    first_items = open_dataset_package(first).read_workload_items("fodb-mixed")
    assert len(first_items) == 2 * len(devices) * len(FODB_PROVENANCES)
    assert len(open_dataset_package(second).read_workload_items("fodb-mixed")) == len(devices) * len(FODB_PROVENANCES)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imread_benchmark.datasets.controlled import ControlledDatasetError, prepare_controlled_ablation
from imread_benchmark.datasets.package import open_dataset_package


def _write_png(path: Path, *, width: int, height: int, seed: int) -> None:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def test_controlled_ablation_builds_matched_resolution_quality_workloads(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_png(sources / "scene-b.png", width=80, height=60, seed=2)
    _write_png(sources / "scene-a.png", width=96, height=64, seed=1)

    descriptor_path = prepare_controlled_ablation(
        sources,
        tmp_path / "controlled",
        source_name="fixture-lossless",
        source_release="2026-07-29",
        source_license="fixture-only",
        source_url="https://example.test/fixture",
        long_edges=(32, 48),
        qualities=(50, 90),
        include_native=True,
        subsampling="4:2:0",
        seed=17,
        compressed_byte_limit=1024**2,
    )
    package = open_dataset_package(descriptor_path)
    provenance = package.descriptor["provenance"]
    expected_workloads = {
        "controlled-le0032-q050",
        "controlled-le0032-q090",
        "controlled-le0048-q050",
        "controlled-le0048-q090",
        "controlled-native-q050",
        "controlled-native-q090",
    }

    assert set(package.descriptor["workloads"]) == expected_workloads
    assert provenance["design"]["factors"] == {
        "encoder_quality": [50, 90],
        "long_edge_pixels": [32, 48, "native"],
    }
    assert provenance["design"]["controls"]["chroma_subsampling"] == "4:2:0"
    assert provenance["design"]["controls"]["source_metadata"] == "stripped"
    assert provenance["encoder"]["implementation"] == "Pillow"
    assert provenance["source_dataset"] == {
        "license": "fixture-only",
        "name": "fixture-lossless",
        "release": "2026-07-29",
        "url": "https://example.test/fixture",
    }
    assert len(provenance["sources"]) == 2
    assert str(sources) not in json.dumps(provenance)

    ordered_source_ids = provenance["ordered_source_ids"]
    assert len(ordered_source_ids) == 2
    for workload_id in expected_workloads:
        workload = provenance["workloads"][workload_id]
        assert workload["ordered_source_ids"] == ordered_source_ids
        assert workload["item_count"] == 2
        items = package.read_workload_items(workload_id)
        assert len(items) == 2
        assert all(item.compressed_bytes > 0 for item in items)
        manifest_path = package.root / package.descriptor["workloads"][workload_id]["manifest"]
        manifest = json.loads(manifest_path.read_text())
        assert all(item["subsampling"] == "4:2:0" for item in manifest["items"])
        assert all(item["has_exif"] is False for item in manifest["items"])

    le32_items = package.read_workload_items("controlled-le0032-q050")
    assert all(max(_jpeg_size(item.data)) == 32 for item in le32_items)
    native_sizes = {tuple(source["normalized_rgb_size"]) for source in provenance["sources"]}
    assert {_jpeg_size(item.data) for item in package.read_workload_items("controlled-native-q050")} == native_sizes
    assert _quantization_tables(le32_items[0].data) != _quantization_tables(
        package.read_workload_items("controlled-le0032-q090")[0].data,
    )


def test_controlled_ablation_is_content_deterministic(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_png(sources / "scene.png", width=64, height=48, seed=3)

    first = prepare_controlled_ablation(
        sources,
        tmp_path / "first",
        source_name="fixture-lossless",
        source_release="test-release",
        source_license="fixture-only",
        long_edges=(32,),
        qualities=(75,),
        include_native=False,
        compressed_byte_limit=1024**2,
    )
    second = prepare_controlled_ablation(
        sources,
        tmp_path / "second",
        source_name="fixture-lossless",
        source_release="test-release",
        source_license="fixture-only",
        long_edges=(32,),
        qualities=(75,),
        include_native=False,
        compressed_byte_limit=1024**2,
    )

    assert json.loads(first.read_text()) == json.loads(second.read_text())


def test_controlled_ablation_rejects_non_lossless_or_undersized_sources(
    tmp_path: Path,
    jpeg_bytes: bytes,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "already-lossy.jpg").write_bytes(jpeg_bytes)

    with pytest.raises(ControlledDatasetError, match="lossless PNG"):
        prepare_controlled_ablation(
            sources,
            tmp_path / "controlled",
            source_name="fixture-lossless",
            source_release="test-release",
            source_license="fixture-only",
            long_edges=(32,),
            qualities=(75,),
        )

    (sources / "already-lossy.jpg").unlink()
    _write_png(sources / "small.png", width=24, height=16, seed=4)
    with pytest.raises(ControlledDatasetError, match="smaller than requested long edge"):
        prepare_controlled_ablation(
            sources,
            tmp_path / "controlled",
            source_name="fixture-lossless",
            source_release="test-release",
            source_license="fixture-only",
            long_edges=(32,),
            qualities=(75,),
        )


def test_controlled_ablation_enforces_each_resident_workload_byte_budget(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_png(sources / "scene.png", width=64, height=48, seed=5)

    with pytest.raises(ControlledDatasetError, match="resident compressed-byte budget"):
        prepare_controlled_ablation(
            sources,
            tmp_path / "controlled",
            source_name="fixture-lossless",
            source_release="test-release",
            source_license="fixture-only",
            long_edges=(32,),
            qualities=(75,),
            compressed_byte_limit=1,
        )


def _jpeg_size(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        image.load()
        assert image.format == "JPEG"
        assert not image.getexif()
        return image.size


def _quantization_tables(data: bytes) -> dict[int, list[int]]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.quantization

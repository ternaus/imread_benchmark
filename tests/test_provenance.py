from __future__ import annotations

import json
from pathlib import Path

import pytest

from imread_benchmark.environments import (
    EnvironmentDescriptor,
    load_environment_descriptor,
    write_environment_descriptor,
)
from imread_benchmark.platforms import (
    PlatformDescriptor,
    capture_current_platform,
    load_platform_descriptor,
    platform_comparison_id,
    platform_location,
    write_platform_descriptor,
)


def test_environment_descriptor_is_content_addressed_and_tamper_checked(tmp_path: Path) -> None:
    descriptor = EnvironmentDescriptor.build(
        dependency_group="mainstream",
        lock_sha256="1" * 64,
        project_sha256="2" * 64,
        runner_revision="3" * 64,
        python={"implementation": "cpython", "version": "3.12.9", "abi": "cpython-312"},
        platform_tags=("macosx_15_0_arm64",),
        distributions=(("numpy", "2.3.1"), ("pillow", "11.3.0")),
        native_backends={"pillow-jpeg": "libjpeg-turbo 3.1.0"},
    )

    path = write_environment_descriptor(tmp_path / "environment.json", descriptor)

    assert load_environment_descriptor(path) == descriptor
    assert (
        descriptor.environment_id
        == EnvironmentDescriptor.build(
            dependency_group="mainstream",
            lock_sha256="1" * 64,
            project_sha256="2" * 64,
            runner_revision="3" * 64,
            python={"implementation": "cpython", "version": "3.12.9", "abi": "cpython-312"},
            platform_tags=("macosx_15_0_arm64",),
            distributions=(("pillow", "11.3.0"), ("numpy", "2.3.1")),
            native_backends={"pillow-jpeg": "libjpeg-turbo 3.1.0"},
        ).environment_id
    )

    document = json.loads(path.read_text())
    document["distributions"][0]["version"] = "999"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="environment_id"):
        load_environment_descriptor(path)


def test_platform_id_uses_stable_identity_but_not_dynamic_runtime(tmp_path: Path) -> None:
    identity = {
        "architecture": "x86_64",
        "cloud_provider": "gcp",
        "cpu_model": "Intel Xeon",
        "location": "us-central1-a",
        "machine_type": "c3-standard-8",
        "logical_cpu_count": 8,
    }
    first = PlatformDescriptor.build(identity=identity, runtime={"kernel": "6.8.0", "free_ram_bytes": 10})
    second = PlatformDescriptor.build(identity=identity, runtime={"kernel": "6.8.1", "free_ram_bytes": 5})

    assert first.platform_id == second.platform_id
    assert first != second
    changed = PlatformDescriptor.build(
        identity={**identity, "machine_type": "c4-standard-8"},
        runtime=first.runtime,
    )
    assert changed.platform_id != first.platform_id

    path = write_platform_descriptor(tmp_path / "platform.json", first)
    assert load_platform_descriptor(path) == first
    document = json.loads(path.read_text())
    document["identity"]["logical_cpu_count"] = 16
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="platform_id"):
        load_platform_descriptor(path)


def test_capture_keeps_zone_as_provenance_not_platform_identity() -> None:
    first = capture_current_platform(
        cloud_provider="gcp",
        machine_type="c4d-standard-16",
        location="us-central1-a",
    )
    second = capture_current_platform(
        cloud_provider="gcp",
        machine_type="c4d-standard-16",
        location="us-central1-b",
    )

    assert first.platform_id == second.platform_id
    assert first.provenance == {"location": "us-central1-a"}
    assert second.provenance == {"location": "us-central1-b"}
    assert "location" not in first.identity


def test_comparison_platform_id_normalizes_legacy_zone_identity() -> None:
    legacy = PlatformDescriptor.build(
        identity={"architecture": "x86_64", "location": "us-central1-a", "machine_type": "c4d-standard-16"},
        runtime={},
    )
    current = PlatformDescriptor.build(
        identity={"architecture": "x86_64", "machine_type": "c4d-standard-16"},
        runtime={},
        provenance={"location": "us-central1-b"},
    )

    assert legacy.platform_id != current.platform_id
    assert platform_comparison_id(legacy.to_dict()) == current.platform_id
    assert platform_location(legacy.to_dict()) == "us-central1-a"
    assert platform_location(current.to_dict()) == "us-central1-b"

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from imread_benchmark.datasets.package import build_dataset_package, open_dataset_package
from imread_benchmark.decoders import BaseDecoder
from imread_benchmark.support import (
    SupportAuditContext,
    audit_decoder_support,
    build_common_support,
    build_operational_support,
    intersect_supported_items,
    load_support_audit,
    load_support_set,
    write_support_audit,
    write_support_set,
)
from imread_benchmark.support.dataloader import audit_decoder_support_in_dataloader


class _AcceptDecoder(BaseDecoder):
    name = "accept"
    package_name = "pytest"

    def decode(self, data: bytes) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)


class _RejectFirstDecoder(_AcceptDecoder):
    name = "reject-first"

    def __init__(self, rejected: bytes) -> None:
        self.rejected = rejected

    def decode(self, data: bytes) -> np.ndarray:
        if data == self.rejected:
            raise ValueError("unsupported test JPEG")
        return super().decode(data)


def test_common_support_is_a_pinned_intersection_in_selection_order(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="fixture-jpegs",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    items = open_dataset_package(descriptor_path).read_workload_items("fixture")
    selection_id = "selection-test"
    context = SupportAuditContext(
        manifest_id="manifest-test",
        selection_id=selection_id,
        process_context="in-process",
        multiprocessing_start_method=None,
        requested_threads=None,
        environment_id="environment-test",
        platform_id="platform-test",
    )
    accepted = audit_decoder_support(
        _AcceptDecoder(),
        items,
        context,
    )
    rejected = audit_decoder_support(
        _RejectFirstDecoder(items[0].data),
        items,
        context,
    )

    support = build_common_support((accepted, rejected), ordered_selection=tuple(item.item_id for item in items))

    assert accepted.successful_item_ids == tuple(item.item_id for item in items)
    assert rejected.successful_item_ids == tuple(item.item_id for item in items[1:])
    assert rejected.failures[0].item_id == items[0].item_id
    assert support.item_ids == tuple(item.item_id for item in items[1:])
    assert support.support_set_id


def test_comparison_intersection_spans_main_process_and_dataloader_contexts(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="comparison-support-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    items = open_dataset_package(descriptor_path).read_workload_items("fixture")
    base = {
        "manifest_id": "manifest-test",
        "selection_id": "selection-test",
        "requested_threads": None,
        "environment_id": "environment-test",
        "platform_id": "platform-test",
    }
    main = audit_decoder_support(
        _AcceptDecoder(),
        items,
        SupportAuditContext(process_context="main-process", multiprocessing_start_method=None, **base),
    )
    worker = audit_decoder_support(
        _RejectFirstDecoder(items[0].data),
        items,
        SupportAuditContext(process_context="dataloader", multiprocessing_start_method="spawn", **base),
    )

    item_ids = intersect_supported_items(
        (main, worker),
        ordered_selection=tuple(item.item_id for item in items),
    )

    assert item_ids == tuple(item.item_id for item in items[1:])


def test_support_artifacts_are_content_addressed_round_trip_and_tamper_checked(
    tmp_path: Path,
    jpeg_dir: Path,
) -> None:
    descriptor_path = build_dataset_package(
        package_name="support-store-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    items = open_dataset_package(descriptor_path).read_workload_items("fixture")
    context = SupportAuditContext(
        manifest_id="1" * 64,
        selection_id="2" * 64,
        process_context="main-process",
        multiprocessing_start_method=None,
        requested_threads=None,
        environment_id="3" * 64,
        platform_id="4" * 64,
    )
    audit = audit_decoder_support(_AcceptDecoder(), items, context)
    support_set = build_operational_support(audit)

    audit_path = write_support_audit(tmp_path / "support", audit)
    set_path = write_support_set(tmp_path / "support", support_set)

    assert audit_path.name == f"{audit.audit_id}.json"
    assert set_path.name == f"{support_set.support_set_id}.json"
    assert load_support_audit(audit_path) == audit
    assert load_support_set(set_path) == support_set
    assert write_support_audit(tmp_path / "support", audit) == audit_path
    assert write_support_set(tmp_path / "support", support_set) == set_path

    document = json.loads(set_path.read_text())
    document["item_ids"] = document["item_ids"][1:]
    set_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="support_set_id"):
        load_support_set(set_path)


def test_support_audit_runs_decode_inside_a_real_dataloader_worker(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="support-dataloader-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    items = open_dataset_package(descriptor_path).read_workload_items("fixture")
    context = SupportAuditContext(
        manifest_id="1" * 64,
        selection_id="2" * 64,
        process_context="dataloader",
        multiprocessing_start_method="spawn",
        requested_threads=None,
        environment_id="3" * 64,
        platform_id="4" * 64,
    )

    result = audit_decoder_support_in_dataloader(
        "pillow",
        items,
        context,
        requested_threads=None,
    )

    assert result.audit.successful_item_ids == tuple(item.item_id for item in items)
    assert result.audit.failures == ()
    assert len(result.worker_handshakes) == 1
    assert result.worker_handshakes[0]["process_id"] != os.getpid()
    assert result.worker_handshakes[0]["effective_threads"] == 1
    assert result.worker_handshakes[0]["multiprocessing_start_method"] == "spawn"

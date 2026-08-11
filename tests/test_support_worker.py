from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from imread_benchmark.datasets.package import build_dataset_package, open_dataset_package
from imread_benchmark.support import load_committed_support_audit
from imread_benchmark.support.spec import SupportAuditIdentity, SupportAuditSpec, write_support_audit_spec


def test_support_audit_worker_is_a_fresh_resumable_subprocess(tmp_path: Path, jpeg_dir: Path) -> None:
    descriptor_path = build_dataset_package(
        package_name="support-worker-fixture",
        workloads={"fixture": jpeg_dir},
        output_root=tmp_path / "packages",
        provenance={"source": "pytest"},
    )
    package = open_dataset_package(descriptor_path)
    workload = package.descriptor["workloads"]["fixture"]
    package_id = package.descriptor["package_id"]
    assert isinstance(workload, dict)
    assert isinstance(package_id, str)
    manifest_id = workload["manifest_id"]
    assert isinstance(manifest_id, str)
    item_ids = tuple(item.item_id for item in package.read_workload_items("fixture"))
    spec = SupportAuditSpec.build(
        identity=SupportAuditIdentity(
            decoder_id="pillow",
            requested_threads=None,
            package_id=package_id,
            workload_id="fixture",
            manifest_id=manifest_id,
            selection_id="2" * 64,
            item_ids=item_ids,
            process_context="main-process",
            multiprocessing_start_method=None,
            environment_id="3" * 64,
            platform_id="4" * 64,
            runner_revision="1" * 40,
        ),
        package_descriptor=descriptor_path,
    )
    spec_path = write_support_audit_spec(tmp_path / "audit-spec.json", spec)
    artifact_root = tmp_path / "artifacts"

    first = subprocess.run(  # noqa: S603 - launches the package's own support worker with controlled argv
        (
            sys.executable,
            "-m",
            "imread_benchmark.support.worker",
            "--audit-spec",
            str(spec_path),
            "--artifact-root",
            str(artifact_root),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    first_result = json.loads(first.stdout)
    audit = load_committed_support_audit(artifact_root / "support", spec.audit_key)

    assert first_result["process_id"] != os.getpid()
    assert first_result["audit_id"] == audit.audit_id
    assert audit.successful_item_ids == item_ids

    second = subprocess.run(  # noqa: S603 - repeats the same controlled worker to verify resume
        (
            sys.executable,
            "-m",
            "imread_benchmark.support.worker",
            "--audit-spec",
            str(spec_path),
            "--artifact-root",
            str(artifact_root),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(second.stdout)["audit_id"] == audit.audit_id

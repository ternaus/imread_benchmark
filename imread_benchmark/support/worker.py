from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from imread_benchmark.datasets.package import open_dataset_package
from imread_benchmark.execution.measurement import configure_decoder
from imread_benchmark.support.audit import SupportAuditContext, audit_decoder_support
from imread_benchmark.support.dataloader import audit_decoder_support_in_dataloader
from imread_benchmark.support.spec import load_support_audit_spec
from imread_benchmark.support.store import commit_support_audit


def run_support_audit_worker(spec_path: str | Path, artifact_root: str | Path) -> dict[str, object]:
    spec = load_support_audit_spec(spec_path)
    identity = spec.identity
    package = open_dataset_package(
        spec.package_descriptor,
        trust_ready=(spec.package_descriptor.parent / ".READY.json").is_file(),
    )
    if package.descriptor.get("package_id") != identity.package_id:
        raise ValueError("support audit package_id does not match descriptor")
    workload = package.descriptor.get("workloads", {}).get(identity.workload_id)
    if not isinstance(workload, dict) or workload.get("manifest_id") != identity.manifest_id:
        raise ValueError("support audit manifest_id does not match descriptor")
    all_items = package.read_workload_items(identity.workload_id)
    by_id = {item.item_id: item for item in all_items}
    if len(by_id) != len(all_items) or any(item_id not in by_id for item_id in identity.item_ids):
        raise ValueError("support audit selection does not match workload items")
    items = tuple(by_id[item_id] for item_id in identity.item_ids)
    context = SupportAuditContext(
        manifest_id=identity.manifest_id,
        selection_id=identity.selection_id,
        process_context=identity.process_context,
        multiprocessing_start_method=identity.multiprocessing_start_method,
        requested_threads=identity.requested_threads,
        environment_id=identity.environment_id,
        platform_id=identity.platform_id,
    )
    if identity.process_context == "dataloader":
        audit = audit_decoder_support_in_dataloader(
            identity.decoder_id,
            items,
            context,
            requested_threads=identity.requested_threads,
        ).audit
    else:
        from imread_benchmark.decoders import REGISTRY

        decoder_class = REGISTRY.get(identity.decoder_id)
        if decoder_class is None:
            raise ValueError(f"unknown decoder {identity.decoder_id!r}")
        decoder = decoder_class()
        configure_decoder(decoder, identity.requested_threads)
        audit = audit_decoder_support(decoder, items, context)
    commit_support_audit(Path(artifact_root) / "support", spec.audit_key, audit)
    return {
        "audit_id": audit.audit_id,
        "audit_key": spec.audit_key,
        "process_id": os.getpid(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-spec", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_support_audit_worker(args.audit_spec, args.artifact_root), sort_keys=True))


if __name__ == "__main__":
    main()

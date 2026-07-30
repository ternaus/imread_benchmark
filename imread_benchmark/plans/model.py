from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from imread_benchmark.datasets.package import DatasetPackageError, open_dataset_package

PLAN_SCHEMA_VERSION = "2.0"


class PlanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    method: str
    item_ids: tuple[str, ...]
    selection_id: str


@dataclass(frozen=True, slots=True)
class PlannedDataset:
    descriptor_path: Path
    package_id: str
    workload_id: str
    manifest_id: str
    manifest: dict[str, Any]
    selection: DatasetSelection
    logical_repeat_factor: int


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    schema_version: str
    experiment_name: str
    seed: int
    repetitions: int
    dataset: PlannedDataset
    matrix: dict[str, Any]
    measurement: dict[str, Any]
    execution: dict[str, Any]


def load_experiment_plan(
    path: str | Path,
    *,
    dataset_descriptor: str | Path | None = None,
) -> ExperimentPlan:
    plan_path = Path(path).resolve()
    document = _read_yaml_object(plan_path)
    if document.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanError(f"unsupported experiment plan schema: {document.get('schema_version')!r}")

    dataset_document = _required_object(document, "dataset")
    descriptor_path = (
        Path(dataset_descriptor)
        if dataset_descriptor is not None
        else Path(_required_string(dataset_document, "descriptor"))
    )
    if descriptor_path.is_absolute():
        descriptor_path = descriptor_path.resolve()
    else:
        descriptor_path = (plan_path.parent / descriptor_path).resolve()
    try:
        package = open_dataset_package(descriptor_path)
    except DatasetPackageError as exc:
        raise PlanError(f"invalid dataset package: {exc}") from exc

    package_id = _required_string(dataset_document, "package_id")
    if package.descriptor.get("package_id") != package_id:
        raise PlanError("dataset package_id does not match descriptor")
    workload_id = _required_string(dataset_document, "workload_id")
    workloads = _required_object(package.descriptor, "workloads")
    workload = workloads.get(workload_id)
    if not isinstance(workload, dict):
        raise PlanError(f"dataset package has no workload {workload_id!r}")
    manifest_id = _required_string(dataset_document, "manifest_id")
    if workload.get("manifest_id") != manifest_id:
        raise PlanError("dataset manifest_id does not match package workload")

    manifest_path = package.root / _required_string(workload, "manifest")
    manifest = _read_json_object(manifest_path)
    selection = _load_selection(_required_object(dataset_document, "selection"), manifest, manifest_id)
    logical_repeat_factor = _required_positive_int(dataset_document, "logical_repeat_factor")
    execution = _load_execution(_required_object(document, "execution"))

    return ExperimentPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        experiment_name=_required_string(document, "experiment_name"),
        seed=_required_int(document, "seed"),
        repetitions=_required_positive_int(document, "repetitions"),
        dataset=PlannedDataset(
            descriptor_path=descriptor_path,
            package_id=package_id,
            workload_id=workload_id,
            manifest_id=manifest_id,
            manifest=manifest,
            selection=selection,
            logical_repeat_factor=logical_repeat_factor,
        ),
        matrix=_required_object(document, "matrix"),
        measurement=_required_object(document, "measurement"),
        execution=execution,
    )


def _load_execution(document: dict[str, Any]) -> dict[str, object]:
    expected = {
        "checkpoint_each_run",
        "maximum_memory_fraction",
        "per_run_subprocess",
        "run_timeout_seconds",
    }
    if set(document) != expected:
        raise PlanError(f"execution fields must be exactly {sorted(expected)}")
    if document.get("per_run_subprocess") is not True:
        raise PlanError("execution.per_run_subprocess must be true")
    if document.get("checkpoint_each_run") is not True:
        raise PlanError("execution.checkpoint_each_run must be true")
    timeout = _required_positive_number(document, "run_timeout_seconds")
    fraction = _required_positive_number(document, "maximum_memory_fraction")
    if fraction > 0.9:
        raise PlanError("execution.maximum_memory_fraction must be at most 0.9")
    return {
        "checkpoint_each_run": True,
        "maximum_memory_fraction": fraction,
        "per_run_subprocess": True,
        "run_timeout_seconds": timeout,
    }


def _load_selection(
    document: dict[str, Any],
    manifest: dict[str, Any],
    manifest_id: str,
) -> DatasetSelection:
    method = _required_string(document, "method")
    if method != "all":
        raise PlanError(f"unsupported dataset selection method: {method!r}")
    items = manifest.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise PlanError("dataset manifest has no valid items list")
    expected_items = _required_positive_int(document, "expected_items")
    if len(items) != expected_items:
        raise PlanError(f"dataset selection expected {expected_items} items, found {len(items)}")
    item_ids = tuple(_required_string(cast("dict[str, Any]", item), "item_id") for item in items)
    identity = {
        "item_ids": item_ids,
        "manifest_id": manifest_id,
        "method": method,
        "schema_version": PLAN_SCHEMA_VERSION,
    }
    return DatasetSelection(method=method, item_ids=item_ids, selection_id=_digest_json(identity))


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PlanError(f"cannot read experiment plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"experiment plan must be a YAML object: {path}")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"dataset manifest must be a JSON object: {path}")
    return value


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PlanError(f"field {key!r} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"field {key!r} must be an integer")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = _required_int(payload, key)
    if value <= 0:
        raise PlanError(f"field {key!r} must be positive")
    return value


def _required_positive_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PlanError(f"field {key!r} must be a positive number")
    return float(value)


def _digest_json(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

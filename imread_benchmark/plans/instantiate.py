from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from imread_benchmark.datasets.package import DatasetPackageError, open_dataset_package
from imread_benchmark.plans.expand import expand_experiment_plan
from imread_benchmark.plans.model import PlanError, load_experiment_plan

_PLACEHOLDERS = ("PACKAGE_ID", "WORKLOAD_ID", "MANIFEST_ID", "ITEM_COUNT")


@dataclass(frozen=True, slots=True)
class InstantiatedPlan:
    workload_id: str
    manifest_id: str
    item_count: int
    plan_id: str
    run_count: int
    path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "item_count": self.item_count,
            "manifest_id": self.manifest_id,
            "path": str(self.path),
            "plan_id": self.plan_id,
            "run_count": self.run_count,
            "workload_id": self.workload_id,
        }


def instantiate_experiment_plans(
    *,
    template_path: str | Path,
    package_descriptor: str | Path,
    output_dir: str | Path,
    workload_ids: tuple[str, ...] = (),
) -> tuple[InstantiatedPlan, ...]:
    """Fill dataset identities in a schema-2 template and validate every output plan."""
    template = _read_template(Path(template_path).resolve())
    descriptor_path = Path(package_descriptor).resolve()
    try:
        package = open_dataset_package(descriptor_path)
    except DatasetPackageError as exc:
        raise PlanError(f"invalid dataset package: {exc}") from exc
    package_id = _required_string(package.descriptor, "package_id")
    workloads = package.descriptor.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        raise PlanError("dataset package has no workloads")
    selected = _select_workloads(workloads, workload_ids)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[InstantiatedPlan] = []
    for workload_id in selected:
        raw_workload = workloads[workload_id]
        if not isinstance(raw_workload, dict):
            raise PlanError(f"dataset package workload {workload_id!r} must be an object")
        manifest_id = _required_string(raw_workload, "manifest_id")
        item_count = _required_positive_int(raw_workload, "item_count")
        document = _replace_placeholders(
            template,
            {
                "ITEM_COUNT": item_count,
                "MANIFEST_ID": manifest_id,
                "PACKAGE_ID": package_id,
                "WORKLOAD_ID": workload_id,
            },
        )
        if _find_placeholders(document):
            raise PlanError(f"unresolved plan template placeholders: {sorted(_find_placeholders(document))}")
        output_path = destination / f"{workload_id}.yaml"
        temporary_path = _write_temporary_yaml(output_path, document)
        try:
            plan = load_experiment_plan(temporary_path, dataset_descriptor=descriptor_path)
            templates = expand_experiment_plan(plan)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        results.append(
            InstantiatedPlan(
                workload_id=workload_id,
                manifest_id=manifest_id,
                item_count=item_count,
                plan_id=templates[0].plan_id,
                run_count=len(templates),
                path=output_path,
            ),
        )
    return tuple(results)


def _read_template(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PlanError(f"cannot read experiment plan template {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise PlanError(f"experiment plan template must be a YAML object: {path}")
    if document.get("schema_version") != "2.0":
        raise PlanError(f"unsupported experiment plan template schema: {document.get('schema_version')!r}")
    return document


def _select_workloads(workloads: dict[str, object], requested: tuple[str, ...]) -> tuple[str, ...]:
    if len(requested) != len(set(requested)):
        raise PlanError("requested workload IDs must be unique")
    selected = requested or tuple(sorted(workloads))
    unknown = tuple(workload_id for workload_id in selected if workload_id not in workloads)
    if unknown:
        raise PlanError(f"dataset package has no requested workloads: {list(unknown)}")
    return selected


def _replace_placeholders(value: object, replacements: dict[str, str | int]) -> object:
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    if value in replacements:
        return replacements[value]
    result = value
    for placeholder, replacement in replacements.items():
        result = result.replace(placeholder, str(replacement))
    return result


def _find_placeholders(value: object) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_find_placeholders(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_find_placeholders(item) for item in value), set())
    if isinstance(value, str):
        return {placeholder for placeholder in _PLACEHOLDERS if placeholder in value}
    return set()


def _write_temporary_yaml(path: Path, payload: object) -> Path:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, mode="w") as file:
            yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    else:
        return temporary


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PlanError(f"field {key!r} must be a non-empty string")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanError(f"field {key!r} must be a positive integer")
    return value

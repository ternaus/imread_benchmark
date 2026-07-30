from __future__ import annotations

import hashlib
import json
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from imread_benchmark.analysis.canonical import RunBundleRecord, load_bundles
from imread_benchmark.analysis.claims import ClaimScope, assert_claim_scope

PUBLICATION_SCHEMA_VERSION = "2.0"
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_OUTPUT_FILES = ("provenance.json", "results.json")


class PublicationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationSpec:
    claim_scope: ClaimScope
    filters: dict[str, object]
    statistic: str
    practical_margin_percent: float
    generator_revision: str


def publish(
    *,
    artifact_root: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    check: bool = False,
) -> None:
    spec = _load_spec(spec_path)
    records = _filter_records(load_bundles(artifact_root), spec.filters)
    assert_claim_scope(records, spec.claim_scope)
    generated = _render(records, spec)
    destination = Path(output_dir)
    if check:
        _check_outputs(destination, generated)
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        _atomic_write(destination / name, content)


def _load_spec(path: str | Path) -> PublicationSpec:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PublicationError(f"cannot read publication spec {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise PublicationError("publication spec must be a YAML object")
    if document.get("schema_version") != PUBLICATION_SCHEMA_VERSION:
        raise PublicationError(f"unsupported publication schema: {document.get('schema_version')!r}")
    try:
        claim_scope = ClaimScope(_required_string(document, "claim_scope"))
    except ValueError as exc:
        raise PublicationError(str(exc)) from exc
    filters = document.get("filters")
    if not isinstance(filters, dict) or not all(isinstance(key, str) for key in filters):
        raise PublicationError("publication filters must be an object with string keys")
    statistic = _required_string(document, "statistic")
    if statistic not in {"images_per_second", "elapsed_seconds", "microseconds_per_image"}:
        raise PublicationError(f"unsupported publication statistic: {statistic!r}")
    margin = document.get("practical_margin_percent")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
        raise PublicationError("practical_margin_percent must be non-negative")
    revision = _required_string(document, "generator_revision")
    if _REVISION.fullmatch(revision) is None:
        raise PublicationError("generator_revision must be a 40- or 64-character hexadecimal revision")
    return PublicationSpec(claim_scope, filters, statistic, float(margin), revision)


def _filter_records(
    records: tuple[RunBundleRecord, ...],
    filters: dict[str, object],
) -> tuple[RunBundleRecord, ...]:
    selected = records
    for path, expected in sorted(filters.items()):
        selected = tuple(record for record in selected if _record_value(record, path) == expected)
    if not selected:
        raise PublicationError("publication filters selected no committed bundles")
    return selected


def _record_value(record: RunBundleRecord, path: str) -> object:
    prefix, separator, key = path.partition(".")
    if not separator or prefix not in {"config", "dataset", "environment", "platform", "runtime"} or not key:
        raise PublicationError(f"invalid publication filter path: {path!r}")
    document = getattr(record, prefix)
    return document.get(key)


def _render(records: tuple[RunBundleRecord, ...], spec: PublicationSpec) -> dict[str, str]:
    ordered = tuple(sorted(records, key=lambda record: record.run_key))
    table = {
        "claim_scope": spec.claim_scope.value,
        "groups": _group_rows(ordered, spec.statistic),
        "practical_margin_percent": spec.practical_margin_percent,
        "rows": [_result_row(record, spec.statistic) for record in ordered],
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "statistic": spec.statistic,
    }
    results_content = _json_content(table)
    provenance_core = {
        "bundle_ids": [record.bundle_id for record in ordered],
        "bundle_keys": [record.run_key for record in ordered],
        "claim_scope": spec.claim_scope.value,
        "filters": spec.filters,
        "generator_revision": spec.generator_revision,
        "output_sha256": hashlib.sha256(results_content.encode()).hexdigest(),
        "practical_margin_percent": spec.practical_margin_percent,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "statistic": spec.statistic,
    }
    provenance = {**provenance_core, "publication_id": _digest_json(provenance_core)}
    return {
        "provenance.json": _json_content(provenance),
        "results.json": results_content,
    }


def _result_row(record: RunBundleRecord, statistic: str) -> dict[str, object]:
    statistics = record.summary.get("statistics")
    if not isinstance(statistics, dict) or not isinstance(statistics.get(statistic), dict):
        raise PublicationError(f"bundle {record.run_key} has no statistic {statistic!r}")
    summary = statistics[statistic]
    return {
        "block_position": record.config.get("block_position"),
        "config_id": record.config.get("config_id"),
        "decoder_id": record.config.get("decoder_id"),
        "environment_id": record.environment.get("environment_id"),
        "manifest_id": record.dataset.get("manifest_id"),
        "mean": summary.get("mean"),
        "median": summary.get("median"),
        "n": summary.get("n"),
        "platform_id": record.platform.get("platform_id"),
        "protocol_id": record.config.get("protocol_id"),
        "raw_samples": [
            _sample_value(sample.elapsed_seconds, sample.items_processed, statistic) for sample in record.samples
        ],
        "repetition": record.config.get("repetition"),
        "run_key": record.run_key,
        "sample_std": summary.get("sample_std"),
        "support_set_id": record.dataset.get("support_set_id"),
        "workload_id": record.dataset.get("workload_id"),
    }


def _group_rows(records: tuple[RunBundleRecord, ...], statistic: str) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[RunBundleRecord]] = {}
    for record in records:
        key = (
            record.config.get("plan_id"),
            record.config.get("config_id"),
            record.dataset.get("workload_id"),
            record.dataset.get("support_set_id"),
            record.environment.get("environment_id"),
            record.platform.get("platform_id"),
        )
        groups.setdefault(key, []).append(record)
    return [_group_row(groups[key], statistic) for key in sorted(groups, key=_sortable_group_key)]


def _sortable_group_key(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str(value) for value in key)


def _group_row(records: list[RunBundleRecord], statistic: str) -> dict[str, object]:
    ordered = sorted(records, key=lambda record: (record.config.get("repetition"), record.run_key))
    repetitions = [record.config.get("repetition") for record in ordered]
    if len(repetitions) != len(set(repetitions)):
        raise PublicationError("publication group contains duplicate repetition blocks")
    values = [_run_mean(record, statistic) for record in ordered]
    first = ordered[0]
    excluded_config_fields = {
        "block_position",
        "plan_id",
        "repetition",
        "run_key",
        "runner_revision",
        "schema_version",
    }
    return {
        "configuration": {
            key: value for key, value in sorted(first.config.items()) if key not in excluded_config_fields
        },
        "environment_id": first.environment.get("environment_id"),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "n": len(values),
        "platform_id": first.platform.get("platform_id"),
        "raw_run_means": values,
        "repetitions": repetitions,
        "run_keys": [record.run_key for record in ordered],
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "support_set_id": first.dataset.get("support_set_id"),
        "workload_id": first.dataset.get("workload_id"),
    }


def _run_mean(record: RunBundleRecord, statistic: str) -> float:
    values = [_sample_value(sample.elapsed_seconds, sample.items_processed, statistic) for sample in record.samples]
    if not values:
        raise PublicationError(f"bundle {record.run_key} contains no timed samples")
    return statistics.fmean(values)


def _sample_value(elapsed_seconds: float, items_processed: int, statistic: str) -> float:
    if statistic == "elapsed_seconds":
        return elapsed_seconds
    if statistic == "images_per_second":
        return items_processed / elapsed_seconds
    return elapsed_seconds * 1_000_000 / items_processed


def _check_outputs(destination: Path, generated: dict[str, str]) -> None:
    observed = {path.name for path in destination.iterdir() if path.is_file()} if destination.is_dir() else set()
    if observed != set(_OUTPUT_FILES):
        raise PublicationError(f"publication output is stale; file set is {sorted(observed)}")
    stale = [name for name, content in generated.items() if (destination / name).read_text() != content]
    if stale:
        raise PublicationError(f"publication output is stale: {', '.join(sorted(stale))}")


def _atomic_write(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            file.write(content)
            temporary = Path(file.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PublicationError(f"publication field {key!r} must be a non-empty string")
    return value


def _json_content(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _digest_json(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

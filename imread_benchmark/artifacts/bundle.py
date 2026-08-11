from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from imread_benchmark.analysis.statistics import summarize_benchmark

BUNDLE_SCHEMA_VERSION = "2.0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FILES = (
    "config.json",
    "dataset.json",
    "environment.json",
    "platform.json",
    "runtime.json",
    "samples.jsonl",
    "failures.jsonl",
    "events.jsonl",
    "summary.json",
)
_ALL_FILES = frozenset((*_DOCUMENT_FILES, "bundle_manifest.json", "COMMITTED.json"))
RUN_BUNDLE_FILES = tuple(sorted(_ALL_FILES))
REMOTE_BUNDLE_FILES = tuple(name for name in RUN_BUNDLE_FILES if name != "COMMITTED.json")
_RESERVED_SUMMARY_FIELDS = frozenset({"schema_version", "statistics", "status"})


class BundleValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunSample:
    sample_index: int
    elapsed_seconds: float
    items_processed: int
    started_at_utc: str | None = None

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        if self.items_processed <= 0:
            raise ValueError("items_processed must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "images_per_second": self.items_processed / self.elapsed_seconds,
            "items_processed": self.items_processed,
            "sample_index": self.sample_index,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "started_at_utc": self.started_at_utc,
        }


@dataclass(frozen=True, slots=True)
class BundleData:
    config: dict[str, object]
    dataset: dict[str, object]
    environment: dict[str, object]
    platform: dict[str, object]
    runtime: dict[str, object]
    samples: tuple[RunSample, ...]
    summary_fields: dict[str, object]
    events: tuple[dict[str, object], ...] = ()


def write_run_bundle(
    *,
    root: str | Path,
    run_key: str,
    data: BundleData,
) -> Path:
    _validate_digest(run_key, field="run_key")
    if not data.samples:
        raise ValueError("run bundle requires at least one sample")
    reserved = _RESERVED_SUMMARY_FIELDS.intersection(data.summary_fields)
    if reserved:
        raise ValueError(f"summary_fields contains reserved names: {', '.join(sorted(reserved))}")

    root_path = Path(root)
    destination = root_path / run_key
    if destination.exists():
        raise FileExistsError(f"run bundles are immutable: {destination}")
    root_path.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_key}.", dir=root_path))
    try:
        _write_payload(staging, data)
        manifest = _build_bundle_manifest(staging, run_key)
        _write_json(staging / "bundle_manifest.json", manifest)
        _write_json(
            staging / "COMMITTED.json",
            {
                "bundle_id": manifest["bundle_id"],
                "run_key": run_key,
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "status": "committed",
            },
        )
        validate_run_bundle(staging, expected_run_key=run_key)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def validate_run_bundle(bundle: str | Path, *, expected_run_key: str | None = None) -> None:
    root = Path(bundle)
    observed_files = {path.name for path in root.iterdir() if path.is_file()} if root.is_dir() else set()
    if observed_files != _ALL_FILES:
        missing = sorted(_ALL_FILES - observed_files)
        extra = sorted(observed_files - _ALL_FILES)
        raise BundleValidationError(f"run bundle file set mismatch; missing={missing}, extra={extra}")

    commit = _read_object(root / "COMMITTED.json")
    manifest = _read_object(root / "bundle_manifest.json")
    run_key = _required_digest(commit, "run_key")
    if expected_run_key is not None and run_key != expected_run_key:
        raise BundleValidationError("run_key does not match expected identity")
    if commit.get("schema_version") != BUNDLE_SCHEMA_VERSION or commit.get("status") != "committed":
        raise BundleValidationError("invalid commit marker")
    if manifest.get("run_key") != run_key or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError("bundle manifest identity mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_DOCUMENT_FILES):
        raise BundleValidationError("bundle manifest file set mismatch")
    manifest_core = {key: value for key, value in manifest.items() if key != "bundle_id"}
    bundle_id = _digest_json(manifest_core)
    if manifest.get("bundle_id") != bundle_id or commit.get("bundle_id") != bundle_id:
        raise BundleValidationError("bundle_id mismatch")
    _validate_payload_checksums(root, files)
    _validate_cross_file_identity(root, run_key)
    _validate_summary(root)


def _write_payload(root: Path, data: BundleData) -> None:
    _write_json(root / "config.json", _versioned(data.config))
    _write_json(root / "dataset.json", _versioned(data.dataset))
    _write_json(root / "environment.json", _versioned(data.environment))
    _write_json(root / "platform.json", _versioned(data.platform))
    _write_json(root / "runtime.json", _versioned(data.runtime))
    _write_jsonl(root / "samples.jsonl", [sample.to_dict() for sample in data.samples])
    _write_jsonl(root / "failures.jsonl", [])
    _write_jsonl(root / "events.jsonl", [_versioned(row) for row in data.events])
    items_processed = {sample.items_processed for sample in data.samples}
    if len(items_processed) != 1:
        raise ValueError("all samples in a run bundle must process the same number of items")
    statistics = summarize_benchmark(
        [sample.elapsed_seconds for sample in data.samples],
        items_processed=items_processed.pop(),
    )
    _write_json(
        root / "summary.json",
        {
            **data.summary_fields,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "statistics": statistics,
            "status": "complete",
        },
    )


def _build_bundle_manifest(root: Path, run_key: str) -> dict[str, object]:
    core: dict[str, object] = {
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": _sha256_file(root / name),
            }
            for name in _DOCUMENT_FILES
        },
        "run_key": run_key,
        "schema_version": BUNDLE_SCHEMA_VERSION,
    }
    return {**core, "bundle_id": _digest_json(core)}


def _validate_payload_checksums(root: Path, files: dict[str, Any]) -> None:
    for name in _DOCUMENT_FILES:
        metadata = files.get(name)
        if not isinstance(metadata, dict):
            raise BundleValidationError(f"bundle manifest metadata missing for {name}")
        path = root / name
        if path.stat().st_size != metadata.get("bytes") or _sha256_file(path) != metadata.get("sha256"):
            raise BundleValidationError(f"checksum mismatch for {name}")


def _validate_summary(root: Path) -> None:
    config = _read_object(root / "config.json")
    summary = _read_object(root / "summary.json")
    samples = _read_jsonl(root / "samples.jsonl")
    if _read_jsonl(root / "failures.jsonl"):
        raise BundleValidationError("a committed successful run cannot contain timed failures")
    expected_count = config.get("timed_passes_per_run")
    if not isinstance(expected_count, int) or expected_count <= 0 or len(samples) != expected_count:
        raise BundleValidationError("sample count does not match config.timed_passes_per_run")
    indices = [sample.get("sample_index") for sample in samples]
    if indices != list(range(len(samples))):
        raise BundleValidationError("sample_index values are not consecutive")
    elapsed = [sample.get("elapsed_seconds") for sample in samples]
    item_counts = {sample.get("items_processed") for sample in samples}
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in elapsed)
        or len(item_counts) != 1
    ):
        raise BundleValidationError("samples contain invalid measurements")
    items_processed = next(iter(item_counts))
    if isinstance(items_processed, bool) or not isinstance(items_processed, int) or items_processed <= 0:
        raise BundleValidationError("samples contain invalid items_processed")
    expected_statistics = summarize_benchmark(
        [float(value) for value in elapsed if isinstance(value, (int, float))],
        items_processed=items_processed,
    )
    if summary.get("schema_version") != BUNDLE_SCHEMA_VERSION or summary.get("status") != "complete":
        raise BundleValidationError("summary identity mismatch")
    if summary.get("statistics") != expected_statistics:
        raise BundleValidationError("summary statistics do not match samples")


def _validate_cross_file_identity(root: Path, run_key: str) -> None:
    from imread_benchmark.environments import load_environment_descriptor
    from imread_benchmark.execution.spec import RunIdentity, compute_run_key
    from imread_benchmark.plans import RunConfiguration
    from imread_benchmark.platforms import load_platform_descriptor

    config = _read_object(root / "config.json")
    dataset = _read_object(root / "dataset.json")
    runtime = _read_object(root / "runtime.json")
    summary = _read_object(root / "summary.json")
    samples = _read_jsonl(root / "samples.jsonl")
    try:
        environment_descriptor = load_environment_descriptor(root / "environment.json")
        platform_descriptor = load_platform_descriptor(root / "platform.json")
    except (OSError, TypeError, ValueError) as exc:
        raise BundleValidationError(f"invalid environment or platform descriptor: {exc}") from exc
    configuration = _configuration_from_document(config, RunConfiguration)
    if config.get("config_id") != configuration.config_id:
        raise BundleValidationError("config_id does not match expanded configuration")
    _require_equal_dataset_identity(configuration, dataset)
    item_ids = _required_string_tuple(dataset, "ordered_item_ids")
    try:
        identity = RunIdentity(
            plan_id=_required_string(config, "plan_id"),
            platform_id=platform_descriptor.platform_id,
            environment_id=environment_descriptor.environment_id,
            runner_revision=_required_string(config, "runner_revision"),
            workload_id=_required_string(dataset, "workload_id"),
            support_set_id=_required_string(dataset, "support_set_id"),
            support_item_ids=item_ids,
            configuration=configuration,
            repetition=_required_non_negative_int(config, "repetition"),
            block_position=_required_non_negative_int(config, "block_position"),
        )
    except BundleValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleValidationError(f"invalid cross-file run identity: {exc}") from exc
    if config.get("run_key") != run_key or compute_run_key(identity) != run_key:
        raise BundleValidationError("run_key does not match cross-file run identity")
    if environment_descriptor.runner_revision != identity.runner_revision:
        raise BundleValidationError("environment runner revision does not match run identity")
    _validate_measurement_counts(configuration, item_ids, samples, summary)
    _validate_runtime(configuration, runtime)
    _validate_events(root)


def _configuration_from_document(config: dict[str, Any], configuration_type: Any) -> Any:
    try:
        return configuration_type(
            **{field.name: config[field.name] for field in fields(configuration_type)},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleValidationError(f"invalid expanded run configuration: {exc}") from exc


def _require_equal_dataset_identity(configuration: Any, dataset: dict[str, Any]) -> None:
    expected = {
        "manifest_id": configuration.manifest_id,
        "package_id": configuration.package_id,
        "selection_id": configuration.selection_id,
        "support_policy": configuration.support_policy,
    }
    mismatched = [key for key, value in expected.items() if dataset.get(key) != value]
    if mismatched:
        raise BundleValidationError(f"config and dataset identity mismatch: {mismatched}")
    expected_context = (
        "dataloader" if configuration.protocol_id == "loader-supply" and configuration.num_workers else "main-process"
    )
    if dataset.get("support_process_context") != expected_context:
        raise BundleValidationError("support process context does not match protocol")
    if dataset.get("support_multiprocessing_start_method") != configuration.multiprocessing_start_method:
        raise BundleValidationError("support multiprocessing start method does not match configuration")


def _validate_measurement_counts(
    configuration: Any,
    item_ids: tuple[str, ...],
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    logical_items = len(item_ids) * configuration.logical_repeat_factor
    if any(sample.get("items_processed") != logical_items for sample in samples):
        raise BundleValidationError("sample items_processed does not match pinned support and logical repeats")
    if (
        summary.get("num_unique_images") != len(item_ids)
        or summary.get("logical_repeat_factor") != configuration.logical_repeat_factor
        or summary.get("logical_decodes_per_pass") != logical_items
    ):
        raise BundleValidationError("summary item counts do not match pinned support and logical repeats")


def _validate_runtime(configuration: Any, runtime: dict[str, Any]) -> None:
    requested = configuration.requested_threads
    if configuration.protocol_id == "decode-memory":
        _validate_effective_threads(runtime.get("effective_threads"), requested)
        return
    _validate_loader_runtime(configuration, runtime, requested)


def _validate_loader_runtime(configuration: Any, runtime: dict[str, Any], requested: int | None) -> None:
    handshakes = runtime.get("worker_handshakes")
    expected_processes = max(1, configuration.num_workers)
    expected_start_method = (
        "in-process" if configuration.num_workers == 0 else configuration.multiprocessing_start_method
    )
    if runtime.get("multiprocessing_start_method") != expected_start_method:
        raise BundleValidationError("runtime multiprocessing start method does not match configuration")
    if not isinstance(handshakes, list) or len(handshakes) != expected_processes:
        raise BundleValidationError("runtime does not contain the expected DataLoader worker handshakes")
    process_ids = _validate_worker_handshakes(
        handshakes,
        expected_processes=expected_processes,
        expected_start_method=expected_start_method,
        requested_threads=requested,
    )
    if runtime.get("persistent_workers_reused") is not True:
        raise BundleValidationError("runtime did not confirm DataLoader process reuse")
    coordinator_process_id = runtime.get("process_id")
    if isinstance(coordinator_process_id, bool) or not isinstance(coordinator_process_id, int):
        raise BundleValidationError("runtime process_id must be an integer")
    if configuration.num_workers == 0 and process_ids != {coordinator_process_id}:
        raise BundleValidationError("workers=0 handshake must come from the run process")
    if configuration.num_workers and coordinator_process_id in process_ids:
        raise BundleValidationError("DataLoader worker handshake came from the run process")


def _validate_worker_handshakes(
    handshakes: list[object],
    *,
    expected_processes: int,
    expected_start_method: str | None,
    requested_threads: int | None,
) -> set[int]:
    process_ids: set[int] = set()
    for handshake in handshakes:
        if not isinstance(handshake, dict):
            raise BundleValidationError("DataLoader worker handshake must be an object")
        process_id = handshake.get("process_id")
        if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
            raise BundleValidationError("DataLoader worker handshake has an invalid process_id")
        process_ids.add(process_id)
        _validate_effective_threads(handshake.get("effective_threads"), requested_threads)
        if handshake.get("multiprocessing_start_method") != expected_start_method:
            raise BundleValidationError("DataLoader worker start method does not match configuration")
    if len(process_ids) != expected_processes:
        raise BundleValidationError("DataLoader worker handshakes contain duplicate process IDs")
    return process_ids


def _validate_effective_threads(effective: object, requested: int | None) -> None:
    if isinstance(effective, bool) or not isinstance(effective, int) or effective <= 0:
        raise BundleValidationError("runtime effective_threads must be a positive integer")
    if requested is not None and effective != requested:
        raise BundleValidationError("runtime effective_threads does not match requested_threads")


def _validate_events(root: Path) -> None:
    events = _read_jsonl(root / "events.jsonl")
    event_names = {event.get("event") for event in events}
    required = {"validation", "warmup", "measurement_complete"}
    if not required.issubset(event_names):
        raise BundleValidationError(f"run events are missing required phases: {sorted(required - event_names)}")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"field {key!r} must be a non-empty string")
    return value


def _required_string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise BundleValidationError(f"field {key!r} must be a non-empty string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise BundleValidationError(f"field {key!r} must contain unique values")
    return result


def _required_non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BundleValidationError(f"field {key!r} must be a non-negative integer")
    return value


def _versioned(payload: dict[str, object]) -> dict[str, object]:
    if "schema_version" in payload and payload["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("payload has an incompatible schema_version")
    return {**payload, "schema_version": BUNDLE_SCHEMA_VERSION}


def _validate_digest(value: str, *, field: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _required_digest(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise BundleValidationError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleValidationError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise BundleValidationError(f"cannot read {path.name}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleValidationError(f"invalid JSON at {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise BundleValidationError(f"{path.name}:{line_number} must contain an object")
        rows.append(value)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _digest_json(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

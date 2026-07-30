from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imread_benchmark.artifacts import RunSample, validate_run_bundle


@dataclass(frozen=True, slots=True)
class RunBundleRecord:
    path: Path
    run_key: str
    bundle_id: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    environment: dict[str, Any]
    platform: dict[str, Any]
    runtime: dict[str, Any]
    samples: tuple[RunSample, ...]
    failures: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    bundle_manifest: dict[str, Any]


def load_bundles(root: str | Path) -> tuple[RunBundleRecord, ...]:
    root_path = Path(root).resolve()
    runs_root = root_path if root_path.name == "runs" else root_path / "runs"
    records: list[RunBundleRecord] = []
    for marker in sorted(runs_root.glob("*/COMMITTED.json")):
        bundle = marker.parent
        run_key = bundle.name
        validate_run_bundle(bundle, expected_run_key=run_key)
        commit = _read_object(marker)
        records.append(
            RunBundleRecord(
                path=bundle,
                run_key=run_key,
                bundle_id=_required_string(commit, "bundle_id"),
                config=_read_object(bundle / "config.json"),
                dataset=_read_object(bundle / "dataset.json"),
                environment=_read_object(bundle / "environment.json"),
                platform=_read_object(bundle / "platform.json"),
                runtime=_read_object(bundle / "runtime.json"),
                samples=tuple(_sample(row) for row in _read_jsonl(bundle / "samples.jsonl")),
                failures=tuple(_read_jsonl(bundle / "failures.jsonl")),
                events=tuple(_read_jsonl(bundle / "events.jsonl")),
                summary=_read_object(bundle / "summary.json"),
                bundle_manifest=_read_object(bundle / "bundle_manifest.json"),
            ),
        )
    return tuple(records)


def _sample(row: dict[str, Any]) -> RunSample:
    sample_index = row.get("sample_index")
    elapsed = row.get("elapsed_seconds")
    items = row.get("items_processed")
    started = row.get("started_at_utc")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise TypeError("elapsed_seconds must be numeric")
    if isinstance(items, bool) or not isinstance(items, int):
        raise TypeError("items_processed must be an integer")
    if started is not None and not isinstance(started, str):
        raise TypeError("started_at_utc must be a string or null")
    return RunSample(
        sample_index=sample_index,
        elapsed_seconds=float(elapsed),
        items_processed=items,
        started_at_utc=started,
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"field {key!r} must be a non-empty string")
    return value

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from imread_benchmark.artifacts import (
    BundleValidationError,
    publish_run_bundle,
    pull_committed_run,
    validate_run_bundle,
)
from imread_benchmark.execution.spec import write_run_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imread_benchmark.datasets.materializer import ObjectStore
    from imread_benchmark.execution.spec import RunSpec


class CoordinatorError(RuntimeError):
    pass


class AttemptStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed-out"


@dataclass(frozen=True, slots=True)
class AttemptResult:
    run_key: str
    status: AttemptStatus
    exit_code: int | None
    process_id: int | None
    attempt_directory: Path | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteCheckpoint:
    store: ObjectStore
    prefix: str = "artifacts"


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    artifact_root: Path
    attempts_root: Path
    timeout_seconds: float
    worker_command: tuple[str, ...] | None = None
    remote: RemoteCheckpoint | None = None


@dataclass(frozen=True, slots=True)
class _AttemptContext:
    directory: Path
    spec: RunSpec
    process_id: int
    started_at: str


def execute_run_specs(
    specs: Sequence[RunSpec],
    config: CoordinatorConfig,
) -> tuple[AttemptResult, ...]:
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if len({spec.run_key for spec in specs}) != len(specs):
        raise CoordinatorError("run specifications contain duplicate run_key values")
    artifact_path = config.artifact_root.resolve()
    attempts_path = config.attempts_root.resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)
    attempts_path.mkdir(parents=True, exist_ok=True)
    command_prefix = config.worker_command or (sys.executable, "-m", "imread_benchmark.execution.worker")
    if not command_prefix:
        raise ValueError("worker_command must not be empty")

    results: list[AttemptResult] = []
    for spec in specs:
        completed = artifact_path / "runs" / spec.run_key
        if not completed.exists() and config.remote is not None:
            pull_committed_run(
                spec.run_key,
                store=config.remote.store,
                artifact_root=artifact_path,
                prefix=config.remote.prefix,
            )
        if completed.exists():
            _validate_existing(completed, spec.run_key)
            results.append(
                AttemptResult(
                    run_key=spec.run_key,
                    status=AttemptStatus.SKIPPED,
                    exit_code=None,
                    process_id=None,
                    attempt_directory=None,
                    detail="valid committed bundle already exists",
                ),
            )
            continue
        result = _execute_one(
            spec,
            artifact_root=artifact_path,
            attempts_root=attempts_path,
            timeout_seconds=config.timeout_seconds,
            command_prefix=command_prefix,
        )
        if result.status is AttemptStatus.COMPLETED and config.remote is not None:
            publish_run_bundle(
                artifact_path / "runs" / spec.run_key,
                store=config.remote.store,
                prefix=config.remote.prefix,
            )
        results.append(result)
    return tuple(results)


def _validate_existing(path: Path, run_key: str) -> None:
    try:
        validate_run_bundle(path, expected_run_key=run_key)
    except (BundleValidationError, OSError) as exc:
        raise CoordinatorError(f"conflicting or incomplete bundle for run {run_key}: {exc}") from exc


def _execute_one(
    spec: RunSpec,
    *,
    artifact_root: Path,
    attempts_root: Path,
    timeout_seconds: float,
    command_prefix: tuple[str, ...],
) -> AttemptResult:
    attempt_id = f"{_utc_compact()}-{uuid.uuid4().hex}"
    attempt = attempts_root / spec.run_key / attempt_id
    attempt.mkdir(parents=True)
    spec_path = write_run_spec(attempt / "run-spec.json", spec)
    command = (
        *command_prefix,
        "--run-spec",
        str(spec_path),
        "--artifact-root",
        str(artifact_root),
    )
    started_at = _utc_now()
    with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(  # noqa: S603 - command is an explicit worker executable plus controlled arguments
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        context = _AttemptContext(
            directory=attempt,
            spec=spec,
            process_id=process.pid,
            started_at=started_at,
        )
        try:
            exit_code = _wait_with_heartbeat(process, context, timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return _finish_attempt(
                context,
                exit_code=process.returncode,
                status=AttemptStatus.TIMED_OUT,
                detail=f"worker exceeded timeout of {timeout_seconds:g} seconds",
            )
        except KeyboardInterrupt:
            _terminate_process_group(process)
            _write_status(
                context,
                exit_code=process.returncode,
                status="interrupted",
                detail="coordinator interrupted",
            )
            raise

    if exit_code != 0:
        return _finish_attempt(
            context,
            exit_code=exit_code,
            status=AttemptStatus.FAILED,
            detail="worker exited abnormally",
        )

    bundle = artifact_root / "runs" / spec.run_key
    try:
        validate_run_bundle(bundle, expected_run_key=spec.run_key)
    except (BundleValidationError, OSError) as exc:
        return _finish_attempt(
            context,
            exit_code=exit_code,
            status=AttemptStatus.FAILED,
            detail=f"worker did not produce a valid bundle: {exc}",
        )
    return _finish_attempt(
        context,
        exit_code=exit_code,
        status=AttemptStatus.COMPLETED,
        detail=None,
    )


def _wait_with_heartbeat(
    process: subprocess.Popen[bytes],
    context: _AttemptContext,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    _write_heartbeat(context)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        try:
            return process.wait(timeout=min(5.0, remaining))
        except subprocess.TimeoutExpired:
            _write_heartbeat(context)


def _write_heartbeat(context: _AttemptContext) -> None:
    document = {
        "process_id": context.process_id,
        "run_key": context.spec.run_key,
        "schema_version": "2.0",
        "status": "running",
        "updated_at_utc": _utc_now(),
    }
    path = context.directory / "heartbeat.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _finish_attempt(
    context: _AttemptContext,
    *,
    exit_code: int | None,
    status: AttemptStatus,
    detail: str | None,
) -> AttemptResult:
    _write_status(
        context,
        exit_code=exit_code,
        status=status.value,
        detail=detail,
    )
    return AttemptResult(
        run_key=context.spec.run_key,
        status=status,
        exit_code=exit_code,
        process_id=context.process_id,
        attempt_directory=context.directory,
        detail=detail,
    )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _write_status(
    context: _AttemptContext,
    *,
    exit_code: int | None,
    status: str,
    detail: str | None,
) -> None:
    document = {
        "detail": detail,
        "exit_code": exit_code,
        "finished_at_utc": _utc_now(),
        "process_id": context.process_id,
        "run_key": context.spec.run_key,
        "schema_version": "2.0",
        "started_at_utc": context.started_at,
        "status": status,
    }
    status_path = context.directory / "status.json"
    status_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _utc_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")

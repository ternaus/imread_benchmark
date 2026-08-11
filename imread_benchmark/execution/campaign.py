from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imread_benchmark.environments import load_environment_descriptor
from imread_benchmark.execution.coordinator import (
    AttemptResult,
    AttemptStatus,
    CoordinatorConfig,
    RemoteCheckpoint,
    execute_run_specs,
)
from imread_benchmark.execution.memory import estimate_peak_memory
from imread_benchmark.execution.spec import RunIdentity, RunSpec, write_run_spec
from imread_benchmark.plans import (
    ExperimentPlan,
    PlanError,
    RunConfiguration,
    RunTemplate,
    expand_experiment_plan,
    load_experiment_plan,
)
from imread_benchmark.platforms import load_platform_descriptor
from imread_benchmark.support import (
    SupportAudit,
    SupportSet,
    build_common_support,
    build_operational_support,
    intersect_supported_items,
    load_committed_support_audit,
    write_support_set,
)
from imread_benchmark.support.spec import SupportAuditIdentity, SupportAuditSpec, write_support_audit_spec

if TYPE_CHECKING:
    from collections.abc import Sequence


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    plan_path: Path
    package_descriptor: Path
    environment_descriptor: Path
    platform_descriptor: Path
    artifact_root: Path
    attempts_root: Path
    runner_revision: str
    worker_python: Path = Path(sys.executable)
    remote: RemoteCheckpoint | None = None


@dataclass(frozen=True, slots=True)
class CampaignResult:
    plan_id: str
    environment_id: str
    platform_id: str
    run_results: tuple[AttemptResult, ...]
    support_audit_count: int
    support_set_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return all(result.status in {AttemptStatus.COMPLETED, AttemptStatus.SKIPPED} for result in self.run_results)

    def to_dict(self) -> dict[str, object]:
        counts = {status.value: 0 for status in AttemptStatus}
        for result in self.run_results:
            counts[result.status.value] += 1
        return {
            "complete": self.complete,
            "environment_id": self.environment_id,
            "plan_id": self.plan_id,
            "platform_id": self.platform_id,
            "run_counts": counts,
            "run_keys": [result.run_key for result in self.run_results],
            "schema_version": "2.0",
            "support_audit_count": self.support_audit_count,
            "support_set_ids": list(self.support_set_ids),
        }


@dataclass(frozen=True, slots=True)
class _SupportContext:
    process_context: str
    multiprocessing_start_method: str | None


@dataclass(frozen=True, slots=True)
class _SupportExecution:
    artifact_root: Path
    attempts_root: Path
    timeout_seconds: float
    worker_python: Path


@dataclass(frozen=True, slots=True)
class _RunSpecContext:
    plan: ExperimentPlan
    environment_id: str
    platform_id: str
    runner_revision: str
    environment_descriptor: Path
    platform_descriptor: Path
    artifact_root: Path


def run_campaign(config: CampaignConfig) -> CampaignResult:
    plan = load_experiment_plan(config.plan_path, dataset_descriptor=config.package_descriptor)
    templates = expand_experiment_plan(plan)
    environment = load_environment_descriptor(config.environment_descriptor)
    platform_descriptor = load_platform_descriptor(config.platform_descriptor)
    if environment.runner_revision != config.runner_revision:
        raise CampaignError("environment runner revision does not match campaign runner revision")
    selected = tuple(
        template
        for template in templates
        if _decoder_group(template.configuration.decoder_id) == environment.dependency_group
    )
    if not selected:
        raise CampaignError(f"plan has no decoders in environment group {environment.dependency_group!r}")
    _validate_memory_budget(selected, plan=plan, platform_runtime=platform_descriptor.runtime)
    for template in selected:
        _validate_configuration_support(template.configuration, environment.distributions)

    artifact_root = config.artifact_root.resolve()
    attempts_root = config.attempts_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    attempts_root.mkdir(parents=True, exist_ok=True)
    audit_specs, audit_key_by_configuration = _support_specs(
        selected,
        plan=plan,
        environment_id=environment.environment_id,
        platform_id=platform_descriptor.platform_id,
        runner_revision=config.runner_revision,
    )
    audits = _execute_support_audits(
        audit_specs,
        artifact_root=artifact_root,
        attempts_root=attempts_root,
        timeout_seconds=float(plan.execution["run_timeout_seconds"]),
        worker_python=config.worker_python,
    )
    support_sets = _build_support_sets(
        selected,
        audits=audits,
        audit_key_by_configuration=audit_key_by_configuration,
        ordered_selection=plan.dataset.selection.item_ids,
        artifact_root=artifact_root,
    )
    run_spec_context = _RunSpecContext(
        plan=plan,
        environment_id=environment.environment_id,
        platform_id=platform_descriptor.platform_id,
        runner_revision=config.runner_revision,
        environment_descriptor=config.environment_descriptor,
        platform_descriptor=config.platform_descriptor,
        artifact_root=artifact_root,
    )
    specs = tuple(
        _run_spec(
            template,
            context=run_spec_context,
            support_set=support_sets[template.configuration.config_id],
        )
        for template in selected
    )
    _write_run_specs(artifact_root, specs)
    results = execute_run_specs(
        specs,
        CoordinatorConfig(
            artifact_root=artifact_root,
            attempts_root=attempts_root / "runs",
            timeout_seconds=float(plan.execution["run_timeout_seconds"]),
            worker_command=(str(config.worker_python), "-m", "imread_benchmark.execution.worker"),
            remote=config.remote,
        ),
    )
    first = selected[0]
    result = CampaignResult(
        plan_id=first.plan_id,
        environment_id=environment.environment_id,
        platform_id=platform_descriptor.platform_id,
        run_results=results,
        support_audit_count=len(audits),
        support_set_ids=tuple(sorted({support.support_set_id for support in support_sets.values()})),
    )
    _write_campaign_index(artifact_root, result)
    return result


def _validate_memory_budget(
    templates: Sequence[RunTemplate],
    *,
    plan: ExperimentPlan,
    platform_runtime: dict[str, object],
) -> None:
    memory_bytes = platform_runtime.get("memory_bytes")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes <= 0:
        raise CampaignError("platform runtime must report positive memory_bytes for campaign preflight")
    fraction = plan.execution.get("maximum_memory_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise CampaignError("plan has no valid maximum_memory_fraction")
    selected_ids = set(plan.dataset.selection.item_ids)
    raw_items = plan.dataset.manifest.get("items")
    if not isinstance(raw_items, list) or not all(isinstance(item, dict) for item in raw_items):
        raise CampaignError("dataset manifest has no valid items for memory preflight")
    items = tuple(item for item in raw_items if item.get("item_id") in selected_ids)
    limit = int(memory_bytes * float(fraction))
    configurations = {template.configuration.config_id: template.configuration for template in templates}
    for configuration in configurations.values():
        try:
            estimate = estimate_peak_memory(configuration, items)
        except ValueError as exc:
            raise CampaignError(f"cannot estimate memory for {configuration.config_id}: {exc}") from exc
        if estimate.estimated_peak_bytes > limit:
            raise CampaignError(
                f"configuration {configuration.config_id} needs an estimated {estimate.estimated_peak_bytes} bytes, "
                f"above the {limit}-byte campaign memory limit ({float(fraction):.3f} of platform RAM)",
            )


def _support_specs(
    templates: Sequence[RunTemplate],
    *,
    plan: ExperimentPlan,
    environment_id: str,
    platform_id: str,
    runner_revision: str,
) -> tuple[tuple[SupportAuditSpec, ...], dict[str, str]]:
    specs: dict[str, SupportAuditSpec] = {}
    keys: dict[str, str] = {}
    for template in templates:
        configuration = template.configuration
        context = _support_context(configuration)
        spec = SupportAuditSpec.build(
            identity=SupportAuditIdentity(
                decoder_id=configuration.decoder_id,
                requested_threads=configuration.requested_threads,
                package_id=plan.dataset.package_id,
                workload_id=plan.dataset.workload_id,
                manifest_id=plan.dataset.manifest_id,
                selection_id=plan.dataset.selection.selection_id,
                item_ids=plan.dataset.selection.item_ids,
                process_context=context.process_context,
                multiprocessing_start_method=context.multiprocessing_start_method,
                environment_id=environment_id,
                platform_id=platform_id,
                runner_revision=runner_revision,
            ),
            package_descriptor=plan.dataset.descriptor_path,
        )
        specs.setdefault(spec.audit_key, spec)
        keys[configuration.config_id] = spec.audit_key
    return tuple(sorted(specs.values(), key=lambda value: value.audit_key)), keys


def _execute_support_audits(
    specs: Sequence[SupportAuditSpec],
    *,
    artifact_root: Path,
    attempts_root: Path,
    timeout_seconds: float,
    worker_python: Path,
) -> dict[str, SupportAudit]:
    support_root = artifact_root / "support"
    spec_root = artifact_root / "specs" / "support"
    audits: dict[str, SupportAudit] = {}
    execution = _SupportExecution(artifact_root, attempts_root / "support", timeout_seconds, worker_python)
    for spec in specs:
        marker = support_root / "keys" / f"{spec.audit_key}.json"
        if not marker.exists():
            spec_path = write_support_audit_spec(spec_root / f"{spec.audit_key}.json", spec)
            _execute_support_audit_process(
                spec_path,
                spec.audit_key,
                execution,
            )
        try:
            audits[spec.audit_key] = load_committed_support_audit(support_root, spec.audit_key)
        except (OSError, TypeError, ValueError) as exc:
            raise CampaignError(f"support audit {spec.audit_key} is incomplete or invalid: {exc}") from exc
    return audits


def _execute_support_audit_process(
    spec_path: Path,
    audit_key: str,
    execution: _SupportExecution,
) -> None:
    attempt = execution.attempts_root / audit_key / f"{_utc_compact()}-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True)
    command = (
        str(execution.worker_python),
        "-m",
        "imread_benchmark.support.worker",
        "--audit-spec",
        str(spec_path),
        "--artifact-root",
        str(execution.artifact_root),
    )
    with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)  # noqa: S603
        try:
            exit_code = process.wait(timeout=execution.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            _write_support_status(attempt, audit_key, "timed-out", process.returncode)
            raise CampaignError(f"support audit {audit_key} exceeded timeout") from exc
    status = "completed" if exit_code == 0 else "failed"
    _write_support_status(attempt, audit_key, status, exit_code)
    if exit_code != 0:
        raise CampaignError(f"support audit {audit_key} failed with exit code {exit_code}")


def _build_support_sets(
    templates: Sequence[RunTemplate],
    *,
    audits: dict[str, SupportAudit],
    audit_key_by_configuration: dict[str, str],
    ordered_selection: Sequence[str],
    artifact_root: Path,
) -> dict[str, SupportSet]:
    configurations = {template.configuration.config_id: template.configuration for template in templates}
    by_configuration: dict[str, SupportSet] = {}
    common_groups: dict[_SupportContext, set[str]] = {}
    for config_id, configuration in configurations.items():
        audit_key = audit_key_by_configuration[config_id]
        if configuration.support_policy == "operational":
            by_configuration[config_id] = build_operational_support(audits[audit_key])
        else:
            common_groups.setdefault(_support_context(configuration), set()).add(audit_key)
    global_common_ids: tuple[str, ...] | None = None
    if common_groups:
        common_audit_keys = sorted({key for keys in common_groups.values() for key in keys})
        global_common_ids = intersect_supported_items(
            tuple(audits[key] for key in common_audit_keys),
            ordered_selection=ordered_selection,
        )
    for context, audit_keys in common_groups.items():
        if global_common_ids is None:
            raise AssertionError("common support intersection was not computed")
        support_set = build_common_support(
            tuple(audits[key] for key in sorted(audit_keys)),
            ordered_selection=global_common_ids,
        )
        for config_id, configuration in configurations.items():
            if configuration.support_policy == "common" and _support_context(configuration) == context:
                by_configuration[config_id] = support_set
    for support_set in set(by_configuration.values()):
        write_support_set(artifact_root / "support", support_set)
    if set(by_configuration) != set(configurations):
        raise AssertionError("campaign did not assign a support set to every configuration")
    return by_configuration


def _run_spec(
    template: RunTemplate,
    *,
    context: _RunSpecContext,
    support_set: SupportSet,
) -> RunSpec:
    configuration = template.configuration
    return RunSpec.build(
        identity=RunIdentity(
            plan_id=template.plan_id,
            platform_id=context.platform_id,
            environment_id=context.environment_id,
            runner_revision=context.runner_revision,
            workload_id=context.plan.dataset.workload_id,
            support_set_id=support_set.support_set_id,
            support_item_ids=support_set.item_ids,
            configuration=configuration,
            repetition=template.repetition,
            block_position=template.position,
        ),
        package_descriptor=context.plan.dataset.descriptor_path,
        support_set_path=context.artifact_root / "support" / "sets" / f"{support_set.support_set_id}.json",
        environment_descriptor=context.environment_descriptor,
        platform_descriptor=context.platform_descriptor,
    )


def _write_run_specs(artifact_root: Path, specs: Sequence[RunSpec]) -> None:
    root = artifact_root / "specs" / "runs"
    for spec in specs:
        write_run_spec(root / f"{spec.run_key}.json", spec)


def _support_context(configuration: RunConfiguration) -> _SupportContext:
    if configuration.protocol_id == "loader-supply" and configuration.num_workers:
        return _SupportContext("dataloader", configuration.multiprocessing_start_method)
    return _SupportContext("main-process", None)


def _decoder_group(decoder_id: str) -> str:
    from imread_benchmark.decoders import REGISTRY

    decoder = REGISTRY.get(decoder_id)
    if decoder is None:
        raise PlanError(f"unknown decoder in experiment plan: {decoder_id}")
    return decoder.group


def _validate_configuration_support(
    configuration: RunConfiguration,
    distributions: tuple[tuple[str, str], ...],
) -> None:
    from imread_benchmark.decoders import REGISTRY

    decoder = REGISTRY[configuration.decoder_id]
    installed = {name for name, _ in distributions}
    normalized_package = _normalize_package_name(decoder.package_name)
    if normalized_package not in installed:
        raise CampaignError(
            f"environment does not contain package {normalized_package!r} for decoder {configuration.decoder_id!r}",
        )
    if not decoder.runs_single_here():
        raise CampaignError(f"decoder {configuration.decoder_id!r} does not support this platform")
    requires_workers = configuration.protocol_id == "loader-supply" and bool(configuration.num_workers)
    if requires_workers and not decoder.runs_dataloader_here():
        raise CampaignError(f"decoder {configuration.decoder_id!r} does not support DataLoader workers here")


def _normalize_package_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


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


def _write_support_status(attempt: Path, audit_key: str, status: str, exit_code: int | None) -> None:
    document = {
        "audit_key": audit_key,
        "exit_code": exit_code,
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "schema_version": "2.0",
        "status": status,
    }
    (attempt / "status.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_campaign_index(artifact_root: Path, result: CampaignResult) -> None:
    path = artifact_root / "campaigns" / result.plan_id / result.platform_id / f"{result.environment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _utc_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")

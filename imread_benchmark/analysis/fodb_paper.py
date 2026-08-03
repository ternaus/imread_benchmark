from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imread_benchmark.analysis.canonical import RunBundleRecord, load_bundles
from imread_benchmark.analysis.claims import ClaimScope, assert_claim_scope

NATIVE_PLAN_ID = "6c746a5c21978f75c58c8c779f6628c2645a199b63489da9b9f1a9bb4f335c8f"
MIXED_PLAN_ID = "e04f54dc177cadf86cb99a9fc34b950afb5b1abee5b46af040bafc8a4aa7910b"
WORKER12_ALL_NATIVE_PLAN_ID = "4af3621c0fc5b75525a54eb4bc7171cb5399938b8515ea2c920a8a5e799d89ed"
WORKER12_ALL_MIXED_PLAN_ID = "1ad7306a07a7d23a602a5a74230e2d4edf5a0aaa55a61184fceb4718b64496f9"
WORKER12_INTEL_NATIVE_PLAN_ID = "c2f9038c67be2282d9aa15511bb57e93f1e9f80e86c392d77aaa6ddcd3d4cf54"
WORKER12_ZEN4_NATIVE_PLAN_ID = "a0e63ca15064755badb98b9b466eda3f9716f39f1515fae5e0b65fb74e6ade74"
WORKER16_MIXED_PLAN_ID = "eeef8895e513b6051c641ec49b97cea3d7878b08b0a2677d519eb2c70310f5e9"
WORKER16_INTEL_NATIVE_PLAN_ID = "4e5cba9f0d95976fd11638cbc71a22d018f044cb80cc89057cefd25ca5180f68"
WORKER16_AXION_NATIVE_PLAN_ID = "e246e31630e83ec09bf77c2f728c5ae3169d8717b630326e3ef568e418db193f"
RUNNER_REVISION = "52a0bea5a2f44079883c9a41472b3add285955f2b471c37674ddd2fb5d4da6ff"

BASE_PLAN_WORKLOADS = {NATIVE_PLAN_ID: "fodb-native", MIXED_PLAN_ID: "fodb-mixed"}
WORKER12_PLAN_WORKLOADS = {
    WORKER12_ALL_NATIVE_PLAN_ID: "fodb-native",
    WORKER12_ALL_MIXED_PLAN_ID: "fodb-mixed",
    WORKER12_INTEL_NATIVE_PLAN_ID: "fodb-native",
    WORKER12_ZEN4_NATIVE_PLAN_ID: "fodb-native",
}
WORKER16_PLAN_WORKLOADS = {
    WORKER16_MIXED_PLAN_ID: "fodb-mixed",
    WORKER16_INTEL_NATIVE_PLAN_ID: "fodb-native",
    WORKER16_AXION_NATIVE_PLAN_ID: "fodb-native",
}
PLAN_WORKLOADS = BASE_PLAN_WORKLOADS | WORKER12_PLAN_WORKLOADS | WORKER16_PLAN_WORKLOADS
EXPECTED_MACHINES = (
    "c4-standard-16",
    "c3d-standard-16",
    "c4d-standard-16",
    "c4a-standard-16",
)
PLATFORM_LABELS = {
    "c4-standard-16": "Intel 8581C",
    "c3d-standard-16": "AMD Zen 4",
    "c4d-standard-16": "AMD Zen 5",
    "c4a-standard-16": "Axion",
}
EXPECTED_REPETITIONS = tuple(range(5))
BASE_WORKERS = (0, 2, 4, 8)
WORKER12_DECISION_GRID = (*BASE_WORKERS, 12)
WORKER_GRID = (*WORKER12_DECISION_GRID, 16)
DEPLOYMENT_WORKERS = (4, 8, 12, 16)
EXPECTED_ITEM_COUNTS = {"fodb-native": 324, "fodb-mixed": 1944}
EXPECTED_TIMED_ITEM_COUNTS = {"fodb-native": 324, "fodb-mixed": 1668}
CONTROLLED_THREADS = {"opencv": 1, "pyvips": 1, "torchvision": 1}
ALL_DECODERS = (
    "ajpegli",
    "imagecodecs",
    "imageio",
    "jpeg4py",
    "kornia",
    "opencv",
    "pillow",
    "pyvips",
    "simplejpeg",
    "skimage",
    "torchvision",
    "turbojpeg",
)
WORKER12_DECODERS = {
    **{("fodb-mixed", machine_type): ALL_DECODERS for machine_type in EXPECTED_MACHINES},
    ("fodb-native", "c4-standard-16"): (
        "ajpegli",
        "imageio",
        "kornia",
        "opencv",
        "pillow",
        "pyvips",
        "skimage",
        "torchvision",
    ),
    ("fodb-native", "c3d-standard-16"): (
        "ajpegli",
        "imageio",
        "opencv",
        "pillow",
        "pyvips",
        "skimage",
        "torchvision",
    ),
    ("fodb-native", "c4d-standard-16"): ALL_DECODERS,
    ("fodb-native", "c4a-standard-16"): ALL_DECODERS,
}
WORKER12_PLAN_SCOPES: dict[str, dict[tuple[str, str], tuple[str, ...]]] = {
    WORKER12_ALL_NATIVE_PLAN_ID: {
        ("fodb-native", "c4d-standard-16"): ALL_DECODERS,
        ("fodb-native", "c4a-standard-16"): ALL_DECODERS,
    },
    WORKER12_ALL_MIXED_PLAN_ID: {("fodb-mixed", machine_type): ALL_DECODERS for machine_type in EXPECTED_MACHINES},
    WORKER12_INTEL_NATIVE_PLAN_ID: {
        ("fodb-native", "c4-standard-16"): WORKER12_DECODERS[("fodb-native", "c4-standard-16")],
    },
    WORKER12_ZEN4_NATIVE_PLAN_ID: {
        ("fodb-native", "c3d-standard-16"): WORKER12_DECODERS[("fodb-native", "c3d-standard-16")],
    },
}
WORKER16_DECODERS = {
    **{("fodb-mixed", machine_type): ALL_DECODERS for machine_type in EXPECTED_MACHINES},
    ("fodb-native", "c4-standard-16"): WORKER12_DECODERS[("fodb-native", "c4-standard-16")],
    ("fodb-native", "c4a-standard-16"): ("ajpegli", "pillow"),
    ("fodb-native", "c3d-standard-16"): (),
    ("fodb-native", "c4d-standard-16"): (),
}
WORKER16_PLAN_SCOPES: dict[str, dict[tuple[str, str], tuple[str, ...]]] = {
    WORKER16_MIXED_PLAN_ID: {("fodb-mixed", machine_type): ALL_DECODERS for machine_type in EXPECTED_MACHINES},
    WORKER16_INTEL_NATIVE_PLAN_ID: {
        ("fodb-native", "c4-standard-16"): WORKER16_DECODERS[("fodb-native", "c4-standard-16")],
    },
    WORKER16_AXION_NATIVE_PLAN_ID: {
        ("fodb-native", "c4a-standard-16"): WORKER16_DECODERS[("fodb-native", "c4a-standard-16")],
    },
}
FOLLOWUP_PLAN_SCOPES = WORKER12_PLAN_SCOPES | WORKER16_PLAN_SCOPES
FOLLOWUP_PLAN_WORKERS = dict.fromkeys(WORKER12_PLAN_SCOPES, 12) | dict.fromkeys(WORKER16_PLAN_SCOPES, 16)
WORKER16_MINIMUM_GAIN = 0.05
PRIMARY_DECODERS = ("pillow", "opencv", "simplejpeg", "torchvision")
MIGRATION_DECODERS = ("pillow", "opencv", "simplejpeg")
PRACTICAL_MARGIN = 0.10
ROBUSTNESS_AUDIT_PACKAGE_ID = "b797eb3938cad77e115b2315b41457c5d6b3062968f7529ee35d1b38dd87eb2f"
ROBUSTNESS_AUDIT_MANIFEST_ID = "47f437595cf8a3215deb958f7278d63bcc6b2a00d4d595d821351431ceda803f"
ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT = 276
ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES = {
    "ajpegli": 0,
    "imagecodecs": 276,
    "imageio": 276,
    "jpeg4py": 276,
    "kornia": 276,
    "opencv": 276,
    "pillow": 276,
    "pyvips": 276,
    "simplejpeg": 276,
    "skimage": 276,
    "torchvision": 276,
    "turbojpeg": 276,
}
ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT = 1
ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES = {
    "ajpegli": 0,
    "imagecodecs": 0,
    "imageio": 0,
    "jpeg4py": 0,
    "kornia": 0,
    "opencv": 1,
    "pillow": 1,
    "pyvips": 0,
    "simplejpeg": 1,
    "skimage": 0,
    "torchvision": 0,
    "turbojpeg": 0,
}
ROBUSTNESS_AUDIT_ITEM_COUNT = ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT + ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT
ROBUSTNESS_AUDIT_SUCCESSES = {
    decoder: successes + ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES[decoder]
    for decoder, successes in ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES.items()
}


class PaperAssetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Measurement:
    workload: str
    machine_type: str
    protocol: str
    decoder: str
    requested_threads: int | None
    workers: int | None
    repetition: int
    images_per_second: float
    run_key: str
    bundle_id: str

    @property
    def configuration_label(self) -> str:
        thread_label = "default" if self.requested_threads is None else str(self.requested_threads)
        return f"{self.decoder}[threads={thread_label}]"


@dataclass(frozen=True, slots=True)
class Aggregate:
    workload: str
    machine_type: str
    protocol: str
    decoder: str
    requested_threads: int | None
    workers: int | None
    repetitions: tuple[int, ...]
    raw_run_means: tuple[float, ...]
    mean: float
    sample_std: float

    @property
    def configuration_label(self) -> str:
        thread_label = "default" if self.requested_threads is None else str(self.requested_threads)
        return f"{self.decoder}[threads={thread_label}]"


def build_paper_assets(*, artifact_root: Path, package_path: Path, output_root: Path) -> dict[str, Any]:
    records = tuple(record for record in load_bundles(artifact_root) if record.config.get("plan_id") in PLAN_WORKLOADS)
    package = _read_object(package_path)
    measurements = _validate_and_measure(records, package)
    aggregates = _aggregate(measurements)
    manifests = _load_package_manifests(package_path, package)
    support_item_ids = _support_item_ids(records)
    workload_descriptors = _workload_descriptors(package, manifests, support_item_ids)
    decisions = _decision_rows(aggregates)
    recommendations = _recommendation_rows(aggregates)
    pillow_rows = _pillow_rows(aggregates)
    thread_rows = _thread_rows(aggregates)
    worker_transfer_rows = _worker_transfer_rows(aggregates)
    worker16_candidates = _worker16_candidate_rows(aggregates)
    sections = {
        "decisions": decisions,
        "pillow_migration": pillow_rows,
        "recommendations": recommendations,
        "thread_controls": thread_rows,
        "worker_transfer": worker_transfer_rows,
        "worker16_candidates": worker16_candidates,
        "workloads": workload_descriptors,
    }
    evidence = _evidence_document(records, measurements, aggregates, sections)

    generated_dir = output_root / "generated"
    figures_dir = output_root / "figures"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_json(generated_dir / "fodb_evidence.json", evidence)
    _write_json(generated_dir / "fodb_worker16_candidates.json", worker16_candidates)
    _write_text(generated_dir / "fodb_results_summary.md", _summary_markdown(evidence))
    _write_text(generated_dir / "table_fodb_workloads.tex", _workload_table(workload_descriptors))
    _write_text(generated_dir / "table_fodb_provenance.tex", _provenance_table(workload_descriptors))
    _write_text(generated_dir / "table_fodb_decisions.tex", _decision_table(decisions))
    _write_text(
        generated_dir / "table_fodb_recommendations.tex",
        _recommendation_table(recommendations),
    )
    _write_text(
        generated_dir / "table_fodb_decoder_coverage.tex",
        _decoder_coverage_table(recommendations),
    )
    _write_text(generated_dir / "table_fodb_pillow.tex", _pillow_table(pillow_rows))
    _write_text(
        generated_dir / "table_fodb_worker_transfer.tex",
        _worker_transfer_table(worker_transfer_rows),
    )
    _write_text(generated_dir / "table_fodb_coverage.tex", _coverage_table(evidence))
    _write_text(generated_dir / "table_fodb_versions.tex", _versions_table(records))
    _plot_workloads(package, manifests, support_item_ids, figures_dir / "fig_fodb_workload_distributions.pdf")
    _plot_worker_scaling(aggregates, figures_dir / "fig_fodb_worker_scaling.pdf")
    _plot_protocol_regret(decisions, figures_dir / "fig_fodb_protocol_regret.pdf")
    _plot_recommendations(recommendations, figures_dir / "fig_fodb_recommendations.pdf")
    return evidence


def _validate_and_measure(records: tuple[RunBundleRecord, ...], package: dict[str, Any]) -> tuple[Measurement, ...]:
    base_total = len(BASE_PLAN_WORKLOADS) * len(EXPECTED_MACHINES) * 75 * len(EXPECTED_REPETITIONS)
    worker12_total = sum(len(decoders) * len(EXPECTED_REPETITIONS) for decoders in WORKER12_DECODERS.values())
    worker16_total = sum(len(decoders) * len(EXPECTED_REPETITIONS) for decoders in WORKER16_DECODERS.values())
    expected_total = base_total + worker12_total + worker16_total
    if len(records) != expected_total:
        raise PaperAssetError(f"expected {expected_total} evidence bundles, found {len(records)}")

    decode_records = tuple(record for record in records if record.config.get("protocol_id") == "decode-memory")
    loader_records = tuple(record for record in records if record.config.get("protocol_id") == "loader-supply")
    assert_claim_scope(decode_records, ClaimScope.DECODER_CAPACITY)
    assert_claim_scope(loader_records, ClaimScope.LOADER_SUPPLY)

    package_workloads = _required_object(package, "workloads")
    manifest_ids = {
        workload: _required_string(_required_object(package_workloads, workload), "manifest_id")
        for workload in EXPECTED_ITEM_COUNTS
    }
    measurements: list[Measurement] = []
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    configuration_repetitions: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    worker12_decoders: dict[tuple[str, str], set[str]] = defaultdict(set)
    worker16_decoders: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        measurement = _measurement(record, manifest_ids)
        measurements.append(measurement)
        cell_counts[(measurement.workload, measurement.machine_type, measurement.protocol)] += 1
        if measurement.protocol == "loader-supply" and measurement.workers == 12:
            worker12_decoders[(measurement.workload, measurement.machine_type)].add(measurement.decoder)
        if measurement.protocol == "loader-supply" and measurement.workers == 16:
            worker16_decoders[(measurement.workload, measurement.machine_type)].add(measurement.decoder)
        config_id = _required_string(record.config, "config_id")
        configuration_repetitions[(measurement.workload, measurement.machine_type, config_id)].add(
            measurement.repetition,
        )

    _validate_matrix_counts(cell_counts, configuration_repetitions)
    for cell, expected_decoders in WORKER12_DECODERS.items():
        if worker12_decoders[cell] != set(expected_decoders):
            raise PaperAssetError(f"workers=12 decoder selection does not match the frozen follow-up for {cell}")
    for cell, expected_decoders in WORKER16_DECODERS.items():
        if worker16_decoders[cell] != set(expected_decoders):
            raise PaperAssetError(f"workers=16 decoder selection does not match the frozen follow-up for {cell}")
    return tuple(measurements)


def _measurement(record: RunBundleRecord, manifest_ids: dict[str, str]) -> Measurement:
    config = record.config
    plan_id = _required_string(config, "plan_id")
    workload = PLAN_WORKLOADS[plan_id]
    machine_type = _required_string(_required_object(record.platform, "identity"), "machine_type")
    if machine_type not in EXPECTED_MACHINES:
        raise PaperAssetError(f"unexpected machine type {machine_type!r}")
    if plan_id in FOLLOWUP_PLAN_SCOPES:
        eligible_decoders = FOLLOWUP_PLAN_SCOPES[plan_id].get((workload, machine_type))
        followup_workers = FOLLOWUP_PLAN_WORKERS[plan_id]
        decoder = _required_string(config, "decoder_id")
        requested_threads = _optional_int(config, "requested_threads")
        followup_checks = (
            (eligible_decoders is not None, "follow-up plan/platform mismatch"),
            (config.get("protocol_id") == "loader-supply", "follow-up protocol is not loader-supply"),
            (config.get("num_workers") == followup_workers, "follow-up worker count is unexpected"),
            (eligible_decoders is not None and decoder in eligible_decoders, "decoder was not eligible for follow-up"),
            (requested_threads == CONTROLLED_THREADS.get(decoder), "follow-up thread profile is not controlled"),
        )
        for valid, message in followup_checks:
            if not valid:
                raise PaperAssetError(f"{message} in {record.run_key}")
    checks = (
        (config.get("runner_revision") == RUNNER_REVISION, "unexpected runner revision"),
        (config.get("output_contract") == "normalized-rgb", "unexpected output contract"),
        (record.dataset.get("workload_id") == workload, "plan/workload mismatch"),
        (record.dataset.get("manifest_id") == manifest_ids[workload], "manifest mismatch"),
        (not record.failures, "timed failures are present"),
        (record.summary.get("status") == "complete", "incomplete summary"),
        (len(record.samples) == 1, "expected one timed traversal"),
    )
    for valid, message in checks:
        if not valid:
            raise PaperAssetError(f"{message} in {record.run_key}")
    ordered_items = record.dataset.get("ordered_item_ids")
    if not isinstance(ordered_items, list) or len(ordered_items) != EXPECTED_TIMED_ITEM_COUNTS[workload]:
        raise PaperAssetError(f"unexpected timed support population for {workload} in {record.run_key}")
    sample = record.samples[0]
    return Measurement(
        workload=workload,
        machine_type=machine_type,
        protocol=_required_string(config, "protocol_id"),
        decoder=_required_string(config, "decoder_id"),
        requested_threads=_optional_int(config, "requested_threads"),
        workers=_optional_int(config, "num_workers"),
        repetition=_required_int(config, "repetition"),
        images_per_second=sample.items_processed / sample.elapsed_seconds,
        run_key=record.run_key,
        bundle_id=record.bundle_id,
    )


def _validate_matrix_counts(
    cell_counts: Counter[tuple[str, str, str]],
    configuration_repetitions: dict[tuple[str, str, str], set[int]],
) -> None:
    for workload in EXPECTED_ITEM_COUNTS:
        for machine_type in EXPECTED_MACHINES:
            if cell_counts[(workload, machine_type, "decode-memory")] != 75:
                raise PaperAssetError(f"incomplete decode-memory matrix for {workload}/{machine_type}")
            worker12_count = len(WORKER12_DECODERS[(workload, machine_type)]) * len(EXPECTED_REPETITIONS)
            worker16_count = len(WORKER16_DECODERS[(workload, machine_type)]) * len(EXPECTED_REPETITIONS)
            followup_count = worker12_count + worker16_count
            if cell_counts[(workload, machine_type, "loader-supply")] != 300 + followup_count:
                raise PaperAssetError(f"incomplete loader-supply matrix for {workload}/{machine_type}")
    expected_repetitions = set(EXPECTED_REPETITIONS)
    if any(repetitions != expected_repetitions for repetitions in configuration_repetitions.values()):
        raise PaperAssetError("at least one configuration lacks the five repetition blocks")


def _aggregate(measurements: tuple[Measurement, ...]) -> tuple[Aggregate, ...]:
    grouped: dict[tuple[object, ...], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        key = (
            measurement.workload,
            measurement.machine_type,
            measurement.protocol,
            measurement.decoder,
            measurement.requested_threads,
            measurement.workers,
        )
        grouped[key].append(measurement)
    aggregates: list[Aggregate] = []
    for aggregate_key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row.repetition)
        repetitions = tuple(row.repetition for row in ordered)
        if repetitions != EXPECTED_REPETITIONS:
            raise PaperAssetError(f"aggregate has repetition blocks {repetitions}, expected {EXPECTED_REPETITIONS}")
        raw = tuple(row.images_per_second for row in ordered)
        workload, machine_type, protocol, decoder, requested_threads, workers = aggregate_key
        aggregates.append(
            Aggregate(
                workload=str(workload),
                machine_type=str(machine_type),
                protocol=str(protocol),
                decoder=str(decoder),
                requested_threads=requested_threads if isinstance(requested_threads, int) else None,
                workers=workers if isinstance(workers, int) else None,
                repetitions=repetitions,
                raw_run_means=raw,
                mean=statistics.fmean(raw),
                sample_std=statistics.stdev(raw),
            ),
        )
    return tuple(sorted(aggregates, key=_aggregate_sort_key))


def _aggregate_sort_key(row: Aggregate) -> tuple[str, ...]:
    return (
        row.workload,
        row.machine_type,
        row.protocol,
        row.decoder,
        str(row.requested_threads),
        str(row.workers),
    )


def _is_controlled(row: Aggregate) -> bool:
    return row.requested_threads == CONTROLLED_THREADS.get(row.decoder)


def _decision_rows(aggregates: tuple[Aggregate, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in EXPECTED_ITEM_COUNTS:
        for machine_type in EXPECTED_MACHINES:
            block = tuple(
                row
                for row in aggregates
                if row.workload == workload and row.machine_type == machine_type and _is_controlled(row)
            )
            decode = tuple(row for row in block if row.protocol == "decode-memory")
            loader = tuple(row for row in block if row.protocol == "loader-supply")
            expected_loader_count = (
                48 + len(WORKER12_DECODERS[(workload, machine_type)]) + len(WORKER16_DECODERS[(workload, machine_type)])
            )
            if len(decode) != 12 or len(loader) != expected_loader_count:
                raise PaperAssetError(f"controlled-thread block is incomplete for {workload}/{machine_type}")
            decode_leader = _best(decode)
            peak_by_decoder = {
                decoder: _best(tuple(row for row in loader if row.decoder == decoder)) for decoder in _decoders(decode)
            }
            loader_leader = _best(tuple(peak_by_decoder.values()))
            selected_loader = peak_by_decoder[decode_leader.decoder]
            aggregate_regret = 1 - selected_loader.mean / loader_leader.mean
            paired_regret = tuple(
                1 - selected / leader
                for selected, leader in zip(selected_loader.raw_run_means, loader_leader.raw_run_means, strict=True)
            )
            decode_ranks = _ranks({row.decoder: row.mean for row in decode})
            loader_ranks = _ranks({decoder: row.mean for decoder, row in peak_by_decoder.items()})
            rank_correlation = statistics.correlation(
                [decode_ranks[decoder] for decoder in sorted(decode_ranks)],
                [loader_ranks[decoder] for decoder in sorted(loader_ranks)],
            )
            threshold = loader_leader.mean * (1 - PRACTICAL_MARGIN)
            tier = [
                {
                    "decoder": row.decoder,
                    "mean_images_per_second": row.mean,
                    "workers": row.workers,
                }
                for row in sorted(peak_by_decoder.values(), key=lambda item: (-item.mean, item.configuration_label))
                if row.mean >= threshold
            ]
            rows.append(
                {
                    "aggregate_regret_percent": aggregate_regret * 100,
                    "decode_leader": decode_leader.decoder,
                    "decode_leader_images_per_second": decode_leader.mean,
                    "loader_leader": loader_leader.decoder,
                    "loader_leader_images_per_second": loader_leader.mean,
                    "loader_leader_workers": loader_leader.workers,
                    "machine_type": machine_type,
                    "paired_regret_percent": [value * 100 for value in paired_regret],
                    "platform": PLATFORM_LABELS[machine_type],
                    "rank_correlation": rank_correlation,
                    "selected_decoder_loader_workers": selected_loader.workers,
                    "strict_leader_match": decode_leader.decoder == loader_leader.decoder,
                    "top_tier": tier,
                    "workload": workload,
                },
            )
    return rows


def _recommendation_rows(
    aggregates: tuple[Aggregate, ...],
    robustness_audit_successes: dict[str, int] | None = None,
) -> dict[str, Any]:
    if robustness_audit_successes is None:
        audit_successes = ROBUSTNESS_AUDIT_SUCCESSES
        uses_recorded_audit = True
    else:
        audit_successes = robustness_audit_successes
        uses_recorded_audit = False
    cells: list[dict[str, Any]] = []
    gaps_by_decoder: dict[str, list[float]] = defaultdict(list)
    for machine_type in EXPECTED_MACHINES:
        for workload in EXPECTED_ITEM_COUNTS:
            candidates = tuple(
                row
                for row in aggregates
                if row.workload == workload
                and row.machine_type == machine_type
                and row.protocol == "loader-supply"
                and row.workers in DEPLOYMENT_WORKERS
                and _is_controlled(row)
            )
            peak_by_decoder = {
                decoder: _best(tuple(row for row in candidates if row.decoder == decoder))
                for decoder in _decoders(candidates)
            }
            if len(peak_by_decoder) != 12:
                raise PaperAssetError(f"recommendation block is incomplete for {workload}/{machine_type}")
            if set(peak_by_decoder) != set(audit_successes):
                raise PaperAssetError("recommendation decoders do not match the robustness audit")
            leader = _best(tuple(peak_by_decoder.values()))
            decoder_rows = []
            for decoder, peak in sorted(peak_by_decoder.items()):
                gap_percent = (1 - peak.mean / leader.mean) * 100
                gaps_by_decoder[decoder].append(gap_percent)
                audited_successes = audit_successes[decoder]
                passes_robustness_audit = audited_successes == ROBUSTNESS_AUDIT_ITEM_COUNT
                decoder_rows.append(
                    {
                        "audited_successes": audited_successes,
                        "coverage_qualified": passes_robustness_audit,
                        "decoder": decoder,
                        "gap_from_leader_percent": gap_percent,
                        "mean_images_per_second": peak.mean,
                        "recommended": passes_robustness_audit and gap_percent <= PRACTICAL_MARGIN * 100,
                        "within_speed_margin": gap_percent <= PRACTICAL_MARGIN * 100,
                        "workers": peak.workers,
                    },
                )
            recommended = [
                row["decoder"]
                for row in sorted(decoder_rows, key=lambda row: (row["gap_from_leader_percent"], row["decoder"]))
                if row["recommended"]
            ]
            speed_shortlist = [
                row["decoder"]
                for row in sorted(decoder_rows, key=lambda row: (row["gap_from_leader_percent"], row["decoder"]))
                if row["within_speed_margin"]
            ]
            cells.append(
                {
                    "decoders": decoder_rows,
                    "leader": leader.decoder,
                    "leader_images_per_second": leader.mean,
                    "leader_workers": leader.workers,
                    "machine_type": machine_type,
                    "platform": PLATFORM_LABELS[machine_type],
                    "recommended": recommended,
                    "speed_shortlist": speed_shortlist,
                    "workload": workload,
                },
            )

    worst_gap_percent = {decoder: max(gaps) for decoder, gaps in gaps_by_decoder.items()}
    robust_decoders = sorted(
        decoder for decoder, successes in audit_successes.items() if successes == ROBUSTNESS_AUDIT_ITEM_COUNT
    )
    if not robust_decoders:
        raise PaperAssetError("no decoder passes the robustness audit")
    portable_decoder = min(
        robust_decoders,
        key=lambda decoder: (worst_gap_percent[decoder], decoder),
    )
    universal_recommendations = sorted(
        decoder for decoder in robust_decoders if worst_gap_percent[decoder] <= PRACTICAL_MARGIN * 100
    )
    portable_speed_candidates = sorted(
        (decoder for decoder, gap in worst_gap_percent.items() if gap <= PRACTICAL_MARGIN * 100),
        key=lambda decoder: (worst_gap_percent[decoder], decoder),
    )
    robustness_audit = {
        "item_count": ROBUSTNESS_AUDIT_ITEM_COUNT,
        "linux_receipts": [
            {
                "job": "imread-20260802-180331-9cbbc8f2",
                "plan_id": "2cab69bcdb92b33c41f8d0ec225ccb91e3e314f792caca517e7280a667578250",
                "result": (
                    "ajpegli support audit completed with 0/277 before the empty support set stopped the campaign"
                ),
            },
            {
                "committed_bundles": 11,
                "job": "imread-20260802-181101-9cbbc8f2",
                "plan_id": "98e8f88e2820b1a0992d0a72cb299d95d52fb2d0d700884567eac54d23282e2e",
                "result": "completed",
            },
        ],
        "manifest_id": ROBUSTNESS_AUDIT_MANIFEST_ID,
        "output_contract": "normalized-rgb",
        "package_id": ROBUSTNESS_AUDIT_PACKAGE_ID,
        "platform": {
            "machine_type": "c3-standard-4",
            "operating_system": "Ubuntu 24.04",
            "zone": "us-central1-a",
        },
        "process_context": "main-process support audit",
        "runner_revision": "9cbbc8f212ad9a384fbabf1dde582023077e83dcda0ac285baf0c16351febb8f",
        "selection": "276 FODB empty-DHT exclusions plus one four-component RGB-contract sentinel",
        "successes": dict(sorted(audit_successes.items())),
    }
    if uses_recorded_audit:
        robustness_audit["categories"] = {
            "empty_dht_bitstream": {
                "description": (
                    "Progressive FODB WhatsApp JPEGs containing empty DHT markers; success measures decoder "
                    "recovery from this malformed bitstream pattern."
                ),
                "item_count": ROBUSTNESS_AUDIT_EMPTY_DHT_ITEM_COUNT,
                "successes": dict(sorted(ROBUSTNESS_AUDIT_EMPTY_DHT_SUCCESSES.items())),
            },
            "four_component_rgb": {
                "description": (
                    "A four-component JPEG; success requires conversion to the normalized three-channel RGB contract."
                ),
                "filename": "ILSVRC2012_val_00019877.JPEG",
                "item_count": ROBUSTNESS_AUDIT_FOUR_COMPONENT_ITEM_COUNT,
                "sha256": "75413aece0dc58bcd9d4b89f664ab04cee3ade28317d81aeace455029e0000ba",
                "successes": dict(sorted(ROBUSTNESS_AUDIT_FOUR_COMPONENT_SUCCESSES.items())),
            },
        }
    return {
        "cells": cells,
        "coverage_qualified_decoders": robust_decoders,
        "robustness_audit": robustness_audit,
        "portable_decoder": portable_decoder,
        "portable_max_gap_percent": worst_gap_percent[portable_decoder],
        "portable_speed_candidates": portable_speed_candidates,
        "universal_recommendations": universal_recommendations,
        "worst_gap_percent": dict(sorted(worst_gap_percent.items())),
    }


def _pillow_rows(aggregates: tuple[Aggregate, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in EXPECTED_ITEM_COUNTS:
        for machine_type in EXPECTED_MACHINES:
            peaks: dict[str, Aggregate] = {}
            for decoder in MIGRATION_DECODERS:
                candidates = tuple(
                    row
                    for row in aggregates
                    if row.workload == workload
                    and row.machine_type == machine_type
                    and row.protocol == "loader-supply"
                    and row.decoder == decoder
                    and row.workers in DEPLOYMENT_WORKERS
                    and _is_controlled(row)
                )
                peaks[decoder] = _best(candidates)
            pillow = peaks["pillow"]
            rows.append(
                {
                    "gains_percent": {
                        decoder: (peaks[decoder].mean / pillow.mean - 1) * 100
                        for decoder in MIGRATION_DECODERS
                        if decoder != "pillow"
                    },
                    "machine_type": machine_type,
                    "peak_workers": {decoder: peak.workers for decoder, peak in peaks.items()},
                    "platform": PLATFORM_LABELS[machine_type],
                    "workload": workload,
                },
            )
    return rows


def _thread_rows(aggregates: tuple[Aggregate, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in EXPECTED_ITEM_COUNTS:
        for machine_type in EXPECTED_MACHINES:
            for decoder in sorted(CONTROLLED_THREADS):
                decode_default = _only(aggregates, (workload, machine_type, "decode-memory", decoder, None, None))
                decode_one = _only(aggregates, (workload, machine_type, "decode-memory", decoder, 1, None))
                loader_default = _best(
                    tuple(
                        row
                        for row in aggregates
                        if row.workload == workload
                        and row.machine_type == machine_type
                        and row.protocol == "loader-supply"
                        and row.decoder == decoder
                        and row.requested_threads is None
                        and row.workers in BASE_WORKERS
                    ),
                )
                loader_one = _best(
                    tuple(
                        row
                        for row in aggregates
                        if row.workload == workload
                        and row.machine_type == machine_type
                        and row.protocol == "loader-supply"
                        and row.decoder == decoder
                        and row.requested_threads == 1
                        and row.workers in BASE_WORKERS
                    ),
                )
                rows.append(
                    {
                        "decode_default_vs_one_percent": (decode_default.mean / decode_one.mean - 1) * 100,
                        "decoder": decoder,
                        "loader_default_vs_one_percent": (loader_default.mean / loader_one.mean - 1) * 100,
                        "machine_type": machine_type,
                        "platform": PLATFORM_LABELS[machine_type],
                        "workload": workload,
                    },
                )
    return rows


def _worker_transfer_rows(aggregates: tuple[Aggregate, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for machine_type in EXPECTED_MACHINES:
        peaks: dict[str, dict[str, Aggregate]] = {}
        for workload in EXPECTED_ITEM_COUNTS:
            peaks[workload] = {
                decoder: _best(
                    tuple(
                        row
                        for row in aggregates
                        if row.workload == workload
                        and row.machine_type == machine_type
                        and row.protocol == "loader-supply"
                        and row.decoder == decoder
                        and _is_controlled(row)
                    ),
                )
                for decoder in sorted({row.decoder for row in aggregates if _is_controlled(row)})
            }
        changed = [
            decoder
            for decoder in sorted(peaks["fodb-native"])
            if peaks["fodb-native"][decoder].workers != peaks["fodb-mixed"][decoder].workers
        ]
        rows.append(
            {
                "changed_decoders": changed,
                "machine_type": machine_type,
                "mixed_peak_worker_counts": dict(
                    sorted(Counter(row.workers for row in peaks["fodb-mixed"].values()).items()),
                ),
                "native_peak_worker_counts": dict(
                    sorted(Counter(row.workers for row in peaks["fodb-native"].values()).items()),
                ),
                "platform": PLATFORM_LABELS[machine_type],
            },
        )
    return rows


def _worker16_candidate_rows(
    aggregates: tuple[Aggregate, ...],
    worker12_decoders: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    selection = WORKER12_DECODERS if worker12_decoders is None else worker12_decoders
    cells: list[dict[str, Any]] = []
    for (workload, machine_type), decoders in sorted(selection.items()):
        selected: list[dict[str, Any]] = []
        for decoder in decoders:
            curve = tuple(
                row
                for row in aggregates
                if row.workload == workload
                and row.machine_type == machine_type
                and row.protocol == "loader-supply"
                and row.decoder == decoder
                and row.requested_threads == CONTROLLED_THREADS.get(decoder)
                and row.workers in WORKER12_DECISION_GRID
            )
            expected_workers = set(WORKER12_DECISION_GRID)
            if len(curve) != len(expected_workers) or {row.workers for row in curve} != expected_workers:
                raise PaperAssetError(
                    f"workers=12 follow-up curve is incomplete for {workload}/{machine_type}/{decoder}",
                )
            worker12 = _only(
                aggregates,
                (workload, machine_type, "loader-supply", decoder, CONTROLLED_THREADS.get(decoder), 12),
            )
            worker8 = _only(
                aggregates,
                (workload, machine_type, "loader-supply", decoder, CONTROLLED_THREADS.get(decoder), 8),
            )
            paired_gain_ratios = tuple(
                worker12_value / worker8_value
                for worker12_value, worker8_value in zip(
                    worker12.raw_run_means,
                    worker8.raw_run_means,
                    strict=True,
                )
            )
            gain_from_workers_8 = worker12.mean / worker8.mean - 1
            if gain_from_workers_8 >= WORKER16_MINIMUM_GAIN and min(paired_gain_ratios) >= 1:
                selected.append(
                    {
                        "decoder": decoder,
                        "gain_from_workers_8_percent": gain_from_workers_8 * 100,
                        "minimum_paired_gain_ratio": min(paired_gain_ratios),
                        "workers_12_images_per_second": worker12.mean,
                    },
                )
        cells.append(
            {
                "decoders": selected,
                "machine_type": machine_type,
                "platform": PLATFORM_LABELS[machine_type],
                "workload": workload,
            },
        )
    candidate_count = sum(len(cell["decoders"]) for cell in cells)
    return {
        "bundle_count_if_launched": candidate_count * len(EXPECTED_REPETITIONS),
        "candidate_count": candidate_count,
        "cells": cells,
        "criterion": (
            "workers=12 is at least 5% faster than workers=8 by the five-block mean "
            "and every paired workers=12/workers=8 block has a ratio of at least 1"
        ),
        "minimum_mean_gain_percent": WORKER16_MINIMUM_GAIN * 100,
        "repetitions_if_launched": len(EXPECTED_REPETITIONS),
    }


def _evidence_document(
    records: tuple[RunBundleRecord, ...],
    measurements: tuple[Measurement, ...],
    aggregates: tuple[Aggregate, ...],
    sections: dict[str, Any],
) -> dict[str, Any]:
    run_keys = sorted(measurement.run_key for measurement in measurements)
    bundle_ids = sorted(measurement.bundle_id for measurement in measurements)
    return {
        "aggregates": [_aggregate_dict(row) for row in aggregates],
        "coverage": {
            "committed_evidence_bundles": len(records),
            "expected_repetitions_per_configuration": len(EXPECTED_REPETITIONS),
            "manifest_items": EXPECTED_ITEM_COUNTS,
            "timed_common_support_items": EXPECTED_TIMED_ITEM_COUNTS,
            "timed_failure_bundles": sum(bool(record.failures) for record in records),
        },
        "decisions": sections["decisions"],
        "matrix": {
            "machine_types": list(EXPECTED_MACHINES),
            "plan_ids": {
                workload: sorted(
                    plan_id for plan_id, plan_workload in PLAN_WORKLOADS.items() if plan_workload == workload
                )
                for workload in EXPECTED_ITEM_COUNTS
            },
            "practical_margin_percent": PRACTICAL_MARGIN * 100,
            "deployment_worker_grid": list(DEPLOYMENT_WORKERS),
            "runner_revision": RUNNER_REVISION,
            "broad_worker_grid": list(BASE_WORKERS),
            "worker_grid": list(WORKER_GRID),
            "workers_12_decoders": {
                f"{workload}/{machine_type}": list(decoders)
                for (workload, machine_type), decoders in sorted(WORKER12_DECODERS.items())
            },
            "workers_16_decoders": {
                f"{workload}/{machine_type}": list(decoders)
                for (workload, machine_type), decoders in sorted(WORKER16_DECODERS.items())
                if decoders
            },
        },
        "pillow_migration": sections["pillow_migration"],
        "recommendations": sections["recommendations"],
        "provenance": {
            "bundle_ids_sha256": _sequence_digest(bundle_ids),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run_keys": run_keys,
            "run_keys_sha256": _sequence_digest(run_keys),
        },
        "schema_version": "1.6",
        "thread_controls": sections["thread_controls"],
        "worker_transfer": sections["worker_transfer"],
        "worker16_candidates": sections["worker16_candidates"],
        "workloads": sections["workloads"],
    }


def _aggregate_dict(row: Aggregate) -> dict[str, Any]:
    return {
        "decoder": row.decoder,
        "machine_type": row.machine_type,
        "mean_images_per_second": row.mean,
        "protocol": row.protocol,
        "raw_run_means": list(row.raw_run_means),
        "repetitions": list(row.repetitions),
        "requested_threads": row.requested_threads,
        "sample_std_images_per_second": row.sample_std,
        "workers": row.workers,
        "workload": row.workload,
    }


def _load_package_manifests(package_path: Path, package: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    archive = _required_object(package, "archive")
    archive_path = package_path.parent / _required_string(archive, "file")
    manifests: dict[str, list[dict[str, Any]]] = {}
    try:
        with tarfile.open(archive_path) as tar:
            for workload in EXPECTED_ITEM_COUNTS:
                source = tar.extractfile(f"manifests/{workload}.json")
                if source is None:
                    raise PaperAssetError(f"package archive has no manifest for {workload}")
                document = json.load(source)
                if not isinstance(document, dict):
                    raise PaperAssetError(f"manifest for {workload} must be an object")
                raw_items = _required_list(document, "items")
                if not all(isinstance(item, dict) for item in raw_items):
                    raise PaperAssetError(f"manifest for {workload} contains a non-object item")
                manifests[workload] = raw_items
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise PaperAssetError(f"cannot read package manifests from {archive_path}: {exc}") from exc
    return manifests


def _support_item_ids(records: tuple[RunBundleRecord, ...]) -> dict[str, set[str]]:
    populations: dict[str, set[str]] = {}
    for record in records:
        workload = _required_string(record.dataset, "workload_id")
        raw_items = record.dataset.get("ordered_item_ids")
        if not isinstance(raw_items, list) or not all(isinstance(item, str) for item in raw_items):
            raise PaperAssetError(f"bundle {record.run_key} has invalid support item IDs")
        item_ids = set(raw_items)
        observed = populations.setdefault(workload, item_ids)
        if observed != item_ids:
            raise PaperAssetError(f"timed support population differs across {workload} bundles")
    if set(populations) != set(EXPECTED_ITEM_COUNTS):
        raise PaperAssetError("timed support populations do not cover both FODB workloads")
    return populations


def _workload_descriptors(
    package: dict[str, Any],
    manifests: dict[str, list[dict[str, Any]]],
    support_item_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    package_items = _required_list(_required_object(package, "provenance"), "items")
    by_sha = {_required_string(item, "jpeg_sha256"): item for item in package_items if isinstance(item, dict)}
    rows: list[dict[str, Any]] = []
    for workload in EXPECTED_ITEM_COUNTS:
        manifest = manifests[workload]
        timed_manifest = [item for item in manifest if _required_string(item, "item_id") in support_item_ids[workload]]
        excluded_manifest = [
            item for item in manifest if _required_string(item, "item_id") not in support_item_ids[workload]
        ]
        selected = [by_sha[_required_string(item, "sha256")] for item in timed_manifest]
        excluded = [by_sha[_required_string(item, "sha256")] for item in excluded_manifest]
        megapixels = [_jpeg_number(item, "megapixels") for item in selected]
        compressed_bytes = [_required_int(item, "jpeg_bytes") for item in selected]
        bits_per_pixel = [_required_number(item, "bits_per_pixel") for item in selected]
        quality = [_jpeg_number(item, "quality_estimate") for item in selected]
        progressive = sum(bool(_required_object(item, "jpeg").get("progressive")) for item in selected)
        provenance = Counter(_required_string(item, "provenance") for item in selected)
        subsampling = Counter(_required_string(_required_object(item, "jpeg"), "subsampling") for item in selected)
        rows.append(
            {
                "bits_per_pixel": _distribution(bits_per_pixel),
                "compressed_bytes": _distribution(compressed_bytes),
                "estimated_quality": _distribution(quality),
                "excluded_items": len(excluded),
                "excluded_profile": _excluded_profile(excluded),
                "items": len(selected),
                "manifest_items": len(manifest),
                "megapixels": _distribution(megapixels),
                "progressive_items": progressive,
                "provenance": dict(sorted(provenance.items())),
                "subsampling": dict(sorted(subsampling.items())),
                "total_compressed_bytes": sum(compressed_bytes),
                "workload": workload,
            },
        )
    return rows


def _excluded_profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"estimated_quality": {}, "progressive_items": 0, "provenance": {}}
    return {
        "estimated_quality": dict(sorted(Counter(_jpeg_number(item, "quality_estimate") for item in items).items())),
        "progressive_items": sum(bool(_required_object(item, "jpeg").get("progressive")) for item in items),
        "provenance": dict(sorted(Counter(_required_string(item, "provenance") for item in items).items())),
    }


def _distribution(values: list[float] | list[int]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    return {
        "maximum": ordered[-1],
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "q10": _linear_quantile(ordered, 0.10),
        "q90": _linear_quantile(ordered, 0.90),
    }


def _linear_quantile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _best(rows: tuple[Aggregate, ...]) -> Aggregate:
    if not rows:
        raise PaperAssetError("cannot select a best configuration from an empty set")
    return min(rows, key=lambda row: (-row.mean, row.configuration_label, row.workers or -1))


def _decoders(rows: tuple[Aggregate, ...]) -> tuple[str, ...]:
    return tuple(sorted({row.decoder for row in rows}))


def _ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for key, _ in ordered[start:end]:
            result[key] = rank
        start = end
    return result


AggregateKey = tuple[str, str, str, str, int | None, int | None]


def _only(rows: tuple[Aggregate, ...], key: AggregateKey) -> Aggregate:
    selected = tuple(
        row
        for row in rows
        if (
            row.workload,
            row.machine_type,
            row.protocol,
            row.decoder,
            row.requested_threads,
            row.workers,
        )
        == key
    )
    if len(selected) != 1:
        raise PaperAssetError(f"expected one aggregate, found {len(selected)}")
    return selected[0]


def _summary_markdown(evidence: dict[str, Any]) -> str:
    robustness_audit = evidence["recommendations"]["robustness_audit"]
    empty_dht = robustness_audit["categories"]["empty_dht_bitstream"]
    four_component = robustness_audit["categories"]["four_component_rgb"]
    universal_recommendations = evidence["recommendations"]["universal_recommendations"]
    lines = [
        "# Generated FODB evidence",
        "",
        "All values below are generated from the exact evidence plan IDs in `fodb_evidence.json`.",
        "Recommendation analysis first keeps decoders within 10% of the local loader leader, "
        "then applies separate empty-DHT bitstream and four-component RGB robustness tests.",
        "The 10% margin is a reporting policy, not a hypothesis test.",
        "",
        "## Workloads",
        "",
        "| Workload | Timed / manifest | Median MP | Median compressed MiB | Median estimated quality | Progressive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            f"| {row['workload']} | {row['items']} / {row['manifest_items']} | "
            f"{row['megapixels']['median']:.3f} | "
            f"{row['compressed_bytes']['median'] / 2**20:.3f} | {row['estimated_quality']['median']:.0f} | "
            f"{row['progressive_items']} |"
        )
        for row in evidence["workloads"]
    )
    lines.extend(
        [
            "",
            "## Deployment recommendation",
            "",
            (
                f"Universal recommendation: {', '.join(f'`{decoder}`' for decoder in universal_recommendations)}."
                if universal_recommendations
                else (
                    "No decoder both passes the robustness audit and remains within 10% "
                    "of the leader in all eight cells."
                )
            ),
            f" Robust minimax candidate (not a universal 10% recommendation): "
            f"`{evidence['recommendations']['portable_decoder']}`; maximum aggregate gap from the local loader "
            f"leader: {evidence['recommendations']['portable_max_gap_percent']:.2f}%.",
            f" Portable speed shortlist before the robustness gate: "
            f"{', '.join(f'`{decoder}`' for decoder in evidence['recommendations']['portable_speed_candidates'])}.",
            "",
            "| Platform | Workload | Within 10% | Passes robustness gate | Highest observed mean |",
            "| --- | --- | --- | --- | --- |",
        ],
    )
    lines.extend(
        (
            f"| {row['platform']} | {row['workload']} | {', '.join(row['speed_shortlist'])} | "
            f"{', '.join(row['recommended'])} | {row['leader']} |"
        )
        for row in evidence["recommendations"]["cells"]
    )
    lines.extend(
        [
            "",
            "## Robustness audit",
            "",
            "The empty-DHT test measures recovery from one malformed bitstream pattern. "
            "The four-component test measures conversion to the normalized three-channel RGB contract; "
            "the sentinel is not classified as corrupt.",
            "",
            "| Decoder | Empty-DHT bitstreams | Four-component RGB | Combined |",
            "| --- | ---: | ---: | ---: |",
        ],
    )
    lines.extend(
        (
            f"| `{decoder}` | {empty_dht['successes'][decoder]}/{empty_dht['item_count']} | "
            f"{four_component['successes'][decoder]}/{four_component['item_count']} | "
            f"{robustness_audit['successes'][decoder]}/{robustness_audit['item_count']} |"
        )
        for decoder in sorted(robustness_audit["successes"])
    )
    lines.extend(
        [
            "",
            "## Protocol decision",
            "",
            "| Workload | Platform | Decode leader | Loader leader | Workers | Regret | "
            "Spearman rho | 10% speed tier before robustness gate |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ],
    )
    for row in evidence["decisions"]:
        tier = ", ".join(item["decoder"] for item in row["top_tier"])
        lines.append(
            f"| {row['workload']} | {row['platform']} | {row['decode_leader']} | {row['loader_leader']} | "
            f"{row['loader_leader_workers']} | {row['aggregate_regret_percent']:.2f}% | "
            f"{row['rank_correlation']:.2f} | {tier} |",
        )
    lines.extend(
        [
            "",
            "## Pillow migration",
            "",
            "| Workload | Platform | OpenCV | simplejpeg |",
            "| --- | --- | ---: | ---: |",
        ],
    )
    for row in evidence["pillow_migration"]:
        gains = row["gains_percent"]
        lines.append(
            f"| {row['workload']} | {row['platform']} | {gains['opencv']:+.1f}% | {gains['simplejpeg']:+.1f}% |",
        )
    lines.extend(
        [
            "",
            "## Peak-worker transfer",
            "",
            "| Platform | Native peak-worker counts | Mixed peak-worker counts | Changed decoders |",
            "| --- | --- | --- | --- |",
        ],
    )
    for row in evidence["worker_transfer"]:
        native = ", ".join(f"w={key}: {value}" for key, value in row["native_peak_worker_counts"].items())
        mixed = ", ".join(f"w={key}: {value}" for key, value in row["mixed_peak_worker_counts"].items())
        lines.append(f"| {row['platform']} | {native} | {mixed} | {', '.join(row['changed_decoders']) or 'none'} |")
    return "\n".join(lines) + "\n"


def _workload_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        line = (
            f"{_latex_workload(row['workload'])} & {row['items']:,}/{row['manifest_items']:,} & "
            f"{row['megapixels']['median']:.2f} [{row['megapixels']['q10']:.2f}, {row['megapixels']['q90']:.2f}] & "
            f"{row['compressed_bytes']['median'] / 2**20:.2f} & {row['bits_per_pixel']['median']:.2f} & "
            f"{row['estimated_quality']['median']:.0f} & {row['progressive_items']:,} "
        )
        body.append(line)
    return (r"\\" + "\n").join(body) + "\n"


def _decision_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        line = (
            f"{_latex_workload(row['workload'])} & {row['platform']} & \\texttt{{{row['decode_leader']}}} & "
            f"\\texttt{{{row['loader_leader']}}} ($w={row['loader_leader_workers']}$) & "
            f"{row['aggregate_regret_percent']:.1f}\\% & {row['rank_correlation']:.2f} "
        )
        body.append(line)
    return (r"\\" + "\n").join(body) + "\n"


def _recommendation_table(recommendations: dict[str, Any]) -> str:
    portable_decoder = recommendations["portable_decoder"]
    universal = set(recommendations["universal_recommendations"])
    portable_speed = set(recommendations["portable_speed_candidates"])
    successes = recommendations["robustness_audit"]["successes"]
    worst_gaps = recommendations["worst_gap_percent"]

    def sort_key(decoder: str) -> tuple[int, float, str]:
        if decoder == portable_decoder:
            group = 0
        elif decoder in portable_speed:
            group = 1
        else:
            group = 2
        return group, worst_gaps[decoder], decoder

    body = []
    shown = portable_speed | {decoder for decoder, count in successes.items() if count == ROBUSTNESS_AUDIT_ITEM_COUNT}
    for decoder in sorted(shown, key=sort_key):
        audited_successes = successes[decoder]
        audit = f"{audited_successes}/{ROBUSTNESS_AUDIT_ITEM_COUNT}"
        if decoder == portable_decoder:
            decision = "portable default"
        elif decoder in universal:
            decision = "portable alternative"
        elif decoder in portable_speed:
            decision = "audit target corpus"
        else:
            decision = "cell-specific"
        body.append(
            f"\\texttt{{{decoder}}} & {worst_gaps[decoder]:.1f}\\% & {audit} & {decision} ",
        )
    return (r"\\" + "\n").join(body) + "\n"


def _decoder_coverage_table(recommendations: dict[str, Any]) -> str:
    audit = recommendations["robustness_audit"]
    categories = audit.get("categories")
    if not isinstance(categories, dict):
        raise PaperAssetError("decoder coverage table requires the recorded two-category robustness audit")
    empty_dht = categories["empty_dht_bitstream"]
    four_component = categories["four_component_rgb"]
    empty_dht_successes = empty_dht["successes"]
    four_component_successes = four_component["successes"]
    combined_successes = audit["successes"]
    body = []
    for decoder in sorted(combined_successes):
        bitstream_result = f"{empty_dht_successes[decoder]}/{empty_dht['item_count']}"
        four_component_result = f"{four_component_successes[decoder]}/{four_component['item_count']}"
        combined_result = f"{combined_successes[decoder]}/{audit['item_count']}"
        body.append(
            f"\\texttt{{{decoder}}} & {bitstream_result} & {four_component_result} & {combined_result} ",
        )
    return (r"\\" + "\n").join(body) + "\n"


def _provenance_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        counts = row["provenance"]
        body.append(
            f"{_latex_workload(row['workload'])} & {counts.get('orig', 0):,} & "
            f"{counts.get('facebook', 0):,} & {counts.get('instagram', 0):,} & "
            f"{counts.get('telegram', 0):,} & {counts.get('twitter', 0):,} & "
            f"{counts.get('whatsapp', 0):,} ",
        )
    return (r"\\" + "\n").join(body) + "\n"


def _pillow_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        gains = row["gains_percent"]
        line = (
            f"{_latex_workload(row['workload'])} & {row['platform']} & {gains['opencv']:+.1f}\\% & "
            f"{gains['simplejpeg']:+.1f}\\% "
        )
        body.append(line)
    return (r"\\" + "\n").join(body) + "\n"


def _worker_transfer_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        native = row["native_peak_worker_counts"]
        mixed = row["mixed_peak_worker_counts"]
        changed = ", ".join(f"\\texttt{{{decoder}}}" for decoder in row["changed_decoders"]) or "none"
        body.append(
            f"{row['platform']} & {native.get(0, 0)} / {native.get(8, 0)} & "
            f"{mixed.get(0, 0)} / {mixed.get(8, 0)} & {len(row['changed_decoders'])}/12 & {changed} ",
        )
    return (r"\\" + "\n").join(body) + "\n"


def _coverage_table(evidence: dict[str, Any]) -> str:
    coverage = evidence["coverage"]
    native = (
        f"FODB-native & {coverage['timed_common_support_items']['fodb-native']:,} / "
        f"{coverage['manifest_items']['fodb-native']:,} "
        "& 0 & 1,500 "
    )
    mixed = (
        f"FODB-mixed & {coverage['timed_common_support_items']['fodb-mixed']:,} / "
        f"{coverage['manifest_items']['fodb-mixed']:,} "
        "& 0 & 1,500 "
    )
    return native + r"\\" + "\n" + mixed + "\n"


def _versions_table(records: tuple[RunBundleRecord, ...]) -> str:
    wanted = {
        "ajpegli",
        "imagecodecs",
        "imageio",
        "jpeg4py",
        "kornia-rs",
        "opencv-python-headless",
        "pillow",
        "pyvips",
        "scikit-image",
        "simplejpeg",
        "torch",
        "torchvision",
        "pyturbojpeg",
    }
    versions: dict[str, set[str]] = defaultdict(set)
    for record in records:
        distributions = record.environment.get("distributions")
        if not isinstance(distributions, list):
            raise PaperAssetError("environment distributions are missing")
        for distribution in distributions:
            if isinstance(distribution, dict) and distribution.get("name") in wanted:
                versions[str(distribution["name"])].add(str(distribution.get("version")))
    if set(versions) != wanted or any(len(values) != 1 for values in versions.values()):
        raise PaperAssetError("decoder package versions differ across evidence environments")
    return (r"\\" + "\n").join(
        f"\\texttt{{{name}}} & \\texttt{{{next(iter(versions[name]))}}} " for name in sorted(versions)
    ) + "\n"


def _plot_workloads(
    package: dict[str, Any],
    manifests: dict[str, list[dict[str, Any]]],
    support_item_ids: dict[str, set[str]],
    destination: Path,
) -> None:
    plt = _pyplot()
    package_items = _required_list(_required_object(package, "provenance"), "items")
    by_sha = {_required_string(item, "jpeg_sha256"): item for item in package_items if isinstance(item, dict)}
    data: dict[str, list[dict[str, Any]]] = {}
    for workload, manifest in manifests.items():
        selected = [item for item in manifest if _required_string(item, "item_id") in support_item_ids[workload]]
        data[workload.replace("fodb-", "FODB-")] = [by_sha[_required_string(item, "sha256")] for item in selected]
    extractors = (
        ("Megapixels", lambda item: _jpeg_number(item, "megapixels"), True),
        ("Compressed size (MiB)", lambda item: _required_int(item, "jpeg_bytes") / 2**20, True),
        ("Bits per pixel", lambda item: _required_number(item, "bits_per_pixel"), False),
        ("Estimated quality", lambda item: _jpeg_number(item, "quality_estimate"), False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)
    for axis, (label, extractor, logarithmic) in zip(axes.flat, extractors, strict=True):
        for workload, workload_items in data.items():
            values = sorted(extractor(item) for item in workload_items)
            probabilities = [(index + 1) / len(values) for index in range(len(values))]
            axis.step(values, probabilities, where="post", label=workload)
        if logarithmic:
            axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("ECDF")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.savefig(destination)
    plt.close(figure)


def _plot_worker_scaling(aggregates: tuple[Aggregate, ...], destination: Path) -> None:
    plt = _pyplot()
    colors = dict(zip(PRIMARY_DECODERS, ("#4C78A8", "#F58518", "#54A24B", "#B279A2"), strict=True))
    figure, axes = plt.subplots(4, 2, figsize=(7.1, 8.4), sharex=True, constrained_layout=True)
    for row_index, machine_type in enumerate(EXPECTED_MACHINES):
        for column_index, workload in enumerate(EXPECTED_ITEM_COUNTS):
            axis = axes[row_index, column_index]
            for decoder_index, decoder in enumerate(PRIMARY_DECODERS):
                curve = sorted(
                    (
                        point
                        for point in aggregates
                        if point.workload == workload
                        and point.machine_type == machine_type
                        and point.protocol == "loader-supply"
                        and point.decoder == decoder
                        and point.requested_threads == CONTROLLED_THREADS.get(decoder)
                        and point.workers is not None
                    ),
                    key=lambda point: point.workers if point.workers is not None else -1,
                )
                expected_workers = set(BASE_WORKERS)
                if decoder in WORKER12_DECODERS[(workload, machine_type)]:
                    expected_workers.add(12)
                if decoder in WORKER16_DECODERS[(workload, machine_type)]:
                    expected_workers.add(16)
                curve_workers = [point.workers for point in curve if point.workers is not None]
                if set(curve_workers) != expected_workers or len(curve_workers) != len(expected_workers):
                    raise PaperAssetError(f"worker curve is incomplete for {workload}/{machine_type}/{decoder}")
                paired_ratios = [
                    tuple(
                        value / baseline
                        for value, baseline in zip(point.raw_run_means, curve[0].raw_run_means, strict=True)
                    )
                    for point in curve
                ]
                means = [statistics.fmean(values) for values in paired_ratios]
                axis.plot(curve_workers, means, marker="o", color=colors[decoder], label=decoder)
                jitter = (decoder_index - 1.5) * 0.06
                for workers, values in zip(curve_workers, paired_ratios, strict=True):
                    axis.scatter(
                        [workers + jitter] * len(values),
                        values,
                        color=colors[decoder],
                        alpha=0.28,
                        s=9,
                        linewidths=0,
                    )
            axis.axhline(1, color="0.45", linewidth=0.8, linestyle="--")
            axis.grid(alpha=0.18)
            axis.set_title(f"{PLATFORM_LABELS[machine_type]} — {workload}", fontsize=9)
            if column_index == 0:
                axis.set_ylabel("Throughput / paired $w=0$")
            if row_index == len(EXPECTED_MACHINES) - 1:
                axis.set_xlabel("DataLoader workers")
            axis.set_xticks(WORKER_GRID)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(destination)
    plt.close(figure)


def _plot_protocol_regret(decisions: list[dict[str, Any]], destination: Path) -> None:
    plt = _pyplot()
    labels = [f"{row['platform']}\n{row['workload'].removeprefix('fodb-')}" for row in decisions]
    paired = [row["paired_regret_percent"] for row in decisions]
    means = [statistics.fmean(values) for values in paired]
    colors = ["#4C78A8" if row["strict_leader_match"] else "#E45756" for row in decisions]
    figure, axis = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    positions = list(range(len(decisions)))
    axis.bar(positions, means, color=colors, alpha=0.82)
    for position, values in zip(positions, paired, strict=True):
        offsets = (-0.12, -0.06, 0, 0.06, 0.12)
        axis.scatter([position + offset for offset in offsets], values, color="black", s=12, alpha=0.65, zorder=3)
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.axhline(PRACTICAL_MARGIN * 100, color="0.3", linewidth=0.9, linestyle="--", label="10% margin")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Loader regret after worker tuning (%)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(destination)
    plt.close(figure)


def _plot_recommendations(recommendations: dict[str, Any], destination: Path) -> None:
    plt = _pyplot()
    import numpy as np
    from matplotlib.patches import Rectangle

    cells = recommendations["cells"]
    cell_by_scenario = {(row["machine_type"], row["workload"]): row for row in cells}
    scenarios = [(machine_type, workload) for machine_type in EXPECTED_MACHINES for workload in EXPECTED_ITEM_COUNTS]
    successes = recommendations["robustness_audit"]["successes"]
    portable_speed = set(recommendations["portable_speed_candidates"])

    def sort_key(decoder: str) -> tuple[int, float, str]:
        if decoder in portable_speed:
            group = 0
        elif successes[decoder] == ROBUSTNESS_AUDIT_ITEM_COUNT:
            group = 1
        else:
            group = 2
        return group, recommendations["worst_gap_percent"][decoder], decoder

    decoders = sorted(recommendations["worst_gap_percent"], key=sort_key)
    gaps = np.array(
        [
            [
                next(
                    item["gap_from_leader_percent"]
                    for item in cell_by_scenario[scenario]["decoders"]
                    if item["decoder"] == decoder
                )
                for scenario in scenarios
            ]
            for decoder in decoders
        ],
    )

    figure, axis = plt.subplots(figsize=(7.1, 5.4), constrained_layout=True)
    image = axis.imshow(gaps, cmap="Blues_r", vmin=0, vmax=max(35, float(gaps.max())), aspect="auto")
    for row_index, _decoder in enumerate(decoders):
        for column_index, _scenario in enumerate(scenarios):
            gap = gaps[row_index, column_index]
            decoder_row = next(item for item in cell_by_scenario[_scenario]["decoders"] if item["decoder"] == _decoder)
            if decoder_row["within_speed_margin"]:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.49, row_index - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.2,
                    ),
                )
            text_color = "white" if gap < 14 else "black"
            axis.text(
                column_index,
                row_index,
                f"{gap:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )

    labels = [
        f"{PLATFORM_LABELS[machine_type]}\n{workload.removeprefix('fodb-')}" for machine_type, workload in scenarios
    ]
    axis.set_xticks(range(len(scenarios)), labels, rotation=32, ha="right")
    axis.set_yticks(range(len(decoders)), decoders)
    for tick, decoder in zip(axis.get_yticklabels(), decoders, strict=True):
        if successes[decoder] == ROBUSTNESS_AUDIT_ITEM_COUNT:
            tick.set_fontweight("bold")
    axis.tick_params(axis="both", length=0, labelsize=8)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Below local best mean loader supply (%)")
    axis.set_xlabel("16-vCPU platform and workload")
    figure.savefig(destination)
    plt.close(figure)


def _pyplot() -> Any:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PaperAssetError("paper figures require `uv run --extra plot`") from exc
    return plt


def _latex_workload(workload: str) -> str:
    return workload.replace("fodb-", "FODB-")


def _latex_decoder_list(value: str) -> str:
    return ", ".join(f"\\texttt{{{item.strip()}}}" for item in value.split(","))


def _sequence_digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _jpeg_number(item: object, key: str) -> float:
    if not isinstance(item, dict):
        raise PaperAssetError("package item must be an object")
    return _required_number(_required_object(item, "jpeg"), key)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperAssetError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperAssetError(f"{path} must contain a JSON object")
    return value


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PaperAssetError(f"field {key!r} must be an object")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise PaperAssetError(f"field {key!r} must be a list")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PaperAssetError(f"field {key!r} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PaperAssetError(f"field {key!r} must be an integer")
    return value


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PaperAssetError(f"field {key!r} must be an integer or null")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperAssetError(f"field {key!r} must be numeric")
    return float(value)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate claim-scoped FODB paper evidence and vector figures.")
    parser.add_argument("--artifacts", type=Path, required=True, help="Hydrated schema-2 artifact root.")
    parser.add_argument("--package", type=Path, required=True, help="FODB package.json used by the campaigns.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Paper directory containing generated/ and figures/.",
    )
    args = parser.parse_args()
    evidence = build_paper_assets(artifact_root=args.artifacts, package_path=args.package, output_root=args.output)
    print(json.dumps({"bundles": evidence["coverage"]["committed_evidence_bundles"], "status": "generated"}))


if __name__ == "__main__":
    main()

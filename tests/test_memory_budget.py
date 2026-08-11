from __future__ import annotations

from imread_benchmark.execution.memory import estimate_peak_memory
from imread_benchmark.plans import RunConfiguration


def test_spawn_memory_estimate_counts_resident_replicas_and_prefetched_rgb_batches() -> None:
    configuration = RunConfiguration(
        protocol_id="loader-supply",
        decoder_id="pillow",
        package_id="a" * 64,
        manifest_id="b" * 64,
        selection_id="c" * 64,
        requested_threads=None,
        num_workers=2,
        batch_size=1,
        prefetch_factor=1,
        persistent_workers=True,
        multiprocessing_start_method="spawn",
        logical_repeat_factor=1,
        warmup_passes=1,
        timed_passes_per_run=2,
        minimum_timed_seconds=0.1,
        output_contract="normalized-rgb",
        support_policy="common",
    )
    items = (
        {"item_id": "one", "compressed_bytes": 100, "width": 10, "height": 20},
        {"item_id": "two", "compressed_bytes": 200, "width": 5, "height": 5},
    )

    estimate = estimate_peak_memory(configuration, items)

    assert estimate.resident_compressed_bytes == 300
    assert estimate.resident_replica_count == 3
    assert estimate.decoded_inflight_bytes == 3_000
    assert estimate.estimated_peak_bytes == 3_900

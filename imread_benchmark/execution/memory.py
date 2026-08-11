from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from imread_benchmark.plans import RunConfiguration


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    resident_compressed_bytes: int
    resident_replica_count: int
    decoded_inflight_bytes: int
    estimated_peak_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "decoded_inflight_bytes": self.decoded_inflight_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "resident_compressed_bytes": self.resident_compressed_bytes,
            "resident_replica_count": self.resident_replica_count,
        }


def estimate_peak_memory(
    configuration: RunConfiguration,
    items: Sequence[Mapping[str, object]],
) -> MemoryEstimate:
    if not items:
        raise ValueError("memory estimate requires at least one manifest item")
    compressed = sum(_positive_int(item, "compressed_bytes") for item in items)
    max_decoded = max(_positive_int(item, "width") * _positive_int(item, "height") * 3 for item in items)
    replicas = _resident_replicas(configuration)
    decoded = _decoded_inflight(configuration, max_decoded)
    return MemoryEstimate(
        resident_compressed_bytes=compressed,
        resident_replica_count=replicas,
        decoded_inflight_bytes=decoded,
        estimated_peak_bytes=compressed * replicas + decoded,
    )


def _resident_replicas(configuration: RunConfiguration) -> int:
    if configuration.protocol_id != "loader-supply" or not configuration.num_workers:
        return 1
    if configuration.multiprocessing_start_method == "fork":
        return 1
    return configuration.num_workers + 1


def _decoded_inflight(configuration: RunConfiguration, max_decoded: int) -> int:
    if configuration.protocol_id == "decode-memory":
        return max_decoded
    if configuration.batch_size is None or configuration.num_workers is None:
        raise ValueError("loader memory estimate requires batch_size and num_workers")
    if configuration.num_workers == 0:
        return (configuration.batch_size + 1) * max_decoded
    if configuration.prefetch_factor is None:
        raise ValueError("loader worker memory estimate requires prefetch_factor")
    queued_batches = configuration.num_workers * configuration.prefetch_factor + 1
    queued_decoded = queued_batches * configuration.batch_size * max_decoded
    worker_temporaries = configuration.num_workers * max_decoded
    return queued_decoded + worker_temporaries


def _positive_int(item: Mapping[str, object], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        item_id = item.get("item_id", "<unknown>")
        raise ValueError(f"manifest item {item_id!r} has invalid {key}")
    return value

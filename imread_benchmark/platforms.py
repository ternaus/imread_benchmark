from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

PLATFORM_SCHEMA_VERSION = "2.0"
_THREAD_ENVIRONMENT_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class PlatformDescriptor:
    platform_id: str
    identity: dict[str, object]
    runtime: dict[str, object]
    provenance: dict[str, object]

    @classmethod
    def build(
        cls,
        *,
        identity: dict[str, object],
        runtime: dict[str, object],
        provenance: dict[str, object] | None = None,
    ) -> PlatformDescriptor:
        if not identity or any(not isinstance(key, str) or not key for key in identity):
            raise ValueError("platform identity must be a non-empty object")
        if any(not isinstance(key, str) or not key for key in runtime):
            raise ValueError("platform runtime keys must be non-empty strings")
        if provenance is not None and any(not isinstance(key, str) or not key for key in provenance):
            raise ValueError("platform provenance keys must be non-empty strings")
        normalized_identity = _canonical_object(identity)
        normalized_runtime = _canonical_object(runtime)
        normalized_provenance = _canonical_object(provenance or {})
        platform_id = _digest(
            {
                "identity": normalized_identity,
                "schema_version": PLATFORM_SCHEMA_VERSION,
            },
        )
        return cls(
            platform_id=platform_id,
            identity=normalized_identity,
            runtime=normalized_runtime,
            provenance=normalized_provenance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "platform_id": self.platform_id,
            "provenance": self.provenance,
            "runtime": self.runtime,
            "schema_version": PLATFORM_SCHEMA_VERSION,
        }

    @property
    def comparison_platform_id(self) -> str:
        """Stable comparison key that deliberately excludes execution zone."""
        return _comparison_platform_id(self.identity)


def capture_current_platform(
    *,
    cloud_provider: str,
    machine_type: str,
    location: str,
) -> PlatformDescriptor:
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info()
    except Exception as exc:
        raise ValueError(f"cannot probe CPU information: {exc}") from exc
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        raise ValueError("cannot determine logical CPU count")
    identity: dict[str, object] = {
        "architecture": platform.machine(),
        "cloud_provider": cloud_provider,
        "cpu_architecture": cpu.get("arch_string_raw") or cpu.get("arch") or platform.machine(),
        "cpu_brand": cpu.get("brand_raw") or "unknown",
        "cpu_family": cpu.get("family"),
        "cpu_model": cpu.get("model"),
        "cpu_stepping": cpu.get("stepping"),
        "cpu_vendor": cpu.get("vendor_id_raw") or "unknown",
        "logical_cpu_count": logical_cpu_count,
        "machine_type": machine_type,
        "system": platform.system(),
    }
    runtime: dict[str, object] = {
        "available_multiprocessing_start_methods": _multiprocessing_start_methods(),
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "libc": list(platform.libc_ver()),
        "memory_bytes": _memory_bytes(),
        "thread_environment": {key: os.environ[key] for key in _THREAD_ENVIRONMENT_VARIABLES if key in os.environ},
    }
    return PlatformDescriptor.build(
        identity=identity,
        runtime=runtime,
        provenance={"location": location},
    )


def write_platform_descriptor(path: str | Path, descriptor: PlatformDescriptor) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(descriptor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"immutable platform descriptor already exists with different content: {path}")
        return path
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError(f"concurrent platform descriptor differs: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_platform_descriptor(path: str | Path) -> PlatformDescriptor:
    source = Path(path)
    try:
        document = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read platform descriptor {source}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != PLATFORM_SCHEMA_VERSION:
        raise ValueError("unsupported platform descriptor schema")
    identity = document.get("identity")
    runtime = document.get("runtime")
    provenance = document.get("provenance", {})
    if not isinstance(identity, dict) or not isinstance(runtime, dict) or not isinstance(provenance, dict):
        raise TypeError("platform descriptor identity, runtime, and provenance must be objects")
    descriptor = PlatformDescriptor.build(identity=identity, runtime=runtime, provenance=provenance)
    if document.get("platform_id") != descriptor.platform_id:
        raise ValueError("platform_id does not match descriptor content")
    return descriptor


def platform_comparison_id(document: dict[str, object]) -> str:
    """
    Return the zone-independent comparison key for a persisted descriptor.

    Legacy descriptors recorded ``location`` in ``identity``.  New descriptors
    record it in ``provenance``.  Normalizing both forms here keeps already
    committed evidence usable without concealing the execution zone.
    """
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("platform descriptor identity must be an object")
    return _comparison_platform_id(identity)


def platform_location(document: dict[str, object]) -> str | None:
    """Return the recorded execution zone from new or legacy provenance."""
    provenance = document.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("location")
        if isinstance(value, str) and value:
            return value
    identity = document.get("identity")
    if isinstance(identity, dict):
        value = identity.get("location")
        if isinstance(value, str) and value:
            return value
    return None


def _canonical_object(value: dict[str, object]) -> dict[str, object]:
    try:
        canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
        loaded = json.loads(canonical)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"platform descriptor contains a non-JSON value: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError("canonical platform object is not a mapping")
    return loaded


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _comparison_platform_id(identity: dict[str, object]) -> str:
    comparison_identity = {key: value for key, value in identity.items() if key != "location"}
    normalized_identity = _canonical_object(comparison_identity)
    return _digest(
        {
            "identity": normalized_identity,
            "schema_version": PLATFORM_SCHEMA_VERSION,
        },
    )


def _multiprocessing_start_methods() -> list[str]:
    import multiprocessing

    return sorted(multiprocessing.get_all_start_methods())


def _memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    return pages * page_size

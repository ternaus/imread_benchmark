from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cpuinfo

logger = logging.getLogger(__name__)


def get_cpu_info() -> dict[str, object]:
    try:
        info = cpuinfo.get_cpu_info()
        return {
            "brand_raw": info.get("brand_raw", "Unknown"),
            "arch": info.get("arch", "Unknown"),
            "hz_advertised_raw": info.get("hz_advertised_raw", "Unknown"),
            "count": multiprocessing.cpu_count(),
        }
    except Exception as exc:
        logger.warning("Failed to get CPU info: %s", exc)
        return {"error": str(exc)}


def get_system_identifier() -> str:
    try:
        info = cpuinfo.get_cpu_info()
        cpu_brand = info.get("brand_raw", "Unknown")
        os_id = "darwin" if platform.system().lower() == "darwin" else "linux"
        cpu_id = cpu_brand.replace(" ", "-")
    except Exception as exc:
        logger.warning("Failed to get system info: %s", exc)
        return "unknown-system"
    else:
        return f"{os_id}_{cpu_id}"


def get_package_versions(library_name: str | None = None) -> dict[str, object]:
    versions: dict[str, object] = {
        "Python": sys.version.split()[0],
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "CPU": get_cpu_info(),
    }

    if library_name is None:
        library_name = os.environ.get("BENCHMARK_LIBRARY")

    if library_name:
        from imread_benchmark.decoders import REGISTRY

        decoder_cls = REGISTRY.get(library_name)
        if decoder_cls is not None:
            pkg = decoder_cls.package_name
            try:
                versions[library_name] = version(pkg)
            except PackageNotFoundError:
                versions[library_name] = f"not installed ({pkg})"
            except Exception as exc:
                versions[library_name] = f"error: {exc}"

    return versions


def collect_jpeg_paths(data_dir: str | Path, num_images: int) -> list[Path]:
    extensions = {".jpg", ".jpeg"}
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in extensions)[:num_images]

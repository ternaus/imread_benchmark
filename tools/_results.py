"""
Shared loader for imread-benchmark JSON results.

Used by both `tools/create_plots.py` (charts) and `tools/render_readme.py`
(README tables) so they agree on schema, column names, and library aliasing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LIBRARY_ORDER: tuple[str, ...] = (
    "simplejpeg",
    "turbojpeg",
    "jpeg4py",
    "kornia-rs",
    "opencv",
    "imagecodecs",
    "pyvips",
    "pillow",
    "skimage",
    "imageio",
    "torchvision",
    "tensorflow",
)


def _alias(library: str) -> str:
    return "kornia-rs" if library == "kornia" else library


def _extract_throughput(results: dict) -> tuple[float | None, float | None]:
    """
    Read mean / std img/s from a benchmark_results dict.

    New format (post-Track 2b): images_per_second_mean / _std as floats.
    Legacy format (older runs): images_per_second as "MEAN ± STD" string.
    """
    mean = results.get("images_per_second_mean")
    std = results.get("images_per_second_std")
    if mean is not None:
        return float(mean), float(std) if std is not None else None

    legacy = results.get("images_per_second")
    if isinstance(legacy, str) and "±" in legacy:
        m_str, s_str = legacy.split("±")
        try:
            return float(m_str.strip()), float(s_str.strip())
        except ValueError:
            return None, None
    return None, None


def load_results(input_dir: Path) -> pd.DataFrame:
    """
    Load every output/<platform>/<lib>_<N>t_results.json into a DataFrame.

    Single-thread / fixed-thread results only — dataloader files are skipped
    here and handled by load_dataloader_results().

    Columns: platform, library, num_threads, num_images, num_runs,
             images_per_second, std_dev, p50, p90, p99, us_per_image,
             cpu_brand, os_name.
    """
    rows: list[dict] = []
    for platform_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        platform = platform_dir.name
        for result_file in sorted(platform_dir.glob("*_results.json")):
            if result_file.name.endswith("_dataloader_results.json"):
                continue
            with result_file.open() as f:
                data = json.load(f)

            results = data.get("benchmark_results", {})
            sysinfo = data.get("system_info", {})
            cpu = sysinfo.get("CPU", {}) if isinstance(sysinfo.get("CPU"), dict) else {}

            mean_ips, std_ips = _extract_throughput(results)

            rows.append(
                {
                    "platform": platform,
                    "library": _alias(data["library"]),
                    "run_tag": data.get("run_tag", ""),
                    "num_threads": data.get("num_threads") or data.get("effective_threads") or 1,
                    "num_images": data.get("num_images"),
                    "num_runs": data.get("num_runs"),
                    "images_per_second": mean_ips,
                    "std_dev": std_ips,
                    "p50": results.get("images_per_second_p50"),
                    "p90": results.get("images_per_second_p90"),
                    "p99": results.get("images_per_second_p99"),
                    "us_per_image": results.get("us_per_image_mean"),
                    "skip_rate": results.get("skip_rate"),
                    "num_images_skipped": results.get("num_images_skipped"),
                    "cpu_brand": cpu.get("brand_raw", "Unknown CPU"),
                    "os_name": sysinfo.get("OS", platform.split("_")[0].title()),
                },
            )
    return pd.DataFrame(rows)


def load_dataloader_results(input_dir: Path) -> pd.DataFrame:
    """
    Load *_dataloader_results.json into a long DataFrame.

    Columns: platform, library, num_workers, images_per_second, std_dev,
             cpu_brand, os_name, num_images, num_runs.
    """
    rows: list[dict] = []
    for platform_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        platform = platform_dir.name
        for result_file in sorted(platform_dir.glob("*_dataloader_results.json")):
            with result_file.open() as f:
                data = json.load(f)

            sysinfo = data.get("system_info", {})
            cpu = sysinfo.get("CPU", {}) if isinstance(sysinfo.get("CPU"), dict) else {}
            library = _alias(data["library"])

            wrs = data.get("worker_results", [])
            rows.extend(
                {
                    "platform": platform,
                    "library": library,
                    "num_workers": wr.get("num_workers"),
                    "images_per_second": wr.get("images_per_second_mean"),
                    "std_dev": wr.get("images_per_second_std"),
                    "cpu_brand": cpu.get("brand_raw", "Unknown CPU"),
                    "os_name": sysinfo.get("OS", platform.split("_")[0].title()),
                    "num_images": data.get("num_images"),
                    "num_runs": data.get("num_runs"),
                }
                for wr in wrs
            )
    return pd.DataFrame(rows)


_KEEP_UPPER = {"AMD", "ARM", "CPU", "EPYC", "GPU", "GHz", "IBM"}


def _prettify_cpu(brand: str) -> str:
    """Trim noise from a cpuinfo brand string and de-shout ALL CAPS tokens."""
    s = brand.replace("(R)", "").replace("(TM)", "")
    s = " ".join(s.split())
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    out: list[str] = []
    for tok in s.split(" "):
        if tok in _KEEP_UPPER or tok.upper() == "CPU":
            if tok.upper() == "CPU":
                continue
            out.append(tok)
        elif tok.isalpha() and tok.isupper() and len(tok) > 2:
            out.append(tok.title())
        else:
            out.append(tok)
    return " ".join(out)


def short_platform(platform: str, cpu_brand: str | None = None) -> str:
    """
    Turn 'linux_AMD-EPYC-9B45' or a recorded CPU brand into a compact label.

    Prefers the recorded `cpu_brand` (e.g. 'AMD EPYC 9B45') when available,
    falling back to splitting the directory name.
    """
    if cpu_brand and cpu_brand != "Unknown CPU":
        return _prettify_cpu(cpu_brand)
    _, _, rest = platform.partition("_")
    return rest.replace("-", " ") if rest else platform

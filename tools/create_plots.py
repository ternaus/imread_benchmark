"""
Generate paper-quality plots from imread-benchmark JSON results.

Reads numeric fields directly (no string parsing) and derives titles from
each result's recorded `system_info`, so adding a new platform is zero-config.

Run via the CLI:
    imread-benchmark plot --input output --output _internal/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="Arial")
sns.set_context("paper", font_scale=1.5)


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

    Columns: platform, library, num_threads, num_images, num_runs,
             images_per_second, std_dev, p50, p90, p99, us_per_image,
             cpu_brand, os_name.
    """
    rows: list[dict] = []
    for platform_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        platform = platform_dir.name
        for result_file in sorted(platform_dir.glob("*_results.json")):
            # Skip dataloader files — they have a different schema and a separate plotter.
            if result_file.name.endswith("_dataloader_results.json"):
                continue
            with result_file.open() as f:
                data = json.load(f)

            results = data.get("benchmark_results", {})
            sysinfo = data.get("system_info", {})
            cpu = sysinfo.get("CPU", {}) if isinstance(sysinfo.get("CPU"), dict) else {}

            library = data["library"]
            if library == "kornia":
                library = "kornia-rs"

            mean_ips, std_ips = _extract_throughput(results)

            rows.append(
                {
                    "platform": platform,
                    "library": library,
                    "num_threads": data.get("num_threads") or 1,
                    "num_images": data.get("num_images"),
                    "num_runs": data.get("num_runs"),
                    "images_per_second": mean_ips,
                    "std_dev": std_ips,
                    "p50": results.get("images_per_second_p50"),
                    "p90": results.get("images_per_second_p90"),
                    "p99": results.get("images_per_second_p99"),
                    "us_per_image": results.get("us_per_image_mean"),
                    "cpu_brand": cpu.get("brand_raw", "Unknown CPU"),
                    "os_name": sysinfo.get("OS", platform.split("_")[0].title()),
                },
            )
    return pd.DataFrame(rows)


def _title_for_platform(df_platform: pd.DataFrame) -> str:
    """Build the figure title from the platform's recorded system_info, no hard-coded map."""
    cpu = df_platform["cpu_brand"].iloc[0]
    os_name = df_platform["os_name"].iloc[0]
    return f"JPEG Decoding Speed — {os_name} / {cpu}"


def plot_platform_performance(df: pd.DataFrame, platform: str, output_path: Path) -> None:
    plt.style.use("default")
    sns.set_theme(style="whitegrid", font="Arial")

    # Single-thread snapshot only (avoids mixing 1-thread and N-thread bars).
    pdata = df[(df["platform"] == platform) & (df["num_threads"] == 1)].copy()
    if pdata.empty:
        return
    pdata = pdata.sort_values("images_per_second", ascending=True)

    plt.figure(figsize=(7, 5))
    n_bars = len(pdata)
    colors = sns.color_palette("Blues", n_colors=n_bars)
    bars = plt.barh(range(len(pdata)), pdata["images_per_second"], height=0.7, color=colors)

    plt.errorbar(
        pdata["images_per_second"],
        range(len(pdata)),
        xerr=pdata["std_dev"],
        fmt="none",
        color="black",
        capsize=4,
        alpha=0.5,
        linewidth=1.5,
    )

    for i, bar in enumerate(bars):
        width = bar.get_width()
        text_color = "white" if i > n_bars / 2 else "black"
        plt.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=14,
            fontweight="bold",
        )

    plt.title(_title_for_platform(pdata), pad=20, fontsize=14, fontweight="bold")
    plt.xlabel("Images per Second (1 thread)", fontsize=14, fontweight="bold")
    plt.yticks(range(len(pdata)), pdata["library"], fontsize=14)

    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=1.5)
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot imread-benchmark single-thread results.")
    parser.add_argument("--input", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("_internal/plots"))
    args = parser.parse_args()

    df = load_results(args.input)
    if df.empty:
        print(f"No results found under {args.input}/<platform>/*_results.json")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for platform in sorted(df["platform"].unique()):
        out_file = args.output / f"performance_{platform}.png"
        plot_platform_performance(df, platform, out_file)
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="Arial")
sns.set_context("paper", font_scale=1.5)


class BenchmarkResults(TypedDict):
    images_per_second: str
    raw_times: list[float]


class SystemInfo(TypedDict):
    Python: str
    OS: str
    OS_Version: NotRequired[str]
    Machine: str
    CPU: dict
    imageio: NotRequired[str]
    kornia: NotRequired[str]
    opencv: NotRequired[str]
    skimage: NotRequired[str]
    tensorflow: NotRequired[str]
    torchvision: NotRequired[str]


class ResultData(TypedDict):
    library: str
    system_info: SystemInfo
    benchmark_results: BenchmarkResults
    num_images: int
    num_runs: int


def load_results(path: str | Path) -> pd.DataFrame:
    """Load all JSON results and convert to DataFrame."""
    results: list[dict] = []
    path = Path(path)

    for platform_dir in path.iterdir():
        if not platform_dir.is_dir():
            continue

        platform = platform_dir.name

        for result_file in platform_dir.glob("*_results.json"):
            with result_file.open() as f:
                data: ResultData = json.load(f)

            library = data["library"]
            if library == "kornia":
                library = "kornia-rs"

            mean_str, std_str = data["benchmark_results"]["images_per_second"].split("±")
            mean = float(mean_str.strip())
            std = float(std_str.strip())

            results.append(
                {
                    "platform": platform,
                    "library": library,
                    "images_per_second": mean,
                    "std_dev": std,
                },
            )

    return pd.DataFrame(results)


def plot_platform_performance(df: pd.DataFrame, platform: str, output_path: str | Path) -> None:
    """Create a publication-quality horizontal bar plot optimized for two-column paper format."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", font="Arial")

    platform_data = df[df["platform"] == platform].copy()
    platform_data = platform_data.sort_values("images_per_second", ascending=True)

    # Figure size for two-column paper
    plt.figure(figsize=(7, 5))

    # Generate colors
    n_bars = len(platform_data)
    colors = sns.color_palette("Blues", n_colors=n_bars)

    # Create horizontal bars
    bars = plt.barh(range(len(platform_data)), platform_data["images_per_second"], height=0.7, color=colors)

    # Add error bars
    plt.errorbar(
        platform_data["images_per_second"],
        range(len(platform_data)),
        xerr=platform_data["std_dev"],
        fmt="none",
        color="black",
        capsize=4,
        alpha=0.5,
        linewidth=1.5,
    )

    # Add value labels inside bars
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

    # Concise, single-line titles
    platform_titles = {
        "darwin": "JPEG Decoding Speed (Apple M4 Max)",
        "linux": "JPEG Decoding Speed (AMD Threadripper 3970X)",
    }
    plt.title(platform_titles[platform], pad=20, fontsize=16, fontweight="bold")
    plt.xlabel("Images per Second", fontsize=14, fontweight="bold")
    plt.yticks(range(len(platform_data)), platform_data["library"], fontsize=14)

    # Thicker axis lines
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)

    # Adjust grid
    plt.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=1.5)

    # Adjust layout
    plt.tight_layout(pad=1.2)

    # Save plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Load and process data
    df = load_results("output")

    # Create visualizations
    # Create separate plots for each platform
    plot_platform_performance(df, "darwin", "performance_darwin.png")
    plot_platform_performance(df, "linux", "performance_linux.png")


if __name__ == "__main__":
    main()

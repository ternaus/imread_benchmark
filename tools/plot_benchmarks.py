from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BenchmarkResults(TypedDict):
    images_per_second: str
    raw_times: list[float]


class ResultData(TypedDict):
    library: str
    benchmark_results: BenchmarkResults
    num_images: int
    num_runs: int
    batch_size: int
    num_workers: int


def load_results(folder: Path) -> pd.DataFrame:
    """Load all JSON results from a folder and convert to DataFrame."""
    results = []

    for json_file in folder.glob("**/*_results.json"):
        with json_file.open() as f:
            data: ResultData = json.load(f)

        # Calculate images per second for each raw time
        raw_times = data["benchmark_results"]["raw_times"]

        # Calculate median and standard deviation
        median = np.median(raw_times)
        std = np.std(raw_times)

        results.append(
            {
                "library": data["library"],
                "images_per_second": median,
                "std_dev": std,
                "batch_size": data["batch_size"],
                "num_workers": data["num_workers"],
            },
        )

    return pd.DataFrame(results)


def create_performance_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create a publication-quality horizontal bar plot."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", font="Arial")

    # Sort by performance
    df = df.sort_values("images_per_second", ascending=True)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Generate colors
    n_bars = len(df)
    colors = sns.color_palette("Blues", n_colors=n_bars)

    # Create horizontal bars
    bars = plt.barh(range(len(df)), df["images_per_second"], height=0.7, color=colors)

    # Add error bars
    plt.errorbar(
        df["images_per_second"],
        range(len(df)),
        xerr=df["std_dev"],
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
            fontsize=12,
            fontweight="bold",
        )

    # Customize plot
    plt.xlabel("Images per Second", fontsize=14, fontweight="bold")

    # Create labels with additional information
    labels = [row["library"] for _, row in df.iterrows()]
    plt.yticks(range(len(df)), labels, fontsize=12)

    # Adjust grid and styling
    plt.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=1.5)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)

    # Adjust layout and save
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plot from benchmark results")
    parser.add_argument("-i", "--input_dir", type=str, help="Directory containing benchmark JSON results")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="performance_comparison.png",
        help="Output path for the plot (default: performance_comparison.png)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data and create plot
    df = load_results(input_dir)
    create_performance_plot(df, output_path)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

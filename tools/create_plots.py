"""
Generate paper-quality plots from imread-benchmark JSON results.

Reads numeric fields directly (no string parsing) and derives titles from
each result's recorded `system_info`, so adding a new platform is zero-config.

Run via the CLI:
    imread-benchmark plot --input output --output docs/assets/benchmarks

Or directly (from the repo root):
    python -m tools.create_plots --input output --output docs/assets/benchmarks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools._results import LIBRARY_ORDER, load_dataloader_results, load_results, short_platform
from tools.paper_assets import (
    CLAIM_FIGURE_BASENAMES,
    plot_fig01_protocol_rank_change,
    plot_fig02_amd_worker_delta,
    plot_fig03_tensorflow_arm_penalty,
    plot_fig04_cross_platform_recommendation,
    validate_paper_data,
)

README_SINGLE_PLOT = "single_thread_overview.png"
README_DATALOADER_PLOT = "dataloader_peak_overview.png"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
sns.set_context("paper", font_scale=1.5)


def _title_for_platform(df_platform: pd.DataFrame) -> str:
    """Build the figure title from the platform's recorded system_info, no hard-coded map."""
    cpu = df_platform["cpu_brand"].iloc[0]
    os_name = df_platform["os_name"].iloc[0]
    return f"JPEG Decoding Speed — {os_name} / {cpu}"


def plot_platform_performance(df: pd.DataFrame, platform: str, output_path: Path) -> None:
    plt.style.use("default")
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

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


def _platform_labels(df: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for platform in sorted(df["platform"].unique()):
        cpu = df[df["platform"] == platform]["cpu_brand"].iloc[0]
        labels[platform] = short_platform(platform, cpu)
    return labels


def _ordered_libraries(df: pd.DataFrame) -> list[str]:
    seen = set(df["library"].unique())
    known = [lib for lib in LIBRARY_ORDER if lib in seen]
    unknown = sorted(seen - set(known))
    return [*known, *unknown]


def _leaderboard_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    labels = _platform_labels(df)
    pivot = df.pivot_table(
        index="library",
        columns="platform",
        values="images_per_second",
        aggfunc="max",
    )
    pivot = pivot.reindex(index=_ordered_libraries(df), columns=labels.keys())
    pivot = pivot.rename(columns=labels)
    return pivot, labels


def _annotate_values(values: pd.DataFrame, relative: pd.DataFrame) -> pd.DataFrame:
    annotations = values.copy().astype("object")
    for row in values.index:
        for col in values.columns:
            value = values.loc[row, col]
            pct = relative.loc[row, col]
            annotations.loc[row, col] = "" if pd.isna(value) else f"{value:,.0f}\n{pct:.0f}%"
    return annotations


def _plot_overview_heatmap(values: pd.DataFrame, title: str, output_path: Path) -> None:
    relative = values.div(values.max(axis=0), axis=1) * 100
    annotations = _annotate_values(values, relative)

    height = max(5.0, 0.42 * len(values.index) + 1.8)
    width = max(8.0, 1.45 * len(values.columns) + 3.2)
    plt.figure(figsize=(width, height))
    ax = sns.heatmap(
        relative,
        vmin=0,
        vmax=100,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        annot=annotations,
        fmt="",
        cbar_kws={"label": "% of fastest on that platform"},
    )
    ax.set_title(title, pad=18, fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_single_thread_overview(df: pd.DataFrame, output_path: Path) -> None:
    single = df[df["run_tag"] == "1t"].copy()
    if single.empty:
        return
    values, _ = _leaderboard_matrix(single)
    _plot_overview_heatmap(values, "Single-thread JPEG decode throughput", output_path)


def plot_dataloader_overview(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    peak_idx = df.groupby(["platform", "library"])["images_per_second"].idxmax()
    peaks = df.loc[peak_idx].copy()
    values, _ = _leaderboard_matrix(peaks)
    _plot_overview_heatmap(values, "Peak PyTorch DataLoader throughput", output_path)


def plot_claim_first_readme_figures(single: pd.DataFrame, dl: pd.DataFrame, output_dir: Path) -> bool:
    """Write the paper/README claim-first figure set when the full paper matrix is present."""
    try:
        validate_paper_data(single, dl)
    except ValueError as exc:
        print(f"paper claim-first figures skipped: {exc}")
        return False

    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", font_scale=1.1)
    plot_fig01_protocol_rank_change(single, dl, output_dir, ("png",))
    plot_fig02_amd_worker_delta(dl, output_dir, ("png",))
    plot_fig03_tensorflow_arm_penalty(single, output_dir, ("png",))
    plot_fig04_cross_platform_recommendation(dl, output_dir, ("png",))
    for basename in CLAIM_FIGURE_BASENAMES:
        print(f"wrote {output_dir / (basename + '.png')}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot imread-benchmark single-thread results.")
    parser.add_argument("--input", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/benchmarks"))
    args = parser.parse_args()

    df = load_results(args.input)
    if df.empty:
        print(f"No results found under {args.input}/<platform>/*_results.json")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    ddf = load_dataloader_results(args.input)
    wrote_claim_figures = plot_claim_first_readme_figures(df, ddf, args.output)
    if not wrote_claim_figures:
        single_overview = args.output / README_SINGLE_PLOT
        plot_single_thread_overview(df, single_overview)
        print(f"wrote {single_overview}")

        dataloader_overview = args.output / README_DATALOADER_PLOT
        plot_dataloader_overview(ddf, dataloader_overview)
        print(f"wrote {dataloader_overview}")

        for platform in sorted(df["platform"].unique()):
            out_file = args.output / f"performance_{platform}.png"
            plot_platform_performance(df, platform, out_file)
            print(f"wrote {out_file}")


if __name__ == "__main__":
    main()

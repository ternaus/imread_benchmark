"""
Generate NeurIPS paper tables and figures from imread-benchmark JSON under output/.

Uses tools._results loaders so numbers agree with README / tools/create_plots.

Usage (from repo root, with plot extras):

    uv run --extra plot python -m tools.paper_assets --all
    uv run --extra plot python -m tools.paper_assets --tables
    uv run --extra plot python -m tools.paper_assets --figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from tools._results import LIBRARY_ORDER, load_dataloader_results, load_results, short_platform

# ---------------------------------------------------------------------------
# Paper scope: five GCP -standard-16 platforms (directory names under output/)
# ---------------------------------------------------------------------------

PAPER_PLATFORMS: tuple[str, ...] = (
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz",
    "linux_AMD-EPYC-9B14",
    "linux_AMD-EPYC-9B45",
    "linux_Neoverse-V2",
    "linux_Neoverse-N1",
)

PLATFORM_MACHINE: dict[str, str] = {
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz": "c4-standard-16",
    "linux_AMD-EPYC-9B14": "c3d-standard-16",
    "linux_AMD-EPYC-9B45": "c4d-standard-16",
    "linux_Neoverse-V2": "c4a-standard-16",
    "linux_Neoverse-N1": "t2a-standard-16",
}

PLATFORM_MICROARCH: dict[str, str] = {
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz": "Intel Emerald Rapids",
    "linux_AMD-EPYC-9B14": "AMD Zen 4 (Genoa)",
    "linux_AMD-EPYC-9B45": "AMD Zen 5 (Turin)",
    "linux_Neoverse-V2": "ARM Neoverse V2 (Google Axion)",
    "linux_Neoverse-N1": "ARM Neoverse N1 (Ampere Altra)",
}

PLATFORM_SMT: dict[str, str] = {
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz": "yes (SMT2)",
    "linux_AMD-EPYC-9B14": "yes (SMT2)",
    "linux_AMD-EPYC-9B45": "yes (SMT2)",
    "linux_Neoverse-V2": "no",
    "linux_Neoverse-N1": "no",
}

ZEN4_PLATFORM = "linux_AMD-EPYC-9B14"
ZEN5_PLATFORM = "linux_AMD-EPYC-9B45"

WORKERS_ORDER = (0, 2, 4, 8)
EXPECTED_SINGLE_DECODERS = set(LIBRARY_ORDER)
EXPECTED_DATALOADER_DECODERS = set(LIBRARY_ORDER) - {"pyvips", "tensorflow"}
EXPECTED_SKIP_DECODERS = {"jpeg4py", "kornia-rs", "turbojpeg"}
ROBUSTNESS_DECODERS = ("jpeg4py", "kornia-rs", "turbojpeg", "pyvips", "tensorflow")

FIG_DPI = 300
CLAIM_FIGURE_BASENAMES: tuple[str, ...] = (
    "fig01_protocol_rank_change",
    "fig02_amd_worker_delta",
    "fig03_tensorflow_arm_penalty",
    "fig04_decoder_recommendation_summary",
)
RANK_CHANGE_PLATFORMS: tuple[str, ...] = (
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz",
    ZEN4_PLATFORM,
    "linux_Neoverse-V2",
)
RANK_CHANGE_HIGHLIGHTS: dict[str, str] = {
    "linux_INTEL(R)-XEON(R)-PLATINUM-8581C-CPU-@-2.30GHz": "imageio",
    ZEN4_PLATFORM: "torchvision",
    "linux_Neoverse-V2": "imageio",
}


def _paper_scope(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["platform"].isin(PAPER_PLATFORMS)].copy()


def _platform_labels(cpu_by_platform: dict[str, str]) -> dict[str, str]:
    return {p: short_platform(p, cpu_by_platform.get(p)) for p in PAPER_PLATFORMS}


def _cpu_by_platform(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in PAPER_PLATFORMS:
        sub = df[df["platform"] == p]
        if not sub.empty:
            out[p] = str(sub["cpu_brand"].iloc[0])
    return out


def _ordered_libs_present(df: pd.DataFrame) -> list[str]:
    seen = set(df["library"].unique())
    return [lib for lib in LIBRARY_ORDER if lib in seen]


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [_md_row(headers), _md_row(["---"] * len(headers))]
    lines.extend(_md_row(r) for r in rows)
    return "\n".join(lines) + "\n"


def _fmt_mean_std(mean: float | None, std: float | None) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.0f}"
    return f"{mean:.0f} ± {std:.0f}"


def _fmt_ips(v: float | None) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.0f}"


def _validate_platform_set(label: str, actual: set[str]) -> None:
    expected_platforms = set(PAPER_PLATFORMS)
    if actual != expected_platforms:
        missing = sorted(expected_platforms - actual)
        extra = sorted(actual - expected_platforms)
        raise ValueError(f"{label} platform mismatch: missing={missing}, extra={extra}")


def _validate_single_decoders(single: pd.DataFrame) -> None:
    for plat in PAPER_PLATFORMS:
        libs = set(single[single["platform"] == plat]["library"].unique())
        if libs != EXPECTED_SINGLE_DECODERS:
            missing = sorted(EXPECTED_SINGLE_DECODERS - libs)
            extra = sorted(libs - EXPECTED_SINGLE_DECODERS)
            raise ValueError(f"{plat} single-thread decoder mismatch: missing={missing}, extra={extra}")


def _validate_dataloader_decoders(dl: pd.DataFrame) -> None:
    expected_workers = set(WORKERS_ORDER)
    for plat in PAPER_PLATFORMS:
        dl_sub = dl[dl["platform"] == plat]
        dl_libs = set(dl_sub["library"].unique())
        if dl_libs != EXPECTED_DATALOADER_DECODERS:
            missing = sorted(EXPECTED_DATALOADER_DECODERS - dl_libs)
            extra = sorted(dl_libs - EXPECTED_DATALOADER_DECODERS)
            raise ValueError(f"{plat} DataLoader decoder mismatch: missing={missing}, extra={extra}")
        for lib in EXPECTED_DATALOADER_DECODERS:
            workers = {int(w) for w in dl_sub[dl_sub["library"] == lib]["num_workers"].unique()}
            if workers != expected_workers:
                raise ValueError(f"{plat}/{lib} worker mismatch: got={sorted(workers)}")


def _validate_skip_decoders(single: pd.DataFrame) -> None:
    skipped = single[single["num_images_skipped"].fillna(0) > 0]
    skipped_libs = set(skipped["library"].unique())
    if skipped_libs != EXPECTED_SKIP_DECODERS:
        raise ValueError(
            "unexpected single-thread skip decoders: "
            f"got={sorted(skipped_libs)}, expected={sorted(EXPECTED_SKIP_DECODERS)}",
        )
    for lib in EXPECTED_SKIP_DECODERS:
        counts = skipped[skipped["library"] == lib]["num_images_skipped"].astype(int).tolist()
        if counts != [1] * len(PAPER_PLATFORMS):
            raise ValueError(f"{lib} skip counts must be 1 on every paper platform, got={counts}")


def validate_paper_data(df_1t: pd.DataFrame, dl: pd.DataFrame) -> dict[str, int]:
    """Validate the result matrix assumed by the paper narrative."""
    single = _paper_scope(df_1t)
    single = single[single["run_tag"] == "1t"]
    dl = _paper_scope(dl)

    _validate_platform_set("single-thread", set(single["platform"].unique()))
    _validate_platform_set("DataLoader", set(dl["platform"].unique()))
    _validate_single_decoders(single)
    _validate_dataloader_decoders(dl)
    _validate_skip_decoders(single)

    return {
        "platforms": len(PAPER_PLATFORMS),
        "single_thread_decoders": len(EXPECTED_SINGLE_DECODERS),
        "dataloader_decoders": len(EXPECTED_DATALOADER_DECODERS),
        "single_thread_rows": len(single),
        "dataloader_worker_rows": len(dl),
    }


def generate_hardware_table(df_1t: pd.DataFrame, dest: Path) -> None:
    cpu_map = _cpu_by_platform(df_1t)
    plat_labels = _platform_labels(cpu_map)
    headers = [
        "Output dir",
        "GCP machine type",
        "CPU (recorded)",
        "Microarchitecture (paper label)",
        "SMT",
        "Short label",
    ]
    rows: list[list[str]] = [
        [
            f"`{plat}/`",
            PLATFORM_MACHINE[plat],
            cpu_map.get(plat, "—"),
            PLATFORM_MICROARCH[plat],
            PLATFORM_SMT[plat],
            plat_labels[plat],
        ]
        for plat in PAPER_PLATFORMS
    ]
    dest.write_text(
        "# Table 1 — Hardware / platform matrix\n\n"
        "_Generated by `python -m tools.paper_assets --tables`._\n\n" + _md_table(headers, rows),
        encoding="utf-8",
    )


def generate_single_thread_table(df_1t: pd.DataFrame, dest: Path) -> None:
    df = _paper_scope(df_1t)
    df = df[df["run_tag"] == "1t"]

    libs = _ordered_libs_present(df)
    plat_labels = _platform_labels(_cpu_by_platform(df))
    headers = ["Decoder"] + [plat_labels[p] for p in PAPER_PLATFORMS]
    rows: list[list[str]] = []
    for lib in libs:
        row = [f"`{lib}`"]
        for plat in PAPER_PLATFORMS:
            sub = df[(df["platform"] == plat) & (df["library"] == lib)]
            if sub.empty:
                row.append("—")
            else:
                row.append(
                    _fmt_mean_std(
                        sub["images_per_second"].iloc[0],
                        sub["std_dev"].iloc[0],
                    ),
                )
        rows.append(row)
    dest.write_text(
        "# Table 2 — Single-thread throughput (images/s, mean ± std)\n\n"
        "_From `*_1t_results.json` (memory decode, one thread)._\n\n" + _md_table(headers, rows),
        encoding="utf-8",
    )


def _peak_dataloader_rows(dl: pd.DataFrame) -> pd.DataFrame:
    dl = dl[dl["num_workers"].isin(WORKERS_ORDER)]
    rows: list[dict] = []
    for plat in PAPER_PLATFORMS:
        libs = dl[dl["platform"] == plat]["library"].unique()
        for lib in libs:
            sub = dl[(dl["platform"] == plat) & (dl["library"] == lib)]
            if sub.empty:
                continue
            idx = sub["images_per_second"].idxmax()
            peak = sub.loc[idx]
            rows.append(
                {
                    "platform": plat,
                    "library": lib,
                    "peak_ips": float(peak["images_per_second"]),
                    "peak_workers": int(peak["num_workers"]),
                },
            )
    return pd.DataFrame(rows)


def generate_peak_dataloader_table(dl: pd.DataFrame, dest: Path) -> None:
    dl = _paper_scope(dl)
    peak = _peak_dataloader_rows(dl)
    plat_labels = _platform_labels(_cpu_by_platform(dl))
    libs = _ordered_libs_present(dl)
    headers = ["Decoder"] + [plat_labels[p] for p in PAPER_PLATFORMS]
    rows: list[list[str]] = []
    for lib in libs:
        row = [f"`{lib}`"]
        for plat in PAPER_PLATFORMS:
            sub = peak[(peak["platform"] == plat) & (peak["library"] == lib)]
            if sub.empty:
                row.append("—")
            else:
                w = int(sub["peak_workers"].iloc[0])
                ips = float(sub["peak_ips"].iloc[0])
                row.append(f"{ips:.0f} (w={w})")
        rows.append(row)
    dest.write_text(
        "# Table 3 — Peak DataLoader throughput (images/s) and best worker count\n\n"
        "_Max over workers ∈ {0, 2, 4, 8}. Decoders without DataLoader JSON omitted._\n\n" + _md_table(headers, rows),
        encoding="utf-8",
    )


def _ips_at_workers(dl: pd.DataFrame, platform: str, lib: str, workers: int) -> float | None:
    sub = dl[(dl["platform"] == platform) & (dl["library"] == lib) & (dl["num_workers"] == workers)]
    if sub.empty:
        return None
    return float(sub["images_per_second"].iloc[0])


def generate_amd_w4_w8_table(dl: pd.DataFrame, dest: Path) -> None:
    dl = _paper_scope(dl)
    libs = [lib for lib in _ordered_libs_present(dl) if lib != "pyvips"]
    headers = ["Decoder", "Zen4 w=4", "Zen4 w=8", "Zen5 w=4", "Zen5 w=8"]
    rows: list[list[str]] = []
    for lib in libs:
        z4_4 = _ips_at_workers(dl, ZEN4_PLATFORM, lib, 4)
        z4_8 = _ips_at_workers(dl, ZEN4_PLATFORM, lib, 8)
        z5_4 = _ips_at_workers(dl, ZEN5_PLATFORM, lib, 4)
        z5_8 = _ips_at_workers(dl, ZEN5_PLATFORM, lib, 8)
        if z4_4 is None and z4_8 is None and z5_4 is None and z5_8 is None:
            continue
        rows.append(
            [
                f"`{lib}`",
                _fmt_ips(z4_4),
                _fmt_ips(z4_8),
                _fmt_ips(z5_4),
                _fmt_ips(z5_8),
            ],
        )
    dest.write_text(
        "# Table 4 — AMD Zen 4 vs Zen 5: DataLoader at w=4 and w=8 (images/s)\n\n"
        "_Zen 4 = `linux_AMD-EPYC-9B14`, Zen 5 = `linux_AMD-EPYC-9B45`._\n\n" + _md_table(headers, rows),
        encoding="utf-8",
    )


def generate_recommendation_table(dl: pd.DataFrame, dest: Path) -> None:
    norm = _peak_pct_of_platform_winner(dl)
    summary = (
        norm.groupby("library", as_index=False)["pct_of_winner"]
        .agg(mean="mean", min="min", max="max")
        .set_index("library")
    )
    recommendation_order = ["torchvision", "simplejpeg", "opencv"]
    headers = [
        "Decoder",
        "Mean % of winner",
        "Min %",
        "Max %",
        "Skipped JPEGs",
        "DataLoader platforms",
    ]
    rows: list[list[str]] = []
    for lib in recommendation_order:
        if lib not in summary.index:
            continue
        vals = summary.loc[lib]
        platform_count = int(norm[norm["library"] == lib]["platform"].nunique())
        skip_text = "1 / 50,000" if lib in EXPECTED_SKIP_DECODERS else "0 / 50,000"
        rows.append(
            [
                f"`{lib}`",
                f"{vals['mean']:.1f}%",
                f"{vals['min']:.1f}%",
                f"{vals['max']:.1f}%",
                skip_text,
                f"{platform_count} / {len(PAPER_PLATFORMS)}",
            ],
        )
    note = (
        "\n_Peak DataLoader throughput normalized to the platform-local winner. "
        "This table lists the zero-skip choices above the 90% practical floor on every paper platform._\n"
    )
    dest.write_text(
        "# Table 6 — Robust zero-skip near-optimal DataLoader choices\n\n" + note + "\n" + _md_table(headers, rows),
        encoding="utf-8",
    )


def _raw_library_name(library: str) -> str:
    return "kornia" if library == "kornia-rs" else library


def _first_skip_example(input_dir: Path, library: str) -> str:
    raw = _raw_library_name(library)
    for plat in PAPER_PLATFORMS:
        for suffix in ("1t_results", "dataloader_results"):
            path = input_dir / plat / f"{raw}_{suffix}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if suffix == "1t_results":
                examples = data.get("benchmark_results", {}).get("skip_examples", [])
            else:
                examples = data.get("skip_examples", [])
            if examples:
                return str(examples[0]).replace("|", "\\|")
    return "—"


def _skip_summary(df_1t: pd.DataFrame, library: str) -> str:
    single = _paper_scope(df_1t)
    single = single[(single["run_tag"] == "1t") & (single["library"] == library)]
    counts = single["num_images_skipped"].fillna(0).astype(int).tolist()
    if not counts or max(counts) == 0:
        return "0 / 50,000 on all five platforms"
    if counts == [1] * len(PAPER_PLATFORMS):
        return "1 / 50,000 on all five platforms"
    return ", ".join(str(c) for c in counts)


def _dataloader_eligibility(input_dir: Path, library: str) -> str:
    raw = _raw_library_name(library)
    present = [(input_dir / plat / f"{raw}_dataloader_results.json").exists() for plat in PAPER_PLATFORMS]
    if all(present):
        return "Yes"
    if not any(present) and library == "pyvips":
        return "No: disabled for forked PyTorch workers"
    if not any(present) and library == "tensorflow":
        return "No: TensorFlow stack not run inside PyTorch DataLoader"
    return f"Partial: {sum(present)} / {len(PAPER_PLATFORMS)} platforms"


def generate_robustness_table(input_dir: Path, df_1t: pd.DataFrame, dest: Path) -> None:
    headers = ["Decoder", "DataLoader eligibility", "Skipped images", "Example failure", "Interpretation"]
    interpretations = {
        "jpeg4py": "Fast path, but needs an explicit CMYK fallback policy.",
        "kornia-rs": "Fast path, but rejects the same uncommon ImageNet image.",
        "turbojpeg": "Fast path, but needs an explicit CMYK fallback policy.",
        "pyvips": "Single-thread numbers only; no loader-scale recommendation in this harness.",
        "tensorflow": "Single-thread portability warning; re-benchmark exact build and pipeline.",
    }
    rows = [
        [
            f"`{lib}`",
            _dataloader_eligibility(input_dir, lib),
            _skip_summary(df_1t, lib),
            _first_skip_example(input_dir, lib),
            interpretations[lib],
        ]
        for lib in ROBUSTNESS_DECODERS
    ]
    dest.write_text(
        "# Table 5 — Robustness and DataLoader eligibility\n\n"
        "_Generated from skip fields in `*_1t_results.json` and `*_dataloader_results.json`._\n\n"
        + _md_table(headers, rows),
        encoding="utf-8",
    )


def _lib_colors(libs: list[str]) -> dict[str, tuple]:
    pal = sns.color_palette("tab20", n_colors=max(20, len(libs)))
    return {lib: pal[i % len(pal)] for i, lib in enumerate(libs)}


def _rank_frame(single: pd.DataFrame, dl: pd.DataFrame, platform: str) -> pd.DataFrame:
    libs = sorted(EXPECTED_DATALOADER_DECODERS)
    single_sub = single[
        (single["platform"] == platform) & (single["run_tag"] == "1t") & (single["library"].isin(libs))
    ].copy()
    peaks = _peak_dataloader_rows(dl)
    peak_sub = peaks[(peaks["platform"] == platform) & (peaks["library"].isin(libs))].copy()
    single_sub["single_rank"] = single_sub["images_per_second"].rank(method="min", ascending=False).astype(int)
    peak_sub["dataloader_rank"] = peak_sub["peak_ips"].rank(method="min", ascending=False).astype(int)
    ranks = single_sub[["library", "single_rank"]].merge(
        peak_sub[["library", "dataloader_rank"]],
        on="library",
        how="inner",
    )
    ranks["rank_delta"] = ranks["single_rank"] - ranks["dataloader_rank"]
    return ranks.sort_values(["rank_delta", "library"], ascending=[True, True])


def plot_fig01_protocol_rank_change(
    single: pd.DataFrame,
    dl: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    single = _paper_scope(single)
    dl = _paper_scope(dl)
    plat_labels = _platform_labels(_cpu_by_platform(single))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 5.4), sharey=False, constrained_layout=True)

    for ax, plat in zip(axes, RANK_CHANGE_PLATFORMS, strict=True):
        ranks = _rank_frame(single, dl, plat)
        highlight = RANK_CHANGE_HIGHLIGHTS[plat]
        y = np.arange(len(ranks))
        colors = []
        for row in ranks.itertuples(index=False):
            if row.library == highlight:
                colors.append("#1f77b4")
            elif row.rank_delta > 0:
                colors.append("#2ca02c")
            elif row.rank_delta < 0:
                colors.append("#e15759")
            else:
                colors.append("0.68")

        ax.barh(y, ranks["rank_delta"], color=colors, alpha=0.9)
        ax.axvline(0, color="0.25", linewidth=1.1)
        for i, row in enumerate(ranks.itertuples(index=False)):
            if row.rank_delta == 0:
                continue
            ha = "left" if row.rank_delta > 0 else "right"
            offset = 0.18 if row.rank_delta > 0 else -0.18
            ax.text(
                row.rank_delta + offset,
                i,
                f"{row.single_rank}->{row.dataloader_rank}",
                ha=ha,
                va="center",
                fontsize=7.5,
                fontweight="bold" if row.library == highlight else "normal",
                color="#1f77b4" if row.library == highlight else "0.2",
            )
        ax.set_title(plat_labels[plat], fontsize=10, fontweight="bold")
        ax.set_xlim(-8.5, 8.5)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.set_yticks(y)
        ax.set_yticklabels(ranks["library"])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.22)
        ax.set_xlabel("Rank change")

    axes[0].set_ylabel("Decoder")
    fig.suptitle("Protocol changes the supported decoder recommendation", fontsize=13, fontweight="bold")
    fig.supxlabel(
        "single-thread rank - peak DataLoader rank (positive = moves up under DataLoader)",
        fontsize=10,
    )

    base = out_dir / CLAIM_FIGURE_BASENAMES[0]
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fig02_amd_worker_delta(dl: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> None:
    dl = _paper_scope(dl)
    libs = [lib for lib in _ordered_libs_present(dl) if lib not in {"pyvips", "tensorflow"}]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, constrained_layout=True)
    labels = {ZEN4_PLATFORM: "AMD Zen 4", ZEN5_PLATFORM: "AMD Zen 5"}
    y = np.arange(len(libs))

    for ax, plat in zip(axes, (ZEN4_PLATFORM, ZEN5_PLATFORM), strict=True):
        deltas = []
        colors = []
        for lib in libs:
            w4 = _ips_at_workers(dl, plat, lib, 4)
            w8 = _ips_at_workers(dl, plat, lib, 8)
            delta = np.nan if not w4 or w8 is None else 100.0 * (w8 / w4 - 1.0)
            deltas.append(delta)
            colors.append("#2ca02c" if delta >= 0 else "#d62728")
        ax.barh(y, deltas, color=colors, alpha=0.88)
        ax.axvline(0, color="0.25", linewidth=1.2)
        for i, delta in enumerate(deltas):
            if pd.isna(delta):
                continue
            ha = "left" if delta >= 0 else "right"
            offset = 0.6 if delta >= 0 else -0.6
            ax.text(delta + offset, i, f"{delta:+.0f}%", va="center", ha=ha, fontsize=8)
        ax.set_title(labels[plat], fontsize=10, fontweight="bold")
        ax.set_xlabel("Throughput change from w=4 to w=8")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(-18, 32)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(libs)
    axes[0].invert_yaxis()
    fig.suptitle("Worker-count scaling differs between AMD generations", fontsize=13, fontweight="bold")

    base = out_dir / CLAIM_FIGURE_BASENAMES[1]
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _peak_pct_of_platform_winner(dl: pd.DataFrame) -> pd.DataFrame:
    dl = _paper_scope(dl)
    peak = _peak_dataloader_rows(dl)
    rows: list[dict] = []
    for plat in PAPER_PLATFORMS:
        sub = peak[peak["platform"] == plat]
        if sub.empty:
            continue
        winner_ips = float(sub["peak_ips"].max())
        if pd.isna(winner_ips) or winner_ips <= 0:
            continue
        rows.extend(
            [
                {
                    "platform": plat,
                    "library": row["library"],
                    "peak_ips": float(row["peak_ips"]),
                    "peak_workers": int(row["peak_workers"]),
                    "pct_of_winner": 100.0 * float(row["peak_ips"]) / winner_ips,
                }
                for row in sub.to_dict("records")
            ],
        )
    return pd.DataFrame(rows)


def plot_fig03_tensorflow_arm_penalty(single: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> None:
    single = _paper_scope(single)
    single = single[single["run_tag"] == "1t"].copy()
    plat_labels = _platform_labels(_cpu_by_platform(single))
    rows: list[dict] = []
    for plat in PAPER_PLATFORMS:
        sub = single[single["platform"] == plat]
        tf = sub[sub["library"] == "tensorflow"]
        if sub.empty or tf.empty:
            continue
        winner = float(sub["images_per_second"].max())
        tf_ips = float(tf["images_per_second"].iloc[0])
        rows.append(
            {
                "platform": plat,
                "label": plat_labels[plat],
                "pct": 100.0 * tf_ips / winner,
                "ips": tf_ips,
            },
        )
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.3), constrained_layout=True)
    colors = ["#d62728" if "Neoverse" in row.label else "#4c78a8" for row in data.itertuples(index=False)]
    bars = ax.bar(data["label"], data["pct"], color=colors, alpha=0.9)
    ax.axhline(100, color="0.35", linewidth=1.0, linestyle="--")
    for bar, row in zip(bars, data.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{row.pct:.0f}%\n{row.ips:.0f} img/s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("TensorFlow single-thread throughput\n(% of platform winner)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("TensorFlow JPEG decode shows a large ARM penalty", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    base = out_dir / CLAIM_FIGURE_BASENAMES[2]
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fig04_cross_platform_recommendation(dl: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> None:
    norm = _peak_pct_of_platform_winner(dl)
    libs = _ordered_libs_present(norm)
    summary = (
        norm.groupby("library", as_index=False)["pct_of_winner"]
        .agg(mean="mean", min="min", max="max")
        .set_index("library")
        .reindex(libs)
        .dropna()
        .sort_values("mean", ascending=True)
    )
    platform_points = norm[norm["library"].isin(summary.index)]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    y = np.arange(len(summary))
    x_min = 70.0
    zero_skip_color = "#2b8cbe"
    skip_color = "#f2a541"
    for i, lib in enumerate(summary.index):
        has_skip = lib in EXPECTED_SKIP_DECODERS
        ax.barh(
            i,
            summary.loc[lib, "mean"] - x_min,
            left=x_min,
            height=0.58,
            color=skip_color if has_skip else zero_skip_color,
            alpha=0.74,
            hatch="//" if has_skip else None,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
    ax.errorbar(
        summary["mean"],
        y,
        xerr=[summary["mean"] - summary["min"], summary["max"] - summary["mean"]],
        fmt="none",
        ecolor="0.25",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    for i, lib in enumerate(summary.index):
        pts = platform_points[platform_points["library"] == lib]["pct_of_winner"]
        has_skip = lib in EXPECTED_SKIP_DECODERS
        ax.scatter(
            pts,
            np.full(len(pts), i),
            s=22,
            color="white",
            edgecolor=skip_color if has_skip else zero_skip_color,
            linewidth=0.8,
            alpha=0.85,
            zorder=4,
        )

    ax.axvline(90, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(90.25, len(summary) - 0.25, "90% practical floor", fontsize=7, color="0.35", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlim(x_min, 103)
    ax.set_xlabel("Peak DataLoader throughput (% of platform winner)")
    ax.set_title("DataLoader speed and observed JPEG robustness", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=zero_skip_color, edgecolor="white", label="zero observed skips"),
            Patch(facecolor=skip_color, edgecolor="white", hatch="//", label="one skipped JPEG"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    base = out_dir / CLAIM_FIGURE_BASENAMES[3]
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_tables(input_dir: Path, paper_dir: Path) -> None:
    df_1t = load_results(input_dir)
    dl = load_dataloader_results(input_dir)
    validate_paper_data(df_1t, dl)
    gen = paper_dir / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    generate_hardware_table(df_1t, gen / "table01_hardware.md")
    generate_single_thread_table(df_1t, gen / "table02_single_thread.md")
    generate_peak_dataloader_table(dl, gen / "table03_peak_dataloader.md")
    generate_amd_w4_w8_table(dl, gen / "table04_amd_w4_w8.md")
    generate_robustness_table(input_dir, df_1t, gen / "table05_robustness.md")
    generate_recommendation_table(dl, gen / "table06_recommendation_tier.md")
    (gen / "README.md").write_text(
        "# Generated paper tables\n\n"
        "- `table01_hardware.md`\n"
        "- `table02_single_thread.md`\n"
        "- `table03_peak_dataloader.md`\n"
        "- `table04_amd_w4_w8.md`\n"
        "- `table05_robustness.md`\n"
        "- `table06_recommendation_tier.md`\n\n"
        "Regenerate from repo root:\n\n"
        "```bash\n"
        "uv run --extra plot python -m tools.paper_assets --tables\n"
        "```\n",
        encoding="utf-8",
    )


def generate_figures(input_dir: Path, paper_dir: Path, formats: tuple[str, ...]) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    df_1t = load_results(input_dir)
    dl = load_dataloader_results(input_dir)
    validate_paper_data(df_1t, dl)
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_fig01_protocol_rank_change(df_1t, dl, fig_dir, formats)
    plot_fig02_amd_worker_delta(dl, fig_dir, formats)
    plot_fig03_tensorflow_arm_penalty(df_1t, fig_dir, formats)
    plot_fig04_cross_platform_recommendation(dl, fig_dir, formats)


def _parse_formats(s: str) -> tuple[str, ...]:
    parts = tuple(p.strip().lower() for p in s.split(",") if p.strip())
    return parts or ("png",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures from benchmark JSON.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output"),
        help="Results directory relative to repo root (default: output).",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("_internal/papers"),
        help="Paper directory relative to repo root (default: _internal/papers).",
    )
    parser.add_argument("--tables", action="store_true", help="Write Markdown tables under paper-dir/generated/.")
    parser.add_argument("--figures", action="store_true", help="Write figures under paper-dir/figures/.")
    parser.add_argument("--all", action="store_true", help="Tables + figures.")
    parser.add_argument("--check", action="store_true", help="Validate the paper result matrix without writing files.")
    parser.add_argument(
        "--format",
        type=str,
        default="png,pdf",
        help="Figure formats, comma-separated (default: png,pdf).",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    input_dir = (root / args.input).resolve()
    paper_dir = (root / args.paper_dir).resolve()
    fmt = _parse_formats(args.format)

    if args.check:
        summary = validate_paper_data(load_results(input_dir), load_dataloader_results(input_dir))
        print(
            "paper data ok: "
            f"{summary['platforms']} platforms, "
            f"{summary['single_thread_decoders']} single-thread decoders, "
            f"{summary['dataloader_decoders']} DataLoader decoders, "
            f"{summary['dataloader_worker_rows']} DataLoader worker rows",
        )
        return

    do_tables = args.tables or args.all
    do_figures = args.figures or args.all
    if not do_tables and not do_figures:
        parser.error("Specify --tables, --figures, or --all")

    if do_tables:
        generate_tables(input_dir, paper_dir)
    if do_figures:
        generate_figures(input_dir, paper_dir, fmt)


if __name__ == "__main__":
    main()

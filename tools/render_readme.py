"""
Regenerate the data tables in README.md from output/<platform>/*.json.

Three blocks are rewritten in place between sentinel HTML comments — anything
outside the markers is left alone.

    <!-- BENCH:single_thread:start -->
    ... auto-generated table ...
    <!-- BENCH:single_thread:end -->

Run via the CLI:
    imread-benchmark render-readme --input output --readme README.md

Or directly (from the repo root):
    python -m tools.render_readme --input output --readme README.md [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from tools._results import (
    LIBRARY_ORDER,
    load_dataloader_results,
    load_results,
    short_platform,
)

MISSING = "—"


def _platform_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return ordered (platform_dirname, short_label) tuples."""
    cols: list[tuple[str, str]] = []
    for platform in sorted(df["platform"].unique()):
        cpu = df[df["platform"] == platform]["cpu_brand"].iloc[0]
        cols.append((platform, short_platform(platform, cpu)))
    return cols


def _markdown_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Build a GitHub-flavored markdown table."""
    if align is None:
        align = ["left"] + ["right"] * (len(headers) - 1)
    sep = []
    for a in align:
        if a == "right":
            sep.append("---:")
        elif a == "center":
            sep.append(":---:")
        else:
            sep.append(":---")
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _fmt_ips(value: float | None) -> str:
    if value is None or pd.isna(value):
        return MISSING
    return f"{value:,.0f}"


def render_single_thread_table(df: pd.DataFrame) -> str:
    """Build the rows = library, cols = platform single-thread leaderboard."""
    df1t = df[df["run_tag"] == "1t"]
    if df1t.empty:
        return "_No single-thread results found._"

    cols = _platform_columns(df1t)
    pivot = df1t.pivot_table(
        index="library",
        columns="platform",
        values="images_per_second",
        aggfunc="max",
    )

    libraries = [lib for lib in LIBRARY_ORDER if lib in pivot.index]
    headers = ["Library", *(short for _, short in cols)]
    rows: list[list[str]] = []
    best_per_col = {p: pivot[p].max() for p, _ in cols if p in pivot.columns}

    for lib in libraries:
        row = [f"`{lib}`"]
        for platform, _ in cols:
            value = pivot.get(platform, pd.Series()).get(lib) if platform in pivot.columns else None
            cell = _fmt_ips(value)
            if value is not None and not pd.isna(value) and value == best_per_col.get(platform):
                cell = f"**{cell}**"
            row.append(cell)
        rows.append(row)

    return _markdown_table(headers, rows)


def render_dataloader_table(ddf: pd.DataFrame) -> str:
    """Peak DataLoader throughput per (library, platform) — value @ best worker count."""
    if ddf.empty:
        return "_No DataLoader results found._"

    idx = ddf.groupby(["platform", "library"])["images_per_second"].idxmax()
    peaks = ddf.loc[idx]
    cols = _platform_columns(ddf)

    headers = ["Library", *(short for _, short in cols)]
    rows: list[list[str]] = []
    best_per_col: dict[str, float] = {}
    for platform, _ in cols:
        sub = peaks[peaks["platform"] == platform]
        if not sub.empty:
            best_per_col[platform] = sub["images_per_second"].max()

    libraries = [lib for lib in LIBRARY_ORDER if lib in peaks["library"].unique()]
    for lib in libraries:
        row = [f"`{lib}`"]
        for platform, _ in cols:
            cell_df = peaks[(peaks["platform"] == platform) & (peaks["library"] == lib)]
            if cell_df.empty:
                row.append(MISSING)
                continue
            ips = cell_df["images_per_second"].iloc[0]
            workers = int(cell_df["num_workers"].iloc[0])
            cell = f"{_fmt_ips(ips)} @ {workers}w"
            if ips == best_per_col.get(platform):
                cell = f"**{cell}**"
            row.append(cell)
        rows.append(row)

    return _markdown_table(headers, rows)


def render_metadata(input_dir: Path, df: pd.DataFrame) -> str:
    """One-liner: platform count, dataset size, run count, latest run timestamp."""
    summaries: list[dict] = []
    for platform_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        summary_file = platform_dir / "run_summary.json"
        if summary_file.exists():
            with summary_file.open() as f:
                summaries.append(json.load(f))

    n_platforms = df["platform"].nunique()
    num_images = int(df["num_images"].dropna().max()) if not df["num_images"].dropna().empty else None
    num_runs = int(df["num_runs"].dropna().max()) if not df["num_runs"].dropna().empty else None

    timestamps = [s["timestamp_utc"] for s in summaries if s.get("timestamp_utc")]
    latest = max(timestamps) if timestamps else None
    latest_date = latest.split("T")[0] if latest else "unknown"

    parts = [f"**{n_platforms} platforms**"]
    if num_images is not None:
        parts.append(f"{num_images:,} images")
    if num_runs is not None:
        parts.append(f"{num_runs} runs each")
    parts.append(f"latest run {latest_date}")
    return "_" + " · ".join(parts) + "_"


def update_readme(readme_path: Path, blocks: dict[str, str]) -> str:
    """Rewrite each `<!-- BENCH:KEY:start -->...<!-- BENCH:KEY:end -->` block."""
    text = readme_path.read_text()
    for key, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- BENCH:{re.escape(key)}:start -->)(.*?)(<!-- BENCH:{re.escape(key)}:end -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(
                f"Marker pair for '{key}' not found in {readme_path}. "
                f"Add `<!-- BENCH:{key}:start -->` and `<!-- BENCH:{key}:end -->`.",
            )
        text = pattern.sub(rf"\1\n\n{body}\n\n\3", text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate README.md benchmark tables.")
    parser.add_argument("--input", type=Path, default=Path("output"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if README would change (CI / pre-commit guard).",
    )
    args = parser.parse_args()

    df = load_results(args.input)
    if df.empty:
        print(f"No results found under {args.input}/<platform>/*_results.json", file=sys.stderr)
        return 1
    ddf = load_dataloader_results(args.input)

    blocks = {
        "single_thread": render_single_thread_table(df),
        "dataloader": render_dataloader_table(ddf),
        "metadata": render_metadata(args.input, df),
    }
    new_text = update_readme(args.readme, blocks)
    old_text = args.readme.read_text()

    if new_text == old_text:
        print(f"{args.readme} already up to date.")
        return 0

    if args.check:
        print(f"{args.readme} is stale. Run: python tools/render_readme.py", file=sys.stderr)
        return 1

    args.readme.write_text(new_text)
    print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

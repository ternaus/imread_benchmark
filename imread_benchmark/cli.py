"""
imread-benchmark CLI — orchestrator that owns venv setup + benchmark dispatch.

Replaces run_benchmarks.sh / run_dataloader_benchmarks.sh. Lives in the
"control plane" venv (anywhere with this package installed); shells out to
per-group worker venvs under venvs/<group>/ for the actual decode work.
This split is required because mainstream / tensorflow cannot coexist
in one Python process (numpy / protobuf pin fights).

Usage:
    imread-benchmark list-libs
    imread-benchmark run --data-dir ~/imagenet/val
    imread-benchmark run --data-dir DATA --mode single --libs opencv,pillow
    imread-benchmark plot --input output --output _internal/plots
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from imread_benchmark.decoders import REGISTRY, BaseDecoder
from imread_benchmark.utils import get_system_identifier

app = typer.Typer(add_completion=False, help="JPEG decoder benchmark orchestrator.")

VENV_ROOT = Path("venvs")
DEFAULT_WORKERS = "0,1,2,4,8"


# ─── lib / group helpers ──────────────────────────────────────────────────────


def _all_lib_names() -> list[str]:
    return sorted(REGISTRY.keys())


def _resolve_libs(libs: str) -> list[str]:
    if libs.strip() in {"", "all"}:
        return _all_lib_names()
    requested = [s.strip() for s in libs.split(",") if s.strip()]
    unknown = [r for r in requested if r not in REGISTRY]
    if unknown:
        typer.echo(f"Unknown library names: {', '.join(unknown)}", err=True)
        typer.echo(f"Known: {', '.join(_all_lib_names())}", err=True)
        raise typer.Exit(2)
    return requested


def _group_of(name: str) -> str:
    return REGISTRY[name].group


def _runs_single(name: str) -> bool:
    return REGISTRY[name].runs_single_here()


def _runs_dataloader(name: str) -> bool:
    return REGISTRY[name].runs_dataloader_here()


# ─── venv management ──────────────────────────────────────────────────────────


def _venv_python(group: str) -> Path:
    """Path to the python interpreter inside a per-group venv."""
    bindir = "Scripts" if platform.system() == "Windows" else "bin"
    suffix = ".exe" if platform.system() == "Windows" else ""
    return VENV_ROOT / group / bindir / f"python{suffix}"


def _ensure_venv(group: str, *, force_reinstall: bool = False) -> Path:
    """
    Create venvs/<group>/ and install .[group] into it.
    Idempotent: if the venv + project are already there, returns the python path immediately.
    """
    if shutil.which("uv") is None:
        typer.echo("FATAL: `uv` not found on PATH. Install it: https://astral.sh/uv", err=True)
        raise typer.Exit(127)

    py = _venv_python(group)
    venv_dir = VENV_ROOT / group
    needs_install = force_reinstall or not py.exists()

    if not py.exists():
        typer.echo(f"[venv] Creating venvs/{group}/ ...")
        subprocess.run(
            ["uv", "venv", str(venv_dir), "--python", "python3", "--seed"],
            check=True,
        )

    if needs_install:
        typer.echo(f"[venv] Installing .[{group}] into venvs/{group}/ ...")
        env = {**os.environ, "UV_LINK_MODE": "copy"}
        subprocess.run(
            ["uv", "pip", "install", "--python", str(py), "-e", f".[{group}]"],
            check=True,
            env=env,
        )
    else:
        typer.echo(f"[venv] Reusing venvs/{group}/")
    return py


# ─── worker invocation ────────────────────────────────────────────────────────


def _run_worker(py: Path, lib: str, module: str, args: list[str]) -> int:
    """Execute a benchmark worker module in a per-group venv subprocess."""
    proc = subprocess.run([str(py), "-m", module, "--library", lib, *args], check=False)
    return proc.returncode


def _query_default_threads(py: Path, lib: str) -> int:
    """Ask the decoder in its venv how many threads it uses by default."""
    code = f"from imread_benchmark.decoders import REGISTRY; print(REGISTRY['{lib}']().get_num_threads())"
    proc = subprocess.run([str(py), "-c", code], check=True, capture_output=True, text=True)
    return int(proc.stdout.strip())


def _run_single_for_lib(
    py: Path,
    lib: str,
    *,
    data_dir: Path,
    output_dir: Path,
    num_images: int,
    num_runs: int,
    mode: str,
) -> bool:
    """Run single-thread + (optionally) library-default-thread benchmarks. Returns success."""
    typer.echo(f"  [single, 1 thread] {lib}")
    rc = _run_worker(
        py,
        lib,
        "imread_benchmark.benchmark_single",
        [
            "--data-dir",
            str(data_dir),
            "--num-images",
            str(num_images),
            "--num-runs",
            str(num_runs),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--num-threads",
            "1",
        ],
    )
    if rc != 0:
        typer.echo(f"  WARN: 1-thread run failed for {lib} (exit {rc})", err=True)
        return False

    default_threads = _query_default_threads(py, lib)
    if default_threads <= 1:
        typer.echo("  [default = 1 thread, no second pass needed]")
        return True

    # If the library ignored set_num_threads(1) (e.g. opencv-on-macOS uses GCD),
    # the 1-thread pass actually ran with `effective_threads != 1`. If that
    # number already equals what the default would do, the second pass is a
    # duplicate — skip it.
    one_t_json = output_dir / get_system_identifier() / f"{lib}_1t_results.json"
    try:
        first_eff = int(json.loads(one_t_json.read_text())["effective_threads"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        first_eff = 1  # be safe, run the second pass

    if first_eff == default_threads:
        typer.echo(
            f"  [skip default pass] {lib} ignores set_num_threads "
            f"(1-thread request → ran with {first_eff}, default also = {default_threads})",
        )
        return True

    typer.echo(f"  [single, default = {default_threads} threads] {lib}")
    rc = _run_worker(
        py,
        lib,
        "imread_benchmark.benchmark_single",
        [
            "--data-dir",
            str(data_dir),
            "--num-images",
            str(num_images),
            "--num-runs",
            str(num_runs),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--num-threads",
            "0",
        ],
    )
    if rc != 0:
        typer.echo(f"  WARN: default-thread run failed for {lib} (exit {rc})", err=True)
        return False
    return True


def _run_dataloader_for_lib(
    py: Path,
    lib: str,
    *,
    data_dir: Path,
    output_dir: Path,
    num_images: int,
    num_runs: int,
    workers: list[int],
) -> bool:
    typer.echo(f"  [dataloader] {lib}  workers={workers}")
    rc = _run_worker(
        py,
        lib,
        "imread_benchmark.benchmark_dataloader",
        [
            "--data-dir",
            str(data_dir),
            "--num-images",
            str(num_images),
            "--num-runs",
            str(num_runs),
            "--output-dir",
            str(output_dir),
            "--workers",
            *(str(w) for w in workers),
        ],
    )
    if rc != 0:
        typer.echo(f"  WARN: dataloader run failed for {lib} (exit {rc})", err=True)
    return rc == 0


# ─── commands ─────────────────────────────────────────────────────────────────


@app.command("list-libs")
def list_libs() -> None:
    """List all known decoders and whether they would run on this machine."""
    sys_str, mach_str = platform.system(), platform.machine()
    typer.echo(f"Platform: {sys_str} {mach_str}\n")
    typer.echo(f"{'name':<14} {'group':<12} {'single':<7} {'dataloader':<10}")
    typer.echo("-" * 50)
    for name in _all_lib_names():
        cls: type[BaseDecoder] = REGISTRY[name]
        s = "yes" if cls.runs_single_here() else "skip"
        d = "yes" if cls.runs_dataloader_here() else "skip"
        typer.echo(f"{name:<14} {cls.group:<12} {s:<7} {d:<10}")


@app.command("run")
def run(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", exists=True, help="Directory of JPEG images"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Where JSON results go"),
    libs: str = typer.Option("all", "--libs", help="Comma-separated lib names, or 'all'"),
    mode: str = typer.Option("both", "--mode", help="single | dataloader | both"),
    num_images: int = typer.Option(50000, "--num-images", "-n"),
    num_runs: int = typer.Option(20, "--num-runs", "-r", help="Timed runs for the single-thread benchmark"),
    dataloader_runs: int = typer.Option(5, "--dataloader-runs", help="Timed runs per worker count"),
    workers: str = typer.Option(DEFAULT_WORKERS, "--workers", help="Comma-separated num_workers values"),
    decode_mode: str = typer.Option("memory", "--decode-mode", help="memory | disk (single benchmark only)"),
    skip_setup: bool = typer.Option(False, "--skip-setup", help="Assume venvs/ already populated"),
) -> None:
    """Benchmark JPEG decoders. Sets up per-group venvs as needed."""
    if mode not in {"single", "dataloader", "both"}:
        typer.echo(f"--mode must be single|dataloader|both, got {mode!r}", err=True)
        raise typer.Exit(2)

    requested = _resolve_libs(libs)
    worker_counts = [int(w) for w in workers.split(",") if w.strip()]

    # Filter by platform predicates per benchmark type.
    do_single = mode in {"single", "both"}
    do_dl = mode in {"dataloader", "both"}

    single_libs = [name for name in requested if do_single and _runs_single(name)]
    dl_libs = [name for name in requested if do_dl and _runs_dataloader(name)]
    all_libs = sorted(set(single_libs) | set(dl_libs))
    skipped = sorted(set(requested) - set(all_libs))

    typer.echo("─" * 60)
    typer.echo(f"Data dir   : {data_dir}")
    typer.echo(f"Output dir : {output_dir}")
    typer.echo(f"Mode       : {mode}")
    typer.echo(f"Images     : {num_images}")
    typer.echo(f"Runs       : single={num_runs}  dataloader={dataloader_runs}")
    typer.echo(f"Workers    : {worker_counts}")
    typer.echo(f"Libs (run) : {', '.join(all_libs) or '(none)'}")
    if skipped:
        typer.echo(f"Skipped    : {', '.join(skipped)}  (platform-incompatible for chosen mode)")
    typer.echo("─" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the venv for each group we'll use.
    needed_groups = sorted({_group_of(name) for name in all_libs})
    venv_python: dict[str, Path] = {}
    for g in needed_groups:
        venv_python[g] = _ensure_venv(g, force_reinstall=False) if not skip_setup else _venv_python(g)

    failures: list[str] = []

    # Iterate libs grouped by their venv to maximise locality (helps log scanning).
    for group in needed_groups:
        py = venv_python[group]
        libs_in_group = [name for name in all_libs if _group_of(name) == group]

        for lib in libs_in_group:
            typer.echo(f"\n=== {lib} (group={group}) ===")
            if lib in single_libs and not _run_single_for_lib(
                py,
                lib,
                data_dir=data_dir,
                output_dir=output_dir,
                num_images=num_images,
                num_runs=num_runs,
                mode=decode_mode,
            ):
                failures.append(f"{lib}/single")
            if lib in dl_libs and not _run_dataloader_for_lib(
                py,
                lib,
                data_dir=data_dir,
                output_dir=output_dir,
                num_images=num_images,
                num_runs=dataloader_runs,
                workers=worker_counts,
            ):
                failures.append(f"{lib}/dataloader")

    typer.echo("\n" + "─" * 60)
    if failures:
        typer.echo(f"Done with {len(failures)} per-decoder failure(s) — other decoders' results are intact:")
        for f in failures:
            typer.echo(f"  - {f}")
    else:
        typer.echo("All benchmarks completed successfully.")

    # Persist a machine-readable summary so the cloud orchestrator can tell
    # "all decoders finished cleanly" from "9 of 12 finished, 3 failed but
    # their JSONs are missing/partial". Without this the only signal is the
    # CLI exit code, which we deliberately keep at 0 for partial failures so
    # vm_startup.sh writes DONE (not FAILED) and we don't lose the 9 clean
    # decoders' work because turbojpeg crashed on a CMYK image.
    summary_path = output_dir / get_system_identifier() / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "system": get_system_identifier(),
        "mode": mode,
        "num_images": num_images,
        "num_runs": num_runs,
        "dataloader_runs": dataloader_runs,
        "workers": worker_counts,
        "libs_requested": requested,
        "libs_skipped_platform": skipped,
        "libs_run": all_libs,
        "failures": failures,
        "exit_status": "ok" if not failures else "partial",
    }
    with summary_path.open("w") as fh:
        json.dump(summary_payload, fh, indent=2)
    typer.echo(f"Run summary: {summary_path}")


@app.command("plot")
def plot(
    input_dir: Path = typer.Option(Path("output"), "--input", "-i", exists=True),
    output_dir: Path = typer.Option(Path("_internal/plots"), "--output", "-o"),
) -> None:
    """Generate paper-quality plots from output/ JSONs. Wraps tools/create_plots.py."""
    import importlib.util

    if any(importlib.util.find_spec(pkg) is None for pkg in ("matplotlib", "seaborn")):
        typer.echo(
            "Plotting requires matplotlib + seaborn. Install with:\n    uv pip install -e '.[plot]'",
            err=True,
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "tools.create_plots", "--input", str(input_dir), "--output", str(output_dir)]
    typer.echo("$ " + " ".join(cmd))
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


@app.command("render-readme")
def render_readme(
    input_dir: Path = typer.Option(Path("output"), "--input", "-i", exists=True),
    readme: Path = typer.Option(Path("README.md"), "--readme", "-r", exists=True),
    check: bool = typer.Option(False, "--check", help="Exit 1 if README would change."),
) -> None:
    """Regenerate the BENCH:* tables in README.md from output/ JSONs."""
    cmd = [sys.executable, "-m", "tools.render_readme", "--input", str(input_dir), "--readme", str(readme)]
    if check:
        cmd.append("--check")
    typer.echo("$ " + " ".join(cmd))
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    app()

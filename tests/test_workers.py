"""
Worker-output schema tests.

`benchmark_single` and `benchmark_dataloader` write JSON files that downstream
plotting / paper tables read by key. A regression that drops or renames a key
silently breaks every downstream consumer. These tests assert the documented
contract:

  - filename pattern (`{lib}_1t_results.json`, `{lib}_default_results.json`,
    `{lib}_dataloader_results.json`)
  - top-level keys (library, run_tag, requested_threads, effective_threads, ...)
  - per-config DataLoader entries written incrementally (one per num_workers)

DataLoader is unit-tested with `num_workers=0` only — multiprocessing in CI
runners is flaky, and the per-config save logic doesn't depend on workers > 0.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

pillow_available = importlib.util.find_spec("PIL") is not None
torch_available = importlib.util.find_spec("torch") is not None

REQUIRED_SINGLE_KEYS = {
    "library",
    "mode",
    "run_tag",
    "requested_threads",
    "effective_threads",
    "system_info",
    "benchmark_results",
    "num_images",
    "num_runs",
}

REQUIRED_BENCHMARK_RESULTS_KEYS = {
    "images_per_second_mean",
    "images_per_second_std",
    "images_per_second_p50",
    "images_per_second_p90",
    "images_per_second_p99",
    "us_per_image_mean",
    "us_per_image_p50",
    "us_per_image_p90",
    "us_per_image_p99",
    "raw_times_s",
    "raw_throughput_ips",
    # Skip-on-bad-image schema. Without these the cloud orchestrator can't tell
    # "all 50k decoded cleanly" from "49997 decoded, 3 skipped (CMYK)" — and
    # losing that distinction in the paper would be a methodological lie.
    "num_images_total",
    "num_images_decoded",
    "num_images_skipped",
    "skip_rate",
    "skip_indices",
    "skip_examples",
}

REQUIRED_DL_WORKER_KEYS = {
    "num_workers",
    "num_warmup",
    "images_per_second_mean",
    "images_per_second_p50",
    "us_per_image_mean",
    "raw_times_s",
}


@pytest.mark.skipif(not pillow_available, reason="pillow not installed")
def test_single_worker_writes_full_schema(tmp_path, jpeg_dir):
    """Run benchmark_single via subprocess (matches how cli.py invokes it) and check JSON."""
    out = tmp_path / "results"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "imread_benchmark.benchmark_single",
            "--library",
            "pillow",
            "--data-dir",
            str(jpeg_dir),
            "--num-images",
            "4",
            "--num-runs",
            "2",
            "--output-dir",
            str(out),
            "--num-threads",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"worker failed: {proc.stderr}"

    files = list(out.rglob("pillow_1t_results.json"))
    assert len(files) == 1, f"expected one pillow_1t_results.json, got {files}"
    j = json.loads(files[0].read_text())

    missing = REQUIRED_SINGLE_KEYS - j.keys()
    assert not missing, f"missing top-level keys: {missing}"

    assert j["library"] == "pillow"
    assert j["run_tag"] == "1t"
    assert j["requested_threads"] == 1
    assert j["effective_threads"] >= 1
    assert j["num_images"] == 4
    assert j["num_runs"] == 2

    br_missing = REQUIRED_BENCHMARK_RESULTS_KEYS - j["benchmark_results"].keys()
    assert not br_missing, f"missing benchmark_results keys: {br_missing}"
    assert len(j["benchmark_results"]["raw_times_s"]) == 2


@pytest.mark.skipif(not pillow_available, reason="pillow not installed")
def test_single_worker_default_threads_filename(tmp_path, jpeg_dir):
    """--num-threads 0 must write `{lib}_default_results.json`, never collide with 1t."""
    out = tmp_path / "results"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "imread_benchmark.benchmark_single",
            "--library",
            "pillow",
            "--data-dir",
            str(jpeg_dir),
            "--num-images",
            "4",
            "--num-runs",
            "1",
            "--output-dir",
            str(out),
            "--num-threads",
            "0",
        ],
        check=False,
    )
    assert proc.returncode == 0
    files = sorted(p.name for p in out.rglob("*.json"))
    assert files == ["pillow_default_results.json"], f"unexpected files: {files}"


@pytest.mark.skipif(not (pillow_available and torch_available), reason="dataloader test needs pillow + torch")
def test_dataloader_incremental_save(tmp_path, jpeg_dir):
    """JSON must be flushed after every num_workers config, not just at the end."""
    out = tmp_path / "results"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "imread_benchmark.benchmark_dataloader",
            "--library",
            "pillow",
            "--data-dir",
            str(jpeg_dir),
            "--num-images",
            "4",
            "--num-runs",
            "1",
            "--output-dir",
            str(out),
            "--workers",
            "0",  # in-process only — avoids fork/spawn flake in CI
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"worker failed: {proc.stderr}"

    files = list(out.rglob("pillow_dataloader_results.json"))
    assert len(files) == 1, f"expected one dataloader json, got {files}"
    j = json.loads(files[0].read_text())

    assert j["library"] == "pillow"
    assert j["benchmark_type"] == "dataloader"
    assert isinstance(j["worker_results"], list)
    assert len(j["worker_results"]) == 1

    wr = j["worker_results"][0]
    missing = REQUIRED_DL_WORKER_KEYS - wr.keys()
    assert not missing, f"missing worker_results keys: {missing}"
    assert wr["num_workers"] == 0
    assert wr["num_warmup"] >= 1

    # Skip-rate fields must surface in the dataloader JSON too — Pillow on
    # synthetic clean fixtures should hit zero skips. If a future change makes
    # the dataloader path silently swallow skips, this catches it.
    for k in (
        "num_images_total",
        "num_images_decoded",
        "num_images_skipped",
        "skip_rate",
        "skip_indices",
        "skip_examples",
    ):
        assert k in j, f"dataloader json missing key: {k}"
    assert j["num_images_skipped"] == 0
    assert j["num_images_decoded"] == j["num_images_total"]
    assert j["skip_rate"] == 0.0


@pytest.mark.skipif(not pillow_available, reason="pillow not installed")
def test_cli_writes_run_summary(tmp_path, jpeg_dir):
    """
    `imread-benchmark run` must write run_summary.json next to the per-decoder
    JSONs, with `exit_status` reflecting whether any decoders failed.
    vm_startup.sh's "DONE vs FAILED" decision relies on this — a missing or
    malformed summary would silently regress us to "any decoder hiccup nukes
    the whole 4-hour run".
    """
    # The CLI shells out to venvs/mainstream/bin/python via --skip-setup. If
    # that doesn't exist in the dev/CI checkout, the subprocess would fail with
    # FileNotFoundError before producing the summary — so the test isn't
    # exercising the actual run_summary.json codepath. Skip cleanly instead of
    # logging a misleading red.
    import pathlib

    if not pathlib.Path("venvs/mainstream/bin/python").exists():
        pytest.skip("venvs/mainstream/bin/python missing — run `imread-benchmark run` once locally first")

    out = tmp_path / "results"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "imread_benchmark.cli",
            "run",
            "--data-dir",
            str(jpeg_dir),
            "--output-dir",
            str(out),
            "--libs",
            "pillow",
            "--mode",
            "single",
            "--num-images",
            "4",
            "--num-runs",
            "1",
            "--skip-setup",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # Even on partial success this must exit 0 so vm_startup.sh writes DONE.
    assert proc.returncode == 0, f"cli failed: {proc.stderr}"

    summaries = list(out.rglob("run_summary.json"))
    assert len(summaries) == 1, f"expected one run_summary.json, got {summaries}"
    s = json.loads(summaries[0].read_text())

    for k in (
        "timestamp_utc",
        "system",
        "mode",
        "num_images",
        "libs_requested",
        "libs_run",
        "failures",
        "exit_status",
    ):
        assert k in s, f"run_summary.json missing key: {k}"
    assert s["exit_status"] == "ok"
    assert s["failures"] == []
    assert s["libs_run"] == ["pillow"]


@pytest.mark.skipif(not pillow_available, reason="pillow not installed")
def test_unknown_library_exits_nonzero(tmp_path, jpeg_dir):
    """Bad --library must fail loudly, not write a garbage JSON."""
    out = tmp_path / "results"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "imread_benchmark.benchmark_single",
            "--library",
            "definitely_not_a_real_decoder",
            "--data-dir",
            str(jpeg_dir),
            "--num-images",
            "4",
            "--num-runs",
            "1",
            "--output-dir",
            str(out),
        ],
        check=False,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert not list(out.rglob("*.json"))

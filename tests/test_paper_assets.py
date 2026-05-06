from __future__ import annotations

from pathlib import Path

import pytest

from tools._results import load_dataloader_results, load_results
from tools.paper_assets import generate_robustness_table, validate_paper_data


def _repo_output() -> Path:
    root = Path("output")
    if not root.exists():
        pytest.skip("paper benchmark output/ directory is not present")
    return root


def test_paper_data_matrix_matches_claimed_scope() -> None:
    root = _repo_output()
    summary = validate_paper_data(load_results(root), load_dataloader_results(root))

    assert summary["platforms"] == 5
    assert summary["single_thread_decoders"] == 12
    assert summary["dataloader_decoders"] == 10
    assert summary["single_thread_rows"] == 60
    assert summary["dataloader_worker_rows"] == 200


def test_robustness_table_surfaces_only_observed_skip_decoders(tmp_path: Path) -> None:
    root = _repo_output()
    dest = tmp_path / "table06_robustness.md"
    generate_robustness_table(root, load_results(root), dest)

    text = dest.read_text(encoding="utf-8")
    assert "`jpeg4py`" in text
    assert "`kornia-rs`" in text
    assert "`turbojpeg`" in text
    assert "`pyvips`" in text
    assert "`tensorflow`" in text
    assert text.count("1 / 50,000 on all five platforms") == 3
    assert "idx=19876" in text
    assert "Unsupported color conversion request" in text

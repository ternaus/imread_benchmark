from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools._results import load_dataloader_results, load_results
from tools.paper_assets import (
    EXPECTED_DATALOADER_DECODERS,
    EXPECTED_SINGLE_DECODERS,
    EXPECTED_SKIP_DECODERS,
    PAPER_PLATFORMS,
    ROBUSTNESS_DECODERS,
    WORKERS_ORDER,
    generate_robustness_table,
    validate_paper_data,
)


def _repo_output() -> Path:
    root = Path("output")
    if not root.exists():
        pytest.skip("paper benchmark output/ directory is not present")
    return root


def _valid_single_thread_frame() -> pd.DataFrame:
    rows = []
    for platform in PAPER_PLATFORMS:
        rows.extend(
            {
                "platform": platform,
                "library": library,
                "run_tag": "1t",
                "num_images_skipped": 1 if library in EXPECTED_SKIP_DECODERS else 0,
            }
            for library in EXPECTED_SINGLE_DECODERS
        )
    return pd.DataFrame(rows)


def _valid_dataloader_frame() -> pd.DataFrame:
    rows = []
    for platform in PAPER_PLATFORMS:
        for library in EXPECTED_DATALOADER_DECODERS:
            rows.extend(
                {
                    "platform": platform,
                    "library": library,
                    "num_workers": worker_count,
                }
                for worker_count in WORKERS_ORDER
            )
    return pd.DataFrame(rows)


def test_paper_data_matrix_matches_claimed_scope() -> None:
    root = _repo_output()
    summary = validate_paper_data(load_results(root), load_dataloader_results(root))

    assert summary["platforms"] == 5
    assert summary["single_thread_decoders"] == 12
    assert summary["dataloader_decoders"] == 10
    assert summary["single_thread_rows"] == 60
    assert summary["dataloader_worker_rows"] == 200


def test_validate_paper_data_missing_platform_raises() -> None:
    single = _valid_single_thread_frame()
    dl = _valid_dataloader_frame()
    missing_platform = PAPER_PLATFORMS[0]

    with pytest.raises(ValueError, match="DataLoader platform mismatch"):
        validate_paper_data(single, dl[dl["platform"] != missing_platform])


def test_validate_paper_data_missing_single_thread_decoder_raises() -> None:
    single = _valid_single_thread_frame()
    dl = _valid_dataloader_frame()
    missing_library = next(iter(EXPECTED_SINGLE_DECODERS))

    with pytest.raises(ValueError, match="single-thread decoder mismatch"):
        validate_paper_data(single[single["library"] != missing_library], dl)


def test_validate_paper_data_extra_unknown_decoder_raises() -> None:
    single = _valid_single_thread_frame()
    dl = _valid_dataloader_frame()
    extra_row = {
        "platform": PAPER_PLATFORMS[0],
        "library": "unknown-decoder",
        "num_workers": WORKERS_ORDER[0],
    }
    dl = pd.concat([dl, pd.DataFrame([extra_row])], ignore_index=True)

    with pytest.raises(ValueError, match="DataLoader decoder mismatch"):
        validate_paper_data(single, dl)


def test_validate_paper_data_incorrect_num_workers_raises() -> None:
    single = _valid_single_thread_frame()
    dl = _valid_dataloader_frame()
    library = next(iter(EXPECTED_DATALOADER_DECODERS))
    row_mask = (dl["platform"] == PAPER_PLATFORMS[0]) & (dl["library"] == library)
    row_mask &= dl["num_workers"] == WORKERS_ORDER[0]
    dl.loc[row_mask, "num_workers"] = 999

    with pytest.raises(ValueError, match="worker mismatch"):
        validate_paper_data(single, dl)


def test_validate_paper_data_incorrect_num_images_skipped_raises() -> None:
    single = _valid_single_thread_frame()
    dl = _valid_dataloader_frame()
    skipped_library = next(iter(EXPECTED_SKIP_DECODERS))
    row_mask = (single["platform"] == PAPER_PLATFORMS[0]) & (single["library"] == skipped_library)
    single.loc[row_mask, "num_images_skipped"] = 2

    with pytest.raises(ValueError, match="skip counts must be 1"):
        validate_paper_data(single, dl)


def test_robustness_table_surfaces_only_observed_skip_decoders(tmp_path: Path) -> None:
    root = _repo_output()
    dest = tmp_path / "table06_robustness.md"
    generate_robustness_table(root, load_results(root), dest)

    text = dest.read_text(encoding="utf-8")
    lines = text.splitlines()
    for decoder in ROBUSTNESS_DECODERS:
        decoder_label = f"`{decoder}`"
        matching_line = next((line for line in lines if decoder_label in line), None)
        assert matching_line is not None, f"Decoder {decoder!r} not found in robustness table."

        expected_prefix = "1 / 50,000" if decoder in EXPECTED_SKIP_DECODERS else "0 / 50,000"
        assert expected_prefix in matching_line

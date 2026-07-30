from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imread_benchmark.analysis.canonical import RunBundleRecord


class ClaimScope(StrEnum):
    DECODER_CAPACITY = "decoder-capacity"
    LOADER_SUPPLY = "loader-supply"
    TRAINING = "training"


def assert_claim_scope(records: Sequence[RunBundleRecord], scope: ClaimScope) -> None:
    if not records:
        raise ValueError("claim evidence set must not be empty")
    protocols = {record.config.get("protocol_id") for record in records}
    if scope is ClaimScope.TRAINING:
        raise ValueError("training claims require end-to-end training evidence; decoder bundles are insufficient")
    if scope is ClaimScope.DECODER_CAPACITY and protocols != {"decode-memory"}:
        raise ValueError("decoder-capacity claims require only decode-memory bundles")
    if scope is ClaimScope.LOADER_SUPPLY and protocols != {"loader-supply"}:
        raise ValueError(f"{scope.value} claims require only loader-supply bundles")

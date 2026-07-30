from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OutputContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OutputContract:
    kind: str
    color_space: str
    dtype: str
    layout: str
    channels: int

    @classmethod
    def normalized_rgb(cls) -> OutputContract:
        return cls(kind="normalized-rgb", color_space="RGB", dtype="uint8", layout="HWC", channels=3)

    def to_dict(self) -> dict[str, object]:
        return {
            "channels": self.channels,
            "color_space": self.color_space,
            "dtype": self.dtype,
            "kind": self.kind,
            "layout": self.layout,
        }


def validate_output(output: object, contract: OutputContract) -> None:
    if not isinstance(output, np.ndarray):
        raise OutputContractError(f"expected numpy.ndarray, got {type(output).__name__}")
    if output.dtype.name != contract.dtype:
        raise OutputContractError(f"expected dtype {contract.dtype}, got {output.dtype.name}")
    if contract.layout != "HWC":
        raise OutputContractError(f"unsupported output layout: {contract.layout}")
    if output.ndim != 3 or output.shape[-1] != contract.channels:
        raise OutputContractError(
            f"expected HWC array with {contract.channels} channels, got shape {output.shape}",
        )
    if contract.color_space != "RGB":
        raise OutputContractError(f"unsupported color space: {contract.color_space}")
    if not output.flags.c_contiguous:
        raise OutputContractError("normalized output must be C-contiguous and fully materialized")

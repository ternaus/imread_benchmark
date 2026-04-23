from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import ClassVar

import numpy as np

ENTRY_POINT_GROUP = "imread_benchmark.decoders"


class BaseDecoder(ABC):
    name: str
    package_name: str

    # Which [project.optional-dependencies] group provides this decoder's wheels.
    # Drives venv selection in the CLI.
    group: ClassVar[str] = "mainstream"

    # Platform skips. Each entry is (system, machine) where either may be "*".
    # system  ∈ {"Linux", "Darwin", "Windows", "*"} (matches platform.system())
    # machine ∈ {"x86_64", "arm64", "aarch64", "*"} (matches platform.machine())
    skip_single: ClassVar[tuple[tuple[str, str], ...]] = ()
    skip_dataloader: ClassVar[tuple[tuple[str, str], ...]] = ()

    # If False, never run inside a PyTorch DataLoader (e.g. tensorflow:
    # nobody wires tf.io.decode_jpeg into a torch worker in practice).
    in_dataloader: ClassVar[bool] = True

    @abstractmethod
    def decode(self, data: bytes) -> np.ndarray:
        """Decode JPEG bytes into an (H, W, 3) uint8 RGB numpy array."""

    def decode_path(self, path: str) -> np.ndarray:
        """Decode a JPEG file by path. Override for library-native path APIs."""
        return self.decode(Path(path).read_bytes())

    def get_num_threads(self) -> int:
        """Return the number of threads the library uses internally. Override per library."""
        return 1

    def set_num_threads(self, n: int) -> None:
        """Set the number of threads the library uses. No-op for single-threaded libraries."""

    @classmethod
    def _matches_current(cls, skips: tuple[tuple[str, str], ...]) -> bool:
        s, m = platform.system(), platform.machine()
        return any((sk in (s, "*")) and (mk in (m, "*")) for sk, mk in skips)

    @classmethod
    def runs_single_here(cls) -> bool:
        return not cls._matches_current(cls.skip_single)

    @classmethod
    def runs_dataloader_here(cls) -> bool:
        return cls.in_dataloader and not cls._matches_current(cls.skip_dataloader)


def _load_registry() -> dict[str, type[BaseDecoder]]:
    """
    Build the decoder registry from entry points declared in pyproject.toml.

    Entry-point loading is lazy per call to .load() — we resolve every name
    eagerly so the CLI can list/filter without importing every decoder's
    third-party deps. Decoder modules themselves import their target
    library lazily inside decode().
    """
    eps: list[EntryPoint] = list(entry_points(group=ENTRY_POINT_GROUP))
    out: dict[str, type[BaseDecoder]] = {}
    for ep in eps:
        cls = ep.load()
        if not isinstance(cls, type) or not issubclass(cls, BaseDecoder):
            raise TypeError(
                f"Entry point '{ep.name}' did not load a BaseDecoder subclass: got {cls!r}",
            )
        out[ep.name] = cls
    return out


REGISTRY: dict[str, type[BaseDecoder]] = _load_registry()

__all__ = ["ENTRY_POINT_GROUP", "REGISTRY", "BaseDecoder"]

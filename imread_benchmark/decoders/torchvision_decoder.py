from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class TorchvisionDecoder(BaseDecoder):
    name = "torchvision"
    package_name = "torchvision"

    def decode(self, data: bytes) -> np.ndarray:
        import torch
        import torchvision

        buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        return torchvision.io.decode_image(buf).permute(1, 2, 0).numpy()

    def decode_path(self, path: str) -> np.ndarray:
        import torchvision

        return torchvision.io.read_image(path).permute(1, 2, 0).numpy()

    def get_num_threads(self) -> int:
        import torch

        return torch.get_num_threads()

    def set_num_threads(self, n: int) -> None:
        import torch

        torch.set_num_threads(n)

from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class OpenCVDecoder(BaseDecoder):
    name = "opencv"
    package_name = "opencv-python-headless"

    def decode(self, data: bytes) -> np.ndarray:
        import cv2

        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB)

    def decode_path(self, path: str) -> np.ndarray:
        import cv2

        return cv2.imread(path, cv2.IMREAD_COLOR_RGB)

    def get_num_threads(self) -> int:
        import cv2

        return cv2.getNumThreads()

    def set_num_threads(self, n: int) -> None:
        import cv2

        cv2.setNumThreads(n)

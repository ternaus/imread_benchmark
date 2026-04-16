from __future__ import annotations

import numpy as np

from imread_benchmark.decoders import BaseDecoder


class TensorFlowDecoder(BaseDecoder):
    name = "tensorflow"
    package_name = "tensorflow"

    def decode(self, data: bytes) -> np.ndarray:
        import tensorflow as tf

        return tf.io.decode_jpeg(data, channels=3).numpy()

    def decode_path(self, path: str) -> np.ndarray:
        import tensorflow as tf

        raw = tf.io.read_file(path)
        return tf.io.decode_image(raw, channels=3).numpy()

    def get_num_threads(self) -> int:
        import tensorflow as tf

        return tf.config.threading.get_intra_op_parallelism_threads() or 1

    def set_num_threads(self, n: int) -> None:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(1)

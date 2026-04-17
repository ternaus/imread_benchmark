# Image Loading Benchmark

## Overview

Benchmarks the speed of reading JPEG images and converting them to RGB numpy arrays across popular Python libraries. Targets machine learning training pipelines on **macOS ARM64 (Apple M-series)** using the ImageNet validation set.


|                                         |
| --------------------------------------- |
|                                         |
| Performance on Apple Silicon (M-series) |


## GitAds Sponsored

[Sponsored by GitAds](https://gitads.dev/v1/ad-track?source=ternaus/imread_benchmark@github)

## Important Note on Image Conversion

All decoders output `(H, W, 3)` uint8 RGB numpy arrays for a fair comparison. Libraries that default to other formats (OpenCV → BGR, torchvision → CHW tensor, TensorFlow → EagerTensor) include a conversion step. Note that in real ML pipelines the conversion is often unnecessary.

## Benchmark Modes

**Memory mode** (default): images are pre-loaded as bytes before the timed loop. This measures pure decode throughput with no disk I/O.

**Disk mode**: each decode call reads the file from disk. Includes I/O latency.

## Dataset

[ImageNet validation set](https://image-net.org) — 50,000 JPEG images, ~500×400px.

```bash
# Download
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir -p imagenet/val
tar -xf ILSVRC2012_img_val.tar -C imagenet/val
```

## System Requirements (macOS)

```bash
brew install jpeg-turbo   # required by simplejpeg and turbojpeg
brew install vips         # required by pyvips (NOT bundled in the pip wheel)
```

## Installation

```bash
# Install uv if needed
pip install uv

# Install the orchestrator (control-plane) into a venv.
# Per-library worker venvs (mainstream / tensorflow / pillow-simd) are
# created lazily on first run, with the right libjpeg-turbo / libvips deps.
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Running the Benchmark

```bash
# What would run on this machine?
imread-benchmark list-libs

# Single + DataLoader for every supported decoder, default 50k images
imread-benchmark run --data-dir /path/to/imagenet/val

# Faster smoke run
imread-benchmark run --data-dir /path/to/imagenet/val \
    --num-images 2000 --num-runs 5 --dataloader-runs 2 \
    --workers 0,2

# Just one library, single-thread benchmark only
imread-benchmark run --data-dir /path/to/imagenet/val \
    --libs opencv --mode single

# Generate plots from output/ JSONs
imread-benchmark plot --input output --output _internal/plots
```

The CLI sets up `venvs/<group>/` for each dependency group it needs. Subsequent runs reuse those venvs, so only the first invocation pays the install cost.

## Running on Google Cloud

Spin up a benchmark VM on GCP, run everything against ImageNet from a GCS bucket, and have it self-delete when done:

```bash
./gcp/run.sh \
    --imagenet-bucket gs://my-bucket/imagenet/val \
    --results-bucket  gs://my-bucket/imread-results \
    --no-wait
```

Built venvs are cached in GCS (keyed by `sha256(uv.lock)`), so reruns on the same machine type skip the ~25-minute install. Use `--force-rebuild` to re-resolve PyPI without editing `uv.lock`. Full details, machine-type matrix, cost, and cache semantics: [docs/gcp_benchmarks.md](docs/gcp_benchmarks.md).

## Results Structure

```
output/
└── darwin_Apple-M4-Max/
    ├── opencv_results.json
    ├── pillow_results.json
    ├── opencv_dataloader_results.json
    └── ...
```

## Libraries Benchmarked

### Direct libjpeg-turbo (fastest)

- **simplejpeg** — CFFI binding; zero-copy decode from bytes
- **turbojpeg** (PyTurboJPEG) — Python binding for libjpeg-turbo
- **jpeg4py** — direct libjpeg-turbo binding (**Linux only**)
- **kornia-rs** — Rust implementation using libjpeg-turbo
- **OpenCV** (opencv-python-headless)

### Comprehensive codec libraries

- **imagecodecs** — uses libjpeg-turbo 3.x; prebuilt ARM64 wheels
- **pyvips** — libvips bindings (bundled in wheels)

### Standard libjpeg

- **Pillow**
- **Pillow-SIMD** (**Linux x86-64 only**)
- **scikit-image**
- **imageio**

### ML framework components

- **torchvision**
- **tensorflow**

## Performance Considerations

- All benchmarks run single-threaded unless using the DataLoader benchmark
- Memory mode is the recommended baseline — it isolates decode speed from storage
- Results based on ImageNet JPEG images (~500×400px)

## Recommendations

### High-throughput ML training

- Use `simplejpeg`, `turbojpeg`, or `kornia-rs` for maximum single-thread decode speed
- Use the DataLoader benchmark to find the best `num_workers` for your CPU

### Cross-platform

- `kornia-rs` and `opencv` offer the most consistent cross-platform performance

### Feature-rich applications

- `opencv` remains the best choice when you need more than just JPEG decoding

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run linters
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new decoder.

## Citation

If you found this work useful, please cite:

```bibtex
@misc{iglovikov2025speed,
      title={Need for Speed: A Comprehensive Benchmark of JPEG Decoders in Python},
      author={Vladimir Iglovikov},
      year={2025},
      eprint={2501.13131},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      doi={10.48550/arXiv.2501.13131}
}
```

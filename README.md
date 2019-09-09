[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
 [![CircleCI](https://circleci.com/gh/ternaus/io_benchmark/tree/master.svg?style=svg)](https://circleci.com/gh/ternaus/io_benchmark/tree/master)

# I/O benchmark
I/O benchmark for different image processing python libraries.

The code is inspired by the benchmark code of [Alex Parinov](https://github.com/creafz) for the [albumentations](https://github.com/albu/albumentations) library.

[https://github.com/albu/albumentations/blob/master/benchmark/benchmark.py](https://github.com/albu/albumentations/blob/master/benchmark/benchmark.py])

The idea is inspired by the work of [Roman Soloviov](https://github.com/zfturbo):

[https://www.kaggle.com/zfturbo/benchmark-2019-speed-of-image-reading](https://www.kaggle.com/zfturbo/benchmark-2019-speed-of-image-reading)

# Installation
```bash
sudo apt install libturbojpeg libvips-dev
```


You can use pip to install `imread_benchmark`:

```bash
pip install imread_benchmark
```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:

```bash
pip install -U git+https://github.com/ternaus/imread_benchmark
```

# To calculate the I/O speed of your SSD/HDD in Linux

```bash
sudo apt-get install hdparm

sudo hdparm -Tt <disk_id>
```
where `disk_id` is of the type `/dev/sda`

As a result you may expect something like:

```bash
/dev/sda:
 Timing cached reads:   26114 MB in  1.99 seconds = 13122.03 MB/sec
 Timing buffered disk reads: 1062 MB in  3.00 seconds = 353.70 MB/sec
```

# To run the benchmark
To get the description of all input parameters
```bash
imread_benchmark -h
```


```bash
imread_benchmark -d <path to images> \
                 -i <number of images to use> \
                 -r <number of repeats>
```

Extra options:
`-p` - to print benchmarked libraries versions
`-s` - to shuffle images on every run
`--show-std` - to show standard deviation for measurements

# Libraries that are benchmarked:

* OpenCV
* pillow-simd (PIL-SIMD)
* jpeg4py
* scikit-image (skimage)
* imageio
* pyvips

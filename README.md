[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# Installation

Requirements:

```bash
sudo apt install requirements.txt
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

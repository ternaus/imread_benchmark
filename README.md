[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
 [![CircleCI](https://circleci.com/gh/ternaus/io_benchmark/tree/master.svg?style=svg)](https://circleci.com/gh/ternaus/io_benchmark/tree/master)

# I/O benchmark
I/O benchmark for different image processing python libraries.

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

# Libraries that are considered:

TODO => Automatic generation of this section

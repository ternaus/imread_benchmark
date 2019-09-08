from imread_benchmark.benchmark import GetSize, get_image_paths, benchmark


def test_get_size_benchmark():
    libraries = ["opencv", "PIL"]

    benchmarks = [
        GetSize(),
        # GetArray()
    ]

    image_paths = get_image_paths("tests/test_images", num_images=100)

    assert benchmark(libraries, benchmarks, image_paths, num_runs=1)

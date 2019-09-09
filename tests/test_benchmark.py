from imread_benchmark.benchmark import GetArray, GetSize, benchmark, get_image_paths  # NOQA


def test_get_size_benchmark():
    libraries = ["opencv", "PIL", "jpeg4py", "skimage", "imageio", "pyvips"]

    benchmarks = [GetSize(), GetArray()]

    image_paths = get_image_paths("tests/test_images", num_images=100)

    assert benchmark(libraries, benchmarks, image_paths, num_runs=1, shuffle=True)

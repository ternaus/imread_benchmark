import logging
import os
import platform
import sys
from importlib.metadata import version

import cpuinfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_LIBRARIES = {
    "opencv": "opencv-python-headless",
    "pil": "pillow",
    "jpeg4py": "jpeg4py",
    "skimage": "scikit-image",
    "imageio": "imageio",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "kornia": "kornia-rs",
}


def get_package_versions():
    import multiprocessing

    # Get CPU info
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_details = {
            "brand_raw": cpu_info.get("brand_raw", "Unknown"),
            "arch": cpu_info.get("arch", "Unknown"),
            "hz_advertised_raw": cpu_info.get("hz_advertised_raw", "Unknown"),
            "count": multiprocessing.cpu_count(),
        }
    except Exception as e:
        logger.warning(f"Failed to get CPU info: {e}")
        cpu_details = {"error": str(e)}

    versions = {
        "Python": sys.version.split()[0],
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "CPU": cpu_details,
    }

    lib_name = os.environ.get("BENCHMARK_LIBRARY")
    if lib_name:
        pkg_name = SUPPORTED_LIBRARIES.get(lib_name)
        if pkg_name:
            try:
                versions[lib_name] = version(pkg_name)
            except Exception as e:
                versions[lib_name] = f"Error getting version: {e!s}"

    return versions


def get_system_identifier() -> str:
    """
    Get a detailed system identifier including OS and CPU.

    Returns:
        str: A string combining OS and CPU model, formatted as 'os_cpu-model'

    """
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_brand = cpu_info.get("brand_raw", "Unknown")

        # Simple OS identification
        os_id = "darwin" if platform.system().lower() == "darwin" else "linux"

        # Replace spaces with hyphens but keep full names
        cpu_id = cpu_brand.replace(" ", "-")
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return "unknown-system"
    else:
        return f"{os_id}_{cpu_id}"

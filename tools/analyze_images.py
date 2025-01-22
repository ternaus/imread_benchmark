import statistics
from pathlib import Path

from PIL import Image


def analyze_images(folder_path: str | Path, limit: int = 2000) -> None:
    """Analyze first N JPEG images in the given folder."""
    folder = Path(folder_path)
    file_sizes = []
    resolutions = []

    # Collect data from first 'limit' images
    for img_path in sorted(folder.glob("*.*"))[:limit]:
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            # Get file size in KB
            file_sizes.append(img_path.stat().st_size / 1024)

            # Get image resolution
            with Image.open(img_path) as img:
                resolutions.append(img.size)

    if not file_sizes:
        print("No JPEG images found in the folder")
        return

    # Analyze file sizes
    avg_size = statistics.mean(file_sizes)
    min_size = min(file_sizes)
    max_size = max(file_sizes)
    median_size = statistics.median(file_sizes)

    # Analyze resolutions
    widths, heights = zip(*resolutions, strict=False)
    min_res = (min(widths), min(heights))
    max_res = (max(widths), max(heights))
    avg_res = (statistics.mean(widths), statistics.mean(heights))
    median_res = (statistics.median(widths), statistics.median(heights))

    # Print results
    print(f"Analysis of first {len(file_sizes)} images:")
    print("\nFile Sizes (KB):")
    print(f"- Average: {avg_size:.1f}")
    print(f"- Median: {median_size:.1f}")
    print(f"- Range: {min_size:.1f} - {max_size:.1f}")
    print("\nResolutions (pixels):")
    print(f"- Average: {avg_res[0]:.0f} x {avg_res[1]:.0f}")
    print(f"- Median: {median_res[0]:.0f} x {median_res[1]:.0f}")
    print(f"- Range: {min_res[0]}x{min_res[1]} - {max_res[0]}x{max_res[1]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_image_folder>")
        sys.exit(1)

    analyze_images(sys.argv[1])

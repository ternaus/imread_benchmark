from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps, features
from PIL import __version__ as pillow_version

from imread_benchmark.datasets.package import build_dataset_package

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_LONG_EDGES = (512, 1024, 2048)
DEFAULT_QUALITIES = (50, 75, 90, 95)
DEFAULT_COMPRESSED_BYTE_LIMIT = 2 * 1024**3
DEFAULT_ORDER_SEED = 20260729

_SUBSAMPLING_CODES = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
_KNOWN_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class ControlledDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    relative_path: str
    source_id: str
    source_bytes: int
    source_sha256: str
    width: int
    height: int
    normalized_rgb_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_rgb_sha256": self.normalized_rgb_sha256,
            "normalized_rgb_size": [self.width, self.height],
            "relative_path": self.relative_path,
            "source_bytes": self.source_bytes,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class _Design:
    long_edges: tuple[int, ...]
    qualities: tuple[int, ...]
    include_native: bool
    subsampling: str
    seed: int
    compressed_byte_limit: int


def prepare_controlled_ablation(  # noqa: PLR0913 - every factor is an explicit experimental control
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_name: str,
    source_release: str,
    source_license: str,
    source_url: str | None = None,
    long_edges: Sequence[int] = DEFAULT_LONG_EDGES,
    qualities: Sequence[int] = DEFAULT_QUALITIES,
    include_native: bool = True,
    subsampling: str = "4:2:0",
    seed: int = DEFAULT_ORDER_SEED,
    compressed_byte_limit: int = DEFAULT_COMPRESSED_BYTE_LIMIT,
) -> Path:
    """Build matched JPEG workloads for a controlled resolution-by-quality design."""
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    source_dataset = _validate_source_dataset(
        name=source_name,
        release=source_release,
        license_name=source_license,
        url=source_url,
    )
    design = _validate_design(
        source_root=source_root,
        output_root=output_root,
        long_edges=long_edges,
        qualities=qualities,
        include_native=include_native,
        subsampling=subsampling,
        seed=seed,
        compressed_byte_limit=compressed_byte_limit,
    )

    sources = _read_sources(source_root, minimum_long_edge=max(design.long_edges, default=0))
    if design.include_native:
        oversized = next((source for source in sources if max(source.width, source.height) > 65_535), None)
        if oversized is not None:
            raise ControlledDatasetError(
                f"native source exceeds the JPEG dimension limit: {oversized.relative_path} is "
                f"{oversized.width}x{oversized.height}",
            )
    ordered_sources = tuple(
        sorted(
            sources,
            key=lambda source: (
                hashlib.sha256(
                    f"controlled-source-order-v2\0{design.seed}\0{source.source_id}".encode(),
                ).digest(),
                source.source_id,
            ),
        ),
    )
    levels: tuple[int | None, ...] = (
        *design.long_edges,
        *((None,) if design.include_native else ()),
    )
    workload_ids = {
        (long_edge, quality): _workload_id(long_edge, quality) for long_edge in levels for quality in design.qualities
    }

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".controlled-ablation.", dir=output_root))
    try:
        workload_roots = {key: staging / "workloads" / workload_id for key, workload_id in workload_ids.items()}
        totals = _encode_workloads(
            ordered_sources=ordered_sources,
            levels=levels,
            design=design,
            workload_ids=workload_ids,
            workload_roots=workload_roots,
        )

        ordered_source_ids = [source.source_id for source in ordered_sources]
        provenance = {
            "dataset": "controlled-resolution-quality-ablation",
            "design": {
                "controls": {
                    "chroma_subsampling": design.subsampling,
                    "color_conversion": "Pillow Image.convert(RGB)",
                    "encoder_optimize": False,
                    "encoder_progressive": False,
                    "encoder_container_headers": "Pillow defaults",
                    "resize_filter": "Pillow LANCZOS",
                    "source_metadata": "stripped",
                    "source_format": "lossless PNG only",
                },
                "factors": {
                    "encoder_quality": list(design.qualities),
                    "long_edge_pixels": [
                        *design.long_edges,
                        *(("native",) if design.include_native else ()),
                    ],
                },
                "pairing_unit": "source_id",
            },
            "encoder": {
                "implementation": "Pillow",
                "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
                "libjpeg_version": features.version("jpg"),
                "pillow_version": pillow_version,
            },
            "order": {"method": "seeded-sha256-order-v2", "seed": design.seed},
            "ordered_source_ids": ordered_source_ids,
            "schema_version": "2.0",
            "source_dataset": source_dataset,
            "sources": [source.to_dict() for source in sources],
            "workloads": {
                workload_ids[key]: {
                    "compressed_byte_limit": design.compressed_byte_limit,
                    "encoder_quality": key[1],
                    "item_count": len(ordered_sources),
                    "long_edge_pixels": key[0] if key[0] is not None else "native",
                    "ordered_source_ids": ordered_source_ids,
                    "total_compressed_bytes": totals[key],
                }
                for key in workload_ids
            },
        }
        return build_dataset_package(
            package_name="controlled-resolution-quality",
            workloads={workload_id: workload_roots[key] for key, workload_id in workload_ids.items()},
            output_root=output_root / "packages",
            provenance=provenance,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_design(  # noqa: PLR0913 - mirrors public experimental controls
    *,
    source_root: Path,
    output_root: Path,
    long_edges: Sequence[int],
    qualities: Sequence[int],
    include_native: bool,
    subsampling: str,
    seed: int,
    compressed_byte_limit: int,
) -> _Design:
    edges = _validated_unique_positive_ints(long_edges, field="long edge", maximum=65_535)
    encoder_qualities = _validated_unique_positive_ints(qualities, field="quality", maximum=95)
    if not isinstance(include_native, bool):
        raise ControlledDatasetError("include_native must be a boolean")
    if not edges and not include_native:
        raise ControlledDatasetError("at least one long edge or the native level is required")
    if not encoder_qualities:
        raise ControlledDatasetError("at least one encoder quality is required")
    if subsampling not in _SUBSAMPLING_CODES:
        raise ControlledDatasetError(f"subsampling must be one of {sorted(_SUBSAMPLING_CODES)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ControlledDatasetError("seed must be an integer")
    if (
        isinstance(compressed_byte_limit, bool)
        or not isinstance(compressed_byte_limit, int)
        or compressed_byte_limit <= 0
    ):
        raise ControlledDatasetError("compressed_byte_limit must be a positive integer")
    if not source_root.is_dir():
        raise ControlledDatasetError(f"source directory does not exist: {source_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ControlledDatasetError("output directory must not be inside the source directory")
    return _Design(
        long_edges=edges,
        qualities=encoder_qualities,
        include_native=include_native,
        subsampling=subsampling,
        seed=seed,
        compressed_byte_limit=compressed_byte_limit,
    )


def _validate_source_dataset(
    *,
    name: str,
    release: str,
    license_name: str,
    url: str | None,
) -> dict[str, str]:
    values = {"license": license_name, "name": name, "release": release}
    for field, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ControlledDatasetError(f"source dataset {field} must be a non-empty string")
    normalized = {field: value.strip() for field, value in values.items()}
    if url is not None:
        if not isinstance(url, str) or not url.strip():
            raise ControlledDatasetError("source dataset URL must be a non-empty string when provided")
        normalized["url"] = url.strip()
    return normalized


def _encode_workloads(
    *,
    ordered_sources: tuple[_Source, ...],
    levels: tuple[int | None, ...],
    design: _Design,
    workload_ids: dict[tuple[int | None, int], str],
    workload_roots: dict[tuple[int | None, int], Path],
) -> dict[tuple[int | None, int], int]:
    totals = dict.fromkeys(workload_ids, 0)
    for order_index, source in enumerate(ordered_sources):
        with _open_normalized_rgb(source.path) as original:
            for long_edge in levels:
                rendered = original.copy() if long_edge is None else _resize_long_edge(original, long_edge)
                try:
                    for quality in design.qualities:
                        key = (long_edge, quality)
                        destination = workload_roots[key] / f"order-{order_index:06d}" / f"{source.source_id}.jpg"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        rendered.save(
                            destination,
                            format="JPEG",
                            quality=quality,
                            subsampling=_SUBSAMPLING_CODES[design.subsampling],
                            optimize=False,
                            progressive=False,
                            exif=b"",
                        )
                        totals[key] += destination.stat().st_size
                        _validate_workload_budget(
                            workload_id=workload_ids[key],
                            resident_bytes=totals[key],
                            limit=design.compressed_byte_limit,
                        )
                finally:
                    rendered.close()
    return totals


def _validate_workload_budget(*, workload_id: str, resident_bytes: int, limit: int) -> None:
    if resident_bytes > limit:
        raise ControlledDatasetError(
            f"{workload_id} exceeds the resident compressed-byte budget: "
            f"{resident_bytes} bytes exceed the {limit}-byte budget",
        )


def _read_sources(root: Path, *, minimum_long_edge: int) -> tuple[_Source, ...]:
    image_paths = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _KNOWN_IMAGE_SUFFIXES),
        ),
    )
    non_png = tuple(path for path in image_paths if path.suffix.lower() != ".png")
    if non_png:
        relative = non_png[0].relative_to(root).as_posix()
        raise ControlledDatasetError(f"controlled sources must be lossless PNG files; found {relative}")
    if not image_paths:
        raise ControlledDatasetError(f"no lossless PNG source images found under {root}")

    sources: list[_Source] = []
    seen_source_ids: set[str] = set()
    for path in image_paths:
        relative_path = path.relative_to(root).as_posix()
        source_sha256 = _sha256_file(path)
        with _open_normalized_rgb(path) as image:
            width, height = image.size
            if max(width, height) < minimum_long_edge:
                raise ControlledDatasetError(
                    f"source {relative_path} is smaller than requested long edge {minimum_long_edge}: {width}x{height}",
                )
            normalized_digest = hashlib.sha256()
            normalized_digest.update(f"RGB\0{width}\0{height}\0".encode())
            normalized_digest.update(image.tobytes())
        source_id = hashlib.sha256(f"{relative_path}\0{source_sha256}".encode()).hexdigest()
        if source_id in seen_source_ids:
            raise ControlledDatasetError(f"duplicate controlled source identity: {relative_path}")
        seen_source_ids.add(source_id)
        sources.append(
            _Source(
                path=path,
                relative_path=relative_path,
                source_id=source_id,
                source_bytes=path.stat().st_size,
                source_sha256=source_sha256,
                width=width,
                height=height,
                normalized_rgb_sha256=normalized_digest.hexdigest(),
            ),
        )
    return tuple(sources)


def _open_normalized_rgb(path: Path) -> Image.Image:
    try:
        encoded = Image.open(path)
    except (OSError, ValueError) as exc:
        raise ControlledDatasetError(f"cannot decode controlled source {path}: {exc}") from exc
    with encoded:
        _validate_source_image(encoded, path)
        normalized = ImageOps.exif_transpose(encoded).convert("RGB")
        normalized.load()
        normalized.info.clear()
        return normalized


def _validate_source_image(image: Image.Image, path: Path) -> None:
    if image.format != "PNG":
        raise ControlledDatasetError(f"controlled source is not a PNG: {path}")
    if getattr(image, "n_frames", 1) != 1:
        raise ControlledDatasetError(f"animated or multi-frame PNG is not supported: {path}")
    if "A" in image.getbands() or "transparency" in image.info:
        raise ControlledDatasetError(f"transparent PNG requires an undeclared compositing policy: {path}")


def _resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    if width >= height:
        target = (long_edge, max(1, (height * long_edge + width // 2) // width))
    else:
        target = (max(1, (width * long_edge + height // 2) // height), long_edge)
    return image.resize(target, resample=Image.Resampling.LANCZOS)


def _validated_unique_positive_ints(
    values: Sequence[int],
    *,
    field: str,
    maximum: int | None = None,
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ControlledDatasetError(f"each {field} must be a positive integer")
        if maximum is not None and value > maximum:
            raise ControlledDatasetError(f"each {field} must be at most {maximum}")
        if value in result:
            raise ControlledDatasetError(f"duplicate {field}: {value}")
        result.append(value)
    return tuple(sorted(result))


def _workload_id(long_edge: int | None, quality: int) -> str:
    size = "native" if long_edge is None else f"le{long_edge:04d}"
    return f"controlled-{size}-q{quality:03d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

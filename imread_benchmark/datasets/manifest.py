from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

MANIFEST_SCHEMA_VERSION = "2.0"

if TYPE_CHECKING:
    from collections.abc import Iterator

_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    },
)


@dataclass(frozen=True, slots=True)
class _JpegProperties:
    width: int | None = None
    height: int | None = None
    components: int | None = None
    progressive: bool | None = None
    subsampling: str | None = None
    has_exif: bool = False
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetItem:
    item_id: str
    relative_path: str
    sha256: str
    compressed_bytes: int
    width: int | None
    height: int | None
    components: int | None
    progressive: bool | None
    subsampling: str | None
    has_exif: bool
    parse_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "components": self.components,
            "compressed_bytes": self.compressed_bytes,
            "has_exif": self.has_exif,
            "height": self.height,
            "item_id": self.item_id,
            "parse_error": self.parse_error,
            "progressive": self.progressive,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "subsampling": self.subsampling,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    root: Path
    dataset_name: str
    manifest_id: str
    items: tuple[DatasetItem, ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        root: str | Path,
        *,
        dataset_name: str,
    ) -> DatasetManifest:
        resolved_root = Path(root).resolve()
        paths = sorted(
            path for path in resolved_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if not paths:
            raise ValueError(f"no JPEG files found under {resolved_root}")

        items = tuple(_build_item(resolved_root, path) for path in paths)
        manifest_id = _manifest_digest(dataset_name, items)
        return cls(
            root=resolved_root,
            dataset_name=dataset_name,
            manifest_id=manifest_id,
            items=items,
        )

    def resolve(self, item: DatasetItem) -> Path:
        return self.root / item.relative_path

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "items": [item.to_dict() for item in self.items],
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
        }


def _build_item(root: Path, path: Path) -> DatasetItem:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    relative_path = path.relative_to(root).as_posix()
    item_id = hashlib.sha256(f"{relative_path}\0{digest}".encode()).hexdigest()
    properties = _parse_jpeg_properties(data)
    return DatasetItem(
        item_id=item_id,
        relative_path=relative_path,
        sha256=digest,
        compressed_bytes=len(data),
        width=properties.width,
        height=properties.height,
        components=properties.components,
        progressive=properties.progressive,
        subsampling=properties.subsampling,
        has_exif=properties.has_exif,
        parse_error=properties.parse_error,
    )


def _manifest_digest(dataset_name: str, items: tuple[DatasetItem, ...]) -> str:
    payload = {
        "dataset_name": dataset_name,
        "items": [item.to_dict() for item in items],
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _parse_jpeg_properties(data: bytes) -> _JpegProperties:
    if not data.startswith(b"\xff\xd8"):
        return _JpegProperties(parse_error="missing JPEG SOI marker")

    has_exif = False
    try:
        for marker, payload in _iter_jpeg_segments(data):
            if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
                has_exif = True
            if marker in _START_OF_FRAME_MARKERS:
                return _properties_from_sof(marker, payload, has_exif=has_exif)
    except ValueError as exc:
        return _JpegProperties(has_exif=has_exif, parse_error=str(exc))

    return _JpegProperties(has_exif=has_exif, parse_error="JPEG start-of-frame marker not found")


def _iter_jpeg_segments(data: bytes) -> Iterator[tuple[int, bytes]]:
    position = 2
    while position < len(data):
        while position < len(data) and data[position] != 0xFF:
            position += 1
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            return

        marker = data[position]
        position += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            return
        if position + 2 > len(data):
            raise ValueError("truncated JPEG segment length")

        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            raise ValueError("invalid JPEG segment length")
        yield marker, data[position + 2 : position + segment_length]
        position += segment_length


def _properties_from_sof(marker: int, payload: bytes, *, has_exif: bool) -> _JpegProperties:
    if len(payload) < 6:
        raise ValueError("truncated JPEG start-of-frame segment")
    height = int.from_bytes(payload[1:3], "big")
    width = int.from_bytes(payload[3:5], "big")
    components = payload[5]
    return _JpegProperties(
        width=width,
        height=height,
        components=components,
        progressive=marker == 0xC2,
        subsampling=_read_subsampling(payload, components),
        has_exif=has_exif,
    )


def _read_subsampling(sof_payload: bytes, components: int) -> str | None:
    if components == 1:
        return "grayscale"
    if components != 3 or len(sof_payload) < 6 + 3 * components:
        return None

    sampling = sof_payload[7]
    horizontal, vertical = sampling >> 4, sampling & 0x0F
    if (horizontal, vertical) == (1, 1):
        return "4:4:4"
    if (horizontal, vertical) == (2, 1):
        return "4:2:2"
    if (horizontal, vertical) == (2, 2):
        return "4:2:0"
    return "other"

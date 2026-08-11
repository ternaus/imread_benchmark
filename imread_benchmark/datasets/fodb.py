from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile, ZipInfo

from imread_benchmark.datasets.manifest import DatasetManifest
from imread_benchmark.datasets.package import build_dataset_package

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FODB_PROVENANCES = ("orig", "facebook", "instagram", "telegram", "twitter", "whatsapp")
FODB_DEVICE_COUNT = 27
DEFAULT_SCENE_COUNT = 12
DEFAULT_SELECTION_SEED = 20260729
DEFAULT_COMPRESSED_BYTE_LIMIT = 2 * 1024**3
QUALITY_ESTIMATOR = "ijg-luma-sum-v2"

_MEMBER_RE = re.compile(
    r"^(?P<device_dir>D(?P<device_number>\d{2})_[^/]+)/"
    r"(?P<provenance>orig|facebook|instagram|telegram|twitter|whatsapp)/"
    r"D(?P=device_number)_img_(?P=provenance)_(?P<scene_id>\d{4})\.jpg$",
)
_SOF_MARKERS = frozenset(
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
_IJG_LUMA_Q50 = (
    16,
    11,
    10,
    16,
    24,
    40,
    51,
    61,
    12,
    12,
    14,
    19,
    26,
    58,
    60,
    55,
    14,
    13,
    16,
    24,
    40,
    57,
    69,
    56,
    14,
    17,
    22,
    29,
    51,
    87,
    80,
    62,
    18,
    22,
    37,
    56,
    68,
    109,
    103,
    77,
    24,
    35,
    55,
    64,
    81,
    104,
    113,
    92,
    49,
    64,
    78,
    87,
    103,
    121,
    120,
    101,
    72,
    92,
    95,
    98,
    112,
    100,
    103,
    99,
)


@dataclass(frozen=True, slots=True)
class FodbMember:
    archive_index: int
    archive_id: str
    archive_path: Path
    zip_member: str
    device_id: str
    device_dir: str
    provenance: str
    scene_id: str
    jpeg_bytes: int
    zip_compressed_bytes: int
    crc32: str


@dataclass(frozen=True, slots=True)
class JpegDescriptor:
    width: int | None
    height: int | None
    components: int | None
    progressive: bool | None
    scan_count: int
    subsampling: str | None
    sampling_factors: tuple[tuple[int, int, int], ...]
    quantization_tables: tuple[tuple[int, tuple[int, ...]], ...]
    quantization_table_sha256: str | None
    quality_estimate: int | None
    quality_estimator: str
    exif_bytes: int
    icc_bytes: int
    other_metadata_bytes: int
    parse_error: str | None

    def to_dict(self) -> dict[str, object]:
        megapixels = None
        if self.width is not None and self.height is not None:
            megapixels = self.width * self.height / 1_000_000
        return {
            "components": self.components,
            "exif_bytes": self.exif_bytes,
            "height": self.height,
            "icc_bytes": self.icc_bytes,
            "megapixels": megapixels,
            "other_metadata_bytes": self.other_metadata_bytes,
            "parse_error": self.parse_error,
            "progressive": self.progressive,
            "quality_estimate": self.quality_estimate,
            "quality_estimator": self.quality_estimator,
            "quantization_table_sha256": self.quantization_table_sha256,
            "quantization_tables": {str(table_id): list(values) for table_id, values in self.quantization_tables},
            "sampling_factors": [list(factors) for factors in self.sampling_factors],
            "scan_count": self.scan_count,
            "subsampling": self.subsampling,
            "width": self.width,
        }


def read_fodb_catalog(archive_paths: Sequence[str | Path]) -> tuple[FodbMember, ...]:
    if not archive_paths:
        raise ValueError("at least one FODB archive is required")

    members: list[FodbMember] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for archive_index, raw_path in enumerate(archive_paths, start=1):
        archive_path = Path(raw_path).expanduser().resolve()
        archive_id = f"part{archive_index:02d}.zip"
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = _parse_member(info, archive_index, archive_id, archive_path)
                if member is None:
                    continue
                key = (member.device_id, member.provenance, member.scene_id)
                if key in seen_keys:
                    raise ValueError(f"duplicate FODB member for {key}: {member.zip_member}")
                seen_keys.add(key)
                members.append(member)

    if not members:
        raise ValueError("no FODB JPEG members found")
    provenance_order = {value: index for index, value in enumerate(FODB_PROVENANCES)}
    return tuple(
        sorted(
            members,
            key=lambda item: (
                item.scene_id,
                item.device_id,
                provenance_order[item.provenance],
            ),
        ),
    )


def complete_scene_ids(
    members: Sequence[FodbMember],
    *,
    expected_device_count: int = FODB_DEVICE_COUNT,
    provenances: Sequence[str] = FODB_PROVENANCES,
) -> tuple[str, ...]:
    device_ids = sorted({member.device_id for member in members})
    if len(device_ids) != expected_device_count:
        raise ValueError(f"expected {expected_device_count} FODB devices, found {len(device_ids)}")

    expected = {(device_id, provenance) for device_id in device_ids for provenance in provenances}
    observed_by_scene: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for member in members:
        if member.provenance in provenances:
            observed_by_scene[member.scene_id].add((member.device_id, member.provenance))
    return tuple(sorted(scene_id for scene_id, observed in observed_by_scene.items() if observed == expected))


def select_scene_ids(scene_ids: Sequence[str], *, count: int, seed: int) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("scene count must be positive")
    ranked = sorted(
        set(scene_ids),
        key=lambda scene_id: (
            hashlib.sha256(f"fodb-scene-v2\0{seed}\0{scene_id}".encode()).digest(),
            scene_id,
        ),
    )
    return tuple(ranked[:count])


def prepare_fodb(  # noqa: PLR0913 - these are the predeclared dataset protocol controls
    archive_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    scene_count: int = DEFAULT_SCENE_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    compressed_byte_limit: int = DEFAULT_COMPRESSED_BYTE_LIMIT,
    expected_device_count: int = FODB_DEVICE_COUNT,
    hash_archives: bool = True,
) -> Path:
    resolved_archives = tuple(Path(path).expanduser().resolve() for path in archive_paths)
    output_root = Path(output_dir).expanduser().resolve()
    members = read_fodb_catalog(resolved_archives)
    complete = complete_scene_ids(members, expected_device_count=expected_device_count)
    selected_scenes = select_scene_ids(complete, count=min(scene_count, len(complete)), seed=seed)
    if not selected_scenes:
        raise ValueError("FODB contains no complete scenes for the requested device/provenance matrix")

    selected = tuple(member for member in members if member.scene_id in selected_scenes)
    expected_items = len(selected_scenes) * expected_device_count * len(FODB_PROVENANCES)
    if len(selected) != expected_items:
        raise ValueError(f"selected matrix should contain {expected_items} JPEGs, found {len(selected)}")
    selection_identity = {
        "expected_device_count": expected_device_count,
        "method": "seeded-sha256-order-v2",
        "provenances": list(FODB_PROVENANCES),
        "schema_version": "2.0",
        "seed": seed,
        "selected_scene_ids": list(selected_scenes),
    }
    selection_id = _digest_json(selection_identity)
    selection_root = output_root / "selections" / selection_id

    workloads = {
        "FODB-native": tuple(member for member in selected if member.provenance == "orig"),
        "FODB-mixed": selected,
    }
    workload_payload: dict[str, object] = {}
    for workload_name, workload_members in workloads.items():
        resident_bytes = sum(member.jpeg_bytes for member in workload_members)
        if resident_bytes > compressed_byte_limit:
            raise ValueError(
                f"{workload_name} exceeds the resident compressed-byte budget: {resident_bytes} bytes exceed the "
                f"{compressed_byte_limit}-byte budget; reduce --scene-count or raise the budget explicitly",
            )

    corpus_root = output_root / "corpus"
    descriptors = _extract_and_describe(selected, corpus_root)
    for workload_name, workload_members in workloads.items():
        workload_payload[workload_name] = _materialize_workload(
            workload_name,
            workload_members,
            corpus_root=corpus_root,
            output_root=selection_root / "workloads" / workload_name.lower(),
            selection_seed=seed,
            compressed_byte_limit=compressed_byte_limit,
        )

    descriptor_by_member = {member_name: descriptor for member_name, descriptor, _, _ in descriptors}
    item_payload = []
    for member in selected:
        descriptor, sha256, crc32 = descriptor_by_member[member.zip_member]
        width, height = descriptor.width, descriptor.height
        bits_per_pixel = None
        if width is not None and height is not None and width > 0 and height > 0:
            bits_per_pixel = member.jpeg_bytes * 8 / (width * height)
        item_payload.append(
            {
                "archive_id": member.archive_id,
                "archive_member_crc32": member.crc32,
                "bits_per_pixel": bits_per_pixel,
                "crc32": crc32,
                "device_dir": member.device_dir,
                "device_id": member.device_id,
                "jpeg_bytes": member.jpeg_bytes,
                "jpeg_sha256": sha256,
                "jpeg": descriptor.to_dict(),
                "provenance": member.provenance,
                "scene_id": member.scene_id,
                "zip_compressed_bytes": member.zip_compressed_bytes,
                "zip_member": member.zip_member,
            },
        )

    archive_payload = []
    for index, archive_path in enumerate(resolved_archives, start=1):
        archive_payload.append(
            {
                "archive_id": f"part{index:02d}.zip",
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256_file(archive_path) if hash_archives else None,
                "source_basename": archive_path.name,
            },
        )

    payload = {
        "archives": archive_payload,
        "complete_scene_count": len(complete),
        "dataset": "Forchheim Image Database (FODB)",
        "device_ids": sorted({member.device_id for member in members}),
        "items": item_payload,
        "license_note": "Research use; do not redistribute selected JPEG files.",
        "provenances": list(FODB_PROVENANCES),
        "schema_version": "2.0",
        "selection": {
            "available_complete_scene_ids": list(complete),
            "requested_scene_count": scene_count,
            "selection_id": selection_id,
            **selection_identity,
        },
        "source_url": "https://faui1-files.cs.fau.de/public/mmsec/datasets/fodb/",
        "workloads": workload_payload,
    }
    return build_dataset_package(
        package_name="fodb-selected",
        workloads={
            "fodb-native": selection_root / "workloads" / "fodb-native",
            "fodb-mixed": selection_root / "workloads" / "fodb-mixed",
        },
        output_root=output_root / "packages",
        provenance=payload,
    )


def _parse_member(
    info: ZipInfo,
    archive_index: int,
    archive_id: str,
    archive_path: Path,
) -> FodbMember | None:
    if info.is_dir():
        return None
    match = _MEMBER_RE.fullmatch(info.filename)
    if match is None:
        return None
    device_id = f"D{match.group('device_number')}"
    return FodbMember(
        archive_index=archive_index,
        archive_id=archive_id,
        archive_path=archive_path,
        zip_member=info.filename,
        device_id=device_id,
        device_dir=match.group("device_dir"),
        provenance=match.group("provenance"),
        scene_id=match.group("scene_id"),
        jpeg_bytes=info.file_size,
        zip_compressed_bytes=info.compress_size,
        crc32=f"{info.CRC:08x}",
    )


def _extract_and_describe(
    members: Sequence[FodbMember],
    corpus_root: Path,
) -> tuple[tuple[str, tuple[JpegDescriptor, str, str], str, str], ...]:
    by_archive: dict[Path, list[FodbMember]] = defaultdict(list)
    for member in members:
        by_archive[member.archive_path].append(member)

    for archive_path, archive_members in by_archive.items():
        with ZipFile(archive_path) as archive:
            for member in archive_members:
                _extract_one(archive, member, corpus_root / member.zip_member)

    result = []
    for member in members:
        path = corpus_root / member.zip_member
        data = path.read_bytes()
        crc32 = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
        if len(data) != member.jpeg_bytes or crc32 != member.crc32:
            raise ValueError(f"extracted JPEG differs from ZIP metadata: {member.zip_member}")
        result.append(
            (
                member.zip_member,
                (_describe_jpeg(data), hashlib.sha256(data).hexdigest(), crc32),
                member.device_id,
                member.scene_id,
            ),
        )
    return tuple(result)


def _extract_one(archive: ZipFile, member: FodbMember, destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != member.jpeg_bytes:
            raise FileExistsError(f"existing extracted file has unexpected size: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        with archive.open(member.zip_member) as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != member.jpeg_bytes:
            raise ValueError(f"truncated extraction for {member.zip_member}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_workload(  # noqa: PLR0913 - explicit output roots keep artifact placement auditable
    workload_name: str,
    members: Sequence[FodbMember],
    *,
    corpus_root: Path,
    output_root: Path,
    selection_seed: int,
    compressed_byte_limit: int,
) -> dict[str, object]:
    ordered_members = sorted(
        members,
        key=lambda member: hashlib.sha256(
            f"fodb-item-v2\0{selection_seed}\0{workload_name}\0{member.zip_member}".encode(),
        ).digest(),
    )
    for order_index, member in enumerate(ordered_members):
        source = corpus_root / member.zip_member
        destination = output_root / f"order-{order_index:04d}" / member.zip_member
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.samefile(source):
                raise FileExistsError(f"existing workload file is not the expected hard link: {destination}")
        else:
            os.link(source, destination)

    manifest = DatasetManifest.build(output_root, dataset_name=workload_name)
    max_decoded_bytes = max((item.width or 0) * (item.height or 0) * 3 for item in manifest.items)

    return {
        "compressed_byte_limit": compressed_byte_limit,
        "max_decoded_rgb_bytes": max_decoded_bytes,
        "num_items": len(members),
        "provenances": sorted({member.provenance for member in members}, key=FODB_PROVENANCES.index),
        "scene_ids": sorted({member.scene_id for member in members}),
        "total_compressed_bytes": sum(member.jpeg_bytes for member in members),
        "traversal_order": "seeded-sha256-order-v2",
    }


def _describe_jpeg(data: bytes) -> JpegDescriptor:
    width = height = components = None
    progressive = None
    sampling_factors: tuple[tuple[int, int, int], ...] = ()
    quantization_tables: dict[int, tuple[int, ...]] = {}
    scan_count = exif_bytes = icc_bytes = other_metadata_bytes = 0
    parse_error = None
    try:
        for marker, payload in _iter_jpeg_segments(data):
            if marker == 0xDA:
                scan_count += 1
            elif marker in _SOF_MARKERS and width is None:
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                components = payload[5]
                progressive = marker == 0xC2
                sampling_factors = _sampling_factors(payload, components)
            elif marker == 0xDB:
                quantization_tables.update(_parse_dqt(payload))
            elif marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
                exif_bytes += len(payload)
            elif marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
                icc_bytes += len(payload)
            elif 0xE0 <= marker <= 0xEF or marker == 0xFE:
                other_metadata_bytes += len(payload)
    except (IndexError, ValueError) as exc:
        parse_error = str(exc)

    table_items = tuple(sorted(quantization_tables.items()))
    table_hash = None
    if table_items:
        canonical = json.dumps({str(key): list(value) for key, value in table_items}, separators=(",", ":"))
        table_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return JpegDescriptor(
        width=width,
        height=height,
        components=components,
        progressive=progressive,
        scan_count=scan_count,
        subsampling=_subsampling_label(sampling_factors, components),
        sampling_factors=sampling_factors,
        quantization_tables=table_items,
        quantization_table_sha256=table_hash,
        quality_estimate=_estimate_quality(quantization_tables.get(0)),
        quality_estimator=QUALITY_ESTIMATOR,
        exif_bytes=exif_bytes,
        icc_bytes=icc_bytes,
        other_metadata_bytes=other_metadata_bytes,
        parse_error=parse_error,
    )


def _iter_jpeg_segments(data: bytes) -> Iterable[tuple[int, bytes]]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("missing JPEG SOI marker")
    position = 2
    while position < len(data):
        marker_start = data.find(b"\xff", position)
        if marker_start < 0:
            return
        position = marker_start + 1
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            return
        marker = data[position]
        position += 1
        if marker == 0xD9:
            return
        if marker in {0x00, 0x01} or 0xD0 <= marker <= 0xD8:
            continue
        if position + 2 > len(data):
            raise ValueError("truncated JPEG segment length")
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            raise ValueError("invalid JPEG segment length")
        payload = data[position + 2 : position + segment_length]
        yield marker, payload
        position += segment_length


def _sampling_factors(payload: bytes, components: int) -> tuple[tuple[int, int, int], ...]:
    if len(payload) < 6 + 3 * components:
        raise ValueError("truncated JPEG start-of-frame segment")
    result = []
    for component_index in range(components):
        offset = 6 + 3 * component_index
        component_id = payload[offset]
        sampling = payload[offset + 1]
        result.append((component_id, sampling >> 4, sampling & 0x0F))
    return tuple(result)


def _subsampling_label(
    sampling_factors: Sequence[tuple[int, int, int]],
    components: int | None,
) -> str | None:
    if components == 1:
        return "grayscale"
    if components != 3 or not sampling_factors:
        return None
    horizontal, vertical = sampling_factors[0][1:]
    if (horizontal, vertical) == (1, 1):
        return "4:4:4"
    if (horizontal, vertical) == (2, 1):
        return "4:2:2"
    if (horizontal, vertical) == (2, 2):
        return "4:2:0"
    return "other"


def _parse_dqt(payload: bytes) -> dict[int, tuple[int, ...]]:
    tables: dict[int, tuple[int, ...]] = {}
    position = 0
    while position < len(payload):
        header = payload[position]
        position += 1
        precision, table_id = header >> 4, header & 0x0F
        value_bytes = 2 if precision else 1
        end = position + 64 * value_bytes
        if precision not in {0, 1} or end > len(payload):
            raise ValueError("invalid JPEG quantization table")
        values = tuple(
            int.from_bytes(payload[offset : offset + value_bytes], "big")
            for offset in range(position, end, value_bytes)
        )
        tables[table_id] = values
        position = end
    return tables


def _estimate_quality(luma_table: Sequence[int] | None) -> int | None:
    if luma_table is None or len(luma_table) != 64:
        return None
    observed_sum = sum(luma_table)
    return min(range(1, 101), key=lambda quality: abs(_ijg_luma_sum(quality) - observed_sum))


def _ijg_luma_sum(quality: int) -> int:
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return sum(min(255, max(1, (value * scale + 50) // 100)) for value in _IJG_LUMA_Q50)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

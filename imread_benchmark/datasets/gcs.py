from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from imread_benchmark.datasets.materializer import (
    ObjectConflictError,
    ObjectMetadata,
    ObjectNotFoundError,
    ObjectStoreError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class GcloudObjectStore:
    base_uri: str
    command_prefix: tuple[str, ...]
    environment: dict[str, str]

    def __init__(
        self,
        base_uri: str,
        *,
        command_prefix: tuple[str, ...] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        normalized = base_uri.rstrip("/")
        if not normalized.startswith("gs://") or normalized.count("/") < 2:
            raise ValueError("base_uri must be a gs:// bucket URI")
        if command_prefix is None:
            executable = shutil.which("gcloud")
            if executable is None:
                raise ObjectStoreError("gcloud executable is not available")
            command_prefix = (executable,)
        if not command_prefix:
            raise ValueError("command_prefix must not be empty")
        object.__setattr__(self, "base_uri", normalized)
        object.__setattr__(self, "command_prefix", command_prefix)
        object.__setattr__(self, "environment", {**os.environ, **(environment or {})})

    def put_create_only(self, source: Path, key: str) -> None:
        uri = self._uri(key)
        result = self._run(("storage", "cp", str(source), uri, "--if-generation-match=0", "--quiet"))
        if result.returncode != 0:
            with tempfile.TemporaryDirectory(prefix="gcs-existing.") as directory:
                existing = Path(directory) / source.name
                try:
                    self.download(key, existing)
                except ObjectStoreError as exc:
                    raise ObjectStoreError(f"create-only upload failed for {uri}: {result.stderr.strip()}") from exc
                if _sha256_file(existing) != _sha256_file(source):
                    raise ObjectConflictError(f"remote object already exists with different content: {uri}")
        if self.metadata(key).size != source.stat().st_size:
            raise ObjectStoreError(f"remote object size mismatch after upload: {uri}")

    def download(self, key: str, destination: Path) -> None:
        self.metadata(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)[1])
        try:
            result = self._run(("storage", "cp", self._uri(key), str(temporary), "--quiet"))
            if result.returncode != 0:
                raise ObjectStoreError(f"download failed for {self._uri(key)}: {result.stderr.strip()}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def metadata(self, key: str) -> ObjectMetadata:
        uri = self._uri(key)
        result = self._run(("storage", "objects", "describe", uri, "--format=json"))
        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "not found" in stderr or "404" in stderr or "matched no objects" in stderr:
                raise ObjectNotFoundError(f"remote object does not exist: {uri}")
            raise ObjectStoreError(f"cannot inspect remote object {uri}: {result.stderr.strip()}")
        try:
            document = json.loads(result.stdout)
            size = int(document["size"])
            generation = str(document["generation"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ObjectStoreError(f"invalid gcloud metadata response for {uri}") from exc
        return ObjectMetadata(size=size, generation=generation)

    def _uri(self, key: str) -> str:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"unsafe object key: {key!r}")
        return f"{self.base_uri}/{key}"

    def _run(self, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - executable is resolved or explicitly injected; argv is not shell text
            (*self.command_prefix, *arguments),
            check=False,
            capture_output=True,
            env=self.environment,
            text=True,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

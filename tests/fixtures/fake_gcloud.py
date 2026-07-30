from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _object_path(uri: str) -> Path:
    if not uri.startswith("gs://"):
        raise ValueError(f"not a GCS URI: {uri}")
    return Path(os.environ["FAKE_GCS_ROOT"]) / uri.removeprefix("gs://")


def _copy(arguments: list[str]) -> int:
    source, destination = arguments[2], arguments[3]
    create_only = "--if-generation-match=0" in arguments
    if source.startswith("gs://"):
        source_path = _object_path(source)
        if not source_path.is_file():
            print("404 Not Found", file=sys.stderr)
            return 1
        destination_path = Path(destination)
    else:
        source_path = Path(source)
        destination_path = _object_path(destination)
        if create_only and destination_path.exists():
            print("412 Precondition Failed", file=sys.stderr)
            return 1
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return 0


def _describe(arguments: list[str]) -> int:
    path = _object_path(arguments[3])
    if not path.is_file():
        print("404 Not Found", file=sys.stderr)
        return 1
    stat = path.stat()
    print(json.dumps({"generation": str(stat.st_mtime_ns), "size": str(stat.st_size)}))
    return 0


def main() -> int:
    arguments = sys.argv[1:]
    if arguments[:2] == ["storage", "cp"]:
        return _copy(arguments)
    if arguments[:3] == ["storage", "objects", "describe"]:
        return _describe(arguments)
    print(f"unsupported fake gcloud command: {arguments}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

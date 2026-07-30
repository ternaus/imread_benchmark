from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    log_path = Path(os.environ["FAKE_UV_LOG"])
    calls = json.loads(log_path.read_text()) if log_path.exists() else []
    calls.append({"argv": sys.argv[1:], "environment": os.environ["UV_PROJECT_ENVIRONMENT"]})
    log_path.write_text(json.dumps(calls))
    environment = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
    bindir = environment / ("Scripts" if sys.platform == "win32" else "bin")
    bindir.mkdir(parents=True)
    target = bindir / ("python.exe" if sys.platform == "win32" else "python")
    target.symlink_to(sys.executable)


if __name__ == "__main__":
    main()

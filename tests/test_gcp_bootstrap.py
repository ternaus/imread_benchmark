from __future__ import annotations

import subprocess
from pathlib import Path


def test_gcp_shell_is_syntax_valid_and_delegates_benchmark_semantics_to_python() -> None:
    root = Path(__file__).parents[1]
    scripts = (root / "gcp" / "run.sh", root / "gcp" / "run-many.sh", root / "gcp" / "vm_startup.sh")

    for script in scripts:
        subprocess.run(("/bin/bash", "-n", str(script)), check=True)  # noqa: S603

    startup = scripts[-1].read_text()
    assert "dataset materialize" in startup
    assert "environment provision" in startup
    assert "campaign run" in startup
    assert "DONE.json" in startup

    launcher = scripts[0].read_text()
    assert 'MACHINE_TYPE="${MACHINE_TYPE:-c4-standard-16}"' in launcher
    assert "c3-standard-16" not in launcher

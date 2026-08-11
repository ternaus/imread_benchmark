from __future__ import annotations

import subprocess
from pathlib import Path


def test_gcp_shell_is_syntax_valid_and_delegates_benchmark_semantics_to_python() -> None:
    root = Path(__file__).parents[1]
    scripts = (root / "gcp" / "run.sh", root / "gcp" / "run-many.sh", root / "gcp" / "vm_startup.sh")
    native_installer = root / "scripts" / "install-libjpeg-turbo.sh"

    for script in (*scripts, native_installer):
        subprocess.run(("/bin/bash", "-n", str(script)), check=True)  # noqa: S603

    startup = scripts[-1].read_text()
    assert "dataset materialize" in startup
    assert "environment provision" in startup
    assert "campaign run" in startup
    assert "DONE.json" in startup
    assert "ulimit -n 65535" in startup
    assert "/snap/google-cloud-cli/current/bin/gcloud" in startup

    launcher = scripts[0].read_text()
    assert 'MACHINE_TYPE="${MACHINE_TYPE:-c4-standard-16}"' in launcher
    assert "c3-standard-16" not in launcher

    installer = native_installer.read_text()
    assert "LIBJPEG_TURBO_VERSION=" in installer
    assert "libjpeg-turbo-official_${LIBJPEG_TURBO_VERSION}_amd64.deb" in installer
    assert "libjpeg-turbo-official_${LIBJPEG_TURBO_VERSION}_arm64.deb" in installer
    assert "sha256sum --check --strict" in installer
    assert "scripts/install-libjpeg-turbo.sh" in startup
    assert "libjpeg-turbo8-dev" not in startup
    assert '--native-backend "libjpeg-turbo=$LIBJPEG_TURBO_BACKEND"' in startup

    project = (root / "pyproject.toml").read_text()
    assert '"pyturbojpeg",' in project
    assert '"pyturbojpeg<2"' not in project

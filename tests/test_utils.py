from __future__ import annotations

from imread_benchmark.utils import collect_jpeg_paths, get_system_identifier


def test_get_system_identifier_is_nonempty():
    ident = get_system_identifier()
    assert isinstance(ident, str)
    assert len(ident) > 0


def test_get_system_identifier_contains_os():
    ident = get_system_identifier().lower()
    assert "darwin" in ident or "linux" in ident or "unknown" in ident


def test_collect_jpeg_paths_returns_subset(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.jpeg").write_bytes(b"fake")
    (tmp_path / "c.png").write_bytes(b"fake")

    paths = collect_jpeg_paths(tmp_path, num_images=10)
    assert len(paths) == 2
    assert all(p.suffix.lower() in {".jpg", ".jpeg"} for p in paths)


def test_collect_jpeg_paths_respects_limit(tmp_path):
    for i in range(5):
        (tmp_path / f"{i}.jpg").write_bytes(b"fake")

    paths = collect_jpeg_paths(tmp_path, num_images=3)
    assert len(paths) == 3

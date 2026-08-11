from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imread_benchmark.analysis.canonical import RunBundleRecord
    from imread_benchmark.analysis.claims import ClaimScope

__all__ = ["ClaimScope", "RunBundleRecord", "assert_claim_scope", "load_bundles", "publish"]


def __getattr__(name: str) -> Any:
    if name in {"RunBundleRecord", "load_bundles"}:
        from imread_benchmark.analysis import canonical

        return getattr(canonical, name)
    if name in {"ClaimScope", "assert_claim_scope"}:
        from imread_benchmark.analysis import claims

        return getattr(claims, name)
    if name == "publish":
        from imread_benchmark.analysis.publication import publish

        return publish
    raise AttributeError(name)

"""Tests for ``src.util.environment``.

The module is best-effort by design (never raises, returns ``None`` for
fields it cannot determine), so the tests pin down the *shape* of the
returned dict rather than concrete values that vary by host.
"""

from __future__ import annotations

import os

from src.util import environment as env_mod
from src.util.environment import capture_environment


def test_capture_returns_dict_with_required_top_level_keys():
    out = capture_environment()
    assert isinstance(out, dict)
    for key in ("python", "platform", "git_commit", "cpu", "mem_gb", "packages"):
        assert key in out, f"missing top-level key: {key}"


def test_cpu_block_shape():
    """``cpu`` is a dict with ``brand`` (string or None) and
    ``logical_count`` (int or None)."""
    out = capture_environment()
    cpu = out["cpu"]
    assert isinstance(cpu, dict)
    assert set(cpu.keys()) >= {"brand", "logical_count"}
    assert cpu["brand"] is None or isinstance(cpu["brand"], str)
    assert cpu["logical_count"] is None or isinstance(cpu["logical_count"], int)
    # If ``os.cpu_count`` returned a value, the env capture should pass it
    # through faithfully.
    if os.cpu_count() is not None:
        assert cpu["logical_count"] == os.cpu_count()


def test_mem_gb_is_float_or_none():
    out = capture_environment()
    mem = out["mem_gb"]
    assert mem is None or isinstance(mem, float)
    if isinstance(mem, float):
        # Sanity: at least 0.5 GB (very generous lower bound for any
        # machine that can run pytest), at most 100 TB.
        assert 0.5 <= mem <= 100_000.0


def test_packages_block_is_dict_of_strings():
    out = capture_environment()
    pkgs = out["packages"]
    assert isinstance(pkgs, dict)
    for name, version in pkgs.items():
        assert isinstance(name, str)
        assert isinstance(version, str)


def test_new_packages_in_default_tuple():
    """The default ``package_names`` tuple should include the four
    packages added by infra(deps): sentence-transformers, bert-score,
    faiss-cpu, pyarrow. These are runtime-load-bearing for the
    retrieval / generation / grounding pipelines and were previously
    invisible to the manifest."""
    import inspect

    sig = inspect.signature(capture_environment)
    default_pkgs = sig.parameters["package_names"].default
    assert "sentence-transformers" in default_pkgs
    assert "bert-score" in default_pkgs
    assert "faiss-cpu" in default_pkgs
    assert "pyarrow" in default_pkgs
    # Pre-existing entries are still there.
    assert "bm25s" in default_pkgs
    assert "transformers" in default_pkgs


def test_capture_never_raises(monkeypatch):
    """``capture_environment`` is documented to never raise. Force the
    underlying helpers into their failure paths and verify the function
    still returns a well-shaped dict."""
    monkeypatch.setattr(env_mod, "_cpu_brand", lambda: None)
    monkeypatch.setattr(env_mod, "_total_mem_bytes", lambda: None)
    monkeypatch.setattr(env_mod, "_git_commit", lambda: None)

    out = capture_environment()
    assert isinstance(out, dict)
    assert out["cpu"]["brand"] is None
    assert out["mem_gb"] is None
    assert out["git_commit"] is None
    # The blocks still exist even when their values are None.
    assert "logical_count" in out["cpu"]


def test_safe_version_returns_none_for_unknown_pkg():
    """The version lookup must never raise on a missing package."""
    assert env_mod._safe_version("definitely-not-an-installed-package") is None


def test_custom_package_names_filter_is_applied():
    """When the caller passes a custom ``package_names`` tuple, only
    those packages appear (and only the installed ones)."""
    out = capture_environment(package_names=("numpy", "this-package-is-fake-xyz"))
    pkgs = out["packages"]
    # numpy is a hard dep of the project; we can assert it's present.
    assert "numpy" in pkgs
    assert "this-package-is-fake-xyz" not in pkgs

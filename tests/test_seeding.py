"""Tests for src.util.seeding.set_global_seed.

The behavioural contract is "all available RNG sinks land at the same
state after two equal-seed calls". We test that by drawing samples
before and after re-seeding, not by mocking the sink functions —
mocks would prove we *called* the seed functions but not that they
*work*.
"""

from __future__ import annotations

import random

import pytest

from src.util.seeding import set_global_seed


# --------------------------------------------------------------------------- #
# Coverage dict shape
# --------------------------------------------------------------------------- #


def test_returns_coverage_dict_with_all_four_sinks():
    coverage = set_global_seed(42, log=False)
    assert set(coverage.keys()) == {"random", "numpy", "torch", "transformers"}
    # Each value is a non-empty string ("ok" or "skipped: ...").
    for sink, status in coverage.items():
        assert isinstance(status, str)
        assert status, f"empty status for {sink}"


def test_random_sink_is_always_ok():
    # stdlib ``random`` is the one sink with no optional-dep risk.
    coverage = set_global_seed(42, log=False)
    assert coverage["random"] == "ok"


# --------------------------------------------------------------------------- #
# Per-sink determinism — re-seed twice, draw, compare
# --------------------------------------------------------------------------- #


def test_stdlib_random_is_deterministic_after_set_global_seed():
    set_global_seed(42, log=False)
    a = [random.random() for _ in range(8)]
    set_global_seed(42, log=False)
    b = [random.random() for _ in range(8)]
    assert a == b


def test_numpy_random_is_deterministic_after_set_global_seed():
    np = pytest.importorskip("numpy")
    set_global_seed(42, log=False)
    a = np.random.rand(8).tolist()
    set_global_seed(42, log=False)
    b = np.random.rand(8).tolist()
    assert a == b


def test_torch_random_is_deterministic_after_set_global_seed():
    torch = pytest.importorskip("torch")
    set_global_seed(42, log=False)
    a = torch.rand(8).tolist()
    set_global_seed(42, log=False)
    b = torch.rand(8).tolist()
    assert a == b


def test_torch_cudnn_determinism_flag_is_set():
    torch = pytest.importorskip("torch")
    set_global_seed(42, log=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


# --------------------------------------------------------------------------- #
# Cross-seed difference — sanity check that the seed value matters
# --------------------------------------------------------------------------- #


def test_different_seeds_produce_different_samples():
    set_global_seed(42, log=False)
    a = random.random()
    set_global_seed(43, log=False)
    b = random.random()
    assert a != b

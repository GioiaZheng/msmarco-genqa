"""Tests for ``src.retrieval.sampling.qrels_anchored_sample``."""

from __future__ import annotations

import pytest

from src.retrieval.sampling import qrels_anchored_sample


def test_includes_all_relevant_doc_ids():
    pool = [f"d{i}" for i in range(1000)]
    qrels = {"q1": {"d10", "d20"}, "q2": {"d30"}}
    sample = qrels_anchored_sample(pool, qrels, target_size=50, seed=42)
    assert {"d10", "d20", "d30"}.issubset(set(sample))
    assert len(sample) == 50


def test_deterministic_for_same_seed():
    pool = [f"d{i}" for i in range(1000)]
    qrels = {"q1": {"d5"}}
    a = qrels_anchored_sample(pool, qrels, target_size=20, seed=42)
    b = qrels_anchored_sample(pool, qrels, target_size=20, seed=42)
    assert a == b


def test_different_seed_changes_sample():
    pool = [f"d{i}" for i in range(1000)]
    qrels = {"q1": {"d5"}}
    a = qrels_anchored_sample(pool, qrels, target_size=20, seed=42)
    b = qrels_anchored_sample(pool, qrels, target_size=20, seed=43)
    assert a != b
    # but both still contain the relevant doc
    assert "d5" in a and "d5" in b


def test_target_smaller_than_relevants_raises():
    pool = [f"d{i}" for i in range(100)]
    qrels = {"q1": set(f"d{i}" for i in range(20))}
    with pytest.raises(ValueError, match="smaller than"):
        qrels_anchored_sample(pool, qrels, target_size=10, seed=42)


def test_target_geq_pool_returns_full_pool():
    pool = [f"d{i}" for i in range(50)]
    qrels = {"q1": {"d5"}}
    sample = qrels_anchored_sample(pool, qrels, target_size=100, seed=42)
    assert sorted(sample) == sorted(pool)


def test_relevant_not_in_pool_raises():
    pool = [f"d{i}" for i in range(50)]
    qrels = {"q1": {"d999"}}  # not in pool
    with pytest.raises(ValueError, match="missing from pool"):
        qrels_anchored_sample(pool, qrels, target_size=20, seed=42)


def test_empty_qrels_returns_pure_random_sample():
    pool = [f"d{i}" for i in range(100)]
    sample = qrels_anchored_sample(pool, qrels={}, target_size=30, seed=42)
    assert len(sample) == 30
    # Sorted deterministic
    assert sample == sorted(sample)


def test_reproducible_across_processes():
    """The sampler must give the same output for the same (seed, qrels, pool)
    inputs regardless of process-level state.

    Earlier versions converted ``pool_set - rel_ids`` to a list directly, so
    distractor ordering depended on Python's randomised hash seed. The fix
    is to ``sorted(...)`` before sampling. We force two distinct hash seeds
    via a subprocess and check the saved doc_ids match.
    """
    import json
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json, sys
        from src.retrieval.sampling import qrels_anchored_sample
        pool = [f"d{i}" for i in range(2000)]
        qrels = {"q1": {"d10", "d20", "d30"}}
        out = qrels_anchored_sample(pool, qrels, target_size=200, seed=42)
        print(json.dumps(out))
        """
    )
    project_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    a = subprocess.run(
        [sys.executable, "-c", script],
        env={"PYTHONHASHSEED": "1", "PYTHONPATH": project_root},
        capture_output=True, check=True,
    )
    b = subprocess.run(
        [sys.executable, "-c", script],
        env={"PYTHONHASHSEED": "999999", "PYTHONPATH": project_root},
        capture_output=True, check=True,
    )
    assert json.loads(a.stdout.strip().splitlines()[-1]) == json.loads(b.stdout.strip().splitlines()[-1])

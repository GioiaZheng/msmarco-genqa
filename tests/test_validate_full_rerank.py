"""Tests for ``scripts/validate_full_rerank.py``.

Synthetic ``run.tsv`` + ``manifest.json`` files in ``tmp_path`` — no
network, no model. The script's job is to surface broken invariants
clearly, so the tests cover both the green path and the four failure
modes the user listed (qid count, max rank, duplicates, manifest
missing resume fields).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "validate_full_rerank",
    PROJECT_ROOT / "scripts" / "validate_full_rerank.py",
)
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def _write_block(path: Path, qid: str, top_k: int) -> None:
    """Append a complete top-K block for ``qid`` to ``path``."""
    with open(path, "a") as f:
        for rank in range(1, top_k + 1):
            f.write(f"{qid}\tQ0\td_{rank}\t{rank}\t{1.0/rank:.6f}\tce\n")


def test_validate_run_tsv_happy_path(tmp_path: Path):
    p = tmp_path / "run.tsv"
    for qid in ("q1", "q2", "q3"):
        _write_block(p, qid, top_k=5)
    stats, errors = validator.validate_run_tsv(p, top_k=5)
    assert errors == []
    assert stats["n_qids"] == 3
    assert stats["n_incomplete"] == 0
    assert stats["n_duplicates"] == 0
    assert stats["max_rank_observed"] == 5


def test_validate_run_tsv_missing_file_is_error(tmp_path: Path):
    stats, errors = validator.validate_run_tsv(tmp_path / "absent.tsv", top_k=5)
    assert stats == {}
    assert errors and "missing" in errors[0]


def test_validate_run_tsv_detects_incomplete_qid(tmp_path: Path):
    p = tmp_path / "run.tsv"
    _write_block(p, "q1", top_k=5)
    # q2 only has rank 1 — partial.
    with open(p, "a") as f:
        f.write("q2\tQ0\td_1\t1\t0.5\tce\n")
    stats, errors = validator.validate_run_tsv(p, top_k=5)
    assert stats["n_incomplete"] == 1
    assert any("incomplete top-5" in e for e in errors)


def test_validate_run_tsv_detects_duplicates(tmp_path: Path):
    p = tmp_path / "run.tsv"
    _write_block(p, "q1", top_k=3)
    # Double-write rank 2 — simulates a buggy resume double-append.
    with open(p, "a") as f:
        f.write("q1\tQ0\td_2_dup\t2\t0.5\tce\n")
    stats, errors = validator.validate_run_tsv(p, top_k=3)
    assert stats["n_duplicates"] == 1
    assert any("duplicate" in e for e in errors)


def test_validate_run_tsv_detects_max_rank_mismatch(tmp_path: Path):
    """A qid whose max rank exceeds top_k (or has gaps) is incomplete."""
    p = tmp_path / "run.tsv"
    # 3 lines but with ranks 1, 3, 5 — max rank 5 ≠ top_k 3.
    with open(p, "a") as f:
        for rank in (1, 3, 5):
            f.write(f"q1\tQ0\td_{rank}\t{rank}\t0.5\tce\n")
    stats, errors = validator.validate_run_tsv(p, top_k=3)
    assert stats["n_incomplete"] == 1


def test_validate_manifest_happy_path(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({
        "extra": {
            "chunk_size": 200,
            "resumed": True,
            "n_resumed_qids": 1000,
            "n_pending_this_run": 5980,
            "n_eval_queries": 6980,
        }
    }))
    stats, errors = validator.validate_manifest(p)
    assert errors == []
    assert stats["chunk_size"] == 200
    assert stats["resumed"] is True


def test_validate_manifest_missing_fields(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"extra": {"some_other_field": True}}))
    stats, errors = validator.validate_manifest(p)
    assert errors and "missing resume/chunking" in errors[0]


def test_validate_manifest_missing_file(tmp_path: Path):
    stats, errors = validator.validate_manifest(tmp_path / "absent.json")
    assert stats == {}
    assert any("missing" in e for e in errors)


def test_validate_manifest_invalid_json(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text("{ not json")
    stats, errors = validator.validate_manifest(p)
    assert any("not valid JSON" in e for e in errors)

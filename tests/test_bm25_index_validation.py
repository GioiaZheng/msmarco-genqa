"""Tests for the R1 fix: BM25Retriever.load() validates expected_config.

We don't actually build a real bm25s index — that's slow and exercised by the
end-to-end smoke tests. Here we just write the two small JSON files the
loader inspects (``doc_ids.json`` and ``config.json``) and verify the
validation logic.
"""

from __future__ import annotations

import json

import pytest

from src.retrieval.bm25 import BM25IndexConfigMismatch, BM25Retriever


def _fake_index_dir(tmp_path, k1=1.5, b=0.75, stopwords="en"):
    p = tmp_path / "fake_index"
    p.mkdir()
    (p / "doc_ids.json").write_text(json.dumps(["d1", "d2"]))
    (p / "config.json").write_text(
        json.dumps({"k1": k1, "b": b, "stopwords": stopwords})
    )
    return p


def test_load_raises_on_k1_mismatch(tmp_path, monkeypatch):
    # Stub bm25s.BM25.load so we don't actually need a real index.
    import bm25s

    class _StubBM25:
        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr(bm25s, "BM25", _StubBM25)

    idx = _fake_index_dir(tmp_path, k1=1.5, b=0.75)
    with pytest.raises(BM25IndexConfigMismatch) as ei:
        BM25Retriever.load(idx, expected_config={"k1": 2.0, "b": 0.75, "stopwords": "en"})
    msg = str(ei.value)
    assert "k1" in msg
    assert "1.5" in msg and "2.0" in msg
    assert "--rebuild-index" in msg


def test_load_raises_on_stopwords_mismatch(tmp_path, monkeypatch):
    import bm25s

    class _StubBM25:
        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr(bm25s, "BM25", _StubBM25)

    idx = _fake_index_dir(tmp_path, stopwords="en")
    with pytest.raises(BM25IndexConfigMismatch) as ei:
        BM25Retriever.load(idx, expected_config={"k1": 1.5, "b": 0.75, "stopwords": None})
    assert "stopwords" in str(ei.value)


def test_load_passes_when_config_matches(tmp_path, monkeypatch):
    import bm25s

    class _StubBM25:
        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr(bm25s, "BM25", _StubBM25)

    idx = _fake_index_dir(tmp_path, k1=1.5, b=0.75, stopwords="en")
    r = BM25Retriever.load(idx, expected_config={"k1": 1.5, "b": 0.75, "stopwords": "en"})
    assert r.k1 == 1.5
    assert r.b == 0.75
    assert r.stopwords == "en"
    assert r.doc_ids == ["d1", "d2"]


def test_load_skips_validation_when_no_expected_config(tmp_path, monkeypatch):
    """Backward compat: callers that don't pass expected_config should still work."""
    import bm25s

    class _StubBM25:
        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr(bm25s, "BM25", _StubBM25)

    idx = _fake_index_dir(tmp_path, k1=2.0, b=0.5)
    # No raise even though k1/b are unusual.
    r = BM25Retriever.load(idx)
    assert r.k1 == 2.0
    assert r.b == 0.5


def test_load_only_validates_keys_present_in_expected(tmp_path, monkeypatch):
    """If only k1 is in expected_config, b/stopwords mismatches are tolerated."""
    import bm25s

    class _StubBM25:
        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr(bm25s, "BM25", _StubBM25)

    idx = _fake_index_dir(tmp_path, k1=1.5, b=0.75, stopwords="en")
    # Only k1 in expected, and it matches; b/stopwords absent from expected
    # so they're not checked.
    r = BM25Retriever.load(idx, expected_config={"k1": 1.5})
    assert r.k1 == 1.5

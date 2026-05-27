"""Tests for ``msmarco_genqa.retrieval.dense``.

We monkey-patch the encoder to avoid downloading SBERT weights, but use
**real** FAISS — it's lightweight and the index correctness is part of
what we want to test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from msmarco_genqa.retrieval.dense import DenseIndexConfigMismatch, DenseRetriever


class _StubEncoder:
    """Tiny deterministic encoder: maps each unique token to a stable basis vector.

    Used so tests don't depend on SentenceTransformer's network state.
    Uses md5 (not Python ``hash()``) because the latter is randomised per
    process on str inputs and would make tests flaky across runs / save+load.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True):
        import hashlib

        def _idx(tok: str) -> int:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            return int.from_bytes(h[:4], "little") % self.dim

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, _idx(tok)] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out.astype("float32")


def _make_retriever(monkeypatch):
    r = DenseRetriever(model_name="stub", device="cpu", normalize=True)
    # Bypass _ensure_model
    r._model = _StubEncoder(dim=8)
    r._embedding_dim = 8
    return r


def test_build_and_retrieve_top1(monkeypatch):
    r = _make_retriever(monkeypatch)
    corpus = [
        "the eiffel tower is in paris",
        "canberra is the capital of australia",
        "william shakespeare wrote hamlet",
    ]
    doc_ids = ["d0", "d1", "d2"]
    r.build(corpus, doc_ids)

    # Query that overlaps strongly with d0
    scores, top = r.retrieve("paris eiffel tower", k=3)
    assert top[0] == "d0"
    # IP scores should be in [0, 1] for normalised vectors and decreasing.
    s_list = [float(s) for s in scores]
    assert s_list == sorted(s_list, reverse=True)


def test_retrieve_batch_shape(monkeypatch):
    r = _make_retriever(monkeypatch)
    corpus = ["alpha beta", "beta gamma", "gamma delta"]
    r.build(corpus, ["d0", "d1", "d2"])
    scores, doc_ids_lists = r.retrieve_batch(["alpha", "delta"], k=2)
    assert scores.shape == (2, 2)
    assert len(doc_ids_lists) == 2
    assert all(len(row) == 2 for row in doc_ids_lists)


def test_retrieve_k_capped_to_corpus_size(monkeypatch):
    r = _make_retriever(monkeypatch)
    r.build(["one two", "three four"], ["d0", "d1"])
    scores, top = r.retrieve("one", k=100)  # k > corpus size
    # IndexFlatIP returns at most ntotal results
    assert len(top) == 2
    assert scores.shape == (2,)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    r = _make_retriever(monkeypatch)
    r.build(["foo bar", "bar baz", "baz qux"], ["a", "b", "c"])
    idx = tmp_path / "idx"
    r.save(idx)

    # Load back; we have to re-stub because load instantiates a new
    # DenseRetriever which would otherwise try to load real SBERT weights.
    loaded = DenseRetriever.load(idx)
    loaded._model = _StubEncoder(dim=8)
    scores, top = loaded.retrieve("foo bar", k=2)
    assert top[0] == "a"


def test_load_raises_on_model_mismatch(tmp_path, monkeypatch):
    r = _make_retriever(monkeypatch)
    r.model_name = "stub-A"
    r.build(["hello"], ["d0"])
    idx = tmp_path / "idx"
    r.save(idx)

    with pytest.raises(DenseIndexConfigMismatch) as ei:
        DenseRetriever.load(idx, expected_config={"model_name": "stub-B"})
    msg = str(ei.value)
    assert "model_name" in msg
    assert "stub-A" in msg and "stub-B" in msg
    assert "--rebuild-index" in msg


def test_load_skips_validation_when_no_expected(tmp_path, monkeypatch):
    r = _make_retriever(monkeypatch)
    r.model_name = "stub-A"
    r.build(["hello"], ["d0"])
    idx = tmp_path / "idx"
    r.save(idx)
    # No raise even though we never set expected_config
    DenseRetriever.load(idx)


def test_save_writes_expected_files(tmp_path, monkeypatch):
    r = _make_retriever(monkeypatch)
    r.build(["x"], ["d0"])
    idx = tmp_path / "idx"
    r.save(idx)
    assert (idx / "index.faiss").exists()
    assert (idx / "doc_ids.json").exists()
    assert (idx / "config.json").exists()
    cfg = json.loads((idx / "config.json").read_text())
    assert cfg["metric"] == "ip"
    assert cfg["normalize"] is True

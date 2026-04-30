"""BM25 retrieval using the ``bm25s`` pure-Python backend.

``bm25s`` is significantly faster than ``rank_bm25`` and supports save/load
of the index. This module wraps it in a small class with a stable API:

    retriever = BM25Retriever(corpus_texts, doc_ids).build()
    retriever.save(index_dir)
    retriever = BM25Retriever.load(index_dir)
    scores, doc_ids = retriever.retrieve(query, k=1000)
    scores, doc_ids_lists = retriever.retrieve_batch(queries, k=1000)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Thin wrapper around ``bm25s.BM25`` with doc_id mapping and persistence."""

    def __init__(
        self,
        corpus_texts: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | None = "en",
    ):
        self.corpus_texts = list(corpus_texts) if corpus_texts is not None else None
        self.doc_ids = list(doc_ids) if doc_ids is not None else None
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self._bm25 = None  # bm25s.BM25 instance

    def build(self) -> "BM25Retriever":
        if self.corpus_texts is None:
            raise ValueError("corpus_texts is required to build the index")
        if self.doc_ids is None:
            raise ValueError("doc_ids is required to map results back to corpus ids")
        if len(self.doc_ids) != len(self.corpus_texts):
            raise ValueError("doc_ids and corpus_texts must have equal length")

        import bm25s

        logger.info("Tokenizing %d passages...", len(self.corpus_texts))
        t0 = time.time()
        tokens = bm25s.tokenize(self.corpus_texts, stopwords=self.stopwords)
        logger.info("Tokenized in %.1f s.", time.time() - t0)

        logger.info("Building BM25 index (k1=%.2f, b=%.2f)...", self.k1, self.b)
        t0 = time.time()
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(tokens)
        logger.info("Indexed in %.1f s.", time.time() - t0)
        return self

    def save(self, path: Path | str) -> None:
        if self._bm25 is None:
            raise ValueError("No index built yet")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path))
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        with open(path / "config.json", "w") as f:
            json.dump(
                {"k1": self.k1, "b": self.b, "stopwords": self.stopwords},
                f,
                indent=2,
            )
        logger.info("Saved index to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Retriever":
        import bm25s

        path = Path(path)
        with open(path / "doc_ids.json") as f:
            doc_ids = json.load(f)
        with open(path / "config.json") as f:
            cfg = json.load(f)
        retriever = cls(
            corpus_texts=None,
            doc_ids=doc_ids,
            k1=cfg["k1"],
            b=cfg["b"],
            stopwords=cfg.get("stopwords"),
        )
        retriever._bm25 = bm25s.BM25.load(str(path))
        return retriever

    def retrieve(self, query: str, k: int = 1000) -> tuple[np.ndarray, list[str]]:
        scores, doc_ids_lists = self.retrieve_batch([query], k=k)
        return scores[0], doc_ids_lists[0]

    def retrieve_batch(
        self,
        queries: Sequence[str],
        k: int = 1000,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Retrieve top-k for a batch of queries.

        Returns
        -------
        scores : np.ndarray of shape (n_queries, k)
        doc_ids_lists : list of length n_queries, each item is a list of k doc_id strings
        """
        if self._bm25 is None:
            raise ValueError("No index built/loaded")
        import bm25s

        tokens = bm25s.tokenize(list(queries), stopwords=self.stopwords)
        results, scores = self._bm25.retrieve(tokens, k=k)
        # results: (n_queries, k) of doc indices into the original corpus order
        doc_ids_lists = [
            [self.doc_ids[int(i)] for i in row] for row in results
        ]
        return scores, doc_ids_lists

"""Dense retrieval with Sentence-Transformers + FAISS.

A small wrapper that mirrors :class:`src.retrieval.bm25.BM25Retriever`'s
API so the experiment scripts and evaluation code can stay
retriever-agnostic:

    retriever = DenseRetriever(model_name=...).build(corpus_texts, doc_ids)
    retriever.save(index_dir)
    retriever = DenseRetriever.load(index_dir)
    scores, doc_ids = retriever.retrieve(query, k=1000)
    scores, doc_ids_lists = retriever.retrieve_batch(queries, k=1000)

Index choice: ``faiss.IndexFlatIP`` over L2-normalised embeddings → exact
inner-product search ≡ cosine similarity. We deliberately avoid IVF/HNSW
for the Week 4 sampled baseline so the only thing measured is the
quality of the encoder, not the ANN approximation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class DenseIndexConfigMismatch(ValueError):
    """Raised when a cached dense index was built with a different model."""


def _force_faiss_single_threaded(faiss_module) -> None:
    """Pin faiss's OpenMP to 1 thread.

    On macOS, ``faiss-cpu`` ships its own ``libomp.dylib`` and ``torch`` ships
    ``libiomp5.dylib``. With multiple OpenMP runtimes loaded in the same
    process, ``faiss::IndexFlat::search`` deadlocks during the OMP join:
    ``__kmpc_fork_call`` is dispatched through faiss's libomp but the
    workers' condition variables get parked in libiomp5, so the join wait
    is never signalled. ``KMP_DUPLICATE_LIB_OK=TRUE`` silences the *abort*
    but not the deadlock.
    Forcing faiss to a single thread sidesteps the OMP fork entirely, at
    the cost of single-threaded search. Search on 50k normalised vectors
    is still well under a second per query at this scale.
    """
    try:
        faiss_module.omp_set_num_threads(1)
    except AttributeError:
        # Older faiss-cpu builds don't expose ``omp_set_num_threads``; if
        # the helper is missing, we accept the deadlock risk and let the
        # user investigate.
        logger.warning("faiss has no omp_set_num_threads; cannot pin threads")


class DenseRetriever:
    """Sentence-Transformers + FAISS dense retriever (cosine via IP)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        encode_batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device  # resolved lazily in _ensure_model
        self.encode_batch_size = encode_batch_size
        self.normalize = normalize

        # Populated by build/load.
        self.doc_ids: list[str] | None = None
        self._index = None  # faiss.Index
        self._model = None  # sentence_transformers.SentenceTransformer
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------ #
    # Encoder
    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        try:
            import torch
            # Force eager init of torch's CPU thread pool. Without this, on
            # macOS the pool is initialised lazily on first encode, which
            # crashes (SIGSEGV) when called after ``ir_datasets`` has been
            # used in the same process — likely an interaction with the file
            # descriptors / signal handlers ir_datasets sets up. Setting the
            # thread count explicitly (even to the existing default) sidesteps
            # the bad lazy path.
            torch.set_num_threads(torch.get_num_threads())
        except ImportError:
            pass

        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        logger.info("Loading dense encoder %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode ``texts`` to a (n, dim) float32 array, optionally L2-normalised."""
        self._ensure_model()

        # Re-assert torch threadpool config at every encode call, not just at
        # model-load time. ``bm25s`` (and other libraries that use
        # multiprocessing) can leave torch's thread state in a wedged form
        # such that the *next* encode SIGSEGVs on macOS. Asserting threads
        # before each encode call is cheap and idempotent.
        try:
            import torch
            torch.set_num_threads(torch.get_num_threads())
        except ImportError:
            pass

        bs = batch_size or self.encode_batch_size
        emb = self._model.encode(
            list(texts),
            batch_size=bs,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        # SentenceTransformer returns float32 by default; be defensive.
        return emb.astype("float32", copy=False)

    # ------------------------------------------------------------------ #
    # Build / save / load
    # ------------------------------------------------------------------ #

    def build(
        self,
        corpus_texts: Sequence[str],
        doc_ids: Sequence[str],
    ) -> "DenseRetriever":
        if len(corpus_texts) != len(doc_ids):
            raise ValueError("corpus_texts and doc_ids must have the same length")

        # Important: load the SBERT encoder BEFORE faiss. On macOS, faiss
        # ships its own libomp.dylib and importing it first interferes with
        # torch's lazy OpenMP init, leading to a SIGSEGV at the first encode
        # batch when called after ir_datasets has been used in the same
        # process. ``_ensure_model`` also calls ``torch.set_num_threads`` to
        # force eager pool init.
        self._ensure_model()

        import faiss
        _force_faiss_single_threaded(faiss)

        self.doc_ids = list(doc_ids)
        logger.info("Encoding %d passages with %s ...", len(corpus_texts), self.model_name)
        t0 = time.time()
        emb = self.encode(corpus_texts)
        logger.info(
            "Encoded %d passages in %.1f s (%.1f passages/s).",
            len(corpus_texts),
            time.time() - t0,
            len(corpus_texts) / max(time.time() - t0, 1e-6),
        )

        self._embedding_dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(emb)
        return self

    def save(self, path: Path | str) -> None:
        if self._index is None:
            raise ValueError("No index built yet")
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        with open(path / "config.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "embedding_dim": self._embedding_dim,
                    "normalize": self.normalize,
                    "metric": "ip",  # IP over normalised embeddings == cosine
                },
                f,
                indent=2,
            )
        logger.info("Saved dense index to %s", path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        device: str | None = None,
        encode_batch_size: int = 32,
        expected_config: dict | None = None,
    ) -> "DenseRetriever":
        """Load a saved FAISS index.

        ``expected_config`` mirrors ``BM25Retriever.load``: if provided, keys
        like ``model_name`` / ``normalize`` are checked against the cached
        ``config.json`` and ``DenseIndexConfigMismatch`` is raised on any
        mismatch (so swapping the encoder in YAML doesn't silently reuse the
        old index).
        """
        import faiss

        path = Path(path)
        with open(path / "doc_ids.json") as f:
            doc_ids = json.load(f)
        with open(path / "config.json") as f:
            cfg = json.load(f)

        if expected_config:
            mismatches = []
            for key in ("model_name", "normalize"):
                if key not in expected_config:
                    continue
                cached = cfg.get(key)
                requested = expected_config[key]
                if cached != requested:
                    mismatches.append((key, cached, requested))
            if mismatches:
                lines = [f"  {k}: cached={c!r}, requested={r!r}" for k, c, r in mismatches]
                raise DenseIndexConfigMismatch(
                    "Cached dense index config does not match requested config:\n"
                    + "\n".join(lines)
                    + f"\n\nIndex location: {path}"
                    + f"\nIndex config:   {cfg}"
                    + f"\nRequested:      {expected_config}"
                    + "\n\nFix one of:"
                    + "\n  - rerun with --rebuild-index to re-encode the corpus"
                    + "\n  - revert configs/baseline.yaml to match the cached index"
                )

        _force_faiss_single_threaded(faiss)
        retriever = cls(
            model_name=cfg["model_name"],
            device=device,
            encode_batch_size=encode_batch_size,
            normalize=cfg.get("normalize", True),
        )
        retriever.doc_ids = list(doc_ids)
        retriever._index = faiss.read_index(str(path / "index.faiss"))
        retriever._embedding_dim = cfg.get("embedding_dim")
        logger.info(
            "Loaded dense index from %s (%d docs, %s)",
            path,
            len(retriever.doc_ids),
            cfg.get("model_name"),
        )
        return retriever

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, k: int = 1000) -> tuple[np.ndarray, list[str]]:
        scores, doc_ids_lists = self.retrieve_batch([query], k=k)
        return scores[0], doc_ids_lists[0]

    def retrieve_batch(
        self,
        queries: Sequence[str],
        k: int = 1000,
        encode_batch_size: int | None = None,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Encode queries, FAISS-search top-k, map indices back to doc_ids."""
        if self._index is None:
            raise ValueError("No index built/loaded")
        if not self.doc_ids:
            raise ValueError("No doc_ids attached to the retriever")

        q_emb = self.encode(
            queries,
            batch_size=encode_batch_size,
            show_progress_bar=show_progress,
        )
        # IndexFlatIP.search returns (scores, indices) of shape (n, k).
        k_eff = min(k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k_eff)
        doc_ids_lists = [
            [self.doc_ids[int(i)] for i in row if int(i) >= 0]
            for row in indices
        ]
        return scores, doc_ids_lists

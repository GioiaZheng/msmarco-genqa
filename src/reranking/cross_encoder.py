"""Cross-encoder reranker over a first-stage retriever's top-k.

Wraps ``sentence_transformers.CrossEncoder`` with a small, retriever-agnostic
interface used by the W5 experiment script:

    reranker = CrossEncoderReranker(model_name=..., batch_size=64)
    reranked = reranker.rerank_batch(queries, candidates_per_query)
    # → for each query, a list of (doc_id, ce_score) sorted by score desc.

Default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — the standard
MS MARCO reranking baseline. It is small enough to run on CPU.

Cross-encoders are O(k) forward passes per query (one per (q, doc) pair),
so rerank-depth dominates wall-clock. We deliberately do *not* default
to k=1000; the runner caps to top-100 unless told otherwise.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Sentence-Transformers ``CrossEncoder`` wrapper.

    Parameters
    ----------
    model_name :
        HuggingFace model id. Defaults to the standard MS MARCO baseline.
    device :
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    batch_size :
        Mini-batch size for ``predict``. 32–64 is a reasonable CPU default;
        larger on GPU.
    max_length :
        Token-length cap for the (query, passage) pair. The MiniLM-L-6
        baseline was trained at 512.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None  # sentence_transformers.CrossEncoder

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        try:
            import torch

            # Mirror the dense retriever's defensive thread-pool init: on
            # macOS, lazy torch thread-pool init after ir_datasets/bm25s have
            # touched signal handlers can SIGSEGV the first forward pass.
            torch.set_num_threads(torch.get_num_threads())
        except ImportError:
            pass

        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        logger.info(
            "Loading cross-encoder %s on %s (max_length=%d, batch_size=%d)",
            self.model_name,
            self.device,
            self.max_length,
            self.batch_size,
        )
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Score a flat list of (query, passage) pairs.

        Returns a 1-D float32 array of length ``len(pairs)``.
        """
        self._ensure_model()
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        scores = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Rerank
    # ------------------------------------------------------------------ #

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates_per_query: Sequence[Sequence[tuple[str, str]]],
        show_progress_bar: bool = True,
    ) -> tuple[list[list[tuple[str, float]]], dict]:
        """Rerank a candidate list per query.

        ``candidates_per_query[i]`` is a list of ``(doc_id, doc_text)``
        for query ``queries[i]``.

        Returns
        -------
        reranked :
            For each query, a list of ``(doc_id, ce_score)`` sorted by
            score descending. Length matches the input candidate list.
        info :
            ``{"n_pairs": int, "score_seconds": float, "queries_per_sec":
            float, "pairs_per_sec": float}`` — runtime telemetry the
            caller will fold into ``metrics.json``.
        """
        if len(queries) != len(candidates_per_query):
            raise ValueError(
                f"queries ({len(queries)}) and candidates_per_query "
                f"({len(candidates_per_query)}) must match in length"
            )

        # Build the flat pair list and remember which query each pair came
        # from. Empty candidate lists are allowed and yield empty outputs.
        pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []  # (start, length) per query
        doc_ids_per_query: list[list[str]] = []
        for q, cands in zip(queries, candidates_per_query):
            start = len(pairs)
            doc_ids: list[str] = []
            for doc_id, text in cands:
                pairs.append((q, text))
                doc_ids.append(doc_id)
            offsets.append((start, len(cands)))
            doc_ids_per_query.append(doc_ids)

        t0 = time.time()
        scores_flat = self.score(pairs, show_progress_bar=show_progress_bar)
        elapsed = time.time() - t0

        reranked: list[list[tuple[str, float]]] = []
        for (start, length), doc_ids in zip(offsets, doc_ids_per_query):
            block = scores_flat[start : start + length]
            order = np.argsort(-block)  # desc by CE score
            reranked.append(
                [(doc_ids[int(i)], float(block[int(i)])) for i in order]
            )

        info = {
            "n_pairs": len(pairs),
            "score_seconds": elapsed,
            "queries_per_sec": (len(queries) / elapsed) if elapsed > 0 else None,
            "pairs_per_sec": (len(pairs) / elapsed) if elapsed > 0 else None,
        }
        return reranked, info

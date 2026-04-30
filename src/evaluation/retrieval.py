"""Retrieval metrics: MRR@k, Recall@k, nDCG@k.

All functions assume binary relevance (relevance > 0 → relevant), which
matches MS MARCO ``qrels.dev.small``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in rel)
    return hits / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], 1)
        if doc_id in rel
    )
    n_rel = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks_mrr: Sequence[int] = (10,),
    ks_recall: Sequence[int] = (100, 1000),
) -> dict[str, float]:
    """Average metrics across queries that have at least one positive qrel.

    Queries present in ``runs`` but absent from ``qrels`` (or with empty qrels)
    are skipped, mirroring the MS MARCO official evaluation behaviour.
    """
    qids = [qid for qid in runs if qrels.get(qid)]
    n = len(qids)
    if n == 0:
        return {"n_queries": 0}

    metrics: dict[str, float] = {}
    for k in ks_mrr:
        metrics[f"mrr@{k}"] = sum(
            reciprocal_rank(runs[q], qrels[q], k) for q in qids
        ) / n
    for k in ks_recall:
        metrics[f"recall@{k}"] = sum(
            recall_at_k(runs[q], qrels[q], k) for q in qids
        ) / n
    metrics["n_queries"] = n
    return metrics

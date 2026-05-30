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


def first_relevant_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int | None = None,
) -> int | None:
    """Return the 1-based rank of the first relevant document, if present."""
    rel = set(relevant)
    if not rel:
        return None
    cutoff = len(retrieved) if k is None else min(k, len(retrieved))
    for rank, doc_id in enumerate(retrieved[:cutoff], 1):
        if doc_id in rel:
            return rank
    return None


def retrieval_shift_bucket(
    before_rank: int | None,
    after_rank: int | None,
) -> str:
    """Classify how the first relevant document moved between two runs."""
    if before_rank is None and after_rank is None:
        return "unchanged_miss"
    if before_rank is None:
        return "new_hit"
    if after_rank is None:
        return "lost_hit"
    if after_rank < before_rank:
        return "promoted"
    if after_rank > before_rank:
        return "demoted"
    return "unchanged_hit"


def query_retrieval_delta(
    qid: str,
    before: Sequence[str],
    after: Sequence[str],
    relevant: Iterable[str],
    *,
    k_rank: int = 10,
    k_recall: int = 100,
) -> dict[str, float | int | str | None]:
    """Compare one query across two ranked retrieval outputs.

    The result is designed for error analysis rather than leaderboard
    reporting: it keeps the headline deltas and a bucket explaining whether
    reranking promoted, demoted, recovered, or lost a qrels-relevant document.
    """
    rel = set(relevant)
    before_rank = first_relevant_rank(before, rel, k_rank)
    after_rank = first_relevant_rank(after, rel, k_rank)
    before_rr = reciprocal_rank(before, rel, k_rank)
    after_rr = reciprocal_rank(after, rel, k_rank)
    before_recall = recall_at_k(before, rel, k_recall)
    after_recall = recall_at_k(after, rel, k_recall)
    rank_movement = (
        before_rank - after_rank
        if before_rank is not None and after_rank is not None
        else None
    )
    return {
        "qid": qid,
        "bucket": retrieval_shift_bucket(before_rank, after_rank),
        "n_relevant": len(rel),
        "before_first_relevant_rank": before_rank,
        "after_first_relevant_rank": after_rank,
        "rank_movement": rank_movement,
        f"before_rr@{k_rank}": before_rr,
        f"after_rr@{k_rank}": after_rr,
        f"rr_delta@{k_rank}": after_rr - before_rr,
        f"before_recall@{k_recall}": before_recall,
        f"after_recall@{k_recall}": after_recall,
        f"recall_delta@{k_recall}": after_recall - before_recall,
    }


def compare_retrieval_runs_per_query(
    before_runs: dict[str, Sequence[str]],
    after_runs: dict[str, Sequence[str]],
    qrels: dict[str, set[str]],
    *,
    k_rank: int = 10,
    k_recall: int = 100,
) -> list[dict[str, float | int | str | None]]:
    """Return query-level deltas for qids shared by two runs and qrels."""
    qids = sorted(
        qid
        for qid in before_runs.keys() & after_runs.keys() & qrels.keys()
        if qrels.get(qid)
    )
    return [
        query_retrieval_delta(
            qid,
            before_runs[qid],
            after_runs[qid],
            qrels[qid],
            k_rank=k_rank,
            k_recall=k_recall,
        )
        for qid in qids
    ]


def evaluate_retrieval(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks_mrr: Sequence[int] = (10,),
    ks_recall: Sequence[int] = (100, 1000),
    ks_ndcg: Sequence[int] = (10,),
) -> dict[str, float]:
    """Average metrics across queries that have at least one positive qrel.

    Queries present in ``runs`` but absent from ``qrels`` (or with empty qrels)
    are skipped, mirroring the MS MARCO official evaluation behaviour.

    Returned dict contains one key per ``mrr@k`` / ``ndcg@k`` / ``recall@k``,
    plus ``n_queries`` (the number of queries that contributed to the
    averages). The runner pulls ``n_queries`` out and promotes it to the
    top-level ``n_examples`` field of the saved ``metrics.json``.
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
    for k in ks_ndcg:
        metrics[f"ndcg@{k}"] = sum(
            ndcg_at_k(runs[q], qrels[q], k) for q in qids
        ) / n
    for k in ks_recall:
        metrics[f"recall@{k}"] = sum(
            recall_at_k(runs[q], qrels[q], k) for q in qids
        ) / n
    metrics["n_queries"] = n
    return metrics

"""Hybrid retrieval fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


RunItems = Mapping[str, Sequence[tuple[str, float]]]
RunsBySource = Mapping[str, RunItems]


@dataclass(frozen=True)
class SourceContribution:
    """Contribution of one source run to one fused document."""

    rank: int
    score: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class FusedHit:
    """One fused retrieval candidate for a query."""

    doc_id: str
    score: float
    sources: dict[str, SourceContribution]

    @property
    def best_rank(self) -> int:
        return min(source.rank for source in self.sources.values())


def _validate_fusion_inputs(
    runs_by_source: RunsBySource,
    rank_constant: float,
    top_k: int | None,
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if not runs_by_source:
        raise ValueError("at least one source run is required")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided")

    source_names = list(runs_by_source)
    if len(set(source_names)) != len(source_names):
        raise ValueError("source names must be unique")

    resolved = {name: 1.0 for name in source_names}
    if weights:
        unknown = sorted(set(weights) - set(source_names))
        if unknown:
            raise ValueError(f"weights provided for unknown source(s): {unknown}")
        for name, weight in weights.items():
            value = float(weight)
            if value < 0:
                raise ValueError(f"weight for source {name!r} must be non-negative")
            resolved[name] = value
    return resolved


def reciprocal_rank_fusion(
    runs_by_source: RunsBySource,
    *,
    rank_constant: float = 60.0,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> dict[str, list[FusedHit]]:
    """Fuse ranked retrieval runs with weighted Reciprocal Rank Fusion.

    Scores follow:

        sum_s weight_s / (rank_constant + rank_s(doc))

    where ranks are one-based within each source run. Ties are resolved
    deterministically by fused score, best observed rank, and document id.
    """
    resolved_weights = _validate_fusion_inputs(
        runs_by_source=runs_by_source,
        rank_constant=rank_constant,
        top_k=top_k,
        weights=weights,
    )
    all_qids = sorted({qid for runs in runs_by_source.values() for qid in runs})
    fused_by_qid: dict[str, list[FusedHit]] = {}

    for qid in all_qids:
        contributions: dict[str, dict[str, SourceContribution]] = {}
        for source_name, runs in runs_by_source.items():
            weight = resolved_weights[source_name]
            seen_in_source: set[str] = set()
            for rank, (doc_id, source_score) in enumerate(runs.get(qid, ()), start=1):
                if doc_id in seen_in_source:
                    raise ValueError(
                        f"duplicate document id {doc_id!r} in source "
                        f"{source_name!r} for query id {qid!r}"
                    )
                seen_in_source.add(doc_id)
                contribution = weight / (rank_constant + rank)
                contributions.setdefault(doc_id, {})[source_name] = SourceContribution(
                    rank=rank,
                    score=float(source_score),
                    weight=weight,
                    contribution=contribution,
                )

        rows = [
            FusedHit(
                doc_id=doc_id,
                score=sum(source.contribution for source in sources.values()),
                sources=sources,
            )
            for doc_id, sources in contributions.items()
        ]
        rows.sort(key=lambda row: (-row.score, row.best_rank, row.doc_id))
        if top_k is not None:
            rows = rows[:top_k]
        fused_by_qid[qid] = rows

    return fused_by_qid


def fused_doc_ids_and_scores(
    fused: Mapping[str, Sequence[FusedHit]],
) -> tuple[list[str], list[list[str]], list[list[float]]]:
    """Return qids, doc ids, and scores in ``write_run_tsv`` shape."""
    qids = sorted(fused)
    doc_ids = [[hit.doc_id for hit in fused[qid]] for qid in qids]
    scores = [[hit.score for hit in fused[qid]] for qid in qids]
    return qids, doc_ids, scores

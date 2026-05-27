"""Build a sub-corpus for sampled retrieval experiments.

Random sampling alone is useless for retrieval evaluation: with 50k random
docs from 8.8M, almost no dev query's relevant doc will be in the pool, so
both BM25 and dense retrievers will score near zero and any comparison is
noise.

The standard fix is **qrels-anchored sampling**: keep every relevant doc
that the eval set cares about, then top up to ``target_size`` with random
distractors. The relevant doc is therefore *always* present in the pool,
so MRR is meaningful and the comparison BM25 ↔ dense on the same pool is
honest. Absolute numbers are still NOT comparable to a full-corpus run
(which is the whole point of the caveat banner on the W4 report).
"""

from __future__ import annotations

import logging
import random
from typing import Iterable

logger = logging.getLogger(__name__)


def qrels_anchored_sample(
    pool_doc_ids: list[str],
    qrels: dict[str, Iterable[str]],
    target_size: int,
    seed: int = 42,
) -> list[str]:
    """Sample ``target_size`` doc_ids that always include all qrels-relevant docs.

    Parameters
    ----------
    pool_doc_ids :
        The universe of doc_ids to sample from (e.g. all 8.8M MS MARCO
        Passage doc_ids).
    qrels :
        Mapping ``query_id -> iterable[relevant_doc_id]`` for the eval set.
        Every doc_id appearing in any inner iterable is unconditionally
        included.
    target_size :
        Desired final corpus size. Must be ≥ number of unique relevant
        doc_ids.
    seed :
        RNG seed for the distractor sample. Same seed → same sample.

    Returns
    -------
    list[str]
        Deterministic, sorted list of doc_ids of length
        ``min(target_size, len(pool_doc_ids))``.

    Raises
    ------
    ValueError
        If ``target_size`` is too small to fit all relevant docs, or if
        any relevant doc is missing from ``pool_doc_ids``.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    pool_set = set(pool_doc_ids)
    rel_ids: set[str] = set()
    for inner in qrels.values():
        rel_ids.update(inner)

    missing = rel_ids - pool_set
    if missing:
        raise ValueError(
            f"{len(missing)} relevant doc_ids are missing from pool_doc_ids "
            f"(first few: {sorted(missing)[:5]}). The pool must be a "
            "superset of the qrels — typically the full corpus's doc_ids."
        )
    if len(rel_ids) > target_size:
        raise ValueError(
            f"target_size ({target_size}) is smaller than the number of unique "
            f"relevant doc_ids ({len(rel_ids)})."
        )

    if target_size >= len(pool_doc_ids):
        logger.info(
            "target_size %d >= pool size %d; returning the full pool.",
            target_size,
            len(pool_doc_ids),
        )
        return sorted(pool_set)

    rng = random.Random(seed)
    n_distractors = target_size - len(rel_ids)
    # ``sorted`` is critical: set-to-list ordering depends on PYTHONHASHSEED,
    # which is randomised per process. Without ``sorted`` the same (seed,
    # qrels, pool) inputs produce different distractor sets across runs and
    # the saved index becomes non-reusable.
    distractor_pool = sorted(pool_set - rel_ids)
    distractors = rng.sample(distractor_pool, n_distractors)

    sampled = sorted(rel_ids | set(distractors))
    logger.info(
        "Sampled %d doc_ids: %d relevant + %d distractors (seed=%d).",
        len(sampled),
        len(rel_ids),
        len(distractors),
        seed,
    )
    return sampled

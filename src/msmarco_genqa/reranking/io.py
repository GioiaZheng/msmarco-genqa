"""I/O helpers for reranking: read/truncate/write TREC-format ``run.tsv``.

Both W2 (BM25) and W4 (dense) write 6-column TREC-format run files:

    qid \t Q0 \t doc_id \t rank \t score \t system

The reranker only needs ``(qid, doc_id, score)`` and the per-query rank
ordering, so we ignore the other columns on read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_run_tsv(path: Path | str) -> dict[str, list[tuple[str, float]]]:
    """Parse a TREC run file into ``{qid: [(doc_id, score), ...]}``.

    Lines are returned in the order they appear in the file (which is
    already the rank order produced by the upstream retriever).
    """
    runs: dict[str, list[tuple[str, float]]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            qid, _q0, doc_id, _rank, score, _sys = parts[:6]
            runs.setdefault(qid, []).append((doc_id, float(score)))
    return runs


def truncate_top_k(
    runs: dict[str, list[tuple[str, float]]],
    k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Keep only the top-``k`` entries per query (preserving order)."""
    return {q: docs[:k] for q, docs in runs.items()}


def collect_unique_doc_ids(
    runs: dict[str, list[tuple[str, float]]],
) -> list[str]:
    """Return the de-duplicated list of doc_ids referenced in ``runs``.

    Order is the first-seen order across queries — useful for batched
    text resolution.
    """
    seen: set[str] = set()
    out: list[str] = []
    for docs in runs.values():
        for doc_id, _ in docs:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


def write_run_tsv(
    path: Path | str,
    qids: Iterable[str],
    doc_ids_lists: Iterable[Iterable[str]],
    scores_lists: Iterable[Iterable[float]],
    system_name: str,
) -> None:
    """Write a TREC-format run file from per-query (doc_ids, scores) lists."""
    with open(path, "w") as f:
        for qid, doc_ids, scores in zip(qids, doc_ids_lists, scores_lists):
            for rank, (d, s) in enumerate(zip(doc_ids, scores), 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{float(s):.6f}\t{system_name}\n")


def append_run_tsv(
    path: Path | str,
    qids: Iterable[str],
    doc_ids_lists: Iterable[Iterable[str]],
    scores_lists: Iterable[Iterable[float]],
    system_name: str,
) -> None:
    """Append a TREC-format chunk to an existing run file (or create it).

    Same wire format as ``write_run_tsv`` but opens the file in append mode
    and flushes after the chunk so a SIGKILL between chunks loses at most
    one in-flight chunk, not the whole run.
    """
    with open(path, "a") as f:
        for qid, doc_ids, scores in zip(qids, doc_ids_lists, scores_lists):
            for rank, (d, s) in enumerate(zip(doc_ids, scores), 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{float(s):.6f}\t{system_name}\n")
        f.flush()


def read_done_qids(path: Path | str, top_k: int) -> set[str]:
    """Return qids that already have a *complete* top-``top_k`` block in the run.

    A qid is considered done only if its highest observed rank equals
    ``top_k`` AND all ranks ``1..top_k`` are present, so a partially-
    written chunk (e.g. one being flushed when SIGKILL hit) is
    automatically retried instead of being treated as complete.
    """
    p = Path(path)
    if not p.exists():
        return set()
    counts: dict[str, int] = {}
    max_rank: dict[str, int] = {}
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, _q0, _doc_id, rank_str = parts[:4]
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            counts[qid] = counts.get(qid, 0) + 1
            max_rank[qid] = max(max_rank.get(qid, 0), rank)
    return {
        qid
        for qid in counts
        if counts[qid] == top_k and max_rank[qid] == top_k
    }


def prune_partial_qids(path: Path | str, keep_qids: set[str]) -> int:
    """Rewrite ``path`` in place keeping only lines whose qid is in ``keep_qids``.

    Returns the number of dropped lines. Used on resume to evict the
    half-flushed lines for a qid that was in flight when the process died:
    without this, re-scoring that qid on resume duplicates its block.
    """
    p = Path(path)
    if not p.exists():
        return 0
    kept_lines: list[str] = []
    dropped = 0
    with open(p) as f:
        for line in f:
            qid_field = line.split("\t", 1)[0]
            if qid_field in keep_qids:
                kept_lines.append(line)
            else:
                dropped += 1
    with open(p, "w") as f:
        f.writelines(kept_lines)
    return dropped

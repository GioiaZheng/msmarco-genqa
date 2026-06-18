"""I/O helpers for reranking: read/truncate/write TREC-format ``run.tsv``.

Both BM25 and dense retrieval write 6-column TREC-format run files:

    qid \t Q0 \t doc_id \t rank \t score \t system

The reranker only needs ``(qid, doc_id, score)`` and the per-query rank
ordering, so we ignore the other columns on read.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


class RunTsvFormatError(ValueError):
    """Raised when a TREC-format run file is malformed."""

    def __init__(self, path: Path | str, line_number: int | None, message: str) -> None:
        self.path = Path(path)
        self.line_number = line_number
        self.reason = message
        location = str(self.path) if line_number is None else f"{self.path}:{line_number}"
        super().__init__(f"{location}: {message}")


def read_run_tsv(path: Path | str) -> dict[str, list[tuple[str, float]]]:
    """Parse a TREC run file into ``{qid: [(doc_id, score), ...]}``.

    Entries are returned in ascending rank order within each query. Malformed
    records fail fast with a line-numbered ``RunTsvFormatError`` instead of
    being skipped silently.
    """
    p = Path(path)
    rows_by_qid: dict[str, list[tuple[int, str, float]]] = {}
    seen_qid_rank: set[tuple[str, int]] = set()
    seen_qid_doc: set[tuple[str, str]] = set()
    try:
        with p.open(encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n").rstrip("\r")
                if not line:
                    raise RunTsvFormatError(p, line_number, "empty line")
                parts = line.split("\t")
                if len(parts) != 6:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"expected 6 tab-separated fields, got {len(parts)}",
                    )
                qid, _q0, doc_id, rank_text, score_text, system = parts
                if not qid:
                    raise RunTsvFormatError(p, line_number, "empty query id")
                if not doc_id:
                    raise RunTsvFormatError(p, line_number, "empty document id")
                if not system:
                    raise RunTsvFormatError(p, line_number, "empty system name")
                if "\ufffd" in qid or "\ufffd" in doc_id or "\ufffd" in system:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        "replacement character found in identifier field",
                    )
                try:
                    rank = int(rank_text)
                except ValueError as exc:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"rank is not an integer: {rank_text!r}",
                    ) from exc
                if rank < 1:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"rank must be positive, got {rank}",
                    )
                qid_rank = (qid, rank)
                if qid_rank in seen_qid_rank:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"duplicate rank {rank} for query id {qid!r}",
                    )
                seen_qid_rank.add(qid_rank)
                qid_doc = (qid, doc_id)
                if qid_doc in seen_qid_doc:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"duplicate document id {doc_id!r} for query id {qid!r}",
                    )
                seen_qid_doc.add(qid_doc)
                try:
                    score = float(score_text)
                except ValueError as exc:
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"score is not numeric: {score_text!r}",
                    ) from exc
                if not math.isfinite(score):
                    raise RunTsvFormatError(
                        p,
                        line_number,
                        f"score must be finite, got {score_text!r}",
                    )
                rows_by_qid.setdefault(qid, []).append((rank, doc_id, score))
    except UnicodeDecodeError as exc:
        raise RunTsvFormatError(p, None, f"file is not valid UTF-8: {exc}") from exc
    return {
        qid: [(doc_id, score) for _rank, doc_id, score in sorted(rows)]
        for qid, rows in rows_by_qid.items()
    }


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

"""TREC-DL 2019 & 2020 passage track loaders.

External-validity coverage beyond the MS MARCO ``dev/small`` sample. Both
tracks reuse the MS MARCO passage collection but ship *deep*, graded
relevance judgments (label scale 0-3) over a small, NIST-pooled query set:
43 judged queries in 2019 and 54 in 2020. This contrasts with
``dev/small``, whose qrels are *sparse* and binary (~1 relevant passage per
query).

The graded judgments are loaded in full into :attr:`TrecDlPassages.graded_qrels`.
For the existing binary metric functions (``mrr@k`` / ``ndcg@k`` /
``recall@k`` in :mod:`msmarco_genqa.evaluation.retrieval`), the labels are
binarized at :data:`DEFAULT_REL_THRESHOLD` (relevance >= 2 counts as
relevant) and exposed as :attr:`TrecDlPassages.qrels` — the same
``dict[str, set[str]]`` schema the metric code already consumes, so those
functions run unchanged.

``ir_datasets`` handles downloading and caching. The collection itself is the
shared MS MARCO passage corpus; only the small query + qrels files are track
specific, so the incremental download is tiny once the corpus is cached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# ir_datasets identifiers for the judged-only query subsets. The ``/judged``
# variant restricts to the NIST-assessed queries (the only ones with qrels),
# which is exactly the evaluable set.
TREC_DL_DATASETS: dict[int, str] = {
    2019: "msmarco-passage/trec-dl-2019/judged",
    2020: "msmarco-passage/trec-dl-2020/judged",
}

# Standard binarization threshold for the MS MARCO TREC-DL passage tracks:
# labels {2, 3} are treated as relevant for MRR/Recall, {0, 1} as not. This
# matches the official TREC-DL passage evaluation convention.
DEFAULT_REL_THRESHOLD = 2


@dataclass
class TrecDlPassages:
    """In-memory TREC-DL passage-track query + qrels bundle.

    Attributes
    ----------
    year :
        Track year (2019 or 2020).
    dataset_name :
        The ``ir_datasets`` identifier the bundle was loaded from.
    rel_threshold :
        Minimum graded label treated as relevant when binarizing
        :attr:`graded_qrels` into :attr:`qrels`.
    queries :
        Mapping from string query id to query text (judged queries only).
    qrels :
        Mapping from query id to the set of relevant doc ids, binarized at
        :attr:`rel_threshold`. Same schema as the MS MARCO loader, so the
        existing binary metric functions consume it unchanged. A query whose
        judgments are all below threshold maps to an empty set (kept, so
        coverage diagnostics can see it).
    graded_qrels :
        Mapping from query id to ``{doc_id: graded_label}`` over every judged
        passage, preserving the full 0-3 scale (including judged-non-relevant
        0/1 labels). Retained for future graded-nDCG work; not consumed by the
        current binary metrics.
    """

    year: int
    dataset_name: str
    rel_threshold: int
    queries: dict[str, str]
    qrels: dict[str, set[str]]
    graded_qrels: dict[str, dict[str, int]]


def binarize_qrels(
    graded_qrels: dict[str, dict[str, int]],
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
) -> dict[str, set[str]]:
    """Collapse graded judgments to positive sets at ``rel_threshold``.

    Every judged query is kept, even if no passage reaches the threshold, so
    that downstream coverage accounting can distinguish "no relevant passage
    above threshold" from "query never judged".
    """
    return {
        qid: {doc_id for doc_id, label in labels.items() if label >= rel_threshold}
        for qid, labels in graded_qrels.items()
    }


def parse_queries_tsv(lines: Iterable[str]) -> dict[str, str]:
    """Parse ``qid<TAB>text`` query lines into ``{qid: text}``.

    Blank lines and ``#`` comments are ignored. The text may itself contain
    tabs; only the first tab is treated as the qid separator.
    """
    queries: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(
                f"queries line {line_number}: expected 'qid<TAB>text', got {line!r}"
            )
        qid, text = line.split("\t", 1)
        queries[qid.strip()] = text.strip()
    return queries


def parse_graded_qrels(lines: Iterable[str]) -> dict[str, dict[str, int]]:
    """Parse TREC-format qrels (``qid iter docid rel``) into graded labels.

    A compact 3-column ``qid docid rel`` form is also accepted. Every judged
    pair is retained, including label 0, so the full judgment depth survives.
    """
    graded: dict[str, dict[str, int]] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 3:
            qid, doc_id, rel_text = parts
        elif len(parts) >= 4:
            qid, _iter, doc_id, rel_text = parts[:4]
        else:
            raise ValueError(
                f"qrels line {line_number}: expected 3 or 4+ columns, got {len(parts)}"
            )
        try:
            rel = int(float(rel_text))
        except ValueError as exc:
            raise ValueError(
                f"qrels line {line_number}: relevance is not numeric: {rel_text!r}"
            ) from exc
        graded.setdefault(qid, {})[doc_id] = rel
    return graded


def build_trec_dl_bundle(
    *,
    year: int,
    dataset_name: str,
    queries: dict[str, str],
    graded_qrels: dict[str, dict[str, int]],
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
) -> TrecDlPassages:
    """Assemble a :class:`TrecDlPassages` from already-parsed dicts.

    Shared by the ``ir_datasets`` path and the file path so both produce an
    identical bundle. Restricts ``graded_qrels`` to judged queries that also
    appear in ``queries`` (the ``/judged`` subset guarantees this for the
    ir_datasets path; the file path may be a subset fixture).
    """
    graded = {qid: labels for qid, labels in graded_qrels.items() if qid in queries}
    return TrecDlPassages(
        year=year,
        dataset_name=dataset_name,
        rel_threshold=rel_threshold,
        queries=queries,
        qrels=binarize_qrels(graded, rel_threshold),
        graded_qrels=graded,
    )


def load_trec_dl(
    year: int,
    cache_dir: Path | str | None = None,
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
) -> TrecDlPassages:
    """Load a TREC-DL passage track (2019 or 2020) via ``ir_datasets``.

    Parameters
    ----------
    year :
        Track year; must be a key of :data:`TREC_DL_DATASETS`.
    cache_dir :
        Directory used as ``IR_DATASETS_HOME``. If None, ir_datasets falls
        back to its default (``~/.ir_datasets``).
    rel_threshold :
        Minimum graded label treated as relevant for the binarized
        :attr:`TrecDlPassages.qrels`.

    Returns
    -------
    TrecDlPassages
    """
    if year not in TREC_DL_DATASETS:
        raise ValueError(
            f"unsupported TREC-DL year {year!r}; expected one of "
            f"{sorted(TREC_DL_DATASETS)}"
        )
    if cache_dir is not None:
        os.environ["IR_DATASETS_HOME"] = str(Path(cache_dir).expanduser().resolve())

    import ir_datasets

    dataset_name = TREC_DL_DATASETS[year]
    logger.info("Loading TREC-DL %d (%s) via ir_datasets...", year, dataset_name)
    dataset = ir_datasets.load(dataset_name)

    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    graded_qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        graded_qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    bundle = build_trec_dl_bundle(
        year=year,
        dataset_name=dataset_name,
        queries=queries,
        graded_qrels=graded_qrels,
        rel_threshold=rel_threshold,
    )
    logger.info(
        "TREC-DL %d: %d judged queries, %d with >=1 relevant passage at rel>=%d.",
        year,
        len(bundle.queries),
        sum(1 for docs in bundle.qrels.values() if docs),
        rel_threshold,
    )
    return bundle


def load_trec_dl_from_files(
    queries_path: Path | str,
    qrels_path: Path | str,
    *,
    year: int,
    dataset_name: str | None = None,
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
) -> TrecDlPassages:
    """Load a TREC-DL bundle from local query/qrels files (no network).

    Used by the deterministic fixture tests and as an offline entry point when
    the track files are already on disk. ``queries_path`` is ``qid<TAB>text``;
    ``qrels_path`` is TREC-format (``qid iter docid rel``).
    """
    queries = parse_queries_tsv(Path(queries_path).read_text(encoding="utf-8").splitlines())
    graded_qrels = parse_graded_qrels(Path(qrels_path).read_text(encoding="utf-8").splitlines())
    return build_trec_dl_bundle(
        year=year,
        dataset_name=dataset_name or f"trec-dl-{year} (local files)",
        queries=queries,
        graded_qrels=graded_qrels,
        rel_threshold=rel_threshold,
    )

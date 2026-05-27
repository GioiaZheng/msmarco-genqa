"""MS MARCO Passage Ranking data loader.

Loads, via ``ir_datasets``:

- The full passage collection (~8.8M passages).
- The ``dev/small`` query split (6,980 queries) and its qrels.

``ir_datasets`` handles downloading and caching automatically. The first call
downloads ~3 GB; subsequent calls reuse the local cache.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MSMarcoPassage:
    """In-memory MS MARCO Passage Ranking dev/small bundle.

    Attributes
    ----------
    corpus_doc_ids :
        Parallel list of doc ids for every passage in the collection. Empty
        if ``load_corpus`` was False at load time.
    corpus_texts :
        Parallel list of passage texts. Same length as ``corpus_doc_ids``.
    queries :
        Mapping from string query id to query text (dev/small).
    qrels :
        Mapping from query id to the set of relevant doc ids
        (only relevance > 0 is kept; MS MARCO dev/small qrels are binary).
    docs_store :
        Optional ``ir_datasets`` docs_store object for random access by
        doc_id. Useful when ``load_corpus=False`` and the caller only needs
        a few specific passages.
    """

    corpus_doc_ids: list[str]
    corpus_texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, set[str]]
    docs_store: Any | None = None


def load_msmarco_passage(
    cache_dir: Path | str | None = None,
    load_corpus: bool = True,
    limit: int | None = None,
) -> MSMarcoPassage:
    """Load the official MS MARCO Passage Ranking dev/small bundle.

    Parameters
    ----------
    cache_dir :
        Directory used as ``IR_DATASETS_HOME``. If None, ir_datasets falls
        back to its default (``~/.ir_datasets``).
    load_corpus :
        When True (default), load the full passage collection into memory.
        Required for index construction. When False, the corpus lists are
        empty and ``docs_store`` is populated for random-access lookup.
    limit :
        If set, load only the first ``limit`` passages from the collection.
        Intended for development/smoke tests; numbers produced this way are
        not comparable to the official baseline.

    Returns
    -------
    MSMarcoPassage
    """
    if cache_dir is not None:
        os.environ["IR_DATASETS_HOME"] = str(Path(cache_dir).expanduser().resolve())

    import ir_datasets

    logger.info("Loading MS MARCO Passage (dev/small) via ir_datasets...")
    collection = ir_datasets.load("msmarco-passage")
    dev_small = ir_datasets.load("msmarco-passage/dev/small")

    queries = {q.query_id: q.text for q in dev_small.queries_iter()}
    logger.info("Loaded %d dev/small queries.", len(queries))

    qrels: dict[str, set[str]] = {}
    for qrel in dev_small.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    logger.info("Loaded qrels for %d queries.", len(qrels))

    doc_ids: list[str] = []
    texts: list[str] = []
    docs_store = None

    if load_corpus:
        logger.info("Loading passage collection into memory (this is slow on first run)...")
        for i, doc in enumerate(collection.docs_iter()):
            if limit is not None and i >= limit:
                break
            doc_ids.append(doc.doc_id)
            texts.append(doc.text)
            if (i + 1) % 1_000_000 == 0:
                logger.info("  %d passages loaded...", i + 1)
        logger.info("Loaded %d passages into memory.", len(doc_ids))
    else:
        docs_store = collection.docs_store()
        logger.info("Skipping eager corpus load; docs_store ready for random access.")

    return MSMarcoPassage(
        corpus_doc_ids=doc_ids,
        corpus_texts=texts,
        queries=queries,
        qrels=qrels,
        docs_store=docs_store,
    )


def get_docs_store(
    dataset_name: str = "msmarco-passage",
    cache_dir: Path | str | None = None,
):
    """Return an ir_datasets docs_store for random-access doc lookup.

    The first call builds a small on-disk index; subsequent calls reuse it.
    """
    if cache_dir is not None:
        os.environ["IR_DATASETS_HOME"] = str(Path(cache_dir).expanduser().resolve())
    import ir_datasets

    return ir_datasets.load(dataset_name).docs_store()

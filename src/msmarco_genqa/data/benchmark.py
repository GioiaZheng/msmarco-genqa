"""Dataset and corpus selection shared by the full-corpus runners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msmarco_genqa.data.msmarco import MSMarcoPassage, load_msmarco_passage
from msmarco_genqa.data.trec_dl import (
    DEFAULT_REL_THRESHOLD,
    TREC_DL_DATASETS,
    binarize_qrels,
    load_trec_dl,
)


MSMARCO_DEV_SMALL = "msmarco-passage/dev/small"
BEIR_NFCORPUS_TEST = "beir/nfcorpus/test"
BEIR_SCIFACT_TEST = "beir/scifact/test"
BEIR_DATASETS = (BEIR_NFCORPUS_TEST, BEIR_SCIFACT_TEST)
SUPPORTED_DATASETS = (MSMARCO_DEV_SMALL, *TREC_DL_DATASETS.values(), *BEIR_DATASETS)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Stable identity and output conventions for one retrieval benchmark."""

    dataset_id: str
    output_slug: str
    corpus_id: str = "msmarco-passage"
    benchmark_family: str = "msmarco"
    topic_scope: str = "dev/small"
    track_year: int | None = None
    rel_threshold: int | None = None
    domain: str | None = None

    @property
    def is_trec_dl(self) -> bool:
        return self.track_year is not None

    @property
    def is_beir(self) -> bool:
        return self.benchmark_family == "beir"

    @property
    def has_graded_qrels(self) -> bool:
        return self.is_trec_dl or self.is_beir

    @property
    def uses_msmarco_corpus(self) -> bool:
        return self.corpus_id == "msmarco-passage"


@dataclass
class BenchmarkCorpus:
    """Corpus documents normalized across supported benchmark families."""

    spec: BenchmarkSpec
    corpus_doc_ids: list[str]
    corpus_texts: list[str]
    docs_store: Any | None = None


@dataclass
class BenchmarkQueries:
    """Query text and judgments normalized across supported benchmarks."""

    spec: BenchmarkSpec
    queries: dict[str, str]
    qrels: dict[str, set[str]]
    graded_qrels: dict[str, dict[str, int]]

    @property
    def judged_topic_count(self) -> int:
        return len(self.graded_qrels)

    @property
    def positive_topic_count(self) -> int:
        return sum(bool(doc_ids) for doc_ids in self.qrels.values())

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_id": self.spec.dataset_id,
            "corpus_id": self.spec.corpus_id,
            "benchmark_family": self.spec.benchmark_family,
            "domain": self.spec.domain,
            "track_year": self.spec.track_year,
            "topic_scope": self.spec.topic_scope,
            "query_count": len(self.queries),
            "judged_topic_count": self.judged_topic_count,
            "positive_topic_count": self.positive_topic_count,
            "qrels_type": "graded" if self.spec.has_graded_qrels else "binary",
            "relevance_threshold": self.spec.rel_threshold,
        }


def get_benchmark_spec(dataset_id: str) -> BenchmarkSpec:
    if dataset_id == MSMARCO_DEV_SMALL:
        return BenchmarkSpec(dataset_id=dataset_id, output_slug="msmarco_dev_small")

    for year, candidate in TREC_DL_DATASETS.items():
        if dataset_id == candidate:
            return BenchmarkSpec(
                dataset_id=dataset_id,
                output_slug=f"trec_dl_{year}",
                benchmark_family="trec-dl",
                topic_scope="judged",
                track_year=year,
                rel_threshold=DEFAULT_REL_THRESHOLD,
            )

    if dataset_id == BEIR_NFCORPUS_TEST:
        return BenchmarkSpec(
            dataset_id=dataset_id,
            output_slug="beir_nfcorpus_test",
            corpus_id=dataset_id,
            benchmark_family="beir",
            topic_scope="test",
            rel_threshold=1,
            domain="medical",
        )

    if dataset_id == BEIR_SCIFACT_TEST:
        return BenchmarkSpec(
            dataset_id=dataset_id,
            output_slug="beir_scifact_test",
            corpus_id=dataset_id,
            benchmark_family="beir",
            topic_scope="test",
            rel_threshold=1,
            domain="scientific",
        )

    raise ValueError(
        f"unsupported dataset {dataset_id!r}; expected one of {SUPPORTED_DATASETS}"
    )


def _set_irds_home(cache_dir: Path | str | None) -> None:
    if cache_dir is not None:
        os.environ["IR_DATASETS_HOME"] = str(Path(cache_dir).expanduser().resolve())


def _record_text(record: object, *, fields: tuple[str, ...]) -> str:
    values: list[str] = []
    for field in fields:
        value = getattr(record, field, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return "\n".join(values)


def query_text(record: object) -> str:
    """Return the canonical query text for common ir_datasets query records."""
    return _record_text(record, fields=("text", "title", "all", "desc"))


def document_text(record: object) -> str:
    """Return searchable text for common ir_datasets document records."""
    return _record_text(record, fields=("title", "text", "abstract", "body"))


def lookup_document_text(docs_store: Any, doc_id: str) -> str:
    """Look up and normalize one document from an ir_datasets docs_store."""
    return document_text(docs_store.get(doc_id))


def load_benchmark_corpus(
    spec: BenchmarkSpec,
    *,
    cache_dir: Path | str | None = None,
    load_corpus: bool = True,
    limit: int | None = None,
) -> BenchmarkCorpus:
    """Load the corpus attached to a benchmark.

    TREC-DL shares the MS MARCO passage corpus. BEIR datasets use their own
    inherited corpora, so using the MS MARCO index for them would be invalid.
    """
    if spec.uses_msmarco_corpus:
        data = load_msmarco_passage(
            cache_dir=cache_dir,
            load_corpus=load_corpus,
            limit=limit,
        )
        return BenchmarkCorpus(
            spec=spec,
            corpus_doc_ids=data.corpus_doc_ids,
            corpus_texts=data.corpus_texts,
            docs_store=data.docs_store,
        )

    _set_irds_home(cache_dir)
    import ir_datasets

    dataset = ir_datasets.load(spec.corpus_id)
    doc_ids: list[str] = []
    texts: list[str] = []
    docs_store = None

    if load_corpus:
        for index, doc in enumerate(dataset.docs_iter()):
            if limit is not None and index >= limit:
                break
            doc_ids.append(doc.doc_id)
            texts.append(document_text(doc))
    else:
        docs_store = dataset.docs_store()

    return BenchmarkCorpus(
        spec=spec,
        corpus_doc_ids=doc_ids,
        corpus_texts=texts,
        docs_store=docs_store,
    )


def load_benchmark_queries(
    dataset_id: str,
    *,
    cache_dir: Path | str | None = None,
    msmarco_data: MSMarcoPassage | None = None,
) -> BenchmarkQueries:
    """Load query text and qrels without loading the passage corpus eagerly."""
    spec = get_benchmark_spec(dataset_id)
    if spec.dataset_id == MSMARCO_DEV_SMALL:
        data = msmarco_data or load_msmarco_passage(
            cache_dir=cache_dir,
            load_corpus=False,
        )
        graded_qrels = {
            qid: {doc_id: 1 for doc_id in doc_ids}
            for qid, doc_ids in data.qrels.items()
        }
        return BenchmarkQueries(spec, data.queries, data.qrels, graded_qrels)

    if spec.is_trec_dl:
        data = load_trec_dl(
            spec.track_year,
            cache_dir=cache_dir,
            rel_threshold=spec.rel_threshold or DEFAULT_REL_THRESHOLD,
        )
        return BenchmarkQueries(spec, data.queries, data.qrels, data.graded_qrels)

    _set_irds_home(cache_dir)
    import ir_datasets

    dataset = ir_datasets.load(spec.dataset_id)
    queries = {q.query_id: query_text(q) for q in dataset.queries_iter()}
    graded_qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id in queries:
            graded_qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    qrels = binarize_qrels(graded_qrels, rel_threshold=spec.rel_threshold or 1)
    return BenchmarkQueries(spec, queries, qrels, graded_qrels)


def default_retrieval_output_dir(spec: BenchmarkSpec, configured: str) -> Path:
    if spec.dataset_id == MSMARCO_DEV_SMALL:
        return Path(configured)
    return Path("outputs") / spec.output_slug / "bm25"


def default_reranker_output_dir(spec: BenchmarkSpec, configured: str) -> Path:
    if spec.dataset_id == MSMARCO_DEV_SMALL:
        return Path(configured)
    return Path("outputs") / spec.output_slug / "cross_encoder_rerank"


def default_retrieval_index_dir(spec: BenchmarkSpec, configured: str) -> Path:
    if spec.uses_msmarco_corpus:
        return Path(configured)
    return Path("data") / "processed" / f"bm25_index_{spec.output_slug}"

"""Dataset selection shared by the full-corpus retrieval runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from msmarco_genqa.data.msmarco import MSMarcoPassage, load_msmarco_passage
from msmarco_genqa.data.trec_dl import DEFAULT_REL_THRESHOLD, TREC_DL_DATASETS, load_trec_dl


MSMARCO_DEV_SMALL = "msmarco-passage/dev/small"
SUPPORTED_DATASETS = (MSMARCO_DEV_SMALL, *TREC_DL_DATASETS.values())


@dataclass(frozen=True)
class BenchmarkSpec:
    """Stable identity and output conventions for one retrieval benchmark."""

    dataset_id: str
    output_slug: str
    track_year: int | None = None
    rel_threshold: int | None = None

    @property
    def is_trec_dl(self) -> bool:
        return self.track_year is not None


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
            "track_year": self.spec.track_year,
            "topic_scope": "judged" if self.spec.is_trec_dl else "dev/small",
            "query_count": len(self.queries),
            "judged_topic_count": self.judged_topic_count,
            "positive_topic_count": self.positive_topic_count,
            "qrels_type": "graded" if self.spec.is_trec_dl else "binary",
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
                track_year=year,
                rel_threshold=DEFAULT_REL_THRESHOLD,
            )
    raise ValueError(
        f"unsupported dataset {dataset_id!r}; expected one of {SUPPORTED_DATASETS}"
    )


def load_benchmark_queries(
    dataset_id: str,
    *,
    cache_dir: Path | str | None = None,
    msmarco_data: MSMarcoPassage | None = None,
) -> BenchmarkQueries:
    """Load query text and qrels without loading the passage corpus eagerly."""
    spec = get_benchmark_spec(dataset_id)
    if not spec.is_trec_dl:
        data = msmarco_data or load_msmarco_passage(
            cache_dir=cache_dir,
            load_corpus=False,
        )
        graded_qrels = {
            qid: {doc_id: 1 for doc_id in doc_ids}
            for qid, doc_ids in data.qrels.items()
        }
        return BenchmarkQueries(spec, data.queries, data.qrels, graded_qrels)

    data = load_trec_dl(
        spec.track_year,
        cache_dir=cache_dir,
        rel_threshold=spec.rel_threshold or DEFAULT_REL_THRESHOLD,
    )
    return BenchmarkQueries(spec, data.queries, data.qrels, data.graded_qrels)


def default_retrieval_output_dir(spec: BenchmarkSpec, configured: str) -> Path:
    if not spec.is_trec_dl:
        return Path(configured)
    return Path("outputs") / spec.output_slug / "bm25"


def default_reranker_output_dir(spec: BenchmarkSpec, configured: str) -> Path:
    if not spec.is_trec_dl:
        return Path(configured)
    return Path("outputs") / spec.output_slug / "cross_encoder_rerank"

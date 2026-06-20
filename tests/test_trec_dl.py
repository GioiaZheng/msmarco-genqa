"""Tests for the TREC-DL 2019/2020 passage-track loaders.

All tests are deterministic and offline: they parse the small committed
fixtures under ``tests/fixtures/trec_dl/`` and never touch ``ir_datasets``
downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from msmarco_genqa.data.trec_dl import (
    DEFAULT_REL_THRESHOLD,
    TREC_DL_DATASETS,
    binarize_qrels,
    build_trec_dl_bundle,
    load_trec_dl,
    load_trec_dl_from_files,
    parse_graded_qrels,
    parse_queries_tsv,
)
from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.evaluation.retrieval_report import read_run_doc_ids

FIXTURES = Path(__file__).parent / "fixtures" / "trec_dl"


def _bundle(year: int):
    return load_trec_dl_from_files(
        FIXTURES / f"queries_{year}.tsv",
        FIXTURES / f"qrels_{year}.txt",
        year=year,
    )


def test_parse_queries_tsv_ignores_comments_and_blanks():
    queries = parse_queries_tsv(
        ["# header", "", "q1\twhat is x", "q2\twhy is y\tz"]
    )
    assert queries == {"q1": "what is x", "q2": "why is y\tz"}


def test_parse_queries_tsv_rejects_missing_tab():
    with pytest.raises(ValueError, match="qid<TAB>text"):
        parse_queries_tsv(["q1 no tab here"])


def test_parse_graded_qrels_keeps_full_scale():
    graded = parse_graded_qrels(
        ["# c", "q1 0 d1 3", "q1 0 d2 0", "q2 d3 2"]
    )
    assert graded == {"q1": {"d1": 3, "d2": 0}, "q2": {"d3": 2}}


def test_parse_graded_qrels_rejects_non_numeric():
    with pytest.raises(ValueError, match="not numeric"):
        parse_graded_qrels(["q1 0 d1 high"])


def test_binarize_qrels_thresholds_and_keeps_empty_queries():
    graded = {"q1": {"d1": 3, "d2": 1}, "q2": {"d3": 0, "d4": 1}}
    binary = binarize_qrels(graded, rel_threshold=2)
    assert binary == {"q1": {"d1"}, "q2": set()}


def test_binarize_qrels_threshold_is_configurable():
    graded = {"q1": {"d1": 1, "d2": 0}}
    assert binarize_qrels(graded, rel_threshold=1) == {"q1": {"d1"}}


@pytest.mark.parametrize("year", [2019, 2020])
def test_loader_returns_msmarco_compatible_schema(year):
    bundle = _bundle(year)
    assert bundle.year == year
    assert bundle.rel_threshold == DEFAULT_REL_THRESHOLD
    # queries: {qid: str}; qrels: {qid: set[str]} — same schema as the
    # MS MARCO loader, so the binary metric functions consume it unchanged.
    assert all(isinstance(text, str) for text in bundle.queries.values())
    assert all(isinstance(docs, set) for docs in bundle.qrels.values())
    assert set(bundle.qrels) <= set(bundle.queries)


def test_loader_2019_binarizes_deep_qrels():
    bundle = _bundle(2019)
    assert bundle.queries.keys() == {"19335", "1037798", "104861"}
    # rel>=2 only: label-1 and label-0 passages drop out.
    assert bundle.qrels["19335"] == {"d_dme_a", "d_dme_b"}
    assert bundle.qrels["1037798"] == {"d_gray_a"}
    # 104861 is judged but has no passage at rel>=2 -> kept, empty.
    assert bundle.qrels["104861"] == set()
    # graded labels retain the full 0-3 scale, including judged-non-relevant.
    assert bundle.graded_qrels["19335"] == {
        "d_dme_a": 3,
        "d_dme_b": 2,
        "d_dme_c": 1,
        "d_dme_d": 0,
    }


def test_loader_2020_loads_both_tracks():
    bundle = _bundle(2020)
    assert bundle.queries.keys() == {"1030303", "1064670"}
    assert bundle.qrels["1030303"] == {"d_h2_a", "d_h2_b"}
    assert bundle.qrels["1064670"] == {"d_shot_a"}


def test_build_bundle_drops_qrels_for_unjudged_queries():
    bundle = build_trec_dl_bundle(
        year=2019,
        dataset_name="unit",
        queries={"q1": "kept"},
        graded_qrels={"q1": {"d1": 3}, "q_unlisted": {"d9": 3}},
    )
    assert set(bundle.graded_qrels) == {"q1"}
    assert set(bundle.qrels) == {"q1"}


def test_metrics_compute_end_to_end_on_binarized_qrels():
    # Acceptance: the same retrieval metric functions run unchanged on the
    # new qrels for a cached-style run file.
    bundle = _bundle(2019)
    runs = read_run_doc_ids(FIXTURES / "run_2019.tsv")
    metrics = evaluate_retrieval(runs, bundle.qrels)
    # 104861 has empty qrels -> skipped, mirroring MS MARCO eval behaviour.
    assert metrics["n_queries"] == 2
    assert metrics["mrr@10"] == pytest.approx(0.5)
    assert metrics["recall@100"] == pytest.approx(1.0)
    # nDCG averages the two evaluable queries (see run fixture header).
    assert metrics["ndcg@10"] == pytest.approx(0.66215, abs=1e-4)


def test_load_trec_dl_rejects_unknown_year():
    with pytest.raises(ValueError, match="unsupported TREC-DL year"):
        load_trec_dl(2018)


def test_dataset_registry_covers_both_tracks():
    assert set(TREC_DL_DATASETS) == {2019, 2020}
    assert all("trec-dl" in name for name in TREC_DL_DATASETS.values())

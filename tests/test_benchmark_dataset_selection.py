"""Offline coverage for dataset selection in the full-corpus runners."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_reranker import (
    first_stage_label,
    load_upstream_benchmark_metadata,
    parse_args as parse_reranker_args,
    resolve_input_run,
    resolve_output_dir as resolve_reranker_output_dir,
    select_eval_qids,
    validate_trec_input_run,
)
from experiments.run_retrieval import (
    parse_args as parse_retrieval_args,
    resolve_output_dir as resolve_retrieval_output_dir,
)
from msmarco_genqa.data import benchmark as benchmark_module
from msmarco_genqa.data.benchmark import (
    MSMARCO_DEV_SMALL,
    SUPPORTED_DATASETS,
    get_benchmark_spec,
    load_benchmark_queries,
)
from msmarco_genqa.data.trec_dl import TREC_DL_DATASETS, load_trec_dl_from_files


FIXTURES = Path(__file__).parent / "fixtures" / "trec_dl"


@pytest.fixture
def cfg() -> dict:
    return {
        "eval_retrieval": {"output_dir": "outputs/bm25_baseline"},
        "reranker": {"output_dir": "outputs/cross_encoder_rerank"},
    }


@pytest.mark.parametrize("year", [2019, 2020])
def test_trec_tracks_load_through_shared_selector_without_network(year, monkeypatch):
    bundle = load_trec_dl_from_files(
        FIXTURES / f"queries_{year}.tsv",
        FIXTURES / f"qrels_{year}.txt",
        year=year,
        dataset_name=TREC_DL_DATASETS[year],
    )
    monkeypatch.setattr(benchmark_module, "load_trec_dl", lambda *args, **kwargs: bundle)

    selected = load_benchmark_queries(TREC_DL_DATASETS[year])

    assert selected.spec.track_year == year
    assert selected.spec.dataset_id == TREC_DL_DATASETS[year]
    assert selected.queries == bundle.queries
    assert selected.qrels == bundle.qrels
    assert selected.graded_qrels == bundle.graded_qrels
    metadata = selected.metadata()
    assert metadata["dataset_id"] == TREC_DL_DATASETS[year]
    assert metadata["track_year"] == year
    assert metadata["topic_scope"] == "judged"
    assert metadata["judged_topic_count"] == len(bundle.graded_qrels)
    assert metadata["qrels_type"] == "graded"
    assert metadata["relevance_threshold"] == 2


def test_supported_datasets_keep_dev_small_as_default():
    assert SUPPORTED_DATASETS == (
        MSMARCO_DEV_SMALL,
        TREC_DL_DATASETS[2019],
        TREC_DL_DATASETS[2020],
    )
    assert parse_retrieval_args([]).dataset == MSMARCO_DEV_SMALL
    assert parse_reranker_args([]).dataset == MSMARCO_DEV_SMALL


@pytest.mark.parametrize("year", [2019, 2020])
def test_trec_default_outputs_are_isolated_by_year(year, cfg, tmp_path):
    dataset_id = TREC_DL_DATASETS[year]
    spec = get_benchmark_spec(dataset_id)
    retrieval_args = parse_retrieval_args(["--dataset", dataset_id])
    reranker_args = parse_reranker_args(["--dataset", dataset_id])

    retrieval_dir = resolve_retrieval_output_dir(
        retrieval_args,
        cfg,
        spec,
        project_root=tmp_path,
    )
    reranker_dir = resolve_reranker_output_dir(
        reranker_args,
        cfg,
        spec,
        project_root=tmp_path,
    )

    assert retrieval_dir == tmp_path / "outputs" / f"trec_dl_{year}" / "bm25"
    assert reranker_dir == (
        tmp_path / "outputs" / f"trec_dl_{year}" / "cross_encoder_rerank"
    )
    assert resolve_input_run(reranker_args, cfg, spec, project_root=tmp_path) == (
        retrieval_dir / "run.tsv"
    )
    assert first_stage_label(spec, retrieval_dir / "run.tsv") == "bm25"


def test_dev_small_paths_remain_backward_compatible(cfg, tmp_path):
    spec = get_benchmark_spec(MSMARCO_DEV_SMALL)
    retrieval_args = parse_retrieval_args([])
    reranker_args = parse_reranker_args([])

    assert resolve_retrieval_output_dir(
        retrieval_args, cfg, spec, project_root=tmp_path
    ) == (tmp_path / "outputs" / "bm25_baseline")
    assert resolve_reranker_output_dir(
        reranker_args, cfg, spec, project_root=tmp_path
    ) == (tmp_path / "outputs" / "cross_encoder_rerank")
    assert resolve_input_run(reranker_args, cfg, spec, project_root=tmp_path) == (
        tmp_path / "outputs" / "dense_retrieval" / "run.tsv"
    )


def test_selected_trec_topics_include_empty_binary_positive_sets():
    runs = {"q1": [("d1", 1.0)], "q2": [("d2", 1.0)], "other": [("d9", 1.0)]}
    queries = {"q1": "one", "q2": "two"}
    qrels = {"q1": {"d1"}, "q2": set()}

    assert select_eval_qids(runs, queries, qrels) == ["q1", "q2"]


def test_trec_reranker_rejects_missing_or_cross_track_topics():
    spec = get_benchmark_spec(TREC_DL_DATASETS[2019])
    queries = {"q1": "one", "q2": "two"}

    with pytest.raises(SystemExit, match="missing 1 topics"):
        validate_trec_input_run(spec, {"q1": [("d1", 1.0)]}, queries, {})

    with pytest.raises(SystemExit, match="Refusing to mix benchmark tracks"):
        validate_trec_input_run(
            spec,
            {"q1": [("d1", 1.0)], "q2": [("d2", 1.0)]},
            queries,
            {"dataset_id": TREC_DL_DATASETS[2020]},
        )


def test_upstream_corpus_scope_is_read_from_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(
        '{"benchmark":{"dataset_id":"track","corpus_scope":"first-N-truncated"}}',
        encoding="utf-8",
    )

    assert load_upstream_benchmark_metadata(tmp_path) == {
        "dataset_id": "track",
        "corpus_scope": "first-N-truncated",
    }

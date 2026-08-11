"""Tests for fixed-output first-stage coverage diagnostics."""

from __future__ import annotations

import math

import pytest

from msmarco_genqa.evaluation.first_stage_coverage import (
    FirstStageCoverageError,
    analyze_first_stage_coverage,
    assert_first_stage_diagnostic_fingerprint,
    deterministic_bucket_samples,
    first_hit_bucket,
    first_stage_diagnostic_fingerprint,
    relevant_coverage_at_100_bucket,
    render_first_stage_coverage_markdown,
)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (1, "top_10"),
        (10, "top_10"),
        (11, "ranks_11_100"),
        (100, "ranks_11_100"),
        (101, "ranks_101_1000"),
        (1000, "ranks_101_1000"),
        (None, "miss_top_1000"),
    ],
)
def test_first_hit_bucket_boundaries(rank, expected):
    assert first_hit_bucket(rank) == expected


@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_first_hit_bucket_rejects_invalid_ranks(rank):
    with pytest.raises(FirstStageCoverageError):
        first_hit_bucket(rank)


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        (0.0, "none"),
        (0.01, "partial_lt_25pct"),
        (0.249999, "partial_lt_25pct"),
        (0.25, "partial_25_to_lt_50pct"),
        (0.499999, "partial_25_to_lt_50pct"),
        (0.5, "partial_50_to_lt_100pct"),
        (0.999999, "partial_50_to_lt_100pct"),
        (1.0, "complete"),
    ],
)
def test_relevant_coverage_bucket_boundaries(coverage, expected):
    assert relevant_coverage_at_100_bucket(coverage) == expected


@pytest.mark.parametrize("coverage", [-0.1, 1.1, math.inf, math.nan])
def test_relevant_coverage_bucket_rejects_invalid_values(coverage):
    with pytest.raises(FirstStageCoverageError):
        relevant_coverage_at_100_bucket(coverage)


def _synthetic_analysis():
    qrels = {
        "q1": {"d1": 1, "d2": 2},
        "q2": {"d3": 1},
        "q3": {"d4": 1, "d5": 1},
        "q4": {"d6": 1},
    }
    queries = {qid: f"query {qid}" for qid in qrels}
    run = {
        "q1": [
            ("d1", 5.0),
            *[(f"x1-{rank}", 4.0 - rank / 1000) for rank in range(2, 101)],
            ("d2", 1.0),
        ],
        "q2": [
            *[(f"x2-{rank}", 4.0 - rank / 1000) for rank in range(1, 50)],
            ("d3", 1.0),
        ],
        "q3": [
            *[(f"x3-{rank}", 4.0 - rank / 1000) for rank in range(1, 500)],
            ("d4", 1.0),
        ],
        "q4": [(f"x4-{rank}", 4.0 - rank / 1000) for rank in range(1, 1001)],
    }
    return analyze_first_stage_coverage(run, qrels, queries)


def test_analysis_reconciles_macro_micro_and_candidate_set_counts():
    report = _synthetic_analysis()

    assert report["scope"]["n_queries"] == 4
    assert report["scope"]["n_positive_qrels"] == 6
    assert report["macro_recall"]["recall@100"] == pytest.approx(0.375)
    assert report["macro_recall"]["recall@1000"] == pytest.approx(0.625)
    assert report["micro_qrels_coverage"]["recall@100"] == pytest.approx(2 / 6)
    assert report["micro_qrels_coverage"]["recall@1000"] == pytest.approx(4 / 6)
    assert report["candidate_set_diagnostic"] == {
        "queries_with_any_relevant_in_top_100": 2,
        "queries_with_no_relevant_in_top_100": 2,
        "queries_with_complete_relevant_coverage_at_100": 1,
        "queries_with_partial_relevant_coverage_at_100": 1,
        "positive_qrels_in_top_100": 2,
        "positive_qrels_outside_top_100": 4,
    }
    assert report["depth_100_to_1000_diagnostic"][
        "additional_positive_qrels_found"
    ] == 2
    assert report["reconciliation"]["first_hit_bucket_queries"] == 4
    assert report["reconciliation"]["coverage_at_100_bucket_queries"] == 4


def test_analysis_assigns_expected_query_buckets():
    rows = {row["qid"]: row for row in _synthetic_analysis()["per_query"]}

    assert rows["q1"]["first_hit_bucket"] == "top_10"
    assert rows["q1"]["coverage_at_100_bucket"] == "partial_50_to_lt_100pct"
    assert rows["q2"]["first_hit_bucket"] == "ranks_11_100"
    assert rows["q2"]["coverage_at_100_bucket"] == "complete"
    assert rows["q3"]["first_hit_bucket"] == "ranks_101_1000"
    assert rows["q3"]["coverage_at_100_bucket"] == "none"
    assert rows["q4"]["first_hit_bucket"] == "miss_top_1000"
    assert rows["q4"]["coverage_at_100_bucket"] == "none"


def test_analysis_rejects_missing_or_extra_run_queries():
    qrels = {"q1": {"d1": 1}}
    queries = {"q1": "query"}

    with pytest.raises(FirstStageCoverageError, match="qid mismatch"):
        analyze_first_stage_coverage({}, qrels, queries)
    with pytest.raises(FirstStageCoverageError, match="qid mismatch"):
        analyze_first_stage_coverage(
            {"q1": [("d1", 1.0)], "extra": [("d2", 0.0)]},
            qrels,
            queries,
        )


def test_deterministic_samples_are_stable_and_cover_observed_buckets():
    rows = _synthetic_analysis()["per_query"]

    first = deterministic_bucket_samples(rows, per_bucket=1, seed="fixed")
    second = deterministic_bucket_samples(rows, per_bucket=1, seed="fixed")

    assert first == second
    assert {
        (row["sample_dimension"], row["sample_bucket"]) for row in first
    } == {
        ("first_hit_bucket", "top_10"),
        ("first_hit_bucket", "ranks_11_100"),
        ("first_hit_bucket", "ranks_101_1000"),
        ("first_hit_bucket", "miss_top_1000"),
        ("coverage_at_100_bucket", "none"),
        ("coverage_at_100_bucket", "partial_50_to_lt_100pct"),
        ("coverage_at_100_bucket", "complete"),
    }


def test_diagnostic_fingerprint_is_exact_and_rejects_drift():
    report = _synthetic_analysis()
    expected = first_stage_diagnostic_fingerprint(report)

    assert_first_stage_diagnostic_fingerprint(report, expected)
    drifted = {**expected, "positive_qrels_in_top_100": 999}
    with pytest.raises(FirstStageCoverageError, match="diagnostic drift"):
        assert_first_stage_diagnostic_fingerprint(report, drifted)


def test_markdown_states_definitions_and_interpretation_boundary():
    report = _synthetic_analysis()
    samples = deterministic_bucket_samples(report["per_query"], per_bucket=1)
    report = {key: value for key, value in report.items() if key != "per_query"}

    markdown = render_first_stage_coverage_markdown(
        report,
        dataset_id="beir/nfcorpus/test",
        contract_path="configs/contract.json",
        release_tag="v1",
        samples=samples,
    )

    assert "`ranks_101_1000`" in markdown
    assert markdown.startswith("# NFCorpus First-Stage Coverage Analysis")
    assert "Macro (headline definition)" in markdown
    assert "Micro qrels coverage (diagnostic only)" in markdown
    assert "Interpretation Boundary" in markdown


def test_markdown_title_uses_dataset_name():
    report = _synthetic_analysis()
    samples = deterministic_bucket_samples(report["per_query"], per_bucket=1)
    report = {key: value for key, value in report.items() if key != "per_query"}

    markdown = render_first_stage_coverage_markdown(
        report,
        dataset_id="beir/scifact/test",
        contract_path="configs/contract.json",
        release_tag="v1",
        samples=samples,
    )

    assert markdown.startswith("# SciFact First-Stage Coverage Analysis")

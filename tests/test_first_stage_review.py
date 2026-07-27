"""Tests for NFCorpus first-stage retrieval review construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msmarco_genqa.evaluation.first_stage_review import (
    FirstStageReviewError,
    build_first_stage_review_cases,
    load_review_taxonomy,
    partition_query_ids_by_source,
    summarize_query_source_diagnostics,
    surface_tokens,
    validate_review_annotations,
)


def _taxonomy() -> dict:
    return {
        "schema": "msmarco-genqa.first-stage-review-taxonomy.v1",
        "dataset_id": "beir/nfcorpus/test",
        "review_statuses": ["pending", "reviewed", "needs_adjudication"],
        "labels": {
            "lexical_competition": {
                "definition": "Relevant and non-relevant documents share query terms.",
                "experiment_implication": "Test ranking discrimination.",
            },
            "other_unclear": {
                "definition": "Evidence is insufficient.",
                "experiment_implication": "Require adjudication.",
            },
        },
        "evidence_note_min_chars": 12,
        "selection": {
            "seed": "fixture",
            "cohorts": {
                "depth_recoverable_101_1000": 1,
                "miss_top_1000": 1,
            },
        },
    }


def _fixture():
    per_query = [
        {
            "qid": "q-depth",
            "query": "diet cataract",
            "first_hit_bucket": "ranks_101_1000",
            "first_relevant_rank": 101,
            "recall@100": 0.0,
            "recall@1000": 1.0,
        },
        {
            "qid": "q-miss",
            "query": "rare syndrome",
            "first_hit_bucket": "miss_top_1000",
            "first_relevant_rank": None,
            "recall@100": 0.0,
            "recall@1000": 0.0,
        },
        {
            "qid": "q-out",
            "query": "outside",
            "first_hit_bucket": "top_10",
            "first_relevant_rank": 1,
            "recall@100": 1.0,
            "recall@1000": 1.0,
        },
    ]
    depth_rows = [(f"dx-{rank}", float(1001 - rank)) for rank in range(1, 101)]
    depth_rows.append(("rel-depth", 0.5))
    depth_rows.extend(
        (f"dx-{rank}", float(1001 - rank))
        for rank in range(102, 1001)
    )
    miss_rows = [(f"mx-{rank}", float(1001 - rank)) for rank in range(1, 1001)]
    run = {
        "q-depth": depth_rows,
        "q-miss": miss_rows,
    }
    qrels = {
        "q-depth": {"rel-depth": 2},
        "q-miss": {"rel-miss": 1},
    }
    query_records = {
        "q-depth": {
            "_id": "q-depth",
            "text": "diet cataract",
            "metadata": {"url": "https://example.test/video/diet-cataract/"},
        },
        "q-miss": {
            "_id": "q-miss",
            "text": "rare syndrome",
            "metadata": {"url": "https://example.test/topics/rare-syndrome/"},
        },
        "q-out": {
            "_id": "q-out",
            "text": "outside",
            "metadata": {"url": "https://example.test/questions/outside/"},
        },
    }
    corpus = {
        "rel-depth": {
            "_id": "rel-depth",
            "title": "Diet and cataract prevention",
            "text": "Diet may affect cataract risk.",
        },
        "rel-miss": {
            "_id": "rel-miss",
            "title": "A different clinical name",
            "text": "The condition is described without the query wording.",
        },
    }
    for doc_id, _score in (*depth_rows, *miss_rows):
        corpus.setdefault(
            doc_id,
            {
                "_id": doc_id,
                "title": "Diet cataract competing result",
                "text": "A ranked document containing the query terms.",
            },
        )
    return per_query, run, qrels, corpus, query_records


def _annotations(cases):
    return [
        {
            "qid": case["qid"],
            "cohort": case["cohort"],
            "review_status": "pending",
            "primary_label": "",
            "secondary_label": "",
            "evidence_note": "",
        }
        for case in cases
    ]


def test_surface_tokens_are_stable_and_drop_common_words():
    assert surface_tokens("What is Diet-and Cataract?") == ["diet", "cataract"]


def test_review_cases_are_complete_deterministic_and_evidence_backed():
    per_query, run, qrels, corpus, query_records = _fixture()

    first = build_first_stage_review_cases(
        per_query,
        run,
        qrels,
        corpus,
        _taxonomy(),
        query_records=query_records,
    )
    second = build_first_stage_review_cases(
        per_query,
        run,
        qrels,
        corpus,
        _taxonomy(),
        query_records=query_records,
    )

    assert first == second
    assert len(first) == 2
    by_qid = {row["qid"]: row for row in first}
    assert by_qid["q-depth"]["cohort"] == "depth_recoverable_101_1000"
    assert by_qid["q-depth"]["first_relevant_rank"] == 101
    assert by_qid["q-depth"]["query_source_type"] == "video"
    assert by_qid["q-depth"]["query_source_url"].endswith("/diet-cataract/")
    assert by_qid["q-depth"]["representative_relevant_documents"][0]["rank"] == 101
    assert by_qid["q-depth"]["max_positive_qrel_query_token_recall"] == 1.0
    assert by_qid["q-depth"]["positive_qrels_by_relevance"] == {"2": 1}
    assert by_qid["q-miss"]["cohort"] == "miss_top_1000"
    assert by_qid["q-miss"]["first_relevant_rank"] is None
    assert by_qid["q-miss"]["positive_qrels_by_relevance"] == {"1": 1}
    assert "only_relevance_level_1_qrels" in by_qid["q-miss"][
        "diagnostic_flags"
    ]
    assert len({row["selection_sha256"] for row in first}) == 2


def test_query_source_diagnostics_use_all_queries_as_denominator():
    per_query, _run, _qrels, _corpus, query_records = _fixture()
    query_records["q-miss"]["metadata"]["url"] = (
        "https://example.test/2014/01/02/rare-syndrome/"
    )

    diagnostics = summarize_query_source_diagnostics(
        per_query,
        query_records,
    )

    assert diagnostics["video"]["n_queries"] == 1
    assert diagnostics["video"]["no_relevant_top_100"] == 1
    assert diagnostics["video"]["depth_recoverable_101_1000"] == 1
    assert diagnostics["dated_article"]["n_queries"] == 1
    assert diagnostics["dated_article"]["miss_top_1000"] == 1
    assert diagnostics["question"]["n_queries"] == 1


def test_query_source_partition_is_complete_and_sorted():
    _rows, _run, _qrels, _corpus, query_records = _fixture()

    groups = partition_query_ids_by_source(
        ["q-miss", "q-out", "q-depth"],
        query_records,
    )

    assert groups == {
        "question": ["q-out"],
        "topic": ["q-miss"],
        "video": ["q-depth"],
    }

    with pytest.raises(FirstStageReviewError, match="duplicate query id"):
        partition_query_ids_by_source(["q-depth", "q-depth"], query_records)


def test_review_case_builder_rejects_cohort_drift():
    per_query, run, qrels, corpus, query_records = _fixture()
    taxonomy = _taxonomy()
    taxonomy["selection"]["cohorts"]["miss_top_1000"] = 2

    with pytest.raises(FirstStageReviewError, match="cohort drift"):
        build_first_stage_review_cases(
            per_query,
            run,
            qrels,
            corpus,
            taxonomy,
            query_records=query_records,
        )


def test_annotation_summary_uses_reviewed_denominator_only():
    per_query, run, qrels, corpus, query_records = _fixture()
    taxonomy = _taxonomy()
    cases = build_first_stage_review_cases(
        per_query,
        run,
        qrels,
        corpus,
        taxonomy,
        query_records=query_records,
    )
    annotations = _annotations(cases)
    annotations[0].update(
        {
            "review_status": "reviewed",
            "primary_label": "lexical_competition",
            "evidence_note": "Query terms occur in both evidence groups.",
        }
    )

    summary = validate_review_annotations(annotations, cases, taxonomy)

    assert summary["n_cases"] == 2
    assert summary["n_reviewed"] == 1
    assert summary["review_coverage"] == 0.5
    assert summary["primary_label_counts"] == {"lexical_competition": 1}
    assert summary["status_counts"] == {"pending": 1, "reviewed": 1}
    assert summary["objective_qrel_evidence_counts"] == {
        "has_relevance_level_2_or_higher": 1,
        "only_relevance_level_1": 1,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("primary_label", "invented", "invalid primary_label"),
        ("cohort", "wrong", "cohort does not match"),
        ("review_status", "done", "invalid review_status"),
    ],
)
def test_annotation_validation_rejects_invalid_values(field, value, match):
    per_query, run, qrels, corpus, query_records = _fixture()
    taxonomy = _taxonomy()
    cases = build_first_stage_review_cases(
        per_query,
        run,
        qrels,
        corpus,
        taxonomy,
        query_records=query_records,
    )
    annotations = _annotations(cases)
    annotations[0][field] = value

    with pytest.raises(FirstStageReviewError, match=match):
        validate_review_annotations(annotations, cases, taxonomy)


def test_reviewed_annotation_requires_evidence_note():
    per_query, run, qrels, corpus, query_records = _fixture()
    taxonomy = _taxonomy()
    cases = build_first_stage_review_cases(
        per_query,
        run,
        qrels,
        corpus,
        taxonomy,
        query_records=query_records,
    )
    annotations = _annotations(cases)
    annotations[0].update(
        {
            "review_status": "reviewed",
            "primary_label": "other_unclear",
            "evidence_note": "short",
        }
    )

    with pytest.raises(FirstStageReviewError, match="at least 12"):
        validate_review_annotations(annotations, cases, taxonomy)


def test_taxonomy_loader_validates_schema_and_label_records(tmp_path: Path):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(_taxonomy()), encoding="utf-8")

    loaded = load_review_taxonomy(path)

    assert loaded["selection"]["seed"] == "fixture"
    assert set(loaded["labels"]) == {"lexical_competition", "other_unclear"}

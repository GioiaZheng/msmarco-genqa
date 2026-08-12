"""Tests for SciFact residual first-stage failure review."""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.scifact_failure_review import (
    SciFactFailureReviewError,
    assert_scifact_failure_fingerprint,
    build_scifact_failure_cases,
    render_scifact_failure_review_markdown,
    scifact_failure_fingerprint,
    summarize_scifact_failure_review,
)


def _fixture():
    per_query = [
        {
            "qid": "q-depth",
            "query": "Protein alpha inhibits tumor growth.",
            "first_hit_bucket": "ranks_101_1000",
            "first_relevant_rank": 101,
            "recall@100": 0.0,
            "recall@1000": 1.0,
        },
        {
            "qid": "q-miss",
            "query": "Obesity decreases life quality.",
            "first_hit_bucket": "miss_top_1000",
            "first_relevant_rank": None,
            "recall@100": 0.0,
            "recall@1000": 0.0,
        },
        {
            "qid": "q-ok",
            "query": "Matched claim.",
            "first_hit_bucket": "top_10",
            "first_relevant_rank": 1,
            "recall@100": 1.0,
            "recall@1000": 1.0,
        },
    ]
    depth_rows = [(f"dx-{rank}", float(1001 - rank)) for rank in range(1, 101)]
    depth_rows.append(("rel-depth", 0.5))
    depth_rows.extend(
        (f"dx-{rank}", float(1001 - rank)) for rank in range(102, 1001)
    )
    miss_rows = [(f"mx-{rank}", float(1001 - rank)) for rank in range(1, 1001)]
    run = {
        "q-depth": depth_rows,
        "q-miss": miss_rows,
        "q-ok": [("rel-ok", 1.0)],
    }
    qrels = {
        "q-depth": {"rel-depth": 1},
        "q-miss": {"rel-shared": 1},
        "q-ok": {"rel-ok": 1},
    }
    corpus = {
        "rel-depth": {
            "_id": "rel-depth",
            "title": "A different biological mechanism",
            "text": "The evidence discusses a pathway with indirect wording.",
        },
        "rel-shared": {
            "_id": "rel-shared",
            "title": "Body mass index and mortality",
            "text": "A population study about BMI and cause-specific mortality.",
        },
        "rel-ok": {
            "_id": "rel-ok",
            "title": "Matched claim",
            "text": "Matched claim.",
        },
    }
    for doc_id, _score in (*depth_rows, *miss_rows):
        title = (
            "Obesity decreases life quality competing result"
            if doc_id.startswith("mx-")
            else "Protein alpha tumor growth competing result"
        )
        corpus.setdefault(
            doc_id,
            {
                "_id": doc_id,
                "title": title,
                "text": "This ranked document repeats the claim terms more directly.",
            },
        )
    return per_query, run, qrels, corpus


def test_scifact_review_builds_residual_cases_only():
    per_query, run, qrels, corpus = _fixture()

    cases = build_scifact_failure_cases(per_query, run, qrels, corpus)
    summary = summarize_scifact_failure_review(cases)

    assert [case["qid"] for case in cases] == ["q-depth", "q-miss"]
    assert summary["n_cases"] == 2
    assert summary["cohort_counts"] == {
        "depth_recoverable_101_1000": 1,
        "miss_top_1000": 1,
    }
    assert summary["diagnostic_flag_counts"]["top_lexical_competition"] == 2
    assert cases[0]["primary_label"] == "terminology_or_evidence_form_mismatch"
    assert cases[1]["primary_label"] == "short_or_broad_claim"


def test_scifact_review_fingerprint_rejects_drift():
    per_query, run, qrels, corpus = _fixture()
    summary = summarize_scifact_failure_review(
        build_scifact_failure_cases(per_query, run, qrels, corpus)
    )
    expected = scifact_failure_fingerprint(summary)

    assert_scifact_failure_fingerprint(summary, expected)
    drifted = {**expected, "n_cases": 999}
    with pytest.raises(SciFactFailureReviewError, match="drift"):
        assert_scifact_failure_fingerprint(summary, drifted)


def test_scifact_review_markdown_states_boundary():
    per_query, run, qrels, corpus = _fixture()
    cases = build_scifact_failure_cases(per_query, run, qrels, corpus)
    markdown = render_scifact_failure_review_markdown(
        cases,
        summarize_scifact_failure_review(cases),
    )

    assert markdown.startswith("# SciFact Residual First-Stage Failure Review")
    assert "retrieval-only" in markdown
    assert "Keep the pipeline frozen" in markdown
    assert "terminology_or_evidence_form_mismatch" in markdown


def test_scifact_review_rejects_missing_corpus_evidence():
    per_query, run, qrels, corpus = _fixture()
    del corpus["rel-depth"]

    with pytest.raises(SciFactFailureReviewError, match="missing document"):
        build_scifact_failure_cases(per_query, run, qrels, corpus)

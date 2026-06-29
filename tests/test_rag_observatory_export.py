from __future__ import annotations

import json
from pathlib import Path

import pytest

from msmarco_genqa.cli import rag_observatory_export as cli
from msmarco_genqa.interop.rag_observatory import (
    EXPORT_FORMAT,
    RagObservatoryExportError,
    build_trace_export,
    load_prediction_rows,
    load_qrels,
    select_prediction_row,
    validate_trace_export,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rag_observatory_export"


def test_build_trace_export_matches_adapter_contract():
    rows = load_prediction_rows(FIXTURE_DIR / "predictions.jsonl")
    row = select_prediction_row(rows, query_id="msmarco-synthetic-q001")
    qrels = load_qrels(FIXTURE_DIR / "qrels.tsv")

    export = build_trace_export(
        row,
        run_id="synthetic-msmarco-genqa-run-001",
        timestamp="2026-06-29T00:00:00Z",
        dataset="synthetic-msmarco-genqa",
        config_hash="synthetic-config-hash",
        code_version="fixture",
        retriever="synthetic-bm25",
        generator="synthetic-generator",
        evaluator="deterministic-rag-triad",
        random_seed=17,
        qrels=qrels,
        export_profile="unit-test",
    )

    validate_trace_export(export)
    assert export["format"] == EXPORT_FORMAT
    assert export["run"]["run_id"] == "synthetic-msmarco-genqa-run-001"
    assert export["query"]["query_id"] == "msmarco-synthetic-q001"
    assert export["retrieved_documents"][0]["is_relevant"] is True
    assert export["retrieved_documents"][1]["is_relevant"] is False
    assert export["selected_context"][0]["doc_id"] == "doc-penicillin"
    assert export["answer"]["citations"][0]["doc_id"] == "doc-penicillin"
    assert {metric["name"] for metric in export["metrics"]} >= {
        "context_relevance",
        "groundedness",
        "answer_relevance",
        "triad",
    }


def test_cli_writes_public_safe_fixture_export(tmp_path: Path):
    output = tmp_path / "trace_export.json"

    cli.main(
        [
            "--predictions",
            str(FIXTURE_DIR / "predictions.jsonl"),
            "--qrels",
            str(FIXTURE_DIR / "qrels.tsv"),
            "--query-id",
            "msmarco-synthetic-q001",
            "--run-id",
            "synthetic-msmarco-genqa-run-001",
            "--timestamp",
            "2026-06-29T00:00:00Z",
            "--dataset",
            "synthetic-msmarco-genqa",
            "--retriever",
            "synthetic-bm25",
            "--generator",
            "synthetic-generator",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_trace_export(payload)
    assert payload["format"] == EXPORT_FORMAT
    assert payload["extra"]["source_predictions"].endswith("predictions.jsonl")


def test_export_rejects_missing_context_text():
    bad_row = {
        "query_id": "q1",
        "query": "Who discovered penicillin?",
        "top_doc_ids": ["doc-penicillin"],
        "passages": [""],
        "prediction": "Alexander Fleming",
        "references": ["Alexander Fleming"],
    }

    with pytest.raises(RagObservatoryExportError, match="missing passage text"):
        build_trace_export(bad_row, timestamp="2026-06-29T00:00:00Z")


def test_validator_rejects_unknown_top_level_field():
    rows = load_prediction_rows(FIXTURE_DIR / "predictions.jsonl")
    export = build_trace_export(rows[0], timestamp="2026-06-29T00:00:00Z")
    export["unexpected"] = True

    with pytest.raises(RagObservatoryExportError, match="unknown field"):
        validate_trace_export(export)


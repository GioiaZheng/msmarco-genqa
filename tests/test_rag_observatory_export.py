from __future__ import annotations

import json
from pathlib import Path

import pytest

from msmarco_genqa.cli import rag_observatory_export as cli
from msmarco_genqa.cli import rag_observatory_sweep_export as sweep_cli
from msmarco_genqa.interop.rag_observatory import (
    EXPORT_FORMAT,
    RagObservatoryExportError,
    SWEEP_EXPORT_FORMAT,
    build_sweep_manifest,
    build_trace_export,
    load_prediction_rows,
    load_qrels,
    select_prediction_row,
    validate_sweep_manifest,
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


def test_build_sweep_manifest_links_configured_traces():
    qrels = load_qrels(FIXTURE_DIR / "qrels.tsv")
    bm25_row = select_prediction_row(
        load_prediction_rows(FIXTURE_DIR / "predictions.jsonl"),
        query_id="msmarco-synthetic-q001",
    )
    dense_row = select_prediction_row(
        load_prediction_rows(FIXTURE_DIR / "predictions_dense_rerank.jsonl"),
        query_id="msmarco-synthetic-q001",
    )

    exports = [
        build_trace_export(
            bm25_row,
            run_id="synthetic-sweep-bm25-msmarco-synthetic-q001",
            timestamp="2026-07-01T00:00:00Z",
            dataset="synthetic-msmarco-genqa",
            retriever="synthetic-bm25",
            generator="synthetic-generator",
            config_id="bm25",
            config={"retriever": "synthetic-bm25", "top_k": 2, "reranking": False},
            qrels=qrels,
            export_profile="config-sweep",
        ),
        build_trace_export(
            dense_row,
            run_id="synthetic-sweep-dense-rerank-msmarco-synthetic-q001",
            timestamp="2026-07-01T00:00:00Z",
            dataset="synthetic-msmarco-genqa",
            retriever="synthetic-dense-rerank",
            reranker="present",
            generator="synthetic-generator",
            config_id="dense-rerank",
            config={"retriever": "synthetic-dense-rerank", "top_k": 2, "reranking": True},
            qrels=qrels,
            export_profile="config-sweep",
        ),
    ]

    assert exports[1]["reranked_documents"][0]["doc_id"] == "reranked:doc-penicillin"
    assert exports[1]["reranked_documents"][0]["extra"]["original_doc_id"] == "doc-penicillin"
    assert exports[1]["reranked_documents"][0]["is_relevant"] is True

    manifest = build_sweep_manifest(
        exports,
        trace_paths=[
            "traces/bm25/msmarco-synthetic-q001.json",
            "traces/dense-rerank/msmarco-synthetic-q001.json",
        ],
        sweep_id="synthetic-trace-sweep-001",
        timestamp="2026-07-01T00:00:00Z",
        dataset="synthetic-msmarco-genqa",
    )

    validate_sweep_manifest(manifest)
    assert manifest["format"] == SWEEP_EXPORT_FORMAT
    assert [row["config_id"] for row in manifest["comparison"]["rows"]] == [
        "bm25",
        "dense-rerank",
    ]
    assert "triad" in manifest["comparison"]["metric_names"]
    assert manifest["configurations"][1]["config"]["has_reranked_documents"] is True


def test_sweep_cli_writes_manifest_and_trace_files(tmp_path: Path):
    output_dir = tmp_path / "sweep"

    sweep_cli.main(
        [
            "--arm",
            f"bm25={FIXTURE_DIR / 'predictions.jsonl'}",
            "--arm",
            f"dense-rerank={FIXTURE_DIR / 'predictions_dense_rerank.jsonl'}",
            "--qrels",
            str(FIXTURE_DIR / "qrels.tsv"),
            "--query-id",
            "msmarco-synthetic-q001",
            "--sweep-id",
            "synthetic-trace-sweep-001",
            "--timestamp",
            "2026-07-01T00:00:00Z",
            "--dataset",
            "synthetic-msmarco-genqa",
            "--generator",
            "synthetic-generator",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "rag_observatory_sweep.json").read_text())
    validate_sweep_manifest(manifest)
    assert (output_dir / "traces" / "bm25" / "msmarco-synthetic-q001.json").exists()
    assert (output_dir / "traces" / "dense-rerank" / "msmarco-synthetic-q001.json").exists()
    assert manifest["sweep"]["supported_dimensions"] == [
        "config_id",
        "retriever",
        "reranker",
        "generator",
        "top_k",
    ]


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

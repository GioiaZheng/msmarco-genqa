from __future__ import annotations

import json

import pytest

from msmarco_genqa.cli import context_packing_report as cli
from msmarco_genqa.evaluation.context_packing_report import (
    PredictionPairingError,
    build_context_packing_report,
    load_prediction_jsonl,
)


def test_build_context_packing_report_compares_matched_qids():
    report = build_context_packing_report(
        {
            "q1": {
                "query": "where was ada born",
                "passages": ["Ada Lovelace was born in London.", "Extra context."],
                "prediction": "London",
                "references": ["London"],
            },
            "q_baseline_only": {
                "query": "unused",
                "passages": ["unused"],
                "prediction": "unused",
                "references": ["unused"],
            },
        },
        {
            "q1": {
                "query": "where was ada born",
                "passages": ["Ada Lovelace was born in London."],
                "prediction": "London",
                "references": ["London"],
                "context_packing": {"retained_doc_ids": ["d1"]},
            },
            "q_compressed_only": {
                "query": "unused",
                "passages": ["unused"],
                "prediction": "unused",
                "references": ["unused"],
            },
        },
        baseline_name="plain",
        compressed_name="packed",
    )

    assert report["coverage"]["n_matched_qids"] == 1
    assert report["coverage"]["n_baseline_only_qids"] == 1
    assert report["coverage"]["n_compressed_only_qids"] == 1
    assert report["metrics"]["plain"]["mean_token_f1"] == pytest.approx(1.0)
    assert report["metrics"]["packed"]["mean_context_chars"] < report["metrics"]["plain"][
        "mean_context_chars"
    ]
    assert report["metrics"]["delta"]["mean_context_chars"] < 0


def test_build_context_packing_report_rejects_reference_mismatch():
    with pytest.raises(PredictionPairingError, match="references differ"):
        build_context_packing_report(
            {"q1": {"prediction": "a", "references": ["a"], "passages": ["a"]}},
            {"q1": {"prediction": "a", "references": ["b"], "passages": ["a"]}},
        )


def test_load_prediction_jsonl_rejects_duplicate_query_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"query_id":"q1","prediction":"a"}\n'
        '{"query_id":"q1","prediction":"b"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate query_id"):
        load_prediction_jsonl(path)


def test_cli_writes_context_packing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "baseline.jsonl"
    compressed = tmp_path / "compressed.jsonl"
    baseline.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "where was ada born",
                "passages": ["Ada Lovelace was born in London.", "Extra context."],
                "prediction": "London",
                "references": ["London"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    compressed.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "where was ada born",
                "passages": ["Ada Lovelace was born in London."],
                "prediction": "London",
                "references": ["London"],
                "context_packing": {"retained_doc_ids": ["d1"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs" / "context_packing"

    cli.main(
        [
            "--baseline-predictions",
            str(baseline),
            "--compressed-predictions",
            str(compressed),
            "--baseline-name",
            "plain",
            "--compressed-name",
            "packed",
            "--output-dir",
            str(output_dir),
        ]
    )

    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["inputs"] == {
        "baseline_predictions": "baseline.jsonl",
        "compressed_predictions": "compressed.jsonl",
    }
    assert comparison["metrics"]["packed"]["mean_token_f1"] == pytest.approx(1.0)
    assert (output_dir / "per_query.jsonl").exists()
    assert "Context packing comparison" in (output_dir / "report.md").read_text(
        encoding="utf-8"
    )

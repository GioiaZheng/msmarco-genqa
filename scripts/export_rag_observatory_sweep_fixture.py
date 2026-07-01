"""Build the small rag-observatory sweep fixture used by make reproduce-small."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msmarco_genqa.cli.rag_observatory_sweep_export import main


if __name__ == "__main__":
    main(
        [
            "--arm",
            "bm25=tests/fixtures/rag_observatory_export/predictions.jsonl",
            "--arm",
            "dense-rerank=tests/fixtures/rag_observatory_export/predictions_dense_rerank.jsonl",
            "--qrels",
            "tests/fixtures/rag_observatory_export/qrels.tsv",
            "--query-id",
            "msmarco-synthetic-q001",
            "--sweep-id",
            "synthetic-trace-sweep-001",
            "--timestamp",
            "2026-07-01T00:00:00Z",
            "--dataset",
            "synthetic-msmarco-genqa",
            "--code-version",
            "fixture",
            "--generator",
            "synthetic-generator",
            "--evaluator",
            "deterministic-rag-triad",
            "--random-seed",
            "17",
            "--output-dir",
            "outputs/reproduce_small/rag_observatory_sweep",
        ]
    )

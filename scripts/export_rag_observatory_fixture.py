"""Build the small rag-observatory export fixture used by make reproduce-small."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msmarco_genqa.cli.rag_observatory_export import main


if __name__ == "__main__":
    main(
        [
            "--predictions",
            "tests/fixtures/rag_observatory_export/predictions.jsonl",
            "--qrels",
            "tests/fixtures/rag_observatory_export/qrels.tsv",
            "--query-id",
            "msmarco-synthetic-q001",
            "--run-id",
            "synthetic-msmarco-genqa-run-001",
            "--timestamp",
            "2026-06-29T00:00:00Z",
            "--dataset",
            "synthetic-msmarco-genqa",
            "--config-hash",
            "synthetic-config-hash",
            "--code-version",
            "fixture",
            "--retriever",
            "synthetic-bm25",
            "--generator",
            "synthetic-generator",
            "--evaluator",
            "deterministic-rag-triad",
            "--random-seed",
            "17",
            "--export-profile",
            "reproduce-small",
            "--output",
            "outputs/reproduce_small/rag_observatory_export.json",
        ]
    )


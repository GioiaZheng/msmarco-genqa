"""Validate and cross-check a retrieval run with TREC-compatible tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msmarco_genqa.evaluation.trec import (
    MetricCrossCheckError,
    OptionalEvaluatorUnavailable,
    QrelsFormatError,
    run_trec_cross_check,
)
from msmarco_genqa.reranking.io import RunTsvFormatError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Six-column run.tsv file.")
    parser.add_argument("--qrels", type=Path, required=True, help="Qrels input file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for canonical run/qrels files and metrics.json.",
    )
    parser.add_argument(
        "--qrels-format",
        choices=("auto", "trec", "irds-tsv", "three-column"),
        default="auto",
        help="Qrels column layout. Use an explicit value for ambiguous numeric rows.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ir-measures", "none"),
        default="auto",
        help=(
            "Independent evaluator. 'auto' skips gracefully when ir-measures "
            "is unavailable; 'ir-measures' requires it."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="msmarco-genqa",
        help="Identifier written to the canonical run's final column.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Maximum absolute metric difference accepted per measure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_trec_cross_check(
            run_path=args.run,
            qrels_path=args.qrels,
            output_dir=args.output_dir,
            run_id=args.run_id,
            qrels_format=args.qrels_format,
            backend=args.backend,
            tolerance=args.tolerance,
        )
    except (
        MetricCrossCheckError,
        OptionalEvaluatorUnavailable,
        QrelsFormatError,
        RunTsvFormatError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

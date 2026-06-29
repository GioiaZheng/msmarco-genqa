"""``mgq-export-rag-observatory`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.interop.rag_observatory import (
    build_trace_export,
    load_prediction_rows,
    load_qrels,
    select_prediction_row,
    write_trace_export,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mgq-export-rag-observatory", description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--code-version", default=None)
    parser.add_argument("--retriever", default=None)
    parser.add_argument("--reranker", default=None)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--evaluator", default="deterministic-rag-triad")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--export-profile", default="single-query")
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predictions_path = _resolve(args.predictions)
    qrels_path = _resolve(args.qrels)
    manifest_path = _resolve(args.manifest)
    output_path = _resolve(args.output)
    assert predictions_path is not None
    assert output_path is not None

    rows = load_prediction_rows(predictions_path)
    row = select_prediction_row(rows, query_id=args.query_id)
    qrels = load_qrels(qrels_path)
    export = build_trace_export(
        row,
        run_id=args.run_id,
        timestamp=args.timestamp,
        dataset=args.dataset,
        config_hash=args.config_hash,
        code_version=args.code_version,
        retriever=args.retriever,
        reranker=args.reranker,
        generator=args.generator,
        evaluator=args.evaluator,
        random_seed=args.random_seed,
        qrels=qrels,
        manifest_path=_display(manifest_path),
        source_predictions=_display(predictions_path),
        source_qrels=_display(qrels_path),
        export_profile=args.export_profile,
        low_score_threshold=args.low_score_threshold,
    )
    written = write_trace_export(output_path, export)
    print(f"Wrote rag-observatory export: {written}")


if __name__ == "__main__":
    main()


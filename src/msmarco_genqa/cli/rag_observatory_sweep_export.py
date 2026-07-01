"""``mgq-export-rag-observatory-sweep`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.interop.rag_observatory import (
    build_sweep_manifest,
    build_trace_export,
    load_prediction_rows,
    load_qrels,
    select_prediction_row,
    write_sweep_manifest,
    write_trace_export,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mgq-export-rag-observatory-sweep",
        description=__doc__,
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="CONFIG_ID=PREDICTIONS",
        help="Configuration id and prediction JSONL path. Repeat for each arm.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--dataset", default="msmarco-genqa")
    parser.add_argument("--code-version", default=None)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--evaluator", default="deterministic-rag-triad")
    parser.add_argument("--random-seed", type=int, default=None)
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


def _parse_arm(raw: str) -> tuple[str, Path]:
    config_id, sep, path = raw.partition("=")
    if not sep or not config_id or not path:
        raise SystemExit("--arm must use CONFIG_ID=PREDICTIONS")
    return config_id, Path(path)


def _config_from_row(row: dict, *, config_id: str) -> dict:
    return {
        "config_id": config_id,
        "retriever": row.get("retrieval_source"),
        "top_k": len(row.get("top_doc_ids") or []),
        "reranking": bool(row.get("reranked_doc_ids")),
        "query_rewriting": bool(row.get("rewritten_query")),
        "context_compression": bool(row.get("context_packing")),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    qrels_path = _resolve(args.qrels)
    output_dir = _resolve(args.output_dir)
    assert output_dir is not None
    qrels = load_qrels(qrels_path)
    trace_exports = []
    trace_paths = []

    for raw_arm in args.arm:
        config_id, raw_predictions = _parse_arm(raw_arm)
        predictions_path = _resolve(raw_predictions)
        assert predictions_path is not None
        rows = load_prediction_rows(predictions_path)
        row = dict(select_prediction_row(rows, query_id=args.query_id))
        query_id = str(row["query_id"])
        config = _config_from_row(row, config_id=config_id)
        trace_relative = Path("traces") / config_id / f"{query_id}.json"
        trace_path = output_dir / trace_relative
        export = build_trace_export(
            row,
            run_id=f"{args.sweep_id}-{config_id}-{query_id}",
            timestamp=args.timestamp,
            dataset=args.dataset,
            code_version=args.code_version,
            retriever=str(row.get("retrieval_source") or config_id),
            reranker="present" if config["reranking"] else None,
            generator=args.generator,
            evaluator=args.evaluator,
            random_seed=args.random_seed,
            config_id=config_id,
            config=config,
            qrels=qrels,
            manifest_path=_display(output_dir / "rag_observatory_sweep.json"),
            source_predictions=_display(predictions_path),
            source_qrels=_display(qrels_path),
            export_profile="config-sweep",
            low_score_threshold=args.low_score_threshold,
        )
        write_trace_export(trace_path, export)
        trace_exports.append(export)
        trace_paths.append(trace_relative.as_posix())

    manifest = build_sweep_manifest(
        trace_exports,
        trace_paths=trace_paths,
        sweep_id=args.sweep_id,
        timestamp=args.timestamp,
        dataset=args.dataset,
    )
    manifest_path = write_sweep_manifest(output_dir / "rag_observatory_sweep.json", manifest)
    print(f"Wrote rag-observatory sweep manifest: {manifest_path}")
    for trace_path in trace_paths:
        print(output_dir / trace_path)


if __name__ == "__main__":
    main()

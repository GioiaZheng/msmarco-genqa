"""``mgq-rag-triad`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.evaluation.rag_triad import (
    build_triad_report,
    load_predictions_jsonl,
    write_triad_outputs,
)
from msmarco_genqa.evaluation.retrieval_report import load_qrels_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mgq-rag-triad", description=__doc__)
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Prediction JSONL for one retrieval/generation config. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help=(
            "Optional qrels TSV. When omitted, context relevance falls back "
            "to lexical query-context overlap."
        ),
    )
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--evaluator", default="deterministic")
    parser.add_argument("--context-top-k", type=int, default=None)
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    parser.add_argument("--max-low-score-cases", type=int, default=100)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    return (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"expected NAME=PATH for --predictions, got: {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"prediction config name is empty in: {value!r}")
    path = _resolve(Path(raw_path.strip()))
    return name, path


def _load_predictions(entries: list[str]) -> tuple[dict[str, list[dict]], dict[str, Path]]:
    out: dict[str, list[dict]] = {}
    paths: dict[str, Path] = {}
    for entry in entries:
        name, path = _parse_named_path(entry)
        if name in out:
            raise SystemExit(f"duplicate prediction config name: {name}")
        if not path.exists():
            raise SystemExit(f"prediction file not found: {path}")
        out[name] = load_predictions_jsonl(path)
        paths[name] = path
    return out, paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predictions, prediction_paths = _load_predictions(args.predictions)
    qrels = None
    qrels_path = None
    if args.qrels is not None:
        qrels_path = _resolve(args.qrels)
        if not qrels_path.exists():
            raise SystemExit(f"qrels file not found: {qrels_path}")
        qrels = load_qrels_tsv(qrels_path)

    report = build_triad_report(
        predictions,
        qrels=qrels,
        evaluator=args.evaluator,
        baseline_config=args.baseline_config,
        context_top_k=args.context_top_k,
        ngram_n=args.ngram_n,
        low_score_threshold=args.low_score_threshold,
        max_low_score_cases=args.max_low_score_cases,
    )
    report["summary"]["inputs"] = {
        "predictions": {
            name: _display_path(path)
            for name, path in prediction_paths.items()
        },
        "qrels": None if qrels_path is None else _display_path(qrels_path),
    }
    paths = write_triad_outputs(_resolve(args.output_dir), report)
    print("Wrote RAG triad evaluation:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()

"""``mgq-retrieval-report`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.evaluation.retrieval_report import (
    compare_runs_report,
    evaluate_run_report,
    load_qrels_tsv,
    read_run_doc_ids,
    render_comparison_markdown,
    render_single_run_markdown,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mgq-retrieval-report", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    evaluate = subcommands.add_parser("evaluate", help="Evaluate one retrieval run.")
    evaluate.add_argument("--run", type=Path, required=True, help="TREC-format run.tsv file.")
    evaluate.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help="Optional qrels TSV file. Defaults to MS MARCO passage dev/small via ir_datasets.",
    )
    evaluate.add_argument("--run-name", default="run")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    _add_metric_args(evaluate)

    compare = subcommands.add_parser("compare", help="Compare two runs on matched qids.")
    compare.add_argument("--baseline-run", type=Path, required=True)
    compare.add_argument("--candidate-run", type=Path, required=True)
    compare.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help="Optional qrels TSV file. Defaults to MS MARCO passage dev/small via ir_datasets.",
    )
    compare.add_argument("--baseline-name", default="baseline")
    compare.add_argument("--candidate-name", default="candidate")
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--k-rank", type=int, default=10)
    compare.add_argument("--k-recall", type=int, default=100)
    _add_metric_args(compare)

    return parser.parse_args(argv)


def _add_metric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ks-mrr", type=int, nargs="+", default=[10])
    parser.add_argument("--ks-ndcg", type=int, nargs="+", default=[10])
    parser.add_argument("--ks-recall", type=int, nargs="+", default=[100, 1000])


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    return (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def _load_qrels(path: Path | None):
    if path is None:
        from msmarco_genqa.data.msmarco import load_msmarco_passage

        bundle = load_msmarco_passage(load_corpus=False)
        return "msmarco-passage/dev/small via ir_datasets", bundle.qrels
    qrels_path = _resolve(path)
    if not qrels_path.exists():
        raise SystemExit(f"qrels file not found: {qrels_path}")
    return qrels_path, load_qrels_tsv(qrels_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    qrels_path, qrels = _load_qrels(args.qrels)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "evaluate":
        run_path = _resolve(args.run)
        if not run_path.exists():
            raise SystemExit(f"run file not found: {run_path}")
        report = evaluate_run_report(
            read_run_doc_ids(run_path),
            qrels,
            run_name=args.run_name,
            ks_mrr=tuple(args.ks_mrr),
            ks_ndcg=tuple(args.ks_ndcg),
            ks_recall=tuple(args.ks_recall),
        )
        report["inputs"] = {
            "run": _display_path(run_path),
            "qrels": _display_path(qrels_path) if isinstance(qrels_path, Path) else qrels_path,
        }
        write_json(output_dir / "metrics.json", report)
        (output_dir / "report.md").write_text(
            render_single_run_markdown(
                report,
                run_path=_display_path(run_path),
                qrels_path=_display_path(qrels_path) if isinstance(qrels_path, Path) else qrels_path,
            ),
            encoding="utf-8",
        )
        print(f"Wrote retrieval report to {output_dir}")
        return

    if args.command == "compare":
        baseline_path = _resolve(args.baseline_run)
        candidate_path = _resolve(args.candidate_run)
        for label, path in (("baseline", baseline_path), ("candidate", candidate_path)):
            if not path.exists():
                raise SystemExit(f"{label} run file not found: {path}")
        report = compare_runs_report(
            read_run_doc_ids(baseline_path),
            read_run_doc_ids(candidate_path),
            qrels,
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            k_rank=args.k_rank,
            k_recall=args.k_recall,
            ks_mrr=tuple(args.ks_mrr),
            ks_ndcg=tuple(args.ks_ndcg),
            ks_recall=tuple(args.ks_recall),
        )
        per_query = report.pop("per_query")
        report["inputs"] = {
            "baseline_run": _display_path(baseline_path),
            "candidate_run": _display_path(candidate_path),
            "qrels": _display_path(qrels_path) if isinstance(qrels_path, Path) else qrels_path,
        }
        write_json(output_dir / "comparison.json", report)
        write_jsonl(output_dir / "per_query.jsonl", per_query)
        (output_dir / "report.md").write_text(
            render_comparison_markdown(
                report,
                qrels_path=_display_path(qrels_path) if isinstance(qrels_path, Path) else qrels_path,
            ),
            encoding="utf-8",
        )
        print(f"Wrote retrieval comparison report to {output_dir}")
        return

    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()

"""``mgq-query-transform-ablation`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.evaluation.query_transform_ablation import (
    build_query_transform_ablation,
    extract_numeric_metrics,
    load_json_mapping,
    read_method_mapping,
    write_ablation_outputs,
)
from msmarco_genqa.util.sweep_summary import write_sweep_summary
from msmarco_genqa.util.tracking import ExperimentTracker


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Method summary spec in the form method=path/to/summary.json.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Optional method metrics spec in the form method=path/to/metrics.json.",
    )
    parser.add_argument("--baseline-method", default="none")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/query_transform/ablation",
    )
    parser.add_argument("--tracking-backend", default="jsonl", help="jsonl, mlflow, wandb, or none.")
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=None,
        help="Directory for per-method tracking events. Defaults to OUTPUT_DIR/tracking.",
    )
    parser.add_argument("--sweep-name", default="query-transform-ablation")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_summaries_from_mapping(mapping: dict[str, Path]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for method, path in mapping.items():
        resolved = _resolve(path)
        if not resolved.exists():
            raise SystemExit(f"summary file not found for {method}: {resolved}")
        summaries[method] = load_json_mapping(resolved)
    return summaries


def _load_metrics_from_mapping(mapping: dict[str, Path]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for method, path in mapping.items():
        resolved = _resolve(path)
        if not resolved.exists():
            raise SystemExit(f"metrics file not found for {method}: {resolved}")
        metrics[method] = extract_numeric_metrics(load_json_mapping(resolved))
    return metrics


def _write_tracking_events(
    *,
    backend: str,
    tracking_dir: Path,
    sweep_name: str,
    baseline_method: str,
    report: dict[str, object],
    summary_paths: dict[str, Path],
    metric_paths: dict[str, Path],
    output_paths: list[Path],
) -> list[Path]:
    if backend.lower() in {"none", "off"}:
        return []

    event_files: list[Path] = []
    for row in report["runs"]:
        method = str(row["method"])
        run_dir = tracking_dir / method
        tracker = ExperimentTracker(
            backend=backend,
            output_dir=run_dir,
            run_name=f"{sweep_name}-{method}",
            tags={
                "sweep": sweep_name,
                "arm": method,
                "component": "query_transform",
                "baseline_method": baseline_method,
            },
        )
        with tracker:
            tracker.log_params(
                {
                    "method": method,
                    "config_hash": row["config_hash"],
                    "n_queries": row["n_queries"],
                    "n_changed": row["n_changed"],
                    "changed_fraction": row["changed_fraction"],
                    "cache_hit": row["cache_hit"],
                }
            )
            if row["metrics"]:
                tracker.log_metrics(row["metrics"])
            tracker.log_artifact(_resolve(summary_paths[method]), name="query_summary")
            if method in metric_paths:
                tracker.log_artifact(_resolve(metric_paths[method]), name="retrieval_metrics")
            for output_path in output_paths:
                tracker.log_artifact(output_path, name=output_path.name)
        event_files.append(run_dir / "events.jsonl")

    write_sweep_summary(event_files, tracking_dir / "summary", sweep_name=sweep_name)
    return event_files


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        summary_paths = read_method_mapping(args.summary, value_name="summary")
        metric_paths = read_method_mapping(args.metrics, value_name="metrics")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summaries = _load_summaries_from_mapping(summary_paths)
    metrics = _load_metrics_from_mapping(metric_paths)
    try:
        report = build_query_transform_ablation(
            summaries,
            metrics=metrics,
            baseline_method=args.baseline_method,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = _resolve(args.output_dir)
    output_paths = write_ablation_outputs(output_dir, report)
    tracking_dir = _resolve(args.tracking_dir) if args.tracking_dir else output_dir / "tracking"
    event_files = _write_tracking_events(
        backend=args.tracking_backend,
        tracking_dir=tracking_dir,
        sweep_name=args.sweep_name,
        baseline_method=args.baseline_method,
        report=report,
        summary_paths=summary_paths,
        metric_paths=metric_paths,
        output_paths=output_paths,
    )
    print(
        "query transformation ablation: "
        f"{len(report['methods'])} methods, output={output_dir}"
    )
    if event_files:
        print(f"tracked sweep: {len(event_files)} runs, tracking={tracking_dir}")


if __name__ == "__main__":
    main()

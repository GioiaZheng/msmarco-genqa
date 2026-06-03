"""Console entry point for the RAG evaluation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.rag_eval import (
    PROJECT_ROOT,
    build_rag_eval_plan,
    filter_rag_eval_plan,
    format_rag_eval_plan,
    load_rag_eval_config,
    run_rag_eval_plan,
)
from msmarco_genqa.util.tracking import ExperimentTracker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Run or inspect the configured evaluation workflow.")
    run.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    run.add_argument("--dry-run", action="store_true", help="Print and track the plan only.")
    run.add_argument("--only", nargs="*", default=[], help="Run only the named stage(s).")
    run.add_argument("--skip", nargs="*", default=[], help="Skip the named stage(s).")
    run.add_argument("--tracking-backend", default=None, help="jsonl, mlflow, wandb, or none.")
    run.add_argument("--tracking-dir", type=Path, default=None)
    run.add_argument(
        "--python",
        default="python",
        help="Python executable used for script-based analysis stages.",
    )
    return parser.parse_args(argv)


def _tracking_settings(cfg: dict) -> dict:
    rag_eval = cfg.get("rag_eval") or {}
    tracking = rag_eval.get("tracking") or cfg.get("tracking") or {}
    return tracking if isinstance(tracking, dict) else {}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command != "run":
        raise SystemExit(f"unsupported command: {args.command}")

    cfg = load_rag_eval_config(args.config)
    plan = build_rag_eval_plan(cfg, config_path=args.config, python=args.python)
    plan = filter_rag_eval_plan(plan, only=set(args.only), skip=set(args.skip))
    print(format_rag_eval_plan(plan))

    tracking = _tracking_settings(cfg)
    tracker = ExperimentTracker(
        backend=args.tracking_backend or tracking.get("backend", "jsonl"),
        output_dir=args.tracking_dir or PROJECT_ROOT / tracking.get("output_dir", "outputs/rag_eval_tracking"),
        run_name=f"rag-eval-{args.config.stem}",
        tags={"config": str(args.config), "command": "run"},
    )
    with tracker:
        run_rag_eval_plan(plan, cwd=PROJECT_ROOT, tracker=tracker, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

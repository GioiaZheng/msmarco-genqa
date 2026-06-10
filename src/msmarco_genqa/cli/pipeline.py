"""Console entry point for the config-driven pipeline runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.pipeline import build_stage_plan, load_pipeline_config, run_stage_plan
from msmarco_genqa.util.tracking import ExperimentTracker

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pipeline.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print and track the plan only.")
    p.add_argument("--only", nargs="*", default=[], help="Run only named stages.")
    p.add_argument("--skip", nargs="*", default=[], help="Skip named stages.")
    p.add_argument("--tracking-backend", default=None, help="jsonl, mlflow, wandb, or none.")
    p.add_argument("--tracking-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_pipeline_config(args.config)
    plan = build_stage_plan(cfg, only=set(args.only), skip=set(args.skip))
    for stage in plan:
        print(f"{stage.name}: {' '.join(stage.command)}")
    tracking_cfg = cfg.get("tracking", {})
    tracker = ExperimentTracker(
        backend=args.tracking_backend or tracking_cfg.get("backend", "jsonl"),
        output_dir=args.tracking_dir or PROJECT_ROOT / tracking_cfg.get("output_dir", "outputs/tracking"),
        run_name=args.config.stem,
        tags={"config": str(args.config)},
    )
    with tracker:
        run_stage_plan(plan, cwd=PROJECT_ROOT, tracker=tracker, dry_run=args.dry_run)

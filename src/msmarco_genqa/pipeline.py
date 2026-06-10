"""Config-driven experiment pipeline planning."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from msmarco_genqa.util.tracking import ExperimentTracker


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    description: str = ""


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg.get("stages"), list):
        raise ValueError("pipeline config must define a list under 'stages'")
    return cfg


def build_stage_plan(
    cfg: dict[str, Any],
    *,
    only: set[str] | None = None,
    skip: set[str] | None = None,
) -> list[Stage]:
    only = only or set()
    skip = skip or set()
    plan: list[Stage] = []
    for raw in cfg["stages"]:
        name = raw["name"]
        if only and name not in only:
            continue
        if name in skip:
            continue
        command = raw.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ValueError(f"stage {name!r} must define command as a list of strings")
        plan.append(Stage(name=name, command=command, description=raw.get("description", "")))
    return plan


def run_stage_plan(
    plan: list[Stage],
    *,
    cwd: str | Path,
    tracker: ExperimentTracker,
    dry_run: bool = False,
) -> None:
    tracker.log_params({"stages": [s.name for s in plan], "dry_run": dry_run})
    for index, stage in enumerate(plan, start=1):
        tracker.log_params({f"stage.{index}.name": stage.name, f"stage.{index}.cmd": stage.command})
        if dry_run:
            continue
        t0 = time.time()
        subprocess.run(stage.command, cwd=cwd, check=True)
        tracker.log_metrics({f"{stage.name}.wall_seconds": time.time() - t0}, step=index)

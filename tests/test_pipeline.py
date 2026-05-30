from __future__ import annotations

from pathlib import Path

import pytest

from msmarco_genqa.pipeline import build_stage_plan, load_pipeline_config, run_stage_plan
from msmarco_genqa.util.tracking import ExperimentTracker


def test_build_stage_plan_filters_only_and_skip():
    cfg = {
        "stages": [
            {"name": "a", "command": ["python", "a.py"]},
            {"name": "b", "command": ["python", "b.py"]},
        ]
    }
    assert [s.name for s in build_stage_plan(cfg, only={"b"})] == ["b"]
    assert [s.name for s in build_stage_plan(cfg, skip={"a"})] == ["b"]


def test_build_stage_plan_rejects_shell_string():
    cfg = {"stages": [{"name": "bad", "command": "python bad.py"}]}
    with pytest.raises(ValueError, match="list of strings"):
        build_stage_plan(cfg)


def test_pipeline_dry_run_tracks_plan_without_executing(tmp_path):
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(
        "stages:\n"
        "  - name: sample\n"
        "    command: [python, missing.py]\n",
        encoding="utf-8",
    )
    cfg = load_pipeline_config(cfg_path)
    tracker = ExperimentTracker(backend="jsonl", output_dir=tmp_path / "tracking")
    run_stage_plan(build_stage_plan(cfg), cwd=Path.cwd(), tracker=tracker, dry_run=True)
    assert (tmp_path / "tracking" / "events.jsonl").exists()

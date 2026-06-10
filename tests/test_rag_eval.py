from __future__ import annotations

from pathlib import Path

import pytest

from msmarco_genqa.cli.rag_eval import main as rag_eval_main
from msmarco_genqa.rag_eval import (
    build_rag_eval_plan,
    filter_rag_eval_plan,
    format_rag_eval_plan,
    load_rag_eval_config,
)


def _write_config(path: Path) -> Path:
    path.write_text(
        """
seed: 42
eval_retrieval:
  output_dir: outputs/bm25
dense:
  output_dir: outputs/dense
reranker:
  output_dir: outputs/reranked
generation:
  output_dir: outputs/generation
  num_eval_queries: 12
rag_eval:
  stages:
    - generation_bm25
    - generation_reranked
    - paired_bootstrap_ci
  num_eval_queries: 99
  bm25_generation_dir: outputs/gen_bm25
  reranked_generation_dir: outputs/gen_reranked
  bootstrap_output_dir: outputs/bootstrap
  tracking:
    backend: none
""",
        encoding="utf-8",
    )
    return path


def test_build_rag_eval_plan_pairs_generation_runs(tmp_path):
    config_path = _write_config(tmp_path / "baseline.yaml")
    cfg = load_rag_eval_config(config_path)

    plan = build_rag_eval_plan(cfg, config_path=config_path, python="python")

    assert [stage.name for stage in plan] == [
        "generation_bm25",
        "generation_reranked",
        "paired_bootstrap_ci",
    ]
    bm25_cmd = plan[0].command
    reranked_cmd = plan[1].command
    assert bm25_cmd[bm25_cmd.index("--input-run") + 1] == "outputs/bm25/run.tsv"
    assert bm25_cmd[bm25_cmd.index("--restrict-to-run") + 1] == "outputs/reranked/run.tsv"
    assert reranked_cmd[reranked_cmd.index("--input-run") + 1] == "outputs/reranked/run.tsv"
    assert reranked_cmd[reranked_cmd.index("--restrict-to-run") + 1] == "outputs/bm25/run.tsv"
    assert "--n-resamples" in plan[2].command


def test_filter_and_format_rag_eval_plan(tmp_path):
    config_path = _write_config(tmp_path / "baseline.yaml")
    cfg = load_rag_eval_config(config_path)
    plan = build_rag_eval_plan(cfg, config_path=config_path)

    filtered = filter_rag_eval_plan(plan, only={"paired_bootstrap_ci"})

    assert [stage.name for stage in filtered] == ["paired_bootstrap_ci"]
    rendered = format_rag_eval_plan(filtered)
    assert "paired_bootstrap_ci" in rendered
    assert "outputs/bootstrap/bootstrap_ci.json" in rendered


def test_build_rag_eval_plan_rejects_unknown_stage(tmp_path):
    config_path = _write_config(tmp_path / "baseline.yaml")
    cfg = load_rag_eval_config(config_path)
    cfg["rag_eval"]["stages"] = ["missing_stage"]

    with pytest.raises(ValueError, match="unknown RAG evaluation stage"):
        build_rag_eval_plan(cfg, config_path=config_path)


def test_rag_eval_cli_dry_run_prints_plan_without_executing(tmp_path, capsys):
    config_path = _write_config(tmp_path / "baseline.yaml")

    rag_eval_main(
        [
            "run",
            "--config",
            str(config_path),
            "--dry-run",
            "--tracking-backend",
            "none",
        ]
    )

    out = capsys.readouterr().out
    assert "generation_bm25" in out
    assert "paired_bootstrap_ci" in out

"""Research-oriented RAG evaluation plan builder."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from msmarco_genqa.util.tracking import ExperimentTracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_STAGE_ORDER: tuple[str, ...] = (
    "query_transformation",
    "bm25_retrieval",
    "dense_retrieval",
    "cross_encoder_rerank",
    "retrieval_quality_report",
    "retrieval_lift_analysis",
    "generation_bm25",
    "generation_reranked",
    "paired_bootstrap_ci",
    "grounding_audit",
)


@dataclass(frozen=True)
class RAGEvalStage:
    name: str
    command: list[str]
    description: str
    expected_outputs: list[str]


def load_rag_eval_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("RAG evaluation config must be a YAML mapping")
    required_sections = ["eval_retrieval", "dense", "reranker", "generation"]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"RAG evaluation config is missing section(s): {missing}")
    return cfg


def _as_posix(value: str | Path) -> str:
    return Path(value).as_posix()


def _cfg_path(cfg: dict[str, Any], section: str, key: str) -> str:
    try:
        value = cfg[section][key]
    except KeyError as exc:
        raise ValueError(f"RAG evaluation config is missing {section}.{key}") from exc
    return _as_posix(value)


def _default_named_dir(base_dir: str, suffix: str) -> str:
    base = Path(base_dir)
    return _as_posix(base.with_name(f"{base.name}_{suffix}"))


def _rag_eval_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("rag_eval", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("rag_eval must be a YAML mapping when present")
    return raw


def _build_all_stages(
    *,
    config_path: str | Path,
    cfg: dict[str, Any],
    python: str,
) -> dict[str, RAGEvalStage]:
    settings = _rag_eval_settings(cfg)
    config_arg = _as_posix(config_path)

    bm25_dir = _cfg_path(cfg, "eval_retrieval", "output_dir")
    dense_dir = _cfg_path(cfg, "dense", "output_dir")
    reranker_dir = _as_posix(settings.get("reranker_output_dir", _cfg_path(cfg, "reranker", "output_dir")))
    generation_dir = _cfg_path(cfg, "generation", "output_dir")

    bm25_run = _as_posix(Path(bm25_dir) / "run.tsv")
    dense_run = _as_posix(Path(dense_dir) / "run.tsv")
    reranker_run = _as_posix(Path(reranker_dir) / "run.tsv")
    query_transform_settings = cfg.get("query_transform", {})
    query_transform_output_dir = "outputs/query_transform/none"
    if isinstance(query_transform_settings, dict):
        query_transform_output_dir = str(
            query_transform_settings.get("output_dir", query_transform_output_dir)
        )
    query_transform_dir = _as_posix(
        settings.get("query_transform_output_dir", query_transform_output_dir)
    )

    bm25_generation_dir = _as_posix(
        settings.get("bm25_generation_dir", _default_named_dir(generation_dir, "bm25_full"))
    )
    reranked_generation_dir = _as_posix(
        settings.get("reranked_generation_dir", _default_named_dir(generation_dir, "reranked_full"))
    )
    bootstrap_dir = _as_posix(
        settings.get("bootstrap_output_dir", _default_named_dir(generation_dir, "bootstrap_full"))
    )
    retrieval_lift_dir = _as_posix(
        settings.get("retrieval_lift_output_dir", "outputs/week05_retrieval_lift_analysis")
    )
    retrieval_report_dir = _as_posix(
        settings.get("retrieval_report_output_dir", "outputs/retrieval_reports/dense_vs_reranked")
    )
    retrieval_qrels_path = settings.get("retrieval_qrels_path")
    grounding_dir = _as_posix(settings.get("grounding_output_dir", "outputs/week07_grounding"))
    num_eval_queries = str(settings.get("num_eval_queries", cfg["generation"].get("num_eval_queries", 200)))
    n_resamples = str(settings.get("bootstrap_resamples", 10000))
    grounding_nli_pairs = str(settings.get("grounding_nli_pairs", 0))

    rerank_command = [
        "mgq-rerank",
        "--config",
        config_arg,
        "--output-dir",
        reranker_dir,
    ]
    if settings.get("reranker_resume", True):
        rerank_command.append("--resume")

    retrieval_report_command = [
        "mgq-retrieval-report",
        "compare",
        "--baseline-run",
        dense_run,
        "--candidate-run",
        reranker_run,
        "--baseline-name",
        "dense",
        "--candidate-name",
        "reranked",
        "--output-dir",
        retrieval_report_dir,
    ]
    if retrieval_qrels_path:
        retrieval_report_command.extend(["--qrels", _as_posix(retrieval_qrels_path)])

    return {
        "query_transformation": RAGEvalStage(
            name="query_transformation",
            description="Deterministic query transformation audit artifacts before retrieval.",
            command=[
                "mgq-transform-queries",
                "--config",
                config_arg,
                "--output-dir",
                query_transform_dir,
            ],
            expected_outputs=[
                _as_posix(Path(query_transform_dir) / "queries.jsonl"),
                _as_posix(Path(query_transform_dir) / "summary.json"),
            ],
        ),
        "bm25_retrieval": RAGEvalStage(
            name="bm25_retrieval",
            description="Full-corpus BM25 retrieval on MS MARCO dev/small.",
            command=["mgq-retrieve", "--config", config_arg],
            expected_outputs=[_as_posix(Path(bm25_dir) / "run.tsv"), _as_posix(Path(bm25_dir) / "metrics.json")],
        ),
        "dense_retrieval": RAGEvalStage(
            name="dense_retrieval",
            description="Dense retrieval on the qrels-anchored sample, with BM25-on-sample comparison.",
            command=["mgq-dense", "--config", config_arg],
            expected_outputs=[_as_posix(Path(dense_dir) / "run.tsv"), _as_posix(Path(dense_dir) / "metrics.json")],
        ),
        "cross_encoder_rerank": RAGEvalStage(
            name="cross_encoder_rerank",
            description="Cross-encoder reranking over dense top-k candidates.",
            command=rerank_command,
            expected_outputs=[
                _as_posix(Path(reranker_dir) / "run.tsv"),
                _as_posix(Path(reranker_dir) / "metrics.json"),
            ],
        ),
        "retrieval_quality_report": RAGEvalStage(
            name="retrieval_quality_report",
            description="Matched-qid retrieval metric report for dense vs reranked runs.",
            command=retrieval_report_command,
            expected_outputs=[
                _as_posix(Path(retrieval_report_dir) / "comparison.json"),
                _as_posix(Path(retrieval_report_dir) / "per_query.jsonl"),
                _as_posix(Path(retrieval_report_dir) / "report.md"),
            ],
        ),
        "retrieval_lift_analysis": RAGEvalStage(
            name="retrieval_lift_analysis",
            description="Query-level retrieval gain/loss buckets after reranking.",
            command=[
                python,
                "scripts/analyze_retrieval_lift.py",
                "--before-run",
                dense_run,
                "--after-run",
                reranker_run,
                "--output-dir",
                retrieval_lift_dir,
            ],
            expected_outputs=[_as_posix(Path(retrieval_lift_dir) / "retrieval_lift.json")],
        ),
        "generation_bm25": RAGEvalStage(
            name="generation_bm25",
            description="Generation from BM25 top passages on the paired evaluation qid set.",
            command=[
                "mgq-generate",
                "--config",
                config_arg,
                "--input-run",
                bm25_run,
                "--output-dir",
                bm25_generation_dir,
                "--retrieval-source",
                "bm25",
                "--restrict-to-run",
                reranker_run,
                "--num-eval-queries",
                num_eval_queries,
            ],
            expected_outputs=[
                _as_posix(Path(bm25_generation_dir) / "predictions.jsonl"),
                _as_posix(Path(bm25_generation_dir) / "metrics.json"),
            ],
        ),
        "generation_reranked": RAGEvalStage(
            name="generation_reranked",
            description="Generation from reranked top passages on the same paired qid set.",
            command=[
                "mgq-generate",
                "--config",
                config_arg,
                "--input-run",
                reranker_run,
                "--output-dir",
                reranked_generation_dir,
                "--retrieval-source",
                "reranked",
                "--restrict-to-run",
                bm25_run,
                "--num-eval-queries",
                num_eval_queries,
            ],
            expected_outputs=[
                _as_posix(Path(reranked_generation_dir) / "predictions.jsonl"),
                _as_posix(Path(reranked_generation_dir) / "metrics.json"),
            ],
        ),
        "paired_bootstrap_ci": RAGEvalStage(
            name="paired_bootstrap_ci",
            description="Paired-bootstrap confidence intervals for reranked minus BM25 generation.",
            command=[
                python,
                "scripts/bootstrap_generation_comparison.py",
                "--bm25-dir",
                bm25_generation_dir,
                "--reranked-dir",
                reranked_generation_dir,
                "--output-dir",
                bootstrap_dir,
                "--n-resamples",
                n_resamples,
                "--seed",
                str(cfg.get("seed", 42)),
            ],
            expected_outputs=[_as_posix(Path(bootstrap_dir) / "bootstrap_ci.json")],
        ),
        "grounding_audit": RAGEvalStage(
            name="grounding_audit",
            description="Grounding metrics over the paired BM25 and reranked generation outputs.",
            command=[
                python,
                "scripts/grounding_audit.py",
                "--bm25-dir",
                bm25_generation_dir,
                "--reranked-dir",
                reranked_generation_dir,
                "--output-dir",
                grounding_dir,
                "--nli-n-pairs",
                grounding_nli_pairs,
            ],
            expected_outputs=[
                _as_posix(Path(grounding_dir) / "summary.json"),
                _as_posix(Path(grounding_dir) / "per_query_grounding.jsonl"),
            ],
        ),
    }


def build_rag_eval_plan(
    cfg: dict[str, Any],
    *,
    config_path: str | Path = "configs/baseline.yaml",
    python: str = "python",
) -> list[RAGEvalStage]:
    settings = _rag_eval_settings(cfg)
    requested = settings.get("stages", list(DEFAULT_STAGE_ORDER))
    if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
        raise ValueError("rag_eval.stages must be a list of stage names")

    all_stages = _build_all_stages(config_path=config_path, cfg=cfg, python=python)
    unknown = [name for name in requested if name not in all_stages]
    if unknown:
        raise ValueError(f"unknown RAG evaluation stage(s): {unknown}")
    return [all_stages[name] for name in requested]


def filter_rag_eval_plan(
    plan: list[RAGEvalStage],
    *,
    only: set[str] | None = None,
    skip: set[str] | None = None,
) -> list[RAGEvalStage]:
    only = only or set()
    skip = skip or set()
    return [stage for stage in plan if (not only or stage.name in only) and stage.name not in skip]


def format_rag_eval_plan(plan: list[RAGEvalStage]) -> str:
    lines: list[str] = []
    for index, stage in enumerate(plan, start=1):
        lines.append(f"{index:02d}. {stage.name}: {' '.join(stage.command)}")
        if stage.description:
            lines.append(f"    {stage.description}")
        if stage.expected_outputs:
            lines.append(f"    outputs: {', '.join(stage.expected_outputs)}")
    return "\n".join(lines)


def run_rag_eval_plan(
    plan: list[RAGEvalStage],
    *,
    cwd: str | Path = PROJECT_ROOT,
    tracker: ExperimentTracker,
    dry_run: bool = False,
) -> None:
    tracker.log_params({"stages": [stage.name for stage in plan], "dry_run": dry_run})
    for index, stage in enumerate(plan, start=1):
        tracker.log_params(
            {
                f"stage.{index}.name": stage.name,
                f"stage.{index}.cmd": stage.command,
                f"stage.{index}.expected_outputs": stage.expected_outputs,
            }
        )
        if dry_run:
            continue
        started_at = time.time()
        subprocess.run(stage.command, cwd=cwd, check=True)
        tracker.log_metrics({f"{stage.name}.wall_seconds": time.time() - started_at}, step=index)

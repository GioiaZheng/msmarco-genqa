"""Fuse retrieval runs with weighted Reciprocal Rank Fusion.

The runner consumes two or more standard TREC-format ``run.tsv`` files,
preserves per-source provenance in ``provenance.jsonl``, writes a fused
``run.tsv``, and records metrics/provenance in ``metrics.json`` and
``manifest.json``.

Example:

    python experiments/run_hybrid_fusion.py \
        --input-run bm25_sample=outputs/W4_dense/run_bm25_sample.tsv \
        --input-run dense=outputs/W4_dense/run.tsv \
        --output-dir outputs/W4_hybrid_rrf \
        --top-k 1000 \
        --qrels data/qrels.dev.small.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.reranking.io import read_run_tsv, write_run_tsv
from msmarco_genqa.retrieval.fusion import fused_doc_ids_and_scores, reciprocal_rank_fusion
from msmarco_genqa.util.environment import capture_environment
from msmarco_genqa.util.manifest import (
    compute_data_fingerprint,
    compute_env_fingerprint,
    compute_resolved_config_hash,
    write_resolved_config,
    write_run_manifest,
)
from msmarco_genqa.util.seeding import set_global_seed

logger = logging.getLogger("run_hybrid_fusion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Named input run. Pass once per source, for example "
            "bm25=outputs/W4_dense/run_bm25_sample.tsv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/W4_hybrid_rrf",
        help="Directory for run.tsv, provenance.jsonl, metrics.json, and manifest.json.",
    )
    parser.add_argument(
        "--rank-constant",
        type=float,
        default=60.0,
        help="RRF rank constant k. Larger values reduce early-rank dominance.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=FLOAT",
        help="Optional source weight. Omitted sources default to 1.0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Maximum fused candidates to keep per query.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help=(
            "Optional qrels TSV for retrieval metrics. Accepts either "
            "'qid docid rel' or 'qid iter docid rel' whitespace-separated rows."
        ),
    )
    parser.add_argument(
        "--system-name",
        default="rrf",
        help="System name written in the sixth TREC run column.",
    )
    parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Refuse to write the manifest if the git tree is dirty.",
    )
    parser.add_argument(
        "--allow-incomplete-manifest",
        action="store_true",
        help="Bypass strict manifest required-field validation during development.",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_named_path(value: str, option_name: str) -> tuple[str, Path]:
    name, sep, raw_path = value.partition("=")
    if not sep or not name or not raw_path:
        raise SystemExit(f"{option_name} must use NAME=PATH, got {value!r}")
    return name, _resolve_path(Path(raw_path))


def _parse_weights(values: list[str], source_names: set[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw in values:
        name, sep, raw_value = raw.partition("=")
        if not sep or not name or not raw_value:
            raise SystemExit(f"--weight must use NAME=FLOAT, got {raw!r}")
        if name not in source_names:
            raise SystemExit(f"--weight names unknown source {name!r}")
        try:
            weight = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"--weight for {name!r} is not numeric: {raw_value!r}") from exc
        if weight < 0:
            raise SystemExit(f"--weight for {name!r} must be non-negative")
        weights[name] = weight
    return weights


def _path_for_payload(path: Path) -> str:
    return (
        str(path.relative_to(PROJECT_ROOT).as_posix())
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 3:
                qid, doc_id, rel_text = parts
            elif len(parts) >= 4:
                qid, _iter, doc_id, rel_text = parts[:4]
            else:
                raise SystemExit(
                    f"{path}:{line_number}: expected 3 or 4+ qrels columns, got {len(parts)}"
                )
            try:
                rel = float(rel_text)
            except ValueError as exc:
                raise SystemExit(
                    f"{path}:{line_number}: relevance is not numeric: {rel_text!r}"
                ) from exc
            if rel > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def _write_provenance(path: Path, fused) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(fused):
            for rank, hit in enumerate(fused[qid], start=1):
                row = {
                    "qid": qid,
                    "doc_id": hit.doc_id,
                    "rank": rank,
                    "rrf_score": hit.score,
                    "sources": {
                        name: {
                            "rank": source.rank,
                            "score": source.score,
                            "weight": source.weight,
                            "contribution": source.contribution,
                        }
                        for name, source in sorted(hit.sources.items())
                    },
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")


def _source_coverage(runs_by_source) -> dict[str, dict[str, int]]:
    return {
        name: {
            "n_queries": len(runs),
            "n_rows": sum(len(rows) for rows in runs.values()),
            "n_unique_docs": len({doc_id for rows in runs.values() for doc_id, _ in rows}),
        }
        for name, runs in sorted(runs_by_source.items())
    }


def _overlap_stats(runs_by_source) -> dict[str, int]:
    qid_sets = [set(runs) for runs in runs_by_source.values()]
    if not qid_sets:
        return {"qids_union": 0, "qids_shared_all": 0}
    return {
        "qids_union": len(set().union(*qid_sets)),
        "qids_shared_all": len(set.intersection(*qid_sets)),
    }


def _build_resolved_config(args: argparse.Namespace, input_runs: dict[str, Path], weights) -> dict[str, Any]:
    return {
        "task": "hybrid_fusion",
        "input_runs": {name: _path_for_payload(path) for name, path in sorted(input_runs.items())},
        "rank_constant": args.rank_constant,
        "weights": {name: float(weights.get(name, 1.0)) for name in sorted(input_runs)},
        "top_k": args.top_k,
        "qrels": _path_for_payload(_resolve_path(args.qrels)) if args.qrels else None,
        "system_name": args.system_name,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    input_runs: dict[str, Path] = {}
    for raw in args.input_run:
        name, path = _parse_named_path(raw, "--input-run")
        if name in input_runs:
            raise SystemExit(f"duplicate --input-run source name {name!r}")
        if not path.exists():
            raise SystemExit(f"input run not found: {path}")
        input_runs[name] = path
    if len(input_runs) < 2:
        raise SystemExit("RRF fusion requires at least two --input-run sources")

    weights = _parse_weights(args.weight, set(input_runs))
    qrels_path = _resolve_path(args.qrels) if args.qrels else None
    if qrels_path is not None and not qrels_path.exists():
        raise SystemExit(f"qrels file not found: {qrels_path}")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    seed_coverage = set_global_seed(seed)
    runs_by_source = {
        name: read_run_tsv(path)
        for name, path in sorted(input_runs.items())
    }
    logger.info(
        "Fusing %d source runs over %d qids.",
        len(runs_by_source),
        _overlap_stats(runs_by_source)["qids_union"],
    )
    fused = reciprocal_rank_fusion(
        runs_by_source,
        rank_constant=args.rank_constant,
        weights=weights,
        top_k=args.top_k,
    )

    qids, doc_ids, scores = fused_doc_ids_and_scores(fused)
    run_path = output_dir / "run.tsv"
    write_run_tsv(run_path, qids, doc_ids, scores, args.system_name)

    provenance_path = output_dir / "provenance.jsonl"
    _write_provenance(provenance_path, fused)

    metrics = {}
    n_examples = None
    if qrels_path is not None:
        qrels = _load_qrels(qrels_path)
        runs_for_eval = {qid: [hit.doc_id for hit in hits] for qid, hits in fused.items()}
        metrics = evaluate_retrieval(runs_for_eval, qrels)
        n_examples = metrics.pop("n_queries", None)

    env_dict = capture_environment()
    resolved_config = _build_resolved_config(args, input_runs, weights)
    payload = {
        "task": "hybrid_fusion",
        "dataset": "msmarco-passage/dev/small" if qrels_path else "run-fusion",
        "n_examples": n_examples,
        "fusion": {
            "method": "weighted_rrf",
            "rank_constant": args.rank_constant,
            "top_k": args.top_k,
            "weights": resolved_config["weights"],
            "input_runs": resolved_config["input_runs"],
        },
        "source_coverage": _source_coverage(runs_by_source),
        "overlap": _overlap_stats(runs_by_source),
        "metrics": {"rrf": metrics} if metrics else {},
        "environment": env_dict,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    logger.info("Wrote fused run and metrics to %s", output_dir)

    resolved_config_path = write_resolved_config(resolved_config, output_dir)
    resolved_config_hash = compute_resolved_config_hash(resolved_config)
    extra_files = {f"input_run.{name}": path for name, path in input_runs.items()}
    if qrels_path is not None:
        extra_files["qrels"] = qrels_path
    data_fingerprint = compute_data_fingerprint(
        cache_dir=PROJECT_ROOT / "data/raw",
        extra_files=extra_files,
    )
    env_fingerprint = compute_env_fingerprint(env_dict)

    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv if argv is None else ["mgq-fuse", *argv],
        config_path=None,
        extra_outputs=[run_path, provenance_path, resolved_config_path],
        extra={
            "task": "hybrid_fusion",
            "method": "weighted_rrf",
            "rank_constant": args.rank_constant,
            "top_k": args.top_k,
            "input_runs": resolved_config["input_runs"],
            "weights": resolved_config["weights"],
            "n_eval_queries": n_examples,
            "seed": seed,
            "seed_coverage": seed_coverage,
            "resolved_config_hash": resolved_config_hash,
            "data_fingerprint": data_fingerprint,
            "env_fingerprint": env_fingerprint,
        },
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
    )

    print("\n=== Hybrid retrieval fusion (RRF) ===")
    print(f"sources: {', '.join(sorted(input_runs))}")
    print(f"qids:    {len(fused)} union")
    print(f"top-K:   {args.top_k}")
    if metrics:
        for key in ("mrr@10", "ndcg@10", "recall@100", "recall@1000"):
            if key in metrics:
                print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

"""End-to-end Week 3 RAG generation baseline.

Pipeline:

1. Load a TREC-format retrieval run (``--input-run``; defaults to the W2
   BM25 run at ``outputs/week02_bm25/run.tsv``).
2. Load dev/small queries and the MS MARCO Passage docs_store (random access).
3. Cross-reference dev/small query ids with MS MARCO QA v2.1 (HuggingFace
   ``ms_marco`` dataset, validation split) to recover human-written answers
   for evaluation.
4. For each evaluated query, take the top-K passages from the run, generate
   an answer with the Seq2Seq model, and score predictions.
5. Persist (under ``--output-dir``, defaults to ``outputs/week03_generation``):
   - ``predictions.jsonl``
   - ``metrics.json``
   - ``examples.jsonl``
   - ``manifest.json``

The runner is **retrieval-source agnostic** — point ``--input-run`` at any
TREC-format ``run.tsv`` (BM25 / dense / reranked) and the rest of the
pipeline is identical. Use ``--restrict-to-run`` to make different
retrieval sources eval on the SAME query subsample (apples-to-apples),
which matters when one source covers fewer queries than another (e.g.
the W5 reranker covers 1,000 dev queries, not all 6,980).

Run from the project root::

    # W3 baseline: BM25 → T5-small (defaults preserve the legacy behaviour)
    python experiments/run_generation_baseline.py

    # Reranked → T5-small, restricted to reranker-covered queries
    python experiments/run_generation_baseline.py \\
        --input-run outputs/week05_reranker/run.tsv \\
        --output-dir outputs/week03_generation_reranked \\
        --retrieval-source reranked
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msmarco import get_docs_store, load_msmarco_passage  # noqa: E402
from src.evaluation.generation import evaluate_generation  # noqa: E402
from src.generation.rag_generator import RAGGenerationConfig, RAGGenerator  # noqa: E402
from src.util.environment import capture_environment  # noqa: E402
from src.util.manifest import write_run_manifest  # noqa: E402
from src.util.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/baseline.yaml",
    )
    parser.add_argument(
        "--num-eval-queries",
        type=int,
        default=None,
        help="Override the eval set size from the config.",
    )
    parser.add_argument(
        "--input-run",
        type=Path,
        default=None,
        help=(
            "TREC-format run.tsv to feed the generator. Defaults to the W2 BM25 "
            "run derived from cfg['eval_retrieval']['output_dir']."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for predictions/metrics/examples/manifest. Defaults to "
            "cfg['generation']['output_dir']."
        ),
    )
    parser.add_argument(
        "--retrieval-source",
        type=str,
        default=None,
        help=(
            "Short label for the upstream retriever (e.g. 'bm25', 'dense', "
            "'reranked'). Recorded in the manifest so reports can keyed by it. "
            "Defaults to a label inferred from --input-run."
        ),
    )
    parser.add_argument(
        "--restrict-to-run",
        type=Path,
        default=None,
        help=(
            "Optional secondary run.tsv whose queries the eval set is further "
            "intersected with. Use this to make BM25-driven and reranked-driven "
            "generation evaluate on the SAME 200-query subsample when one "
            "upstream run covers fewer queries than the other."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override cfg['generation']['max_new_tokens']. Use when running a "
            "controlled generation-budget sweep without editing the config; "
            "the manifest's argv record captures the override for provenance."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override cfg['generation']['model_name'] (e.g. ``t5-base``, "
            "``google/flan-t5-base``). Used by the W7-B generator horizontal."
        ),
    )
    parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help=(
            "Refuse to write the manifest if the git working tree has "
            "uncommitted changes. Use for canonical / headline runs where "
            "the recorded commit must be sufficient to reproduce."
        ),
    )
    return parser.parse_args(argv)


def resolve_input_run(args: argparse.Namespace, cfg: dict, project_root: Path) -> Path:
    """Pick the run.tsv to feed the generator. CLI > config-derived default."""
    if args.input_run is not None:
        p = args.input_run
        return p if p.is_absolute() else project_root / p
    return project_root / cfg["eval_retrieval"]["output_dir"] / "run.tsv"


def resolve_output_dir(args: argparse.Namespace, cfg: dict, project_root: Path) -> Path:
    """Pick the output directory. CLI > config."""
    if args.output_dir is not None:
        p = args.output_dir
        return p if p.is_absolute() else project_root / p
    return project_root / cfg["generation"]["output_dir"]


def infer_retrieval_source(input_run: Path) -> str:
    """Best-effort short label derived from the input run path.

    Falls back to 'bm25' for the W2 path and 'unknown' otherwise. The CLI
    flag ``--retrieval-source`` is preferred whenever the caller knows.
    """
    name = input_run.parent.name.lower()
    if "week02" in name or "bm25" in name:
        return "bm25"
    if "week04" in name or "dense" in name:
        return "dense"
    if "week05" in name or "rerank" in name:
        return "reranked"
    return "unknown"


def compute_eligible(
    runs: dict[str, list[str]],
    queries: dict[str, str],
    qa_answers: dict[str, list[str]],
    restrict_qids: set[str] | None = None,
) -> list[str]:
    """Intersect the three sources that a query needs to be evaluable, plus
    an optional ``restrict_qids`` filter (queries covered by another run).
    """
    eligible = set(runs) & set(queries) & set(qa_answers)
    if restrict_qids is not None:
        eligible &= restrict_qids
    return sorted(eligible)


def load_runs(run_path: Path) -> dict[str, list[str]]:
    """Read a TREC-format run file into qid -> ranked doc_ids."""
    runs: dict[str, list[str]] = {}
    with open(run_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, _, doc_id, rank = parts[0], parts[1], parts[2], int(parts[3])
            bucket = runs.setdefault(qid, [])
            # Pad if rank is sparse / out-of-order
            while len(bucket) < rank:
                bucket.append(None)  # type: ignore[arg-type]
            bucket[rank - 1] = doc_id
    # Drop any None gaps (rank gaps shouldn't really exist for our writer)
    for q in runs:
        runs[q] = [d for d in runs[q] if d is not None]
    return runs


def load_qa_references(cache_dir: Path | None) -> dict[str, list[str]]:
    """Build ``query_id -> list[answer]`` from MS MARCO QA v2.1 validation."""
    from datasets import load_dataset

    logger.info("Loading MS MARCO QA v2.1 validation for answer references...")
    ds = load_dataset("ms_marco", "v2.1", split="validation")
    qid_to_answers: dict[str, list[str]] = {}
    for row in ds:
        answers = [
            a.strip()
            for a in (row.get("answers") or [])
            if a and not a.lower().startswith("no answer")
        ]
        if answers:
            qid_to_answers[str(row["query_id"])] = answers
    logger.info("Got answer references for %d queries.", len(qid_to_answers))
    return qid_to_answers


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_new_tokens is not None:
        # Override before config snapshot is captured into the manifest so
        # the snapshot reflects the effective value, not the file default.
        cfg.setdefault("generation", {})["max_new_tokens"] = args.max_new_tokens
    seed = cfg.get("seed", 42)
    seed_coverage = set_global_seed(seed)

    cache_dir = PROJECT_ROOT / cfg["data"].get("cache_dir", "data/raw")
    run_path = resolve_input_run(args, cfg, PROJECT_ROOT)
    output_dir = resolve_output_dir(args, cfg, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_source = args.retrieval_source or infer_retrieval_source(run_path)

    if not run_path.exists():
        logger.error(
            "Missing input run at %s — pass --input-run or run the upstream "
            "retrieval/reranker script first.",
            run_path,
        )
        sys.exit(1)

    # ---- 1. Inputs ----
    logger.info("Loading retrieval run from %s (source=%s)", run_path, retrieval_source)
    runs = load_runs(run_path)
    logger.info("Loaded retrieval results for %d queries.", len(runs))

    restrict_qids: set[str] | None = None
    restrict_run_rel: str | None = None
    if args.restrict_to_run is not None:
        restrict_path = (
            args.restrict_to_run
            if args.restrict_to_run.is_absolute()
            else PROJECT_ROOT / args.restrict_to_run
        )
        if not restrict_path.exists():
            logger.error("Missing --restrict-to-run file at %s", restrict_path)
            sys.exit(1)
        logger.info("Restricting eligibility to queries in %s", restrict_path)
        restrict_qids = set(load_runs(restrict_path).keys())
        logger.info("  ↳ %d queries in restriction set.", len(restrict_qids))
        restrict_run_rel = (
            str(restrict_path.relative_to(PROJECT_ROOT))
            if restrict_path.is_relative_to(PROJECT_ROOT)
            else str(restrict_path)
        )

    # We only need queries (not the corpus) at this point; the corpus is
    # accessed lazily through the ir_datasets docs_store.
    data = load_msmarco_passage(cache_dir=cache_dir, load_corpus=False)
    docs_store = data.docs_store or get_docs_store(cache_dir=cache_dir)

    qid_to_answers = load_qa_references(cache_dir=cache_dir)

    # ---- 2. Eligible eval set ----
    eligible = compute_eligible(runs, data.queries, qid_to_answers, restrict_qids)
    logger.info(
        "Eligible queries (in run + dev/small + QA references%s): %d",
        " + restriction" if restrict_qids is not None else "",
        len(eligible),
    )
    n_eval = args.num_eval_queries or int(
        cfg["generation"].get("num_eval_queries", 200)
    )
    rng = random.Random(seed)
    sample_qids = rng.sample(eligible, min(n_eval, len(eligible)))
    logger.info("Evaluating on %d queries.", len(sample_qids))

    top_k_passages = int(cfg["generation"].get("top_k_passages", 3))

    # ---- 3. Build (query, passages, references) batches ----
    queries: list[str] = []
    passages_per_query: list[list[str]] = []
    references_per_query: list[list[str]] = []
    top_doc_ids_per_query: list[list[str]] = []
    for qid in sample_qids:
        top_ids = runs[qid][:top_k_passages]
        passages = []
        for d in top_ids:
            try:
                passages.append(docs_store.get(d).text)
            except KeyError:
                passages.append("")
        queries.append(data.queries[qid])
        passages_per_query.append(passages)
        references_per_query.append(qid_to_answers[qid])
        top_doc_ids_per_query.append(top_ids)

    # ---- 4. Generate ----
    gen_model_name = args.model_name or cfg["generation"].get("model_name", "t5-small")
    gen_cfg = RAGGenerationConfig(
        model_name=gen_model_name,
        max_input_length=int(cfg["generation"].get("max_input_length", 512)),
        max_new_tokens=int(cfg["generation"].get("max_new_tokens", 64)),
        top_k_passages=top_k_passages,
    )
    generator = RAGGenerator(gen_cfg)

    logger.info("Generating answers...")
    t0 = time.time()
    predictions = generator.generate_batch(queries, passages_per_query)
    gen_time = time.time() - t0
    logger.info(
        "Generated %d answers in %.1f s (%.1f ms / query).",
        len(predictions),
        gen_time,
        gen_time * 1000 / max(len(predictions), 1),
    )

    # ---- 5. Persist predictions.jsonl ----
    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for qid, q, passages, top_ids, pred, refs in zip(
            sample_qids,
            queries,
            passages_per_query,
            top_doc_ids_per_query,
            predictions,
            references_per_query,
        ):
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": q,
                        "top_doc_ids": top_ids,
                        "passages": passages,
                        "prediction": pred,
                        "references": refs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Wrote predictions to %s", pred_path)

    # ---- 6. examples.jsonl (small qualitative subset) ----
    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid, q, passages, pred, refs in list(
            zip(sample_qids, queries, passages_per_query, predictions, references_per_query)
        )[:20]:
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": q,
                        "passages": passages,
                        "prediction": pred,
                        "references": refs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Wrote %d qualitative examples to %s", min(20, len(predictions)), examples_path)

    # ---- 7. Metrics (unified schema across W2/W3) ----
    metrics = evaluate_generation(predictions, references_per_query)
    logger.info("Metrics: %s", metrics)

    n_examples = metrics.pop("n_predictions", len(predictions))
    payload = {
        "task": "generation",
        "dataset": "msmarco-passage/dev/small ∩ ms_marco/v2.1/validation",
        "n_examples": n_examples,
        "config": cfg,
        "metrics": metrics,
        "wall_clock_seconds": {"generation": gen_time},
        "environment": capture_environment(),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    input_run_rel = (
        str(run_path.relative_to(PROJECT_ROOT))
        if run_path.is_relative_to(PROJECT_ROOT)
        else str(run_path)
    )
    manifest_extra: dict[str, object] = {
        "task": "generation",
        "model_name": gen_cfg.model_name,
        "top_k_passages": top_k_passages,
        "n_eval_queries": len(predictions),
        "seed": seed,
        "seed_coverage": seed_coverage,
        "input_run": input_run_rel,
        "retrieval_source": retrieval_source,
        "run_name": output_dir.name,
    }
    if restrict_run_rel is not None:
        manifest_extra["restrict_to_run"] = restrict_run_rel
    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv,
        config_path=args.config,
        extra_outputs=[pred_path, examples_path],
        extra=manifest_extra,
        require_clean_tree=args.require_clean_tree,
    )

    print("\n=== RAG generation ===")
    print(f"retrieval source: {retrieval_source}")
    print(f"input run:        {input_run_rel}")
    print(f"queries evaluated: {len(predictions)}")
    for key in ("rouge-l", "bleu", "exact-match", "token-f1"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

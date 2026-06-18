"""End-to-end RAG generation baseline.

Pipeline:

1. Load a TREC-format retrieval run (``--input-run``; defaults to the
   BM25 run at ``outputs/bm25_baseline/run.tsv``).
2. Load dev/small queries and the MS MARCO Passage docs_store (random access).
3. Cross-reference dev/small query ids with MS MARCO QA v2.1 (HuggingFace
   ``ms_marco`` dataset, validation split) to recover human-written answers
   for evaluation.
4. For each evaluated query, take the top-K passages from the run, generate
   an answer with the Seq2Seq model, and score predictions.
5. Persist (under ``--output-dir``, defaults to ``outputs/generation``):
   - ``predictions.jsonl``
   - ``metrics.json``
   - ``examples.jsonl``
   - ``manifest.json``

The runner is **retrieval-source agnostic** — point ``--input-run`` at any
TREC-format ``run.tsv`` (BM25 / dense / reranked) and the rest of the
pipeline is identical. Use ``--restrict-to-run`` to make different
retrieval sources eval on the SAME query subsample (apples-to-apples),
which matters when one source covers fewer queries than another (e.g.
the reranker covers 1,000 dev queries, not all 6,980).

Run from the project root::

    # generation baseline: BM25 → T5-small (defaults preserve the legacy behaviour)
    python experiments/run_generation_baseline.py

    # Reranked → T5-small, restricted to reranker-covered queries
    python experiments/run_generation_baseline.py \\
        --input-run outputs/cross_encoder_rerank/run.tsv \\
        --output-dir outputs/generation_reranked \\
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

from msmarco_genqa.data.msmarco import get_docs_store, load_msmarco_passage
from msmarco_genqa.evaluation.generation import evaluate_generation
from msmarco_genqa.generation.context_packing import ContextPackingConfig, pack_context
from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator
from msmarco_genqa.reranking.io import read_run_tsv
from msmarco_genqa.util.environment import capture_environment
from msmarco_genqa.util.manifest import (
    compute_data_fingerprint,
    compute_env_fingerprint,
    compute_resolved_config_hash,
    compute_sampling_block,
    write_resolved_config,
    write_run_manifest,
)
from msmarco_genqa.util.seeding import set_global_seed

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
            "TREC-format run.tsv to feed the generator. Defaults to the BM25 "
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
            "``google/flan-t5-base``). Used by the generator-capacity sweep."
        ),
    )
    parser.add_argument(
        "--context-packing",
        action="store_true",
        help="Enable deterministic context packing for this generation run.",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=None,
        help="Override generation.context_packing.max_context_chars.",
    )
    parser.add_argument(
        "--context-max-passage-chars",
        type=int,
        default=None,
        help="Override generation.context_packing.max_passage_chars.",
    )
    parser.add_argument(
        "--context-sentence-selection",
        choices=["head", "query_overlap"],
        default=None,
        help="Override generation.context_packing.sentence_selection.",
    )
    parser.add_argument(
        "--context-ordering",
        choices=["rank", "shorter_first"],
        default=None,
        help="Override generation.context_packing.ordering.",
    )
    parser.add_argument(
        "--no-context-deduplicate",
        action="store_true",
        help="Disable generation.context_packing.deduplicate for this run.",
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
    parser.add_argument(
        "--allow-incomplete-manifest",
        action="store_true",
        help=(
            "Bypass the schema-v2 required-field contract on manifest write. "
            "Development-only escape hatch; production / headline runs must "
            "leave this off so missing reproducibility fields fail loudly."
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

    Falls back to 'bm25' for the BM25 path and 'unknown' otherwise. The CLI
    flag ``--retrieval-source`` is preferred whenever the caller knows.
    """
    name = input_run.parent.name.lower()
    if "bm25" in name:
        return "bm25"
    if "dense" in name:
        return "dense"
    if "rerank" in name:
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
    return {
        qid: [doc_id for doc_id, _score in docs]
        for qid, docs in read_run_tsv(run_path).items()
    }


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
    packing_overrides = cfg.setdefault("generation", {}).setdefault("context_packing", {})
    if args.context_packing:
        packing_overrides["enabled"] = True
    if args.context_max_chars is not None:
        packing_overrides["max_context_chars"] = args.context_max_chars
    if args.context_max_passage_chars is not None:
        packing_overrides["max_passage_chars"] = args.context_max_passage_chars
    if args.context_sentence_selection is not None:
        packing_overrides["sentence_selection"] = args.context_sentence_selection
    if args.context_ordering is not None:
        packing_overrides["ordering"] = args.context_ordering
    if args.no_context_deduplicate:
        packing_overrides["deduplicate"] = False
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
    context_packing_cfg = ContextPackingConfig.from_mapping(
        cfg["generation"].get("context_packing", {})
    )

    # ---- 3. Build (query, passages, references) batches ----
    queries: list[str] = []
    passages_per_query: list[list[str]] = []
    references_per_query: list[list[str]] = []
    top_doc_ids_per_query: list[list[str]] = []
    context_packing_per_query: list[dict | None] = []
    for qid in sample_qids:
        raw_top_ids = runs[qid][:top_k_passages]
        raw_passages = []
        for d in raw_top_ids:
            try:
                raw_passages.append(docs_store.get(d).text)
            except KeyError:
                raw_passages.append("")
        query = data.queries[qid]
        if context_packing_cfg.enabled:
            packed = pack_context(
                query=query,
                doc_ids=raw_top_ids,
                passages=raw_passages,
                config=context_packing_cfg,
            )
            passages = packed.passages
            top_ids = packed.doc_ids
            context_packing = packed.to_json(context_packing_cfg)
        else:
            passages = raw_passages
            top_ids = raw_top_ids
            context_packing = None
        queries.append(query)
        passages_per_query.append(passages)
        references_per_query.append(qid_to_answers[qid])
        top_doc_ids_per_query.append(top_ids)
        context_packing_per_query.append(context_packing)

    # ---- 4. Generate ----
    gen_model_name = args.model_name or cfg["generation"].get("model_name", "t5-small")
    # If --model-name is used to override the default checkpoint (e.g. the
    # generator-capacity sweep swapping t5-small for t5-base), the
    # baked-in revision pin from the config no longer applies — fall back
    # to unpinned to avoid loading the wrong revision under the wrong name.
    revision_from_cfg = cfg["generation"].get("revision")
    gen_revision = revision_from_cfg if args.model_name is None else None
    gen_cfg = RAGGenerationConfig(
        model_name=gen_model_name,
        revision=gen_revision,
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
        for qid, q, passages, top_ids, pred, refs, context_packing in zip(
            sample_qids,
            queries,
            passages_per_query,
            top_doc_ids_per_query,
            predictions,
            references_per_query,
            context_packing_per_query,
        ):
            row = {
                "query_id": qid,
                "query": q,
                "top_doc_ids": top_ids,
                "passages": passages,
                "prediction": pred,
                "references": refs,
            }
            if context_packing is not None:
                row["context_packing"] = context_packing
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )
    logger.info("Wrote predictions to %s", pred_path)

    # ---- 6. examples.jsonl (small qualitative subset) ----
    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid, q, passages, pred, refs, context_packing in list(
            zip(
                sample_qids,
                queries,
                passages_per_query,
                predictions,
                references_per_query,
                context_packing_per_query,
            )
        )[:20]:
            row = {
                "query_id": qid,
                "query": q,
                "passages": passages,
                "prediction": pred,
                "references": refs,
            }
            if context_packing is not None:
                row["context_packing"] = context_packing
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )
    logger.info("Wrote %d qualitative examples to %s", min(20, len(predictions)), examples_path)

    # ---- 7. Metrics (unified schema across retrieval and generation) ----
    metrics = evaluate_generation(predictions, references_per_query)
    logger.info("Metrics: %s", metrics)

    n_examples = metrics.pop("n_predictions", len(predictions))
    env_dict = capture_environment()
    # Generation inherits sampling context from its upstream retrieval run.
    # bm25 (full corpus) → not sampled. dense / reranked → qrels-anchored
    # sample. Generation's own metrics (Token-F1, ROUGE-L, etc.) are
    # answer-vs-reference, NOT recall-based, so the caveat is contextual
    # rather than directly affecting metric direction — but the provenance
    # is still load-bearing for any cross-source comparison (BM25-driven
    # vs dense-driven generation are NOT apples-to-apples without it).
    _generation_is_sampled = retrieval_source in ("dense", "reranked")
    payload = {
        "task": "generation",
        "dataset": "msmarco-passage/dev/small ∩ ms_marco/v2.1/validation",
        "n_examples": n_examples,
        "config": cfg,
        "metrics": metrics,
        "sampling": compute_sampling_block(
            is_sampled=_generation_is_sampled,
            method="qrels-anchored (via upstream retrieval)"
            if _generation_is_sampled
            else None,
        ),
        "wall_clock_seconds": {"generation": gen_time},
        "environment": env_dict,
    }
    if context_packing_cfg.enabled:
        packing_rows = [row for row in context_packing_per_query if row is not None]
        original_chars = [int(row["original_context_chars"]) for row in packing_rows]
        packed_chars = [int(row["packed_context_chars"]) for row in packing_rows]
        payload["context_packing"] = {
            "config": context_packing_cfg.to_json(),
            "mean_original_context_chars": _mean_ints(original_chars),
            "mean_packed_context_chars": _mean_ints(packed_chars),
            "mean_compression_ratio": _mean_floats(
                [float(row["compression_ratio"]) for row in packing_rows]
            ),
            "n_queries_with_empty_context": sum(1 for chars in packed_chars if chars == 0),
        }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    input_run_rel = (
        str(run_path.relative_to(PROJECT_ROOT))
        if run_path.is_relative_to(PROJECT_ROOT)
        else str(run_path)
    )
    resolved_config_path = write_resolved_config(cfg, output_dir)
    resolved_config_hash = compute_resolved_config_hash(cfg)
    data_fingerprint = compute_data_fingerprint(
        cache_dir=cache_dir,
        extra_files={"input_run": run_path},
    )
    env_fingerprint = compute_env_fingerprint(env_dict)

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
        "context_packing": context_packing_cfg.to_json(),
        "resolved_config_hash": resolved_config_hash,
        "data_fingerprint": data_fingerprint,
        "env_fingerprint": env_fingerprint,
    }
    if restrict_run_rel is not None:
        manifest_extra["restrict_to_run"] = restrict_run_rel
    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv,
        config_path=args.config,
        extra_outputs=[pred_path, examples_path, resolved_config_path],
        extra=manifest_extra,
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
    )

    print("\n=== RAG generation ===")
    print(f"retrieval source: {retrieval_source}")
    print(f"input run:        {input_run_rel}")
    print(f"queries evaluated: {len(predictions)}")
    for key in ("rouge-l", "bleu", "exact-match", "token-f1"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


def _mean_ints(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_floats(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()

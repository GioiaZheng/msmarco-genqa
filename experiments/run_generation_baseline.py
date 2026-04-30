"""End-to-end Week 3 RAG generation baseline.

Pipeline:

1. Load the Week 2 BM25 run (``outputs/week02_bm25/run.tsv``).
2. Load dev/small queries and the MS MARCO Passage docs_store (random access).
3. Cross-reference dev/small query ids with MS MARCO QA v2.1 (HuggingFace
   ``ms_marco`` dataset, validation split) to recover human-written answers
   for evaluation.
4. For each evaluated query, take the top-K BM25 passages, generate an
   answer with the Seq2Seq model, and score predictions.
5. Persist:
   - ``outputs/week03_generation/predictions.jsonl``
   - ``outputs/week03_generation/metrics.json``
   - ``outputs/week03_generation/examples.jsonl``

Run from the project root::

    python experiments/run_generation_baseline.py
    python experiments/run_generation_baseline.py --config configs/baseline.yaml
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

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
    seed = cfg.get("seed", 42)
    random.seed(seed)

    cache_dir = PROJECT_ROOT / cfg["data"].get("cache_dir", "data/raw")
    w2_dir = PROJECT_ROOT / cfg["eval_retrieval"]["output_dir"]
    w3_dir = PROJECT_ROOT / cfg["generation"]["output_dir"]
    w3_dir.mkdir(parents=True, exist_ok=True)

    run_path = w2_dir / "run.tsv"
    if not run_path.exists():
        logger.error(
            "Missing %s — run experiments/run_retrieval.py first.", run_path
        )
        sys.exit(1)

    # ---- 1. Inputs ----
    logger.info("Loading retrieval run from %s", run_path)
    runs = load_runs(run_path)
    logger.info("Loaded retrieval results for %d queries.", len(runs))

    # We only need queries (not the corpus) at this point; the corpus is
    # accessed lazily through the ir_datasets docs_store.
    data = load_msmarco_passage(cache_dir=cache_dir, load_corpus=False)
    docs_store = data.docs_store or get_docs_store(cache_dir=cache_dir)

    qid_to_answers = load_qa_references(cache_dir=cache_dir)

    # ---- 2. Eligible eval set ----
    eligible = sorted(set(runs) & set(data.queries) & set(qid_to_answers))
    logger.info(
        "Eligible queries (in run + dev/small + QA references): %d",
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
    gen_cfg = RAGGenerationConfig(
        model_name=cfg["generation"].get("model_name", "t5-small"),
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
    pred_path = w3_dir / "predictions.jsonl"
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
    examples_path = w3_dir / "examples.jsonl"
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

    # ---- 7. Metrics ----
    metrics = evaluate_generation(predictions, references_per_query)
    logger.info("Metrics: %s", metrics)

    payload = {
        "config": cfg,
        "metrics": metrics,
        "wall_clock_seconds": {"generation": gen_time},
        "n_eval": len(predictions),
    }
    with open(w3_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", w3_dir / "metrics.json")

    print("\n=== Week 3 RAG generation baseline ===")
    print(f"queries evaluated: {len(predictions)}")
    for key in ("rouge-l", "bleu", "exact-match", "token-f1"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {w3_dir}")


if __name__ == "__main__":
    main()

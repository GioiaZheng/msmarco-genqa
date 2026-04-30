"""Build the official BM25 baseline on MS MARCO Passage / dev/small.

End-to-end:

1. Load corpus, dev/small queries, and qrels via ``ir_datasets``.
2. Build (or load from cache) a ``bm25s`` index over the corpus.
3. Retrieve top-k for every dev query.
4. Compute MRR@10, Recall@100, Recall@1000.
5. Persist:
   - ``outputs/week02_bm25/metrics.json``
   - ``outputs/week02_bm25/run.tsv``  (TREC-format run, top-1000 by default)
   - ``outputs/week02_bm25/examples.jsonl``  (qualitative samples)

Run from the project root::

    python experiments/run_retrieval.py
    python experiments/run_retrieval.py --config configs/baseline.yaml
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

from src.data.msmarco import load_msmarco_passage  # noqa: E402
from src.evaluation.retrieval import evaluate_retrieval  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402


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
        "--rebuild-index",
        action="store_true",
        help="Force a fresh BM25 index even if a cached one exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    random.seed(seed)

    output_dir = PROJECT_ROOT / cfg["eval_retrieval"]["output_dir"]
    index_dir = PROJECT_ROOT / cfg["retrieval"]["index_dir"]
    cache_dir = PROJECT_ROOT / cfg["data"].get("cache_dir", "data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(cfg["retrieval"]["top_k"])
    ks_mrr = tuple(cfg["eval_retrieval"].get("ks_mrr", [10]))
    ks_recall = tuple(cfg["eval_retrieval"].get("ks_recall", [100, 1000]))
    n_examples = int(cfg["eval_retrieval"].get("n_examples", 20))
    corpus_limit = cfg["data"].get("corpus_limit")

    # ---- 1. Data ----
    have_index = (index_dir / "config.json").exists() and not args.rebuild_index
    data = load_msmarco_passage(
        cache_dir=cache_dir,
        load_corpus=not have_index,
        limit=corpus_limit,
    )

    # ---- 2. Index ----
    index_time: float | None = None
    if have_index:
        logging.info("Loading cached BM25 index from %s", index_dir)
        retriever = BM25Retriever.load(index_dir)
    else:
        retriever = BM25Retriever(
            corpus_texts=data.corpus_texts,
            doc_ids=data.corpus_doc_ids,
            k1=float(cfg["retrieval"]["k1"]),
            b=float(cfg["retrieval"]["b"]),
            stopwords=cfg["retrieval"].get("stopwords", "en"),
        )
        t0 = time.time()
        retriever.build()
        index_time = time.time() - t0
        retriever.save(index_dir)

    # ---- 3. Retrieval ----
    qids = list(data.queries.keys())
    queries_text = [data.queries[q] for q in qids]

    logging.info("Retrieving top-%d for %d queries...", top_k, len(qids))
    t0 = time.time()
    scores, doc_ids_lists = retriever.retrieve_batch(queries_text, k=top_k)
    search_time = time.time() - t0
    logging.info(
        "Retrieval done in %.1f s (%.1f ms/query).",
        search_time,
        search_time * 1000 / max(len(qids), 1),
    )

    # ---- 4. Build runs dict + write TREC run.tsv ----
    runs: dict[str, list[str]] = {}
    run_path = output_dir / "run.tsv"
    with open(run_path, "w") as f:
        for qid, doc_ids_for_q, score_row in zip(qids, doc_ids_lists, scores):
            runs[qid] = doc_ids_for_q
            for rank, (doc_id, score) in enumerate(zip(doc_ids_for_q, score_row), 1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{float(score):.6f}\tbm25\n")
    logging.info("Wrote TREC run to %s", run_path)

    # ---- 5. Metrics ----
    metrics = evaluate_retrieval(
        runs=runs,
        qrels=data.qrels,
        ks_mrr=ks_mrr,
        ks_recall=ks_recall,
    )
    logging.info("Metrics: %s", metrics)

    # ---- 6. Qualitative examples ----
    rng = random.Random(seed)
    eligible = [q for q in qids if data.qrels.get(q)]
    sample = rng.sample(eligible, min(n_examples, len(eligible)))
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    # Resolve passage text. If we built fresh, ``data`` already has the corpus
    # in memory; otherwise we use the docs_store for random access.
    if data.corpus_texts:
        id_to_text = dict(zip(data.corpus_doc_ids, data.corpus_texts))

        def get_text(doc_id: str) -> str:
            return id_to_text.get(doc_id, "")
    else:
        from src.data.msmarco import get_docs_store

        store = get_docs_store(cache_dir=cache_dir)

        def get_text(doc_id: str) -> str:
            try:
                return store.get(doc_id).text
            except KeyError:
                return ""

    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid in sample:
            relevant = data.qrels.get(qid, set())
            top_doc_ids = runs[qid][:10]
            score_row = scores[qid_to_idx[qid]]
            top_results = [
                {
                    "doc_id": d,
                    "rank": i + 1,
                    "score": float(score_row[i]),
                    "passage": get_text(d),
                    "is_relevant": d in relevant,
                }
                for i, d in enumerate(top_doc_ids)
            ]
            first_rank = next(
                (r["rank"] for r in top_results if r["is_relevant"]),
                None,
            )
            example = {
                "query_id": qid,
                "query": data.queries[qid],
                "relevant_doc_ids": sorted(relevant),
                "first_relevant_rank_in_top10": first_rank,
                "top_results": top_results,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logging.info("Wrote %d examples to %s", len(sample), examples_path)

    # ---- 7. metrics.json ----
    payload = {
        "config": cfg,
        "metrics": metrics,
        "wall_clock_seconds": {
            "indexing": index_time,
            "search": search_time,
        },
        "top_k": top_k,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logging.info("Wrote metrics to %s", output_dir / "metrics.json")

    # ---- 8. Friendly summary ----
    print("\n=== Week 2 BM25 baseline ===")
    print(f"queries evaluated: {metrics.get('n_queries')}")
    for key in ("mrr@10", "recall@100", "recall@1000"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

"""Synthetic-data smoke test for the Week 2 pipeline.

Bypasses ``ir_datasets`` (so no ~3 GB download) and runs the rest of the
Week 2 orchestration as written in ``experiments/run_retrieval.py``:
``BM25Retriever.build`` → save/load → batch retrieve → ``evaluate_retrieval``
→ ``run.tsv`` / ``metrics.json`` / ``examples.jsonl`` serialization.

This validates everything **except** the ir_datasets HTTP loader. The numbers
produced are meaningless and clearly labelled as such in the output payload.

Usage::

    python scripts/smoke_test_week02.py

After it succeeds, run::

    python -m src.reporting.build_report --week week02
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch the data loader BEFORE importing the experiment script, so that
# ``from src.data.msmarco import load_msmarco_passage`` inside the runner
# resolves to our stub.
import src.data.msmarco as _msmarco  # noqa: E402

from src.data.msmarco import MSMarcoPassage  # noqa: E402
from src.evaluation.retrieval import evaluate_retrieval  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

SYNTHETIC_PASSAGES = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Canberra is the capital city of Australia, located in the Australian Capital Territory.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.",
    "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "William Shakespeare was an English playwright, poet, and actor of the Elizabethan era.",
    "DNA carries genetic information used in the growth, development, functioning, and reproduction of all known organisms.",
    "Sydney is the largest city in Australia, but it is not the capital.",
    "Paris is the capital and most populous city of France, located on the river Seine.",
] + [f"Filler passage number {i} with no useful content." for i in range(40)]

SYNTHETIC_DOC_IDS = [f"doc{i}" for i in range(len(SYNTHETIC_PASSAGES))]

SYNTHETIC_QUERIES = {
    "q1": "what is the capital of australia",
    "q2": "where is the eiffel tower located",
    "q3": "what is photosynthesis",
    "q4": "who was shakespeare",
    "q5": "what is the largest ocean",
}

# Map query id -> set of relevant doc ids (synthetic ground truth)
SYNTHETIC_QRELS = {
    "q1": {"doc1"},
    "q2": {"doc0"},
    "q3": {"doc4"},
    "q4": {"doc6"},
    "q5": {"doc5"},
}


def _stub_loader(*args, **kwargs):
    return MSMarcoPassage(
        corpus_doc_ids=list(SYNTHETIC_DOC_IDS),
        corpus_texts=list(SYNTHETIC_PASSAGES),
        queries=dict(SYNTHETIC_QUERIES),
        qrels={k: set(v) for k, v in SYNTHETIC_QRELS.items()},
        docs_store=None,
    )


# ---------------------------------------------------------------------------
# Orchestration (mirrors experiments/run_retrieval.py)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Override the loader at module level
    _msmarco.load_msmarco_passage = _stub_loader

    import yaml
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/baseline.yaml").read_text())

    output_dir = PROJECT_ROOT / cfg["eval_retrieval"]["output_dir"]
    index_dir = PROJECT_ROOT / "data/processed/bm25_index_smoke"  # use a separate dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark this output as a smoke run, not a real baseline
    cfg.setdefault("data", {})["expected_corpus_size"] = (
        f"{len(SYNTHETIC_PASSAGES)} (synthetic smoke test — NOT the official corpus)"
    )
    cfg["data"]["expected_dev_queries"] = (
        f"{len(SYNTHETIC_QUERIES)} (synthetic)"
    )
    cfg.setdefault("retrieval", {})["index_dir"] = str(index_dir.relative_to(PROJECT_ROOT))

    seed = cfg.get("seed", 42)
    random.seed(seed)
    top_k = int(cfg["retrieval"]["top_k"])
    top_k = min(top_k, len(SYNTHETIC_PASSAGES))  # cap to corpus size

    data = _stub_loader()

    # Build / save / reload the index (cover the persistence path too)
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
    logging.info("Reloading index from disk to validate save/load path...")
    retriever = BM25Retriever.load(index_dir)

    qids = list(data.queries.keys())
    queries_text = [data.queries[q] for q in qids]
    t0 = time.time()
    scores, doc_ids_lists = retriever.retrieve_batch(queries_text, k=top_k)
    search_time = time.time() - t0

    # run.tsv
    runs: dict[str, list[str]] = {}
    run_path = output_dir / "run.tsv"
    with open(run_path, "w") as f:
        for qid, docs, score_row in zip(qids, doc_ids_lists, scores):
            runs[qid] = docs
            for rank, (d, s) in enumerate(zip(docs, score_row), 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{float(s):.6f}\tbm25\n")

    metrics = evaluate_retrieval(
        runs, data.qrels, ks_mrr=(10,), ks_recall=(100, 1000),
    )

    # examples.jsonl
    id_to_text = dict(zip(data.corpus_doc_ids, data.corpus_texts))
    qid_to_idx = {q: i for i, q in enumerate(qids)}
    sample = list(qids)  # tiny set, dump all
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
                    "passage": id_to_text.get(d, ""),
                    "is_relevant": d in relevant,
                }
                for i, d in enumerate(top_doc_ids)
            ]
            first = next(
                (r["rank"] for r in top_results if r["is_relevant"]), None
            )
            f.write(json.dumps({
                "query_id": qid,
                "query": data.queries[qid],
                "relevant_doc_ids": sorted(relevant),
                "first_relevant_rank_in_top10": first,
                "top_results": top_results,
            }, ensure_ascii=False) + "\n")

    payload = {
        "smoke_test": True,
        "config": cfg,
        "metrics": metrics,
        "wall_clock_seconds": {"indexing": index_time, "search": search_time},
        "top_k": top_k,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n=== smoke test (synthetic data, NOT the official baseline) ===")
    print(f"queries evaluated: {metrics.get('n_queries')}")
    for k in ("mrr@10", "recall@100", "recall@1000"):
        if k in metrics:
            print(f"  {k:14s} = {metrics[k]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

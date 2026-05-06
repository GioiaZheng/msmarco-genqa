"""Build the official BM25 baseline on MS MARCO Passage / dev/small.

End-to-end:

1. Load corpus, dev/small queries, and qrels via ``ir_datasets``.
2. Build (or load from cache) a ``bm25s`` index over the corpus.
3. Retrieve top-k for every dev query, in chunks; append each chunk to
   ``run.tsv`` so a killed run can be resumed instead of restarting from
   query 0.
4. Compute MRR@10, Recall@100, Recall@1000.
5. Persist:
   - ``outputs/week02_bm25/metrics.json``
   - ``outputs/week02_bm25/run.tsv``  (TREC-format run, top-1000 by default)
   - ``outputs/week02_bm25/examples.jsonl``  (qualitative samples)

Run from the project root::

    python experiments/run_retrieval.py
    python experiments/run_retrieval.py --resume                       # pick up where a killed run stopped
    python experiments/run_retrieval.py --rebuild-index                # force fresh BM25 index
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
from src.util.environment import capture_environment  # noqa: E402

logger = logging.getLogger("run_retrieval")


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip queries that already have a complete top-K block in run.tsv "
            "and append only the missing ones. Without --resume, run.tsv is "
            "truncated at the start of the run."
        ),
    )
    return parser.parse_args()


def _read_done_qids(run_path: Path, top_k: int) -> set[str]:
    """Return query ids that already have a *complete* top-K block in ``run_path``.

    A qid is considered done only if its highest observed rank equals ``top_k``
    AND all ranks 1..top_k are present, so a partially-written chunk (e.g. the
    one being flushed when SIGKILL hit) is automatically retried.
    """
    if not run_path.exists():
        return set()
    counts: dict[str, int] = {}
    max_rank: dict[str, int] = {}
    with open(run_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, _, _doc_id, rank_str = parts[0], parts[1], parts[2], parts[3]
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            counts[qid] = counts.get(qid, 0) + 1
            max_rank[qid] = max(max_rank.get(qid, 0), rank)
    return {
        qid
        for qid in counts
        if counts[qid] == top_k and max_rank[qid] == top_k
    }


def _read_runs_from_tsv(run_path: Path) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Read run.tsv into ``{qid: [doc_id sorted by rank]}`` and ``{qid: [score sorted by rank]}``."""
    by_qid: dict[str, list[tuple[int, str, float]]] = {}
    with open(run_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            qid, _q0, doc_id, rank_str, score_str, _sys = parts[:6]
            try:
                rank = int(rank_str)
                score = float(score_str)
            except ValueError:
                continue
            by_qid.setdefault(qid, []).append((rank, doc_id, score))
    runs: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for qid, triples in by_qid.items():
        triples.sort(key=lambda t: t[0])
        runs[qid] = [t[1] for t in triples]
        scores[qid] = [t[2] for t in triples]
    return runs, scores


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
    chunk_size = int(cfg["retrieval"].get("chunk_size", 200))
    n_threads = int(cfg["retrieval"].get("n_threads", 0))
    bm25s_chunksize = int(cfg["retrieval"].get("bm25s_chunksize", 50))
    ks_mrr = tuple(cfg["eval_retrieval"].get("ks_mrr", [10]))
    ks_ndcg = tuple(cfg["eval_retrieval"].get("ks_ndcg", [10]))
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
        logger.info("Loading cached BM25 index from %s", index_dir)
        # Validate that the cached index was built with the same BM25
        # parameters as the current YAML config. Otherwise the user's k1/b
        # changes would be silently ignored — see R1 in the rigor review.
        expected = {
            "k1": float(cfg["retrieval"]["k1"]),
            "b": float(cfg["retrieval"]["b"]),
            "stopwords": cfg["retrieval"].get("stopwords", "en"),
        }
        retriever = BM25Retriever.load(
            index_dir,
            n_threads=n_threads,
            chunksize=bm25s_chunksize,
            expected_config=expected,
        )
    else:
        retriever = BM25Retriever(
            corpus_texts=data.corpus_texts,
            doc_ids=data.corpus_doc_ids,
            k1=float(cfg["retrieval"]["k1"]),
            b=float(cfg["retrieval"]["b"]),
            stopwords=cfg["retrieval"].get("stopwords", "en"),
            n_threads=n_threads,
            chunksize=bm25s_chunksize,
        )
        t0 = time.time()
        retriever.build()
        index_time = time.time() - t0
        retriever.save(index_dir)

    # ---- 3. Plan retrieval ----
    qids = list(data.queries.keys())
    run_path = output_dir / "run.tsv"

    if args.resume:
        done = _read_done_qids(run_path, top_k=top_k)
        if done:
            logger.info(
                "Resume mode: %d / %d queries already complete in %s; "
                "skipping those.",
                len(done),
                len(qids),
                run_path,
            )
            # Rewrite run.tsv keeping only entries for done qids. Without this,
            # a kill mid-chunk leaves partial lines for an incomplete qid; on
            # resume we re-retrieve that qid in full and append, ending up with
            # duplicate entries.
            kept_lines = []
            with open(run_path) as rf:
                for line in rf:
                    qid_field = line.split("\t", 1)[0]
                    if qid_field in done:
                        kept_lines.append(line)
            with open(run_path, "w") as wf:
                wf.writelines(kept_lines)
        else:
            logger.info(
                "Resume mode requested but no complete entries found; "
                "starting from query 0 (no truncation since file may be empty)."
            )
        pending_qids = [q for q in qids if q not in done]
        run_file_mode = "a"
    else:
        # Fresh run: truncate run.tsv. (We do this here so that aborted partial
        # writes from prior runs don't pollute the new output.)
        if run_path.exists():
            logger.info("Truncating existing %s (use --resume to keep it).", run_path)
            run_path.unlink()
        pending_qids = list(qids)
        run_file_mode = "w"

    # Pre-compute the qualitative-example sample BEFORE the loop, using a fixed
    # RNG seed, so the same set of examples is chosen regardless of whether we
    # resumed mid-run. We capture top-10 doc_ids+scores per sampled qid as we
    # go (cheap; bounded size) but otherwise rely on run.tsv for evaluation.
    rng = random.Random(seed)
    eligible = [q for q in qids if data.qrels.get(q)]
    sample_qids = rng.sample(eligible, min(n_examples, len(eligible)))
    sample_qids_set = set(sample_qids)

    # ---- 4. Chunked retrieval ----
    logger.info(
        "Retrieving top-%d for %d queries (%d pending) in chunks of %d "
        "(n_threads=%d).",
        top_k,
        len(qids),
        len(pending_qids),
        chunk_size,
        n_threads,
    )
    t_search_start = time.time()
    chunks_done = 0
    queries_done_this_run = 0

    if pending_qids:
        with open(run_path, run_file_mode) as run_f:
            for chunk_start in range(0, len(pending_qids), chunk_size):
                chunk_qids = pending_qids[chunk_start : chunk_start + chunk_size]
                chunk_texts = [data.queries[q] for q in chunk_qids]

                t0 = time.time()
                chunk_scores, chunk_doc_ids = retriever.retrieve_batch(
                    chunk_texts,
                    k=top_k,
                )
                chunk_seconds = time.time() - t0

                for qid, doc_ids_for_q, score_row in zip(
                    chunk_qids, chunk_doc_ids, chunk_scores
                ):
                    for rank, (doc_id, score) in enumerate(
                        zip(doc_ids_for_q, score_row), 1
                    ):
                        run_f.write(
                            f"{qid}\tQ0\t{doc_id}\t{rank}\t{float(score):.6f}\tbm25\n"
                        )
                run_f.flush()  # checkpoint: ensure chunk is durable on disk

                chunks_done += 1
                queries_done_this_run += len(chunk_qids)
                logger.info(
                    "chunk %d: %d queries in %.1f s (%.1f ms/query); "
                    "this-run progress %d / %d",
                    chunks_done,
                    len(chunk_qids),
                    chunk_seconds,
                    chunk_seconds * 1000 / max(len(chunk_qids), 1),
                    queries_done_this_run,
                    len(pending_qids),
                )
    else:
        logger.info("All queries already retrieved; nothing to do for this phase.")

    search_time = time.time() - t_search_start
    if pending_qids:
        logger.info(
            "Retrieval done in %.1f s for the %d pending queries (%.1f ms/query).",
            search_time,
            len(pending_qids),
            search_time * 1000 / max(len(pending_qids), 1),
        )

    # ---- 5. Materialise runs from disk for evaluation ----
    logger.info("Reading %s for evaluation...", run_path)
    runs, scores_by_qid = _read_runs_from_tsv(run_path)

    metrics = evaluate_retrieval(
        runs=runs,
        qrels=data.qrels,
        ks_mrr=ks_mrr,
        ks_recall=ks_recall,
        ks_ndcg=ks_ndcg,
    )
    logger.info("Metrics: %s", metrics)

    # ---- 6. Qualitative examples ----
    # Resolve passage text. If we built the index in this run, the corpus is
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
        for qid in sample_qids:
            relevant = data.qrels.get(qid, set())
            ranked_doc_ids = runs.get(qid, [])
            ranked_scores = scores_by_qid.get(qid, [])
            top_doc_ids = ranked_doc_ids[:10]
            top_scores = ranked_scores[:10]
            top_results = [
                {
                    "doc_id": d,
                    "rank": i + 1,
                    "score": float(s),
                    "passage": get_text(d),
                    "is_relevant": d in relevant,
                }
                for i, (d, s) in enumerate(zip(top_doc_ids, top_scores))
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
    logger.info("Wrote %d examples to %s", len(sample_qids), examples_path)

    # ---- 7. metrics.json (unified schema across W2/W3) ----
    n_examples = metrics.pop("n_queries", None)  # promote count to top level
    payload = {
        "task": "retrieval",
        "dataset": "msmarco-passage/dev/small",
        "n_examples": n_examples,
        "config": cfg,
        "metrics": metrics,
        "wall_clock_seconds": {
            "indexing": index_time,
            "search": search_time,
            "search_pending_count": len(pending_qids),
        },
        "environment": capture_environment(),
        "top_k": top_k,
        "resumed": args.resume and bool(set(qids) - set(pending_qids)),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    # ---- 8. Friendly summary ----
    print("\n=== Week 2 BM25 baseline ===")
    print(f"queries evaluated: {n_examples}")
    for key in ("mrr@10", "ndcg@10", "recall@100", "recall@1000"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

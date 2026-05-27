"""Week 5: cross-encoder reranking on top of the W4 dense retrieval run.

Pipeline:

1. Load a first-stage ``run.tsv`` (defaults to the W4 dense run) and
   truncate to top-K per query (default K = 100).
2. Load the W4 sample doc_id pool + sample-restricted qrels (so the
   evaluation is apples-to-apples with the W4 dense numbers).
3. Resolve passage texts via ``ir_datasets``' docs_store, one chunk at
   a time so memory stays bounded.
4. Score (query, passage) pairs with ``cross-encoder/ms-marco-MiniLM-L-6-v2``,
   chunk-by-chunk; append each chunk's reranked block to ``run.tsv`` and
   ``flush()`` so a SIGKILL between chunks loses at most one chunk.
5. ``--resume`` reads the existing ``run.tsv``, identifies qids that
   already have a complete top-K block, prunes any half-written ones,
   and only scores the remainder. Mirrors the W2 BM25 resume pattern.
6. Evaluate dense (input) and dense+CE (reranked) on the SAME qrels and
   the SAME query set, so the delta is purely the reranker effect.
7. Persist:
   - ``<output-dir>/metrics.json``   (dense vs rerank deltas)
   - ``<output-dir>/run.tsv``        (reranked TREC run, append-built)
   - ``<output-dir>/examples.jsonl`` (before/after per query)
   - ``<output-dir>/manifest.json``  (git/config/dep hashes + extras)

Usage::

    # default 1,000-query CPU subsample, output at outputs/week05_reranker/
    python experiments/run_reranker.py --num-eval-queries 1000

    # full dev/small, resume-safe, distinct output dir so the 1k-query
    # historical result stays untouched
    OMP_NUM_THREADS=12 python experiments/run_reranker.py \\
        --output-dir outputs/week05_reranker_full \\
        --resume

The W4 dense run is the source of truth; we never re-encode the corpus
or rebuild FAISS here.
"""

from __future__ import annotations

# Mirror W4: keep faiss/torch libomp from clashing on macOS even though
# this script doesn't itself touch faiss — the dense W4 outputs may have
# left state behind, and downstream torch loads inherit the env.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import random
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.data.msmarco import get_docs_store, load_msmarco_passage
from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.reranking.cross_encoder import CrossEncoderReranker
from msmarco_genqa.reranking.io import (
    append_run_tsv,
    collect_unique_doc_ids,
    prune_partial_qids,
    read_done_qids,
    read_run_tsv,
    truncate_top_k,
)
from msmarco_genqa.util.environment import capture_environment
from msmarco_genqa.util.manifest import (
    compute_data_fingerprint,
    compute_env_fingerprint,
    compute_resolved_config_hash,
    write_resolved_config,
    write_run_manifest,
)
from msmarco_genqa.util.seeding import set_global_seed

logger = logging.getLogger("run_reranker")


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
        "--input-run",
        type=Path,
        default=PROJECT_ROOT / "outputs/week04_dense/run.tsv",
        help="First-stage run.tsv to rerank. Defaults to the W4 dense run.",
    )
    parser.add_argument(
        "--input-week",
        type=str,
        default="week04_dense",
        help="Output dir name of the input retriever (used to find sample_doc_ids.json).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Rerank depth K. Defaults to reranker.rerank_top_k from config.",
    )
    parser.add_argument(
        "--num-eval-queries",
        type=int,
        default=None,
        help="Subsample queries (deterministic). Default: use all.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override reranker.model_name from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override reranker.batch_size from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for run.tsv / metrics.json / examples.jsonl / "
            "manifest.json. Defaults to cfg['reranker']['output_dir']. Pass an "
            "alternate path to avoid overwriting a prior reranker run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip queries that already have a complete top-K block in the "
            "output run.tsv and append only the missing ones. Without "
            "--resume, run.tsv is truncated at the start of the run."
        ),
    )
    parser.add_argument(
        "--rerank-chunk-size",
        type=int,
        default=None,
        help=(
            "Number of queries to score per chunk before flushing the "
            "appended block to run.tsv. Default: cfg['reranker']['chunk_size'] "
            "(falls back to 200). Smaller chunks = more durable but more I/O."
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
    parser.add_argument(
        "--allow-incomplete-manifest",
        action="store_true",
        help=(
            "Bypass the schema-v2 required-field contract on manifest write. "
            "Development-only escape hatch; production / headline runs must "
            "leave this off so missing reproducibility fields fail loudly."
        ),
    )
    return parser.parse_args()


def _peak_memory_mb() -> float:
    """Return peak RSS of the current process, in MiB.

    ``resource.getrusage`` reports bytes on macOS, kilobytes on Linux.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _resolve_passages(doc_ids: list[str], docs_store) -> dict[str, str]:
    """Look up passage text for each doc_id; missing → empty string."""
    n_missing = 0
    out: dict[str, str] = {}
    for d in doc_ids:
        try:
            out[d] = docs_store.get(d).text
        except KeyError:
            out[d] = ""
            n_missing += 1
    if n_missing:
        logger.warning("%d / %d doc_ids missing from docs_store", n_missing, len(doc_ids))
    return out


def _load_sample_qrels(input_week_dir: Path, all_qrels: dict[str, set[str]]) -> dict[str, set[str]]:
    """Restrict qrels to the W4 sample (apples-to-apples eval).

    If ``sample_doc_ids.json`` exists in the input week's output dir we
    use it; otherwise we fall back to the full qrels — this lets the
    runner also work against the W2 full-corpus run.
    """
    sample_path = input_week_dir / "sample_doc_ids.json"
    if not sample_path.exists():
        logger.info(
            "No sample_doc_ids.json at %s; using full qrels (assume non-sampled run).",
            sample_path,
        )
        return all_qrels
    with open(sample_path) as f:
        sample_doc_ids = set(json.load(f))
    sample_qrels = {
        q: {d for d in rel if d in sample_doc_ids}
        for q, rel in all_qrels.items()
    }
    return {q: r for q, r in sample_qrels.items() if r}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seed_coverage = set_global_seed(seed)

    rerank_cfg = cfg.get("reranker", {})
    model_name = args.model_name or rerank_cfg.get(
        "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_k = int(args.rerank_top_k or rerank_cfg.get("rerank_top_k", 100))
    batch_size = int(args.batch_size or rerank_cfg.get("batch_size", 64))
    max_length = int(rerank_cfg.get("max_length", 512))
    cfg_output_dir = rerank_cfg.get("output_dir", "outputs/week05_reranker")
    if args.output_dir is not None:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT / args.output_dir
        )
    else:
        output_dir = PROJECT_ROOT / cfg_output_dir
    chunk_size = int(
        args.rerank_chunk_size
        or rerank_cfg.get("chunk_size", 200)
    )
    n_examples_to_save = int(rerank_cfg.get("n_examples", 20))
    cache_dir = PROJECT_ROOT / cfg["data"].get("cache_dir", "data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_run_path = args.input_run
    if not input_run_path.exists():
        raise SystemExit(
            f"Input run not found at {input_run_path}.\n"
            "Run experiments/run_dense_retrieval.py first (or pass --input-run)."
        )
    input_week_dir = PROJECT_ROOT / "outputs" / args.input_week

    # ---------------------------------------------------------------- #
    # 1. Load first-stage run + truncate to top-K
    # ---------------------------------------------------------------- #
    logger.info("Reading first-stage run from %s ...", input_run_path)
    t0 = time.time()
    full_runs = read_run_tsv(input_run_path)
    logger.info(
        "Read %d queries in %.1f s.", len(full_runs), time.time() - t0
    )
    runs_topk = truncate_top_k(full_runs, rerank_top_k)
    logger.info("Truncated to top-%d per query.", rerank_top_k)

    # Optional query subsample (deterministic).
    if args.num_eval_queries is not None and args.num_eval_queries < len(runs_topk):
        rng = random.Random(seed)
        subsample = sorted(rng.sample(list(runs_topk.keys()), args.num_eval_queries))
        runs_topk = {q: runs_topk[q] for q in subsample}
        logger.info("Subsampled to %d eval queries (seed=%d).", len(runs_topk), seed)

    # ---------------------------------------------------------------- #
    # 2. Queries + qrels
    # ---------------------------------------------------------------- #
    data = load_msmarco_passage(cache_dir=cache_dir, load_corpus=False)
    docs_store = data.docs_store or get_docs_store(cache_dir=cache_dir)
    sample_qrels = _load_sample_qrels(input_week_dir, data.qrels)

    # Restrict to queries that have at least one positive qrel AND have a
    # text in the queries map. ``runs_topk`` already only contains queries
    # the first-stage retrieved for, so the intersection is what matters.
    eval_qids = [
        q for q in runs_topk
        if q in sample_qrels and q in data.queries
    ]
    skipped = len(runs_topk) - len(eval_qids)
    if skipped:
        logger.info(
            "Skipping %d / %d queries with no qrel in eval set.",
            skipped,
            len(runs_topk),
        )
    runs_topk = {q: runs_topk[q] for q in eval_qids}
    logger.info("Will rerank %d queries × top-%d.", len(eval_qids), rerank_top_k)

    # ---------------------------------------------------------------- #
    # 3. Resume bookkeeping: which eval qids are already on disk, fully?
    # ---------------------------------------------------------------- #
    rerank_run_path = output_dir / "run.tsv"
    eval_qids_set = set(eval_qids)
    if args.resume:
        on_disk_done = read_done_qids(rerank_run_path, top_k=rerank_top_k)
        # Only count "done" qids that are part of the CURRENT eval set; an
        # output dir reused with a different --num-eval-queries shouldn't
        # silently inherit unrelated entries.
        done_qids = on_disk_done & eval_qids_set
        if on_disk_done and not done_qids:
            logger.warning(
                "Resume: %d done qids on disk but none overlap the current "
                "eval set — was the eval-set selection changed? Starting fresh.",
                len(on_disk_done),
            )
        if done_qids:
            dropped = prune_partial_qids(rerank_run_path, keep_qids=done_qids)
            logger.info(
                "Resume: %d / %d eval qids already complete on disk "
                "(pruned %d half-written lines).",
                len(done_qids),
                len(eval_qids),
                dropped,
            )
        else:
            logger.info("Resume requested but no complete entries — fresh start.")
    else:
        # Fresh run: truncate any prior file so we don't re-evaluate stale lines.
        if rerank_run_path.exists():
            logger.info(
                "Truncating existing %s (pass --resume to keep it).",
                rerank_run_path,
            )
            rerank_run_path.unlink()
        done_qids = set()

    pending_qids = [q for q in eval_qids if q not in done_qids]
    logger.info(
        "Will rerank %d pending queries × top-%d in chunks of %d.",
        len(pending_qids),
        rerank_top_k,
        chunk_size,
    )

    # ---------------------------------------------------------------- #
    # 4. Chunked rerank loop (append + flush after each chunk)
    # ---------------------------------------------------------------- #
    reranker = CrossEncoderReranker(
        model_name=model_name,
        revision=rerank_cfg.get("revision"),
        device=rerank_cfg.get("device"),
        batch_size=batch_size,
        max_length=max_length,
    )

    total_pairs_scored = 0
    score_seconds_total = 0.0
    resolve_seconds = 0.0
    chunks_done = 0
    t_wall = time.time()

    for chunk_start in range(0, len(pending_qids), chunk_size):
        chunk_qids = pending_qids[chunk_start : chunk_start + chunk_size]

        # Resolve passages JUST for this chunk's candidates. Memory stays
        # bounded at O(chunk_size × rerank_top_k) text strings.
        chunk_runs = {q: runs_topk[q] for q in chunk_qids}
        needed = collect_unique_doc_ids(chunk_runs)
        t0 = time.time()
        text_by_id = _resolve_passages(needed, docs_store)
        resolve_seconds += time.time() - t0

        queries_text_chunk = [data.queries[q] for q in chunk_qids]
        candidates_chunk = [
            [(d, text_by_id[d]) for d, _ in chunk_runs[q]]
            for q in chunk_qids
        ]

        chunk_reranked, chunk_info = reranker.rerank_batch(
            queries_text_chunk,
            candidates_chunk,
            show_progress_bar=False,
        )

        # Append the chunk's reranked block to run.tsv and flush.
        append_run_tsv(
            rerank_run_path,
            chunk_qids,
            [[d for d, _ in row] for row in chunk_reranked],
            [[s for _, s in row] for row in chunk_reranked],
            system_name="dense+ce_minilm_l6",
        )

        total_pairs_scored += chunk_info["n_pairs"]
        score_seconds_total += chunk_info["score_seconds"]
        chunks_done += 1
        elapsed = time.time() - t_wall
        done_so_far = chunk_start + len(chunk_qids)
        pps = total_pairs_scored / max(elapsed, 1e-6)
        remaining = len(pending_qids) - done_so_far
        eta_min = (remaining * rerank_top_k) / max(pps, 1e-6) / 60.0
        logger.info(
            "chunk %d: %d queries (%d pairs) in %.1fs; progress %d / %d "
            "(%.0f pairs/s overall, ETA %.0f min).",
            chunks_done,
            len(chunk_qids),
            chunk_info["n_pairs"],
            chunk_info["score_seconds"],
            done_so_far,
            len(pending_qids),
            pps,
            eta_min,
        )

    rerank_wall_seconds = time.time() - t_wall
    peak_mem_mb = _peak_memory_mb()
    logger.info(
        "Reranked %d pending queries × top-%d (%d pairs) in %.1f s "
        "(%.1f q/s, %.0f pairs/s; peak RSS %.0f MiB).",
        len(pending_qids),
        rerank_top_k,
        total_pairs_scored,
        rerank_wall_seconds,
        len(pending_qids) / max(rerank_wall_seconds, 1e-6),
        total_pairs_scored / max(rerank_wall_seconds, 1e-6),
        peak_mem_mb,
    )

    # ---------------------------------------------------------------- #
    # 5. Materialise the (resumed + freshly-written) run.tsv from disk
    # ---------------------------------------------------------------- #
    logger.info("Reading reranked run from %s for evaluation...", rerank_run_path)
    final_runs = read_run_tsv(rerank_run_path)
    rerank_runs_eval: dict[str, list[str]] = {
        q: [d for d, _ in final_runs.get(q, [])] for q in eval_qids
    }
    missing_after_load = [q for q in eval_qids if not rerank_runs_eval[q]]
    if missing_after_load:
        logger.warning(
            "%d qids missing from run.tsv after rerank loop — partial state? "
            "These will not contribute to metrics.",
            len(missing_after_load),
        )

    # ---------------------------------------------------------------- #
    # 6. Evaluate dense (input order) AND reranked, on the same query set
    # ---------------------------------------------------------------- #
    dense_runs_eval = {q: [d for d, _ in runs_topk[q]] for q in eval_qids}

    dense_metrics = evaluate_retrieval(dense_runs_eval, sample_qrels)
    rerank_metrics = evaluate_retrieval(rerank_runs_eval, sample_qrels)
    logger.info("Dense (input)   metrics: %s", dense_metrics)
    logger.info("Dense + CE rerank metrics: %s", rerank_metrics)

    # ---------------------------------------------------------------- #
    # 7. Qualitative examples (before vs after) — reads reranked from disk
    # ---------------------------------------------------------------- #
    rng = random.Random(seed + 2)  # different stream than the W4 examples
    eligible = sorted(eval_qids)
    sample_qids_for_examples = rng.sample(
        eligible, min(n_examples_to_save, len(eligible))
    )

    def _block(doc_score_pairs, relevant: set[str], n: int = 10):
        return [
            {
                "doc_id": d,
                "rank": j + 1,
                "score": float(s),
                "is_relevant": d in relevant,
            }
            for j, (d, s) in enumerate(doc_score_pairs[:n])
        ]

    def _first_rank(block):
        return next((r["rank"] for r in block if r["is_relevant"]), None)

    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid in sample_qids_for_examples:
            relevant = sample_qrels.get(qid, set())
            dense_block = _block(runs_topk[qid], relevant)
            # final_runs[qid] is the disk-materialised reranked block (doc, score).
            ce_block = _block(final_runs.get(qid, []), relevant)
            entry = {
                "query_id": qid,
                "query": data.queries[qid],
                "relevant_doc_ids": sorted(relevant),
                "dense_top10": dense_block,
                "rerank_top10": ce_block,
                "dense_first_rank_in_top10": _first_rank(dense_block),
                "rerank_first_rank_in_top10": _first_rank(ce_block),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote %d qualitative examples to %s",
        len(sample_qids_for_examples),
        examples_path,
    )

    # ---------------------------------------------------------------- #
    # 8. metrics.json (unified schema)
    # ---------------------------------------------------------------- #
    n_queries = dense_metrics.pop("n_queries", None)
    rerank_metrics.pop("n_queries", None)

    payload = {
        "task": "reranking",
        "dataset": "msmarco-passage/dev/small (qrels-anchored sample, via W4 dense run)",
        "n_examples": n_queries,
        "config": cfg,
        "rerank": {
            "input_run": str(input_run_path.relative_to(PROJECT_ROOT))
            if input_run_path.is_relative_to(PROJECT_ROOT)
            else str(input_run_path),
            "input_week": args.input_week,
            "rerank_top_k": rerank_top_k,
            "model_name": model_name,
            "batch_size": batch_size,
            "max_length": max_length,
        },
        "metrics": {
            "dense": dense_metrics,
            "rerank": rerank_metrics,
        },
        "wall_clock_seconds": {
            "resolve_passages": resolve_seconds,
            "rerank": rerank_wall_seconds,
            "score_only": score_seconds_total,
        },
        "throughput": {
            "queries_per_sec": len(pending_qids) / max(rerank_wall_seconds, 1e-6),
            "pairs_per_sec": total_pairs_scored / max(rerank_wall_seconds, 1e-6),
            "n_pairs": total_pairs_scored,
        },
        "resume": {
            "enabled": bool(args.resume),
            "chunk_size": chunk_size,
            "n_resumed_qids": len(done_qids),
            "n_chunks_this_run": chunks_done,
        },
        "peak_memory_mib": peak_mem_mb,
        "environment": (env_dict := capture_environment()),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    resolved_config_path = write_resolved_config(cfg, output_dir)
    resolved_config_hash = compute_resolved_config_hash(cfg)
    data_fingerprint = compute_data_fingerprint(
        cache_dir=cache_dir,
        extra_files={"input_run": input_run_path},
    )
    env_fingerprint = compute_env_fingerprint(env_dict)

    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv,
        config_path=args.config,
        extra_outputs=[rerank_run_path, examples_path, resolved_config_path],
        extra={
            "task": "reranking",
            "model_name": model_name,
            "rerank_top_k": rerank_top_k,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "n_eval_queries": len(eval_qids),
            "n_pending_this_run": len(pending_qids),
            "n_resumed_qids": len(done_qids),
            "n_pairs_this_run": total_pairs_scored,
            "input_run": str(input_run_path.relative_to(PROJECT_ROOT))
            if input_run_path.is_relative_to(PROJECT_ROOT)
            else str(input_run_path),
            "resumed": bool(args.resume) and len(done_qids) > 0,
            "seed": seed,
            "seed_coverage": seed_coverage,
            "resolved_config_hash": resolved_config_hash,
            "data_fingerprint": data_fingerprint,
            "env_fingerprint": env_fingerprint,
        },
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
    )

    # ---------------------------------------------------------------- #
    # 9. Friendly summary
    # ---------------------------------------------------------------- #
    print("\n=== Week 5 cross-encoder reranking ===")
    print(f"input run:   {input_run_path}")
    print(f"rerank model: {model_name}")
    print(
        f"queries:     {len(eval_qids)} eval  ({len(done_qids)} resumed, "
        f"{len(pending_qids)} fresh this run)  |  rerank top-K: {rerank_top_k}"
    )
    if pending_qids:
        print(
            f"throughput:  {len(pending_qids) / max(rerank_wall_seconds, 1e-6):.2f} q/s, "
            f"{total_pairs_scored / max(rerank_wall_seconds, 1e-6):.0f} pairs/s"
        )
        print(
            f"wall clock:  {rerank_wall_seconds:.1f} s rerank "
            f"(+ {resolve_seconds:.1f} s text resolve)"
        )
    print(f"peak RSS:    {peak_mem_mb:.0f} MiB")
    print(f"  {'metric':14s}  {'dense':>10s}  {'+CE':>10s}  {'Δ':>9s}")
    for key in ("mrr@10", "ndcg@10", "recall@100", "recall@1000"):
        d = dense_metrics.get(key)
        r = rerank_metrics.get(key)
        delta = (r - d) if (d is not None and r is not None) else None
        d_s = f"{d:.4f}" if d is not None else "  —  "
        r_s = f"{r:.4f}" if r is not None else "  —  "
        delta_s = f"{delta:+.4f}" if delta is not None else "   —  "
        print(f"  {key:14s}  {d_s:>10s}  {r_s:>10s}  {delta_s:>9s}")
    print(f"outputs:     {output_dir}")


if __name__ == "__main__":
    # Same hard-exit reasoning as run_dense_retrieval.py: torch + faiss
    # libomp interactions on macOS can wedge the interpreter shutdown.
    # Even though this script doesn't import faiss directly, sentence-
    # transformers and ir_datasets together can pin a worker thread.
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

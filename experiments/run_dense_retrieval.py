"""Dense retrieval baseline on a *sampled* MS MARCO sub-corpus.

Pipeline:

1. Load dev/small queries + qrels via ``ir_datasets`` (no eager full corpus).
2. Build a qrels-anchored sample of ``sample_size`` doc_ids — every dev
   relevant doc is included, then random distractors fill to size.
3. Resolve those doc_ids to passage texts via ir_datasets' ``docs_store``.
4. Build (or load) a FAISS dense index over normalised SBERT embeddings.
5. Build (or load) a parallel ``bm25s`` index over the SAME sample, so
   BM25-vs-dense is a head-to-head comparison on the same restricted pool.
6. Retrieve top-K from both and evaluate MRR@10 / nDCG@10 / Recall@100,1000.
7. Persist:
   - ``outputs/dense_retrieval/metrics.json``  (unified schema, both retrievers)
   - ``outputs/dense_retrieval/run.tsv``       (dense run, TREC format)
   - ``outputs/dense_retrieval/run_bm25_sample.tsv`` (BM25 on sample)
   - ``outputs/dense_retrieval/examples.jsonl``

Usage::

    python experiments/run_dense_retrieval.py
    python experiments/run_dense_retrieval.py --sample-size 30000
    python experiments/run_dense_retrieval.py --rebuild-index

The numbers produced here are NOT comparable to the BM25 full-corpus
baseline. The qrels-anchored sample makes the relevant doc always present
in the pool, which inflates absolute retrieval metrics. The valid
comparison is BM25-on-sample vs dense-on-sample.
"""

from __future__ import annotations

# macOS libomp workaround: faiss-cpu and torch each ship their own libomp,
# and loading both in the same process aborts with a duplicate-symbol error.
# Must be set BEFORE any import that pulls in faiss or torch.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.data.msmarco import get_docs_store, load_msmarco_passage
from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.retrieval.bm25 import BM25Retriever
from msmarco_genqa.retrieval.dense import DenseRetriever
from msmarco_genqa.retrieval.query_transform import materialize_query_transform
from msmarco_genqa.retrieval.sampling import qrels_anchored_sample
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

logger = logging.getLogger("run_dense_retrieval")


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
        help="Force a fresh dense (and BM25-sample) index even if cached ones exist.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the sample size from the config.",
    )
    parser.add_argument(
        "--no-bm25-comparison",
        action="store_true",
        help="Skip the BM25-on-sample comparison (dense only).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override ``dense.model_name`` from the config. Used by the "
            "same-tier encoder comparison (bge-small-en-v1.5, "
            "all-MiniLM-L12-v2, …)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the output directory. Defaults to ``outputs/dense_retrieval``. "
            "Pass a fresh path per-encoder so same-tier encoder comparison "
            "runs don't collide."
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


def _load_pool_doc_ids(project_root: Path) -> list[str]:
    """Return the universe of doc_ids to sample from.

    We reuse the ``doc_ids.json`` already produced by the BM25 index
    build (96 MB) so we don't have to iterate the full corpus a second time.
    """
    cached = project_root / "data/processed/bm25_index_msmarco/doc_ids.json"
    if cached.exists():
        logger.info("Reusing pool doc_ids from %s", cached)
        with open(cached) as f:
            return json.load(f)
    raise SystemExit(
        f"Pool doc_ids not found at {cached}.\n"
        "Run experiments/run_retrieval.py first to populate the BM25 index "
        "(this script reuses its doc_ids.json instead of re-iterating the "
        "full 8.8M-passage corpus)."
    )


def _resolve_passages(doc_ids: list[str], docs_store) -> list[str]:
    """Look up passage text for each doc_id via ir_datasets' docs_store."""
    n_missing = 0
    texts = []
    for d in doc_ids:
        try:
            texts.append(docs_store.get(d).text)
        except KeyError:
            texts.append("")
            n_missing += 1
    if n_missing:
        logger.warning("%d / %d doc_ids missing from docs_store", n_missing, len(doc_ids))
    return texts


def _write_run_tsv(path: Path, qids: list[str], doc_ids_lists: list[list[str]],
                   scores, system_name: str) -> None:
    with open(path, "w") as f:
        for qid, docs, score_row in zip(qids, doc_ids_lists, scores):
            for rank, (d, s) in enumerate(zip(docs, score_row), 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{float(s):.6f}\t{system_name}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seed_coverage = set_global_seed(seed)

    dense_cfg = cfg["dense"]
    sample_size = args.sample_size or int(dense_cfg["sample_size"])
    # CLI overrides for the same-tier encoder comparison + qrel-density
    # sweep: model_name + output dir + auto-keyed FAISS index dir.
    # The cached index is invalidated by EITHER a different encoder OR a
    # different sample size, so we key the dir on (model_safe, sample_size)
    # whenever --model-name is overridden. The dense baseline (50k MiniLM-L6,
    # written via the unmodified default code path) is left untouched.
    if args.model_name is not None:
        dense_cfg["model_name"] = args.model_name
        safe = args.model_name.replace("/", "_").replace(":", "_")
        dense_cfg["index_dir"] = (
            f"data/processed/dense_index_{safe}_n{sample_size}"
        )
        # BM25-on-sample is a function of sample_size only (not model_name);
        # auto-key the cache dir on sample_size whenever the dense override
        # is in play so the qrel-density sweep doesn't collide with the
        # dense baseline 50k BM25-sample index.
        dense_cfg["bm25_sample_index_dir"] = (
            f"data/processed/bm25_sample_index_n{sample_size}"
        )
    if args.output_dir is not None:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT / args.output_dir
        )
    else:
        output_dir = PROJECT_ROOT / dense_cfg["output_dir"]
    cache_dir = PROJECT_ROOT / cfg["data"].get("cache_dir", "data/raw")
    dense_index_dir = PROJECT_ROOT / dense_cfg["index_dir"]
    bm25_sample_index_dir = PROJECT_ROOT / dense_cfg.get(
        "bm25_sample_index_dir", "data/processed/bm25_sample_index"
    )
    top_k = int(dense_cfg.get("top_k", 1000))
    do_bm25 = (not args.no_bm25_comparison) and dense_cfg.get("compare_bm25_on_sample", True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Data
    # ---------------------------------------------------------------- #
    data = load_msmarco_passage(cache_dir=cache_dir, load_corpus=False)
    docs_store = data.docs_store or get_docs_store(cache_dir=cache_dir)
    query_text_by_qid, query_transform_summary, query_transform_outputs = (
        materialize_query_transform(
            data.queries,
            cfg.get("query_transform"),
            output_dir=output_dir / "query_transform",
        )
    )
    if query_transform_summary["method"] != "none":
        logger.info(
            "Query transformation %s changed %d / %d queries.",
            query_transform_summary["method"],
            query_transform_summary["n_changed"],
            query_transform_summary["n_queries"],
        )

    pool_doc_ids = _load_pool_doc_ids(PROJECT_ROOT)
    sample_doc_ids = qrels_anchored_sample(
        pool_doc_ids=pool_doc_ids,
        qrels=data.qrels,
        target_size=sample_size,
        seed=seed,
    )

    # Persist the sampled doc_id list for auditability.
    sample_path = output_dir / "sample_doc_ids.json"
    with open(sample_path, "w") as f:
        json.dump(sample_doc_ids, f)
    logger.info("Wrote sampled doc_ids to %s (%d docs)", sample_path, len(sample_doc_ids))

    sample_set = set(sample_doc_ids)
    sample_qrels = {
        q: {d for d in rel if d in sample_set}
        for q, rel in data.qrels.items()
    }
    sample_qrels = {q: r for q, r in sample_qrels.items() if r}
    logger.info(
        "After restricting qrels to sampled docs: %d / %d queries still have ≥1 relevant.",
        len(sample_qrels),
        len(data.qrels),
    )

    # We will only need passage texts if we have to (re)build at least one
    # of the indexes; resolving 50k docs from the docs_store is the slowest
    # step besides encoding.
    have_dense = (dense_index_dir / "config.json").exists() and not args.rebuild_index
    have_bm25_sample = (
        do_bm25
        and (bm25_sample_index_dir / "config.json").exists()
        and not args.rebuild_index
    )
    sample_texts: list[str] | None = None
    if not have_dense or (do_bm25 and not have_bm25_sample):
        logger.info("Resolving %d sampled passages from docs_store...", len(sample_doc_ids))
        t0 = time.time()
        sample_texts = _resolve_passages(sample_doc_ids, docs_store)
        logger.info("Resolved in %.1f s.", time.time() - t0)

    # ---------------------------------------------------------------- #
    # 2. Dense index
    # ---------------------------------------------------------------- #
    expected_dense_cfg = {
        "model_name": dense_cfg["model_name"],
        "normalize": True,
    }
    encode_seconds: float | None = None
    if have_dense:
        dense = DenseRetriever.load(
            dense_index_dir,
            device=dense_cfg.get("device"),
            encode_batch_size=int(dense_cfg.get("encode_batch_size", 32)),
            expected_config=expected_dense_cfg,
        )
        # Validate that the cached index covers our exact sample.
        if set(dense.doc_ids) != sample_set:
            raise SystemExit(
                f"Cached dense index at {dense_index_dir} covers {len(dense.doc_ids)} "
                f"doc_ids but the current sample has {len(sample_set)}. They must match. "
                "Pass --rebuild-index to rebuild."
            )
    else:
        dense = DenseRetriever(
            model_name=dense_cfg["model_name"],
            revision=dense_cfg.get("revision"),
            device=dense_cfg.get("device"),
            encode_batch_size=int(dense_cfg.get("encode_batch_size", 32)),
            normalize=True,
        )
        t0 = time.time()
        dense.build(sample_texts, sample_doc_ids)
        encode_seconds = time.time() - t0
        dense.save(dense_index_dir)

    # ---------------------------------------------------------------- #
    # 3. (Optional) BM25 on the same sample
    # ---------------------------------------------------------------- #
    bm25_build_seconds: float | None = None
    bm25 = None
    if do_bm25:
        if have_bm25_sample:
            expected_bm25 = {
                "k1": float(cfg["retrieval"]["k1"]),
                "b": float(cfg["retrieval"]["b"]),
                "stopwords": cfg["retrieval"].get("stopwords", "en"),
            }
            bm25 = BM25Retriever.load(
                bm25_sample_index_dir,
                expected_config=expected_bm25,
            )
            if set(bm25.doc_ids) != sample_set:
                raise SystemExit(
                    f"Cached BM25-sample index covers a different sample "
                    f"({len(bm25.doc_ids)} docs) than the current ({len(sample_set)}). "
                    "Pass --rebuild-index."
                )
        else:
            bm25 = BM25Retriever(
                corpus_texts=sample_texts,
                doc_ids=sample_doc_ids,
                k1=float(cfg["retrieval"]["k1"]),
                b=float(cfg["retrieval"]["b"]),
                stopwords=cfg["retrieval"].get("stopwords", "en"),
            )
            t0 = time.time()
            bm25.build()
            bm25_build_seconds = time.time() - t0
            bm25.save(bm25_sample_index_dir)

    # ---------------------------------------------------------------- #
    # 4. Retrieval
    # ---------------------------------------------------------------- #
    qids = list(data.queries.keys())
    queries_text = [query_text_by_qid[q] for q in qids]
    top_k_eff = min(top_k, len(sample_doc_ids))

    logger.info("Dense retrieval: top-%d for %d queries...", top_k_eff, len(qids))
    t0 = time.time()
    dense_scores, dense_doc_ids_lists = dense.retrieve_batch(queries_text, k=top_k_eff)
    dense_search_seconds = time.time() - t0
    logger.info(
        "Dense done in %.1f s (%.1f ms/query)",
        dense_search_seconds,
        dense_search_seconds * 1000 / max(len(qids), 1),
    )

    bm25_scores = None
    bm25_doc_ids_lists = None
    bm25_search_seconds: float | None = None
    if bm25 is not None:
        logger.info("BM25-on-sample retrieval: top-%d for %d queries...", top_k_eff, len(qids))
        t0 = time.time()
        bm25_scores, bm25_doc_ids_lists = bm25.retrieve_batch(queries_text, k=top_k_eff)
        bm25_search_seconds = time.time() - t0
        logger.info(
            "BM25-on-sample done in %.1f s (%.1f ms/query)",
            bm25_search_seconds,
            bm25_search_seconds * 1000 / max(len(qids), 1),
        )

    # ---------------------------------------------------------------- #
    # 5. Persist run files
    # ---------------------------------------------------------------- #
    dense_run_path = output_dir / "run.tsv"
    _write_run_tsv(dense_run_path, qids, dense_doc_ids_lists, dense_scores, "dense")
    logger.info("Wrote dense run to %s", dense_run_path)

    bm25_run_path: Path | None = None
    if bm25 is not None:
        bm25_run_path = output_dir / "run_bm25_sample.tsv"
        _write_run_tsv(bm25_run_path, qids, bm25_doc_ids_lists, bm25_scores, "bm25_sample")
        logger.info("Wrote BM25-on-sample run to %s", bm25_run_path)

    # ---------------------------------------------------------------- #
    # 6. Evaluate (against the *sample-restricted* qrels)
    # ---------------------------------------------------------------- #
    dense_runs = {q: docs for q, docs in zip(qids, dense_doc_ids_lists)}
    dense_metrics = evaluate_retrieval(dense_runs, sample_qrels)
    logger.info("Dense metrics: %s", dense_metrics)

    bm25_metrics = None
    if bm25 is not None:
        bm25_runs = {q: docs for q, docs in zip(qids, bm25_doc_ids_lists)}
        bm25_metrics = evaluate_retrieval(bm25_runs, sample_qrels)
        logger.info("BM25-on-sample metrics: %s", bm25_metrics)

    # ---------------------------------------------------------------- #
    # 7. Qualitative examples
    # ---------------------------------------------------------------- #
    rng = random.Random(seed + 1)  # different stream than the sampler
    eligible = sorted(sample_qrels.keys())
    n_examples = int(dense_cfg.get("n_examples", 20))
    sample_qids_for_examples = rng.sample(eligible, min(n_examples, len(eligible)))
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    def _top10_block(scores_arr, doc_ids_lists_local, qid: str, relevant: set[str]):
        i = qid_to_idx[qid]
        return [
            {
                "doc_id": d,
                "rank": j + 1,
                "score": float(scores_arr[i][j]),
                "is_relevant": d in relevant,
            }
            for j, d in enumerate(doc_ids_lists_local[i][:10])
        ]

    def _first_rank(block):
        return next((r["rank"] for r in block if r["is_relevant"]), None)

    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid in sample_qids_for_examples:
            relevant = sample_qrels.get(qid, set())
            dense_block = _top10_block(dense_scores, dense_doc_ids_lists, qid, relevant)
            entry = {
                "query_id": qid,
                "query": data.queries[qid],
                "relevant_doc_ids": sorted(relevant),
                "dense_top10": dense_block,
                "dense_first_rank_in_top10": _first_rank(dense_block),
            }
            if query_transform_summary["method"] != "none":
                entry["transformed_query"] = query_text_by_qid[qid]
            if bm25 is not None:
                bm25_block = _top10_block(bm25_scores, bm25_doc_ids_lists, qid, relevant)
                entry["bm25_sample_top10"] = bm25_block
                entry["bm25_sample_first_rank_in_top10"] = _first_rank(bm25_block)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d qualitative examples to %s", len(sample_qids_for_examples), examples_path)

    # ---------------------------------------------------------------- #
    # 8. metrics.json (unified schema)
    # ---------------------------------------------------------------- #
    n_examples_total = dense_metrics.pop("n_queries", None)
    if bm25_metrics is not None:
        bm25_metrics.pop("n_queries", None)

    env_dict = capture_environment()
    sampling_block = compute_sampling_block(
        is_sampled=True,
        method="qrels-anchored",
        sample_size=len(sample_doc_ids),
    )
    payload = {
        "task": "retrieval",
        "dataset": "msmarco-passage/dev/small (qrels-anchored sample)",
        "n_examples": n_examples_total,
        "config": cfg,
        "metrics": {
            "dense": dense_metrics,
            **({"bm25_sample": bm25_metrics} if bm25_metrics is not None else {}),
        },
        "sampling": sampling_block,
        "wall_clock_seconds": {
            "encode_corpus": encode_seconds,
            "dense_search": dense_search_seconds,
            "bm25_sample_build": bm25_build_seconds,
            "bm25_sample_search": bm25_search_seconds,
        },
        "environment": env_dict,
        "sample": {
            "size": len(sample_doc_ids),
            "n_qrels_doc_ids_in_sample": sum(len(v) for v in sample_qrels.values()),
            "n_eval_queries_with_qrels_in_sample": len(sample_qrels),
            "method": "qrels-anchored",
            "doc_ids_path": str(sample_path.relative_to(PROJECT_ROOT)),
        },
        "top_k": top_k_eff,
        "query_transform": query_transform_summary,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    resolved_config_path = write_resolved_config(cfg, output_dir)
    resolved_config_hash = compute_resolved_config_hash(cfg)
    data_fingerprint = compute_data_fingerprint(
        cache_dir=cache_dir,
        extra_files={"sample_doc_ids": sample_path},
    )
    env_fingerprint = compute_env_fingerprint(env_dict)

    manifest_outputs = [
        dense_run_path,
        examples_path,
        sample_path,
        resolved_config_path,
        *query_transform_outputs,
    ]
    if bm25_run_path is not None:
        manifest_outputs.append(bm25_run_path)
    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv,
        config_path=args.config,
        extra_outputs=manifest_outputs,
        extra={
            "task": "dense_retrieval",
            "model_name": dense_cfg["model_name"],
            "sample_size": len(sample_doc_ids),
            "sampling_method": "qrels-anchored",
            "top_k": top_k_eff,
            "n_eval_queries": n_examples_total,
            "compared_against_bm25_sample": bm25 is not None,
            "seed": seed,
            "seed_coverage": seed_coverage,
            "resolved_config_hash": resolved_config_hash,
            "data_fingerprint": data_fingerprint,
            "env_fingerprint": env_fingerprint,
            "query_transform": query_transform_summary,
        },
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
    )

    # ---------------------------------------------------------------- #
    # 9. Friendly summary
    # ---------------------------------------------------------------- #
    print("\n=== dense retrieval (sampled corpus) ===")
    print(f"sample size: {len(sample_doc_ids):,}  |  eval queries: {n_examples_total}")
    print(f"  {'metric':14s}  {'dense':>10s}  {'bm25_sample':>12s}  {'Δ':>9s}")
    for key in ("mrr@10", "ndcg@10", "recall@100", "recall@1000"):
        d = dense_metrics.get(key)
        b = bm25_metrics.get(key) if bm25_metrics else None
        delta = (d - b) if (d is not None and b is not None) else None
        d_s = f"{d:.4f}" if d is not None else "  —  "
        b_s = f"{b:.4f}" if b is not None else "  —  "
        delta_s = f"{delta:+.4f}" if delta is not None else "   —  "
        print(f"  {key:14s}  {d_s:>10s}  {b_s:>12s}  {delta_s:>9s}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    # ``os._exit`` skips the interpreter shutdown path. On macOS with both
    # faiss-cpu and torch loaded, that shutdown wedges in ``pthread_join``
    # trying to reap an OpenMP worker thread owned by the *other* libomp
    # instance (faiss ships libomp.dylib, torch ships libiomp5.dylib).
    # All real work has finished by the time we reach this line, so a hard
    # exit is safe and avoids a 30+ s hang at the end of every run.
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

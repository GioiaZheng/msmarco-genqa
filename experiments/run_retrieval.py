"""Run full-corpus BM25 on MS MARCO dev/small or TREC-DL judged topics.

End-to-end:

1. Load the shared MS MARCO passage corpus and the selected query/qrels set.
2. Build (or load from cache) a ``bm25s`` index over the corpus.
3. Retrieve top-k for every selected query, in chunks; append each chunk to
   ``run.tsv`` so a killed run can be resumed instead of restarting from
   query 0.
4. Compute MRR@10, Recall@100, Recall@1000.
5. Persist:
   - ``outputs/bm25_baseline/metrics.json``
   - ``outputs/bm25_baseline/run.tsv``  (TREC-format run, top-1000 by default)
   - ``outputs/bm25_baseline/examples.jsonl``  (qualitative samples)

Run from the project root::

    python experiments/run_retrieval.py
    python experiments/run_retrieval.py --resume                       # pick up where a killed run stopped
    python experiments/run_retrieval.py --rebuild-index                # force fresh BM25 index
    python experiments/run_retrieval.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.data.benchmark import (
    BEIR_NFCORPUS_TEST,
    MSMARCO_DEV_SMALL,
    SUPPORTED_DATASETS,
    BenchmarkQueries,
    BenchmarkSpec,
    default_retrieval_index_dir,
    default_retrieval_output_dir,
    get_benchmark_spec,
    load_benchmark_corpus,
    load_benchmark_queries,
    lookup_document_text,
)
from msmarco_genqa.data.nfcorpus_video import (
    SUPPORTED_REPRESENTATIONS,
    NFCorpusVideoQueryBundle,
    load_nfcorpus_video_query_representation,
    validate_frozen_title_metrics,
    write_nfcorpus_video_query_artifacts,
)
from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.evaluation.trec import evaluate_trec_retrieval, trec_metric_contract
from msmarco_genqa.retrieval.bm25 import BM25Retriever
from msmarco_genqa.retrieval.query_transform import materialize_query_transform
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

logger = logging.getLogger("run_retrieval")
DEFAULT_NFCORPUS_VIDEO_CONTRACT = (
    PROJECT_ROOT / "configs/nfcorpus_video_query_representation.json"
)


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
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=MSMARCO_DEV_SMALL,
        help="Query/qrels set to run against the shared MS MARCO passage corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override. TREC-DL defaults to isolated, year-specific "
            "directories; dev/small keeps eval_retrieval.output_dir."
        ),
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
    parser.add_argument(
        "--query-representation",
        choices=SUPPORTED_REPRESENTATIONS,
        default=None,
        help=(
            "Run the predeclared 102-query NFCorpus video representation "
            "experiment. Requires --dataset beir/nfcorpus/test and an explicit "
            "--output-dir so the full benchmark output cannot be overwritten."
        ),
    )
    parser.add_argument(
        "--query-representation-contract",
        type=Path,
        default=None,
        help=(
            "Pinned NFCorpus video experiment contract. When "
            "--query-representation is set, defaults to "
            "configs/nfcorpus_video_query_representation.json."
        ),
    )
    parser.add_argument(
        "--no-query-source-download",
        action="store_true",
        help=(
            "Refuse to download the pinned official NFCorpus archive when it "
            "is absent. Integrity checks are always enforced."
        ),
    )
    return parser.parse_args(argv)


def resolve_output_dir(
    args: argparse.Namespace,
    cfg: dict,
    spec: BenchmarkSpec,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    configured = cfg["eval_retrieval"]["output_dir"]
    path = args.output_dir or default_retrieval_output_dir(spec, configured)
    return path if path.is_absolute() else project_root / path


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
    parsed = read_run_tsv(run_path)
    return (
        {qid: [doc_id for doc_id, _score in docs] for qid, docs in parsed.items()},
        {qid: [score for _doc_id, score in docs] for qid, docs in parsed.items()},
    )


def _positive_score_recall(
    runs: dict[str, list[str]],
    scores_by_qid: dict[str, list[float]],
    qrels: dict[str, set[str]],
    *,
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    """Macro recall after excluding non-retrieved (score <= 0) fillers."""

    qids = [qid for qid in qrels if qid in runs]
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        values = []
        for qid in qids:
            positive_docs = [
                doc_id
                for doc_id, score in zip(
                    runs[qid][:cutoff],
                    scores_by_qid.get(qid, [])[:cutoff],
                )
                if score > 0.0
            ]
            relevant = qrels[qid]
            values.append(
                len(set(positive_docs) & relevant) / len(relevant)
                if relevant
                else 0.0
            )
        metrics[f"positive_score_recall@{cutoff}"] = (
            sum(values) / len(values) if values else 0.0
        )
    return metrics


def _index_fingerprint(index_dir: Path) -> dict[str, object]:
    """Hash a small experiment index as an ordered set of files."""

    digest = hashlib.sha256()
    files = sorted(path for path in index_dir.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(index_dir).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{relative}\0{size}\0".encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
        "path": index_dir.relative_to(PROJECT_ROOT).as_posix(),
    }


def _validate_query_representation_args(
    args: argparse.Namespace,
    cfg: dict,
) -> Path | None:
    """Validate the experiment boundary and return its resolved contract path."""

    if args.query_representation is None:
        if args.query_representation_contract is not None:
            raise SystemExit(
                "--query-representation-contract requires --query-representation"
            )
        if args.no_query_source_download:
            raise SystemExit(
                "--no-query-source-download requires --query-representation"
            )
        return None

    if args.dataset != BEIR_NFCORPUS_TEST:
        raise SystemExit(
            "--query-representation is restricted to --dataset beir/nfcorpus/test"
        )
    if args.output_dir is None:
        raise SystemExit(
            "--query-representation requires an explicit --output-dir to avoid "
            "overwriting the full NFCorpus benchmark"
        )
    query_transform_method = str(
        (cfg.get("query_transform") or {}).get("method", "none")
    )
    if query_transform_method != "none":
        raise SystemExit(
            "query_transform.method must remain 'none' for the controlled "
            "NFCorpus query-representation experiment"
        )

    path = args.query_representation_contract or DEFAULT_NFCORPUS_VIDEO_CONTRACT
    return path if path.is_absolute() else PROJECT_ROOT / path


def _select_video_query_cohort(
    benchmark: BenchmarkQueries,
    bundle: NFCorpusVideoQueryBundle,
) -> BenchmarkQueries:
    """Apply a constructed query cohort to judgments after construction."""

    query_ids = set(bundle.queries)
    missing_qrels = sorted(query_ids - set(benchmark.graded_qrels))
    if missing_qrels:
        raise SystemExit(
            f"{len(missing_qrels)} NFCorpus video queries are missing graded qrels"
        )
    return BenchmarkQueries(
        spec=benchmark.spec,
        queries=dict(bundle.queries),
        qrels={qid: set(benchmark.qrels.get(qid, set())) for qid in bundle.queries},
        graded_qrels={
            qid: dict(benchmark.graded_qrels[qid]) for qid in bundle.queries
        },
    )


def _validate_representation_resume(
    output_dir: Path,
    bundle: NFCorpusVideoQueryBundle,
    *,
    resume: bool,
) -> None:
    """Prevent a resumed run from mixing query representations."""

    run_path = output_dir / "run.tsv"
    if not resume or not run_path.exists():
        return
    summary_path = output_dir / "query_representation" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"refusing to resume {run_path}: query-representation summary is missing"
        )
    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"refusing to resume {run_path}: query-representation summary is invalid"
        ) from exc
    expected = bundle.summary
    for key in (
        "representation",
        "qid_sha256",
        "official_query_records_sha256",
        "effective_queries_sha256",
    ):
        if existing.get(key) != expected.get(key):
            raise SystemExit(
                f"refusing to resume {run_path}: query representation {key} differs"
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    benchmark_spec = get_benchmark_spec(args.dataset)
    query_representation_contract = _validate_query_representation_args(args, cfg)
    seed = cfg.get("seed", 42)
    seed_coverage = set_global_seed(seed)

    output_dir = resolve_output_dir(args, cfg, benchmark_spec)
    index_dir_config = default_retrieval_index_dir(
        benchmark_spec,
        cfg["retrieval"]["index_dir"],
    )
    index_dir = (
        index_dir_config
        if index_dir_config.is_absolute()
        else PROJECT_ROOT / index_dir_config
    )
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
    corpus_data = load_benchmark_corpus(
        benchmark_spec,
        cache_dir=cache_dir,
        load_corpus=not have_index,
        limit=corpus_limit,
    )
    benchmark = load_benchmark_queries(
        args.dataset,
        cache_dir=cache_dir,
    )
    query_representation_bundle: NFCorpusVideoQueryBundle | None = None
    query_representation_outputs: list[Path] = []
    query_representation_summary: dict[str, object] = {
        "representation": "benchmark_default",
        "n_queries": len(benchmark.queries),
    }
    if query_representation_contract is not None:
        query_representation_bundle = load_nfcorpus_video_query_representation(
            benchmark.queries,
            representation=args.query_representation,
            contract_path=query_representation_contract,
            project_root=PROJECT_ROOT,
            download_if_missing=not args.no_query_source_download,
        )
        _validate_representation_resume(
            output_dir,
            query_representation_bundle,
            resume=args.resume,
        )
        benchmark = _select_video_query_cohort(
            benchmark,
            query_representation_bundle,
        )
        (
            query_representation_summary,
            query_representation_outputs,
        ) = write_nfcorpus_video_query_artifacts(
            query_representation_bundle,
            output_dir / "query_representation",
        )
        logger.info(
            "NFCorpus video query representation %s: %d validated queries.",
            args.query_representation,
            len(benchmark.queries),
        )
    query_text_by_qid, query_transform_summary, query_transform_outputs = (
        materialize_query_transform(
            benchmark.queries,
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
            corpus_texts=corpus_data.corpus_texts,
            doc_ids=corpus_data.corpus_doc_ids,
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
    if query_representation_bundle is not None:
        query_representation_summary["index_fingerprint"] = _index_fingerprint(
            index_dir
        )

    # ---- 3. Plan retrieval ----
    qids = list(benchmark.queries.keys())
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
    eligible = [q for q in qids if benchmark.qrels.get(q)]
    sample_qids = rng.sample(eligible, min(n_examples, len(eligible)))

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
                chunk_texts = [query_text_by_qid[q] for q in chunk_qids]

                t0 = time.time()
                chunk_scores, chunk_doc_ids = retriever.retrieve_batch(
                    chunk_texts,
                    k=top_k,
                    deterministic_ties=query_representation_bundle is not None,
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

    if benchmark_spec.has_graded_qrels:
        metrics = evaluate_trec_retrieval(
            runs=runs,
            qrels=benchmark.graded_qrels,
            rel_threshold=benchmark_spec.rel_threshold or 1,
            ks_mrr=tuple(ks_mrr),
            ks_recall=tuple(ks_recall),
            ks_ndcg=tuple(ks_ndcg),
        )
    else:
        metrics = evaluate_retrieval(
            runs=runs,
            qrels=benchmark.qrels,
            ks_mrr=ks_mrr,
            ks_recall=ks_recall,
            ks_ndcg=ks_ndcg,
        )
    logger.info("Metrics: %s", metrics)
    if query_representation_bundle is not None:
        positive_score_metrics = _positive_score_recall(
            runs,
            scores_by_qid,
            benchmark.qrels,
            cutoffs=tuple(ks_recall),
        )
        query_representation_summary["positive_score_metrics"] = (
            positive_score_metrics
        )
        query_representation_summary["title_baseline_reproduction"] = (
            validate_frozen_title_metrics(
                query_representation_bundle,
                metrics,
                positive_score_metrics=positive_score_metrics,
            )
        )
        summary_path = output_dir / "query_representation" / "summary.json"
        with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                query_representation_summary,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

    # ---- 6. Qualitative examples ----
    # Resolve passage text. If we built the index in this run, the corpus is
    # in memory; otherwise we use the docs_store for random access.
    if corpus_data.corpus_texts:
        id_to_text = dict(zip(corpus_data.corpus_doc_ids, corpus_data.corpus_texts))

        def get_text(doc_id: str) -> str:
            return id_to_text.get(doc_id, "")
    else:
        store = corpus_data.docs_store or load_benchmark_corpus(
            benchmark_spec,
            cache_dir=cache_dir,
            load_corpus=False,
        ).docs_store

        def get_text(doc_id: str) -> str:
            try:
                return lookup_document_text(store, doc_id)
            except KeyError:
                return ""

    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for qid in sample_qids:
            relevant = benchmark.qrels.get(qid, set())
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
                "query": benchmark.queries[qid],
                "relevant_doc_ids": sorted(relevant),
                "first_relevant_rank_in_top10": first_rank,
                "top_results": top_results,
            }
            if query_transform_summary["method"] != "none":
                example["transformed_query"] = query_text_by_qid[qid]
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(sample_qids), examples_path)

    # ---- 7. metrics.json (unified schema across retrieval and generation) ----
    n_examples = metrics.pop("n_queries", None)  # promote count to top level
    env_dict = capture_environment()
    # BM25 baseline runs the full 8.8M corpus by default. corpus_limit is
    # only set for smoke / dev iteration; when set, it's a first-N truncation
    # (not qrels-anchored — that's the dense runner's pattern).
    sampling_block = compute_sampling_block(
        is_sampled=corpus_limit is not None,
        method="first-N-truncated" if corpus_limit is not None else None,
        sample_size=corpus_limit,
    )
    benchmark_metadata = benchmark.metadata()
    judged_qids = set(benchmark.graded_qrels)
    benchmark_metadata.update(
        {
            "corpus_id": benchmark_spec.corpus_id,
            "corpus_scope": "first-N-truncated" if corpus_limit is not None else "full",
            "run_topic_count": len(set(runs) & set(benchmark.queries)),
            "judged_topic_coverage": (
                len(set(runs) & judged_qids) / len(judged_qids) if judged_qids else 0.0
            ),
            "query_representation": query_representation_summary,
        }
    )
    payload = {
        "task": "retrieval",
        "dataset": benchmark_spec.dataset_id,
        "benchmark": benchmark_metadata,
        "n_examples": n_examples,
        "config": cfg,
        "metrics": metrics,
        "sampling": sampling_block,
        "wall_clock_seconds": {
            "indexing": index_time,
            "search": search_time,
            "search_pending_count": len(pending_qids),
        },
        "environment": env_dict,
        "top_k": top_k,
        "query_transform": query_transform_summary,
        "query_representation": query_representation_summary,
        "resumed": args.resume and bool(set(qids) - set(pending_qids)),
    }
    if benchmark_spec.has_graded_qrels:
        payload["evaluation"] = {
            **trec_metric_contract(
                rel_threshold=benchmark_spec.rel_threshold or 1,
                ks_mrr=tuple(ks_mrr),
                ks_ndcg=tuple(ks_ndcg),
                ks_recall=tuple(ks_recall),
                run_depth=top_k,
            ),
            "qrels_source": benchmark_spec.dataset_id,
            "internal_backend": "msmarco_genqa.evaluation.trec",
        }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")

    resolved_config_path = write_resolved_config(cfg, output_dir)
    resolved_config_hash = compute_resolved_config_hash(cfg)
    data_fingerprint = compute_data_fingerprint(
        cache_dir=cache_dir,
        corpus_limit=corpus_limit,
        data_sources={
            "dataset_id": benchmark_spec.dataset_id,
            "corpus_id": benchmark_spec.corpus_id,
        },
        extra_files=(
            {
                "query_representation_contract": query_representation_bundle.contract_path,
                "query_representation_archive": query_representation_bundle.archive_path,
            }
            if query_representation_bundle is not None
            else None
        ),
    )
    env_fingerprint = compute_env_fingerprint(env_dict)

    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        command=sys.argv,
        config_path=args.config,
        extra_outputs=[
            run_path,
            examples_path,
            resolved_config_path,
            *query_transform_outputs,
            *query_representation_outputs,
        ],
        extra={
            "task": "retrieval",
            "backend": cfg["retrieval"].get("backend", "bm25s"),
            "k1": cfg["retrieval"].get("k1"),
            "b": cfg["retrieval"].get("b"),
            "stopwords": cfg["retrieval"].get("stopwords"),
            "top_k": top_k,
            "n_eval_queries": len(qids),
            "n_metric_queries": n_examples,
            "dataset": benchmark_spec.dataset_id,
            "corpus_id": benchmark_spec.corpus_id,
            "track_year": benchmark_spec.track_year,
            "judged_topic_count": benchmark.judged_topic_count,
            "run_topic_count": len(set(runs) & set(benchmark.queries)),
            "corpus_scope": benchmark_metadata["corpus_scope"],
            "resumed": bool(args.resume and (set(qids) - set(pending_qids))),
            "seed": seed,
            "seed_coverage": seed_coverage,
            "resolved_config_hash": resolved_config_hash,
            "data_fingerprint": data_fingerprint,
            "env_fingerprint": env_fingerprint,
            "query_transform": query_transform_summary,
            "query_representation": query_representation_summary,
        },
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
    )

    # ---- 8. Friendly summary ----
    print(f"\n=== BM25 baseline: {benchmark_spec.dataset_id} ===")
    print(f"queries evaluated: {n_examples}")
    for key in ("mrr@10", "ndcg@10", "recall@100", "recall@1000"):
        if key in metrics:
            print(f"  {key:14s} = {metrics[key]:.4f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()

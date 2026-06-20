"""End-to-end driver for the BM25-vs-reranked generation comparison on the
full eligible dev/small pool.

Five phases, each idempotent and skippable when its outputs already exist:

  0. preflight    — print eligible-pool stats, top-k, retrieval source,
                    expected wall-clock
  1. validate     — assert the full reranked ``run.tsv`` satisfies the
                    invariants (qid count, top-K, no duplicates, manifest
                    has resume/chunking fields)
  2. gen_bm25     — generate answers using BM25 top-3, restricted to the
                    reranker-covered eligible pool
  3. gen_reranked — generate answers using the reranked top-3
  4. analyse      — compute per-query metrics, by-query-type breakdown,
                    bucket assignments, qualitative examples, and a
                    markdown report

Idempotency
-----------
- A phase whose outputs are present and well-formed is **skipped** by
  default (printed as ``SKIP (already done)``). Use ``--force`` to
  re-run everything.
- Before phase 4 the driver asserts that the two prediction files cover
  **exactly the same query_ids**. Mismatch ⇒ hard error (exit 2).

Modes
-----
- ``--dry-run``: tiny sample run (``--num-eval-queries 5``) wired through
  every phase, writing to ``outputs/dryrun/...`` so it can't disturb a
  real artefact. Useful as a wiring smoke test before the long run.

Usage::

    # Real run (after the full reranker completes)
    python scripts/run_full_generation_and_analysis.py

    # Dry run — proves the orchestration end-to-end with 5 queries
    python scripts/run_full_generation_and_analysis.py --dry-run

    # Re-run a single phase
    python scripts/run_full_generation_and_analysis.py --force --only gen_bm25
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("driver")


# --------------------------------------------------------------------------- #
# Config dataclass — every phase consumes one of these
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriverConfig:
    reranker_run: Path
    reranker_manifest: Path
    bm25_run: Path
    bm25_out: Path
    rerank_out: Path
    analysis_out: Path
    log_dir: Path
    n_eval_queries: int
    expected_qids: int
    rerank_top_k: int
    retrieval_source_bm25: str
    retrieval_source_reranked: str
    seconds_per_query_estimate: float
    require_full_manifest: bool
    max_new_tokens: int | None = None


def _suffix(s: str) -> str:
    if not s:
        return ""
    return s if s.startswith("_") else "_" + s


def build_default_config(args: argparse.Namespace) -> DriverConfig:
    """Production defaults: full dev/small via outputs/cross_encoder_rerank_full/.

    ``--out-suffix`` (e.g. ``_mnt128``) appends to the three output directories
    so a budget-sweep run leaves the canonical _full directories untouched.
    """
    suf = _suffix(args.out_suffix)
    # Raise the per-query estimate when the generator gets a bigger budget;
    # the estimate is only used to warn the user about wall-clock in the
    # preflight, so over-estimating is safer than under.
    spq = 0.30 if not args.max_new_tokens or args.max_new_tokens <= 64 else 0.50
    return DriverConfig(
        reranker_run=PROJECT_ROOT / "outputs/cross_encoder_rerank_full/run.tsv",
        reranker_manifest=PROJECT_ROOT / "outputs/cross_encoder_rerank_full/manifest.json",
        bm25_run=PROJECT_ROOT / "outputs/bm25_baseline/run.tsv",
        bm25_out=PROJECT_ROOT / f"outputs/generation_bm25_full{suf}",
        rerank_out=PROJECT_ROOT / f"outputs/generation_reranked_full{suf}",
        analysis_out=PROJECT_ROOT / f"outputs/generation_analysis{suf}",
        log_dir=PROJECT_ROOT / f"logs{suf}",
        n_eval_queries=args.num_eval_queries or 99999,
        expected_qids=args.expected_qids,
        rerank_top_k=args.rerank_top_k,
        retrieval_source_bm25="bm25",
        retrieval_source_reranked="reranked",
        seconds_per_query_estimate=spq,
        require_full_manifest=True,
        max_new_tokens=args.max_new_tokens,
    )


def build_dryrun_config(args: argparse.Namespace) -> DriverConfig:
    """Dry-run defaults: tiny sample using whatever reranker output is
    already on disk (historical reranker 1k-qid run by default). Writes to
    outputs/dryrun/... so a real run is never disturbed.
    """
    suf = _suffix(args.out_suffix)
    return DriverConfig(
        reranker_run=PROJECT_ROOT / args.dryrun_reranker_run,
        reranker_manifest=PROJECT_ROOT / args.dryrun_reranker_run.replace(
            "run.tsv", "manifest.json"
        ),
        bm25_run=PROJECT_ROOT / "outputs/bm25_baseline/run.tsv",
        bm25_out=PROJECT_ROOT / f"outputs/dryrun/gen_bm25{suf}",
        rerank_out=PROJECT_ROOT / f"outputs/dryrun/gen_reranked{suf}",
        analysis_out=PROJECT_ROOT / f"outputs/dryrun/analysis{suf}",
        log_dir=PROJECT_ROOT / f"outputs/dryrun/logs{suf}",
        n_eval_queries=args.dryrun_n,
        expected_qids=0,  # don't enforce qid coverage in dry-run
        rerank_top_k=args.rerank_top_k,
        retrieval_source_bm25="bm25",
        retrieval_source_reranked="reranked",
        seconds_per_query_estimate=0.30,
        require_full_manifest=False,  # historical 1k run may pre-date the manifest schema
        max_new_tokens=args.max_new_tokens,
    )


# --------------------------------------------------------------------------- #
# Phase output-completeness checks (idempotency)
# --------------------------------------------------------------------------- #


def _is_valid_metrics_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    metrics = data.get("metrics") or {}
    # Expect at least one of the four generation metrics with a numeric value.
    return any(isinstance(metrics.get(k), (int, float)) for k in (
        "rouge-l", "bleu", "token-f1", "exact-match",
    ))


def _is_valid_predictions_jsonl(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        return False
    try:
        n = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    n += 1
                if n >= min_rows:
                    return True
        return False
    except OSError:
        return False


def generation_outputs_complete(output_dir: Path, expected_qids: int | None) -> bool:
    """A generation phase is complete iff predictions.jsonl + metrics.json
    both exist and are well-formed. If ``expected_qids`` is given we also
    require predictions.jsonl to have exactly that many rows."""
    preds = output_dir / "predictions.jsonl"
    metrics = output_dir / "metrics.json"
    if not (_is_valid_metrics_json(metrics) and _is_valid_predictions_jsonl(preds)):
        return False
    if expected_qids is not None and expected_qids > 0:
        with open(preds) as f:
            n = sum(1 for line in f if line.strip())
        return n == expected_qids
    return True


def analysis_outputs_complete(output_dir: Path) -> bool:
    summary = output_dir / "summary.json"
    report = output_dir / "report.md"
    return summary.exists() and report.exists()


# --------------------------------------------------------------------------- #
# qid-pool mismatch (fail loudly)
# --------------------------------------------------------------------------- #


def assert_matched_qid_pools(bm25_dir: Path, rerank_dir: Path) -> None:
    bm25_qids: set[str] = set()
    rerank_qids: set[str] = set()
    with open(bm25_dir / "predictions.jsonl") as f:
        for line in f:
            if line.strip():
                bm25_qids.add(str(json.loads(line)["query_id"]))
    with open(rerank_dir / "predictions.jsonl") as f:
        for line in f:
            if line.strip():
                rerank_qids.add(str(json.loads(line)["query_id"]))
    bm25_only = bm25_qids - rerank_qids
    rerank_only = rerank_qids - bm25_qids
    if bm25_only or rerank_only:
        msg = (
            f"qid-pool mismatch between BM25 and reranked predictions:\n"
            f"  bm25={len(bm25_qids)}  rerank={len(rerank_qids)}  shared={len(bm25_qids & rerank_qids)}\n"
            f"  bm25-only first 5: {sorted(bm25_only)[:5]}\n"
            f"  rerank-only first 5: {sorted(rerank_only)[:5]}"
        )
        raise SystemExit("FATAL: " + msg)


# --------------------------------------------------------------------------- #
# Preflight summary
# --------------------------------------------------------------------------- #


def _count_qids_in_run(path: Path) -> int:
    if not path.exists():
        return 0
    qids: set[str] = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                qids.add(line.split("\t", 1)[0])
    return len(qids)


def preflight_summary(cfg: DriverConfig) -> dict[str, object]:
    """Build a small dict describing what the orchestrator is about to do.

    Pure-Python; no model loads. Counts qids in the input runs and
    estimates wall-clock from ``seconds_per_query_estimate``.
    """
    n_rerank_qids = _count_qids_in_run(cfg.reranker_run)
    n_bm25_qids = _count_qids_in_run(cfg.bm25_run)
    # The eligible pool is bounded above by the smaller of the two
    # retrieval coverages — that's what restrict-to-run enforces.
    eligible_estimate = min(n_rerank_qids, n_bm25_qids)
    # Actual eval count after --num-eval-queries clamping.
    eval_count = min(cfg.n_eval_queries, eligible_estimate)
    # Each generation pass runs once; analysis is negligible.
    est_seconds = 2 * eval_count * cfg.seconds_per_query_estimate
    return {
        "rerank_run": str(cfg.reranker_run),
        "rerank_qids_on_disk": n_rerank_qids,
        "expected_qids": cfg.expected_qids,
        "rerank_top_k": cfg.rerank_top_k,
        "bm25_run": str(cfg.bm25_run),
        "bm25_qids_on_disk": n_bm25_qids,
        "eligible_pool_estimate": eligible_estimate,
        "n_eval_queries_after_clamp": eval_count,
        "retrieval_source_bm25": cfg.retrieval_source_bm25,
        "retrieval_source_reranked": cfg.retrieval_source_reranked,
        "expected_wall_clock_minutes": round(est_seconds / 60.0, 1),
        "outputs": {
            "bm25": str(cfg.bm25_out),
            "reranked": str(cfg.rerank_out),
            "analysis": str(cfg.analysis_out),
        },
    }


def print_preflight(summary: dict[str, object]) -> None:
    print()
    print("=== PREFLIGHT ===")
    print(f"  rerank run        : {summary['rerank_run']}")
    print(f"  rerank qids       : {summary['rerank_qids_on_disk']:>6} on disk"
          f"  (expected {summary['expected_qids']})")
    print(f"  rerank top-K      : {summary['rerank_top_k']}")
    print(f"  BM25 run          : {summary['bm25_run']}")
    print(f"  BM25 qids         : {summary['bm25_qids_on_disk']:>6} on disk")
    print(f"  eligible pool     : {summary['eligible_pool_estimate']:>6} "
          "(min of bm25 and rerank coverage)")
    print(f"  --num-eval-queries: {summary['n_eval_queries_after_clamp']:>6} "
          "(after clamp)")
    print(f"  expected runtime  : {summary['expected_wall_clock_minutes']:>6} min")
    print(f"  retrieval labels  : "
          f"{summary['retrieval_source_bm25']!r}, "
          f"{summary['retrieval_source_reranked']!r}")
    print(f"  outputs           : {summary['outputs']['bm25']}")
    print(f"                      {summary['outputs']['reranked']}")
    print(f"                      {summary['outputs']['analysis']}")
    print()


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], log_path: Path) -> None:
    """Run a subprocess, streaming stdout/stderr to ``log_path`` (append).

    Raises ``SystemExit`` on non-zero exit so the driver fails loudly.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s", " ".join(cmd))
    logger.info("  (log appended to %s)", log_path)
    t0 = time.time()
    with open(log_path, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  $ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    logger.info("  ↳ exit %d in %.1fs", rc, elapsed)
    if rc != 0:
        raise SystemExit(
            f"FATAL: command failed (exit {rc}): {' '.join(cmd)}\n"
            f"       see {log_path}"
        )


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def phase_validate(cfg: DriverConfig, force: bool) -> str:
    """Phase 1: validate the reranker output."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/validate_full_rerank.py"),
        "--run-tsv", str(cfg.reranker_run),
        "--manifest", str(cfg.reranker_manifest),
        "--expected-qids", str(cfg.expected_qids),
        "--rerank-top-k", str(cfg.rerank_top_k),
    ]
    # Soft-mode the validator when we're not enforcing full coverage
    # (dry-run, or historical run without the resume manifest schema).
    if not cfg.require_full_manifest or cfg.expected_qids == 0:
        cmd.append("--partial-ok")
    _run(cmd, cfg.log_dir / "validate.log")
    return "OK"


def phase_gen_bm25(cfg: DriverConfig, force: bool) -> str:
    if not force and generation_outputs_complete(cfg.bm25_out, cfg.n_eval_queries if cfg.expected_qids else None):
        return "SKIP (already done)"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/run_generation_baseline.py"),
        "--input-run", str(cfg.bm25_run),
        "--output-dir", str(cfg.bm25_out),
        "--retrieval-source", cfg.retrieval_source_bm25,
        "--restrict-to-run", str(cfg.reranker_run),
        "--num-eval-queries", str(cfg.n_eval_queries),
    ]
    if cfg.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(cfg.max_new_tokens)]
    _run(cmd, cfg.log_dir / "gen_bm25.log")
    return "OK"


def phase_gen_reranked(cfg: DriverConfig, force: bool) -> str:
    if not force and generation_outputs_complete(cfg.rerank_out, cfg.n_eval_queries if cfg.expected_qids else None):
        return "SKIP (already done)"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/run_generation_baseline.py"),
        "--input-run", str(cfg.reranker_run),
        "--output-dir", str(cfg.rerank_out),
        "--retrieval-source", cfg.retrieval_source_reranked,
        "--num-eval-queries", str(cfg.n_eval_queries),
    ]
    if cfg.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(cfg.max_new_tokens)]
    _run(cmd, cfg.log_dir / "gen_reranked.log")
    return "OK"


def phase_analyse(cfg: DriverConfig, force: bool) -> str:
    if not force and analysis_outputs_complete(cfg.analysis_out):
        return "SKIP (already done)"
    # Fail loudly on mismatched qid pools BEFORE invoking the analysis,
    # so the error is clearly attributed.
    assert_matched_qid_pools(cfg.bm25_out, cfg.rerank_out)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/analyze_generation_rerank.py"),
        "--bm25-dir", str(cfg.bm25_out),
        "--reranked-dir", str(cfg.rerank_out),
        "--output-dir", str(cfg.analysis_out),
    ]
    _run(cmd, cfg.log_dir / "analyse.log")
    return "OK"


PHASES = {
    "validate": phase_validate,
    "gen_bm25": phase_gen_bm25,
    "gen_reranked": phase_gen_reranked,
    "analyse": phase_analyse,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with --num-eval-queries 5 against the historical 1k reranker "
             "output, writing to outputs/dryrun/. Proves orchestration wiring.",
    )
    p.add_argument(
        "--dryrun-n",
        type=int,
        default=5,
        help="Number of queries for the dry-run sample (default 5).",
    )
    p.add_argument(
        "--dryrun-reranker-run",
        type=str,
        default="outputs/cross_encoder_rerank/run.tsv",
        help="Reranker run.tsv to use in dry-run mode.",
    )
    p.add_argument(
        "--num-eval-queries",
        type=int,
        default=None,
        help="Cap the eval pool. Default: all (99999, the runner clamps).",
    )
    p.add_argument(
        "--expected-qids",
        type=int,
        default=6980,
        help="Validator enforces this many qids in the reranker run (default 6980).",
    )
    p.add_argument(
        "--rerank-top-k",
        type=int,
        default=100,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing outputs and re-run every phase.",
    )
    p.add_argument(
        "--only",
        choices=list(PHASES.keys()),
        default=None,
        help="Run only this phase (still respects --force).",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the preflight summary print (useful in scripted pipelines).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override generator max_new_tokens for both gen phases. Pass-through "
            "to run_generation_baseline.py; the runner records it in each "
            "manifest's argv. Leave unset to use cfg['generation']."
        ),
    )
    p.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix appended to bm25_out, rerank_out, and analysis_out "
            "so a budget-sweep run writes to fresh directories without "
            "clobbering the canonical _full outputs. Example: --out-suffix _mnt128."
        ),
    )
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = build_dryrun_config(args) if args.dry_run else build_default_config(args)

    if not args.skip_preflight:
        summary = preflight_summary(cfg)
        print_preflight(summary)
        # Also persist the preflight JSON so the analysis report can cite it.
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_dir / "preflight.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Run phases.
    phases_to_run = [args.only] if args.only else list(PHASES.keys())
    results: dict[str, str] = {}
    for name in phases_to_run:
        logger.info("=== phase: %s ===", name)
        try:
            results[name] = PHASES[name](cfg, force=args.force)
        except SystemExit:
            results[name] = "FAILED"
            raise

    # Friendly final summary.
    print()
    print("=== driver summary ===")
    width = max(len(p) for p in phases_to_run)
    for name in phases_to_run:
        print(f"  {name:{width}}  {results[name]}")
    print()
    print(f"BM25 generation  : {cfg.bm25_out}")
    print(f"Rerank generation: {cfg.rerank_out}")
    print(f"Analysis         : {cfg.analysis_out}")
    print(f"  report          : {cfg.analysis_out / 'report.md'}")


if __name__ == "__main__":
    main()

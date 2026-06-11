"""Pre-flight validation for ``outputs/W5_reranker_full/run.tsv``.

Run this AFTER the full dev/small rerank finishes and BEFORE kicking off
the BM25-vs-reranked generation comparison on the full eligible pool.

Validates the four invariants the comparison relies on:

1. All 6,980 dev/small qids are present.
2. Every qid has exactly ``rerank_top_k`` ranks (default 100), and the
   max observed rank equals ``rerank_top_k``.
3. No duplicate ``(qid, rank)`` pairs (a paranoid extra check that
   guards against a resume bug double-appending a chunk).
4. ``manifest.json`` records resume/chunking info: at minimum the
   ``chunk_size`` field and the ``resumed`` boolean from the runner.

Exits 0 on success, 1 on any failure (prints which check failed).

Usage::

    python scripts/validate_full_rerank.py \\
        --run-tsv outputs/W5_reranker_full/run.tsv \\
        --manifest outputs/W5_reranker_full/manifest.json \\
        --expected-qids 6980 \\
        --rerank-top-k 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tsv",
        type=Path,
        default=PROJECT_ROOT / "outputs/W5_reranker_full/run.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/W5_reranker_full/manifest.json",
    )
    parser.add_argument(
        "--expected-qids",
        type=int,
        default=6980,
        help="Number of dev/small queries expected in the run.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help=(
            "Treat a missing manifest.json as a failure. Off by default so "
            "this script can also validate a still-incomplete partial output."
        ),
    )
    parser.add_argument(
        "--partial-ok",
        action="store_true",
        help=(
            "Treat 'fewer than expected_qids' as a soft warning instead of "
            "a failure. Useful for sanity-checking a paused/partial rerank."
        ),
    )
    return parser.parse_args()


def validate_run_tsv(path: Path, top_k: int) -> tuple[dict, list[str]]:
    """Return (stats, errors). Pure-Python, single pass over the file."""
    errors: list[str] = []
    if not path.exists():
        return ({}, [f"run.tsv missing at {path}"])

    counts_per_qid: dict[str, int] = {}
    max_rank_per_qid: dict[str, int] = {}
    qid_rank_pairs: Counter[tuple[str, int]] = Counter()
    n_lines = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                errors.append(f"malformed line: {line!r}")
                continue
            qid, _q0, _doc_id, rank_str = parts[0], parts[1], parts[2], parts[3]
            try:
                rank = int(rank_str)
            except ValueError:
                errors.append(f"non-integer rank: {rank_str!r}")
                continue
            counts_per_qid[qid] = counts_per_qid.get(qid, 0) + 1
            max_rank_per_qid[qid] = max(max_rank_per_qid.get(qid, 0), rank)
            qid_rank_pairs[(qid, rank)] += 1
            n_lines += 1

    n_qids = len(counts_per_qid)
    incomplete = [
        q for q, c in counts_per_qid.items()
        if c != top_k or max_rank_per_qid[q] != top_k
    ]
    duplicates = [pair for pair, c in qid_rank_pairs.items() if c > 1]

    if incomplete:
        errors.append(
            f"{len(incomplete)} qids have an incomplete top-{top_k} block "
            f"(first 5: {incomplete[:5]})"
        )
    if duplicates:
        errors.append(
            f"{len(duplicates)} duplicate (qid, rank) pairs — possible "
            f"resume double-append (first 5: {duplicates[:5]})"
        )
    return (
        {
            "n_lines": n_lines,
            "n_qids": n_qids,
            "n_incomplete": len(incomplete),
            "n_duplicates": len(duplicates),
            "max_rank_observed": max(max_rank_per_qid.values()) if max_rank_per_qid else 0,
        },
        errors,
    )


def validate_manifest(path: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return ({}, [f"manifest.json missing at {path}"])
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return ({}, [f"manifest.json is not valid JSON: {e}"])
    extra = data.get("extra", {})
    missing = [k for k in ("chunk_size", "resumed") if k not in extra]
    if missing:
        errors.append(
            f"manifest.extra missing resume/chunking fields: {missing}"
        )
    return (
        {
            "chunk_size": extra.get("chunk_size"),
            "resumed": extra.get("resumed"),
            "n_resumed_qids": extra.get("n_resumed_qids"),
            "n_pending_this_run": extra.get("n_pending_this_run"),
            "n_eval_queries": extra.get("n_eval_queries"),
        },
        errors,
    )


def main() -> int:
    args = parse_args()

    run_stats, run_errors = validate_run_tsv(args.run_tsv, top_k=args.rerank_top_k)

    if args.manifest.exists() or args.require_manifest:
        mf_stats, mf_errors = validate_manifest(args.manifest)
    else:
        mf_stats, mf_errors = ({}, [])
        mf_stats["note"] = "manifest absent (rerank not finalised yet)"

    # Coverage check (separate so it can be downgraded with --partial-ok).
    qid_short = max(0, args.expected_qids - run_stats.get("n_qids", 0))
    if qid_short and not args.partial_ok:
        run_errors.append(
            f"only {run_stats.get('n_qids', 0)} / {args.expected_qids} "
            f"qids present in run.tsv (short by {qid_short})"
        )

    payload = {
        "run_tsv": str(args.run_tsv),
        "manifest": str(args.manifest),
        "expected_qids": args.expected_qids,
        "rerank_top_k": args.rerank_top_k,
        "run_stats": run_stats,
        "manifest_stats": mf_stats,
        "run_errors": run_errors,
        "manifest_errors": mf_errors,
    }
    if qid_short and args.partial_ok:
        payload["partial_warning"] = (
            f"Run is partial: {run_stats.get('n_qids', 0)} / "
            f"{args.expected_qids} qids present."
        )

    print(json.dumps(payload, indent=2))
    if run_errors or mf_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R5 metric-robustness: NLI grounding factorial runner.

For each NLI backbone in :data:`msmarco_genqa.evaluation.nli_factorial.NLI_BACKBONES`,
this scores the *same* paired W3 predictions (BM25-fed vs reranked-fed) the
W7-A audit used, then sweeps the score-formula x threshold grid for free off
the cached 3-class probabilities. The question it answers: does the W7-A
"rerank lowers NLI grounding" sign-reversal survive across backbones and
across the formula/threshold grid, or is it an artifact of one choice?

The expensive part is one NLI forward pass per query per arm per backbone.
``deberta-v3-small`` and ``minilm-l6`` are CPU-minutes each; ``roberta-large-mnli``
is CPU-hours over 6,980 x 2 pairs — run it overnight or pass ``--max-pairs``
for a smoke. Each backbone x direction writes its own output dir with:

- ``per_query_probs.jsonl`` — cached 3-class probs for both arms (so the
  grid can be re-derived without re-running the model).
- ``summary.json``          — every cell's paired-bootstrap CI + the
  sign-reversal verdict + full provenance.
- ``manifest.json``         — schema-v2 manifest under the ``nli_grounding``
  profile (7 ``extra.nli.*`` fields enforced at write time).

Inputs (same convention as ``scripts/grounding_audit.py``):

- ``<bm25-dir>/predictions.jsonl``
- ``<reranked-dir>/predictions.jsonl``

Usage::

    python scripts/run_nli_factorial.py \\
        --bm25-dir outputs/week03_generation_bm25_full \\
        --reranked-dir outputs/week03_generation_reranked_full \\
        --output-dir outputs/r5_nli_factorial \\
        --backbone all

    # smoke on 50 pairs with the small backbone only
    python scripts/run_nli_factorial.py --backbone deberta-v3-small --max-pairs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.nli_factorial import (
    DEFAULT_FORMULAS,
    DEFAULT_THRESHOLDS,
    NLI_BACKBONES,
    run_factorial,
)
from msmarco_genqa.evaluation.nli_grounding import (
    per_query_nli_probs,
    resolve_label_indices,
)
from msmarco_genqa.util.environment import capture_environment
from msmarco_genqa.util.manifest import (
    compute_data_fingerprint,
    compute_env_fingerprint,
    compute_resolved_config_hash,
    write_resolved_config,
    write_run_manifest,
)

logger = logging.getLogger("run_nli_factorial")

DIRECTIONS = ("passages_to_prediction", "prediction_to_passages")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bm25-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week03_generation_bm25_full",
    )
    parser.add_argument(
        "--reranked-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/week03_generation_reranked_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/r5_nli_factorial",
        help="Base dir; each backbone x direction writes a subdir under it.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="all",
        choices=["all", *NLI_BACKBONES.keys()],
        help="Which registry backbone to run, or 'all' (default).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="passages_to_prediction",
        choices=["both", *DIRECTIONS],
        help="Premise/hypothesis direction, or 'both' to sweep the axis.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Smoke cap on paired qids (first N, deterministic). 0 = all.",
    )
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument(
        "--nli-device",
        type=str,
        default=None,
        help="Override device for NLI scoring; auto-detect when omitted.",
    )
    parser.add_argument("--require-clean-tree", action="store_true")
    parser.add_argument("--allow-incomplete-manifest", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def load_predictions_ordered(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def assert_paired(bm25: list[dict], rerank: list[dict]) -> None:
    if len(bm25) != len(rerank):
        raise SystemExit(
            f"Prediction files have different lengths: "
            f"bm25={len(bm25)} reranked={len(rerank)}."
        )
    bm25_qids = [str(r["query_id"]) for r in bm25]
    rerank_qids = [str(r["query_id"]) for r in rerank]
    if bm25_qids != rerank_qids:
        if set(bm25_qids) == set(rerank_qids):
            raise SystemExit(
                "Prediction files cover the same qids but in different order."
            )
        raise SystemExit(
            "Prediction files cover different qid sets — pairing impossible."
        )


def _rel_to_root(path: Path) -> str:
    """Path relative to PROJECT_ROOT when possible, else its string form.

    The plain ``.relative_to(PROJECT_ROOT)`` crashes when inputs live
    outside the project tree (e.g. a worktree run whose data dir is
    symlinked elsewhere). Mirror the runner convention in
    ``experiments/run_generation_baseline.py`` and degrade to the absolute
    string rather than failing after the expensive scoring is already done.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT))
    return str(resolved)


# --------------------------------------------------------------------------- #
# One backbone x direction run
# --------------------------------------------------------------------------- #

def run_one(
    *,
    name: str,
    spec: dict[str, str],
    direction: str,
    bm25: list[dict],
    rerank: list[dict],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_id = spec["model_id"]
    revision = spec.get("revision")

    # Resolve the label->column mapping from the model config alone (a tiny
    # config-only download), so the manifest records it without a second
    # full model load.
    from transformers import AutoConfig

    cfg_kwargs = {"revision": revision} if revision else {}
    hf_config = AutoConfig.from_pretrained(model_id, **cfg_kwargs)
    label_idx = resolve_label_indices(hf_config.id2label)

    bm25_preds = [r.get("prediction") or "" for r in bm25]
    bm25_psgs = [list(r.get("passages") or []) for r in bm25]
    rerank_preds = [r.get("prediction") or "" for r in rerank]
    rerank_psgs = [list(r.get("passages") or []) for r in rerank]

    logger.info(
        "[%s | %s] scoring %d paired qids (model=%s) ...",
        name, direction, len(bm25), model_id,
    )
    t0 = time.time()
    probs_bm25 = per_query_nli_probs(
        bm25_preds, bm25_psgs,
        model_type=model_id, revision=revision, direction=direction,
        batch_size=args.nli_batch_size, device=args.nli_device,
        max_length=args.nli_max_length,
    )
    logger.info("  BM25 arm done in %.1f s; reranked arm ...", time.time() - t0)
    t1 = time.time()
    probs_rerank = per_query_nli_probs(
        rerank_preds, rerank_psgs,
        model_type=model_id, revision=revision, direction=direction,
        batch_size=args.nli_batch_size, device=args.nli_device,
        max_length=args.nli_max_length,
    )
    score_secs = time.time() - t0
    logger.info("  reranked arm done in %.1f s.", time.time() - t1)

    cells = run_factorial(
        probs_bm25, probs_rerank,
        formulas=DEFAULT_FORMULAS, thresholds=DEFAULT_THRESHOLDS,
        n_resamples=args.n_resamples, ci=args.ci, seed=args.bootstrap_seed,
    )
    n_reverse = sum(1 for c in cells if c["reverses_sign"])
    baseline = next(
        c for c in cells
        if c["formula"] == "entailment" and c["threshold"] is None
    )

    out_dir = args.output_dir / f"{name}__{direction}"
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_path = out_dir / "per_query_probs.jsonl"
    with open(probs_path, "w") as f:
        for i in range(len(bm25)):
            f.write(json.dumps({
                "query_id": str(bm25[i]["query_id"]),
                "bm25": probs_bm25[i],
                "reranked": probs_rerank[i],
            }) + "\n")

    summary: dict[str, Any] = {
        "task": "r5_nli_grounding_factorial",
        "backbone": name,
        "model_id": model_id,
        "revision": revision,
        "premise_hypothesis_direction": direction,
        "label_index_mapping": label_idx,
        "n_paired_qids": len(bm25),
        "max_pairs_cap": int(args.max_pairs) or None,
        "scoring_seconds": round(score_secs, 2),
        "inputs": {
            "bm25_predictions": _rel_to_root(args.bm25_dir / "predictions.jsonl"),
            "reranked_predictions": _rel_to_root(args.reranked_dir / "predictions.jsonl"),
        },
        "verdict": {
            "baseline_cell_reverses_sign": baseline["reverses_sign"],
            "baseline_delta": baseline["bootstrap"]["mean_delta"],
            "baseline_ci": [
                baseline["bootstrap"]["ci_low"], baseline["bootstrap"]["ci_high"],
            ],
            "n_cells": len(cells),
            "n_cells_reverse_sign": n_reverse,
        },
        "cells": cells,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- schema-v2 manifest under the nli_grounding profile ----
    run_cfg: dict[str, Any] = {
        "task": "r5_nli_grounding_factorial",
        "backbone": name,
        "model_id": model_id,
        "revision": revision,
        "direction": direction,
        "formulas": list(DEFAULT_FORMULAS),
        "thresholds": list(DEFAULT_THRESHOLDS),
        "n_resamples": args.n_resamples,
        "ci": args.ci,
        "bootstrap_seed": args.bootstrap_seed,
        "max_pairs": int(args.max_pairs),
        "bm25_dir": str(args.bm25_dir),
        "reranked_dir": str(args.reranked_dir),
    }
    resolved_config_path = write_resolved_config(run_cfg, out_dir)
    extra: dict[str, Any] = {
        "task": "r5_nli_grounding_factorial",
        "seed": int(args.bootstrap_seed),
        "n_paired_qids": len(bm25),
        "resolved_config_hash": compute_resolved_config_hash(run_cfg),
        "data_fingerprint": compute_data_fingerprint(
            cache_dir=PROJECT_ROOT / "data/raw",
            extra_files={
                "bm25_predictions": args.bm25_dir / "predictions.jsonl",
                "reranked_predictions": args.reranked_dir / "predictions.jsonl",
            },
        ),
        "env_fingerprint": compute_env_fingerprint(capture_environment()),
        "nli": {
            "backbone": model_id,
            "revision": revision,
            "score_formula": list(DEFAULT_FORMULAS),
            "threshold": list(DEFAULT_THRESHOLDS),
            "premise_hypothesis_direction": direction,
            "label_index_mapping": label_idx,
            "aggregation": ["mean_score", "grounded_rate"],
        },
    }
    write_run_manifest(
        project_root=PROJECT_ROOT,
        output_dir=out_dir,
        command=sys.argv,
        extra_outputs=[probs_path, summary_path, resolved_config_path],
        extra=extra,
        require_clean_tree=args.require_clean_tree,
        allow_incomplete=args.allow_incomplete_manifest,
        profile="nli_grounding",
    )

    logger.info(
        "[%s | %s] baseline Δ=%+.4f CI=[%+.4f, %+.4f] reverses=%s; "
        "%d/%d cells reverse. -> %s",
        name, direction, baseline["bootstrap"]["mean_delta"],
        baseline["bootstrap"]["ci_low"], baseline["bootstrap"]["ci_high"],
        baseline["reverses_sign"], n_reverse, len(cells), out_dir,
    )
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    bm25 = load_predictions_ordered(args.bm25_dir / "predictions.jsonl")
    rerank = load_predictions_ordered(args.reranked_dir / "predictions.jsonl")
    assert_paired(bm25, rerank)
    if args.max_pairs and 0 < args.max_pairs < len(bm25):
        bm25 = bm25[: args.max_pairs]
        rerank = rerank[: args.max_pairs]
        logger.info("Smoke cap: first %d paired qids.", args.max_pairs)

    backbones = (
        list(NLI_BACKBONES) if args.backbone == "all" else [args.backbone]
    )
    directions = (
        list(DIRECTIONS) if args.direction == "both" else [args.direction]
    )

    results: list[dict[str, Any]] = []
    for name in backbones:
        for direction in directions:
            results.append(run_one(
                name=name, spec=NLI_BACKBONES[name], direction=direction,
                bm25=bm25, rerank=rerank, args=args,
            ))

    print()
    print("=== R5 NLI grounding factorial — baseline cell per backbone ===")
    print(f"  {'backbone':22s} {'direction':24s} {'Δ':>9s}  {'reverses':>8s}  {'cells↺':>7s}")
    for s in results:
        v = s["verdict"]
        print(
            f"  {s['backbone']:22s} {s['premise_hypothesis_direction']:24s} "
            f"{v['baseline_delta']:>+9.4f}  {str(v['baseline_cell_reverses_sign']):>8s}  "
            f"{v['n_cells_reverse_sign']}/{v['n_cells']:>2}"
        )
    print()
    print(
        "Sign-reversal holds where Δ<0 with CI upper bound below 0 across "
        "backbones AND most cells. See each subdir's summary.json."
    )


if __name__ == "__main__":
    main()

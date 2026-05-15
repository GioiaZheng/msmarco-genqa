"""Render markdown reports (and optionally PDFs) from experiment outputs.

Usage::

    python -m src.reporting.build_report --week week02
    python -m src.reporting.build_report --week week03
    python -m src.reporting.build_report --week review_all

For ``weekNN`` the renderer reads:

- ``outputs/<week>/metrics.json``
- ``outputs/<week>/examples.jsonl`` (W2)
- ``outputs/<week>/predictions.jsonl`` (W3)

substitutes ``{{...}}`` placeholders inside ``reports/templates/<week>.md``,
and writes ``reports/generated/<week>.md``.

For ``review_all`` the renderer just stamps a timestamp into the static
``reports/templates/review_all.md`` template and writes
``reports/generated/review_all.md`` — no metrics.json is consulted.

If ``pandoc`` is on PATH it attempts to produce ``reports/generated/<name>.pdf``;
missing pandoc is a warning, not an error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = PROJECT_ROOT / "reports" / "templates"
GENERATED_DIR = PROJECT_ROOT / "reports" / "generated"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fmt_float(x: Any, places: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{places}f}"
    except (TypeError, ValueError):
        return str(x)


def _truncate(text: str, n: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n].rstrip() + "..."


def _signed_delta(after: Any, before: Any, places: int = 4) -> str:
    if after is None or before is None:
        return "—"
    try:
        return f"{float(after) - float(before):+.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _substitute(template: str, fields: dict[str, str]) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


# --------------------------------------------------------------------------- #
# Week 01 — EDA. No metrics.json required: this is a notebook-driven report
# that just frames the figures the W1 notebook produces.
# --------------------------------------------------------------------------- #

def build_week01(out_dir: Path) -> str:
    """Render the W1 EDA report.

    Unlike W2/W3, W1 has no ``metrics.json`` — it's an EDA notebook.
    The template just embeds the figures that ``notebooks/week01_eda.ipynb``
    writes to ``figures/`` and substitutes a generation timestamp.
    ``out_dir`` is unused but kept in the signature for API symmetry.
    """
    template = (TEMPLATES_DIR / "week01_eda.md").read_text()
    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Week 02
# --------------------------------------------------------------------------- #

def _format_week02_case_studies(examples: list[dict], n: int = 5) -> str:
    hits = [e for e in examples if e.get("first_relevant_rank_in_top10")]
    hits = hits[:n]
    if not hits:
        return "*No examples with a top-10 hit were sampled.*"
    blocks: list[str] = []
    for e in hits:
        block = [
            f"### `{e['query_id']}` — {e['query']}",
            "",
            f"- First relevant passage rank: **{e['first_relevant_rank_in_top10']}**",
            f"- Relevant doc id(s): {', '.join(e.get('relevant_doc_ids') or []) or '—'}",
            "",
            "Top 3 retrieved:",
            "",
        ]
        for r in e["top_results"][:3]:
            mark = "✓" if r["is_relevant"] else " "
            block.append(
                f"{r['rank']}. [{mark}] (score={r['score']:.3f}) {_truncate(r['passage'])}"
            )
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _format_week02_error_analysis(examples: list[dict], n: int = 5) -> str:
    misses = [e for e in examples if not e.get("first_relevant_rank_in_top10")]
    misses = misses[:n]
    if not misses:
        return "*All sampled queries had a relevant passage in the top-10.*"
    blocks: list[str] = []
    for e in misses:
        block = [
            f"### `{e['query_id']}` — {e['query']}",
            "",
            f"- Relevant doc id(s): {', '.join(e.get('relevant_doc_ids') or []) or '—'}",
            "- BM25 top 3:",
            "",
        ]
        for r in e["top_results"][:3]:
            block.append(
                f"  {r['rank']}. (score={r['score']:.3f}) {_truncate(r['passage'])}"
            )
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_week02(out_dir: Path) -> str:
    metrics_path = out_dir / "metrics.json"
    examples_path = out_dir / "examples.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiments/run_retrieval.py first."
        )
    payload = _read_json(metrics_path)
    examples = _read_jsonl(examples_path)

    cfg = payload.get("config", {})
    metrics = payload.get("metrics", {})
    timing = payload.get("wall_clock_seconds", {})
    # Schema migration: ``n_examples`` is the new top-level field; older
    # metrics.json files keep the count inside ``metrics["n_queries"]``.
    n_eval = (
        payload.get("n_examples")
        or metrics.get("n_queries")
        or 0
    )
    # Old schema used "search_this_run"; new schema uses "search".
    search = timing.get("search") if timing.get("search") is not None else timing.get("search_this_run")
    ms_per_q = (search * 1000 / n_eval) if (search and n_eval) else None

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_corpus": cfg.get("data", {}).get("expected_corpus_size", "—"),
        "n_queries_total": cfg.get("data", {}).get("expected_dev_queries", "—"),
        "n_queries_eval": n_eval,
        "stopwords": cfg.get("retrieval", {}).get("stopwords", "—"),
        "k1": _fmt_float(cfg.get("retrieval", {}).get("k1"), 2),
        "b": _fmt_float(cfg.get("retrieval", {}).get("b"), 2),
        "top_k": cfg.get("retrieval", {}).get("top_k", "—"),
        "indexing_seconds": _fmt_float(timing.get("indexing"), 1),
        "search_seconds": _fmt_float(search, 1),
        "search_ms_per_query": _fmt_float(ms_per_q, 1),
        "mrr_at_10": _fmt_float(metrics.get("mrr@10")),
        "ndcg_at_10": _fmt_float(metrics.get("ndcg@10")),
        "recall_at_100": _fmt_float(metrics.get("recall@100")),
        "recall_at_1000": _fmt_float(metrics.get("recall@1000")),
        "case_studies": _format_week02_case_studies(examples),
        "error_analysis": _format_week02_error_analysis(examples),
    }

    template = (TEMPLATES_DIR / "week02_bm25.md").read_text()
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Week 03
# --------------------------------------------------------------------------- #

def _format_week03_examples(records: list[dict], n: int = 5) -> str:
    if not records:
        return "*No predictions available.*"
    blocks: list[str] = []
    for r in records[:n]:
        refs = r.get("references", []) or []
        ref_str = " | ".join(refs) if refs else "—"
        block = [
            f"### `{r.get('query_id')}` — {r.get('query', '')}",
            "",
            f"- **Reference(s):** {ref_str}",
            f"- **Prediction:** {r.get('prediction', '')}",
            "",
            "Retrieved context (top passages used):",
            "",
        ]
        for j, p in enumerate(r.get("passages", [])[:3], 1):
            block.append(f"  {j}. {_truncate(p)}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _format_week03_errors(records: list[dict], n: int = 5) -> str:
    """Pick predictions whose token-level F1 (computed on first reference) is 0."""
    from src.evaluation.generation import token_f1  # lazy import

    losers = []
    for r in records:
        refs = r.get("references", []) or []
        if not refs:
            continue
        if token_f1(r.get("prediction", ""), refs) == 0:
            losers.append(r)
    if not losers:
        return "*No zero-F1 predictions in the sampled set.*"
    blocks: list[str] = []
    for r in losers[:n]:
        refs = r.get("references", []) or []
        block = [
            f"### `{r.get('query_id')}` — {r.get('query', '')}",
            "",
            f"- **Reference(s):** {' | '.join(refs) or '—'}",
            f"- **Prediction:** {r.get('prediction', '')}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_week03(out_dir: Path) -> str:
    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "predictions.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. "
            "Run experiments/run_generation_baseline.py first."
        )
    payload = _read_json(metrics_path)
    records = _read_jsonl(predictions_path)
    cfg = payload.get("config", {})
    metrics = payload.get("metrics", {})
    # Schema migration: prefer top-level ``n_examples`` (new), fall back to
    # ``metrics.n_predictions`` or ``n_eval`` from older metrics.json files.
    n_eval = (
        payload.get("n_examples")
        or metrics.get("n_predictions")
        or payload.get("n_eval")
        or 0
    )

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": cfg.get("generation", {}).get("model_name", "—"),
        "top_k_passages": cfg.get("generation", {}).get("top_k_passages", "—"),
        "max_input_length": cfg.get("generation", {}).get("max_input_length", "—"),
        "max_new_tokens": cfg.get("generation", {}).get("max_new_tokens", "—"),
        "n_eval": n_eval,
        "rouge_l": _fmt_float(metrics.get("rouge-l")),
        "bleu": _fmt_float(metrics.get("bleu")),
        "exact_match": _fmt_float(metrics.get("exact-match")),
        "token_f1": _fmt_float(metrics.get("token-f1")),
        "examples": _format_week03_examples(records),
        "error_analysis": _format_week03_errors(records),
    }

    template = (TEMPLATES_DIR / "week03_generation.md").read_text()
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Week 04 — dense retrieval (sampled), with head-to-head BM25-on-sample.
# --------------------------------------------------------------------------- #

def _format_week04_case_studies(examples: list[dict], n: int = 5) -> str:
    """Show queries where the dense vs BM25 first-rank disagrees most."""
    if not examples:
        return "*No examples available.*"

    def _gap(e):
        d = e.get("dense_first_rank_in_top10")
        b = e.get("bm25_sample_first_rank_in_top10")
        d_eff = d if d is not None else 11
        b_eff = b if b is not None else 11
        return abs(d_eff - b_eff), -(d_eff + b_eff)  # bigger gap first, then smaller ranks

    ranked = sorted(examples, key=_gap, reverse=True)
    blocks: list[str] = []
    for e in ranked[:n]:
        d = e.get("dense_first_rank_in_top10")
        b = e.get("bm25_sample_first_rank_in_top10")
        block = [
            f"### `{e.get('query_id')}` — {e.get('query', '')}",
            "",
            f"- Dense first-relevant rank: **{d if d is not None else '>10'}**",
            f"- BM25 first-relevant rank:  **{b if b is not None else '>10'}**",
            f"- Relevant doc id(s): {', '.join(e.get('relevant_doc_ids') or []) or '—'}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _format_week04_discussion(dense_m: dict, bm25_m: dict | None) -> str:
    """Generate a small bulleted discussion grounded in the actual numbers."""
    if bm25_m is None:
        return (
            "- BM25 comparison was disabled for this run; only dense numbers reported."
        )
    bullets: list[str] = []

    def _diff(key, label):
        d = dense_m.get(key)
        b = bm25_m.get(key)
        if d is None or b is None:
            return None
        delta = d - b
        if abs(delta) < 1e-3:
            return f"- **{label}**: dense and BM25 are within 0.001 ({d:.4f} vs {b:.4f})."
        winner = "dense" if delta > 0 else "BM25"
        return (
            f"- **{label}**: {winner} wins by {abs(delta):.4f} "
            f"(dense {d:.4f} vs BM25 {b:.4f})."
        )

    for k, lbl in [
        ("mrr@10", "MRR@10"),
        ("ndcg@10", "nDCG@10"),
        ("recall@100", "Recall@100"),
        ("recall@1000", "Recall@1000"),
    ]:
        line = _diff(k, lbl)
        if line:
            bullets.append(line)

    bullets.append(
        "- Caveat: every dev relevant doc is in the sampled pool by construction, "
        "so absolute numbers are upper-bounded and will drop on a larger / "
        "non-anchored sample."
    )
    return "\n".join(bullets) if bullets else "- (no numeric comparison available)"


def build_week04(out_dir: Path) -> str:
    metrics_path = out_dir / "metrics.json"
    examples_path = out_dir / "examples.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiments/run_dense_retrieval.py first."
        )
    payload = _read_json(metrics_path)
    examples = _read_jsonl(examples_path)

    cfg = payload.get("config", {})
    metrics = payload.get("metrics", {})
    dense_m = metrics.get("dense", {})
    bm25_m = metrics.get("bm25_sample") or None
    timing = payload.get("wall_clock_seconds", {})
    sample = payload.get("sample", {})

    n_eval = payload.get("n_examples", 0)
    n_total_queries = cfg.get("data", {}).get("expected_dev_queries", 6980)
    sample_size = sample.get("size", "—")
    encode_s = timing.get("encode_corpus")
    encode_per_doc_ms = (
        (encode_s * 1000 / sample_size) if (encode_s and isinstance(sample_size, int) and sample_size) else None
    )

    def _delta(key):
        d = dense_m.get(key)
        b = (bm25_m or {}).get(key)
        if d is None or b is None:
            return "—"
        return f"{d - b:+.4f}"

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": cfg.get("dense", {}).get("model_name", "—"),
        "sample_size": f"{sample_size:,}" if isinstance(sample_size, int) else str(sample_size),
        "k1": _fmt_float(cfg.get("retrieval", {}).get("k1"), 2),
        "b": _fmt_float(cfg.get("retrieval", {}).get("b"), 2),
        "n_eval_queries": n_eval,
        "n_total_queries": n_total_queries,
        "n_qrels_doc_ids_in_sample": sample.get("n_qrels_doc_ids_in_sample", "—"),
        "seed": cfg.get("seed", "—"),
        "top_k": payload.get("top_k", "—"),
        "encode_seconds": _fmt_float(encode_s, 1),
        "encode_per_doc_ms": _fmt_float(encode_per_doc_ms, 2),
        "dense_search_seconds": _fmt_float(timing.get("dense_search"), 1),
        "bm25_build_seconds": _fmt_float(timing.get("bm25_sample_build"), 1),
        "bm25_search_seconds": _fmt_float(timing.get("bm25_sample_search"), 1),
        "dense_mrr10": _fmt_float(dense_m.get("mrr@10")),
        "dense_ndcg10": _fmt_float(dense_m.get("ndcg@10")),
        "dense_r100": _fmt_float(dense_m.get("recall@100")),
        "dense_r1000": _fmt_float(dense_m.get("recall@1000")),
        "bm25_mrr10": _fmt_float((bm25_m or {}).get("mrr@10")),
        "bm25_ndcg10": _fmt_float((bm25_m or {}).get("ndcg@10")),
        "bm25_r100": _fmt_float((bm25_m or {}).get("recall@100")),
        "bm25_r1000": _fmt_float((bm25_m or {}).get("recall@1000")),
        "delta_mrr10": _delta("mrr@10"),
        "delta_ndcg10": _delta("ndcg@10"),
        "delta_r100": _delta("recall@100"),
        "delta_r1000": _delta("recall@1000"),
        "case_studies": _format_week04_case_studies(examples),
        "discussion_bullets": _format_week04_discussion(dense_m, bm25_m),
    }

    template = (TEMPLATES_DIR / "week04_dense.md").read_text()
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Week 05 — cross-encoder reranking on top of the W4 dense run.
# --------------------------------------------------------------------------- #


def _format_week05_case_studies(examples: list[dict], n: int = 5) -> str:
    """Show queries where the reranker moved the relevant doc the most."""
    if not examples:
        return "*No examples available.*"

    def _gap(e):
        d = e.get("dense_first_rank_in_top10")
        r = e.get("rerank_first_rank_in_top10")
        # Use 11 to mean "not in top-10". Bigger improvement first.
        d_eff = d if d is not None else 11
        r_eff = r if r is not None else 11
        improvement = d_eff - r_eff
        return improvement, -r_eff

    ranked = sorted(examples, key=_gap, reverse=True)
    blocks: list[str] = []
    for e in ranked[:n]:
        d = e.get("dense_first_rank_in_top10")
        r = e.get("rerank_first_rank_in_top10")
        block = [
            f"### `{e.get('query_id')}` — {e.get('query', '')}",
            "",
            f"- Dense first-relevant rank:  **{d if d is not None else '>10'}**",
            f"- Rerank first-relevant rank: **{r if r is not None else '>10'}**",
            f"- Relevant doc id(s): {', '.join(e.get('relevant_doc_ids') or []) or '—'}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _format_week05_discussion(
    dense_m: dict, rerank_m: dict, examples: list[dict]
) -> str:
    bullets: list[str] = []

    def _diff(key, label):
        d = dense_m.get(key)
        r = rerank_m.get(key)
        if d is None or r is None:
            return None
        delta = r - d
        if abs(delta) < 1e-3:
            return f"- **{label}**: rerank ≈ dense ({r:.4f} vs {d:.4f}, Δ={delta:+.4f})."
        verb = "improves" if delta > 0 else "regresses"
        return (
            f"- **{label}**: cross-encoder {verb} the metric by {abs(delta):.4f} "
            f"(dense {d:.4f} → rerank {r:.4f})."
        )

    for k, lbl in [("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]:
        line = _diff(k, lbl)
        if line:
            bullets.append(line)

    # Recall@100 is unchanged by reranking; call that out explicitly so the
    # narrative ("recall saturated, reranker improves ordering") is grounded.
    d_r100 = dense_m.get("recall@100")
    r_r100 = rerank_m.get("recall@100")
    if d_r100 is not None and r_r100 is not None:
        if abs(r_r100 - d_r100) < 1e-6:
            bullets.append(
                "- **Recall@100** is unchanged ({:.4f}) — by construction, "
                "the reranker only re-orders the top-K dense candidates and "
                "cannot recover docs the dense retriever missed.".format(d_r100)
            )

    # Promotion stats from the examples.
    promoted = sum(
        1
        for e in examples
        if (e.get("rerank_first_rank_in_top10") or 99)
        < (e.get("dense_first_rank_in_top10") or 99)
    )
    demoted = sum(
        1
        for e in examples
        if (e.get("rerank_first_rank_in_top10") or 99)
        > (e.get("dense_first_rank_in_top10") or 99)
    )
    if examples:
        bullets.append(
            f"- In the {len(examples)} sampled examples, the relevant passage "
            f"was promoted in {promoted} and demoted in {demoted} "
            f"(remaining cases unchanged or both >10)."
        )
    return "\n".join(bullets) if bullets else "- (no numeric comparison available)"


def build_week05(out_dir: Path) -> str:
    metrics_path = out_dir / "metrics.json"
    examples_path = out_dir / "examples.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiments/run_reranker.py first."
        )
    payload = _read_json(metrics_path)
    examples = _read_jsonl(examples_path)

    rerank_info = payload.get("rerank", {})
    metrics = payload.get("metrics", {})
    dense_m = metrics.get("dense", {})
    rerank_m = metrics.get("rerank", {})
    timing = payload.get("wall_clock_seconds", {})
    throughput = payload.get("throughput", {})

    # Try to grab the W4 dense Recall@100 (for the "recall already
    # saturated" framing) — fall back to the input dense top-K's recall.
    w4_metrics_path = PROJECT_ROOT / "outputs" / rerank_info.get(
        "input_week", "week04_dense"
    ) / "metrics.json"
    dense_r100_w4 = None
    if w4_metrics_path.exists():
        w4 = _read_json(w4_metrics_path)
        dense_r100_w4 = (w4.get("metrics", {}).get("dense") or {}).get("recall@100")

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": rerank_info.get("model_name", "—"),
        "rerank_top_k": rerank_info.get("rerank_top_k", "—"),
        "input_run": rerank_info.get("input_run", "—"),
        "n_queries": payload.get("n_examples", "—"),
        "n_pairs": throughput.get("n_pairs", "—"),
        "rerank_seconds": _fmt_float(timing.get("rerank"), 1),
        "resolve_seconds": _fmt_float(timing.get("resolve_passages"), 1),
        "queries_per_sec": _fmt_float(throughput.get("queries_per_sec"), 2),
        "pairs_per_sec": _fmt_float(throughput.get("pairs_per_sec"), 0),
        "peak_memory_mib": _fmt_float(payload.get("peak_memory_mib"), 0),
        "batch_size": rerank_info.get("batch_size", "—"),
        "max_length": rerank_info.get("max_length", "—"),
        "dense_mrr10": _fmt_float(dense_m.get("mrr@10")),
        "dense_ndcg10": _fmt_float(dense_m.get("ndcg@10")),
        "dense_r100": _fmt_float(dense_m.get("recall@100")),
        "rerank_mrr10": _fmt_float(rerank_m.get("mrr@10")),
        "rerank_ndcg10": _fmt_float(rerank_m.get("ndcg@10")),
        "rerank_r100": _fmt_float(rerank_m.get("recall@100")),
        "delta_mrr10": _signed_delta(rerank_m.get("mrr@10"), dense_m.get("mrr@10")),
        "delta_ndcg10": _signed_delta(rerank_m.get("ndcg@10"), dense_m.get("ndcg@10")),
        "delta_r100": _signed_delta(rerank_m.get("recall@100"), dense_m.get("recall@100")),
        "dense_recall_at_100_w4": _fmt_float(dense_r100_w4) if dense_r100_w4 else "—",
        "case_studies": _format_week05_case_studies(examples),
        "discussion_bullets": _format_week05_discussion(dense_m, rerank_m, examples),
    }

    template = (TEMPLATES_DIR / "week05_reranker.md").read_text()
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Week 06 — evaluation layer (BERTScore semantic proxy + regression taxonomy).
#
# Unlike W2-W5 this week has no single ``metrics.json`` of its own; the
# numbers live in two sibling output dirs:
#   - outputs/week06_bertscore_proxy/bertscore_proxy_ci.json
#   - outputs/week06_analysis/{summary.json, regression_taxonomy.json}
# ``out_dir`` is the canonical W6 dir (``outputs/week06_analysis``); the
# BERTScore proxy JSON is looked up by sibling path inside the function.
# --------------------------------------------------------------------------- #


def _fmt_pct(value: Any, denom: float | int, places: int = 1) -> str:
    """Render ``value`` as a percent of ``denom`` (e.g. 14/40 → '35.0 %')."""
    try:
        f = float(value) / float(denom) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"
    return f"{f:.{places}f} %"


def build_week06(out_dir: Path) -> str:
    summary_path = out_dir / "summary.json"
    taxonomy_path = out_dir / "regression_taxonomy.json"
    bertscore_path = (
        out_dir.parent / "week06_bertscore_proxy" / "bertscore_proxy_ci.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. "
            "Run scripts/analyze_generation_rerank.py first."
        )
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"{taxonomy_path} not found. "
            "Run scripts/regression_failure_taxonomy.py first."
        )
    if not bertscore_path.exists():
        raise FileNotFoundError(
            f"{bertscore_path} not found. "
            "Run scripts/bertscore_paired_eval.py first."
        )

    summary = _read_json(summary_path)
    taxonomy = _read_json(taxonomy_path)
    bert = _read_json(bertscore_path)

    # Headline / counts
    n_shared = summary.get("n_shared_qids", 0)
    n_regression = (summary.get("headline") or {}).get("n_strict_regressions", 0)

    # BERTScore proxy
    bs_boot = bert.get("bootstrap", {}) or {}
    bs_meta = bert.get("bertscore", {}) or {}
    bs_subsample = bert.get("subsample", {}) or {}
    n_pairs = bs_subsample.get("n_sampled", 0)
    win = bert.get("win_rate_rerank_strictly_better", 0.0)
    tie = bert.get("tie_rate", 0.0)
    loss = bert.get("loss_rate_bm25_strictly_better", 0.0)

    # Regression taxonomy — the JSON omits categories whose count is zero, so
    # default-to-zero on lookup.
    tax_counts = taxonomy.get("counts", {}) or {}
    tax_n_sampled = taxonomy.get("n_sampled", 0)
    cats = (
        "truncation_short",
        "truncation_midword",
        "topic_drift",
        "extractive_passage_bias",
        "semantic_mismatch",
    )
    tax_values = {c: int(tax_counts.get(c, 0)) for c in cats}
    truncation_total = (
        tax_values["truncation_short"] + tax_values["truncation_midword"]
    )

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_shared_qids": f"{n_shared:,}",
        "n_regression": str(n_regression),
        # BERTScore proxy
        "bertscore_model": bs_meta.get("model_type", "—"),
        "bertscore_n_pairs": f"{n_pairs:,}" if isinstance(n_pairs, int) else str(n_pairs),
        "bertscore_n_resamples": f"{bs_boot.get('n_resamples', 0):,}",
        "bertscore_seed": bs_boot.get("seed", "—"),
        "bertscore_mean_bm25": _fmt_float(bs_boot.get("mean_a")),
        "bertscore_mean_rerank": _fmt_float(bs_boot.get("mean_b")),
        "bertscore_delta": f"{bs_boot.get('mean_delta', 0.0):+.4f}",
        "bertscore_ci_low": f"{bs_boot.get('ci_low', 0.0):+.4f}",
        "bertscore_ci_high": f"{bs_boot.get('ci_high', 0.0):+.4f}",
        "bertscore_p_two_sided": (
            "< 0.001"
            if (bs_boot.get("p_two_sided") or 0.0) < 0.001
            else _fmt_float(bs_boot.get("p_two_sided"), 3)
        ),
        "bertscore_win_rate_pct": f"{win * 100:.1f} %",
        "bertscore_tie_rate_pct": f"{tie * 100:.1f} %",
        "bertscore_loss_rate_pct": f"{loss * 100:.1f} %",
        "bertscore_subsample_pct": (
            f"{n_pairs / n_shared * 100:.1f} %" if n_shared else "—"
        ),
        # Regression taxonomy
        "tax_n_sampled": str(tax_n_sampled),
        "tax_seed": taxonomy.get("seed", "—"),
        "tax_truncation_short": str(tax_values["truncation_short"]),
        "tax_truncation_midword": str(tax_values["truncation_midword"]),
        "tax_topic_drift": str(tax_values["topic_drift"]),
        "tax_extractive_passage_bias": str(tax_values["extractive_passage_bias"]),
        "tax_semantic_mismatch": str(tax_values["semantic_mismatch"]),
        "tax_truncation_short_pct": _fmt_pct(
            tax_values["truncation_short"], tax_n_sampled
        ),
        "tax_truncation_midword_pct": _fmt_pct(
            tax_values["truncation_midword"], tax_n_sampled
        ),
        "tax_topic_drift_pct": _fmt_pct(tax_values["topic_drift"], tax_n_sampled),
        "tax_extractive_passage_bias_pct": _fmt_pct(
            tax_values["extractive_passage_bias"], tax_n_sampled
        ),
        "tax_semantic_mismatch_pct": _fmt_pct(
            tax_values["semantic_mismatch"], tax_n_sampled
        ),
        "tax_truncation_total_pct": _fmt_pct(truncation_total, tax_n_sampled),
        "tax_truncation_total_pct_rounded": (
            f"{round(truncation_total / tax_n_sampled * 100):d} %"
            if tax_n_sampled
            else "—"
        ),
    }

    template = (TEMPLATES_DIR / "week06_eval_layer.md").read_text()
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# Cross-week review (static template, no metrics.json dependency)
# --------------------------------------------------------------------------- #


def build_review_all(out_dir: Path) -> str:
    """Render the cross-week progress report.

    The template is hand-written narrative — no per-week metrics.json
    is read here. The only substitution is ``{{generated_at}}`` so each
    rebuild stamps a fresh timestamp at the top.

    ``out_dir`` is unused; kept in the signature for API symmetry with
    the per-week builders.
    """
    template = (TEMPLATES_DIR / "review_all.md").read_text()
    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return _substitute(template, fields)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

def _augmented_env() -> dict:
    """Environment for the pandoc subprocess.

    Adds the user-local Python ``bin/`` directory (where ``pip install --user``
    drops ``weasyprint``) to ``PATH``, and sets ``DYLD_LIBRARY_PATH`` to
    Homebrew's ``/usr/local/lib`` so weasyprint can find its native deps
    (Pango, GLib, fontconfig, ...) on macOS.
    """
    import os
    import sys

    env = os.environ.copy()
    user_bin = Path(sys.prefix).parent / "bin"  # may not exist; cheap to add
    user_site_bin = Path.home() / "Library" / "Python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "bin"
    extra = [str(p) for p in (user_bin, user_site_bin) if p.exists()]
    if extra:
        env["PATH"] = ":".join([*extra, env.get("PATH", "")])
    if sys.platform == "darwin":
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = ":".join(filter(None, ["/usr/local/lib", existing]))
    return env


def try_pdf(md_path: Path) -> tuple[bool, str]:
    """Attempt to render ``md_path`` to PDF via pandoc.

    Tries engines in order: ``xelatex`` → ``pdflatex`` → ``weasyprint``. Returns
    (success, message). Never raises — callers should print the message and
    continue.
    """
    if not shutil.which("pandoc"):
        return False, (
            "pandoc not found on PATH; PDF skipped.\n"
            "  install with e.g. `brew install pandoc` (macOS) or "
            "`sudo apt-get install pandoc texlive-xetex` (Debian/Ubuntu)."
        )

    pdf_path = md_path.with_suffix(".pdf")
    env = _augmented_env()

    # Engine candidates: each is (engine_name, label, extra_args).
    candidates: list[tuple[str, str, list[str]]] = [
        ("xelatex", "xelatex (LaTeX)", ["-V", "mainfont=Helvetica"]),
        ("pdflatex", "pdflatex (LaTeX)", []),
        ("weasyprint", "weasyprint (HTML/CSS)", []),
    ]

    last_err = ""
    for engine, label, extra in candidates:
        # ``shutil.which`` won't see weasyprint in --user bin unless we widen
        # the PATH the same way we do for the subprocess.
        if not shutil.which(engine, path=env.get("PATH")):
            continue
        cmd = [
            "pandoc",
            str(md_path),
            "-o",
            str(pdf_path),
            "--from=gfm",
            f"--pdf-engine={engine}",
            *extra,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, env=env)
            return True, f"Wrote PDF: {pdf_path} (engine: {label})"
        except subprocess.CalledProcessError as exc:
            last_err = (exc.stderr or b"").decode(errors="replace").strip()[:400]

    return False, (
        "pandoc found but no working PDF engine. Install one of:\n"
        "  - xelatex: `brew install --cask basictex` (macOS) "
        "or `sudo apt-get install texlive-xetex` (Debian/Ubuntu)\n"
        "  - weasyprint: `pip install --user weasyprint && brew install pango fontconfig`\n"
        f"{('Last engine stderr: ' + last_err) if last_err else ''}"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week",
        required=True,
        choices=["week01", "week02", "week03", "week04", "week05", "week06", "review_all"],
        help=(
            "Which report to build. ``weekNN`` builds the per-week report "
            "from outputs/weekNN_*/. ``review_all`` builds the cross-week "
            "progress report from the static review_all template."
        ),
    )
    args = parser.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if args.week == "week01":
        out_dir = OUTPUTS_DIR / "week01_eda"  # unused, notebook-driven
        md = build_week01(out_dir)
        out_md = GENERATED_DIR / "week01_eda.md"
    elif args.week == "week02":
        out_dir = OUTPUTS_DIR / "week02_bm25"
        md = build_week02(out_dir)
        out_md = GENERATED_DIR / "week02_bm25.md"
    elif args.week == "week03":
        out_dir = OUTPUTS_DIR / "week03_generation"
        md = build_week03(out_dir)
        out_md = GENERATED_DIR / "week03_generation.md"
    elif args.week == "week04":
        out_dir = OUTPUTS_DIR / "week04_dense"
        md = build_week04(out_dir)
        out_md = GENERATED_DIR / "week04_dense.md"
    elif args.week == "week05":
        out_dir = OUTPUTS_DIR / "week05_reranker"
        md = build_week05(out_dir)
        out_md = GENERATED_DIR / "week05_reranker.md"
    elif args.week == "week06":
        # Two source dirs: analysis (summary + taxonomy) + bertscore proxy.
        # ``out_dir`` is the canonical analysis dir; bertscore JSON is
        # discovered relative to it inside ``build_week06``.
        out_dir = OUTPUTS_DIR / "week06_analysis"
        md = build_week06(out_dir)
        out_md = GENERATED_DIR / "week06_eval_layer.md"
    else:
        # ``review_all``: cross-week narrative, no metrics.json dependency.
        md = build_review_all(OUTPUTS_DIR)
        out_md = GENERATED_DIR / "review_all.md"

    out_md.write_text(md)
    print(f"Wrote markdown: {out_md}")

    pdf_ok, msg = try_pdf(out_md)
    print(msg)
    if not pdf_ok:
        # exit 0; missing pandoc is not a hard failure
        sys.exit(0)


if __name__ == "__main__":
    main()

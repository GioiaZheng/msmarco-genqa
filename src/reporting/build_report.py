"""Render weekly markdown reports (and optionally PDFs) from experiment outputs.

Usage::

    python -m src.reporting.build_report --week week02
    python -m src.reporting.build_report --week week03

The renderer reads:

- ``outputs/<week>/metrics.json``
- ``outputs/<week>/examples.jsonl`` (W2)
- ``outputs/<week>/predictions.jsonl`` (W3)

substitutes ``{{...}}`` placeholders inside ``reports/templates/<week>.md``,
and writes ``reports/generated/<week>.md``. If ``pandoc`` is on PATH it
attempts to produce ``reports/generated/<week>.pdf``; missing pandoc is a
warning, not an error.
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


def _substitute(template: str, fields: dict[str, str]) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


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
    n_eval = metrics.get("n_queries", 0)
    search = timing.get("search")
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

    fields = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": cfg.get("generation", {}).get("model_name", "—"),
        "top_k_passages": cfg.get("generation", {}).get("top_k_passages", "—"),
        "max_input_length": cfg.get("generation", {}).get("max_input_length", "—"),
        "max_new_tokens": cfg.get("generation", {}).get("max_new_tokens", "—"),
        "n_eval": metrics.get("n_predictions", payload.get("n_eval", 0)),
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
# PDF
# --------------------------------------------------------------------------- #

def try_pdf(md_path: Path) -> tuple[bool, str]:
    """Attempt to render `md_path` to PDF via pandoc.

    Returns (success, message). The function never raises — callers should
    print the message and continue.
    """
    if not shutil.which("pandoc"):
        return False, (
            "pandoc not found on PATH; PDF skipped.\n"
            "  install with e.g. `brew install pandoc` (macOS) or "
            "`sudo apt-get install pandoc texlive-xetex` (Debian/Ubuntu)."
        )
    pdf_path = md_path.with_suffix(".pdf")
    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(pdf_path),
        "--from=gfm",
        "--pdf-engine=xelatex",
        "-V",
        "mainfont=Helvetica",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, f"Wrote PDF: {pdf_path}"
    except subprocess.CalledProcessError as exc:
        # Retry without xelatex (in case xelatex is not installed)
        cmd_fallback = [
            "pandoc",
            str(md_path),
            "-o",
            str(pdf_path),
            "--from=gfm",
        ]
        try:
            subprocess.run(cmd_fallback, check=True, capture_output=True)
            return True, f"Wrote PDF: {pdf_path} (default engine, no xelatex)"
        except subprocess.CalledProcessError as exc2:
            return False, (
                "pandoc found but PDF generation failed (likely missing LaTeX engine).\n"
                "  install xelatex with `brew install --cask basictex` (macOS) "
                "or `sudo apt-get install texlive-xetex` (Debian/Ubuntu).\n"
                f"  pandoc stderr: {exc2.stderr.decode().strip()[:400]}"
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
        choices=["week02", "week03"],
        help="Which week's report to build.",
    )
    args = parser.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if args.week == "week02":
        out_dir = OUTPUTS_DIR / "week02_bm25"
        md = build_week02(out_dir)
        out_md = GENERATED_DIR / "week02_bm25.md"
    else:
        out_dir = OUTPUTS_DIR / "week03_generation"
        md = build_week03(out_dir)
        out_md = GENERATED_DIR / "week03_generation.md"

    out_md.write_text(md)
    print(f"Wrote markdown: {out_md}")

    pdf_ok, msg = try_pdf(out_md)
    print(msg)
    if not pdf_ok:
        # exit 0; missing pandoc is not a hard failure
        sys.exit(0)


if __name__ == "__main__":
    main()

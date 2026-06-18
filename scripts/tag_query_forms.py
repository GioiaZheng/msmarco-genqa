"""Tag dev/small queries with a wh-word question-form label.

Complementary axis to the MS MARCO QA v2.1 native ``query_type``
(DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION) already attached
by ``scripts/analyze_generation_rerank.py``. This script adds a second
axis — *question form* (who / what / when / where / why / how / which /
yes_no / other) — and dumps:

- ``query_forms.jsonl``  — one row per qid:
    ``{query_id, query, question_form, ms_marco_query_type}``
- ``summary.json``       — aggregate distribution, plus a crosstab vs
                            the native ``query_type`` so the orthogonality
                            of the two axes is auditable.
- ``summary.md``         — human-readable inspection table.

Default input is ``outputs/generation_analysis/per_query_metrics.jsonl``,
which already carries the native ``query_type`` field. Pass
``--queries-only`` to skip the join and tag dev/small queries
loaded directly via ``ir_datasets`` (useful before the generation
analysis exists, e.g. for a fresh checkout).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.query_form import (
    QUESTION_FORM_CATEGORIES,
    classify_question_form,
)

logger = logging.getLogger("tag_query_forms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-query-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs/generation_analysis/per_query_metrics.jsonl",
        help=(
            "Input jsonl with one row per paired qid (generation-analysis output). "
            "Each row must have 'query_id', 'query'; 'query_type' is "
            "joined into the output when present."
        ),
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help=(
            "Skip the generation-analysis per-query file and tag dev/small "
            "queries directly via ir_datasets. Used when no generation "
            "analysis output exists yet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/query_type_analysis",
    )
    return parser.parse_args()


def load_per_query_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_queries_via_irdatasets() -> dict[str, str]:
    from msmarco_genqa.data.msmarco import load_msmarco_passage

    bundle = load_msmarco_passage(load_corpus=False)
    return bundle.queries


def build_rows(
    *,
    per_query: list[dict[str, Any]] | None,
    queries: dict[str, str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if per_query is not None:
        for r in per_query:
            qid = str(r["query_id"])
            query = r.get("query") or ""
            form = classify_question_form(query)
            out.append(
                {
                    "query_id": qid,
                    "query": query,
                    "question_form": form,
                    "ms_marco_query_type": r.get("query_type"),
                }
            )
        return out
    assert queries is not None
    for qid, query in queries.items():
        out.append(
            {
                "query_id": str(qid),
                "query": query,
                "question_form": classify_question_form(query),
                "ms_marco_query_type": None,
            }
        )
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    form_counts = Counter(r["question_form"] for r in rows)
    distribution = {
        cat: {
            "count": form_counts.get(cat, 0),
            "share": form_counts.get(cat, 0) / n if n else 0.0,
        }
        for cat in QUESTION_FORM_CATEGORIES
    }

    # Crosstab vs MS MARCO native query_type (only rows that have one).
    crosstab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    qt_counts: Counter[str] = Counter()
    for r in rows:
        qt = r.get("ms_marco_query_type")
        if not qt:
            continue
        crosstab[r["question_form"]][qt] += 1
        qt_counts[qt] += 1
    crosstab_serialisable = {
        form: dict(inner) for form, inner in crosstab.items()
    }

    return {
        "n_total": n,
        "categories": list(QUESTION_FORM_CATEGORIES),
        "distribution": distribution,
        "ms_marco_query_type_counts": dict(qt_counts),
        "crosstab_form_x_ms_marco_query_type": crosstab_serialisable,
    }


def render_markdown(rows: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Query-form tagging — generation-analysis complement")
    lines.append("")
    lines.append(
        f"Tagged **{agg['n_total']}** queries with a wh-word / yes-no / "
        "other label. This is a *question-form* axis, complementary to "
        "the MS MARCO QA v2.1 native `query_type` (which classifies by "
        "*answer type*: DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION)."
    )
    lines.append("")
    lines.append("## 1. Distribution")
    lines.append("")
    lines.append("| question_form | n | share |")
    lines.append("|---|---:|---:|")
    for cat in QUESTION_FORM_CATEGORIES:
        d = agg["distribution"][cat]
        lines.append(f"| `{cat}` | {d['count']} | {d['share']:.1%} |")
    lines.append(f"| **total** | **{agg['n_total']}** | 100 % |")
    lines.append("")

    if agg["ms_marco_query_type_counts"]:
        lines.append("## 2. Crosstab — question_form × MS MARCO `query_type`")
        lines.append("")
        qt_keys = sorted(agg["ms_marco_query_type_counts"].keys())
        header = "| form | " + " | ".join(qt_keys) + " | total |"
        sep = "|---|" + "|".join(["---:"] * (len(qt_keys) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for cat in QUESTION_FORM_CATEGORIES:
            inner = agg["crosstab_form_x_ms_marco_query_type"].get(cat, {})
            total = sum(inner.values())
            cells = [str(inner.get(qt, 0)) for qt in qt_keys]
            lines.append(f"| `{cat}` | " + " | ".join(cells) + f" | {total} |")
        col_totals = [
            str(agg["ms_marco_query_type_counts"][qt]) for qt in qt_keys
        ]
        grand_total = sum(agg["ms_marco_query_type_counts"].values())
        lines.append("| **total** | " + " | ".join(col_totals) + f" | {grand_total} |")
        lines.append("")
        lines.append(
            "Read the row totals as the question-form distribution restricted "
            "to queries with a native MS MARCO `query_type`. Off-diagonal mass "
            "is the load-bearing observation: the two axes are not redundant."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.queries_only:
        logger.info("Loading dev/small queries via ir_datasets...")
        queries = load_queries_via_irdatasets()
        logger.info("Loaded %d queries.", len(queries))
        rows = build_rows(per_query=None, queries=queries)
    else:
        if not args.per_query_metrics.exists():
            raise SystemExit(
                f"Missing {args.per_query_metrics}. Re-run "
                "scripts/analyze_generation_rerank.py first, or pass "
                "--queries-only to tag dev/small directly."
            )
        logger.info("Loading per-query rows from %s...", args.per_query_metrics)
        per_query = load_per_query_rows(args.per_query_metrics)
        logger.info("Loaded %d rows.", len(per_query))
        rows = build_rows(per_query=per_query, queries=None)

    agg = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.output_dir / "query_forms.jsonl"
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", jsonl_path, len(rows))

    json_path = args.output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Wrote %s", json_path)

    md_path = args.output_dir / "summary.md"
    md_path.write_text(render_markdown(rows, agg))
    logger.info("Wrote %s", md_path)

    print()
    print(f"=== Query-form tagging (n={agg['n_total']}) ===")
    print(f"  {'form':10s}  {'n':>5s}  {'share':>6s}")
    for cat in QUESTION_FORM_CATEGORIES:
        d = agg["distribution"][cat]
        print(f"  {cat:10s}  {d['count']:>5d}  {d['share']:>6.1%}")


if __name__ == "__main__":
    main()

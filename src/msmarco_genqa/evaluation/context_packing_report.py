"""Reports for baseline vs compressed RAG generation prompts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.generation import exact_match, token_f1


class PredictionPairingError(ValueError):
    """Raised when two prediction files cannot be compared safely."""


def load_prediction_jsonl(path: Path | str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_number}: invalid JSON") from exc
            qid = str(row.get("query_id", ""))
            if not qid:
                raise ValueError(f"{p}:{line_number}: missing query_id")
            if qid in rows:
                raise ValueError(f"{p}:{line_number}: duplicate query_id {qid!r}")
            rows[qid] = row
    return rows


def build_context_packing_report(
    baseline_rows: Mapping[str, Mapping[str, Any]],
    compressed_rows: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str = "baseline",
    compressed_name: str = "compressed",
) -> dict[str, Any]:
    matched_qids = sorted(set(baseline_rows) & set(compressed_rows))
    if not matched_qids:
        raise PredictionPairingError("prediction files do not share any query ids")

    per_query: list[dict[str, Any]] = []
    for qid in matched_qids:
        baseline = baseline_rows[qid]
        compressed = compressed_rows[qid]
        baseline_refs = _references(baseline)
        compressed_refs = _references(compressed)
        if baseline_refs != compressed_refs:
            raise PredictionPairingError(f"references differ for query_id {qid!r}")
        baseline_prediction = str(baseline.get("prediction", ""))
        compressed_prediction = str(compressed.get("prediction", ""))
        baseline_context_chars = _context_chars(baseline)
        compressed_context_chars = _context_chars(compressed)
        row = {
            "query_id": qid,
            "query": baseline.get("query") or compressed.get("query") or "",
            "references": baseline_refs,
            "baseline": {
                "name": baseline_name,
                "prediction": baseline_prediction,
                "context_chars": baseline_context_chars,
                "token_f1": token_f1(baseline_prediction, baseline_refs),
                "exact_match": exact_match(baseline_prediction, baseline_refs),
            },
            "compressed": {
                "name": compressed_name,
                "prediction": compressed_prediction,
                "context_chars": compressed_context_chars,
                "token_f1": token_f1(compressed_prediction, baseline_refs),
                "exact_match": exact_match(compressed_prediction, baseline_refs),
                "context_packing": compressed.get("context_packing"),
            },
        }
        row["deltas"] = {
            "context_chars": compressed_context_chars - baseline_context_chars,
            "context_compression_ratio": _safe_ratio(
                compressed_context_chars,
                baseline_context_chars,
            ),
            "token_f1": row["compressed"]["token_f1"] - row["baseline"]["token_f1"],
            "exact_match": row["compressed"]["exact_match"]
            - row["baseline"]["exact_match"],
        }
        per_query.append(row)

    summary = {
        "baseline_name": baseline_name,
        "compressed_name": compressed_name,
        "coverage": {
            "n_baseline_qids": len(baseline_rows),
            "n_compressed_qids": len(compressed_rows),
            "n_matched_qids": len(matched_qids),
            "n_baseline_only_qids": len(set(baseline_rows) - set(compressed_rows)),
            "n_compressed_only_qids": len(set(compressed_rows) - set(baseline_rows)),
        },
        "metrics": {
            baseline_name: _aggregate_side(per_query, "baseline"),
            compressed_name: _aggregate_side(per_query, "compressed"),
            "delta": {
                "mean_context_chars": _mean(
                    [row["deltas"]["context_chars"] for row in per_query]
                ),
                "mean_context_compression_ratio": _mean(
                    [row["deltas"]["context_compression_ratio"] for row in per_query]
                ),
                "mean_token_f1": _mean([row["deltas"]["token_f1"] for row in per_query]),
                "mean_exact_match": _mean(
                    [row["deltas"]["exact_match"] for row in per_query]
                ),
            },
        },
        "per_query": per_query,
    }
    return summary


def render_context_packing_markdown(report: Mapping[str, Any]) -> str:
    baseline = str(report["baseline_name"])
    compressed = str(report["compressed_name"])
    metric_block = report["metrics"]
    lines = [
        "# Context packing comparison",
        "",
        f"- Baseline: `{baseline}`",
        f"- Compressed: `{compressed}`",
        f"- Matched qids: **{report['coverage']['n_matched_qids']}**",
        "",
        "| metric | baseline | compressed | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ("mean_context_chars", "mean_token_f1", "mean_exact_match"):
        b = float(metric_block[baseline][key])
        c = float(metric_block[compressed][key])
        d = float(metric_block["delta"][key])
        lines.append(f"| {key} | {b:.4f} | {c:.4f} | {d:+.4f} |")
    lines.append(
        "| mean_context_compression_ratio | 1.0000 | "
        f"{float(metric_block['delta']['mean_context_compression_ratio']):.4f} | "
        f"{float(metric_block['delta']['mean_context_compression_ratio']) - 1.0:+.4f} |"
    )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| field | value |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(report["coverage"].items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append(
        "Metric deltas are compressed minus baseline on the matched qid set. "
        "The report uses context character count as a deterministic token-cost proxy."
    )
    return "\n".join(lines) + "\n"


def write_context_packing_outputs(
    report: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query = list(report.get("per_query", []))
    summary = {key: value for key, value in report.items() if key != "per_query"}
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    with (output_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    (output_dir / "report.md").write_text(
        render_context_packing_markdown(summary),
        encoding="utf-8",
    )


def _references(row: Mapping[str, Any]) -> list[str]:
    refs = row.get("references") or []
    if isinstance(refs, str):
        return [refs]
    return [str(ref) for ref in refs]


def _context_chars(row: Mapping[str, Any]) -> int:
    passages = row.get("passages") or []
    if not isinstance(passages, Sequence) or isinstance(passages, str):
        return 0
    return len(" ".join(str(passage).strip() for passage in passages if str(passage).strip()))


def _aggregate_side(rows: Sequence[Mapping[str, Any]], side: str) -> dict[str, float]:
    return {
        "mean_context_chars": _mean([row[side]["context_chars"] for row in rows]),
        "mean_token_f1": _mean([row[side]["token_f1"] for row in rows]),
        "mean_exact_match": _mean([row[side]["exact_match"] for row in rows]),
    }


def _mean(values: Sequence[float | int]) -> float:
    return mean([float(value) for value in values]) if values else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0

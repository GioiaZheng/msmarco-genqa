"""Check that machine-readable headline metrics match RESULTS.md.

This keeps ``metadata.json`` aligned with the public result summary without
re-running the expensive MS MARCO pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "metadata.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "RESULTS.md"
TOLERANCE = 1e-4


def _to_float(value: str) -> float:
    return float(value.strip().replace(",", "").lstrip("+"))


def _section(markdown: str, heading: str) -> str:
    match = re.search(rf"^##+\s+{re.escape(heading)}\s*$", markdown, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"section not found: {heading}")

    tail = markdown[match.end() :]
    next_heading = re.search(r"\n#{2,3} ", tail)
    if next_heading:
        return tail[: next_heading.start()]
    return tail


def _metric_row(section_text: str, metric_name: str) -> list[str]:
    pattern = rf"^\|\s*{re.escape(metric_name)}\s*\|(.+)\|$"
    match = re.search(pattern, section_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"metric row not found: {metric_name}")
    return [part.strip() for part in match.group(1).split("|")]


def collect_results_metrics(results_markdown: str) -> dict[str, Any]:
    reliability = _section(results_markdown, "Statistical Reliability")
    dense = _section(results_markdown, "Dense Retrieval on 50k Qrels-Anchored Sample")
    rerank = _section(results_markdown, "Cross-Encoder Reranking")

    rouge_delta, rouge_ci = _metric_row(reliability, "ROUGE-L")
    token_f1_delta, token_f1_ci = _metric_row(reliability, "Token-F1")
    ci_match = re.fullmatch(r"\[\+?([0-9.]+),\s*\+?([0-9.]+)\]", token_f1_ci)
    if ci_match is None:
        raise ValueError("Token-F1 confidence interval has unexpected format")

    bm25_sample_mrr, dense_mrr = _metric_row(dense, "MRR@10")
    dense_rerank_mrr = _metric_row(rerank, "MRR@10")

    return {
        "bm25_to_reranked_t5_small": {
            "token_f1_delta": _to_float(token_f1_delta),
            "rouge_l_delta": _to_float(rouge_delta),
            "paired_bootstrap_ci_token_f1": [
                _to_float(ci_match.group(1)),
                _to_float(ci_match.group(2)),
            ],
        },
        "dense_retrieval_sample": {
            "bm25_sample_mrr_at_10": _to_float(bm25_sample_mrr),
            "dense_mrr_at_10": _to_float(dense_mrr),
            "cross_encoder_mrr_at_10": _to_float(dense_rerank_mrr[1]),
        },
    }


def _compare_float(path: str, recorded: float, expected: float) -> list[str]:
    if abs(recorded - expected) <= TOLERANCE:
        return []
    return [f"{path}: metadata={recorded:.4f}, RESULTS.md={expected:.4f}"]


def _compare_list(path: str, recorded: list[float], expected: list[float]) -> list[str]:
    if len(recorded) != len(expected):
        return [f"{path}: metadata length={len(recorded)}, RESULTS.md length={len(expected)}"]

    failures: list[str] = []
    for idx, (got, want) in enumerate(zip(recorded, expected)):
        failures.extend(_compare_float(f"{path}[{idx}]", float(got), float(want)))
    return failures


def compare_headline_metrics(metadata: dict[str, Any], results_metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    recorded = metadata.get("headline_metrics", {})

    for group_name, expected_group in results_metrics.items():
        recorded_group = recorded.get(group_name, {})
        if not isinstance(recorded_group, dict):
            failures.append(f"{group_name}: missing from metadata.json")
            continue

        for key, expected_value in expected_group.items():
            path = f"{group_name}.{key}"
            if key not in recorded_group:
                failures.append(f"{path}: missing from metadata.json")
                continue

            recorded_value = recorded_group[key]
            if isinstance(expected_value, list):
                failures.extend(_compare_list(path, list(recorded_value), expected_value))
            else:
                failures.extend(_compare_float(path, float(recorded_value), float(expected_value)))

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    results_metrics = collect_results_metrics(args.results.read_text(encoding="utf-8"))
    failures = compare_headline_metrics(metadata, results_metrics)

    if failures:
        print("Headline metric consistency check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Headline metric consistency check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

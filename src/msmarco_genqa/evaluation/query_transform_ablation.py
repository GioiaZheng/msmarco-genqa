"""Query transformation ablation reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MetricMap = dict[str, float]


def load_json_mapping(path: Path | str) -> dict[str, Any]:
    """Load a JSON object from disk."""

    p = Path(path)
    with p.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{p}: expected a JSON object")
    return payload


def extract_numeric_metrics(payload: Mapping[str, Any]) -> MetricMap:
    """Extract numeric metrics from common repository metric JSON shapes.

    The preferred input is the ``metrics.json`` emitted by
    ``mgq-retrieval-report evaluate``, where metric names live directly under
    ``payload["metrics"]``. For convenience, nested metric groups are flattened
    as ``group.metric``.
    """

    raw = payload.get("metrics", payload)
    if not isinstance(raw, Mapping):
        return {}

    metrics: MetricMap = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
        elif isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if isinstance(child_value, (int, float)):
                    metrics[f"{key}.{child_key}"] = float(child_value)
    return dict(sorted(metrics.items()))


def build_query_transform_ablation(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    metrics: Mapping[str, Mapping[str, float]] | None = None,
    baseline_method: str = "none",
) -> dict[str, Any]:
    """Build a method-level ablation report from summaries and optional metrics."""

    if baseline_method not in summaries:
        raise ValueError(f"baseline method {baseline_method!r} is missing from summaries")

    rows: list[dict[str, Any]] = []
    metric_by_method = metrics or {}
    metric_keys = sorted({key for values in metric_by_method.values() for key in values})

    for method, summary in summaries.items():
        reported_method = str(summary.get("method", method))
        if reported_method != method:
            raise ValueError(
                f"summary for {method!r} reports method {reported_method!r}; "
                "method labels must match"
            )
        row = {
            "method": method,
            "config_hash": str(summary.get("config_hash", "")),
            "n_queries": int(summary.get("n_queries", 0)),
            "n_changed": int(summary.get("n_changed", 0)),
            "changed_fraction": float(summary.get("changed_fraction", 0.0)),
            "cache_hit": bool(summary.get("cache_hit", False)),
            "metrics": dict(sorted(metric_by_method.get(method, {}).items())),
        }
        rows.append(row)

    baseline_metrics = metric_by_method.get(baseline_method, {})
    deltas: list[dict[str, Any]] = []
    for row in rows:
        method = str(row["method"])
        if method == baseline_method:
            continue
        for key in metric_keys:
            if key not in baseline_metrics or key not in row["metrics"]:
                continue
            value = float(row["metrics"][key])
            baseline_value = float(baseline_metrics[key])
            deltas.append(
                {
                    "method": method,
                    "metric": key,
                    "baseline": baseline_value,
                    "value": value,
                    "delta": value - baseline_value,
                }
            )

    return {
        "baseline_method": baseline_method,
        "methods": [row["method"] for row in rows],
        "metric_keys": metric_keys,
        "runs": rows,
        "metric_deltas_vs_baseline": deltas,
        "notes": [
            "Transformation summaries compare query rewriting coverage.",
            "Metric deltas are reported only when matched retrieval metrics are provided.",
        ],
    }


def render_query_transform_ablation_markdown(report: Mapping[str, Any]) -> str:
    """Render a query transformation ablation report as Markdown."""

    metric_keys = [str(key) for key in report.get("metric_keys", [])]
    lines = [
        "# Query transformation ablation report",
        "",
        f"- Baseline method: `{report['baseline_method']}`",
        f"- Methods: {', '.join(f'`{method}`' for method in report['methods'])}",
        "",
        "## Method coverage",
        "",
    ]

    columns = [
        "method",
        "config hash",
        "queries",
        "changed",
        "changed %",
        *metric_keys,
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|")
    for row in report["runs"]:
        values = [
            f"`{row['method']}`",
            f"`{row['config_hash']}`" if row["config_hash"] else "",
            str(row["n_queries"]),
            str(row["n_changed"]),
            f"{100.0 * float(row['changed_fraction']):.2f}",
        ]
        for key in metric_keys:
            metric_value = row["metrics"].get(key)
            values.append("--" if metric_value is None else f"{float(metric_value):.4f}")
        lines.append("| " + " | ".join(values) + " |")

    deltas = list(report.get("metric_deltas_vs_baseline", []))
    if deltas:
        lines.extend(
            [
                "",
                "## Metric deltas vs baseline",
                "",
                "| method | metric | baseline | value | delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in deltas:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['method']}`",
                        str(row["metric"]),
                        f"{float(row['baseline']):.4f}",
                        f"{float(row['value']):.4f}",
                        f"{float(row['delta']):+.4f}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_ablation_outputs(output_dir: Path, report: Mapping[str, Any]) -> list[Path]:
    """Write machine-readable and Markdown ablation artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation.json"
    md_path = output_dir / "report.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path.write_text(render_query_transform_ablation_markdown(report), encoding="utf-8")
    return [json_path, md_path]


def read_method_mapping(
    specs: Sequence[str],
    *,
    value_name: str,
) -> dict[str, Path]:
    """Parse ``method=path`` CLI specs into a mapping."""

    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{value_name} spec must be method=path, got {spec!r}")
        method, raw_path = spec.split("=", 1)
        method = method.strip()
        raw_path = raw_path.strip()
        if not method:
            raise ValueError(f"{value_name} spec has an empty method: {spec!r}")
        if not raw_path:
            raise ValueError(f"{value_name} spec has an empty path for {method!r}")
        if method in parsed:
            raise ValueError(f"duplicate {value_name} method {method!r}")
        parsed[method] = Path(raw_path)
    return parsed

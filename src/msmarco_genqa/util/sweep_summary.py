"""Sweep-level summaries for local experiment-tracking events."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


EVENT_FILE = "events.jsonl"


def discover_event_files(paths: Sequence[Path | str]) -> list[Path]:
    """Return sorted, de-duplicated ``events.jsonl`` files under paths."""

    discovered: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            if path.name != EVENT_FILE:
                raise ValueError(f"expected an {EVENT_FILE} file, got {path}")
            discovered[str(path.resolve())] = path
            continue
        if path.is_dir():
            for event_path in sorted(path.rglob(EVENT_FILE)):
                discovered[str(event_path.resolve())] = event_path
            continue
        raise ValueError(f"tracking path not found: {path}")
    return [discovered[key] for key in sorted(discovered)]


def load_tracking_events(path: Path | str) -> list[dict[str, Any]]:
    """Load and validate JSONL tracking events from one run."""

    event_path = Path(path)
    rows: list[dict[str, Any]] = []
    with event_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{event_path}:{line_no}: invalid JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{event_path}:{line_no}: expected a JSON object")
            if not isinstance(record.get("kind"), str):
                raise ValueError(f"{event_path}:{line_no}: missing string kind")
            if not isinstance(record.get("run_name"), str):
                raise ValueError(f"{event_path}:{line_no}: missing string run_name")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"{event_path}:{line_no}: missing object payload")
            rows.append(record)
    if not rows:
        raise ValueError(f"{event_path}: no tracking events found")
    return rows


def summarize_event_file(path: Path | str) -> dict[str, Any]:
    """Collapse one run's tracking events into tags, params, metrics, and artifacts."""

    event_path = Path(path)
    events = load_tracking_events(event_path)
    run_name = str(events[0]["run_name"])
    tags: dict[str, Any] = {}
    params: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    metric_step: int | None = None

    for event in events:
        run_name = str(event.get("run_name") or run_name)
        kind = str(event["kind"])
        payload = event["payload"]
        if kind == "run":
            raw_tags = payload.get("tags", {})
            if isinstance(raw_tags, Mapping):
                tags.update({str(key): value for key, value in raw_tags.items()})
        elif kind == "params":
            params.update({str(key): value for key, value in payload.items()})
        elif kind == "metrics":
            raw_metrics = payload.get("metrics", {})
            if isinstance(raw_metrics, Mapping):
                metrics.update({str(key): value for key, value in raw_metrics.items()})
            if isinstance(payload.get("step"), int):
                metric_step = int(payload["step"])
        elif kind == "artifact":
            artifacts.append(
                {
                    "name": payload.get("name") or f"artifact_{len(artifacts) + 1}",
                    "path": str(payload.get("path", "")),
                }
            )

    return {
        "run_name": run_name,
        "event_file": str(event_path),
        "tags": dict(sorted(tags.items())),
        "params": dict(sorted(params.items())),
        "metrics": dict(sorted(metrics.items())),
        "artifacts": artifacts,
        "last_metric_step": metric_step,
    }


def build_sweep_summary(
    event_files: Sequence[Path | str],
    *,
    sweep_name: str | None = None,
) -> dict[str, Any]:
    """Build a run-by-run summary table from tracking event files."""

    runs = [summarize_event_file(path) for path in event_files]
    runs.sort(key=lambda row: str(row["run_name"]))
    return {
        "sweep_name": sweep_name,
        "n_runs": len(runs),
        "runs": runs,
        "columns": {
            "tags": _sorted_union(run["tags"] for run in runs),
            "params": _sorted_union(run["params"] for run in runs),
            "metrics": _sorted_union(run["metrics"] for run in runs),
        },
    }


def flatten_summary_rows(summary: Mapping[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Return stable CSV/Markdown columns plus flattened run rows."""

    columns = summary.get("columns", {})
    tag_cols = [str(col) for col in columns.get("tags", [])]
    param_cols = [str(col) for col in columns.get("params", [])]
    metric_cols = [str(col) for col in columns.get("metrics", [])]
    fields = (
        ["run_name", "event_file"]
        + [f"tag.{col}" for col in tag_cols]
        + [f"param.{col}" for col in param_cols]
        + [f"metric.{col}" for col in metric_cols]
        + ["artifact_count", "artifacts"]
    )

    rows: list[dict[str, str]] = []
    for run in summary.get("runs", []):
        tags = run.get("tags", {})
        params = run.get("params", {})
        metrics = run.get("metrics", {})
        artifacts = run.get("artifacts", [])
        row = {
            "run_name": _format_value(run.get("run_name", "")),
            "event_file": _format_value(run.get("event_file", "")),
        }
        row.update({f"tag.{col}": _format_value(tags.get(col, "")) for col in tag_cols})
        row.update({f"param.{col}": _format_value(params.get(col, "")) for col in param_cols})
        row.update({f"metric.{col}": _format_value(metrics.get(col, "")) for col in metric_cols})
        row["artifact_count"] = str(len(artifacts))
        row["artifacts"] = "; ".join(
            f"{artifact.get('name')}={artifact.get('path')}" for artifact in artifacts
        )
        rows.append(row)
    return fields, rows


def render_sweep_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown comparison table."""

    fields, rows = flatten_summary_rows(summary)
    title = str(summary.get("sweep_name") or "tracked sweep")
    lines = [
        f"# {title} summary",
        "",
        f"- Runs: {summary.get('n_runs', len(rows))}",
        "",
    ]
    if not rows:
        lines.append("_No runs found._")
        return "\n".join(lines) + "\n"

    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_sweep_summary(
    event_files: Sequence[Path | str],
    output_dir: Path | str,
    *,
    sweep_name: str | None = None,
) -> list[Path]:
    """Write JSON, CSV, and Markdown summaries for tracked sweep runs."""

    summary = build_sweep_summary(event_files, sweep_name=sweep_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "sweep_summary.json"
    csv_path = out_dir / "sweep_summary.csv"
    md_path = out_dir / "sweep_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    fields, rows = flatten_summary_rows(summary)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(render_sweep_summary_markdown(summary), encoding="utf-8")
    return [json_path, csv_path, md_path]


def _sorted_union(mappings: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({str(key) for mapping in mappings for key in mapping})


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

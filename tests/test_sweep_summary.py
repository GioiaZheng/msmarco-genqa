from __future__ import annotations

import csv
import json

from msmarco_genqa.cli import sweep_summary as cli
from msmarco_genqa.util.sweep_summary import (
    build_sweep_summary,
    discover_event_files,
    flatten_summary_rows,
    write_sweep_summary,
)
from msmarco_genqa.util.tracking import ExperimentTracker


def _write_run(tmp_path, method: str, metric: float):
    run_dir = tmp_path / "tracking" / method
    with ExperimentTracker(
        backend="jsonl",
        output_dir=run_dir,
        run_name=f"query-transform-{method}",
        tags={"sweep": "query-transform", "arm": method},
    ) as tracker:
        tracker.log_params({"method": method, "config_hash": f"{method}-hash"})
        tracker.log_metrics({"mrr@10": metric})
        tracker.log_artifact(f"outputs/query_transform/{method}/summary.json", name="query_summary")
    return run_dir / "events.jsonl"


def test_build_sweep_summary_flattens_local_jsonl_runs(tmp_path):
    event_files = [
        _write_run(tmp_path, "none", 0.2),
        _write_run(tmp_path, "normalize", 0.25),
    ]

    summary = build_sweep_summary(event_files, sweep_name="query-transform")
    fields, rows = flatten_summary_rows(summary)

    assert summary["n_runs"] == 2
    assert "metric.mrr@10" in fields
    assert rows[0]["tag.arm"] == "none"
    assert rows[1]["metric.mrr@10"] == "0.25"
    assert rows[1]["artifact_count"] == "1"


def test_write_sweep_summary_outputs_json_csv_and_markdown(tmp_path):
    event_files = [_write_run(tmp_path, "none", 0.2)]

    paths = write_sweep_summary(
        event_files,
        tmp_path / "summary",
        sweep_name="query-transform",
    )

    assert {path.name for path in paths} == {
        "sweep_summary.csv",
        "sweep_summary.json",
        "sweep_summary.md",
    }
    payload = json.loads((tmp_path / "summary" / "sweep_summary.json").read_text())
    assert payload["runs"][0]["metrics"]["mrr@10"] == 0.2

    with (tmp_path / "summary" / "sweep_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["param.method"] == "none"
    assert "# query-transform summary" in (
        tmp_path / "summary" / "sweep_summary.md"
    ).read_text(encoding="utf-8")


def test_discover_event_files_rejects_non_tracking_file(tmp_path):
    other = tmp_path / "metrics.json"
    other.write_text("{}", encoding="utf-8")

    try:
        discover_event_files([other])
    except ValueError as exc:
        assert "expected an events.jsonl file" in str(exc)
    else:
        raise AssertionError("expected non-tracking file to fail")


def test_cli_writes_summary_from_tracking_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    _write_run(tmp_path, "none", 0.2)

    cli.main(
        [
            "tracking",
            "--name",
            "query-transform",
            "--output-dir",
            "outputs/sweep_summary",
        ]
    )

    assert (tmp_path / "outputs" / "sweep_summary" / "sweep_summary.json").exists()
    assert "sweep summary: 1 runs" in capsys.readouterr().out

from __future__ import annotations

import json

from msmarco_genqa.util.tracking import ExperimentTracker


def test_jsonl_tracker_writes_params_metrics_and_artifact(tmp_path):
    with ExperimentTracker(
        backend="jsonl",
        output_dir=tmp_path,
        run_name="smoke",
        tags={"sweep": "fixture", "arm": "baseline"},
    ) as tracker:
        tracker.log_params({"model": "t5-small"})
        tracker.log_metrics({"token_f1": 0.37}, step=2)
        tracker.log_artifact("outputs/run/metrics.json")

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["run", "params", "metrics", "artifact"]
    assert rows[0]["payload"]["tags"]["sweep"] == "fixture"
    assert rows[1]["payload"]["model"] == "t5-small"
    assert rows[2]["payload"]["metrics"]["token_f1"] == 0.37
    assert rows[2]["payload"]["step"] == 2


def test_tracker_rejects_unknown_backend(tmp_path):
    try:
        ExperimentTracker(backend="unknown", output_dir=tmp_path)
    except ValueError as exc:
        assert "unknown tracking backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")

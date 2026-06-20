from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


_CHECK_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_fixture_headline_metrics.py"
)
_spec = importlib.util.spec_from_file_location("check_fixture_headline_metrics", _CHECK_PATH)
check_fixture_headline_metrics = importlib.util.module_from_spec(_spec)
sys.modules["check_fixture_headline_metrics"] = check_fixture_headline_metrics
_spec.loader.exec_module(check_fixture_headline_metrics)  # type: ignore[union-attr]


def test_fixture_metrics_match_committed_goldens():
    config = check_fixture_headline_metrics._load_json(
        check_fixture_headline_metrics.DEFAULT_CONFIG
    )
    golden = check_fixture_headline_metrics._load_json(
        check_fixture_headline_metrics.DEFAULT_GOLDEN
    )
    observed = check_fixture_headline_metrics.compute_observed(config)
    assert check_fixture_headline_metrics.compare_to_golden(observed, golden) == []


def test_fixture_metric_drift_is_reported():
    config = check_fixture_headline_metrics._load_json(
        check_fixture_headline_metrics.DEFAULT_CONFIG
    )
    golden = check_fixture_headline_metrics._load_json(
        check_fixture_headline_metrics.DEFAULT_GOLDEN
    )
    drifted = copy.deepcopy(golden)
    drifted["metrics"]["generation.mean_token_f1"]["expected"] = 0.0

    observed = check_fixture_headline_metrics.compute_observed(config)
    failures = check_fixture_headline_metrics.compare_to_golden(observed, drifted)

    assert len(failures) == 1
    assert "generation.mean_token_f1" in failures[0]
    assert "observed" in failures[0]


def test_dump_observed_does_not_modify_goldens(capsys):
    before = check_fixture_headline_metrics.DEFAULT_GOLDEN.read_text(encoding="utf-8")

    rc = check_fixture_headline_metrics.main(["--dump-observed"])

    after = check_fixture_headline_metrics.DEFAULT_GOLDEN.read_text(encoding="utf-8")
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert before == after
    assert payload["seed"] == 42
    assert "retrieval.mrr@10" in payload["metrics"]

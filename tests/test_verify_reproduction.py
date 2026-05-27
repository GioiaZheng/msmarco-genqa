"""Tests for scripts/verify_reproduction.py.

The verifier is load-bearing for the reproducibility-protocol contract:
a broken verifier silently lets non-reproducible runs pass audit. So we
test classification on synthetic inputs (PASS / FAIL for each of the
five checks).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml as _yaml

from msmarco_genqa.util.manifest import (
    SCHEMA_VERSION,
    compute_resolved_config_hash,
)

# Load the script as a module without exposing it under a fragile path.
_VERIFY_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_reproduction.py"
_spec = importlib.util.spec_from_file_location("verify_reproduction", _VERIFY_PATH)
verify_reproduction = importlib.util.module_from_spec(_spec)
sys.modules["verify_reproduction"] = verify_reproduction
_spec.loader.exec_module(verify_reproduction)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Fixtures: build a v2-compliant run directory on disk.
# --------------------------------------------------------------------------- #


def _build_compliant_run(
    output_dir: Path,
    *,
    schema_override: str | None = None,
    cfg_override: dict[str, Any] | None = None,
    metrics_blob_override: dict[str, Any] | None = None,
    extra_override: dict[str, Any] | None = None,
) -> Path:
    """Lay down a synthetic run dir that the verifier should classify as
    PASS by default. Each ``*_override`` knob lets a specific test
    deviate one aspect of the recorded run to trigger one FAIL check."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = cfg_override or {
        "seed": 42,
        "retrieval": {"backend": "bm25s", "k1": 1.5, "b": 0.75, "top_k": 1000},
    }
    resolved_path = output_dir / "resolved_config.yaml"
    with open(resolved_path, "w") as f:
        _yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

    metrics_blob = metrics_blob_override or {
        "task": "retrieval",
        "metrics": {"mrr@10": 0.1703, "recall@100": 0.6212},
        "sampling": {"is_sampled": False},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_blob, indent=2))

    # Recompute the metrics.json sha256_16 so the manifest record is honest.
    metrics_sha = verify_reproduction._file_sha256_16(metrics_path)

    extra = {
        "seed": 42,
        "resolved_config_hash": compute_resolved_config_hash(cfg),
        "data_fingerprint": "a" * 64,
        "env_fingerprint": "b" * 64,
    }
    if extra_override:
        extra.update(extra_override)

    manifest = {
        "schema": schema_override or SCHEMA_VERSION,
        "timestamp_utc": "2026-05-27T12:00:00+00:00",
        "git": {"commit": "deadbeef1234", "dirty": False},
        "command": ["python", "experiments/run_retrieval.py"],
        "config": [],
        "dependencies": [],
        "outputs": [{"path": f"{output_dir.name}/metrics.json", "sha256_16": metrics_sha}],
        "python": {"version": "3.10.0", "executable": "python3", "platform": "darwin"},
        "extra": extra,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_verify_compliant_run_returns_zero_failures(tmp_path: Path):
    """A freshly built v2-compliant run dir should pass all five checks
    (one WARN on git mismatch since tmp_path's manifest commit != HEAD,
    but that does not count as a failure)."""
    out = _build_compliant_run(tmp_path / "outputs" / "fake_run")
    assert verify_reproduction.verify_one(out) == 0


def test_verify_missing_manifest_returns_one(tmp_path: Path):
    """No manifest at all → exactly 1 failure (the first check shortcuts)."""
    empty = tmp_path / "outputs" / "no_manifest"
    empty.mkdir(parents=True)
    assert verify_reproduction.verify_one(empty) == 1


def test_verify_wrong_schema_fails_schema_check(tmp_path: Path):
    """Schema v1 (or any non-v2 string) must fail check #1."""
    out = _build_compliant_run(
        tmp_path / "outputs" / "v1_run",
        schema_override="msmarco-genqa.manifest.v1",
    )
    n = verify_reproduction.verify_one(out)
    assert n >= 1  # schema check fails; others may still pass


def test_verify_missing_required_field_fails(tmp_path: Path):
    """If extra.env_fingerprint is None, check #2 must surface a FAIL."""
    out = _build_compliant_run(
        tmp_path / "outputs" / "no_env_fp",
        extra_override={"env_fingerprint": None},
    )
    assert verify_reproduction.verify_one(out) >= 1


def test_verify_resolved_config_drift_fails(tmp_path: Path):
    """If the resolved_config.yaml on disk drifts from what the recorded
    hash represents, check #3 must FAIL."""
    out = _build_compliant_run(tmp_path / "outputs" / "drift")
    # Mutate the yaml so its content no longer matches the recorded hash.
    yaml_path = out / "resolved_config.yaml"
    cfg = _yaml.safe_load(yaml_path.read_text())
    cfg["retrieval"]["k1"] = 99.0
    with open(yaml_path, "w") as f:
        _yaml.safe_dump(cfg, f, sort_keys=True)
    assert verify_reproduction.verify_one(out) >= 1


def test_verify_metrics_json_drift_fails(tmp_path: Path):
    """If metrics.json on disk drifts after the manifest was written,
    check #4 must FAIL."""
    out = _build_compliant_run(tmp_path / "outputs" / "metrics_drift")
    metrics_path = out / "metrics.json"
    blob = json.loads(metrics_path.read_text())
    blob["metrics"]["mrr@10"] = 0.9999  # someone hand-edited the metrics
    metrics_path.write_text(json.dumps(blob, indent=2))
    assert verify_reproduction.verify_one(out) >= 1


def test_extract_headline_metrics_flattens_nested():
    """Dense runner emits ``{"dense": {...}, "bm25_sample": {...}}``.
    The extractor should produce namespaced keys so both arms surface
    on the verifier's printout — apples-to-apples comparability is the
    point."""
    blob = {
        "dense": {"mrr@10": 0.45, "recall@100": 0.91},
        "bm25_sample": {"mrr@10": 0.42},
    }
    out = verify_reproduction._extract_headline_metrics(blob)
    assert out["dense.mrr@10"] == 0.45
    assert out["dense.recall@100"] == 0.91
    assert out["bm25_sample.mrr@10"] == 0.42


def test_extract_headline_metrics_filters_non_numeric():
    """Non-numeric values (booleans, strings) shouldn't crash the
    formatter — they're skipped entirely from the headline display."""
    blob = {"mrr@10": 0.5, "is_sampled": True, "method": "qrels"}
    out = verify_reproduction._extract_headline_metrics(blob)
    assert out == {"mrr@10": 0.5}


def test_extract_headline_metrics_handles_non_dict():
    """Defensive: blob = None / list / scalar must NOT raise."""
    assert verify_reproduction._extract_headline_metrics(None) == {}
    assert verify_reproduction._extract_headline_metrics([1, 2, 3]) == {}
    assert verify_reproduction._extract_headline_metrics("not a dict") == {}


def test_get_dotted_distinguishes_missing_from_none():
    """The internal walker must classify ``key absent`` vs
    ``key present with value None`` differently — the validator treats
    both as violations but the verifier's reporting differs."""
    d = {"git": {"commit": None, "dirty": False}}
    assert verify_reproduction._get_dotted(d, "git.commit") is None
    assert verify_reproduction._get_dotted(d, "git.dirty") is False
    assert verify_reproduction._get_dotted(d, "git.absent") is verify_reproduction._MISSING


@pytest.mark.parametrize(
    "value,expected_max_len",
    [
        ("short", 5),
        ("a" * 100, 28),
        (42, 2),
        (None, 4),  # "None"
    ],
)
def test_truncate_for_display(value, expected_max_len):
    """Display truncation must not crash on non-strings and must cap
    long strings to the default n=28 (matches the verifier's one-line
    output budget)."""
    s = verify_reproduction._truncate(value)
    assert len(s) <= max(28, expected_max_len)

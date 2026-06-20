"""Tests for ``scripts/backfill_provenance.py``.

The script's output is deliberately distinct from a runtime manifest so a
reader can never mistake the two. The tests pin down the load-bearing
parts of that distinction (schema string, ``unknown`` block keys) and
the basic correctness of ``build_backfilled_provenance``.

``scripts/`` is not a Python package, so the script is loaded by path
via :mod:`importlib`.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "backfill_provenance.py"
)


def _load_backfill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backfill_provenance", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load_backfill_module()


EXPECTED_UNKNOWN_KEYS = {
    "production_commit",
    "production_command_line",
    "production_timestamp",
    "git_dirty_at_production",
    "python_version_at_production",
    "package_versions_at_production",
    "production_input_files",
    "production_random_seed_effectiveness",
}


def _make_output_dir(tmp_path: Path, name: str) -> Path:
    output_dir = tmp_path / "outputs" / name
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text('{"task": "fake"}\n')
    (output_dir / "run.tsv").write_text("q1\tQ0\td1\t1\t1.0\tfake\n")
    return output_dir


def _build(tmp_path: Path, output_dir: Path) -> dict:
    return bp.build_backfilled_provenance(
        output_dir=output_dir,
        anchor_tag="v1.0-fake",
        anchor_commit="abcdef012345",
        config_path="configs/baseline.yaml",
        config_sha_16="0123456789abcdef",
        project_root=tmp_path,
        now_utc=dt.datetime(2026, 5, 22, 12, 0, 0, tzinfo=dt.timezone.utc),
    )


def test_schema_string_and_distinctness(tmp_path: Path):
    """The schema must be the backfill schema, *not* the runtime-manifest one.

    This is the load-bearing invariant: a reader who confuses
    ``provenance.backfill.json`` with ``manifest.json`` would assume the
    run has tighter provenance than it actually does.
    """
    output_dir = _make_output_dir(tmp_path, "stage_schema")
    doc = _build(tmp_path, output_dir)

    assert doc["schema"] == "msmarco-genqa.backfilled-provenance.v1"
    assert doc["schema"].startswith("msmarco-genqa.backfilled-provenance")
    assert doc["schema"] != "msmarco-genqa.manifest.v1"
    assert "BACKFILLED" in doc["note"]
    assert doc["produced_by"] == "scripts/backfill_provenance.py"


def test_unknown_block_has_all_eight_keys(tmp_path: Path):
    """The ``unknown`` block enumerates exactly the eight unrecoverable
    dimensions of runtime provenance. Tests pin this set down so a
    well-meaning refactor doesn't quietly drop one of the explanations."""
    output_dir = _make_output_dir(tmp_path, "stage_unknown")
    doc = _build(tmp_path, output_dir)

    assert isinstance(doc["unknown"], dict)
    assert set(doc["unknown"].keys()) == EXPECTED_UNKNOWN_KEYS
    for key, explanation in doc["unknown"].items():
        assert isinstance(explanation, str)
        assert explanation.strip(), f"unknown.{key} should not be empty"


def test_unknown_seed_block_references_seeding_patch(tmp_path: Path):
    """The seed-effectiveness footnote is the file's reason-for-being.
    It must mention the seeding patch commit so a future reader can
    trace why the file is distinct from a real manifest."""
    output_dir = _make_output_dir(tmp_path, "stage_seed_note")
    doc = _build(tmp_path, output_dir)

    seed_note = doc["unknown"]["production_random_seed_effectiveness"]
    assert "seeding" in seed_note.lower()
    assert "src/msmarco_genqa/util/seeding.py" in seed_note
    assert bp.SEEDING_PATCH_COMMIT_SHORT in seed_note


def test_anchor_and_config_blocks(tmp_path: Path):
    """``anchor`` and ``config_at_anchor`` carry the tag/commit/hash
    triple and a comment explaining what the hash does *not* prove."""
    output_dir = _make_output_dir(tmp_path, "stage_anchor")
    doc = _build(tmp_path, output_dir)

    assert doc["anchor"]["tag"] == "v1.0-fake"
    assert doc["anchor"]["commit"] == "abcdef012345"
    assert "EARLIER" in doc["anchor"]["note"]

    assert doc["config_at_anchor"]["path"] == "configs/baseline.yaml"
    assert doc["config_at_anchor"]["sha256_16"] == "0123456789abcdef"
    assert "cannot verify" in doc["config_at_anchor"]["comment"]


def test_outputs_listing_is_repo_relative_and_hashed(tmp_path: Path):
    """Each entry in ``outputs_on_disk_now`` has a repo-relative path,
    a 16-char sha256, and a non-zero size."""
    output_dir = _make_output_dir(tmp_path, "stage_outputs")
    doc = _build(tmp_path, output_dir)

    outputs = doc["outputs_on_disk_now"]
    paths = [r["path"] for r in outputs]
    assert any(p.endswith("metrics.json") for p in paths)
    assert any(p.endswith("run.tsv") for p in paths)

    for rec in outputs:
        assert not rec["path"].startswith("/"), f"absolute path leak: {rec['path']}"
        assert str(tmp_path) not in rec["path"]
        assert len(rec["sha256_16"]) == 16
        assert rec["size_bytes"] > 0


def test_idempotence_excludes_self(tmp_path: Path):
    """If a stale ``provenance.backfill.json`` is already in the output
    dir (re-running the backfill), the regenerated listing must NOT
    record itself — otherwise its own hash drifts on every run."""
    output_dir = _make_output_dir(tmp_path, "stage_idempotent")
    (output_dir / "provenance.backfill.json").write_text('{"stale": true}\n')

    doc = _build(tmp_path, output_dir)

    paths = [r["path"] for r in doc["outputs_on_disk_now"]]
    assert not any(
        p.endswith("provenance.backfill.json") for p in paths
    ), f"backfill should not list itself; got {paths}"


def test_write_backfill_for_dir_round_trips(tmp_path: Path):
    """End-to-end: ``write_backfill_for_dir`` writes valid JSON that
    parses back to the same schema string and ``unknown`` keys."""
    output_dir = _make_output_dir(tmp_path, "stage_writeback")

    written = bp.write_backfill_for_dir(
        output_dir=output_dir,
        anchor_tag="v1.0-fake",
        anchor_commit="abcdef012345",
        config_path="configs/baseline.yaml",
        config_sha_16="0123456789abcdef",
        project_root=tmp_path,
    )
    assert written == output_dir / "provenance.backfill.json"
    assert written.exists()

    loaded = json.loads(written.read_text())
    assert loaded["schema"] == "msmarco-genqa.backfilled-provenance.v1"
    assert set(loaded["unknown"].keys()) == EXPECTED_UNKNOWN_KEYS


def test_timestamp_is_iso_utc(tmp_path: Path):
    """``backfill_created_at`` is an ISO 8601 UTC timestamp; pinning the
    format here so accidental locale-formatting regressions are caught."""
    output_dir = _make_output_dir(tmp_path, "stage_ts")
    doc = _build(tmp_path, output_dir)

    ts = doc["backfill_created_at"]
    parsed = dt.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)


def test_skips_subdirectories_when_listing_outputs(tmp_path: Path):
    """``outputs_on_disk_now`` lists files one level deep; nested dirs
    are not recursed into (kept simple on purpose — the canonical
    output dirs are flat)."""
    output_dir = _make_output_dir(tmp_path, "stage_subdir")
    (output_dir / "nested").mkdir()
    (output_dir / "nested" / "leaf.txt").write_text("ignored\n")

    doc = _build(tmp_path, output_dir)

    paths = [r["path"] for r in doc["outputs_on_disk_now"]]
    assert not any("nested" in p for p in paths)


# --------------------------------------------------------------------------- #
# Hashing helpers — small, but they back the schema's correctness claim.
# --------------------------------------------------------------------------- #


def test_sha256_hex_16_length_and_determinism():
    out1 = bp._sha256_hex_16(b"hello world")
    out2 = bp._sha256_hex_16(b"hello world")
    assert out1 == out2
    assert len(out1) == 16
    assert all(c in "0123456789abcdef" for c in out1)


def test_file_sha256_hex_16_matches_in_memory(tmp_path: Path):
    payload = b"some bytes\nspread\nacross\nlines\n" * 100
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    assert bp._file_sha256_hex_16(p) == bp._sha256_hex_16(payload)


# --------------------------------------------------------------------------- #
# Module-level sanity
# --------------------------------------------------------------------------- #


def test_schema_constants_are_exported():
    """The module pins these as constants; downstream tests rely on them
    holding the documented values."""
    assert bp.SCHEMA == "msmarco-genqa.backfilled-provenance.v1"
    assert bp.PRODUCED_BY == "scripts/backfill_provenance.py"
    assert "outputs/bm25_baseline" in bp.DEFAULT_TARGETS
    assert "outputs/dense_retrieval" in bp.DEFAULT_TARGETS
    assert "outputs/cross_encoder_rerank" in bp.DEFAULT_TARGETS


@pytest.mark.parametrize("commit", ["abcdef012345", "5a35de9c18ea"])
def test_unknown_block_embeds_anchor_commit(commit: str):
    """The ``production_commit`` and ``production_random_seed_effectiveness``
    explanations reference the anchor commit so readers can verify which
    snapshot the file is rooted at."""
    block = bp._unknown_block(commit)
    assert commit in block["production_commit"]
    assert commit in block["production_random_seed_effectiveness"]

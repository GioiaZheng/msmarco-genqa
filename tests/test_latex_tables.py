from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from msmarco_genqa.reporting.latex_tables import (
    ARTIFACT_SCHEMA,
    IncompatibleArtifactError,
    MissingArtifactError,
    SchemaValidationError,
    StaleSourceError,
    export_tables,
    load_artifact,
    validate_compatible,
    validate_sidecar_current,
)
from scripts.export_report_tables import resolve_from_project_root


def _artifact_payload(table_id: str = "table_one") -> dict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": "fixture",
        "metric_schema": "fixture.metrics.v1",
        "query_set": {"id": "dev-small", "n_queries": 2},
        "model_revision": {"retriever": "fixture-retriever"},
        "tables": [
            {
                "id": table_id,
                "columns": [
                    {"heading": "Metric", "align": "l"},
                    {"heading": "Value", "align": "r"},
                ],
                "rows": [
                    {"cells": ["MRR@10", "0.5000"]},
                    {"cells": ["Recall@100", "0.7500"]},
                ],
            }
        ],
    }


def _write_artifact(path: Path, payload: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _artifact_payload(), indent=2) + "\n")
    return path


def test_cli_paths_resolve_from_project_root():
    resolved = resolve_from_project_root(Path("reports/generated/tables"))

    assert resolved.is_absolute()
    assert resolved.parts[-3:] == ("reports", "generated", "tables")


def test_export_tables_writes_fragment_and_repo_relative_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    artifact = _write_artifact(tmp_path / "reports" / "generated" / "artifacts" / "a.json")

    written = export_tables(
        artifact_paths=[artifact],
        output_dir=tmp_path / "reports" / "generated" / "tables",
    )

    assert [path.name for path in written] == ["table_one.tex"]
    table_text = written[0].read_text()
    assert r"\begin{tabular}{lr}" in table_text
    assert "MRR@10 & 0.5000" in table_text

    sidecar = written[0].with_suffix(".sources.json")
    payload = json.loads(sidecar.read_text())
    source_path = payload["sources"][0]["path"]
    assert source_path == "reports/generated/artifacts/a.json"
    assert str(tmp_path) not in source_path


def test_load_artifact_missing_path_raises(tmp_path: Path):
    with pytest.raises(MissingArtifactError):
        load_artifact(tmp_path / "absent.json")


def test_load_artifact_rejects_missing_schema(tmp_path: Path):
    bad = _artifact_payload()
    del bad["schema"]
    path = _write_artifact(tmp_path / "bad.json", bad)

    with pytest.raises(SchemaValidationError, match="schema"):
        load_artifact(path)


def test_load_artifact_rejects_wrong_row_width(tmp_path: Path):
    bad = _artifact_payload()
    bad["tables"][0]["rows"][0]["cells"] = ["only-one-cell"]
    path = _write_artifact(tmp_path / "bad-width.json", bad)

    with pytest.raises(SchemaValidationError, match="2 cells"):
        load_artifact(path)


def test_validate_compatible_rejects_query_set_mismatch(tmp_path: Path):
    first = _write_artifact(tmp_path / "a.json")
    second_payload = copy.deepcopy(_artifact_payload("table_two"))
    second_payload["query_set"] = {"id": "other", "n_queries": 2}
    second = _write_artifact(tmp_path / "b.json", second_payload)

    with pytest.raises(IncompatibleArtifactError, match="query_set"):
        validate_compatible([load_artifact(first), load_artifact(second)])


def test_export_tables_rejects_incompatible_comparison_group(tmp_path: Path):
    first_payload = _artifact_payload("table_one")
    first_payload["compatibility_group"] = "retrieval-comparison"
    first = _write_artifact(tmp_path / "a.json", first_payload)

    second_payload = copy.deepcopy(_artifact_payload("table_two"))
    second_payload["compatibility_group"] = "retrieval-comparison"
    second_payload["metric_schema"] = "fixture.metrics.v2"
    second = _write_artifact(tmp_path / "b.json", second_payload)

    with pytest.raises(IncompatibleArtifactError, match="metric_schema"):
        export_tables(
            artifact_paths=[first, second],
            output_dir=tmp_path / "reports" / "generated" / "tables",
        )


def test_validate_sidecar_current_rejects_stale_artifact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    artifact = _write_artifact(tmp_path / "reports" / "generated" / "artifacts" / "a.json")
    written = export_tables(
        artifact_paths=[artifact],
        output_dir=tmp_path / "reports" / "generated" / "tables",
    )

    payload = _artifact_payload()
    payload["tables"][0]["rows"][0]["cells"] = ["MRR@10", "0.6000"]
    _write_artifact(artifact, payload)

    with pytest.raises(StaleSourceError):
        validate_sidecar_current(written[0].with_suffix(".sources.json"))

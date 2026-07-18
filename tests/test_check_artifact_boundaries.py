from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_CHECK_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_artifact_boundaries.py"
_spec = importlib.util.spec_from_file_location("check_artifact_boundaries", _CHECK_PATH)
check_artifact_boundaries = importlib.util.module_from_spec(_spec)
sys.modules["check_artifact_boundaries"] = check_artifact_boundaries
_spec.loader.exec_module(check_artifact_boundaries)  # type: ignore[union-attr]


def test_allows_small_external_artifact_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "artifacts" / "run.json"
    pointer.parent.mkdir()
    pointer.write_text("{}\n", encoding="utf-8")

    errors = check_artifact_boundaries.check_paths(
        ["artifacts/run.json"],
        project_root=tmp_path,
        max_pointer_bytes=100,
    )

    assert errors == []


def test_rejects_oversized_external_artifact_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "artifacts" / "run.json"
    pointer.parent.mkdir()
    pointer.write_text("x" * 101, encoding="utf-8")

    errors = check_artifact_boundaries.check_paths(
        ["artifacts/run.json"],
        project_root=tmp_path,
        max_pointer_bytes=100,
    )

    assert errors == ["artifacts/run.json: external artifact pointers must stay small"]


def test_rejects_payload_file_in_external_artifact_directory(tmp_path: Path) -> None:
    payload = tmp_path / "artifacts" / "run.tsv"
    payload.parent.mkdir()
    payload.write_text("qid Q0 doc 1 1.0 run\n", encoding="utf-8")

    errors = check_artifact_boundaries.check_paths(
        ["artifacts/run.tsv"],
        project_root=tmp_path,
        max_pointer_bytes=100,
    )

    assert errors == [
        "artifacts/run.tsv: artifacts/ may contain only JSON pointers and README.md"
    ]

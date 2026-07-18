from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECK_PATH = PROJECT_ROOT / "scripts" / "check_artifact_registry.py"
REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "registry.json"
_spec = importlib.util.spec_from_file_location("check_artifact_registry", CHECK_PATH)
check_artifact_registry = importlib.util.module_from_spec(_spec)
sys.modules["check_artifact_registry"] = check_artifact_registry
_spec.loader.exec_module(check_artifact_registry)  # type: ignore[union-attr]


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _validate_copy(tmp_path: Path, registry: dict) -> list[str]:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return check_artifact_registry.validate_registry(path, project_root=PROJECT_ROOT)


def test_current_registry_is_consistent() -> None:
    assert check_artifact_registry.validate_registry(
        REGISTRY_PATH, project_root=PROJECT_ROOT
    ) == []


def test_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    registry = _registry()
    registry["entries"][0]["artifact_record"]["sha256_lf"] = "0" * 64

    errors = _validate_copy(tmp_path, registry)

    assert any("sha256_lf mismatch" in error for error in errors)


def test_rejects_duplicate_registry_id(tmp_path: Path) -> None:
    registry = _registry()
    registry["entries"][1]["id"] = registry["entries"][0]["id"]

    errors = _validate_copy(tmp_path, registry)

    assert any("duplicate registry id" in error for error in errors)


def test_historical_production_commit_must_remain_unknown(tmp_path: Path) -> None:
    registry = _registry()
    registry["entries"][0]["provenance"]["production_commit"] = "0" * 40

    errors = _validate_copy(tmp_path, registry)

    assert any(
        "historical_partial production_commit must be null" in error for error in errors
    )


def test_exact_commit_must_match_artifact_record(tmp_path: Path) -> None:
    registry = _registry()
    exact_entry = deepcopy(registry["entries"][-1])
    exact_entry["provenance"]["production_commit"] = "0" * 40
    registry["entries"][-1] = exact_entry

    errors = _validate_copy(tmp_path, registry)

    assert any("does not match artifact experiment.git_commit" in error for error in errors)


def test_tracked_manifest_path_must_exist(tmp_path: Path) -> None:
    registry = _registry()
    manifest = registry["entries"][0]["provenance"]["manifests"][0]
    manifest["availability"] = "tracked"
    manifest["path"] = "outputs/definitely-missing/manifest.json"

    errors = _validate_copy(tmp_path, registry)

    assert any("referenced file does not exist" in error for error in errors)


def test_rejects_current_lockfile_hash_drift(tmp_path: Path) -> None:
    registry = _registry()
    registry["current_lockfile"]["sha256_lf"] = "0" * 64

    errors = _validate_copy(tmp_path, registry)

    assert any("requirements-lock.txt changed" in error for error in errors)

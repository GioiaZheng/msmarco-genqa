from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_CHECK_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_notebooks.py"
_spec = importlib.util.spec_from_file_location("check_notebooks", _CHECK_PATH)
check_notebooks = importlib.util.module_from_spec(_spec)
sys.modules["check_notebooks"] = check_notebooks
_spec.loader.exec_module(check_notebooks)  # type: ignore[union-attr]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text(
        '[project.scripts]\nrag-eval     = "msmarco_genqa.cli.rag_eval:main"\n',
        encoding="utf-8",
    )


def _write_notebook(path: Path, cells: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str, *, outputs: list | None = None, execution_count=None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
    }


def test_current_notebooks_pass_demo_checks():
    assert check_notebooks.check_notebooks(
        PROJECT_ROOT / "notebooks",
        project_root=PROJECT_ROOT,
    ) == []


def test_rejects_stored_outputs(tmp_path: Path):
    _write_pyproject(tmp_path)
    notebook = _write_notebook(
        tmp_path / "notebooks" / "demo.ipynb",
        [
            _markdown("Use `rag-eval run --dry-run`."),
            _code(
                "from msmarco_genqa.rag_eval import build_rag_eval_plan\n",
                outputs=[{"output_type": "stream", "text": "hello"}],
            ),
        ],
    )

    errors = check_notebooks.check_notebook(notebook, project_root=tmp_path)

    assert any("must not store outputs" in error for error in errors)


def test_rejects_heavy_model_loading(tmp_path: Path):
    _write_pyproject(tmp_path)
    notebook = _write_notebook(
        tmp_path / "notebooks" / "demo.ipynb",
        [
            _markdown("Thin demo over package APIs."),
            _code(
                "from msmarco_genqa.rag_eval import build_rag_eval_plan\n"
                "model = AutoModel.from_pretrained('example')\n"
            ),
        ],
    )

    errors = check_notebooks.check_notebook(notebook, project_root=tmp_path)

    assert any("model loading belongs in package code" in error for error in errors)


def test_rejects_missing_package_or_cli_reference(tmp_path: Path):
    _write_pyproject(tmp_path)
    notebook = _write_notebook(
        tmp_path / "notebooks" / "demo.ipynb",
        [_markdown("A prose-only notebook."), _code("x = 1\n")],
    )

    errors = check_notebooks.check_notebook(notebook, project_root=tmp_path)

    assert any("must reference package APIs" in error for error in errors)

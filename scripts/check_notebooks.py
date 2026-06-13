"""Check that notebooks stay lightweight and demo-oriented."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_CODE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?m)^\s*[!%]"), "shell commands and notebook magics are not allowed"),
    (re.compile(r"\bir_datasets\.load\s*\("), "full benchmark loaders belong in scripts"),
    (re.compile(r"\bload_dataset\s*\("), "dataset downloads belong in scripts"),
    (re.compile(r"\.download\s*\("), "downloads belong in scripts"),
    (re.compile(r"\bfrom_pretrained\s*\("), "model loading belongs in package code"),
    (re.compile(r"\bSentenceTransformer\s*\("), "model loading belongs in package code"),
    (re.compile(r"\bCrossEncoder\s*\("), "model loading belongs in package code"),
    (re.compile(r"\bAuto(Model|Tokenizer)\b"), "model loading belongs in package code"),
    (re.compile(r"\bsubprocess\."), "CLI execution belongs in scripts or documented commands"),
    (re.compile(r"\brequests\."), "network calls are not allowed in notebooks"),
    (re.compile(r"\burllib\.request\b"), "network calls are not allowed in notebooks"),
    (re.compile(r"\bfaiss\.(read|write)_index\s*\("), "index IO belongs in package code"),
)

PACKAGE_REFERENCES = (
    "from msmarco_genqa",
    "import msmarco_genqa",
    "msmarco_genqa.",
)


def cell_source(cell: dict[str, Any]) -> str:
    """Return a notebook cell source as one string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def package_imports(source: str) -> set[str]:
    """Collect imported msmarco_genqa modules from one code cell."""
    imports: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "msmarco_genqa" or alias.name.startswith("msmarco_genqa."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "msmarco_genqa" or node.module.startswith("msmarco_genqa."):
                imports.add(node.module)
    return imports


def check_notebook(path: Path, *, project_root: Path) -> list[str]:
    """Return validation errors for one notebook."""
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid notebook JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append(f"{path}: nbformat must be 4")

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return [*errors, f"{path}: cells must be a list"]

    combined_source: list[str] = []
    imported_modules: set[str] = set()
    for index, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        source = cell_source(cell)
        combined_source.append(source)

        if cell_type != "code":
            continue

        if cell.get("outputs"):
            errors.append(f"{path}: code cell {index} must not store outputs")
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: code cell {index} must have execution_count=null")

        for pattern, message in FORBIDDEN_CODE_PATTERNS:
            if pattern.search(source):
                errors.append(f"{path}: code cell {index}: {message}")

        if source.strip():
            try:
                imported_modules.update(package_imports(source))
            except SyntaxError as exc:
                errors.append(f"{path}: code cell {index} is not plain Python: {exc.msg}")

    all_source = "\n".join(combined_source)
    has_package_reference = any(token in all_source for token in PACKAGE_REFERENCES)
    has_cli_reference = "rag-eval" in all_source
    if not (has_package_reference or has_cli_reference):
        errors.append(f"{path}: notebook must reference package APIs or a registered CLI")

    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    if has_cli_reference and 'rag-eval     = "msmarco_genqa.cli.rag_eval:main"' not in pyproject:
        errors.append(f"{path}: references rag-eval, but the CLI is not registered")

    for module in sorted(imported_modules):
        if importlib.util.find_spec(module) is None:
            errors.append(f"{path}: imported module is not available: {module}")

    return errors


def discover_notebooks(root: Path) -> list[Path]:
    """Return notebooks to validate."""
    return sorted(root.glob("*.ipynb"))


def check_notebooks(notebook_root: Path, *, project_root: Path) -> list[str]:
    """Return validation errors for all notebooks under a root."""
    notebooks = discover_notebooks(notebook_root)
    if not notebooks:
        return [f"{notebook_root}: no notebooks found"]

    errors: list[str] = []
    for notebook in notebooks:
        errors.extend(check_notebook(notebook, project_root=project_root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-root",
        type=Path,
        default=Path("notebooks"),
        help="Directory containing demo notebooks.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for pyproject.toml and import checks.",
    )
    args = parser.parse_args(argv)

    errors = check_notebooks(args.notebook_root, project_root=args.project_root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Notebook demo checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

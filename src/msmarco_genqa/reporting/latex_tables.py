"""Export checked LaTeX table fragments from report artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_SCHEMA = "msmarco-genqa.table-artifact.v1"
SOURCE_SCHEMA = "msmarco-genqa.table-sources.v1"


class TableArtifactError(RuntimeError):
    """Base class for table artifact failures."""


class MissingArtifactError(TableArtifactError):
    """Raised when a configured artifact path does not exist."""


class SchemaValidationError(TableArtifactError):
    """Raised when an artifact does not satisfy the table schema."""


class IncompatibleArtifactError(TableArtifactError):
    """Raised when artifacts describe incompatible experiment contexts."""


class StaleSourceError(TableArtifactError):
    """Raised when a sidecar source hash no longer matches its artifact."""


@dataclass(frozen=True)
class TableArtifact:
    """A loaded table artifact and its stable source digest."""

    path: Path
    payload: dict[str, Any]
    sha256_16: str

    @property
    def relative_path(self) -> str:
        try:
            return self.path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return self.path.as_posix()


def file_sha256_16(path: Path) -> str:
    """Return an LF-normalized, 16-character SHA-256 for a text artifact."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def load_artifact(path: Path) -> TableArtifact:
    """Load and validate one table artifact."""
    if not path.exists():
        raise MissingArtifactError(f"artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{path}: invalid JSON: {exc}") from exc

    validate_artifact_payload(payload, source=path.as_posix())
    return TableArtifact(path=path, payload=payload, sha256_16=file_sha256_16(path))


def validate_artifact_payload(payload: dict[str, Any], *, source: str = "<memory>") -> None:
    """Validate the minimal schema used by the LaTeX exporter."""
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise SchemaValidationError(
            f"{source}: schema must be {ARTIFACT_SCHEMA!r}, got {payload.get('schema')!r}"
        )

    for key in ("artifact_id", "metric_schema", "query_set", "tables"):
        if key not in payload:
            raise SchemaValidationError(f"{source}: missing required key {key!r}")

    query_set = payload["query_set"]
    if not isinstance(query_set, dict):
        raise SchemaValidationError(f"{source}: query_set must be an object")
    if not query_set.get("id") or not isinstance(query_set.get("n_queries"), int):
        raise SchemaValidationError(f"{source}: query_set requires id and integer n_queries")

    tables = payload["tables"]
    if not isinstance(tables, list) or not tables:
        raise SchemaValidationError(f"{source}: tables must be a non-empty list")

    for table in tables:
        validate_table_payload(table, source=source)


def validate_table_payload(table: dict[str, Any], *, source: str = "<memory>") -> None:
    """Validate one table block inside an artifact."""
    for key in ("id", "columns", "rows"):
        if key not in table:
            raise SchemaValidationError(f"{source}: table is missing {key!r}")

    columns = table["columns"]
    rows = table["rows"]
    if not isinstance(columns, list) or not columns:
        raise SchemaValidationError(f"{source}: table {table.get('id')!r} has no columns")
    if not isinstance(rows, list) or not rows:
        raise SchemaValidationError(f"{source}: table {table.get('id')!r} has no rows")

    for column in columns:
        if not isinstance(column, dict) or "heading" not in column:
            raise SchemaValidationError(
                f"{source}: table {table.get('id')!r} columns require heading"
            )
        align = column.get("align", "l")
        if align not in {"l", "c", "r"}:
            raise SchemaValidationError(
                f"{source}: table {table.get('id')!r} has invalid align {align!r}"
            )

    width = len(columns)
    for row in rows:
        cells = row.get("cells") if isinstance(row, dict) else None
        if not isinstance(cells, list) or len(cells) != width:
            raise SchemaValidationError(
                f"{source}: table {table.get('id')!r} rows must have {width} cells"
            )


def validate_compatible(artifacts: Iterable[TableArtifact]) -> None:
    """Refuse to combine artifacts with incompatible run contexts."""
    items = list(artifacts)
    if len(items) < 2:
        return

    first = items[0].payload
    expected = {
        "query_set": first.get("query_set"),
        "metric_schema": first.get("metric_schema"),
        "model_revision": first.get("model_revision"),
    }
    for artifact in items[1:]:
        payload = artifact.payload
        for key, value in expected.items():
            if payload.get(key) != value:
                raise IncompatibleArtifactError(
                    f"{artifact.relative_path}: incompatible {key}; "
                    f"expected {value!r}, got {payload.get(key)!r}"
                )


def validate_compatibility_groups(artifacts: Iterable[TableArtifact]) -> None:
    """Validate artifacts that explicitly opt into a shared comparison group."""
    groups: dict[str, list[TableArtifact]] = {}
    for artifact in artifacts:
        group = artifact.payload.get("compatibility_group")
        if group:
            groups.setdefault(str(group), []).append(artifact)

    for members in groups.values():
        validate_compatible(members)


def render_table(table: dict[str, Any]) -> str:
    """Render a booktabs tabular fragment from a validated table block."""
    validate_table_payload(table)
    align = "".join(column.get("align", "l") for column in table["columns"])
    headings = [str(column["heading"]) for column in table["columns"]]
    rows = ["    " + " & ".join(headings) + r" \\"]
    rows.append(r"    \midrule")
    for row in table["rows"]:
        rows.append("    " + " & ".join(str(cell) for cell in row["cells"]) + r" \\")

    lines = [
        "% Refresh with: python scripts/export_report_tables.py",
        rf"\begin{{tabular}}{{{align}}}",
        r"    \toprule",
        *rows,
        r"    \bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def source_sidecar(table_id: str, artifact: TableArtifact) -> dict[str, Any]:
    """Return deterministic source metadata for one generated table."""
    payload = artifact.payload
    return {
        "schema": SOURCE_SCHEMA,
        "table_id": table_id,
        "hash_convention": "sha256_lf",
        "sources": [
            {
                "path": artifact.relative_path,
                "sha256_16": artifact.sha256_16,
                "artifact_schema": payload["schema"],
                "artifact_id": payload["artifact_id"],
                "metric_schema": payload["metric_schema"],
                "query_set": payload["query_set"],
                "model_revision": payload.get("model_revision"),
            }
        ],
    }


def write_table_fragment(
    *,
    table: dict[str, Any],
    artifact: TableArtifact,
    output_dir: Path,
) -> Path:
    """Write one table fragment and its source sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_id = str(table["id"])
    table_path = output_dir / f"{table_id}.tex"
    source_path = output_dir / f"{table_id}.sources.json"

    table_path.write_text(render_table(table), encoding="utf-8")
    source_path.write_text(
        json.dumps(source_sidecar(table_id, artifact), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return table_path


def export_tables(*, artifact_paths: Iterable[Path], output_dir: Path) -> list[Path]:
    """Export all tables from the given artifact paths."""
    artifacts = [load_artifact(path) for path in artifact_paths]
    validate_compatibility_groups(artifacts)

    written: list[Path] = []
    seen: set[str] = set()
    for artifact in artifacts:
        for table in artifact.payload["tables"]:
            table_id = str(table["id"])
            if table_id in seen:
                raise SchemaValidationError(f"duplicate table id: {table_id}")
            seen.add(table_id)
            written.append(
                write_table_fragment(table=table, artifact=artifact, output_dir=output_dir)
            )
    return written


def validate_sidecar_current(sidecar_path: Path) -> None:
    """Validate that a table source sidecar still matches its artifact hashes."""
    if not sidecar_path.exists():
        raise MissingArtifactError(f"source sidecar not found: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SOURCE_SCHEMA:
        raise SchemaValidationError(
            f"{sidecar_path}: schema must be {SOURCE_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    if payload.get("hash_convention") != "sha256_lf":
        raise SchemaValidationError(
            f"{sidecar_path}: hash_convention must be 'sha256_lf'"
        )
    for source in payload.get("sources", []):
        source_path = Path(source["path"])
        expected = source["sha256_16"]
        actual = file_sha256_16(source_path)
        if actual != expected:
            raise StaleSourceError(
                f"{sidecar_path}: {source_path} hash changed from {expected} to {actual}"
            )

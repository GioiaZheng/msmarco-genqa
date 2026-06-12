"""Shared input validation helpers for runners and serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class InputValidationError(ValueError):
    """Typed error for malformed external inputs."""

    message: str
    field: str | None = None
    line_number: int | None = None
    path: Path | None = None

    def __str__(self) -> str:
        parts: list[str] = []
        if self.path is not None:
            parts.append(str(self.path))
        if self.line_number is not None:
            parts.append(str(self.line_number))
        prefix = ":".join(parts)
        field = f"{self.field}: " if self.field else ""
        return f"{prefix + ': ' if prefix else ''}{field}{self.message}"


def normalize_text(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
    max_chars: int | None = None,
    replacement_ratio_limit: float = 0.20,
    line_number: int | None = None,
    path: Path | None = None,
) -> str:
    """Normalize user/data text and reject clearly corrupted inputs.

    Very long text is truncated at a word boundary when possible. This keeps
    prompt assembly deterministic while still letting runners handle large
    passages without crashing.
    """
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    if text:
        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        if replacement_ratio > replacement_ratio_limit:
            raise InputValidationError(
                "replacement-character ratio is too high",
                field=field,
                line_number=line_number,
                path=path,
            )
    text = " ".join(text.replace("\ufffd", " ").split())
    if not text and not allow_empty:
        raise InputValidationError(
            "must not be empty",
            field=field,
            line_number=line_number,
            path=path,
        )
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = truncate_text(text, max_chars)
    return text


def normalize_passage_list(
    passages: Iterable[Any],
    *,
    max_passage_chars: int | None = None,
    allow_empty_list: bool = True,
) -> list[str]:
    """Normalize passage text while dropping empty optional passages."""
    out: list[str] = []
    for index, passage in enumerate(passages):
        text = normalize_text(
            passage,
            field=f"passages[{index}]",
            allow_empty=True,
            max_chars=max_passage_chars,
        )
        if text:
            out.append(text)
    if not out and not allow_empty_list:
        raise InputValidationError("must include at least one non-empty passage", field="passages")
    return out


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut or text[:max_chars].rstrip()


def iter_jsonl_objects(path: Path | str) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield JSON objects from a UTF-8 JSONL file with line-numbered errors."""
    p = Path(path)
    try:
        with p.open(encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputValidationError(
                        "invalid JSON",
                        line_number=line_number,
                        path=p,
                    ) from exc
                if not isinstance(row, dict):
                    raise InputValidationError(
                        "record must be a JSON object",
                        line_number=line_number,
                        path=p,
                    )
                yield line_number, row
    except UnicodeDecodeError as exc:
        raise InputValidationError("file is not valid UTF-8", path=p) from exc

from __future__ import annotations

import pytest

from msmarco_genqa.util.input_validation import (
    InputValidationError,
    iter_jsonl_objects,
    normalize_text,
)


def test_normalize_text_collapses_whitespace_and_replacement_characters():
    assert normalize_text("  hello \ufffd   world  ", field="query") == "hello world"


def test_normalize_text_truncates_at_word_boundary():
    assert normalize_text("alpha beta gamma", field="text", max_chars=12) == "alpha beta"


def test_normalize_text_rejects_replacement_heavy_text():
    with pytest.raises(InputValidationError, match="replacement-character ratio"):
        normalize_text("\ufffd" * 20, field="text")


def test_iter_jsonl_objects_rejects_non_object_record(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('["not", "object"]\n', encoding="utf-8")
    with pytest.raises(InputValidationError, match="record must be a JSON object"):
        list(iter_jsonl_objects(path))


def test_iter_jsonl_objects_rejects_non_utf8(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"id":"p1"}\n\xff')
    with pytest.raises(InputValidationError, match="not valid UTF-8"):
        list(iter_jsonl_objects(path))

from __future__ import annotations

import json

import pytest

from msmarco_genqa.service.app import GenerationService, create_app, load_passages_jsonl
from msmarco_genqa.util.input_validation import InputValidationError


class FakeGenerator:
    def generate(self, query, passages):
        return f"{query} :: {len(passages)}"


def test_generation_service_wraps_generator():
    service = GenerationService(FakeGenerator())
    out = service.answer("what is bm25?", ["p1", "p2"])
    assert out["answer"] == "what is bm25? :: 2"
    assert out["n_passages"] == 2


def test_generation_service_rejects_empty_query():
    service = GenerationService(FakeGenerator())
    with pytest.raises(InputValidationError, match="query: must not be empty"):
        service.answer("   ", ["p1"])


def test_generation_service_normalizes_passages():
    service = GenerationService(FakeGenerator())
    out = service.answer("  what   is bm25? ", ["  p1  ", "", "p2\ufffdtext"])
    assert out["query"] == "what is bm25?"
    assert out["n_passages"] == 2


def test_load_passages_jsonl(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "text": "first"}) + "\n"
        + json.dumps({"id": "p2", "text": "second"}) + "\n",
        encoding="utf-8",
    )
    assert load_passages_jsonl(path) == {"p1": "first", "p2": "second"}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"text": "missing id"}, "missing id"),
        ({"id": "p1"}, "missing text"),
        ({"id": "   ", "text": "empty id"}, "id: must not be empty"),
        ({"id": "p1", "text": "   "}, "text: must not be empty"),
        ({"id": "p1", "text": "\ufffd" * 20}, "replacement-character ratio"),
    ],
)
def test_load_passages_jsonl_rejects_malformed_records(tmp_path, payload, message):
    path = tmp_path / "passages.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        load_passages_jsonl(path)


def test_load_passages_jsonl_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "text": "first"}) + "\n"
        + json.dumps({"id": "p1", "text": "second"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError, match="duplicate passage id"):
        load_passages_jsonl(path)


def test_load_passages_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text('{"id": "p1", "text": "first"\n', encoding="utf-8")
    with pytest.raises(InputValidationError, match="invalid JSON"):
        load_passages_jsonl(path)


def test_generate_endpoint_returns_structured_validation_error():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(generator=FakeGenerator()))
    response = client.post("/generate", json={"query": "   ", "passages": ["p1"]})
    assert response.status_code == 422
    assert response.json()["error"] == {
        "type": "input_validation",
        "message": "must not be empty",
        "field": "query",
    }

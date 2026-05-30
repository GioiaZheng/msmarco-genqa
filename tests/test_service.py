from __future__ import annotations

import json

from msmarco_genqa.service.app import GenerationService, load_passages_jsonl


class FakeGenerator:
    def generate(self, query, passages):
        return f"{query} :: {len(passages)}"


def test_generation_service_wraps_generator():
    service = GenerationService(FakeGenerator())
    out = service.answer("what is bm25?", ["p1", "p2"])
    assert out["answer"] == "what is bm25? :: 2"
    assert out["n_passages"] == 2


def test_load_passages_jsonl(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "text": "first"}) + "\n"
        + json.dumps({"id": "p2", "text": "second"}) + "\n",
        encoding="utf-8",
    )
    assert load_passages_jsonl(path) == {"p1": "first", "p2": "second"}

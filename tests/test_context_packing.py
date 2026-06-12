from __future__ import annotations

import pytest

from msmarco_genqa.generation.context_packing import (
    ContextPackingConfig,
    pack_context,
)


def test_disabled_context_packing_preserves_non_empty_passages():
    packed = pack_context(
        query="what is bm25",
        doc_ids=["d1", "d2"],
        passages=[" first passage ", ""],
        config=ContextPackingConfig(enabled=False),
    )

    assert packed.passages == ["first passage"]
    assert packed.doc_ids == ["d1"]
    assert packed.spans[0].start_char == 0
    assert packed.spans[0].end_char == len("first passage")


def test_query_overlap_sentence_selection_keeps_relevant_sentence():
    packed = pack_context(
        query="where was ada lovelace born",
        doc_ids=["d1"],
        passages=[
            "This sentence is about engines and mathematics. "
            "Ada Lovelace was born in London. "
            "This trailing sentence is less useful."
        ],
        config=ContextPackingConfig(
            enabled=True,
            max_passage_chars=40,
            sentence_selection="query_overlap",
        ),
    )

    assert packed.passages == ["Ada Lovelace was born in London."]
    assert packed.spans[0].selected_sentence_count == 1
    assert packed.spans[0].truncated is True


def test_context_budget_trims_last_retained_passage_at_word_boundary():
    packed = pack_context(
        query="budget",
        doc_ids=["d1", "d2"],
        passages=["alpha beta gamma", "delta epsilon zeta"],
        config=ContextPackingConfig(enabled=True, max_context_chars=24),
    )

    assert packed.passages == ["alpha beta gamma", "delta"]
    assert packed.packed_context_chars <= 24
    assert packed.spans[1].truncated is True
    assert packed.spans[1].start_char == len("alpha beta gamma ")


def test_deduplication_drops_repeated_normalized_passage():
    packed = pack_context(
        query="same",
        doc_ids=["d1", "d2", "d3"],
        passages=["Same passage.", "  same   passage. ", "Different passage."],
        config=ContextPackingConfig(enabled=True, deduplicate=True),
    )

    assert packed.doc_ids == ["d1", "d3"]
    assert packed.dropped_doc_ids == ["d2"]


def test_shorter_first_ordering_is_deterministic():
    packed = pack_context(
        query="order",
        doc_ids=["d1", "d2", "d3"],
        passages=["longer passage text", "short", "short"],
        config=ContextPackingConfig(enabled=True, deduplicate=False, ordering="shorter_first"),
    )

    assert packed.doc_ids == ["d2", "d3", "d1"]
    assert [span.source_rank for span in packed.spans] == [2, 3, 1]


def test_context_packing_rejects_mismatched_doc_ids_and_passages():
    with pytest.raises(ValueError, match="same length"):
        pack_context(
            query="q",
            doc_ids=["d1"],
            passages=["p1", "p2"],
            config=ContextPackingConfig(enabled=True),
        )

"""Deterministic context packing for RAG generation prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ContextPackingConfig:
    """CPU-friendly prompt packing knobs.

    Character budgets are used as a deterministic proxy for tokenizer cost so
    tests do not need to download model tokenizers.
    """

    enabled: bool = False
    max_context_chars: int | None = None
    max_passage_chars: int | None = None
    sentence_selection: str = "query_overlap"
    deduplicate: bool = True
    ordering: str = "rank"
    min_sentence_chars: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ContextPackingConfig":
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise ValueError("generation.context_packing must be a mapping")
        cfg = cls(
            enabled=bool(raw.get("enabled", False)),
            max_context_chars=_optional_positive_int(raw.get("max_context_chars")),
            max_passage_chars=_optional_positive_int(raw.get("max_passage_chars")),
            sentence_selection=str(raw.get("sentence_selection", "query_overlap")),
            deduplicate=bool(raw.get("deduplicate", True)),
            ordering=str(raw.get("ordering", "rank")),
            min_sentence_chars=max(1, int(raw.get("min_sentence_chars", 1))),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.sentence_selection not in {"head", "query_overlap"}:
            raise ValueError(
                "generation.context_packing.sentence_selection must be 'head' or "
                "'query_overlap'"
            )
        if self.ordering not in {"rank", "shorter_first"}:
            raise ValueError("generation.context_packing.ordering must be 'rank' or 'shorter_first'")

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_context_chars": self.max_context_chars,
            "max_passage_chars": self.max_passage_chars,
            "sentence_selection": self.sentence_selection,
            "deduplicate": self.deduplicate,
            "ordering": self.ordering,
            "min_sentence_chars": self.min_sentence_chars,
        }


@dataclass(frozen=True)
class PackedPassage:
    doc_id: str
    source_rank: int
    packed_index: int
    text: str
    original_char_count: int
    selected_char_count: int
    start_char: int
    end_char: int
    truncated: bool = False
    selected_sentence_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_rank": self.source_rank,
            "packed_index": self.packed_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "original_char_count": self.original_char_count,
            "selected_char_count": self.selected_char_count,
            "truncated": self.truncated,
            "selected_sentence_count": self.selected_sentence_count,
        }


@dataclass(frozen=True)
class PackedContext:
    passages: list[str]
    spans: list[PackedPassage]
    original_doc_ids: list[str]
    dropped_doc_ids: list[str] = field(default_factory=list)
    original_context_chars: int = 0
    packed_context_chars: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return [span.doc_id for span in self.spans]

    def to_json(self, config: ContextPackingConfig) -> dict[str, Any]:
        return {
            "config": config.to_json(),
            "original_doc_ids": self.original_doc_ids,
            "retained_doc_ids": self.doc_ids,
            "dropped_doc_ids": self.dropped_doc_ids,
            "original_context_chars": self.original_context_chars,
            "packed_context_chars": self.packed_context_chars,
            "compression_ratio": _safe_ratio(
                self.packed_context_chars,
                self.original_context_chars,
            ),
            "spans": [span.to_json() for span in self.spans],
        }


def pack_context(
    *,
    query: str,
    doc_ids: Sequence[str],
    passages: Sequence[str],
    config: ContextPackingConfig,
) -> PackedContext:
    """Pack passages under the configured context budget."""
    if len(doc_ids) != len(passages):
        raise ValueError("doc_ids and passages must have the same length")
    config.validate()
    original_doc_ids = [str(doc_id) for doc_id in doc_ids]
    original_texts = [_normalize_text(text) for text in passages]
    original_context_chars = len(" ".join(text for text in original_texts if text))

    if not config.enabled:
        retained_texts = [text for text in original_texts if text]
        retained_doc_ids = [
            doc_id for doc_id, text in zip(original_doc_ids, original_texts) if text
        ]
        spans = _build_spans(
            doc_ids=retained_doc_ids,
            source_ranks=[
                index + 1 for index, text in enumerate(original_texts) if text
            ],
            texts=retained_texts,
            original_lengths=[len(text) for text in original_texts if text],
            selected_sentence_counts=[0 for text in retained_texts],
            truncated_flags=[False for text in retained_texts],
        )
        return PackedContext(
            passages=retained_texts,
            spans=spans,
            original_doc_ids=original_doc_ids,
            original_context_chars=original_context_chars,
            packed_context_chars=len(" ".join(retained_texts)),
        )

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for index, (doc_id, text) in enumerate(zip(original_doc_ids, original_texts), start=1):
        if not text:
            dropped.append(doc_id)
            continue
        selected_text, sentence_count = _select_passage_text(query, text, config)
        if not selected_text:
            dropped.append(doc_id)
            continue
        dedup_key = _dedup_key(selected_text)
        if config.deduplicate and dedup_key in seen:
            dropped.append(doc_id)
            continue
        seen.add(dedup_key)
        selected.append(
            {
                "doc_id": doc_id,
                "source_rank": index,
                "text": selected_text,
                "original_char_count": len(text),
                "selected_sentence_count": sentence_count,
                "truncated": len(selected_text) < len(text),
            }
        )

    if config.ordering == "shorter_first":
        selected.sort(key=lambda row: (len(row["text"]), row["source_rank"]))

    retained_rows = _apply_context_budget(selected, config.max_context_chars)
    retained_texts = [row["text"] for row in retained_rows]
    retained_doc_ids = [row["doc_id"] for row in retained_rows]
    spans = _build_spans(
        doc_ids=retained_doc_ids,
        source_ranks=[row["source_rank"] for row in retained_rows],
        texts=retained_texts,
        original_lengths=[row["original_char_count"] for row in retained_rows],
        selected_sentence_counts=[row["selected_sentence_count"] for row in retained_rows],
        truncated_flags=[row["truncated"] for row in retained_rows],
    )
    retained_set = set(retained_doc_ids)
    dropped.extend(doc_id for doc_id in original_doc_ids if doc_id not in retained_set and doc_id not in dropped)
    return PackedContext(
        passages=retained_texts,
        spans=spans,
        original_doc_ids=original_doc_ids,
        dropped_doc_ids=dropped,
        original_context_chars=original_context_chars,
        packed_context_chars=len(" ".join(retained_texts)),
    )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("context packing budgets must be positive integers")
    return parsed


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\ufffd", " ").split())


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _sentences(text: str, *, min_chars: int) -> list[str]:
    parts = [_normalize_text(part) for part in _SENTENCE_RE.split(text)]
    return [part for part in parts if len(part) >= min_chars] or [_normalize_text(text)]


def _select_passage_text(
    query: str,
    text: str,
    config: ContextPackingConfig,
) -> tuple[str, int]:
    budget = config.max_passage_chars
    if budget is None or len(text) <= budget:
        return text, len(_sentences(text, min_chars=config.min_sentence_chars))

    sentences = _sentences(text, min_chars=config.min_sentence_chars)
    if config.sentence_selection == "head":
        ordered = list(enumerate(sentences))
    else:
        query_tokens = _token_set(query)
        ordered = sorted(
            enumerate(sentences),
            key=lambda item: (
                -len(query_tokens & _token_set(item[1])),
                item[0],
            ),
        )

    chosen: list[tuple[int, str]] = []
    used = 0
    for original_index, sentence in ordered:
        extra = len(sentence) + (1 if chosen else 0)
        if used + extra <= budget:
            chosen.append((original_index, sentence))
            used += extra
        elif not chosen:
            chosen.append((original_index, _truncate_text(sentence, budget)))
            break
    chosen.sort(key=lambda item: item[0])
    return " ".join(sentence for _index, sentence in chosen), len(chosen)


def _apply_context_budget(
    rows: Sequence[Mapping[str, Any]],
    max_context_chars: int | None,
) -> list[dict[str, Any]]:
    if max_context_chars is None:
        return [dict(row) for row in rows]

    retained: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        text = str(row["text"])
        separator = 1 if retained else 0
        remaining = max_context_chars - used - separator
        if remaining <= 0:
            break
        if len(text) <= remaining:
            retained.append(dict(row))
            used += separator + len(text)
            continue
        trimmed = _truncate_text(text, remaining)
        if trimmed:
            updated = dict(row)
            updated["text"] = trimmed
            updated["truncated"] = True
            retained.append(updated)
        break
    return retained


def _truncate_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    cut = text[:budget].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut or text[:budget].rstrip()


def _build_spans(
    *,
    doc_ids: Sequence[str],
    source_ranks: Sequence[int],
    texts: Sequence[str],
    original_lengths: Sequence[int],
    selected_sentence_counts: Sequence[int],
    truncated_flags: Sequence[bool],
) -> list[PackedPassage]:
    spans: list[PackedPassage] = []
    offset = 0
    for packed_index, (
        doc_id,
        source_rank,
        text,
        original_length,
        sentence_count,
        truncated,
    ) in enumerate(
        zip(
            doc_ids,
            source_ranks,
            texts,
            original_lengths,
            selected_sentence_counts,
            truncated_flags,
        ),
        start=1,
    ):
        start = offset
        end = start + len(text)
        spans.append(
            PackedPassage(
                doc_id=doc_id,
                source_rank=source_rank,
                packed_index=packed_index,
                text=text,
                original_char_count=original_length,
                selected_char_count=len(text),
                start_char=start,
                end_char=end,
                truncated=truncated,
                selected_sentence_count=sentence_count,
            )
        )
        offset = end + 1
    return spans


def _dedup_key(text: str) -> str:
    return _normalize_text(text).lower()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0

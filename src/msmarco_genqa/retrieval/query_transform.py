"""Deterministic query transformation utilities for retrieval ablations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


SUPPORTED_METHODS = {"none", "normalize", "lexical_expansion", "decontextualize"}

DEFAULT_EXPANSION_TERMS: dict[str, tuple[str, ...]] = {
    "covid": ("coronavirus",),
    "fda": ("food and drug administration",),
    "flu": ("influenza",),
    "heart attack": ("myocardial infarction",),
    "irs": ("internal revenue service",),
    "nyc": ("new york city",),
    "u.s.": ("united states",),
    "uk": ("united kingdom",),
    "usa": ("united states",),
}

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)?")
_TERMINAL_PUNCT_RE = re.compile(r"[\s?!.]+$")
_ELLIPTICAL_PREFIX_RE = re.compile(
    r"^(and|also|how about|it|that|these|they|this|those|what about)\b"
)


@dataclass(frozen=True)
class QueryTransformConfig:
    """Configuration for deterministic query transformation."""

    method: str = "none"
    lowercase: bool = True
    strip_terminal_punctuation: bool = True
    max_query_terms_for_expansion: int = 6
    max_expansion_terms: int = 4
    expansion_terms: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_EXPANSION_TERMS)
    )
    decontextualization_context: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "QueryTransformConfig":
        settings = dict(raw or {})
        method = str(settings.get("method", "none"))
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported query transformation method {method!r}; "
                f"expected one of {sorted(SUPPORTED_METHODS)}"
            )

        expansion_terms = _parse_expansion_terms(
            settings.get("expansion_terms", DEFAULT_EXPANSION_TERMS)
        )
        return cls(
            method=method,
            lowercase=_as_bool(settings.get("lowercase", True)),
            strip_terminal_punctuation=_as_bool(
                settings.get("strip_terminal_punctuation", True)
            ),
            max_query_terms_for_expansion=int(
                settings.get("max_query_terms_for_expansion", 6)
            ),
            max_expansion_terms=int(settings.get("max_expansion_terms", 4)),
            expansion_terms=expansion_terms,
            decontextualization_context=_optional_str(
                settings.get("decontextualization_context")
            ),
        )


@dataclass(frozen=True)
class QueryTransformationRecord:
    """One auditable query transformation record."""

    query_id: str
    original_query: str
    transformed_query: str
    method: str
    config_hash: str
    changed: bool
    added_terms: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["added_terms"] = list(self.added_terms)
        return payload


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_expansion_terms(raw: object) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("query_transform.expansion_terms must be a mapping")

    parsed: dict[str, tuple[str, ...]] = {}
    for trigger, terms in raw.items():
        trigger_text = str(trigger).strip()
        if not trigger_text:
            continue
        if isinstance(terms, str):
            values = (terms,)
        elif isinstance(terms, Sequence):
            values = tuple(str(term) for term in terms)
        else:
            raise ValueError(
                f"expansion terms for {trigger_text!r} must be a string or sequence"
            )
        cleaned = tuple(term.strip() for term in values if term.strip())
        if cleaned:
            parsed[trigger_text] = cleaned
    return parsed


def normalize_query(
    query: str,
    *,
    lowercase: bool = True,
    strip_terminal_punctuation: bool = True,
) -> str:
    """Collapse whitespace and optionally lowercase a query."""

    text = str(query)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _SPACE_RE.sub(" ", text.strip())
    if strip_terminal_punctuation:
        text = _TERMINAL_PUNCT_RE.sub("", text).strip()
    if lowercase:
        text = text.lower()
    return text


def query_transform_config_hash(config: QueryTransformConfig) -> str:
    """Return a stable short hash for artifact compatibility checks."""

    payload = {
        "method": config.method,
        "lowercase": config.lowercase,
        "strip_terminal_punctuation": config.strip_terminal_punctuation,
        "max_query_terms_for_expansion": config.max_query_terms_for_expansion,
        "max_expansion_terms": config.max_expansion_terms,
        "expansion_terms": {
            key: list(value) for key, value in sorted(config.expansion_terms.items())
        },
        "decontextualization_context": config.decontextualization_context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def transform_query(
    query: str,
    config: QueryTransformConfig,
    *,
    context: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Transform one query and return ``(transformed_query, added_terms)``."""

    if config.method == "none":
        return str(query), ()

    normalized = normalize_query(
        str(query),
        lowercase=config.lowercase,
        strip_terminal_punctuation=config.strip_terminal_punctuation,
    )
    if config.method == "normalize":
        return normalized, ()

    if config.method == "lexical_expansion":
        added_terms = _expansion_terms_for_query(normalized, config)
        if not added_terms:
            return normalized, ()
        return " ".join([normalized, *added_terms]).strip(), added_terms

    if config.method == "decontextualize":
        ctx = context or config.decontextualization_context
        if not ctx or not normalized or not _ELLIPTICAL_PREFIX_RE.search(normalized):
            return normalized, ()
        prefix = normalize_query(
            ctx,
            lowercase=config.lowercase,
            strip_terminal_punctuation=config.strip_terminal_punctuation,
        )
        return f"{prefix} {normalized}".strip(), ()

    raise ValueError(f"unsupported query transformation method {config.method!r}")


def _expansion_terms_for_query(
    normalized_query: str,
    config: QueryTransformConfig,
) -> tuple[str, ...]:
    tokens = _TOKEN_RE.findall(normalized_query)
    if not tokens or len(tokens) > config.max_query_terms_for_expansion:
        return ()

    token_set = set(tokens)
    additions: list[str] = []
    seen_additions: set[str] = set()
    for trigger, raw_terms in sorted(config.expansion_terms.items()):
        trigger_normalized = normalize_query(
            trigger,
            lowercase=config.lowercase,
            strip_terminal_punctuation=config.strip_terminal_punctuation,
        )
        if not _trigger_matches(trigger_normalized, normalized_query, token_set):
            continue
        for term in raw_terms:
            normalized_term = normalize_query(
                term,
                lowercase=config.lowercase,
                strip_terminal_punctuation=config.strip_terminal_punctuation,
            )
            if not normalized_term:
                continue
            if normalized_term in normalized_query or normalized_term in seen_additions:
                continue
            additions.append(normalized_term)
            seen_additions.add(normalized_term)
            if len(additions) >= config.max_expansion_terms:
                return tuple(additions)
    return tuple(additions)


def _trigger_matches(trigger: str, normalized_query: str, token_set: set[str]) -> bool:
    if " " not in trigger and trigger in token_set:
        return True
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", normalized_query) is not None


def transform_queries(
    queries: Mapping[str, str],
    config: QueryTransformConfig,
    *,
    contexts: Mapping[str, str] | None = None,
) -> list[QueryTransformationRecord]:
    """Transform a query mapping into ordered audit records."""

    config_hash = query_transform_config_hash(config)
    records: list[QueryTransformationRecord] = []
    for qid, query in queries.items():
        transformed, added_terms = transform_query(
            query,
            config,
            context=(contexts or {}).get(qid),
        )
        records.append(
            QueryTransformationRecord(
                query_id=str(qid),
                original_query=str(query),
                transformed_query=transformed,
                method=config.method,
                config_hash=config_hash,
                changed=str(query) != transformed,
                added_terms=added_terms,
            )
        )
    return records


def summarize_transformations(
    records: Sequence[QueryTransformationRecord],
    config: QueryTransformConfig,
    *,
    cache_hit: bool = False,
) -> dict[str, object]:
    config_hash = query_transform_config_hash(config)
    n_queries = len(records)
    n_changed = sum(1 for record in records if record.changed)
    return {
        "method": config.method,
        "config_hash": config_hash,
        "n_queries": n_queries,
        "n_changed": n_changed,
        "changed_fraction": (n_changed / n_queries) if n_queries else 0.0,
        "cache_hit": cache_hit,
    }


def transformed_query_map(
    records: Sequence[QueryTransformationRecord],
) -> dict[str, str]:
    return {record.query_id: record.transformed_query for record in records}


def write_transformation_artifacts(
    records: Sequence[QueryTransformationRecord],
    output_dir: Path,
    config: QueryTransformConfig,
    *,
    cache_hit: bool = False,
) -> tuple[dict[str, object], list[Path]]:
    """Write query transformation JSONL and summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    summary_path = output_dir / "summary.json"

    with queries_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    summary = summarize_transformations(records, config, cache_hit=cache_hit)
    summary["queries_path"] = str(queries_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, [queries_path, summary_path]


def read_cached_transformations(
    path: Path,
    *,
    expected_config_hash: str,
    expected_query_ids: Sequence[str],
) -> list[QueryTransformationRecord] | None:
    """Read cached records if the config hash and qid order match."""

    if not path.exists():
        return None

    records: list[QueryTransformationRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            added_terms = tuple(payload.get("added_terms", ()))
            records.append(
                QueryTransformationRecord(
                    query_id=str(payload["query_id"]),
                    original_query=str(payload["original_query"]),
                    transformed_query=str(payload["transformed_query"]),
                    method=str(payload["method"]),
                    config_hash=str(payload["config_hash"]),
                    changed=bool(payload["changed"]),
                    added_terms=added_terms,
                )
            )

    if [record.query_id for record in records] != [str(qid) for qid in expected_query_ids]:
        return None
    if any(record.config_hash != expected_config_hash for record in records):
        return None
    return records


def materialize_query_transform(
    queries: Mapping[str, str],
    settings: Mapping[str, object] | None,
    *,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, object], list[Path]]:
    """Return transformed query text plus optional audit artifacts for runners."""

    config = QueryTransformConfig.from_mapping(settings)
    records = transform_queries(queries, config)
    if config.method == "none":
        return dict(queries), summarize_transformations(records, config), []
    summary, paths = write_transformation_artifacts(records, output_dir, config)
    return transformed_query_map(records), summary, paths

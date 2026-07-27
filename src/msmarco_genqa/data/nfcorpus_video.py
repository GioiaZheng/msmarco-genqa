"""Pinned NFCorpus video query representations for the controlled ablation.

The loader deliberately accepts query text only. It has no qrels, corpus,
ranked-run, or metric input, which keeps query construction outside the
outcome-data path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


CONTRACT_SCHEMA = "msmarco-genqa.nfcorpus-video-query-representation.v1"
SUPPORTED_REPRESENTATIONS = ("title", "description", "title_plus_description")
TITLE_MEMBER = "nfcorpus/test.vid-titles.queries"
DESCRIPTION_MEMBER = "nfcorpus/test.vid-desc.queries"
_SPACE_RE = re.compile(r"\s+")
_ALIGNMENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MAX_QUERY_MEMBER_BYTES = 5 * 1024 * 1024


class NFCorpusVideoContractError(ValueError):
    """Raised when a pinned query-representation input violates its contract."""


@dataclass(frozen=True)
class NFCorpusVideoQueryRecord:
    """One official video record aligned to its BEIR query title."""

    query_id: str
    beir_title: str
    official_title: str
    description: str
    effective_query: str

    def to_json(self, *, representation: str) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "representation": representation,
            "beir_title": self.beir_title,
            "official_title": self.official_title,
            "description": self.description,
            "effective_query": self.effective_query,
        }


@dataclass(frozen=True)
class NFCorpusVideoQueryBundle:
    """Validated 102-query treatment bundle."""

    representation: str
    queries: dict[str, str]
    records: tuple[NFCorpusVideoQueryRecord, ...]
    summary: dict[str, object]
    contract_path: Path
    archive_path: Path
    frozen_title_metrics: dict[str, float]
    frozen_title_rerank_metrics: dict[str, float]
    positive_score_title_metrics: dict[str, float]
    deterministic_title_metrics: dict[str, float] | None
    deterministic_title_rerank_metrics: dict[str, float] | None


def normalize_query_field(text: str) -> str:
    """Normalize Unicode and collapse whitespace without lexical rewriting."""

    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text)).strip())


def _alignment_key(text: str) -> str:
    """Loose key used only to verify official/BEIR title identity."""

    return " ".join(_ALIGNMENT_TOKEN_RE.findall(str(text).casefold()))


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NFCorpusVideoContractError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, object], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise NFCorpusVideoContractError(f"{label}.{key} must be a non-empty string")
    return value


def _resolve_repo_path(raw_path: str, project_root: Path) -> Path:
    root = project_root.resolve()
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise NFCorpusVideoContractError(
            f"official archive path must stay inside the project root: {raw_path!r}"
        )
    return resolved


def _download_pinned_archive(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise NFCorpusVideoContractError("official archive URL must use HTTPS")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https":
                raise NFCorpusVideoContractError(
                    "official archive download redirected away from HTTPS"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise NFCorpusVideoContractError(
                    "official archive Content-Length differs from the contract"
                )
            downloaded = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                downloaded += len(chunk)
                if downloaded > expected_bytes:
                    raise NFCorpusVideoContractError(
                        "official archive download exceeds the contracted byte size"
                    )
                output.write(chunk)
            if downloaded != expected_bytes:
                raise NFCorpusVideoContractError(
                    "official archive download ended before the contracted byte size"
                )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _verify_archive(path: Path, contract: Mapping[str, object]) -> None:
    expected_bytes = contract.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise NFCorpusVideoContractError(
            "inputs.official_nfcorpus_v1.bytes must be a positive integer"
        )
    if path.stat().st_size != expected_bytes:
        raise NFCorpusVideoContractError(
            "official archive byte-size drift: "
            f"expected {expected_bytes}, found {path.stat().st_size}"
        )

    for algorithm in ("md5", "sha256"):
        expected = _require_string(
            contract,
            algorithm,
            "inputs.official_nfcorpus_v1",
        ).casefold()
        observed = _file_hash(path, algorithm)
        if observed != expected:
            raise NFCorpusVideoContractError(
                f"official archive {algorithm.upper()} drift: "
                f"expected {expected}, found {observed}"
            )


def _read_query_member(archive: tarfile.TarFile, member_name: str) -> dict[str, str]:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise NFCorpusVideoContractError(
            f"official archive is missing {member_name!r}"
        ) from exc
    if not member.isfile():
        raise NFCorpusVideoContractError(f"{member_name!r} is not a regular file")
    if member.size > _MAX_QUERY_MEMBER_BYTES:
        raise NFCorpusVideoContractError(
            f"{member_name!r} exceeds the query-member size limit"
        )

    extracted = archive.extractfile(member)
    if extracted is None:
        raise NFCorpusVideoContractError(f"could not read {member_name!r}")
    try:
        text = extracted.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NFCorpusVideoContractError(f"{member_name!r} is not UTF-8") from exc

    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            query_id, value = line.split("\t", 1)
        except ValueError as exc:
            raise NFCorpusVideoContractError(
                f"{member_name}:{line_number} is not a two-column TSV row"
            ) from exc
        query_id = query_id.strip()
        if not query_id or not value.strip():
            raise NFCorpusVideoContractError(
                f"{member_name}:{line_number} contains an empty id or text field"
            )
        if query_id in records:
            raise NFCorpusVideoContractError(
                f"{member_name}:{line_number} duplicates query id {query_id!r}"
            )
        records[query_id] = value
    return records


def _qid_hash(query_ids: list[str]) -> str:
    canonical = "".join(f"{query_id}\n" for query_id in sorted(query_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record_hash(titles: Mapping[str, str], descriptions: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(titles):
        row = {
            "description": descriptions[query_id],
            "qid": query_id,
            "title": titles[query_id],
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(f"{encoded}\n".encode("utf-8"))
    return digest.hexdigest()


def _effective_query_hash(queries: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(queries):
        row = {"query_id": query_id, "text": queries[query_id]}
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(f"{encoded}\n".encode("utf-8"))
    return digest.hexdigest()


def _build_effective_query(
    representation: str,
    *,
    beir_title: str,
    description: str,
) -> str:
    title = normalize_query_field(beir_title)
    desc = normalize_query_field(description)
    if representation == "title":
        return title
    if representation == "description":
        return desc
    if representation == "title_plus_description":
        return f"{title}\n{desc}"
    raise NFCorpusVideoContractError(
        f"unsupported query representation {representation!r}; "
        f"expected one of {SUPPORTED_REPRESENTATIONS}"
    )


def load_nfcorpus_video_query_representation(
    baseline_queries: Mapping[str, str],
    *,
    representation: str,
    contract_path: Path,
    project_root: Path,
    download_if_missing: bool = True,
) -> NFCorpusVideoQueryBundle:
    """Load and validate one predeclared query representation.

    Only ``baseline_queries`` and the pinned official title/description members
    are visible to this function. The caller applies the returned qid cohort to
    qrels after query construction.
    """

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise NFCorpusVideoContractError(
            f"unsupported query representation {representation!r}; "
            f"expected one of {SUPPORTED_REPRESENTATIONS}"
        )

    contract_path = contract_path.resolve()
    try:
        contract_raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NFCorpusVideoContractError(
            f"could not read query-representation contract {contract_path}"
        ) from exc
    contract = _require_mapping(contract_raw, "contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise NFCorpusVideoContractError(
            f"unsupported contract schema {contract.get('schema')!r}"
        )

    declared_representations = _require_mapping(
        contract.get("query_representations"),
        "query_representations",
    )
    if representation not in declared_representations:
        raise NFCorpusVideoContractError(
            f"representation {representation!r} is not predeclared by the contract"
        )

    inputs = _require_mapping(contract.get("inputs"), "inputs")
    official = _require_mapping(
        inputs.get("official_nfcorpus_v1"),
        "inputs.official_nfcorpus_v1",
    )
    archive_path = _resolve_repo_path(
        _require_string(official, "path", "inputs.official_nfcorpus_v1"),
        project_root,
    )
    expected_bytes = official.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise NFCorpusVideoContractError(
            "inputs.official_nfcorpus_v1.bytes must be a positive integer"
        )
    if not archive_path.exists():
        if not download_if_missing:
            raise NFCorpusVideoContractError(
                f"official archive is missing and downloads are disabled: {archive_path}"
            )
        _download_pinned_archive(
            _require_string(official, "url", "inputs.official_nfcorpus_v1"),
            archive_path,
            expected_bytes=expected_bytes,
        )
    _verify_archive(archive_path, official)

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            official_titles = _read_query_member(archive, TITLE_MEMBER)
            descriptions = _read_query_member(archive, DESCRIPTION_MEMBER)
    except (tarfile.TarError, OSError) as exc:
        raise NFCorpusVideoContractError("official archive is not a valid tar.gz") from exc

    if set(official_titles) != set(descriptions):
        raise NFCorpusVideoContractError(
            "official video title and description query-id sets differ"
        )

    cohort = _require_mapping(contract.get("cohort"), "cohort")
    expected_count = cohort.get("n_queries")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise NFCorpusVideoContractError("cohort.n_queries must be a positive integer")
    if len(official_titles) != expected_count:
        raise NFCorpusVideoContractError(
            f"video cohort size drift: expected {expected_count}, found {len(official_titles)}"
        )

    observed_qid_hash = _qid_hash(list(official_titles))
    expected_qid_hash = _require_string(cohort, "qid_sha256", "cohort").casefold()
    if observed_qid_hash != expected_qid_hash:
        raise NFCorpusVideoContractError(
            f"video cohort qid hash drift: expected {expected_qid_hash}, "
            f"found {observed_qid_hash}"
        )

    observed_record_hash = _record_hash(official_titles, descriptions)
    expected_record_hash = _require_string(
        cohort,
        "official_query_records_sha256",
        "cohort",
    ).casefold()
    if observed_record_hash != expected_record_hash:
        raise NFCorpusVideoContractError(
            f"official video record hash drift: expected {expected_record_hash}, "
            f"found {observed_record_hash}"
        )

    missing_baseline = sorted(set(official_titles) - set(baseline_queries))
    if missing_baseline:
        raise NFCorpusVideoContractError(
            f"{len(missing_baseline)} video qids are absent from the BEIR test queries"
        )

    mismatched_titles = [
        query_id
        for query_id, title in official_titles.items()
        if _alignment_key(title) != _alignment_key(baseline_queries[query_id])
    ]
    if mismatched_titles:
        raise NFCorpusVideoContractError(
            f"{len(mismatched_titles)} official titles do not align with BEIR query text"
        )

    records: list[NFCorpusVideoQueryRecord] = []
    queries: dict[str, str] = {}
    for query_id in sorted(official_titles):
        effective_query = _build_effective_query(
            representation,
            beir_title=baseline_queries[query_id],
            description=descriptions[query_id],
        )
        record = NFCorpusVideoQueryRecord(
            query_id=query_id,
            beir_title=normalize_query_field(baseline_queries[query_id]),
            official_title=normalize_query_field(official_titles[query_id]),
            description=normalize_query_field(descriptions[query_id]),
            effective_query=effective_query,
        )
        records.append(record)
        queries[query_id] = effective_query

    frozen = _require_mapping(contract.get("frozen_title_baseline"), "frozen_title_baseline")
    frozen_bm25 = _require_mapping(frozen.get("bm25"), "frozen_title_baseline.bm25")
    frozen_bm25_ce = _require_mapping(
        frozen.get("bm25_ce"),
        "frozen_title_baseline.bm25_ce",
    )
    frozen_title_metrics = {
        metric: float(frozen_bm25[metric])
        for metric in ("mrr@10", "ndcg@10", "recall@100", "recall@1000")
    }
    frozen_title_rerank_metrics = {
        metric: float(frozen_bm25_ce[metric])
        for metric in ("mrr@10", "ndcg@10", "recall@100")
    }
    positive_score_bm25 = _require_mapping(
        frozen.get("positive_score_bm25"),
        "frozen_title_baseline.positive_score_bm25",
    )
    positive_score_title_metrics = {
        metric: float(positive_score_bm25[metric])
        for metric in (
            "positive_score_recall@100",
            "positive_score_recall@1000",
        )
    }
    deterministic_bm25_raw = frozen.get("deterministic_tie_bm25")
    deterministic_title_metrics = (
        {
            metric: float(deterministic_bm25_raw[metric])
            for metric in ("mrr@10", "ndcg@10", "recall@100", "recall@1000")
        }
        if isinstance(deterministic_bm25_raw, dict)
        else None
    )
    deterministic_ce_raw = frozen.get("deterministic_tie_bm25_ce")
    deterministic_title_rerank_metrics = (
        {
            metric: float(deterministic_ce_raw[metric])
            for metric in ("mrr@10", "ndcg@10", "recall@100")
        }
        if isinstance(deterministic_ce_raw, dict)
        else None
    )
    summary: dict[str, object] = {
        "schema": CONTRACT_SCHEMA,
        "dataset_id": _require_string(
            official,
            "dataset_id",
            "inputs.official_nfcorpus_v1",
        ),
        "representation": representation,
        "n_queries": len(queries),
        "qid_sha256": observed_qid_hash,
        "official_query_records_sha256": observed_record_hash,
        "effective_queries_sha256": _effective_query_hash(queries),
        "source_archive": {
            "path": archive_path.relative_to(project_root.resolve()).as_posix(),
            "bytes": archive_path.stat().st_size,
            "md5": _file_hash(archive_path, "md5"),
            "sha256": _file_hash(archive_path, "sha256"),
        },
        "title_alignment": {
            "matched": len(queries),
            "mismatched": 0,
        },
        "ranking_tie_break": {
            "order": "score descending, then doc_id ascending",
            "candidate_pool": "full 3,633-document corpus before top-k truncation",
            "reason": "Cross-platform BM25 top-k selection is unstable among equal scores.",
        },
        "leakage_boundary": {
            "inputs": [
                "frozen_video_qids",
                "beir_query_title",
                "official_video_title",
                "official_video_description",
            ],
            "excluded": [
                "qrels",
                "corpus_documents",
                "ranked_outputs",
                "metrics",
                "manual_rewrites",
            ],
        },
    }
    return NFCorpusVideoQueryBundle(
        representation=representation,
        queries=queries,
        records=tuple(records),
        summary=summary,
        contract_path=contract_path,
        archive_path=archive_path,
        frozen_title_metrics=frozen_title_metrics,
        frozen_title_rerank_metrics=frozen_title_rerank_metrics,
        positive_score_title_metrics=positive_score_title_metrics,
        deterministic_title_metrics=deterministic_title_metrics,
        deterministic_title_rerank_metrics=deterministic_title_rerank_metrics,
    )


def write_nfcorpus_video_query_artifacts(
    bundle: NFCorpusVideoQueryBundle,
    output_dir: Path,
) -> tuple[dict[str, object], list[Path]]:
    """Write the effective-query audit trail for one retrieval run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    summary_path = output_dir / "summary.json"
    with queries_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in bundle.records:
            handle.write(
                json.dumps(
                    record.to_json(representation=bundle.representation),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    summary = dict(bundle.summary)
    summary["queries_path"] = "query_representation/queries.jsonl"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary, [queries_path, summary_path]


def validate_frozen_title_metrics(
    bundle: NFCorpusVideoQueryBundle,
    metrics: Mapping[str, object],
    *,
    positive_score_metrics: Mapping[str, object],
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Validate the frozen deterministic baseline and report historical deltas."""

    if bundle.representation != "title":
        return {"required": False, "passed": None}

    published_reference_deltas: dict[str, float] = {}
    for metric, expected in bundle.frozen_title_metrics.items():
        if metric in metrics:
            published_reference_deltas[metric] = float(metrics[metric]) - expected
    for metric, expected in bundle.positive_score_title_metrics.items():
        if metric in positive_score_metrics:
            published_reference_deltas[metric] = (
                float(positive_score_metrics[metric]) - expected
            )

    deterministic_deltas: dict[str, float] | None = None
    deterministic_status = "pending_freeze"
    if bundle.deterministic_title_metrics is not None:
        deterministic_deltas = {}
        for metric, expected in bundle.deterministic_title_metrics.items():
            if metric not in metrics:
                raise NFCorpusVideoContractError(
                    f"deterministic title baseline is missing metric {metric}"
                )
            observed = float(metrics[metric])
            delta = observed - expected
            deterministic_deltas[metric] = delta
            if abs(delta) > tolerance:
                raise NFCorpusVideoContractError(
                    f"deterministic title metric drift for {metric}: "
                    f"expected {expected:.16g}, found {observed:.16g}, "
                    f"delta {delta:.3g}"
                )
        deterministic_status = "passed"
    return {
        "required": True,
        "passed": True,
        "tolerance": tolerance,
        "published_reference_deltas": published_reference_deltas,
        "deterministic_baseline": {
            "status": deterministic_status,
            "metric_deltas": deterministic_deltas,
        },
    }


def validate_frozen_title_reranker_metrics(
    bundle: NFCorpusVideoQueryBundle,
    input_metrics: Mapping[str, object],
    rerank_metrics: Mapping[str, object],
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Require title-only BM25 and CE metrics to reproduce the frozen slice."""

    if bundle.representation != "title":
        return {"required": False, "passed": None}

    if (
        bundle.deterministic_title_metrics is None
        or bundle.deterministic_title_rerank_metrics is None
    ):
        return {
            "required": False,
            "passed": None,
            "status": "pending_deterministic_baseline_freeze",
        }

    metric_deltas: dict[str, dict[str, float]] = {"bm25": {}, "bm25_ce": {}}
    expected_groups = (
        ("bm25", bundle.deterministic_title_metrics, input_metrics),
        ("bm25_ce", bundle.deterministic_title_rerank_metrics, rerank_metrics),
    )
    for group, expected_metrics, observed_metrics in expected_groups:
        for metric, expected in expected_metrics.items():
            if metric == "recall@1000":
                continue
            if metric not in observed_metrics:
                raise NFCorpusVideoContractError(
                    f"title reranker reproduction is missing {group} metric {metric}"
                )
            observed = float(observed_metrics[metric])
            delta = observed - expected
            metric_deltas[group][metric] = delta
            if abs(delta) > tolerance:
                raise NFCorpusVideoContractError(
                    f"title reranker metric drift for {group}.{metric}: "
                    f"expected {expected:.16g}, found {observed:.16g}, "
                    f"delta {delta:.3g}"
                )
    return {
        "required": True,
        "passed": True,
        "tolerance": tolerance,
        "metric_deltas": metric_deltas,
    }

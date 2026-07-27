"""Tests for frozen retrieval data and metric contracts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from msmarco_genqa.evaluation.retrieval_contract import (
    RetrievalContractError,
    verify_retrieval_contract,
)


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_run(
    path: Path,
    rows: dict[str, list[tuple[str, float]]],
    *,
    system: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{system}"
        for qid in sorted(rows)
        for rank, (doc_id, score) in enumerate(rows[qid], start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _make_contract(root: Path) -> Path:
    source = root / "inputs" / "source.zip"
    source.parent.mkdir(parents=True, exist_ok=True)
    query_rows = [
        {"_id": "q1", "text": "one"},
        {"_id": "q2", "text": "two"},
        {"_id": "train-only", "text": "extra"},
    ]
    corpus_rows = [
        {"_id": "d1", "text": "one"},
        {"_id": "d2", "text": "two"},
        {"_id": "x1", "text": "other"},
        {"_id": "x2", "text": "other"},
    ]
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "nfcorpus/queries.jsonl",
            "".join(json.dumps(row) + "\n" for row in query_rows),
        )
        archive.writestr(
            "nfcorpus/corpus.jsonl",
            "".join(json.dumps(row) + "\n" for row in corpus_rows),
        )

    qrels = root / "inputs" / "qrels.trec"
    qrels.write_text("q1 0 d1 1\nq2 0 d2 2\n", encoding="utf-8", newline="\n")
    bm25 = root / "inputs" / "bm25.tsv"
    ce = root / "inputs" / "ce.tsv"
    _write_run(
        bm25,
        {"q1": [("d1", 2.0), ("x1", 1.0)], "q2": [("d2", 2.0), ("x2", 1.0)]},
        system="bm25",
    )
    _write_run(
        ce,
        {"q1": [("d1", 3.0)], "q2": [("d2", 3.0)]},
        system="bm25_ce",
    )

    release_asset = root / "inputs" / "release.zip"
    release_asset.write_bytes(b"release fixture")
    pointer = {
        "release": {
            "repository": "GioiaZheng/msmarco-genqa",
            "tag": "fixture",
            "asset": "release.zip",
        },
        "download": {
            "bytes": release_asset.stat().st_size,
            "sha256": hashlib.sha256(release_asset.read_bytes()).hexdigest(),
        },
    }
    pointer_path = root / "artifacts" / "pointer.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8", newline="\n")

    contract = {
        "schema": "msmarco-genqa.retrieval-data-metric-contract.v1",
        "dataset_id": "beir/nfcorpus/test",
        "analysis_base_commit": "a" * 40,
        "experiment_commit": "b" * 12,
        "release": {
            "pointer_path": pointer_path.relative_to(root).as_posix(),
            **pointer["release"],
        },
        "inputs": {
            "release_asset": _file_record(release_asset, root),
            "source_archive": {
                **_file_record(source, root),
                "query_member": "nfcorpus/queries.jsonl",
                "corpus_member": "nfcorpus/corpus.jsonl",
            },
            "qrels": _file_record(qrels, root),
            "bm25_run": {
                **_file_record(bm25, root),
                "query_count": 2,
                "row_count": 4,
                "depth": 2,
            },
            "ce_run": {
                **_file_record(ce, root),
                "query_count": 2,
                "row_count": 2,
                "depth": 1,
            },
        },
        "expected_counts": {
            "source_queries": 3,
            "corpus_documents": 4,
            "test_queries": 2,
            "qrels_judgments": 2,
        },
        "qrels_format": "trec",
        "binary_relevance_threshold": 1,
        "expected_bm25_metrics": {
            "mrr@10": 1.0,
            "ndcg@10": 1.0,
            "recall@100": 1.0,
            "recall@1000": 1.0,
        },
        "metric_tolerance": 1e-12,
    }
    contract_path = root / "configs" / "contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return contract_path


def test_contract_verifies_structure_identifiers_and_metrics(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)

    report = verify_retrieval_contract(contract_path, project_root=tmp_path)

    assert report["verified"] is True
    assert report["scope"]["test_queries"] == 2
    assert report["scope"]["bm25"] == {
        "query_count": 2,
        "row_count": 4,
        "depth": 2,
    }
    assert report["scope"]["ce"]["depth"] == 1
    assert report["max_abs_delta"] == 0.0
    assert all(report["integrity"].values())


def test_contract_rejects_release_or_input_hash_drift(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    bm25 = tmp_path / contract["inputs"]["bm25_run"]["path"]
    bm25.write_text(
        bm25.read_text(encoding="utf-8") + "q3\tQ0\td1\t1\t1.0\tbm25\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RetrievalContractError, match="byte-size drift"):
        verify_retrieval_contract(contract_path, project_root=tmp_path)


def test_contract_rejects_unknown_document_ids(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    bm25 = tmp_path / contract["inputs"]["bm25_run"]["path"]
    text = bm25.read_text(encoding="utf-8").replace("x2", "unknown")
    bm25.write_text(text, encoding="utf-8", newline="\n")
    contract["inputs"]["bm25_run"].update(_file_record(bm25, tmp_path))
    contract_path.write_text(json.dumps(contract), encoding="utf-8", newline="\n")

    with pytest.raises(RetrievalContractError, match="absent from the source"):
        verify_retrieval_contract(contract_path, project_root=tmp_path)


def test_contract_rejects_ce_candidate_set_changes(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ce = tmp_path / contract["inputs"]["ce_run"]["path"]
    text = ce.read_text(encoding="utf-8").replace(
        "q2\tQ0\td2", "q2\tQ0\tx2"
    )
    ce.write_text(text, encoding="utf-8", newline="\n")
    contract["inputs"]["ce_run"].update(_file_record(ce, tmp_path))
    contract_path.write_text(json.dumps(contract), encoding="utf-8", newline="\n")

    with pytest.raises(RetrievalContractError, match="candidate set differs"):
        verify_retrieval_contract(contract_path, project_root=tmp_path)


def test_contract_rejects_metric_drift(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["expected_bm25_metrics"]["mrr@10"] = 0.5
    contract_path.write_text(json.dumps(contract), encoding="utf-8", newline="\n")

    with pytest.raises(RetrievalContractError, match="metric drift"):
        verify_retrieval_contract(contract_path, project_root=tmp_path)


def test_contract_rejects_non_finite_expected_metric(tmp_path: Path) -> None:
    contract_path = _make_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["expected_bm25_metrics"]["mrr@10"] = float("nan")
    contract_path.write_text(json.dumps(contract), encoding="utf-8", newline="\n")

    with pytest.raises(RetrievalContractError, match="finite number"):
        verify_retrieval_contract(contract_path, project_root=tmp_path)

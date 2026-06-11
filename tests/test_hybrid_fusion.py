"""Tests for weighted RRF hybrid retrieval fusion."""

from __future__ import annotations

import json

import pytest

from experiments import run_hybrid_fusion
from msmarco_genqa.retrieval.fusion import fused_doc_ids_and_scores, reciprocal_rank_fusion


def test_rrf_accumulates_duplicate_docs_across_sources():
    fused = reciprocal_rank_fusion(
        {
            "bm25": {"q1": [("d_sparse", 9.0), ("d_shared", 8.0)]},
            "dense": {"q1": [("d_shared", 0.9), ("d_dense", 0.8)]},
        },
        rank_constant=60,
    )

    q1 = fused["q1"]
    assert q1[0].doc_id == "d_shared"
    assert q1[0].score == pytest.approx((1 / 62) + (1 / 61))
    assert set(q1[0].sources) == {"bm25", "dense"}
    assert q1[0].sources["bm25"].rank == 2
    assert q1[0].sources["dense"].rank == 1


def test_rrf_weights_change_ranking():
    runs = {
        "bm25": {"q1": [("d_sparse", 10.0)]},
        "dense": {"q1": [("d_dense", 1.0)]},
    }

    equal = reciprocal_rank_fusion(runs, rank_constant=60)
    weighted = reciprocal_rank_fusion(runs, rank_constant=60, weights={"bm25": 2.0})

    assert [hit.doc_id for hit in equal["q1"]] == ["d_dense", "d_sparse"]
    assert [hit.doc_id for hit in weighted["q1"]] == ["d_sparse", "d_dense"]
    assert weighted["q1"][0].sources["bm25"].weight == 2.0


def test_rrf_handles_missing_query_in_one_source():
    fused = reciprocal_rank_fusion(
        {
            "bm25": {"q1": [("d1", 1.0)]},
            "dense": {"q2": [("d2", 1.0)]},
        }
    )

    assert sorted(fused) == ["q1", "q2"]
    assert fused["q1"][0].doc_id == "d1"
    assert fused["q2"][0].doc_id == "d2"


def test_rrf_ties_are_deterministic_by_score_best_rank_doc_id():
    fused = reciprocal_rank_fusion(
        {
            "a": {"q1": [("d_b", 1.0)]},
            "b": {"q1": [("d_a", 1.0)]},
        },
        rank_constant=60,
    )

    assert [hit.doc_id for hit in fused["q1"]] == ["d_a", "d_b"]


def test_rrf_rejects_duplicate_doc_within_source_query():
    with pytest.raises(ValueError, match="duplicate document id"):
        reciprocal_rank_fusion(
            {"bm25": {"q1": [("d1", 2.0), ("d1", 1.0)]}},
        )


def test_rrf_validates_weight_names():
    with pytest.raises(ValueError, match="unknown source"):
        reciprocal_rank_fusion(
            {"bm25": {"q1": [("d1", 1.0)]}},
            weights={"dense": 1.5},
        )


def test_fused_doc_ids_and_scores_shape():
    fused = reciprocal_rank_fusion(
        {"a": {"q2": [("d2", 1.0)], "q1": [("d1", 1.0)]}},
        top_k=1,
    )

    qids, doc_ids, scores = fused_doc_ids_and_scores(fused)
    assert qids == ["q1", "q2"]
    assert doc_ids == [["d1"], ["d2"]]
    assert len(scores) == 2


def test_hybrid_fusion_runner_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_hybrid_fusion, "PROJECT_ROOT", tmp_path)
    bm25 = tmp_path / "bm25.tsv"
    bm25.write_text(
        "q1\tQ0\td_sparse\t1\t10.0\tbm25\n"
        "q1\tQ0\td_shared\t2\t9.0\tbm25\n"
        "q2\tQ0\td_other\t1\t8.0\tbm25\n",
        encoding="utf-8",
    )
    dense = tmp_path / "dense.tsv"
    dense.write_text(
        "q1\tQ0\td_shared\t1\t1.0\tdense\n"
        "q1\tQ0\td_dense\t2\t0.5\tdense\n"
        "q2\tQ0\td_other\t1\t0.9\tdense\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1 0 d_shared 1\nq2 0 d_other 1\n", encoding="utf-8")

    output_dir = tmp_path / "outputs" / "rrf"
    run_hybrid_fusion.main(
        [
            "--input-run",
            f"bm25={bm25}",
            "--input-run",
            f"dense={dense}",
            "--output-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--qrels",
            str(qrels),
            "--allow-incomplete-manifest",
        ]
    )

    run_lines = (output_dir / "run.tsv").read_text(encoding="utf-8").splitlines()
    assert run_lines[0].startswith("q1\tQ0\td_shared\t1\t")
    assert (output_dir / "provenance.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "manifest.json").exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["fusion"]["method"] == "weighted_rrf"
    assert metrics["fusion"]["input_runs"]["bm25"] == "bm25.tsv"
    assert metrics["metrics"]["rrf"]["mrr@10"] == pytest.approx(1.0)
    assert metrics["overlap"] == {"qids_shared_all": 2, "qids_union": 2}

    first_provenance = json.loads(
        (output_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_provenance["doc_id"] == "d_shared"
    assert set(first_provenance["sources"]) == {"bm25", "dense"}

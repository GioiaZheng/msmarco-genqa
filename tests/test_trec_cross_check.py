"""Fixture tests for TREC-compatible retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from msmarco_genqa.evaluation.trec import (
    MetricCrossCheckError,
    OptionalEvaluatorUnavailable,
    QrelsFormatError,
    compare_metric_sets,
    read_qrels,
    run_trec_cross_check,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trec_eval"


class _FakeMeasure:
    def __init__(self, name: str) -> None:
        self.name = name

    def __matmul__(self, cutoff: int) -> "_FakeMeasure":
        return _FakeMeasure(f"{self.name}@{cutoff}")

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeMeasure) and self.name == other.name


class _FakeIrMeasures:
    RR = _FakeMeasure("RR")
    nDCG = _FakeMeasure("nDCG")
    R = _FakeMeasure("R")

    @staticmethod
    def calc_aggregate(measures, qrels, run):
        assert set(qrels) == {"q1", "q2"}
        assert run["q1"] == {"d1": 1.0, "d3": 0.5}
        expected = {
            "RR@10": 0.75,
            "nDCG@10": 0.622038473168458,
            "R@1000": 0.75,
        }
        return {measure: expected[measure.name] for measure in measures}


def test_fixture_export_and_cross_check(tmp_path):
    report = run_trec_cross_check(
        run_path=FIXTURE_DIR / "run.tsv",
        qrels_path=FIXTURE_DIR / "qrels.trec",
        output_dir=tmp_path,
        qrels_format="trec",
        backend="ir-measures",
        ir_measures_module=_FakeIrMeasures,
    )

    assert report["scope"] == {
        "run_queries": 2,
        "qrels_queries": 2,
        "evaluated_queries": 2,
    }
    assert report["internal_metrics"]["mrr@10"] == pytest.approx(0.75)
    assert report["internal_metrics"]["ndcg@10"] == pytest.approx(0.622038473168458)
    assert report["internal_metrics"]["recall@1000"] == pytest.approx(0.75)
    assert report["external_evaluator"]["status"] == "passed"
    assert max(report["external_evaluator"]["absolute_deltas"].values()) < 1e-12

    assert (tmp_path / "run.trec").read_text().splitlines() == [
        "q1\tQ0\td1\t1\t1\tmsmarco-genqa",
        "q1\tQ0\td3\t2\t0.5\tmsmarco-genqa",
        "q2\tQ0\td9\t1\t1\tmsmarco-genqa",
        "q2\tQ0\td2\t2\t0.5\tmsmarco-genqa",
    ]
    assert (tmp_path / "qrels.trec").read_text().splitlines() == [
        "q1\t0\td1\t1",
        "q1\t0\td4\t1",
        "q2\t0\td2\t1",
    ]
    assert (tmp_path / "metrics.json").exists()


def test_missing_run_query_counts_as_zero(tmp_path):
    run_path = tmp_path / "run.tsv"
    run_path.write_text("q1\tQ0\td1\t1\t1.0\tfixture\n")
    report = run_trec_cross_check(
        run_path=run_path,
        qrels_path=FIXTURE_DIR / "qrels.trec",
        output_dir=tmp_path / "out",
        qrels_format="trec",
        backend="none",
    )
    assert report["internal_metrics"]["mrr@10"] == pytest.approx(0.5)
    assert report["internal_metrics"]["recall@1000"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("content", "qrels_format"),
    [
        ("q1 0 d1 1\n", "auto"),
        ("q1 d1 1 0\n", "auto"),
        ("q1 d1 1\n", "auto"),
        ("q1 d1 1 0\n", "irds-tsv"),
    ],
)
def test_qrels_layouts(tmp_path, content, qrels_format):
    path = tmp_path / "qrels.txt"
    path.write_text(content)
    assert read_qrels(path, qrels_format=qrels_format) == {"q1": {"d1": 1}}


def test_ambiguous_qrels_requires_explicit_format(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1 0\n")
    with pytest.raises(QrelsFormatError, match="ambiguous 4-field qrels row"):
        read_qrels(path)


def test_duplicate_qrels_fail_fast(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1 1\nq1 0 d1 1\n")
    with pytest.raises(QrelsFormatError, match="duplicate judgment"):
        read_qrels(path, qrels_format="trec")


def test_cross_check_rejects_graded_qrels(tmp_path):
    qrels_path = tmp_path / "graded.qrels"
    qrels_path.write_text("q1 0 d1 2\n")
    with pytest.raises(ValueError, match="only binary relevance levels"):
        run_trec_cross_check(
            run_path=FIXTURE_DIR / "run.tsv",
            qrels_path=qrels_path,
            output_dir=tmp_path / "out",
            qrels_format="trec",
            backend="none",
        )


def test_metric_mismatch_fails_gate():
    internal = {"mrr@10": 0.5, "ndcg@10": 0.6, "recall@1000": 0.7}
    external = {"mrr@10": 0.5, "ndcg@10": 0.61, "recall@1000": 0.7}
    with pytest.raises(MetricCrossCheckError, match="ndcg@10"):
        compare_metric_sets(internal, external, tolerance=1e-6)


def test_auto_backend_records_unavailable(monkeypatch, tmp_path):
    def unavailable(_name):
        raise ModuleNotFoundError("ir_measures")

    monkeypatch.setattr("msmarco_genqa.evaluation.trec.importlib.import_module", unavailable)
    report = run_trec_cross_check(
        run_path=FIXTURE_DIR / "run.tsv",
        qrels_path=FIXTURE_DIR / "qrels.trec",
        output_dir=tmp_path,
        qrels_format="trec",
        backend="auto",
    )
    assert report["external_evaluator"]["status"] == "unavailable"
    assert "pip install" in report["external_evaluator"]["reason"]


def test_required_backend_fails_when_unavailable(monkeypatch, tmp_path):
    def unavailable(_name):
        raise ModuleNotFoundError("ir_measures")

    monkeypatch.setattr("msmarco_genqa.evaluation.trec.importlib.import_module", unavailable)
    with pytest.raises(OptionalEvaluatorUnavailable):
        run_trec_cross_check(
            run_path=FIXTURE_DIR / "run.tsv",
            qrels_path=FIXTURE_DIR / "qrels.trec",
            output_dir=tmp_path,
            qrels_format="trec",
            backend="ir-measures",
        )


def test_real_ir_measures_backend_when_installed(tmp_path):
    pytest.importorskip("ir_measures")
    report = run_trec_cross_check(
        run_path=FIXTURE_DIR / "run.tsv",
        qrels_path=FIXTURE_DIR / "qrels.trec",
        output_dir=tmp_path,
        qrels_format="trec",
        backend="ir-measures",
    )
    assert report["external_evaluator"]["status"] == "passed"
    assert max(report["external_evaluator"]["absolute_deltas"].values()) < 1e-12

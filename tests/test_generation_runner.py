"""Tests for the generation-runner plumbing in
``experiments/run_generation_baseline.py``.

The runner itself wires together a T5 generator and MS MARCO data, both of
which are slow / network-bound. These tests cover only the *pure* helpers
that decide which run.tsv to read, where to write outputs, and which query
ids are eligible — i.e. the surface that gained CLI overrides as part of
making the runner retrieval-source agnostic.

Nothing here downloads a model or touches HuggingFace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_generation_baseline import (
    compute_eligible,
    infer_retrieval_source,
    load_runs,
    parse_args,
    resolve_input_run,
    resolve_output_dir,
)
from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator
from msmarco_genqa.reranking.io import RunTsvFormatError


# --------------------------------------------------------------------------- #
# load_runs — the TREC-format reader the runner uses
# --------------------------------------------------------------------------- #


def _write_run(path: Path, rows: list[tuple[str, str, int, float]]) -> None:
    """Helper: write a minimal TREC-format run with (qid, doc, rank, score)."""
    with open(path, "w") as f:
        for qid, doc, rank, score in rows:
            f.write(f"{qid}\tQ0\t{doc}\t{rank}\t{score:.4f}\ttest\n")


class TestLoadRuns:
    def test_basic_round_trip(self, tmp_path: Path):
        run = tmp_path / "run.tsv"
        _write_run(
            run,
            [
                ("q1", "d_a", 1, 9.0),
                ("q1", "d_b", 2, 8.0),
                ("q1", "d_c", 3, 7.0),
                ("q2", "d_x", 1, 5.0),
            ],
        )
        runs = load_runs(run)
        assert runs == {"q1": ["d_a", "d_b", "d_c"], "q2": ["d_x"]}

    def test_rejects_malformed_short_lines(self, tmp_path: Path):
        run = tmp_path / "run.tsv"
        run.write_text("q1\tQ0\td_a\t1\t9.0\ttest\nbroken\nq2\tQ0\td_x\t1\t1.0\ttest\n")
        with pytest.raises(RunTsvFormatError, match="expected 6 tab-separated fields"):
            load_runs(run)

    def test_handles_out_of_order_ranks(self, tmp_path: Path):
        run = tmp_path / "run.tsv"
        _write_run(
            run,
            [
                ("q1", "d_b", 2, 8.0),
                ("q1", "d_a", 1, 9.0),
                ("q1", "d_c", 3, 7.0),
            ],
        )
        runs = load_runs(run)
        assert runs["q1"] == ["d_a", "d_b", "d_c"]

    def test_rejects_duplicate_document_ids(self, tmp_path: Path):
        run = tmp_path / "run.tsv"
        _write_run(
            run,
            [
                ("q1", "d_a", 1, 9.0),
                ("q1", "d_a", 2, 8.0),
            ],
        )
        with pytest.raises(RunTsvFormatError, match="duplicate document id"):
            load_runs(run)


# --------------------------------------------------------------------------- #
# parse_args + resolve_* — CLI vs config-derived defaults
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_cfg() -> dict:
    return {
        "eval_retrieval": {"output_dir": "outputs/bm25_baseline"},
        "generation": {"output_dir": "outputs/generation"},
    }


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    return tmp_path


class TestResolveInputRun:
    def test_default_is_bm25_w2_run(self, fake_cfg, fake_root):
        args = parse_args([])
        assert resolve_input_run(args, fake_cfg, fake_root) == (
            fake_root / "outputs/bm25_baseline/run.tsv"
        )

    def test_cli_override_relative(self, fake_cfg, fake_root):
        args = parse_args(["--input-run", "outputs/cross_encoder_rerank/run.tsv"])
        assert resolve_input_run(args, fake_cfg, fake_root) == (
            fake_root / "outputs/cross_encoder_rerank/run.tsv"
        )

    def test_cli_override_absolute(self, fake_cfg, fake_root, tmp_path):
        abs_path = tmp_path / "elsewhere" / "run.tsv"
        args = parse_args(["--input-run", str(abs_path)])
        # Absolute paths must NOT be re-rooted under PROJECT_ROOT.
        assert resolve_input_run(args, fake_cfg, fake_root) == abs_path


class TestResolveOutputDir:
    def test_default_is_w3_output_dir(self, fake_cfg, fake_root):
        args = parse_args([])
        assert resolve_output_dir(args, fake_cfg, fake_root) == (
            fake_root / "outputs/generation"
        )

    def test_cli_override(self, fake_cfg, fake_root):
        args = parse_args(["--output-dir", "outputs/generation_reranked"])
        assert resolve_output_dir(args, fake_cfg, fake_root) == (
            fake_root / "outputs/generation_reranked"
        )


# --------------------------------------------------------------------------- #
# infer_retrieval_source — manifest label
# --------------------------------------------------------------------------- #


class TestInferRetrievalSource:
    @pytest.mark.parametrize(
        "path, expected",
        [
            (Path("outputs/bm25_baseline/run.tsv"), "bm25"),
            (Path("outputs/bm25_full/run.tsv"), "bm25"),
            (Path("outputs/dense_retrieval/run.tsv"), "dense"),
            (Path("outputs/dense_minilm/run.tsv"), "dense"),
            (Path("outputs/cross_encoder_rerank/run.tsv"), "reranked"),
            (Path("outputs/some_rerank/run.tsv"), "reranked"),
            (Path("outputs/unlabeled_run/run.tsv"), "unknown"),
        ],
    )
    def test_labels(self, path: Path, expected: str):
        assert infer_retrieval_source(path) == expected


# --------------------------------------------------------------------------- #
# compute_eligible — the apples-to-apples query selector
# --------------------------------------------------------------------------- #


class TestComputeEligible:
    def test_three_way_intersection(self):
        runs = {"q1": ["d1"], "q2": ["d2"], "q3": ["d3"]}
        queries = {"q1": "t1", "q2": "t2", "q3": "t3", "q4": "t4"}
        qa = {"q1": ["a1"], "q2": ["a2"], "q4": ["a4"]}
        # q1, q2 are in all three; q3 has no QA; q4 has no run.
        assert compute_eligible(runs, queries, qa) == ["q1", "q2"]

    def test_restrict_to_run_further_narrows(self):
        runs = {"q1": ["d1"], "q2": ["d2"], "q3": ["d3"]}
        queries = {"q1": "t1", "q2": "t2", "q3": "t3"}
        qa = {"q1": ["a1"], "q2": ["a2"], "q3": ["a3"]}
        # Without restriction: all three are eligible.
        assert compute_eligible(runs, queries, qa) == ["q1", "q2", "q3"]
        # With restriction to {q2, q3, q99}: only q2 and q3.
        assert (
            compute_eligible(runs, queries, qa, restrict_qids={"q2", "q3", "q99"})
            == ["q2", "q3"]
        )

    def test_restriction_can_empty_the_eval_set(self):
        runs = {"q1": ["d1"]}
        queries = {"q1": "t1"}
        qa = {"q1": ["a1"]}
        assert compute_eligible(runs, queries, qa, restrict_qids=set()) == []

    def test_output_is_sorted_deterministically(self):
        runs = {"qz": ["d"], "qa": ["d"], "qm": ["d"]}
        queries = {"qz": "tz", "qa": "ta", "qm": "tm"}
        qa = {"qz": ["a"], "qa": ["a"], "qm": ["a"]}
        # Sorted lexicographically — important so seeded sampling is reproducible.
        assert compute_eligible(runs, queries, qa) == ["qa", "qm", "qz"]


# --------------------------------------------------------------------------- #
# End-to-end (parse_args path): the CLI we documented in the README works
# --------------------------------------------------------------------------- #


def test_documented_reranked_invocation_parses(fake_cfg, fake_root):
    """The exact invocation the README will show users:
    --input-run X --output-dir Y --retrieval-source reranked
    """
    argv = [
        "--input-run",
        "outputs/cross_encoder_rerank/run.tsv",
        "--output-dir",
        "outputs/generation_reranked",
        "--retrieval-source",
        "reranked",
    ]
    args = parse_args(argv)
    assert resolve_input_run(args, fake_cfg, fake_root) == (
        fake_root / "outputs/cross_encoder_rerank/run.tsv"
    )
    assert resolve_output_dir(args, fake_cfg, fake_root) == (
        fake_root / "outputs/generation_reranked"
    )
    assert args.retrieval_source == "reranked"
    # restrict_to_run not set in this canonical invocation.
    assert args.restrict_to_run is None


def test_context_packing_invocation_parses():
    args = parse_args(
        [
            "--context-packing",
            "--context-max-chars",
            "900",
            "--context-max-passage-chars",
            "320",
            "--context-sentence-selection",
            "query_overlap",
            "--context-ordering",
            "rank",
            "--no-context-deduplicate",
        ]
    )

    assert args.context_packing is True
    assert args.context_max_chars == 900
    assert args.context_max_passage_chars == 320
    assert args.context_sentence_selection == "query_overlap"
    assert args.context_ordering == "rank"
    assert args.no_context_deduplicate is True


def test_rag_generator_prompt_normalizes_and_truncates_without_model_load():
    generator = RAGGenerator.__new__(RAGGenerator)
    generator.config = RAGGenerationConfig(
        top_k_passages=2,
        max_query_chars=16,
        max_passage_chars=9,
    )

    prompt = generator.build_prompt(
        "  what   is dense retrieval exactly? ",
        [" first passage text ", "", "third passage"],
    )

    assert prompt == "question: what is dense context: first"

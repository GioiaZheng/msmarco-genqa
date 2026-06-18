from __future__ import annotations

import json

from msmarco_genqa.cli import query_transform as cli
from msmarco_genqa.retrieval.query_transform import (
    QueryTransformConfig,
    normalize_query,
    query_transform_config_hash,
    read_cached_transformations,
    transform_queries,
    transform_query,
    write_transformation_artifacts,
)


def test_normalize_query_handles_empty_and_whitespace():
    assert normalize_query("   ") == ""
    assert normalize_query("  What   IS   NLP?  ") == "what is nlp"


def test_lexical_expansion_adds_terms_for_short_queries():
    config = QueryTransformConfig.from_mapping({"method": "lexical_expansion"})

    transformed, added = transform_query("NYC weather?", config)

    assert transformed == "nyc weather new york city"
    assert added == ("new york city",)


def test_lexical_expansion_leaves_specific_queries_stable():
    config = QueryTransformConfig.from_mapping({"method": "lexical_expansion"})

    transformed, added = transform_query(
        "what is the capital city of france in europe",
        config,
    )

    assert transformed == "what is the capital city of france in europe"
    assert added == ()


def test_decontextualize_prepends_context_only_for_elliptical_query():
    config = QueryTransformConfig.from_mapping(
        {
            "method": "decontextualize",
            "decontextualization_context": "Symptoms of influenza",
        }
    )

    transformed, added = transform_query("what about children?", config)

    assert transformed == "symptoms of influenza what about children"
    assert added == ()


def test_config_hash_changes_with_method():
    normalize = QueryTransformConfig.from_mapping({"method": "normalize"})
    expansion = QueryTransformConfig.from_mapping({"method": "lexical_expansion"})

    assert query_transform_config_hash(normalize) != query_transform_config_hash(expansion)


def test_artifact_cache_requires_matching_hash_and_qids(tmp_path):
    config = QueryTransformConfig.from_mapping({"method": "normalize"})
    records = transform_queries({"q1": " What is NLP? "}, config)
    _summary, paths = write_transformation_artifacts(records, tmp_path, config)
    queries_path = paths[0]

    cached = read_cached_transformations(
        queries_path,
        expected_config_hash=query_transform_config_hash(config),
        expected_query_ids=["q1"],
    )
    assert cached == records

    assert (
        read_cached_transformations(
            queries_path,
            expected_config_hash="different",
            expected_query_ids=["q1"],
        )
        is None
    )
    assert (
        read_cached_transformations(
            queries_path,
            expected_config_hash=query_transform_config_hash(config),
            expected_query_ids=["q2"],
        )
        is None
    )


def test_transform_queries_records_original_and_transformed_text():
    config = QueryTransformConfig.from_mapping({"method": "normalize"})

    records = transform_queries({"q1": "  What   is NLP? "}, config)

    assert records[0].original_query == "  What   is NLP? "
    assert records[0].transformed_query == "what is nlp"
    assert records[0].changed is True


def test_cli_writes_local_query_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "query_transform:\n"
        "  method: normalize\n"
        "  output_dir: outputs/query_transform/normalize\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.tsv"
    queries_path.write_text("q1\t What   is NLP? \n", encoding="utf-8")

    cli.main(
        [
            "--config",
            str(config_path),
            "--queries-tsv",
            str(queries_path),
        ]
    )

    output_dir = tmp_path / "outputs" / "query_transform" / "normalize"
    lines = (output_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(lines[0])

    assert record["original_query"] == " What   is NLP? "
    assert record["transformed_query"] == "what is nlp"
    assert summary["n_changed"] == 1
    assert "query transformations: 1 queries, 1 changed" in capsys.readouterr().out

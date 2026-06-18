from __future__ import annotations

import json

from msmarco_genqa.cli import query_transform_ablation as cli
from msmarco_genqa.evaluation.query_transform_ablation import (
    build_query_transform_ablation,
    extract_numeric_metrics,
    read_method_mapping,
    render_query_transform_ablation_markdown,
)


def _summary(method: str, *, changed: int) -> dict[str, object]:
    return {
        "method": method,
        "config_hash": f"{method}-hash",
        "n_queries": 10,
        "n_changed": changed,
        "changed_fraction": changed / 10,
        "cache_hit": False,
    }


def test_build_query_transform_ablation_computes_metric_deltas():
    report = build_query_transform_ablation(
        {
            "none": _summary("none", changed=0),
            "normalize": _summary("normalize", changed=2),
        },
        metrics={
            "none": {"mrr@10": 0.20, "recall@100": 0.70},
            "normalize": {"mrr@10": 0.25, "recall@100": 0.69},
        },
    )

    assert report["methods"] == ["none", "normalize"]
    assert report["runs"][1]["changed_fraction"] == 0.2
    assert {
        (row["method"], row["metric"], round(row["delta"], 4))
        for row in report["metric_deltas_vs_baseline"]
    } == {
        ("normalize", "mrr@10", 0.05),
        ("normalize", "recall@100", -0.01),
    }


def test_render_query_transform_ablation_markdown_includes_method_table():
    report = build_query_transform_ablation(
        {
            "none": _summary("none", changed=0),
            "lexical_expansion": _summary("lexical_expansion", changed=3),
        }
    )

    rendered = render_query_transform_ablation_markdown(report)

    assert "# Query transformation ablation report" in rendered
    assert "| `lexical_expansion` | `lexical_expansion-hash` | 10 | 3 | 30.00 |" in rendered


def test_extract_numeric_metrics_flattens_nested_groups():
    metrics = extract_numeric_metrics(
        {"metrics": {"dense": {"mrr@10": 0.4}, "wall_clock_seconds": {"run": 12.0}}}
    )

    assert metrics == {"dense.mrr@10": 0.4, "wall_clock_seconds.run": 12.0}


def test_read_method_mapping_rejects_duplicate_methods():
    try:
        read_method_mapping(["none=a.json", "none=b.json"], value_name="summary")
    except ValueError as exc:
        assert "duplicate summary method" in str(exc)
    else:
        raise AssertionError("expected duplicate method to fail")


def test_cli_writes_ablation_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    none_dir = tmp_path / "outputs" / "query_transform" / "none"
    norm_dir = tmp_path / "outputs" / "query_transform" / "normalize"
    none_dir.mkdir(parents=True)
    norm_dir.mkdir(parents=True)
    (none_dir / "summary.json").write_text(
        json.dumps(_summary("none", changed=0)),
        encoding="utf-8",
    )
    (norm_dir / "summary.json").write_text(
        json.dumps(_summary("normalize", changed=4)),
        encoding="utf-8",
    )
    (none_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"mrr@10": 0.1}}),
        encoding="utf-8",
    )
    (norm_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"mrr@10": 0.2}}),
        encoding="utf-8",
    )

    cli.main(
        [
            "--summary",
            "none=outputs/query_transform/none/summary.json",
            "--summary",
            "normalize=outputs/query_transform/normalize/summary.json",
            "--metrics",
            "none=outputs/query_transform/none/metrics.json",
            "--metrics",
            "normalize=outputs/query_transform/normalize/metrics.json",
            "--output-dir",
            "outputs/query_transform/ablation",
        ]
    )

    output_dir = tmp_path / "outputs" / "query_transform" / "ablation"
    report = json.loads((output_dir / "ablation.json").read_text(encoding="utf-8"))
    assert report["metric_deltas_vs_baseline"][0]["delta"] == 0.1
    assert (output_dir / "report.md").exists()
    assert "query transformation ablation: 2 methods" in capsys.readouterr().out

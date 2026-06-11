"""``mgq-transform-queries`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import yaml

from msmarco_genqa.data.msmarco import load_msmarco_passage
from msmarco_genqa.retrieval.query_transform import (
    QueryTransformConfig,
    query_transform_config_hash,
    read_cached_transformations,
    transform_queries,
    write_transformation_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/baseline.yaml",
        help="Pipeline YAML containing the query_transform section.",
    )
    parser.add_argument(
        "--method",
        choices=["none", "normalize", "lexical_expansion", "decontextualize"],
        default=None,
        help="Override query_transform.method from the config.",
    )
    parser.add_argument(
        "--queries-tsv",
        type=Path,
        default=None,
        help="Optional two-column TSV of qid and query. Defaults to MS MARCO dev/small.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Defaults to query_transform.output_dir from the config.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Reuse a JSONL artifact when qids and config hash match.",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SystemExit("config must be a YAML mapping")
    return raw


def _load_queries(path: Path | None, cfg: Mapping[str, object]) -> dict[str, str]:
    if path is None:
        data_cfg = cfg.get("data", {})
        cache_dir = None
        if isinstance(data_cfg, Mapping):
            cache_dir_value = data_cfg.get("cache_dir")
            if cache_dir_value:
                cache_dir = PROJECT_ROOT / str(cache_dir_value)
        bundle = load_msmarco_passage(cache_dir=cache_dir, load_corpus=False)
        return bundle.queries

    queries: dict[str, str] = {}
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    with resolved.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise SystemExit(f"{resolved}:{line_number}: expected qid<TAB>query")
            qid, query = parts
            if not qid:
                raise SystemExit(f"{resolved}:{line_number}: empty query id")
            queries[qid] = query
    return queries


def _settings_from_config(
    cfg: Mapping[str, object],
    *,
    method_override: str | None,
) -> dict[str, object]:
    raw = cfg.get("query_transform", {})
    if raw is None:
        settings: dict[str, object] = {}
    elif isinstance(raw, Mapping):
        settings = dict(raw)
    else:
        raise SystemExit("query_transform must be a YAML mapping")
    if method_override is not None:
        settings["method"] = method_override
    return settings


def _output_dir_from_settings(
    settings: Mapping[str, object],
    override: Path | None,
) -> Path:
    if override is not None:
        return override if override.is_absolute() else PROJECT_ROOT / override
    configured = settings.get("output_dir", "outputs/query_transform")
    path = Path(str(configured))
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    settings = _settings_from_config(cfg, method_override=args.method)
    transform_config = QueryTransformConfig.from_mapping(settings)
    queries = _load_queries(args.queries_tsv, cfg)
    query_ids = list(queries)
    config_hash = query_transform_config_hash(transform_config)

    cache_hit = False
    records = None
    if args.cache_file is not None:
        cache_path = args.cache_file if args.cache_file.is_absolute() else PROJECT_ROOT / args.cache_file
        records = read_cached_transformations(
            cache_path,
            expected_config_hash=config_hash,
            expected_query_ids=query_ids,
        )
        cache_hit = records is not None
    if records is None:
        records = transform_queries(queries, transform_config)

    output_dir = _output_dir_from_settings(settings, args.output_dir)
    summary, _paths = write_transformation_artifacts(
        records,
        output_dir,
        transform_config,
        cache_hit=cache_hit,
    )
    print(
        "query transformations: "
        f"{summary['n_queries']} queries, {summary['n_changed']} changed, "
        f"method={summary['method']}, output={output_dir}"
    )


if __name__ == "__main__":
    main()

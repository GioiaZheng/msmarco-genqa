# Query Transformation Protocol

Query transformation is an optional pre-retrieval ablation layer. It keeps the
original MS MARCO query text alongside the transformed text so retrieval gains
can be audited without losing the baseline input.

## Methods

- `none`: preserve the original query exactly. This is the default for
  historical baseline reproducibility.
- `normalize`: collapse whitespace, lowercase text, and strip terminal
  punctuation.
- `lexical_expansion`: normalize the query, then append deterministic curated
  expansion terms for short queries such as `nyc` -> `new york city`.
- `decontextualize`: normalize elliptical follow-up queries and prepend a
  configured context string when one is available.

The first production ablation should compare `none`, `normalize`, and
`lexical_expansion` before changing retriever or reranker parameters.

## Artifacts

Run:

```bash
mgq-transform-queries \
  --config configs/baseline.yaml \
  --method lexical_expansion \
  --output-dir outputs/query_transform/lexical_expansion
```

The command writes:

- `queries.jsonl`: one record per query with `query_id`, `original_query`,
  `transformed_query`, `method`, `config_hash`, `changed`, and `added_terms`.
- `summary.json`: query count, changed count, changed fraction, method, config
  hash, and cache status.

The same transformation config is used by `mgq-retrieve` and `mgq-dense`.
When the method is not `none`, each runner writes its own
`query_transform/queries.jsonl` and `query_transform/summary.json` under the
run output directory.

## Ablation Report

After running each method and evaluating matched retrieval outputs, combine the
method summaries and optional metric files:

```bash
mgq-query-transform-ablation \
  --summary none=outputs/query_transform/none/summary.json \
  --summary normalize=outputs/query_transform/normalize/summary.json \
  --summary lexical_expansion=outputs/query_transform/lexical_expansion/summary.json \
  --metrics none=outputs/query_transform/none/metrics.json \
  --metrics normalize=outputs/query_transform/normalize/metrics.json \
  --metrics lexical_expansion=outputs/query_transform/lexical_expansion/metrics.json \
  --output-dir outputs/query_transform/ablation
```

The report writes:

- `ablation.json`: method-level changed-query coverage and metric deltas
  versus the baseline method.
- `report.md`: a compact table suitable for README, report, or issue updates.

## Reproducibility Notes

- Keep `query_transform.method: none` for canonical no-transform baselines.
- Record the `config_hash` when reporting metric deltas.
- Compare transformed and untransformed runs on matched query ids with
  `mgq-retrieval-report compare`.
- Do not tune expansion terms using answer labels from the evaluation split.

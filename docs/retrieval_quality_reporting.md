# Retrieval Quality Reporting

`mgq-retrieval-report` evaluates any TREC-format `run.tsv` against a qrels
file and writes reproducible JSON and Markdown artifacts. It is intended for
BM25, dense, hybrid RRF, and reranked runs, as long as each input follows the
standard six-column run format:

```text
qid Q0 doc_id rank score system
```

## Single-Run Metrics

Use `evaluate` when a run should be scored independently:

```bash
mgq-retrieval-report evaluate \
  --run outputs/W4_dense/run.tsv \
  --run-name dense \
  --output-dir outputs/retrieval_reports/dense
```

Artifacts:

- `metrics.json`: input run/qrels paths, metric values, metric settings,
  evaluated qid count, and skipped-qid coverage diagnostics.
- `report.md`: a compact Markdown summary for experiment notes.

By default the report computes `MRR@10`, `nDCG@10`, `Recall@100`, and
`Recall@1000`. Override cutoffs with `--ks-mrr`, `--ks-ndcg`, and
`--ks-recall`.

## Matched-Qid Comparison

Use `compare` when the claim is comparative. The command restricts both runs
to the same qid set with positive qrels before computing deltas, which avoids
misleading comparisons when two retrieval stages cover different queries.

```bash
mgq-retrieval-report compare \
  --baseline-run outputs/W4_dense/run.tsv \
  --candidate-run outputs/W4_hybrid_rrf/run.tsv \
  --baseline-name dense \
  --candidate-name rrf \
  --output-dir outputs/retrieval_reports/dense_vs_rrf
```

Artifacts:

- `comparison.json`: input run/qrels paths, matched metrics,
  candidate-minus-baseline deltas, coverage counts, and movement-bucket
  summary.
- `per_query.jsonl`: query-level promoted, demoted, new-hit, lost-hit,
  unchanged-hit, and unchanged-miss diagnostics.
- `report.md`: a compact Markdown summary.

The coverage block is part of the result, not a footnote. Check
`n_baseline_only_qids`, `n_candidate_only_qids`, and
`n_shared_without_positive_qrels` before interpreting metric deltas.

## Multi-Run Matrix

Use `matrix` when a research claim compares more than two systems, such as
BM25-on-sample, dense, RRF, and RRF followed by cross-encoder reranking. The
command restricts every row to qids shared by all input runs and positive
qrels before computing metrics:

```bash
mgq-retrieval-report matrix \
  --run bm25_sample=outputs/W4_dense/run_bm25_sample.tsv \
  --run dense=outputs/W4_dense/run.tsv \
  --run rrf=outputs/W4_hybrid_rrf/run.tsv \
  --run rrf_reranked=outputs/W5_hybrid_rrf_reranker/run.tsv \
  --baseline-name bm25_sample \
  --output-dir outputs/retrieval_reports/hybrid_matrix
```

Artifacts:

- `matrix.json`: run order, per-run metrics, candidate-minus-baseline deltas,
  best run per metric, shared-qid coverage, and movement diagnostics.
- `pairwise_deltas.jsonl`: one row per candidate vs the named baseline.
- `report.md`: Markdown tables for the metrics matrix, deltas, best run per
  metric, and coverage.

Prefer this command over hand-copying metrics from separate runs. Separate
`metrics.json` files may cover different query sets; the matrix report makes
the shared-qid restriction explicit and repeatable.

## Qrels Format

If `--qrels` is omitted, the command loads MS MARCO Passage dev/small qrels
through `ir_datasets`. Pass `--qrels` when evaluating a custom split,
sample-specific qrels file, or downloaded TREC qrels artifact.

The command accepts standard four-column TREC qrels:

```text
qid iter doc_id relevance
```

It also accepts compact three-column qrels:

```text
qid doc_id relevance
```

Only positive relevance labels count as relevant. Non-positive labels are
kept as empty-qrel coverage evidence so the report can distinguish missing
qrels from explicitly non-relevant qrels.

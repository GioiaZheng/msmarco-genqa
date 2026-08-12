# Cross-Dataset First-Stage Error Analysis

## Result

NFCorpus and SciFact fail in different first-stage regimes under the same
frozen BM25 -> cross-encoder setup. NFCorpus has the stronger candidate-set
ceiling; SciFact has a smaller residual coverage tail.

| Diagnostic | NFCorpus | SciFact |
|---|---:|---:|
| Queries | 323 | 300 |
| Positive qrels | 12,334 | 339 |
| BM25 Recall@100 | 0.2378 | 0.8759 |
| BM25 Recall@1000 | 0.4572 | 0.9606 |
| No relevant document in top 100 | 72 (22.3%) | 35 (11.7%) |
| Complete relevant coverage at top 100 | 19 (5.9%) | 259 (86.3%) |
| First relevant hit only at ranks 101-1000 | 24 (7.4%) | 24 (8.0%) |
| No relevant hit at depth 1000 | 48 (14.9%) | 11 (3.7%) |

## Cross-Dataset Deltas

| Delta | Value | Interpretation |
|---|---:|---|
| SciFact - NFCorpus Recall@100 | +0.6381 | SciFact has a much healthier fixed top-100 candidate set. |
| SciFact - NFCorpus complete top-100 coverage share | +80.5% | Complete relevant-document coverage is common on SciFact and rare on NFCorpus. |
| NFCorpus - SciFact no-hit-at-100 share | +10.6% | The reranker is more often given no relevant candidate on NFCorpus. |
| NFCorpus - SciFact no-hit-at-1000 share | +11.2% | The unrecovered lexical first-stage tail is larger on NFCorpus. |

## Failure Partition

| Partition | NFCorpus | SciFact | Meaning |
|---|---:|---:|---|
| Candidate-set absence at 100 | 72 | 35 | No judged relevant document is available to the fixed top-100 reranker. |
| Depth-recoverable absence | 24 | 24 | BM25 can find a relevant document, but only after the reranker cutoff. |
| Residual top-1000 miss | 48 | 11 | Deeper BM25 still does not retrieve a judged relevant document. |
| Partial top-100 coverage | 232 | 6 | At least one relevant document is reachable, but some positive qrels remain outside the candidate set. |
| Complete top-100 coverage | 19 | 259 | All judged positive qrels are already reachable by the reranker. |

## Qualitative Boundary

The complete NFCorpus review covers all 72 queries with no relevant document in
BM25 top 100. It labels 67/72 cases as `source_context_dependency`, which
supports a compact-query representation explanation for that dataset.

The bounded SciFact residual review covers all 35 queries with no relevant
document in BM25 top 100. It labels 28/35 as
`terminology_or_evidence_form_mismatch`, 4/35 as
`lexical_competition_at_depth_cutoff`, and 3/35 as `short_or_broad_claim`.
This does not reproduce the NFCorpus source-context pattern. The comparison
therefore supports a retrieval-reachability conclusion plus dataset-specific
failure descriptions, not a claim that both datasets share the same semantic
failure causes.

## Decision

Do not change the pipeline yet. The cross-dataset evidence separates three
effects:

- NFCorpus has a severe fixed-candidate limitation and a documented
  compact-query/source-context component.
- SciFact generalizes better at the first retrieval stage under the same frozen
  BM25 setup.
- The cross-encoder improves ranking when relevant documents are present, but
  cannot recover candidates missing from the first stage.

The next controlled change, if any, should be retrieval-side and predeclared:
candidate depth, query representation, hybrid retrieval, or a stronger
first-stage retriever. It should not be selected merely from the
already-inspected failure cases.

## Reproduction

```bash
make analyze-cross-dataset-errors
```

The target first verifies the BEIR release bundle and both dataset contracts,
then writes:

- `outputs/analysis/cross_dataset_errors/summary.json`
- `outputs/analysis/cross_dataset_errors/report.md`

The checked drift contract is
[`configs/cross_dataset_error_analysis.json`](../configs/cross_dataset_error_analysis.json).

Source diagnostics:

- [`docs/nfcorpus_first_stage_error_analysis.md`](nfcorpus_first_stage_error_analysis.md)
- [`docs/scifact_first_stage_error_analysis.md`](scifact_first_stage_error_analysis.md)
- [`docs/nfcorpus_first_stage_taxonomy_review.md`](nfcorpus_first_stage_taxonomy_review.md)
- [`docs/scifact_failure_review.md`](scifact_failure_review.md)

## Limitations

- This is retrieval-only evidence; no generation run is evaluated on NFCorpus
  or SciFact.
- The analysis uses public qrels as the relevance ground truth and does not
  infer relevance for unjudged documents.
- The comparison uses the frozen BM25 and cross-encoder outputs from the
  released BEIR bundle; it does not measure a new retriever.
- The SciFact review is a bounded residual no-hit@100 review, not a manual
  taxonomy over all 300 SciFact queries.
- NFCorpus and SciFact have very different qrels densities, so macro recall,
  query-count failures, and positive-qrel mass are reported separately.

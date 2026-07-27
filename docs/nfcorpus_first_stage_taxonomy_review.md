# NFCorpus First-Stage Failure Review

## Main finding

The 72 NFCorpus queries with no judged relevant document in the BM25 top 100
are dominated by missing source-page context rather than a single lexical
ranking defect. A complete review assigned `source_context_dependency` to
67/72 cases (93.1%), `vocabulary_or_form_mismatch` to 4/72 (5.6%), and
`underspecified_or_ambiguous_query` to 1/72 (1.4%).

This does not make the measured candidate-set limitation disappear: under the
published BEIR task definition, BM25 still misses every positive qrel in its
top 100 for these queries. It does change the engineering interpretation.
Most reviewed failures do not yet justify modifying the reranker or generator.
The next controlled comparison should separate topic pages from the official
non-topic query subset or restore richer query fields before an architectural
change is attributed to retrieval quality.

## Scope

This review uses the fixed NFCorpus BM25 and cross-encoder outputs published in
`v2.2-beir-cross-domain-baselines`. It does not rerun retrieval, change a
model, or alter the architecture.

The full census contains:

- 24 queries whose first positive qrel appears at ranks 101–1000;
- 48 queries with no positive qrel in the BM25 top 1000;
- all 72 queries with no positive qrel in the BM25 top 100.

Binary retrieval metrics use the frozen `relevance >= 1` rule. The input
archive, qrels, run sizes, hashes, and headline metrics are pinned in the
[data and metric contract](../configs/nfcorpus_first_stage_contract.json).
The preceding quantitative analysis is in the
[first-stage coverage report](nfcorpus_first_stage_error_analysis.md).

## Evidence

### Manual review result

| Primary label | Ranks 101–1000 | Miss top 1000 | Total | Share |
|---|---:|---:|---:|---:|
| `source_context_dependency` | 22 | 45 | 67 | 93.1% |
| `vocabulary_or_form_mismatch` | 2 | 2 | 4 | 5.6% |
| `underspecified_or_ambiguous_query` | 0 | 1 | 1 | 1.4% |
| `lexical_competition` | 0 | 0 | 0 | 0.0% |
| `qrels_or_scope_gap` | 0 | 0 | 0 | 0.0% |
| `other_unclear` | 0 | 0 | 0 | 0.0% |

Review coverage is 72/72 with no pending or adjudication rows. The compact
case-level decisions are stored in
[`reports/annotations/nfcorpus_first_stage_review_v1.csv`](../reports/annotations/nfcorpus_first_stage_review_v1.csv).
Generated document snippets remain under ignored `outputs/` and are not
committed.

### Qrel linkage evidence

NFCorpus judgments are derived from links among NutritionFacts pages and
medical articles rather than independent topical judgments. The
[dataset authors](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
describe three strengths: direct citation links, indirect links through
another NutritionFacts page, and topic/tag connections. The
[ir_datasets representation](https://ir-datasets.com/nfcorpus.html)
documents the BEIR-style positive levels used here as direct and indirect
links.

Within the 72 reviewed cases:

| Published qrel evidence | Cases | Share |
|---|---:|---:|
| only relevance level 1 | 62 | 86.1% |
| at least one relevance level 2 qrel | 10 | 13.9% |

The prevalence of level-1-only evidence is consistent with the manual finding:
for many queries, relevance is visible in the source-page link graph but not
recoverable from the exported title alone. This is evidence about benchmark
construction, not evidence that the qrels are invalid.

### Coverage by query source

The source URL in the pinned query archive provides an objective page type.
The following table uses all 323 test queries, not only the failure census.

| Source type | Queries | No positive top 100 | Rate | First hit 101–1000 | Miss top 1000 | Macro R@100 | Macro R@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dated article | 32 | 0 | 0.0% | 0 | 0 | 0.1950 | 0.4215 |
| question | 10 | 3 | 30.0% | 0 | 3 | 0.4858 | 0.5975 |
| topic | 179 | 58 | 32.4% | 19 | 39 | 0.2063 | 0.4132 |
| video | 102 | 11 | 10.8% | 5 | 6 | 0.2821 | 0.5320 |

Topic pages are 179/323 (55.4%) of the test set but 58/72 (80.6%) of the
no-hit-at-100 census. This concentration is the strongest objective signal in
the analysis.

### Fixed-run metrics by source

| Source | System | Queries | MRR@10 | nDCG@10 | Recall@100 |
|---|---|---:|---:|---:|---:|
| dated article | BM25 | 32 | 0.6123 | 0.2703 | 0.1950 |
| dated article | BM25 + CE | 32 | 0.7706 | 0.3793 | 0.1950 |
| question | BM25 | 10 | 0.4700 | 0.3542 | 0.4858 |
| question | BM25 + CE | 10 | 0.6500 | 0.3978 | 0.4858 |
| topic | BM25 | 179 | 0.5277 | 0.3428 | 0.2063 |
| topic | BM25 + CE | 179 | 0.5530 | 0.3562 | 0.2063 |
| video | BM25 | 102 | 0.4780 | 0.2492 | 0.2821 |
| video | BM25 + CE | 102 | 0.5171 | 0.2972 | 0.2821 |
| non-topic combined | BM25 | 144 | 0.5073 | 0.2612 | 0.2769 |
| non-topic combined | BM25 + CE | 144 | 0.5827 | 0.3224 | 0.2769 |

The cross-encoder improves MRR@10 and nDCG@10 in every source group while
Recall@100 remains fixed by design. On the 144 non-topic queries, MRR@10 rises
from 0.5073 to 0.5827 and nDCG@10 from 0.2612 to 0.3224. The topic subset has
the smallest MRR gain and the highest absolute number of no-hit-at-100 cases.
This is consistent with an effective reranker operating under a source-title
candidate bottleneck.

## Method

1. Re-verify the frozen input hashes, 323-query scope, qid/docid joins, run
   depths, and headline metrics.
2. Select the complete pre-declared census of 24 depth-recoverable and 48
   top-1000-miss queries. No sampling replacement is used.
3. For each case, record the query source URL and type, qrel level counts,
   positive-document snippets, BM25 top-document snippets, first positive
   rank, and exact query-token overlap.
4. Assign one primary label and an optional secondary label using the frozen
   [retrieval taxonomy](../configs/nfcorpus_retrieval_review_taxonomy.json).
5. Validate that every frozen qid is present exactly once and that reviewed
   rows contain an admissible label and evidence note.
6. Recompute source-level BM25 and cross-encoder metrics from the fixed ranked
   runs.

The review order is deterministic from the configured seed. Exact tables are
used instead of charts because this is a 72-case full census with small
categorical counts; a chart would not add resolution beyond the stated
denominators.

Reproduce the evidence export and summary with:

```bash
python scripts/export_nfcorpus_first_stage_review.py \
  --annotations reports/annotations/nfcorpus_first_stage_review_v1.csv
```

## Limitations

- This is a single-reviewer descriptive taxonomy. It has not yet received an
  independent second annotation or inter-annotator agreement estimate.
- The BEIR archive exports a compact query text. The original NFCorpus
  interfaces also expose combined query fields and dedicated non-topic/video
  subsets; this review does not reconstruct those richer fields.
- The taxonomy covers only the 72 queries with zero Recall@100. It does not
  estimate error-category prevalence across all 323 queries.
- Exact token overlap is a review aid and not a semantic similarity metric.
- The labels explain observable evidence; they do not establish causal model
  effects.
- The 10-query question group is too small for strong subgroup claims.

## Decision and next experiment

The current evidence supports keeping the architecture unchanged. Before
testing a new first-stage retriever, the lowest-risk follow-up is a controlled
query-representation comparison:

1. preserve the fixed models and evaluation contract;
2. evaluate the official non-topic subset separately as a stable reference;
3. if richer NFCorpus title-plus-context fields can be recovered
   reproducibly, compare them with the BEIR title-only representation;
4. only then test query expansion or hybrid retrieval on the cases that remain
   unresolved.

This sequence distinguishes benchmark representation effects from retriever
capacity and avoids selecting an architecture to compensate for omitted query
context.

## Further questions

- Does the reranking gain remain stable when topic pages are excluded?
- How much first-stage coverage is recovered by richer query fields without
  changing BM25?
- Among failures that remain after query enrichment, does hybrid retrieval
  outperform normalization or lightweight expansion?
- Would a second reviewer reproduce the distinction between
  `source_context_dependency` and `vocabulary_or_form_mismatch`?

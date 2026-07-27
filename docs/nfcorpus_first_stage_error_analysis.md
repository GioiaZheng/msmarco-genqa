# NFCorpus First-Stage Coverage: Query-Level Diagnostic

## Technical summary

The fixed NFCorpus BM25 top-100 candidate set contains at least one
qrels-relevant document for **251 of 323 queries (77.7%)**. The remaining
**72 queries (22.3%)** give the cross-encoder no relevant candidate to rank.
Under the current qrels, this is a hard candidate-set limit: reranker changes
cannot recover a judged relevant document for those queries without changing
first-stage membership.

The limitation is broader than the 72 complete misses. Among all queries,
**232 (71.8%)** have only partial relevant-document coverage at depth 100, and
only **19 (5.9%)** have complete coverage. Extending the same BM25 run from
depth 100 to 1000 raises macro Recall from **0.2378 to 0.4572**, finding 3,616
additional positive qrels across 200 queries. Even at depth 1000, 48 queries
have no relevant hit and 6,753 of 12,334 positive qrels remain unretrieved.

These are descriptive results for one frozen BM25 run. They establish where
the current candidate-set ceiling occurs; they do not yet identify the
semantic cause of each miss or evaluate a replacement retriever.

## The failure is both query-level absence and incomplete coverage

The first relevant hit separates queries by whether a fixed top-100 reranker
can reach any relevant document:

| First relevant BM25 hit | Queries | Share | Interpretation |
|---|---:|---:|---|
| Rank 1-10 | 218 | 67.5% | A relevant document is already inside the headline rank window. |
| Rank 11-100 | 33 | 10.2% | A relevant document is reachable by the fixed top-100 reranker. |
| Rank 101-1000 | 24 | 7.4% | BM25 finds a relevant document, but outside the reranker candidate set. |
| No hit in top 1000 | 48 | 14.9% | Deeper BM25 retrieval still does not find a relevant document. |

The relevant-document coverage view shows why a binary “any hit” diagnostic is
not sufficient for NFCorpus, where one query may have many positive qrels:

| Recall@100 bucket | Queries | Share |
|---|---:|---:|
| 0 | 72 | 22.3% |
| Greater than 0 and below 0.25 | 145 | 44.9% |
| 0.25 to below 0.50 | 51 | 15.8% |
| 0.50 to below 1 | 36 | 11.1% |
| 1 | 19 | 5.9% |

The two partitions reconcile exactly to all 323 queries. In particular, the
24 first hits at ranks 101-1000 plus the 48 top-1000 misses equal the 72
queries with Recall@100 equal to zero.

## Going deeper helps, but does not remove the bottleneck

| Diagnostic | Depth 100 | Depth 1000 | Change |
|---|---:|---:|---:|
| Macro Recall | 0.2378 | 0.4572 | +0.2195 |
| Micro positive-qrel coverage | 0.1593 | 0.4525 | +0.2932 |
| Positive qrels retrieved | 1,965 | 5,581 | +3,616 |

Macro Recall gives every query equal weight and remains the benchmark
headline definition. Micro coverage weights queries by their number of
positive qrels and is reported only as a diagnostic decomposition. The
difference between the two is expected because NFCorpus has between 1 and 475
positive qrels per query (median 16, mean 38.19).

Ranks 101-1000 add at least one relevant document for 200 queries. This shows
that a deeper candidate pool contains useful evidence for many queries, but
it is not itself an end-to-end result: reranking 1,000 candidates would change
runtime and would require a separate controlled experiment.

## Scope and metric definitions

- Dataset: `beir/nfcorpus/test`.
- Population: all 323 test queries with positive qrels.
- Relevance rule: `rel >= 1`.
- First-stage run: the immutable BM25 depth-1000 output published in
  `v2.2-beir-cross-domain-baselines`.
- Reranker reachability cutoff: BM25 top 100.
- First-hit buckets: ranks 1-10, 11-100, 101-1000, or no hit at depth 1000.
- Coverage buckets: Recall@100 equal to 0; partial intervals
  `(0, 0.25)`, `[0.25, 0.50)`, `[0.50, 1)`; or complete coverage at 1.

The underlying data and metric contract fixes the input byte sizes and
SHA-256 digests, requires exact qid/docid correspondence, and rejects missing
queries, duplicate documents, rank gaps, and non-finite scores:
[`configs/nfcorpus_first_stage_contract.json`](../configs/nfcorpus_first_stage_contract.json).

## Methodology and reproduction

The analysis reads the already-published BM25 run, public NFCorpus query
records, and qrels. It does not rebuild an index, rerun BM25, invoke the
cross-encoder, or modify any model.

From a configured clone:

```bash
make analyze-nfcorpus-first-stage
```

The target first downloads and verifies the public BEIR release into
`outputs/reproductions/beir_cross_domain_v1`, recovers the original NFCorpus
files into the repository-local `outputs/reproductions/beir_irds_cache`, and
recomputes the released metrics. It then writes:

- `outputs/analysis/nfcorpus_first_stage/summary.json`;
- `outputs/analysis/nfcorpus_first_stage/per_query.jsonl`;
- `outputs/analysis/nfcorpus_first_stage/examples.jsonl`;
- `outputs/analysis/nfcorpus_first_stage/report.md`.

The query-level file records the positive-qrel count, first relevant rank,
hit counts and recall at 10/100/1000, relevant hits inside and outside the
top-100 candidate set, and relevant document IDs still missing after depth
1000. Diagnostic examples are selected deterministically by SHA-256 within
each bucket rather than chosen manually.

Tables are used instead of charts because both diagnostic dimensions are
small, mutually exclusive partitions and exact reconciliation is the primary
audit requirement.

## Limitations and robustness checks

- The reported Recall@100 and Recall@1000 values independently reconcile to
  the frozen headline metrics with tolerance `1e-12`.
- Macro and micro results are labeled separately and are not compared as if
  they used the same denominator.
- The analysis uses qrels as the available relevance ground truth. Unjudged
  documents are not treated as newly relevant, and qrels incompleteness may
  affect qualitative interpretation.
- A first hit outside rank 100 shows candidate-depth exclusion, not the reason
  BM25 ranked it late.
- A top-1000 miss does not by itself prove vocabulary mismatch, domain shift,
  or a need for dense retrieval.
- No generation claim follows from this retrieval-only analysis.

## Recommended next step

Keep the architecture unchanged while reviewing two pre-declared cohorts:

1. The 24 queries whose first relevant hit appears at ranks 101-1000. These
   isolate ranking-depth failures where BM25 can retrieve relevant evidence.
2. The 48 queries with no relevant hit at depth 1000. These isolate the harder
   first-stage failures for lexical-overlap and terminology analysis.

For each cohort, compare query terms with the judged relevant documents and
record a small failure taxonomy before selecting a new retrieval baseline.
That evidence can determine whether the next controlled test should vary
candidate depth, query processing, or first-stage retrieval method.

## Further questions

- Are top-100 misses concentrated in queries with many positive qrels, short
  keyword queries, or specialized medical terminology?
- Within the 24 depth-only failures, how many relevant documents enter early
  enough that a larger reranking pool would be computationally credible?
- Do relevance grades or qrels density change the apparent failure mix?
- Would a separately versioned dense or hybrid first stage improve coverage
  without erasing the current BM25 control?

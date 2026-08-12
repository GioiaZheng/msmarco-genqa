# SciFact First-Stage Coverage: Query-Level Diagnostic

## Technical summary

The fixed SciFact BM25 top-100 candidate set contains at least one
qrels-relevant document for **265 of 300 queries (88.3%)**. The remaining
**35 queries (11.7%)** give the cross-encoder no relevant candidate to rank.

This is a real candidate-set ceiling, but it is much smaller than the
NFCorpus bottleneck. SciFact BM25 Recall@100 is **0.8759**, compared with
**0.2378** on NFCorpus. Extending the same SciFact BM25 run from depth 100 to
1000 raises macro Recall from **0.8759** to **0.9606**, finds 28 additional
positive qrels across 28 queries, and leaves 11 queries with no relevant hit
at depth 1000.

The analysis supports a narrow conclusion: the unchanged pipeline has some
SciFact first-stage misses, but the main cross-domain candidate-set problem
observed on NFCorpus does not repeat at the same scale on SciFact.

## First relevant hit

The first relevant hit separates queries by whether the fixed top-100 reranker
can reach any relevant document:

| First relevant BM25 hit | Queries | Share | Interpretation |
|---|---:|---:|---|
| Rank 1-10 | 239 | 79.7% | A relevant document is already inside the headline rank window. |
| Rank 11-100 | 26 | 8.7% | A relevant document is reachable by the fixed top-100 reranker. |
| Rank 101-1000 | 24 | 8.0% | BM25 finds a relevant document, but outside the reranker candidate set. |
| No hit in top 1000 | 11 | 3.7% | Deeper BM25 retrieval still does not find a relevant document. |

The relevant-document coverage view is also less severe than NFCorpus:

| Recall@100 bucket | Queries | Share |
|---|---:|---:|
| 0 | 35 | 11.7% |
| Greater than 0 and below 0.25 | 0 | 0.0% |
| 0.25 to below 0.50 | 0 | 0.0% |
| 0.50 to below 1 | 6 | 2.0% |
| 1 | 259 | 86.3% |

Most SciFact test queries have exactly one positive qrel. The positive-qrel
distribution is therefore very different from NFCorpus: SciFact has median 1,
mean 1.13, and maximum 5 positive qrels per query, while NFCorpus has many
queries with much denser relevance sets. This makes complete coverage at
depth 100 easier to interpret on SciFact.

## Depth 100 to 1000

| Diagnostic | Depth 100 | Depth 1000 | Change |
|---|---:|---:|---:|
| Macro Recall | 0.8759 | 0.9606 | +0.0847 |
| Micro positive-qrel coverage | 0.8791 | 0.9617 | +0.0826 |
| Positive qrels retrieved | 298 | 326 | +28 |

Ranks 101-1000 add at least one relevant document for 28 queries, including
24 queries whose first relevant hit appears only after rank 100. This suggests
that a larger reranking pool could recover some additional SciFact evidence,
but it would be a new latency and cost condition, not a free improvement.

## Cross-dataset comparison

| Diagnostic | NFCorpus | SciFact |
|---|---:|---:|
| Test queries | 323 | 300 |
| Positive qrels | 12,334 | 339 |
| BM25 Recall@100 | 0.2378 | 0.8759 |
| BM25 Recall@1000 | 0.4572 | 0.9606 |
| Queries with no relevant document in top 100 | 72 | 35 |
| Queries with complete relevant coverage at 100 | 19 | 259 |
| Queries still missing at depth 1000 | 48 | 11 |
| Positive qrels still missing at depth 1000 | 6,753 | 13 |

NFCorpus remains primarily candidate-set and representation limited under the
fixed BM25 top-100 setting. SciFact shows a smaller residual first-stage
issue: most queries already have complete relevant-document coverage at depth
100, and the reranker is usually operating on a candidate set that contains a
judged relevant document.

## Scope and metric definitions

- Dataset: `beir/scifact/test`.
- Population: all 300 test queries with positive qrels.
- Relevance rule: `rel >= 1`.
- First-stage run: the immutable BM25 depth-1000 output published in
  `v2.2-beir-cross-domain-baselines`.
- Reranker reachability cutoff: BM25 top 100.
- First-hit buckets: ranks 1-10, 11-100, 101-1000, or no hit at depth 1000.
- Coverage buckets: Recall@100 equal to 0; partial intervals
  `(0, 0.25)`, `[0.25, 0.50)`, `[0.50, 1)`; or complete coverage at 1.

The data and metric contract fixes the input byte sizes and SHA-256 digests,
requires exact qid/docid correspondence, and rejects missing queries,
duplicate documents, rank gaps, and non-finite scores:
[`configs/scifact_first_stage_contract.json`](../configs/scifact_first_stage_contract.json).

## Methodology and reproduction

The analysis reads the already-published BM25 run, public SciFact query
records, and qrels. It does not rebuild an index, rerun BM25, invoke the
cross-encoder, or modify any model.

From a configured clone:

```bash
make analyze-scifact-first-stage
```

The target downloads and verifies the public BEIR release into
`outputs/reproductions/beir_cross_domain_v1`, recovers the original SciFact
files into the repository-local `outputs/reproductions/beir_irds_cache`, and
recomputes the released metrics. It then writes:

- `outputs/analysis/scifact_first_stage/summary.json`;
- `outputs/analysis/scifact_first_stage/per_query.jsonl`;
- `outputs/analysis/scifact_first_stage/examples.jsonl`;
- `outputs/analysis/scifact_first_stage/report.md`.

The query-level file records the positive-qrel count, first relevant rank,
hit counts and recall at 10/100/1000, relevant hits inside and outside the
top-100 candidate set, and relevant document IDs still missing after depth
1000. Diagnostic examples are selected deterministically by SHA-256 within
each bucket rather than chosen manually.

## Limitations and robustness checks

- The reported Recall@100 and Recall@1000 values independently reconcile to
  the frozen headline metrics with tolerance `1e-12`.
- Macro and micro results are labeled separately and are not compared as if
  they used the same denominator.
- The analysis uses qrels as the available relevance ground truth. Unjudged
  documents are not treated as newly relevant.
- A first hit outside rank 100 shows candidate-depth exclusion, not the reason
  BM25 ranked it late.
- A top-1000 miss does not by itself prove vocabulary mismatch, domain shift,
  or a need for dense retrieval.
- No generation claim follows from this retrieval-only analysis.

## Follow-up review

The residual 35-query review is now recorded in
[`docs/scifact_failure_review.md`](scifact_failure_review.md). It covers the
24 queries whose first relevant hit appears at ranks 101-1000 and the 11
queries with no relevant hit at depth 1000.

The review does not reproduce the NFCorpus source-context failure pattern.
Most residual SciFact misses are better described as scientific claim/evidence
formulation mismatches under exact lexical first-stage retrieval. The current
decision is therefore to keep the retrieval pipeline frozen for the report and
only consider a predeclared retrieval-side intervention, such as candidate
depth sensitivity, scientific-claim query rewriting, or hybrid lexical/dense
retrieval, if the next experiment explicitly targets this failure mode.

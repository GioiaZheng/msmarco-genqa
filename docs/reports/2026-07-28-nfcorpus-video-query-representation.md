# NFCorpus Video Query Representation

## Question

Can bounded source-page context repair part of the NFCorpus first-stage
candidate limitation without changing the corpus, qrels, BM25 implementation,
cross-encoder, or architecture?

## Setup

The controlled comparison uses the official 102-query NFCorpus test/video
subset and three predeclared representations:

1. the existing BEIR title;
2. the official video description;
3. title plus description.

All conditions use the same 3,633-document BEIR NFCorpus corpus, `rel >= 1`
qrels, BM25 parameters, index fingerprint, and deterministic score-descending,
document-id-ascending tie rule. BM25 retrieves to depth 1,000. The fixed
`cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8` reranks the corresponding top 100
without adding candidates.

The query cohort, official source archive, field construction, leakage
boundary, and stopping rules were frozen before the treatment runs in
[`docs/nfcorpus_video_query_representation_protocol.md`](../nfcorpus_video_query_representation_protocol.md).

## Result

### BM25

| Representation | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|---:|
| Title | 0.478019 | 0.249252 | 0.282065 | 0.491892 |
| Description | 0.530781 | 0.292925 | 0.327210 | 0.641349 |
| Title + description | **0.603568** | **0.345704** | **0.370030** | **0.672296** |

### BM25 plus cross-encoder

| Representation | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|
| Title | 0.522012 | 0.298003 | 0.282065 |
| Description | 0.635699 | 0.356809 | 0.327210 |
| Title + description | **0.668857** | **0.385310** | **0.370030** |

Recall@100 is identical before and after reranking within each representation
because the cross-encoder only reorders that representation's BM25 top 100.

## Paired evidence

The predeclared primary metric is Recall@100. Confidence intervals and
two-sided p-values use 10,000 paired-bootstrap resamples over the same 102
queries with seed `20260727`.

| Treatment vs title | Metric | Mean delta | 95% CI | p-value |
|---|---|---:|---:|---:|
| Description | Recall@100 | +0.045145 | [-0.002419, +0.094286] | 0.0610 |
| Title + description | Recall@100 | **+0.087965** | **[+0.053494, +0.126537]** | **< 0.0002** |
| Title + description | BM25 MRR@10 | +0.125549 | [+0.054279, +0.200058] | 0.0002 |
| Title + description | BM25 nDCG@10 | +0.096452 | [+0.060503, +0.136063] | < 0.0002 |
| Title + description | CE MRR@10 | +0.146845 | [+0.068063, +0.224665] | < 0.0002 |
| Title + description | CE nDCG@10 | +0.087307 | [+0.049858, +0.124981] | < 0.0002 |

Description alone has a positive point estimate, but its primary Recall@100
interval crosses zero. It should not be described as a robust candidate-recall
improvement on this cohort.

Title plus description also improves Recall@100 over description alone by
`+0.042820`, with a 95% interval of `[+0.020324, +0.070412]` and
`p < 0.0002`. After reranking, however, its MRR@10 and nDCG@10 differences
from description alone are not conclusive: their intervals cross zero
(`p = 0.2338` and `p = 0.0732`, respectively).

## Candidate-coverage diagnostic

| Representation | No hit at 100 | No hit at 1,000 |
|---|---:|---:|
| Title | 11 | 8 |
| Description | 5 | 0 |
| Title + description | 4 | 0 |

Relative to title, title plus description recovers 10 of the 11 no-hit-at-100
queries but loses coverage for 3 previously covered queries, giving a net
reduction from 11 to 4. At depth 1,000 it recovers all 8 title misses and
introduces none.

These counts are query-level hit diagnostics. They do not imply complete
coverage of all relevant documents; even the best Recall@100 remains only
`0.370030`.

## Validation

- The three BM25 runs contain the same 102 qids at depth 1,000 and were
  produced from clean commit `119a9c695547`.
- All runs share BM25 index SHA-256
  `bf3fbede5fb624bd3c4def66de8d690bdfcda1f3f43c47637a5328cc8459730f`.
- The three reranked runs contain 102 qids at depth 100 and were produced from
  clean commit `cc6ae1f9ac08`.
- All 306 query-condition candidate-set checks confirm that each reranked run
  contains exactly its corresponding BM25 top-100 documents.
- Run parsing rejects duplicate documents, rank gaps, non-finite scores, and
  malformed records.
- Independent `ir-measures` evaluation reproduces every aggregate metric
  within the `1e-12` acceptance tolerance.

Run the complete contract, metric, candidate-set, and paired-bootstrap check
with:

```bash
python -X utf8 scripts/analyze_nfcorpus_query_representations.py \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260727 \
  --tolerance 1e-12
```

## Interpretation

The result supports a narrow conclusion: on the official NFCorpus video
subset, the compact BEIR title omits useful source context, and combining that
title with the bounded official description materially improves first-stage
candidate coverage under the fixed BM25 system.

This does not establish that BM25 was replaced, that the architecture
improved, or that the result transfers to the other 221 NFCorpus test queries.
The description is source-page context connected to how NFCorpus was built,
not an independently authored user query. The experiment is retrieval-only
and provides no evidence about answer generation or grounding.

## Follow-up

The architecture gate remains closed. The best bounded representation raises
Recall@100, but the remaining value of `0.370030` still leaves a material
candidate ceiling. A later hybrid or dense retrieval comparison is justified
only as a separately predeclared experiment; it should not be folded into
this result.

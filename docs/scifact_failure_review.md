# SciFact Residual First-Stage Failure Review

## Main Finding

The 35 SciFact queries with no judged relevant document in BM25 top 100 do not
reproduce the NFCorpus source-context failure pattern. They are mostly
claim/evidence formulation misses: judged positive abstracts often share few
exact content tokens with the claim, while higher-ranked BM25 documents match
the claim wording more directly.

## Scope

- Dataset: `beir/scifact/test`.
- Review population: all 35 queries with BM25 Recall@100 equal to 0.
- Cohorts: 24 depth-recoverable cases and 11 top-1000 misses.
- Evidence: frozen BM25 depth-1000 run, public qrels, SciFact claims, and
  corpus abstracts.
- Boundary: retrieval-only; no retrieval, reranking, or generation is rerun.

## Residual Failure Categories

| Primary label | Cases | Share | Interpretation |
|---|---:|---:|---|
| `terminology_or_evidence_form_mismatch` | 28 | 80.0% | The judged evidence uses a substantially different surface formulation from the claim, so exact lexical matching gives it weak BM25 support. |
| `lexical_competition_at_depth_cutoff` | 4 | 11.4% | The judged evidence has moderate surface overlap, but higher-ranked documents match the claim text more strongly and push it beyond the top-100 reranker cutoff. |
| `short_or_broad_claim` | 3 | 8.6% | The claim is short or broad enough that exact terms retrieve many plausible non-relevant documents before the judged evidence. |
| `other_unclear` | 0 | 0.0% | The available query, qrels, and BM25 evidence do not support a more specific descriptive label. |

## Cohorts

| Cohort | Cases | Interpretation |
|---|---:|---|
| `depth_recoverable_101_1000` | 24 | BM25 eventually retrieves a judged positive document, but below the fixed top-100 reranker cutoff. |
| `miss_top_1000` | 11 | The judged positive document is absent even after extending BM25 to depth 1000. |

## Cross-Cutting Signals

| Signal | Cases | Interpretation |
|---|---:|---|
| `top_lexical_competition` | 35 | The strongest BM25 candidates match more claim tokens than the judged positive evidence. |
| `low_positive_surface_overlap` | 31 | The judged positive evidence covers at most 25% of claim content tokens. |
| `polarity_or_directional_claim` | 13 | The claim contains a negation, direction, activation, inhibition, or comparative cue. |
| `shared_positive_evidence_doc` | 10 | Multiple residual claims point to the same judged evidence document, often with paired or directional wording. |

Shared judged-positive evidence groups:

- `11335781`: `1278`, `1279`
- `25649714`: `1196`, `1197`
- `3203590`: `913`, `914`
- `5304891`: `1332`, `975`
- `8646760`: `820`, `821`

## Interpretation

The answer to the residual SciFact question is narrower than a new retrieval
architecture. The cross-encoder can only rerank the fixed top-100 candidate
set, and these 35 cases show that the first-stage candidate boundary still
matters. However, SciFact does not show the same broad candidate-set collapse
as NFCorpus: 259/300 SciFact queries already have complete relevant-document
coverage at top 100.

The strongest observed SciFact pattern is lexical competition under scientific
claim formulation. Many claims are written as compact statements, while the
judged evidence appears in abstracts with different wording, background
framing, or polarity/directional structure. Increasing candidate depth could
recover the 24 depth-recoverable cases, but it would change latency and
reranking cost. The 11 top-1000 misses are better treated as
terminology/formulation or scope limitations before selecting a new retriever.

## Decision

Keep the pipeline frozen for the current report. The next intervention should
be predeclared and retrieval-side only if it targets this failure mode directly,
such as candidate-depth sensitivity, query rewriting for scientific claims, or
hybrid lexical/dense retrieval. The current evidence does not justify changing
the reranker or generator.

## Reproduction

```bash
make review-scifact-first-stage
```

The target writes:

- `outputs/analysis/scifact_first_stage/review/review_cases.jsonl`
- `outputs/analysis/scifact_first_stage/review/review_summary.json`
- `outputs/analysis/scifact_first_stage/review/review.md`

The checked drift contract is
[`configs/scifact_failure_review.json`](../configs/scifact_failure_review.json).

## Limitations

- The labels are a bounded descriptive review aided by exact token-overlap
  features; they are not causal ground truth.
- The review uses public qrels only and does not infer relevance for unjudged
  documents.
- The review covers residual no-hit@100 cases, not all 300 SciFact test
  queries.
- It does not evaluate generated answers or groundedness on SciFact.

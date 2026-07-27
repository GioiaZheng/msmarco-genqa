# Failure Taxonomy

This taxonomy is for paired BM25-vs-reranked generation analysis. It keeps
error review grounded in query-level evidence instead of broad impressions.

Use it when a reranked output is worse than the BM25 output, when grounding
metrics disagree, or when a new ablation changes a headline metric enough to
deserve inspection.

## NFCorpus First-Stage Retrieval Labels

The NFCorpus first-stage review is a retrieval-only analysis and does not use
the generation labels below. Its unit of review is one query from either the
`depth_recoverable_101_1000` or `miss_top_1000` cohort. The machine-readable
definitions and frozen cohort sizes live in
[`configs/nfcorpus_retrieval_review_taxonomy.json`](../configs/nfcorpus_retrieval_review_taxonomy.json).

| Label | Review decision | Experiment implication |
|---|---|---|
| `lexical_competition` | Relevant text contains the query terms but receives a weaker lexical score than competing documents. | Test first-stage discrimination or hybrid retrieval. |
| `vocabulary_or_form_mismatch` | Query and relevant text express the same concept through different forms or terminology. | Test normalization, expansion, dense retrieval, or hybrid retrieval. |
| `underspecified_or_ambiguous_query` | The exported query text alone cannot distinguish the qrels target from other plausible collection topics. | Test clarification or richer query fields. |
| `source_context_dependency` | The qrel is supported by the source page or its link graph, but the exported page title omits the context needed to infer that relation. | Evaluate richer NFCorpus query fields or the official source-type subsets before changing the model. |
| `qrels_or_scope_gap` | Query and judged document have a material scope mismatch not explained by wording or known source-page context. | Audit judgments or segmentation before treating the case as model failure. |
| `other_unclear` | Available evidence does not support one of the preceding labels. | Adjudicate or collect more evidence. |

Each row has one primary label, an optional secondary label, and a short
evidence note. `pending` rows cannot contain judgments;
`needs_adjudication` rows cannot assert a primary label. Published shares use
reviewed cases only.

The current 72-case census and its limitations are documented in
[`docs/nfcorpus_first_stage_taxonomy_review.md`](nfcorpus_first_stage_taxonomy_review.md).

## Unit Of Review

One review item is a paired query record:

- query id
- query text
- reference answer(s)
- BM25 passages and generated answer
- reranked passages and generated answer
- per-query metric scores
- optional grounding scores

The label should describe the most likely primary cause of the metric
difference. If two causes are equally plausible, use a primary label plus a
note rather than inventing a combined category.

## Current Regression Labels

### `truncation_midword`

The generated answer ends inside a word or phrase, or produces a fragment that
looks like a clipped extract. The upstream passage may be relevant, but the
generator output is malformed enough to hurt the metric or NLI entailment.

Likely layer: generation.

### `truncation_short`

The answer is grammatical enough, but too short to express the reference
answer. It often copies a title, entity name, or local phrase when the passage
contains a fuller answer nearby.

Likely layer: generation.

### `topic_drift`

The retrieved/reranked passage is about a neighbouring concept but not the
asked entity or relation. The generated answer follows the shown passage, so
the downstream failure is caused by ranking the wrong evidence.

Likely layer: retrieval or reranking.

### `extractive_passage_bias`

The generator copies a salient span from the prompt even when the reference
answer requires synthesis or a less prominent span. This is not pure retrieval
failure: the answer may be present, but the generator picks the wrong surface
form.

Likely layer: generation conditioned on passage layout.

### `semantic_mismatch`

The produced answer has lexical overlap with the reference but changes the
meaning. Examples include wrong numeric value, wrong date, wrong person, or
negated relation.

Likely layer: retrieval, generation, or annotation depending on context.

### `annotation_or_reference_gap`

The output appears correct or defensible from the passages, but the reference
answer is narrow, missing a paraphrase, or mismatched to the query wording.

Likely layer: dataset/evaluation.

### `metric_artifact`

The output is semantically acceptable but penalised by the chosen surface
metric, or an NLI/semantic scorer fails because the answer is a fragment,
list, unit expression, or formatting variant.

Likely layer: evaluation.

### `passage_context_conflict`

Top passages contain conflicting answers or incompatible contexts. The
generator selects one plausible answer, but the reference or another passage
supports a different one.

Likely layer: retrieval set composition.

## Grounding-Specific Labels

### `paraphrase_reorder`

Content words are present in the passages, but word order or phrasing differs
enough to reduce n-gram or NLI scores.

### `partial_external`

Most of the answer is grounded, but a small token, unit, morphology, or
normalisation choice is not present exactly in the prompt.

### `parametric_or_external`

The answer introduces content that is not supported by the shown passages.
This is the label to watch if a future generator becomes less extractive.

## Labeling Procedure

1. Fix a random seed before sampling cases.
2. Sample from the target bucket, not from all predictions.
3. Read the query, references, both outputs, and both top-k passage sets.
4. Assign one primary label.
5. Add a short evidence note when the case is non-obvious.
6. Report counts and shares, but do not over-interpret tiny samples.

Recommended sample sizes:

- 30 cases for a quick sanity check.
- 40-50 cases for a reportable regression taxonomy.
- 100+ cases if a paper section depends on the taxonomy.

## Current Snapshot

The existing 40-query seeded regression triage found that most regressions are
generation-side:

| Label | Count | Share |
|---|---:|---:|
| `truncation_midword` | 22 | 55% |
| `truncation_short` | 14 | 35% |
| `topic_drift` | 2 | 5% |
| `extractive_passage_bias` | 2 | 5% |
| `semantic_mismatch` | 0 | 0% |

Interpretation: reranking usually improves evidence placement, but T5-small
often turns richer evidence into short or fragmentary extracts. That supports
the next research step: change generator capacity and prompt format before
claiming the reranker itself is the residual bottleneck.

## When To Add A New Label

Add a label only if at least three reviewed cases do not fit the existing
taxonomy and the new distinction would change an experiment decision.

When a label is added, update:

- this file
- the relevant taxonomy script
- any report table that consumes the label set
- tests that pin expected labels or summary columns

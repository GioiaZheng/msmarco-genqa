# Evaluation Protocol

This protocol defines how to produce, compare, and report the repository's
main RAG evaluation results. It is written for repeatable research work, not
for a one-off demo.

## Claim Under Test

Primary claim:

> Replacing BM25 top-3 passages with cross-encoder-reranked dense top-3
> passages improves generation quality on the same MS MARCO dev/small query
> set under the same frozen generator.

The claim is comparative. It is valid only when the two generation arms share
the same query ids, prompt template, generator checkpoint, decoding settings,
and evaluation code.

## Dataset

- Dataset: MS MARCO Passage Ranking.
- Evaluation split: `dev/small`.
- Query count: 6,980.
- Corpus: roughly 8.8M passages.
- Large data is downloaded through `ir_datasets` and is not committed.

The dense retrieval stage currently uses a qrels-anchored 50k-passage sample.
That sampling makes dense-vs-BM25-on-sample comparisons valid, but it does not
make the dense absolute metric directly comparable to full-corpus BM25.

## Canonical Command Surface

Inspect the evaluation plan:

```bash
rag-eval run --config configs/baseline.yaml --dry-run
```

Run the configured workflow:

```bash
rag-eval run --config configs/baseline.yaml
```

For focused work, run a subset:

```bash
rag-eval run --config configs/baseline.yaml --only generation_bm25 generation_reranked paired_bootstrap_ci
```

The command expands to the stage-specific `mgq-*` and analysis script calls
recorded in `configs/baseline.yaml` under `rag_eval`.

## Stage Order

1. `bm25_retrieval`
2. `dense_retrieval`
3. `cross_encoder_rerank`
4. `retrieval_quality_report`
5. `retrieval_lift_analysis`
6. `generation_bm25`
7. `generation_reranked`
8. `paired_bootstrap_ci`
9. `grounding_audit`

Each stage writes under `outputs/`. Output directories are gitignored; metrics,
manifests, and summaries should be copied into reports only after they are
checked against this protocol.

## Pairing Rules

For generation comparison:

- BM25 generation input: `outputs/week02_bm25/run.tsv`.
- Reranked generation input: `outputs/week05_reranker_full/run.tsv`.
- Both generation arms must use `--restrict-to-run` against the other arm's
  upstream run.
- The final `predictions.jsonl` files must contain the same query ids in the
  same order.

The bootstrap script enforces matching length, qid set, and qid order. If it
fails, do not manually align the results in a spreadsheet. Fix the upstream
generation command or write a checked sorter.

## Metrics

Surface metrics:

- Token-F1
- ROUGE-L
- sentence BLEU
- Exact match

Retrieval metrics:

- MRR@10
- nDCG@10
- Recall@100 / Recall@1000, depending on stage

Use `retrieval_quality_report` for matched-qid retrieval comparisons before
interpreting deltas. Coverage counts are part of the report and should be
checked before comparing dense, RRF, or reranked runs.

Grounding metrics:

- lexical content-token grounding
- n-gram grounding
- optional NLI entailment

Semantic proxy:

- BERTScore on a fixed paired subsample when the scorer is available.

No single metric is treated as the whole story. The current result is strong
because the retrieval, surface-generation, bootstrap, and error-analysis
signals are read together.

## Statistical Test

Use paired bootstrap over query-level scores:

- Default resamples: 10,000.
- Default confidence level: 95 percent.
- Default seed: 42.
- Delta direction: reranked minus BM25.

Report the mean delta and confidence interval. If the interval crosses zero,
the claim is not statistically supported for that metric and setting.

## Reproducibility Requirements

For a result to be reportable, record:

- git commit and dirty-tree status
- config file and resolved config hash
- seed
- upstream run files
- output directory
- dependency file hashes
- environment fingerprint

The manifest schema in `docs/reproducibility_protocol.md` is the detailed
contract. Runs that bypass the manifest contract are allowed during
development, but should not be used for headline claims.

## Reporting Rules

When updating `RESULTS.md` or a report:

- State whether retrieval metrics are full-corpus or sampled.
- State whether generation is full dev/small or a smaller smoke subset.
- State the exact paired query count.
- State the generator checkpoint and decoding budget.
- State bootstrap seed, resample count, and confidence level.
- Separate confirmed results from hypotheses and queued follow-ups.

Avoid language that implies general QA capability. The current system is an
MS MARCO Passage RAG evaluation pipeline with a frozen generator; the evidence
does not support broader claims without additional datasets and models.

## Failure Review Gate

Before treating a new run as a real improvement, inspect regressions:

1. Generate the paired bootstrap summary.
2. Run the regression/failure taxonomy scripts.
3. Sample at least 30-50 regressions with a fixed seed.
4. Record whether the failure mode is retrieval-side, generation-side,
   metric-side, or annotation-side.

`docs/failure_taxonomy.md` defines the labels and adjudication notes.

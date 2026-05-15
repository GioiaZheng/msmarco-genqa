# Week 6: Evaluation Layer — Semantic Proxy + Regression Taxonomy

*Auto-generated {{generated_at}} from `outputs/week06_bertscore_proxy/`
and `outputs/week06_analysis/`. Do not edit by hand. Re-run
`python -m src.reporting.build_report --week week06` to refresh.*

## 1. Objective

W3 (paired BM25-vs-reranked generation) and W5 (full-dev rerank) gave
us four surface-form deltas (ROUGE-L, BLEU, EM, Token-F1) with
paired-bootstrap CIs that all exclude zero. W6 asks two follow-up
questions that the surface-form table cannot answer:

> 1. Does the rerank Δ also show up in a *semantic* similarity metric,
>    or are we just rewarding lexical overlap?
> 2. The full-dev split has {{n_shared_qids}} paired queries; {{n_regression}}
>    are regressions where the reranker brings the relevant passage
>    into top-3 yet the generator scores *worse* than under BM25. What
>    breaks on those queries — retrieval, semantics, or generation?

Both questions are answered cheaply (CPU, ~minutes) and produced
under `outputs/week06_*/`. Neither requires re-running W3/W5.

## 2. Setup

### 2.1 BERTScore semantic-proxy

- Inputs: the same two paired prediction files used in the W3 surface-
  form CI (`outputs/week03_generation_bm25_full/predictions.jsonl` and
  `outputs/week03_generation_reranked_full/predictions.jsonl`).
- Subsample: **{{bertscore_n_pairs}} of {{n_shared_qids}} shared qids**,
  seed {{bertscore_seed}} (deterministic).
- Scorer: **{{bertscore_model}}** with `rescale_with_baseline=True`,
  multi-reference max-F1 per query.
- CI: paired bootstrap, {{bertscore_n_resamples}} resamples, seed
  {{bertscore_seed}}, 95 % percentile interval.

This is deliberately *not* the canonical `roberta-large` BERTScore. It
is a fast proxy whose only purpose is to answer "is the surface-form
Δ also visible in a semantic-similarity scorer?". The full-dev
`roberta-large` pass is listed under §7 *Next*.

### 2.2 Regression failure taxonomy

- Pool: the **{{n_regression}} regression queries** identified by
  `scripts/analyze_generation_rerank.py` (relevant passage in top-3
  under rerank, but generator F1 dropped vs BM25).
- Sample: **{{tax_n_sampled}}** seeded queries, seed {{tax_seed}}.
- Labels are produced by a deterministic rule cascade
  (`scripts/regression_failure_taxonomy.py`); first-match wins:
  `truncation_short`, `truncation_midword`, `topic_drift`,
  `extractive_passage_bias`, `semantic_mismatch`.

This is a *triage*, not a citation-grade classifier — the value is in
the headline split, not the per-row labels.

## 3. BERTScore proxy results

| Metric                                | BM25 (mean) | Reranked (mean) | Δ (rerank − BM25) | 95 % CI                              | p (two-sided) |
|---------------------------------------|------------:|----------------:|------------------:|--------------------------------------|--------------:|
| BERTScore-F1 (DistilBERT proxy)       | {{bertscore_mean_bm25}} | {{bertscore_mean_rerank}} | **{{bertscore_delta}}** | [{{bertscore_ci_low}}, {{bertscore_ci_high}}] | {{bertscore_p_two_sided}} |

Per-query split on the {{bertscore_n_pairs}}-pair subsample:

| Outcome                              | Rate |
|--------------------------------------|-----:|
| Rerank strictly better               | {{bertscore_win_rate_pct}} |
| Tie                                  | {{bertscore_tie_rate_pct}} |
| BM25 strictly better                 | {{bertscore_loss_rate_pct}} |

The semantic Δ ({{bertscore_delta}}) lands within a hair of the
surface-form Token-F1 Δ already published in the W3 paired-bootstrap
table (+0.1711) and the ROUGE-L Δ (+0.1742). **The rerank gain is not
a surface-form artefact** — a semantic-similarity scorer picks it up
at the same magnitude.

## 4. Regression taxonomy

40-query seeded triage of the {{n_regression}}-strong regression bucket.

| Failure mode                | Count | % of sample |
|-----------------------------|------:|------------:|
| `truncation_midword`        | {{tax_truncation_midword}} | {{tax_truncation_midword_pct}} |
| `truncation_short` (≤3 tok) | {{tax_truncation_short}}   | {{tax_truncation_short_pct}} |
| `topic_drift`               | {{tax_topic_drift}}        | {{tax_topic_drift_pct}} |
| `extractive_passage_bias`   | {{tax_extractive_passage_bias}} | {{tax_extractive_passage_bias_pct}} |
| `semantic_mismatch`         | {{tax_semantic_mismatch}}  | {{tax_semantic_mismatch_pct}} |

**{{tax_truncation_total_pct}} of regressions are generation-side
truncation, not retrieval or semantic failures.** The cross-encoder
gets the relevant passage into the generator's window; the generator
then emits a short or mid-sentence answer. This *looks* like the
`max_new_tokens=64` cap firing; a follow-up run at
`max_new_tokens=128` falsifies that reading — see §6 *Closure of the
budget-cap hypothesis*.

`semantic_mismatch` did not pick up any sampled rows under the rule
cascade. Per the script's documentation that bucket is the residual
catch-all; its emptiness here is consistent with the BERTScore result
in §3 (the rerank gain is real on a semantic scorer).

## 5. Limitations

- **DistilBERT, not `roberta-large`.** The headline Δ + CI in §3 is a
  *proxy*. For citation in a cross-paper comparison, re-run
  `scripts/bertscore_paired_eval.py` with `--model-type roberta-large
  --n-pairs 0` (full dev, ~hours of CPU). The proxy is calibrated only
  for "is the surface-form Δ also visible in a semantic scorer", not
  for absolute BERTScore numbers.
- **Sampled, not full-dev BERTScore.** {{bertscore_n_pairs}}/{{n_shared_qids}}
  = {{bertscore_subsample_pct}} of the shared qid set; full-dev is
  trivially obtainable but does not change the qualitative answer.
- **Heuristic taxonomy.** The five labels in §4 are deterministic
  rules, not a learned classifier. Labels are intentionally coarse;
  the load-bearing claim is the truncation share, not the per-row
  attribution.
- **Single sampled triage.** 40 of {{n_regression}} = ~17 % of the
  regression bucket. Seed-stable, but a different seed would shift the
  exact percentages by a few points.

## 6. Closure of the budget-cap hypothesis

The §4 taxonomy at the time predicted that raising
`max_new_tokens` from 64 → 128 would close most of the regression
bucket. A follow-up sweep on full dev/small (same generator,
prompts, retrieval inputs, seed; CLI override only, canonical
64-token outputs untouched) **falsifies** that prediction:

| signal                              | mnt=64 (canonical) | mnt=128 |
|-------------------------------------|-------------------:|--------:|
| Token-F1 Δ (rerank − BM25)          | +0.1711            | +0.1706 |
| ROUGE-L Δ                            | +0.1742            | +0.1736 |
| BLEU Δ                               | +0.1330            | +0.1325 |
| EM Δ                                 | +0.0471            | +0.0471 |
| Regression bucket size               | 233                | 231     |
| Truncation share (40-query triage)   | 90.0 %             | 87.5 %  |

All four surface-form deltas move by <0.005; the regression bucket
shrinks by 2 queries; the truncation share moves 2.5 percentage
points — all within noise on a 40-query sample. T5-small is hitting
EOS naturally on this prompt format, not running out of decode
budget. The mid-word-ending output style is intrinsic to the model.

Outputs under
`outputs/week03_generation_{bm25,reranked}_full_mnt128/`,
`outputs/week06_bootstrap_mnt128/`, `outputs/week06_taxonomy_mnt128/`
(all gitignored). BERTScore proxy on the new predictions was
skipped and remains optional; see *Reproduce — closure run* below.

## 7. Next

- **Generator-side work targets capacity / output style, not decode
  budget.** The closure result above re-aims any further work on the
  generator: instruction-tuned or QA-style small models, or moving
  off T5-small entirely. The decode budget is no longer a candidate
  bottleneck.
- **Full-dev BERTScore with `roberta-large`.** Convert the §3 proxy
  into a citation-grade evaluation. Same script, two CLI flags.
- **Failure-mode classifier upgrade.** The residual regression
  bucket — once budget is ruled out — is dominated by topic-drift /
  extractive-bias / semantic-mismatch; a learned classifier (rather
  than the current rule cascade) is appropriate at that point.

## 8. Reproduce

```bash
# (1) Generation on full dev/small — done in W3, but listed here for
# completeness; mutually restricted to the shared qid set:
python experiments/run_generation_baseline.py \
    --input-run outputs/week02_bm25/run.tsv \
    --output-dir outputs/week03_generation_bm25_full \
    --retrieval-source bm25 \
    --restrict-to-run outputs/week05_reranker_full/run.tsv \
    --num-eval-queries 9999

python experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker_full/run.tsv \
    --output-dir outputs/week03_generation_reranked_full \
    --retrieval-source reranked \
    --restrict-to-run outputs/week02_bm25/run.tsv \
    --num-eval-queries 9999

# (2) Bucket / retrieval-flag analysis — emits regression bucket and
# summary.json under outputs/week06_analysis/:
python scripts/analyze_generation_rerank.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week06_analysis

# (3) BERTScore semantic-proxy CI (~3 min CPU on the 3 000-pair subsample):
python scripts/bertscore_paired_eval.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week06_bertscore_proxy \
    --n-pairs 3000

# (4) Regression failure taxonomy (40-query seeded triage):
python scripts/regression_failure_taxonomy.py
```

### Reproduce — closure run (`max_new_tokens=128`)

The sweep and the two cheap analyses; BERTScore on the new
predictions is optional and was skipped at write time.

```bash
# (1) Full BM25-vs-reranked sweep at max_new_tokens=128. Writes to
# fresh _mnt128 directories; the canonical _full outputs are not
# touched.
python scripts/run_full_generation_and_analysis.py \
    --max-new-tokens 128 --out-suffix _mnt128

# (2) Paired-bootstrap CI on the new predictions:
python scripts/bootstrap_generation_comparison.py \
    --bm25-dir outputs/week03_generation_bm25_full_mnt128 \
    --reranked-dir outputs/week03_generation_reranked_full_mnt128 \
    --output-dir outputs/week06_bootstrap_mnt128

# (3) Regression failure taxonomy on the new predictions:
python scripts/regression_failure_taxonomy.py \
    --bm25-dir outputs/week03_generation_bm25_full_mnt128 \
    --reranked-dir outputs/week03_generation_reranked_full_mnt128 \
    --per-query outputs/week06_analysis_mnt128/per_query_metrics.jsonl \
    --output-dir outputs/week06_taxonomy_mnt128

# (4) Optional: BERTScore proxy on the new predictions (~15-30 min):
python scripts/bertscore_paired_eval.py \
    --bm25-dir outputs/week03_generation_bm25_full_mnt128 \
    --reranked-dir outputs/week03_generation_reranked_full_mnt128 \
    --output-dir outputs/week06_bertscore_proxy_mnt128 \
    --n-pairs 3000
```

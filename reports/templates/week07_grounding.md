# Week 7: Grounding audit — is T5-small extracting from the prompt?

*Auto-generated {{generated_at}} from `outputs/week07_grounding/`. Do
not edit by hand. Re-run `python -m src.reporting.build_report --week
week07` to refresh.*

## 1. Objective

W3/W5/W6 established that **reranked retrieval roughly doubles every
W3 generation metric** (Token-F1 +0.171, ROUGE-L +0.174, both 95 %
CIs strictly above zero), that this is **not a surface-form artefact**
(BERTScore proxy Δ +0.173), and that the **decode budget is not the
bottleneck** (mnt=64 → 128 closure run did not move metrics or shrink
the regression bucket). One question that remained unanswered: *what
is T5-small actually doing on this prompt format?* Specifically:

> Is the model **extracting** content from the retrieved passages, or
> **generating** from its parametric memory and happening to overlap?

This is the *grounding* question. It bounds how much of the W3 gain
should be attributed to retrieval (passage content makes it into the
prompt) vs to the generator (content is produced de novo). W7 answers
it with two cheap deterministic metrics computed on the **already-on-
disk W3 predictions** — no new generation, no model load.

## 2. Setup

Inputs: the same two paired prediction files used by every W3/W5/W6
analysis script — `outputs/week03_generation_bm25_full/predictions.jsonl`
and `outputs/week03_generation_reranked_full/predictions.jsonl`. The
key property is that each row already contains the *exact* top-K
passage texts that the generator saw at prompt time, alongside the
prediction. So grounding can be scored without re-running anything.

Two metrics, both pure-Python regex-tokenised CPU functions
(`src/evaluation/grounding.py`, ~150 lines, no model load, no NLTK or
sklearn dependency):

- **Lexical content-token grounding.** Fraction of unique non-stopword
  tokens in the prediction (lowercased) that appear anywhere in the
  union of passage texts. A local 85-word stopword list keeps the
  metric self-contained.
- **3-gram grounding.** Fraction of the prediction's contiguous 3-grams
  (case-folded, stopwords retained, since n-gram order is the signal)
  that appear as a contiguous span in at least one passage. Catches
  phrase-level extractiveness the lexical metric is blind to.

Edge cases (documented, stable, *reported separately*): predictions
with no content tokens score lexical 1.0 vacuously; predictions
shorter than n tokens score n-gram 1.0 vacuously. The audit reports
the count of such predictions per arm so this convention cannot
silently inflate the headline number.

NLI-based grounding (entailment of prediction by passages) is the
canonical semantic faithfulness metric and is the obvious extension;
it is **deliberately not in this commit** (see §7 *Next*).

## 3. Lexical content-token grounding

Per-query metric: fraction of the prediction's unique non-stopword
tokens that appear anywhere in the prompt's passage union. 1.0 means
every distinct content word the model emitted is present in one of
the top-K passages it was conditioned on.

| Metric                  | BM25 (mean) | Reranked (mean) | Δ (rerank − BM25) | 95 % CI                              | p (two-sided) |
|-------------------------|------------:|----------------:|------------------:|--------------------------------------|--------------:|
| Lexical (content-token) | {{lex_mean_bm25}} | {{lex_mean_rerank}} | **{{lex_delta}}** | [{{lex_ci_low}}, {{lex_ci_high}}] | {{lex_p_two_sided}} |

Per-query split:

| Outcome                        | Rate |
|--------------------------------|-----:|
| Rerank strictly more grounded  | {{lex_win_rate_pct}} |
| Tie                            | {{lex_tie_rate_pct}} |
| BM25 strictly more grounded    | {{lex_loss_rate_pct}} |

**Headline read.** Both arms sit at the lexical ceiling — almost
every content word the model emits is already in the prompt. T5-small
on the `question: ... context: ...` shape is effectively performing
**extractive QA**, not generative QA. The Δ between arms is tiny
({{lex_delta}}); the rerank advantage on lexical grounding alone is
near-zero because there is essentially no slack for retrieval to take
advantage of at the *word* level.

## 4. 3-gram grounding

Per-query metric: fraction of the prediction's contiguous 3-grams
that match a contiguous span in any single passage. Tighter than the
lexical metric — catches "the model strung together the right words
in the wrong order" cases that the bag-of-content-tokens score misses.

| Metric          | BM25 (mean) | Reranked (mean) | Δ (rerank − BM25) | 95 % CI                              | p (two-sided) |
|-----------------|------------:|----------------:|------------------:|--------------------------------------|--------------:|
| 3-gram          | {{ngram_mean_bm25}} | {{ngram_mean_rerank}} | **{{ngram_delta}}** | [{{ngram_ci_low}}, {{ngram_ci_high}}] | {{ngram_p_two_sided}} |

Per-query split:

| Outcome                        | Rate |
|--------------------------------|-----:|
| Rerank strictly more grounded  | {{ngram_win_rate_pct}} |
| Tie                            | {{ngram_tie_rate_pct}} |
| BM25 strictly more grounded    | {{ngram_loss_rate_pct}} |

Both arms remain near the ceiling at the phrase level too. The Δ is
small but the CI strictly excludes zero: reranking does push 3-gram
grounding up — slightly — because better-ranked passages contain
contiguous phrasings closer to the gold answer.

## 5. Edge cases

Predictions that fall into the documented "vacuous → 1.0" path of
each metric (see `src/evaluation/grounding.py` docstrings):

| Arm     | Lex vacuous (no content tokens) | 3-gram vacuous (<3 tokens) |
|---------|--------------------------------:|---------------------------:|
| BM25    | {{bm25_lex_vacuous}}            | {{bm25_ngram_vacuous}}     |
| Rerank  | {{rerank_lex_vacuous}}          | {{rerank_ngram_vacuous}}   |

The 3-gram vacuous counts (~{{ngram_vacuous_pct_rounded}} of {{n_shared_qids}}
shared qids per arm) are exactly the `truncation_short` /
`extractive_passage_bias` pattern surfaced by the W6 regression
taxonomy — predictions of 1–2 tokens, often a passage title. They
inflate the n-gram headline by a few tenths of a percentage point
*identically on both arms*, so the paired Δ is unaffected. The
lexical vacuous counts are negligible (<0.3 % of either arm).

## 6. Limitations

- **Extractiveness, not faithfulness.** The two metrics measure
  whether the prediction *literally derives from* the prompt
  passages. A heavy paraphrase that is perfectly faithful semantically
  will score low; a verbatim copy from a *distractor* passage will
  score high. The right tool for the semantic faithfulness question
  is a small NLI cross-encoder over (passages → prediction); it is
  listed in §7 *Next*.
- **Top-K passages only.** Grounding is measured against the top-3
  passages T5-small actually saw at prompt time, not against the full
  retrieved top-100 or the broader corpus. This is the relevant
  notion of grounding for *this* generator: the question is "did the
  model use the prompt", not "is the answer findable somewhere in
  MS MARCO".
- **Word- and phrase-level signal only.** Both metrics are bag-of-
  unigrams / bag-of-3-grams. They do not see word order beyond n=3,
  nor any morphology / synonymy / tense variation. A future NLI pass
  would close all three gaps.
- **The ceiling effect is the headline.** Because both arms sit at
  ~99 % lexical grounding, the W7 rerank-vs-BM25 Δ for grounding is
  essentially zero. This is a *finding*, not a limitation, but it
  does mean the audit's primary value here is the *level*, not the
  Δ: it bounds how much of the W3 surface-form rerank gain is
  attributable to the generator vs to retrieval. The answer is
  "almost none to the generator at this prompt format".

## 7. Next

- **Span-level citation linking.** For each prediction, attribute its
  content tokens to the specific source passage. Cheap (~50 lines)
  but explicitly out of scope for this commit per the W7 scope.

### 7.1 W7 follow-ups (done; live in the README)

The two forward-pointers in the original W7 §7 (NLI-based grounding
and per-query bucket / type breakdown) were both delivered in a later
pass. Numbers are not embedded here to keep the W7 snapshot stable;
see the README §1 Week 7 for the live writeup.

- *W7-A* — NLI-entailment grounding via
  `cross-encoder/nli-deberta-v3-small` on a 3 000-paired-qid subsample.
  Module `src/evaluation/nli_grounding.py`, driven by this same
  `scripts/grounding_audit.py` with `--nli-n-pairs 3000`. Headline:
  the NLI Δ is **negative and strictly excludes zero** (BM25 0.227 →
  Reranked 0.082, Δ −0.145, 95 % CI [−0.16, −0.13]) — the only metric
  in this project whose Δ reverses sign vs the W3 surface-form story.
- *W7-C* — per-query Spearman / Pearson + binned (≥0.9 vs <0.9)
  Mann-Whitney joining lex / 3-gram / NLI grounding against W6
  Token-F1 and a freshly-cached per-qid BERTScore-F1
  (`scripts/grounding_correlation.py`, output
  `outputs/week07_grounding_correlation/`). All 12 binned cells have
  Δ(high − low) > 0; magnitudes 0.04–0.10 on Token-F1 / BERTScore.
- *W7-D* — seeded 30-case study of the rerank-arm low-grounding tail
  (197 queries with lex < 0.9 OR ngram < 0.9), with a coarse
  rule-cascade label per case. **0 / 30 hallucinations** (no case had
  lex < 0.5); 77 % are paraphrase / reordering, 23 % are tokeniser /
  morphology artefacts. Script
  `scripts/low_grounding_case_study.py`, output
  `outputs/week07_low_grounding_cases/cases.md`.

## 8. Reproduce

```bash
# The audit runs on the existing W3 paired-prediction files. No new
# generation, no model load; CPU, ~2 minutes end-to-end (~2 s scoring
# + the paired bootstrap on two metrics).
python scripts/grounding_audit.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week07_grounding
```

Outputs (gitignored):

- `outputs/week07_grounding/per_query_grounding.jsonl` — per-qid scores
  for both arms and both metrics, ready for join with the W6 bucket /
  query-type tables.
- `outputs/week07_grounding/summary.json` — means, paired-bootstrap
  CIs, win/tie/loss, edge-case diagnostics, full provenance.

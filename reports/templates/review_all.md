# MS MARCO Generative QA — Progress Report

*Generated {{generated_at}}. For per-week depth see `reports/generated/weekNN_*.pdf`.*

## Project goal

Build an end-to-end open-domain QA system on MS MARCO that goes
**query → retrieve passages → generate an answer**, and measure each
stage in isolation so it's clear where errors come from. The aim is not
SOTA; the aim is a clean, reproducible pipeline whose retrieval,
reranking, and generation stages can each be ablated.

## Methods (by week)

| Week | Stage                     | Data                                                  | Method                                                                  |
|------|---------------------------|-------------------------------------------------------|-------------------------------------------------------------------------|
| W1   | EDA                       | MS MARCO QA v2.1 validation, 5k-row sample            | Length / type / answer-type distributions; sizing decisions for W2–W5   |
| W2   | Sparse retrieval (BM25)   | MS MARCO Passage (8.8M docs), dev/small (6 980 q)     | `bm25s` backend, `k1=1.5`, `b=0.75`, top-1000 per query                 |
| W3   | RAG generation            | BM25 / reranked top-3 ∩ QA references, full dev/small (6 980 q) | T5-small (no fine-tuning), prompt = `question: <q> context: <p1 p2 p3>` |
| W4   | Dense retrieval           | Qrels-anchored 50k sample of Passage, dev/small (6 980 q) | `all-MiniLM-L6-v2` + FAISS `IndexFlatIP`; same sample re-indexed with BM25 for apples-to-apples comparison |
| W5   | Cross-encoder reranking   | W4 dense top-100, full dev/small (6 980 q)            | `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank, depth K=100              |
| W6   | Evaluation layer + decode-budget closure | W3 paired predictions (6 980 q), 3 000-pair BERTScore subsample, 40-query taxonomy triage, mnt=128 sweep | DistilBERT BERTScore proxy; deterministic 5-bucket regression taxonomy; `max_new_tokens` 64→128 falsification run; question-form / structural-feature breakdowns (W6-A/B/C) |
| W7   | Grounding audit           | W3 paired predictions (6 980 q), 3 000-pair NLI subsample | Lexical content-token + 3-gram grounding (deterministic CPU, ~2 s); DeBERTa-v3 NLI entailment proxy; grounding ↔ Token-F1 / BERTScore correlation; low-grounding case study (W7-A/C/D) |
| W5-A | First-stage × rerank head-to-head | BM25 full-corpus top-100 + W4 dense top-100 (both 6 980 q) | Same `ms-marco-MiniLM-L-6-v2` reranker on both first stages; reports Δ and constrained recovery rate |

All runs share a single config (`configs/baseline.yaml`); the four
entrypoints in `experiments/run_*.py` produce the W1–W5 metrics each
report cites. W6 and W7 are *post-hoc analysis layers* over the
already-on-disk W3 predictions — no new generation or model load, all
under `scripts/*.py`.

## Key results

**W1 (EDA)** — Queries are short (5–10 words median); passages ~10× longer
(median ~50 words). `DESCRIPTION` and `NUMERIC` dominate the query-type
buckets; a non-trivial fraction of queries has no answer and must be
filtered for any SFT supervision. Most queries have 0 or 1 marked-relevant
passage (sparse, binary qrels).

**W2 (BM25 baseline, full 8.8M corpus, 6 980 dev/small queries)**

| Metric      | Value  |
|-------------|--------|
| MRR@10      | 0.1703 |
| Recall@100  | 0.6212 |
| Recall@1000 | 0.8154 |

Published Anserini/Lucene reference is ~0.184 MRR@10; the ~0.014 gap is
attributable to tokenizer differences between Lucene and `bm25s`. No
`k1`/`b` tuning. Search cost: 586 ms/query.

**W3 (RAG, T5-small + retrieved top-3, full dev/small 6 980 queries)**

Two paired runs over the **same 6 980 dev/small qids** (mutually
restricted via `--restrict-to-run`), differing only in upstream
retrieval source:

| Retrieval &rarr; T5-small | ROUGE-L | BLEU   | EM     | Token-F1 |
|---------------------------|--------:|-------:|-------:|---------:|
| BM25 top-3                | 0.1859  | 0.0717 | 0.0135 | 0.1966   |
| Reranked top-3            | 0.3621  | 0.2922 | 0.0606 | 0.3677   |
| **Δ (rerank − BM25)**     | **+0.1763** | **+0.2206** | **+0.0471** | **+0.1711** |

Reranking the first stage **roughly doubles every generation metric**
on the full benchmark. 95 % paired-bootstrap CIs (n = 6 980, 10 k
resamples, seed 42) on the per-query Δ all sit strictly above zero:
ROUGE-L [+0.1663, +0.1820], BLEU [+0.1265, +0.1395], EM [+0.0417,
+0.0527], Token-F1 [+0.1632, +0.1789]; two-sided p < 0.001 in every
case.

Mechanism (retrieval flag): the rate of having at least one
relevance-judged passage in the top-3 jumps from **20.8 % (BM25) to
96.9 % (reranked)** — the cross-encoder is what gets the relevant
passage into the generator's window.

By query type (token-F1 Δ), `DESCRIPTION` (n=3725) gains most
(+0.2050); `NUMERIC` (n=1665) gains least (+0.1238) — short numeric
answers depend more on lexical surface match than on which-of-the-
near-duplicates the reranker picks. Per-query split:
**4 015 improvements / 1 766 regressions / 1 199 ties** over 6 980
queries.

T5-small is pretrained-only (no SFT) so absolute level numbers are
modest. The load-bearing claim is the **paired Δ between rows**, which
is statistically robust on n = 6 980.

A *low-cost semantic evaluation sanity check on 3 000 paired examples
using DistilBERT-based BERTScore* (rescaled, paired-bootstrap CI; seed
42) gives BM25 0.2192 → reranked 0.3920, **Δ +0.1728** ([+0.1608,
+0.1850], p < 0.001). Per-query: rerank strictly better on 64.8 % of
qids; tie 5.7 %; strictly worse on 29.5 %. The semantic-proxy Δ
(+0.1728) is within a hair of the surface-form Token-F1 Δ (+0.1711)
and ROUGE-L Δ (+0.1742), so the reranker improvement **is not a
surface-form artefact** — it shows up in a semantic-similarity scorer
at the same magnitude. (DistilBERT is intentionally not the canonical
BERTScore encoder; this is a proxy that complements rather than
replaces a full `roberta-large` evaluation.)

A 40-query seeded triage of the 233-strong `regression` bucket shows
that **~90 % of regressions are generation-side truncation, not
retrieval or semantic failures**: 55 % `truncation_midword`, 35 %
`truncation_short` (≤ 3 tokens, generator extracted only a title-
like fragment), 5 % `topic_drift`, 5 % `extractive_passage_bias`,
0 % `semantic_mismatch`. The midword pattern initially looked like
the `max_new_tokens = 64` cap firing; W6's closure run below
**falsifies** that reading.

**W4 (Dense vs BM25 on the same 50k qrels-anchored sample, 6 980 queries)**

| Metric      | BM25 (sample) | Dense (sample) | Δ          |
|-------------|---------------|----------------|------------|
| MRR@10      | 0.6948        | 0.8830         | **+0.1882** |
| nDCG@10     | 0.7270        | 0.9041         | +0.1771    |
| Recall@100  | 0.9338        | 0.9946         | +0.0608    |
| Recall@1000 | 0.9686        | 0.9991         | +0.0305    |

Dense wins on every metric. Caveat: every dev/small relevant doc is
unconditionally included in the 50k pool, so absolute numbers are
upper-bounded; the *delta* between dense and BM25 on the same pool is
what's meaningful. Recall@100 = 0.9946 means dense has effectively hit
the ceiling on this sampled setting → from W5 onward, recall is no
longer the bottleneck.

**W5 (Cross-encoder rerank over W4 dense top-100, full 6 980-query dev/small)**

| Metric      | Dense (top-100) | Dense + CE rerank | Δ          |
|-------------|-----------------|-------------------|------------|
| MRR@10      | 0.8830          | 0.9304            | **+0.0474** |
| nDCG@10     | 0.9041          | 0.9434            | +0.0393    |
| Recall@100  | 0.9946          | 0.9946            | +0.0000    |

Recall@100 is unchanged by construction (the reranker only re-orders the
top-100). The interesting columns are MRR@10 and nDCG@10: even with
recall saturated, the cross-encoder still buys roughly +0.04 by tightening
*local ordering* — pushing the relevant passage from rank 2–3 to rank 1.
Cost: 4 h 37 min wall-clock for 538 k pairs (~32 pairs/s), peak RSS
3.3 GiB; the CE is the slowest stage by ~2 orders of magnitude. The
earlier 1 000-query subsample of dev/small produced consistent deltas
(MRR Δ +0.0435, nDCG Δ +0.0398), confirming the gain is not an artefact
of the subsample.

**W5-A (Rerank head-to-head across first stages, same 6 980 dev/small queries)**

The same `ms-marco-MiniLM-L-6-v2` reranker applied to two first stages
of very different starting strength:

| First stage           | MRR@10 (FS) | MRR@10 (+ CE) | Δ_MRR     | nDCG@10 (FS) | nDCG@10 (+ CE) | Δ_nDCG    | Constrained MRR recovery |
|-----------------------|------------:|--------------:|----------:|-------------:|---------------:|----------:|-------------------------:|
| BM25 (full 8.8M)      | 0.1703      | 0.3554        | **+0.1851** | 0.2138       | 0.4017         | **+0.1879** | 41.1 %                   |
| Dense MiniLM-L6 (50k) | 0.8830      | 0.9304        | +0.0474   | 0.9041       | 0.9434         | +0.0393   | 42.4 %                   |

Absolute deltas favour the BM25 arm (more room to recover), but the
**constrained recovery rate** — Δ ⁄ (Recall@100 − first-stage) — comes
in tied near ~42 % MRR / ~44 % nDCG once you account for the recall
ceiling of each arm (BM25 full-corpus Recall@100 = 0.6212 caps what the
reranker can promote; dense-on-sample sits at 0.9946). Read together
with W2 + W4 + W5: the cross-encoder closes a fixed *fraction* of the
ordering gap, regardless of first-stage strength.

**W6 (Evaluation layer — semantic proxy, regression taxonomy, decode-budget closure)**

Layered on the W3 paired predictions, no new generation:

- *Semantic-proxy BERTScore* on a 3 000-pair subsample (DistilBERT,
  rescaled, paired bootstrap 10 k resamples seed 42): BM25 0.2192 →
  reranked 0.3920, **Δ +0.1728 [+0.1608, +0.1850], p < 0.001**. Per-
  query: rerank strictly better 64.8 %, tie 5.7 %, BM25 better 29.5 %.
  Lands within ±0.002 of the surface-form Token-F1 Δ (+0.1711) and
  ROUGE-L Δ (+0.1742) — so the rerank gain is **not a surface-form
  artefact**.
- *Regression-bucket triage* (40-query seeded sample of the 233-strong
  `regression` bucket): 90 % generator-side truncation as summarised
  under W3 above.
- *Decoding-budget closure (`mnt = 64 → 128`, full dev/small)*: same
  generator, same prompts, same retrieval inputs, same seed; only the
  decode cap changes. Result: Δ Token-F1 +0.1706 (CI [+0.163,
  +0.178]), Δ ROUGE-L +0.1736, Δ BLEU +0.1325, Δ EM +0.0471 — all
  four deltas move by < 0.005, the regression bucket shrinks 233 →
  231, and the 40-query truncation share moves 90.0 % → 87.5 %. **The
  W6 budget-cap hypothesis is falsified**: T5-small is hitting EOS
  naturally on this prompt format, not running out of decode budget.
  The midword-ending style is intrinsic to the model.
- *W6-A/B/C* offline breakdowns: question-form tagger over all 6 980
  queries; per-form Δ Token-F1 (only `which`, n = 120, has a CI on
  Δ Token-F1 that includes zero — Δ = +0.0250, CI [−0.028, +0.077],
  p = 0.33, despite its retrieval-side Δ MRR@10 = +0.71); Mann-Whitney
  U on five structural features of regression vs non-regression
  queries (only signal: regression-arm top-3 passage lengths are
  marginally shorter, Δ median = −2.7 tokens, p = 0.0073, r = −0.103
  — consistent with the truncation finding).

**W7 (Grounding audit — what is T5-small actually doing?)**

Deterministic CPU pass over the same W3 paired predictions; ~2 s of
scoring + bootstrap, no model load except for the optional NLI proxy.

- *Lexical content-token grounding* (fraction of unique non-stopword
  prediction tokens appearing in the prompt's top-3 passages): BM25
  **0.9972** → reranked **0.9977**, **Δ +0.0005 [−0.0003, +0.0014],
  p = 0.24**. Both arms at the ceiling.
- *3-gram grounding*: BM25 **0.9873** → reranked **0.9905**, **Δ
  +0.0032 [+0.0015, +0.0050], p < 0.001**. Slight uplift, both arms
  still ~99 %.
- **Read:** T5-small on this prompt format is performing **extractive
  QA**. The W3/W5 Δ Token-F1 = +0.171 is almost entirely downstream
  of retrieval — the reranker puts the right words into the prompt
  and the generator copies them. This is a *calibration* of the W3
  result, not a contradiction.
- *W7-A — NLI-entailment grounding* (3 000-paired subsample,
  `cross-encoder/nli-deberta-v3-small`): BM25 **0.2270** → reranked
  **0.0821**, **Δ −0.1448 [−0.1597, −0.1297], p < 0.001**. The only
  metric in this project whose paired Δ **reverses sign** versus the
  W3/W5/W6 surface-form story. Likely mechanism: reranked-arm
  predictions are more fragmentary / mid-word-cut, which inflates
  word and 3-gram overlap but scores low on a sentence-level NLI
  cross-encoder that cannot entail a fragmentary hypothesis. To be
  tested by a generator swap (W7-B follow-up: T5-base / instruction-
  tuned small model).
- *W7-C — grounding ↔ downstream correlation*: every binned cell
  (high-vs-low grounding Mann-Whitney) shows downstream-better-on-
  high direction, but magnitudes are 0.04–0.10 in absolute Token-F1 /
  BERTScore — consistent with the ceiling effect. Per-query Spearman /
  Pearson are near-zero, again because almost all rows sit ≥ 0.99 on
  lex / 3-gram.
- *W7-D — low-grounding case study* (30-query seeded triage of the
  197 rerank-arm queries with `lex < 0.9` OR `ngram < 0.9`): 77 %
  `paraphrase_reorder`, 23 % `partial_external` (mostly tokeniser /
  morphology artefacts: `350oF` vs `350°F`; `competed` vs
  `compete`), **0 % parametric-or-external hallucinations**.
  Reinforces W7's headline: T5-small is extractive even on its
  worst-grounded outputs.

## Limitations

- **Sampled corpus from W4 onward.** Dense + rerank numbers are on a
  50k qrels-anchored sample where every relevant doc is present by
  construction. Comparable *within* W4/W5; the W4/W5 absolute MRR is
  *not* directly comparable to the W2 full-corpus number.
- **No fine-tuning anywhere.** Generic encoders/generator used as-is
  (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, `t5-small`). A
  MS-MARCO-tuned encoder (`msmarco-MiniLM-L6-cos-v5`) or SFT generator
  would close most of the gap to leaderboard baselines.
- **Dev/small only, single seed.** No held-out test split; distractor
  sample is deterministic but not averaged across seeds.
- **No fine-tuning of T5-small.** W3 absolute level numbers (ROUGE-L
  ~0.18 BM25 → ~0.36 reranked) are modest because the generator is
  pretrained-only; the Δ between paired rows is the load-bearing
  signal. A SFT pass on `(question, gold passage, answer)` triples is
  in scope for a future week.
- **Surface-form generation metrics only at W3.** The W6 BERTScore
  proxy (semantic-similarity) and the W7 grounding metrics
  (extractiveness) partially close this gap, but the canonical
  `roberta-large` BERTScore on full dev/small is still open.
- **NLI-grounding sign flip on the rerank arm (W7-A) is unresolved.**
  Direction reversal is statistically robust (CI strictly above zero
  in magnitude); the mechanism is hypothesised — not yet confirmed —
  to be the fragmentary-output artefact rather than a true semantic
  regression. A generator swap is needed to disentangle.
- **No fine-tuning anywhere in the pipeline.** Every model is used as
  a generic pretrained checkpoint; no SFT pass on `(question, gold
  passage, answer)` triples has been done. This bounds how far the
  surface-form / semantic deltas can go.

## Next steps

The W6 closure + W7 grounding audit together re-aim the remaining
follow-up work away from the decode budget and toward the generator
itself and the prompt format:

- **W7-B — generator capacity, not decode budget.** The closure run
  + the 99 % extractiveness ceiling rule out the `max_new_tokens =
  128` intervention. The candidate axes are: (i) swap T5-small for a
  larger or instruction-tuned variant on the same prompts (T5-base
  is the planned first step under the reduced-scope ddl); (ii) move
  off the `question: ... context: ...` extractive-QA shape toward a
  prompt that demands multi-passage synthesis or explicit citation;
  (iii) light SFT on `(question, gold passage, answer)` triples —
  cheap to try, but the W7 ceiling suggests gains will be bounded by
  the prompt shape, not by lack of supervision.
- **Resolve the W7-A NLI sign flip.** The reranked arm's NLI Δ is
  the only metric reversing direction in the whole project; the
  current best guess is that fragmentary midword outputs score low
  on a sentence-level cross-encoder even when lexically grounded.
  A T5-base swap on the same prompts would disambiguate "generator
  fragment artefact" from "true semantic regression".
- **Citation-grade BERTScore.** Re-run the W6 proxy with
  `roberta-large` on full dev/small (~hours of CPU). The proxy and
  the surface-form deltas agree at the same magnitude, so this is a
  paper-writing requirement, not a research question.
- **W4 follow-ups.** *W4-B*: head-to-head on the same 50 k qrels-
  anchored sample for `bge-small-en-v1.5`, `all-MiniLM-L12-v2`, and
  the existing `all-MiniLM-L6-v2` baseline (the L12 run is on disk;
  BGE is queued). *W4-A*: relevant-doc-density sensitivity (5 % /
  10 % only, per ddl scope) with the W4-B winner.
- **W5-B — K ∈ {50, 100, 200} sweep** on both first stages for the
  performance–latency Pareto. Subsample, per ddl scope.
- **Single-query demo CLI** (polish tier). The pipeline is currently
  batch-eval-oriented; a thin `python -m src.demo.ask "<question>"`
  wrapper that bundles the doc_id → passage map would make the
  end-to-end system approachable without running a full retrieval
  sweep.

## How this maps to the repo

- Conclusions per week: `reports/generated/weekNN_*.pdf`
- Process logs per week: `notebooks/weekNN_*.ipynb`
- Reproducible pipeline: `experiments/run_*.py` + `src/` + `configs/baseline.yaml`
- Reproduce locally: `make` targets (see `Makefile`); outputs land in
  `outputs/` (not committed).

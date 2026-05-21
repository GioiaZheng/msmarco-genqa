# MS MARCO GenQA

## TL;DR

A reproducible, single-machine end-to-end MS MARCO retrieval-augmented QA
pipeline built up across six weekly stages:

> **W1 EDA → W2 BM25 retrieval → W3 RAG generation (T5-small) → W4 dense
> retrieval (SBERT + FAISS) → W5 cross-encoder reranking → W6 semantic-proxy
> evaluation + regression-failure taxonomy.**

**Headline result.** Swapping the first-stage retriever from BM25 to a
cross-encoder reranked dense top-3 — same T5-small generator, same 6 980
dev/small queries — roughly **doubles every generation metric**: Token-F1
**0.197 → 0.368** (Δ +0.171), ROUGE-L **0.193 → 0.368** (Δ +0.174), with
95 % paired-bootstrap CIs strictly above 0 on all four surface-form
metrics. A DistilBERT-based BERTScore proxy on a 3 000-pair subsample
recovers the same Δ (+0.173), so the gain is **not** a surface-form
artefact. A 40-query triage of the regression bucket shows ~90 % of the
remaining regressions surface as *mid-word / short generation-side
output*, not retrieval or semantic failures. A follow-up
`max_new_tokens=64→128` sweep on full dev/small **falsifies** the
budget-cap reading (truncation share 90 % → 87.5 %; all four
surface-form deltas move by <0.005) — the bottleneck is generator
capacity / output style, not decode budget. See *Week 6 — Evaluation
layer* §*Closure* below.

**How to inspect.** The full statistical write-up lives in
[§1 Status](#1-status) below (each weekly stage is self-contained with a
headline number + script + report pointer); the BM25-vs-reranked
generation comparison is consolidated in *Generation × retrieval source*.
PDF per-week reports live in `reports/generated/*.pdf`.

**How to reproduce.** `make install`, then run the per-week scripts in
[§4 Run the official baselines](#4-run-the-official-baselines). The W3
full-dev comparison + W6 evaluation overlay are a single block under
*Generation × retrieval source*. The repo is **batch-eval oriented** by
design — there is no one-shot `--question` CLI; [§4.5](#45-single-query-demo)
shows the minimal in-Python composition for an honest single-query
sanity check.

## Project layout

The repo has two parallel tracks:

- **Script pipeline** (`experiments/`, `src/`) — reproducible, official-corpus, structured outputs + auto-generated reports. **This is the source of truth for benchmark numbers.**
- **Notebooks** (`notebooks/`) — small prototype experiments on hand-written toy corpora or sampled data. They exist for narrative + visualization + smoke-checking the API. **Numbers in notebooks are illustrative only — do not cite them as benchmarks.** Every notebook has a "Limitations" section that says so explicitly, and points at the equivalent script for the honest result.

## 1. Status

### Week 1 — EDA &nbsp;&nbsp;✅ done (notebook only)

- Notebook: [`notebooks/week01_eda.ipynb`](notebooks/week01_eda.ipynb) — runs end-to-end
- Report: `python -m src.reporting.build_report --week week01` →
  [`reports/generated/week01_eda.md`](reports/generated/) *(gitignored)*

### Week 2 — BM25 retrieval &nbsp;&nbsp;✅ done

- Script: [`experiments/run_retrieval.py`](experiments/run_retrieval.py)
- **MRR@10 = 0.1703** &nbsp;·&nbsp; **Recall@100 = 0.6212** &nbsp;·&nbsp; **Recall@1000 = 0.8154**
  on `dev/small` (6,980 queries, full 8.8M-passage corpus)
- Prototype notebook: [`notebooks/week02_retrieval.ipynb`](notebooks/week02_retrieval.ipynb)
  — sampled closed-set (MRR@10 = 0.1956, n=30, structurally optimistic; not a benchmark)

### Week 3 — RAG generation &nbsp;&nbsp;✅ done

- Script: [`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py)
- **Sampled baseline** — 200-query sample of dev/small (seed 42), T5-small (no fine-tuning),
  top-3 BM25 passages from the W2 run, best-of-N reference scoring:
  - **ROUGE-L = 0.1626** &nbsp;·&nbsp; **BLEU = 0.0574** &nbsp;·&nbsp; **EM = 0.0050** &nbsp;·&nbsp; **Token-F1 = 0.1756**
  - Run config: [`configs/baseline.yaml`](configs/baseline.yaml) · Command: `python experiments/run_generation_baseline.py` ·
    Manifest: `outputs/week03_generation/manifest.json` *(gitignored; regenerated each run)*
  - **Not** a full dev/small benchmark — 200 / 6,980 queries, CPU-friendly.
- Prototype notebook: [`notebooks/week03_generation.ipynb`](notebooks/week03_generation.ipynb)
  — 3-passage toy demo with T5-small (smoke test, not a benchmark)
- A full-dev comparison of this same generator against a *reranked* upstream
  retriever lives in [*Generation × retrieval source*](#generation--retrieval-source--done)
  below — that is the headline result, not this 200-query subsample.

### Week 4 — Dense retrieval (sampled) &nbsp;&nbsp;✅ done

- Script: [`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py)
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP` over
  L2-normalised embeddings, qrels-anchored 50k-passage sample.
- **Dense MRR@10 = 0.8830** &nbsp;·&nbsp; **nDCG@10 = 0.9041** &nbsp;·&nbsp; **Recall@100 = 0.9946**
  vs **BM25-on-sample MRR@10 = 0.6948** &nbsp;·&nbsp; **Recall@100 = 0.9338** (same 50k pool).
- Numbers are **upper-bounded** by qrels-anchoring (every dev relevant doc is
  in the pool by construction). The valid comparison is *dense vs BM25 on the
  same sample*, not against the W2 full-corpus number.

### Week 5 — Cross-encoder reranking &nbsp;&nbsp;✅ done

- Script: [`experiments/run_reranker.py`](experiments/run_reranker.py)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the W4 dense top-100.
- **Full dev/small (6 980 queries)** — `outputs/week05_reranker_full/` (gitignored):

  | Metric      | Dense (W4) | + CE rerank | Δ          |
  |-------------|-----------:|------------:|-----------:|
  | MRR@10      | 0.8830     | 0.9304      | **+0.0474** |
  | nDCG@10     | 0.9041     | 0.9434      | +0.0393    |
  | Recall@100  | 0.9946     | 0.9946      | +0.0000    |

  Runtime ~4 h 37 min on a 6-core MacBook (538 000 (query, passage) pairs
  at ~32 pairs/s, peak RSS ~3.3 GiB). Recall@100 is unchanged by
  construction — reranking is order-only.
- **Sanity check vs the 1 000-query CPU subsample run.** An earlier
  pilot on a 1 000-query subsample of dev/small produced
  MRR Δ +0.0435, nDCG Δ +0.0398 (level: 0.8846 → 0.9282 / 0.9014 → 0.9412).
  The full-dev deltas (+0.0474 / +0.0393) sit on top of the subsample
  deltas with a similar magnitude, so the subsample was not biased — the
  reranker gain is a property of dev/small, not of the 1 000-query slice.
- Prototype notebook: [`notebooks/week05_reranker.ipynb`](notebooks/week05_reranker.ipynb)
  — 8-passage toy demo showing the score-margin sharpening (bi-encoder gap
  ~0.08 → cross-encoder gap ~7).
- Narrative: W4 closed the recall gap (semantic matching). W5 closes the
  *local-ordering* gap — once the relevant passage is somewhere in the top-100,
  the cross-encoder is what pushes it to top-1 / top-3.

### Generation × retrieval source &nbsp;&nbsp;✅ done

*Cross-week synthesis (not a new stage): uses the W2 BM25 run.tsv and the
W5 reranked run.tsv as inputs and produces the full-dev paired
predictions that W6's evaluation layer scores below.*

Does feeding reranked top-K back into the T5-small generator actually
improve answer quality? Apples-to-apples comparison on **full dev/small
(6 980 queries)**: same T5-small (no fine-tuning), same top-3 passages
— only the upstream retrieval source changes. Both runs are mutually
restricted via `--restrict-to-run` so the two predictions cover the
**exact same 6 980 query ids** (verified by qid set-diff = ∅).

| Retrieval source &rarr; T5-small | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25 &nbsp; *(`outputs/week02_bm25/run.tsv`)*                              | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked &nbsp; *(`outputs/week05_reranker_full/run.tsv`)*                 | **0.3621** | **0.2922** | **0.0606** | **0.3677** |
| **Δ (rerank − BM25)** | **+0.1763** | **+0.2206** | **+0.0471** | **+0.1711** |

**Reranking the first stage roughly doubles every generation metric on
full dev/small, and the 95% paired-bootstrap CI on the per-query Δ
excludes 0 for all four metrics** (see the bootstrap table below).
Cross-encoder reordering of top-100 → top-3 delivers materially better
passages into the generator's context window, and surface-form metrics
pick that up even with a frozen pretrained T5-small. The structural
mechanism is visible in retrieval flags: the rate of having at least
one relevance-judged passage in the top-3 jumps from **20.8 %** (BM25)
to **96.9 %** (reranked) over the 6 980 paired queries.

Per-query token-F1 splits 4 015 strict improvements / 1 766
regressions / 1 199 ties — i.e. **net +2 249 / 6 980 queries (32 %)**
strictly improved by feeding reranked passages, with the regressions
mostly clustered around already-easy queries where BM25 happened to
surface a usable passage.

Qualitative example (qid 1043064, *"what is the chemical formula for
oxygen tetrafluoride?"*): BM25 → *"Li 2 O"*; reranked → *"N2F4"*
(matches the reference).

#### Paired-bootstrap 95% CI on Δ (rerank − BM25)

Full 6 980 paired qids, 10 000 bootstrap resamples, seed 42.
ROUGE-L and BLEU per-query scores here come from `rouge_score.RougeScorer`
and NLTK `sentence_bleu` (smoothing method 1) so the per-query means
differ slightly from the HF corpus-level numbers above; what matters is
that all four CIs lie strictly above 0 with effectively zero overlap.

| Metric            | BM25 (per-query mean) | Reranked | Δ        | 95% CI on Δ           | p₂ (10k resamples) |
|-------------------|----------------------:|---------:|---------:|----------------------:|-------------------:|
| ROUGE-L           | 0.1934                | 0.3677   | +0.1742  | [+0.1663, +0.1820]    | < 0.001            |
| BLEU (sentence)   | 0.0423                | 0.1754   | +0.1330  | [+0.1265, +0.1395]    | < 0.001            |
| Exact-Match       | 0.0135                | 0.0606   | +0.0471  | [+0.0417, +0.0527]    | < 0.001            |
| Token-F1          | 0.1966                | 0.3677   | +0.1711  | [+0.1632, +0.1789]    | < 0.001            |

CIs are ~6× tighter than the earlier 200-query subsample (n=200
vs. n=6 980), and the full-dev Δ point estimates all sit inside the
200-query CIs — the original direction-and-magnitude claim survives
the move to full dev/small intact, with the level numbers naturally
deflating to the harder full benchmark.

#### Bucket analysis by query type (token-F1)

| query_type     |    n |   BM25 |  Reranked |        Δ |
|----------------|-----:|-------:|----------:|---------:|
| DESCRIPTION    | 3725 | 0.1889 |    0.3939 | **+0.2050** |
| ENTITY         |  631 | 0.1765 |    0.3186 | +0.1421  |
| LOCATION       |  498 | 0.2495 |    0.3928 | +0.1433  |
| NUMERIC        | 1665 | 0.1997 |    0.3235 | +0.1238  |
| PERSON         |  461 | 0.2186 |    0.3557 | +0.1371  |

`DESCRIPTION` queries (53 % of the eval set) benefit most from
reranking; `NUMERIC` benefits least — consistent with the intuition
that short numeric answers depend more on lexical surface match in
the passage than on which-of-the-near-duplicates the reranker picks.
Full bucket counts (`rerank_fixed_generation_improved` = 2 184,
`rerank_fixed_generation_still_failing` = 2 684, `regression` = 233,
…) in [`outputs/week06_analysis/summary.json`](outputs/week06_analysis/) (gitignored).

#### Semantic-proxy BERTScore sanity check (3 000-paired subsample)

The four metrics above are all surface-form (overlap, n-gram, exact).
T5-small produces verbose extractive outputs that surface-form metrics
under-credit when reworded, so the load-bearing question is: **does
the +0.17 Token-F1 Δ also show up in a semantic-similarity metric?**

A *low-cost semantic evaluation sanity check on 3 000 paired examples
using DistilBERT-based BERTScore* (rescaled, paired-bootstrap CI; seed
42). DistilBERT, not the canonical `roberta-large` — the absolute
level is a proxy, but the paired Δ between two systems on the same
3 000 qids is what matters and is stable under the encoder choice:

| Metric                          | BM25 (per-q mean) | Reranked | Δ        | 95% CI on Δ           | p₂      |
|---------------------------------|------------------:|---------:|---------:|----------------------:|--------:|
| BERTScore-F1 (DistilBERT proxy) | 0.2192            | 0.3920   | **+0.1728** | [+0.1608, +0.1850]  | < 0.001 |

Per-query: rerank strictly better on **64.8 %** of qids; tie 5.7 %;
strictly worse on 29.5 %.

**This is the load-bearing finding**: the semantic-proxy Δ (+0.1728) is
within a hair of the surface-form Token-F1 Δ (+0.1711) and ROUGE-L Δ
(+0.1742). The reranker improvement is *not* a surface-form artefact;
it shows up in a semantic-similarity scorer at essentially the same
magnitude. This is the result that justifies acting on the surface-
form CIs above without first running a full `roberta-large` BERTScore.

Reproduce: `python scripts/bertscore_paired_eval.py --n-pairs 3000`
(writes `outputs/week06_bertscore_proxy/bertscore_proxy_ci.json`,
gitignored). Caveat: DistilBERT is intentionally not the canonical
BERTScore encoder — for cross-paper comparison, re-run with `--model-
type microsoft/deberta-xlarge-mnli` or `--model-type roberta-large`
on a longer time budget.

#### Regression failure taxonomy (40-query triage of the 233-strong regression bucket)

Of the 6 980 paired qids, 233 land in the `regression` bucket
(reranker brings a relevant passage into top-3, yet token-F1 drops
vs BM25). A 40-query seeded triage with deterministic heuristic
labels (`scripts/regression_failure_taxonomy.py`, seed 42) reveals
that **regressions are dominated by generation-side truncation, not
retrieval failures or semantic drift**:

| label                    |  n | share |
|--------------------------|---:|------:|
| `truncation_midword`     | 22 | 55 %  |
| `truncation_short`       | 14 | 35 %  |
| `topic_drift`            |  2 |  5 %  |
| `extractive_passage_bias`|  2 |  5 %  |
| `semantic_mismatch`      |  0 |  0 %  |

`truncation_midword` (rerank prediction ends without terminal
punctuation on an alphabetic char — initially attributed to the
`max_new_tokens=64` cap; see *Closure* below for the falsification)
and `truncation_short` (≤ 3 tokens; generator extracted only a
title-like fragment) together account for **90 %** of the sample.
Example (qid 49802, query "belizean cuisine"): BM25 → 24-token
description; reranked → "Belizean cuisine" (the title from the
passage). The reranker is doing its job — bringing in richer
passages — but T5-small's tendency to emit short / mid-sentence
answers turns richer context into less overlap-matching output on
this bucket.

Full markdown report with 40 per-example detail rows in
[`outputs/week06_analysis/regression_taxonomy.md`](outputs/week06_analysis/) (gitignored).

Reproduce:

```bash
# 1. Generation on full dev/small (~1 h each on a 6-core CPU; mutually
# restricted to the same qid set):
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

# 2. Four-metric paired bootstrap CI:
python scripts/bootstrap_generation_comparison.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week03_generation_bootstrap_full

# 3. Bucket + retrieval-flag analysis (with token-F1 / EM CIs inline):
python scripts/analyze_generation_rerank.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week06_analysis

# 4. Semantic-proxy BERTScore sanity check on a 3,000-paired subsample
# (~3 min CPU with DistilBERT; pass --n-pairs 0 for full-dev):
python scripts/bertscore_paired_eval.py \
    --n-pairs 3000

# 5. Regression failure taxonomy (40-query triage, deterministic labels):
python scripts/regression_failure_taxonomy.py
```

The earlier 200-query subsample comparison (BM25 0.2131 / Rerank
0.4006 ROUGE-L, Δ +0.1875 with CI [+0.1344, +0.2389]) lives in
`outputs/week03_generation_{bm25,reranked}/` (gitignored). The Δ point
estimates and conclusions are consistent across the two scales; the
full-dev numbers above are the version to cite.

### Week 6 — Evaluation layer &nbsp;&nbsp;✅ done

Two follow-up evaluations layered on top of the full-dev paired
predictions produced by *Generation × retrieval source* above —
neither requires a new model run, both are CPU-only:

- **Semantic-proxy BERTScore** on a 3 000-pair subsample of the 6 980 shared
  qids (script: [`scripts/bertscore_paired_eval.py`](scripts/bertscore_paired_eval.py),
  output: `outputs/week06_bertscore_proxy/bertscore_proxy_ci.json`):
  BM25 0.2192 → Reranked 0.3920, **Δ +0.1728, 95 % paired-bootstrap CI
  [+0.1608, +0.1850], p < 0.001**. Per-query: rerank strictly better
  64.8 %, tie 5.7 %, BM25 strictly better 29.5 %.
  Scorer is **DistilBERT-based BERTScore** (`distilbert-base-uncased`,
  `rescale_with_baseline=True`) — a deliberate proxy for the canonical
  `roberta-large` BERTScore; the proxy answers "is the rerank Δ also
  visible in a semantic-similarity scorer?", *not* "what is the
  citation-grade BERTScore?". The semantic Δ sits within a hair of the
  surface-form Token-F1 Δ (+0.1711) and ROUGE-L Δ (+0.1742), so the
  rerank gain is not a surface-form artefact. Full detail (table,
  per-query split, caveats) under *Semantic-proxy BERTScore sanity check*
  above.
- **Regression failure taxonomy** on a 40-query seeded triage of the
  233-strong regression bucket (queries where rerank brought the
  relevant passage into top-3 but generator F1 *dropped* vs BM25;
  script: [`scripts/regression_failure_taxonomy.py`](scripts/regression_failure_taxonomy.py),
  output: `outputs/week06_analysis/regression_taxonomy.{json,md}`):
  **~90 % of regressions are generation-side truncation, not retrieval
  or semantic failures** (55 % `truncation_midword`, 35 %
  `truncation_short`, 5 % `topic_drift`, 5 % `extractive_passage_bias`,
  0 % `semantic_mismatch`). At the time of the W6 report this pointed
  at a cheap budget-cap intervention; the *Closure* bullet below
  records the falsification of that reading.
- **Decoding-budget closure — `max_new_tokens=64→128` on full dev/small.**
  Same generator, same prompts, same retrieval inputs, same seed;
  only `max_new_tokens` changes (CLI override, canonical 64-token
  outputs untouched). Result: rerank Δ is **statistically and
  practically unchanged** — Token-F1 Δ +0.1706 (CI [+0.163, +0.178]),
  ROUGE-L Δ +0.1736 (CI [+0.166, +0.181]), BLEU Δ +0.1325, EM Δ
  +0.0471, all four CIs strictly above zero. The regression bucket
  shrank only 233 → 231 queries and the truncation share dropped
  90.0 % → **87.5 %** (40-query seeded triage, same rule cascade).
  **The W6 budget-cap hypothesis is falsified**: T5-small is hitting
  EOS naturally on this prompt format, not running out of decode
  budget. The mid-word-ending output style is intrinsic to the model,
  not a symptom of the 64-token cap. Outputs under
  `outputs/week03_generation_{bm25,reranked}_full_mnt128/`,
  `outputs/week06_bootstrap_mnt128/`,
  `outputs/week06_taxonomy_mnt128/` (gitignored). BERTScore proxy on
  the new predictions was **skipped**; it remains optional and can
  be reproduced with one command (see *Reproduce* under
  `reports/templates/week06_eval_layer.md`).
- Per-week report: [`reports/generated/week06_eval_layer.pdf`](reports/generated/week06_eval_layer.pdf)
  (regenerate with `python -m src.reporting.build_report --week week06`).
- **W6 follow-ups — three offline analyses on the existing
  per-query metrics, no new generation / reranker runs:**
  - *W6-A: question-form tagging* of all 6 980 dev/small queries
    (who / what / when / where / why / how / which / yes_no / other;
    complementary to MS MARCO QA's native answer-type `query_type`).
    Script: [`scripts/tag_query_forms.py`](scripts/tag_query_forms.py).
    Output: `outputs/week06_querytype/`.
  - *W6-B: rerank Δ by question-form* — joins W6-A onto the W3/W5/W6
    per-query metrics; reports per-form ΔToken-F1 / ΔROUGE-L / ΔEM /
    ΔMRR@10 / ΔnDCG@10 with paired-bootstrap CIs. **`which` (n=120) is
    the only form whose 95 % CI on ΔToken-F1 includes zero** (Δ = +0.0250,
    CI [−0.028, +0.077], p = 0.33), even though its retrieval-side
    ΔMRR@10 is +0.71 — the reranker is doing its job on retrieval but
    the gain doesn't transfer downstream on selection-style queries.
    Factual wh-forms (what / where / why / how / who) all see Δ in the
    +0.12 to +0.23 range with CIs strictly above zero. Script:
    [`scripts/analyze_rerank_by_query_form.py`](scripts/analyze_rerank_by_query_form.py).
    Output: `outputs/week06_rerank_by_form/`.
  - *W6-C: regression vs non-regression query profile* — Mann-Whitney
    U on five structural features (query length tokens / chars, qrels
    density, BM25 + rerank top-3 mean passage length). Regression
    queries do *not* differ from the rest on the query side at any
    meaningful effect size; the only detectable signal is a small
    downward shift in retrieved passage length on both arms (rerank
    Δmedian −2.7 tokens, p = 0.0073, r = −0.103). Consistent with
    the W6 taxonomy finding that ~90 % of regressions are
    generator-side truncation: shorter passages give T5-small less
    material to extract. Script:
    [`scripts/regression_query_profile.py`](scripts/regression_query_profile.py);
    three boxplots under [`figures/w6c_regression_vs_other_*.png`](figures/).

### Week 7 — Grounding audit &nbsp;&nbsp;✅ done

Cheap deterministic CPU pass over the existing full-dev paired
predictions to answer the question the W6 closure left open: *what
is T5-small actually doing on this prompt format — extracting from
the passages, or generating from parametric memory?* Two metrics in
[`src/evaluation/grounding.py`](src/evaluation/grounding.py), driven
by [`scripts/grounding_audit.py`](scripts/grounding_audit.py); no
new generation, no model load, ~2 s scoring + bootstrap.

- **Lexical content-token grounding** (fraction of unique non-stopword
  prediction tokens that appear anywhere in the prompt's top-3
  passages): BM25 **0.9972** → Reranked **0.9977**, **Δ +0.0005
  (95 % paired-bootstrap CI [−0.0003, +0.0014], p=0.24)**. Both arms
  sit at the lexical ceiling — almost every content word T5-small
  emits is already in the prompt. The rerank advantage on grounding
  at the word level is indistinguishable from zero because there is
  no slack.
- **3-gram grounding** (fraction of the prediction's contiguous
  3-grams that appear as a contiguous span in any single passage):
  BM25 **0.9873** → Reranked **0.9905**, **Δ +0.0032 (95 %
  paired-bootstrap CI [+0.0015, +0.0050], p < 0.001)**. Reranking
  slightly raises phrase-level grounding too, but both arms are
  again ~99 %.
- **Headline read:** T5-small on the `question: ... context: ...`
  prompt is effectively performing **extractive QA**. The W3/W5
  rerank gain (Δ Token-F1 +0.171) is almost entirely downstream of
  retrieval — the reranker puts the right words into the prompt and
  the generator copies them. This is a *calibration* of the earlier
  result, not a contradiction: the W3 surface-form deltas are real,
  but the mechanism is "better passages → better extractive output",
  not "better passages → better neural reasoning".
- **Edge cases reported separately:** ~10 % of predictions per arm
  are <3 tokens (the `truncation_short` / `extractive_passage_bias`
  pattern from the W6 taxonomy), scoring n-gram grounding as 1.0
  vacuously. Lexical-vacuous predictions are negligible (<0.3 %).
  Counts are identical-shape on both arms, so the paired Δ is
  unaffected.
- Outputs:
  [`outputs/week07_grounding/{per_query_grounding.jsonl,summary.json}`](outputs/)
  (gitignored).
- Per-week report:
  [`reports/generated/week07_grounding.pdf`](reports/generated/week07_grounding.pdf)
  (regenerate with `python -m src.reporting.build_report --week week07`).
- **W7-A — NLI-entailment grounding (semantic-faithfulness companion).**
  Scores `entailment(passages → prediction)` with
  `cross-encoder/nli-deberta-v3-small` on a 3 000-paired-qid subsample
  (seed 42, same paired-bootstrap-CI convention as the W6 BERTScore
  proxy). Result: BM25 **0.2270** → Reranked **0.0821**, **Δ −0.1448
  (95 % CI [−0.1597, −0.1297], p < 0.001)**. The NLI Δ is *negative
  and strictly excludes zero* — the only metric in this project whose
  Δ reverses sign vs the W3/W5/W6 surface-form story. Likely mechanism
  (to be tested by a generator swap): T5-small on reranked top-3 emits
  more fragmentary / mid-word-cut snippets that inflate word and
  3-gram overlap (positive lex / ngram / Token-F1) but score low on a
  sentence-level NLI cross-encoder which cannot entail a fragmentary
  hypothesis. Module:
  [`src/evaluation/nli_grounding.py`](src/evaluation/nli_grounding.py);
  driver: the same `grounding_audit.py` script with `--nli-n-pairs 3000`.
- **W7-C — grounding ↔ downstream correlation.** Per-query
  Spearman / Pearson + binned (≥0.9 vs <0.9) Mann-Whitney comparing
  lex / 3-gram / NLI grounding against Token-F1 and a freshly-scored
  DistilBERT BERTScore-F1 (full 6 980 paired qids; NLI cells restricted
  to the 3 000-pair subsample). Every binned cell has Δ(high − low) > 0;
  magnitudes 0.04–0.10 in absolute Token-F1 / BERTScore. Direction is
  consistent across all 12 cells (high grounding → higher downstream);
  magnitude is small because both arms sit near the lex / 3-gram
  ceiling on most queries. BERTScore is cached per-qid for re-use by
  follow-on analyses. Script:
  [`scripts/grounding_correlation.py`](scripts/grounding_correlation.py).
- **W7-D — low-grounding case study (30-query seeded triage).** Of the
  197 rerank-arm queries with `lex_rerank < 0.9` OR `ngram_rerank < 0.9`,
  sampled 30 at seed = 42 and dumped (query, top-3 passages, BM25 +
  rerank predictions, references) with a coarse rule-cascade label.
  **23 / 30 (77 %) `paraphrase_reorder`** (content words present in
  the prompt, order / phrasing differs); **7 / 30 (23 %)
  `partial_external`**, most of which are tokeniser / morphology
  artefacts (`350oF` vs `350°F`; `competed` vs `compete`). **0 / 30
  `parametric_or_external`** (lex < 0.5) — no genuine hallucinations
  in the sample. Reinforces the W7 headline: T5-small on this prompt
  format is performing extractive QA even on its worst-grounded outputs.
  Script:
  [`scripts/low_grounding_case_study.py`](scripts/low_grounding_case_study.py).

### Reference points

- Published Anserini/Lucene BM25 baseline on MS MARCO `dev/small`: MRR@10 ≈ 0.184. Our `bm25s`-based **0.1703** is in the same ballpark; the gap is consistent with tokenizer differences.

## 2. Directory layout

```
configs/        baseline.yaml — paths + retrieval/generation/reranker/eval knobs
experiments/    run_retrieval.py, run_dense_retrieval.py,
                run_generation_baseline.py, run_reranker.py
src/
  data/         msmarco.py — ir_datasets loader for the official corpus
  retrieval/    bm25.py    — bm25s wrapper with save/load + chunked retrieve
                dense.py   — Sentence-Transformers + FAISS dense retriever
                sampling.py — qrels-anchored sub-corpus sampling
  reranking/    cross_encoder.py — Cross-encoder reranker wrapper (W5)
                io.py            — TREC run.tsv read/truncate/write helpers
  generation/   rag_generator.py — T5/BART RAG generator
  evaluation/   retrieval.py (MRR/Recall/nDCG), generation.py (ROUGE/BLEU/EM/F1)
  reporting/    build_report.py — fills markdown templates from outputs/
notebooks/      prototype, narrative, plots
reports/
  templates/    week02_bm25.md, week03_generation.md  (committed)
  generated/    filled-in markdown + optional PDF     (gitignored)
outputs/        run.tsv, metrics.json, examples.jsonl per week  (gitignored)
data/           raw/, processed/, cache/ — all gitignored, .gitkeep tracked
figures/        plots from notebooks (committed)
scripts/        smoke tests + the notebook regenerator
```

Everything runs from the project root. Scripts add `PROJECT_ROOT` to `sys.path` themselves; no `PYTHONPATH` needed.

## 3. Setup

Python 3.10+ recommended (3.9 also works).

```bash
# Fast development install — loose lower bounds, latest compatible versions.
pip install -r requirements.txt
pip install -e .                       # register `src` as a real package
```

For an environment that reproduces the numbers checked into the reports,
pin to the lockfile instead:

```bash
pip install -r requirements-lock.txt   # exact versions, no upgrades
pip install -e .
```

Or, equivalently:

```bash
make install                           # uses requirements.txt + editable install
```

Optional, only needed for PDF report generation (markdown reports work without):

```bash
brew install pandoc                    # macOS
brew install --cask basictex           # macOS LaTeX engine
# Linux: sudo apt-get install pandoc texlive-xetex
```

## 4. Run the official baselines

### Week 2 — BM25 retrieval

```bash
python experiments/run_retrieval.py
python -m src.reporting.build_report --week week02
```

First run: ~5 min download (~1 GB), ~15 min `ir_datasets` encoding fix pass, ~10 min `bm25s` index build, ~70 min retrieve. Total ≈ 1h40m on a 16 GB MacBook.

Subsequent runs reuse the cached index (`data/processed/bm25_index_msmarco/`, 2.1 GB) and skip download/index — only retrieve runs (~70 min).

If a run is killed mid-retrieve:

```bash
python experiments/run_retrieval.py --resume     # picks up at next chunk boundary
```

To force a fresh index:

```bash
python experiments/run_retrieval.py --rebuild-index
```

Outputs:
- `outputs/week02_bm25/metrics.json`
- `outputs/week02_bm25/run.tsv` (TREC-format top-1000, ~250 MB)
- `outputs/week02_bm25/examples.jsonl`
- `reports/generated/week02_bm25.md` (+ `.pdf` if pandoc installed)

### Week 3 — RAG generation baseline

Requires Week 2 to have produced `outputs/week02_bm25/run.tsv`.

```bash
python experiments/run_generation_baseline.py
python -m src.reporting.build_report --week week03
```

Default: 200 dev queries, T5-small, top-3 passages from BM25. ROUGE-L / BLEU / EM / Token-F1 against MS MARCO QA v2.1 reference answers. CPU runtime ~5–15 min.

Tunable knobs in `configs/baseline.yaml`:
- `generation.model_name` (`t5-small`, `t5-base`, `facebook/bart-base`, …)
- `generation.num_eval_queries`
- `generation.top_k_passages`

The runner is **retrieval-source agnostic** — feed it any TREC-format
`run.tsv` (BM25 / dense / reranked) via the CLI flags. Defaults preserve
the W3 BM25 baseline behaviour exactly; explicit flags pick a different
upstream:

```bash
# Reranked → T5-small, on the queries the reranker actually covers
python experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker/run.tsv \
    --output-dir outputs/week03_generation_reranked \
    --retrieval-source reranked
```

Use `--restrict-to-run <other_run.tsv>` to force two generation runs to
evaluate on the SAME 200-query subsample even when their upstream
retrievers cover different query sets — see the "Generation × retrieval
source" section above for the BM25-vs-reranked comparison this enables.

### Week 4 — Dense retrieval (sampled corpus)

Requires Week 2 to have produced `data/processed/bm25_index_msmarco/doc_ids.json`
(the doc_id pool the qrels-anchored sampler draws from).

```bash
python experiments/run_dense_retrieval.py
python -m src.reporting.build_report --week week04
```

First run: ~13.5 min to encode the 50k sampled passages (810 s on a 6-core
MacBook CPU, batch 32; `all-MiniLM-L12-v2` takes ~26 min), ~20 s for FAISS search.
Subsequent runs reuse the cached FAISS index. Tunable knobs:
- `dense.model_name` (e.g. `sentence-transformers/all-MiniLM-L6-v2`,
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5`)
- `dense.sample_size` (default 50000; grows the pool, shrinks recall)
- `dense.compare_bm25_on_sample` (head-to-head on the same sample)

### Week 5 — Cross-encoder reranking

Requires Week 4 to have produced `outputs/week04_dense/run.tsv`.

```bash
python experiments/run_reranker.py
python -m src.reporting.build_report --week week05
```

Default: rerank the top-100 dense candidates per query with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. CPU-only runtime scales linearly with
the number of queries — full 6,980 queries × top-100 is ~6 hours on a 6-core
laptop. Use `--num-eval-queries` for a deterministic subsample.

```bash
# fast smoke test (~1 min)
python experiments/run_reranker.py --num-eval-queries 50 --rerank-top-k 100
# canonical baseline (~50 min on CPU with batch 128, OMP=12)
OMP_NUM_THREADS=12 python experiments/run_reranker.py --num-eval-queries 1000
```

Tunable knobs in `configs/baseline.yaml` under `reranker:`:
- `reranker.model_name` (any HF cross-encoder)
- `reranker.rerank_top_k` (depth; cost is O(K))
- `reranker.batch_size`, `reranker.max_length`

## 4.5. Single-query demo

The pipeline above is **batch-eval oriented by design**: every entrypoint
in `experiments/` consumes a retrieval run (`run.tsv`) over the full dev
set and emits metrics + paired predictions. There is intentionally no
`--question "..."` CLI wrapper — the project optimises for reproducible
benchmark numbers, not for serving one-off queries. The block below is
the smallest in-Python composition for an honest "ask one question"
sanity check; it is **not** a shipped CLI command.

### A. Generator-only single-shot (runs as written)

The generator has a single-shot `generate(query, passages)` method. With
`pip install -e .` in place, this block runs end-to-end on CPU in
roughly the time it takes to download T5-small:

```python
# Real, runnable as-is — same generator config the W3 baseline uses
# (T5-small, top-3 passages folded into the prompt, max_new_tokens=64).
from src.generation.rag_generator import RAGGenerationConfig, RAGGenerator

gen = RAGGenerator(RAGGenerationConfig())
answer = gen.generate(
    query="what is bm25",
    passages=[
        "BM25 is a probabilistic ranking function used by search engines "
        "to estimate the relevance of documents to a given search query.",
        "BM25 was developed in the 1970s and 1980s as part of the Okapi "
        "information retrieval system at City University, London.",
        "Unlike plain TF-IDF, BM25 saturates term frequency and normalises "
        "for document length.",
    ],
)
print(answer)
```

This is honest about what the project ships: a generator the user feeds
passages to. The passages here are hand-written, not retrieved — so the
answer reflects generator behaviour, not the end-to-end pipeline.

### B. Composition sketch: retrieve + generate (not packaged)

To put real retrieval in front of the generator, load one of the
on-disk indexes the batch scripts use, then map the returned `doc_ids`
back to passage text. There is no shipped helper — the snippet below is
the **minimal composition pattern**, not a runnable copy-paste:

```python
# Sketch — assumes W2 BM25 index already built and a doc_id → passage map
# is available. The batch runners build that map from ir_datasets; pulling
# it out into a standalone helper is a TODO (see §8 Next, src.demo.ask).
from src.retrieval.bm25 import BM25Retriever

bm25 = BM25Retriever.load("data/processed/bm25_index_msmarco")
scores, doc_ids = bm25.retrieve("what is bm25", k=3)
passages = [passage_by_id[d] for d in doc_ids]  # supply your own map
answer = gen.generate(query="what is bm25", passages=passages)
```

`BM25Retriever.retrieve` (and `DenseRetriever.retrieve` over the W4 dense
index) both return `(scores, doc_ids)`; resolving `doc_ids` to passage
text is the missing piece for a one-command demo. A thin
`python -m src.demo.ask "<question>"` wrapper that bundles that mapping
is listed under §8 *Next*.

## 5. Run the prototype notebooks

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/week01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/week02_retrieval.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/week03_generation.ipynb
```

First run downloads:
- HuggingFace `ms_marco` v2.1 validation split (~500 MB) — used by W1, W2
- T5-small (~250 MB) — used by W3

Notebook results are **prototypes**: small samples, closed-set retrieval, hand-written eval queries. Do not cite as benchmark numbers.

## 6. Configuration

All knobs live in [configs/baseline.yaml](configs/baseline.yaml). Key ones:

| Key | Effect |
|-----|--------|
| `retrieval.k1`, `retrieval.b` | BM25 hyperparameters |
| `retrieval.top_k` | Depth of saved run (default 1000) |
| `retrieval.chunk_size` | Checkpoint cadence in queries (default 200). Smaller = more durable, more I/O. |
| `retrieval.n_threads` | `0` = sequential (default, matches our 0.1703 baseline). `-1` = all CPUs. Try `-1` for new runs. |
| `data.corpus_limit` | Set to a small int for a smoke test that does **not** reproduce official numbers; leave `null` for real runs. |
| `generation.num_eval_queries` | Size of the W3 eval subset |

## 6.5. Reproducibility status

Current state of the engineering scaffold (what works today; what's still TODO).

| Area | Status | Notes |
|---|---|---|
| Default unit tests | ✅ | `make test` / `pytest -q` — 78 tests, no network, no heavy deps. Slow tests excluded by `[tool.pytest.ini_options]`. |
| Slow tests | ✅ (skips gracefully) | `make test-slow` includes `@pytest.mark.slow`. HF metric scripts skip if unavailable; never hard-fail offline. |
| Lockfile | ✅ basic | `requirements-lock.txt` is pip-freeze-style; sub-dep transitive closure + hash pinning are TODO (would need pip-tools / uv). |
| Installable package | ✅ basic | `pip install -e .` registers `src` via `pyproject.toml`. Existing `sys.path.insert` shims in `experiments/` and `scripts/` are kept for now to avoid touching unrelated code; removing them is a TODO. |
| CI | ✅ basic | `.github/workflows/ci.yml`: pytest + ruff on push/PR to main. Does not run slow tests or download MS MARCO data. |
| Lint | ✅ minimal | `ruff` with `F` + `W` (pyflakes + whitespace). Style rules (`E`, `I`, `UP`, …) intentionally OFF on the first pass. |
| Artifact manifest | ✅ wired | `src/util/manifest.py` provides `build_manifest()` / `write_manifest()` / `write_run_manifest()`. All 4 runners write `outputs/<week>/manifest.json` alongside `metrics.json`. Captures git commit + dirty flag, command, config hash, dependency-file hashes (requirements / lockfile / pyproject), and per-output sha256 (truncated). |
| Numbers in `reports/generated/*.pdf` | ⚠️ historical | Reflect the dev environment at the time the PDF was committed. Re-running with `requirements-lock.txt` is the closest we get to reproduction today. |

Current limitations to be aware of:

- The lockfile reflects the author's **macOS CPU-only** dev environment. Linux / CUDA may resolve different versions; install `torch` from the appropriate PyTorch index *first*.
- The corpus, encoder, and reranker checkpoints are downloaded by `ir_datasets` / HuggingFace at first run and **are not checksummed by the project**. If upstream changes silently, numbers may shift.
- `experiments/run_*.py` still rely on `sys.path.insert(0, PROJECT_ROOT)` at the top of the file. `pip install -e .` makes this unnecessary, but the shim is kept until a future pass removes them.

## 7. Known limitations

- **Tokenizer mismatch with Anserini.** Our 0.1703 vs reference 0.184 is mostly tokenizer-induced (`bm25s` default tokenizer ≠ Lucene `EnglishAnalyzer`). Acceptable for a single-machine pure-Python pipeline.
- **CPU-only retrieve is slow at 8.8M docs.** ~70 min for 6,980 queries. `n_threads=-1` may help; not yet benchmarked on this corpus.
- **Generation: pretrained T5-small, no fine-tuning.** Numbers will be low on overlap-based metrics. Fine-tuning is in scope for a future week.
- **NumPy 2.x runtime warning.** Some compiled deps (torch) were built against NumPy 1.x. Cosmetic on this codebase; downgrade to `numpy<2` if it ever causes a real failure.

## 8. Next

- **W5 follow-ups (in progress / queued).**
  *W5-A*: rerank the W2 full-corpus BM25 top-100 with the same
  `ms-marco-MiniLM-L-6-v2` and compare ΔMRR@10 / ΔnDCG@10 head-to-head
  with the existing W5 dense+rerank. The cleaner question: does the
  cross-encoder *recover more* from a weaker first stage? Driver:
  the existing `experiments/run_reranker.py` with
  `--input-run outputs/week02_bm25/run.tsv`; summary table:
  [`scripts/compare_rerank_first_stages.py`](scripts/compare_rerank_first_stages.py).
  *W5-B*: K ∈ {50, 100, 200} sweep on both first stages for the
  performance–latency Pareto.
- **W4 follow-ups.** *W4-B*: head-to-head on the same 50 k
  qrels-anchored sample for `bge-small-en-v1.5`, `all-MiniLM-L12-v2`,
  and the existing `all-MiniLM-L6-v2` baseline — pick the best
  same-tier general-purpose encoder before scaling. *W4-A*: relevant-
  document-density sensitivity (1 % / 5 % / 10 %) with the W4-B winner
  vs BM25 on the same sample, to track how the BM25 ↔ dense gap shifts
  as the pool dilutes.
- **W7-B — generator capacity, not decode budget.** The W6 closure
  closure (`max_new_tokens=64→128`, full dev/small) plus the W7
  ceiling (~99 % extractiveness on both arms) together imply
  generator-side work should target either *richer prompt formats
  that demand reasoning* (multi-passage synthesis, citation-aware
  decoding) or *a different model* — not the decode budget. The
  current `question: ... context: ...` shape is fundamentally an
  extractive-QA prompt; T5-small is doing what the prompt asks.
- **Try a MS-MARCO-tuned dense encoder.** Swap
  `sentence-transformers/all-MiniLM-L6-v2` for
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5` (same architecture,
  domain-tuned weights) and re-run W4 on the same qrels-anchored 50 k
  sample. Measures how much of the current dense vs BM25 gap is
  attributable to generic-encoder choice vs the retrieval setup.
- **Add a simple single-query demo CLI** (optional, polish-tier). The
  repo is currently batch-eval oriented — see §4.5 for the minimal
  in-Python composition. A thin `python -m src.demo.ask "<question>"`
  wrapper around it would make the pipeline approachable for non-eval
  users.

## 9. License

See [LICENSE](LICENSE).

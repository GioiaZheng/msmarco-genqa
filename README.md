# MS MARCO GenQA

[![CI status](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/ci.yml?query=branch%3Amain)
[![Secret scan](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/secret-scan.yml/badge.svg?branch=main)](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/secret-scan.yml?query=branch%3Amain)
[![CodeQL](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/GioiaZheng/msmarco-genqa/actions/workflows/codeql.yml?query=branch%3Amain)
[![Latest release](https://img.shields.io/github/v/release/GioiaZheng/msmarco-genqa?display_name=tag)](https://github.com/GioiaZheng/msmarco-genqa/releases/latest)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-benchmark%20runs-FFD21E)](https://huggingface.co/datasets/GioiaZheng/msmarco-genqa-benchmark-runs)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://github.com/GioiaZheng/msmarco-genqa/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/GioiaZheng/msmarco-genqa)](https://github.com/GioiaZheng/msmarco-genqa/blob/main/LICENSE)

## TL;DR

This repository is a reproducible research-engineering implementation of an
MS MARCO retrieval-augmented QA pipeline. It moves from lexical retrieval to
dense retrieval, cross-encoder reranking, generation, statistical evaluation,
and grounding analysis on the full `dev/small` split (6,980 queries).

**Main result.** Replacing BM25 top-3 passages with cross-encoder-reranked
dense top-3 passages — same T5-small generator, same paired query set —
increases Token-F1 from 0.197 to 0.368 (Δ +0.171) and ROUGE-L from 0.193 to
0.368 (Δ +0.174). Paired-bootstrap 95% CIs are strictly above zero across all
surface metrics, and a DistilBERT BERTScore proxy recovers a similar lift
magnitude (Δ +0.173).

**Scope.** The repo is batch-evaluation oriented. It includes experiment
manifests, config-driven runners, query-level diagnostics, CI checks, report
artifacts, and reproducibility notes alongside the code. Detailed experiment
notes live in [`docs/experiments.md`](docs/experiments.md); the result summary
is [`RESULTS.md`](RESULTS.md); reproducibility entry points are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); the current repository report is
[`reports/repo_report/report.pdf`](reports/repo_report/report.pdf), with
[`report.html`](reports/repo_report/report.html) kept alongside it; the compact
ACL-style findings write-up is
[`reports/acl_findings/report.pdf`](reports/acl_findings/report.pdf).

The 14 exact TREC-DL, BEIR cross-domain, and NFCorpus video-query ranked
runs are also published as a
[Hugging Face Dataset](https://huggingface.co/datasets/GioiaZheng/msmarco-genqa-benchmark-runs),
with a normalized Parquet preview, raw TREC-format outputs, checksum
manifests, and the original Release archives. It contains ranked identifiers,
ranks, and scores only; benchmark text and qrels are not redistributed.

## Results at a glance

| Question | Comparison | Result |
|---|---|---:|
| Does dense retrieval beat lexical retrieval on the same pool? | BM25-on-sample vs SBERT + FAISS | MRR@10 0.6948 → 0.8830 |
| Does reranking add value after dense retrieval? | Dense top-100 vs cross-encoder reranked top-100 | MRR@10 0.8830 → 0.9304 |
| Does retrieval lift transfer to generation? | BM25 top-3 → T5-small vs reranked top-3 → T5-small | Token-F1 0.1966 → 0.3677 |
| Is the generation lift statistically reliable? | 6,980 paired qids, 10,000 bootstrap resamples | ΔToken-F1 +0.1711, 95% CI [+0.1632, +0.1789] |
| Does the unchanged reranker help on external corpora? | BEIR NFCorpus / SciFact BM25 top-100 → CE rerank | MRR@10 +9.18% / +3.25% relative |

## Evidence status

| Status | What can be claimed | Evidence and boundary |
|---|---|---|
| Validated result | On MS MARCO `dev/small`, reranked dense top-3 improves T5-small surface metrics over BM25 top-3 on 6,980 paired queries. | [`RESULTS.md`](RESULTS.md) reports the paired metrics and confidence intervals. Dense retrieval and reranking use the documented 50k qrels-anchored candidate pool, not a full-corpus dense first stage. |
| Validated external retrieval benchmark | On TREC-DL 2019 and 2020, cross-encoder reranking improves MRR@10 and graded nDCG@10 over a full-corpus BM25 first stage on all 43 and 54 judged topics. | [`docs/trec_dl_external_validity.md`](docs/trec_dl_external_validity.md) records the run commit, models, candidate depths, runtimes, independent `ir-measures` cross-check, and links to the checked artifact. This is retrieval evidence, not generation evidence. |
| Validated external-domain retrieval benchmark | On BEIR NFCorpus and SciFact, the unchanged cross-encoder improves MRR@10 and nDCG@10 over BM25 on all 323 and 300 test queries. | [`docs/cross_domain_benchmarks.md`](docs/cross_domain_benchmarks.md) records separate corpora, full query coverage, candidate depth, CPU runtimes, manifests, and an independent `ir-measures` cross-check. This supports transfer on two retrieval collections, not broad cross-domain RAG or generation generalization. |
| Validated first-stage error analysis | The NFCorpus/SciFact comparison separates fixed top-100 candidate absence, depth-recoverable misses, residual top-1000 misses, and query/dataset effects. | [`docs/cross_dataset_error_analysis.md`](docs/cross_dataset_error_analysis.md) shows that NFCorpus has the stronger first-stage ceiling, while SciFact has much healthier candidate coverage. [`docs/scifact_failure_review.md`](docs/scifact_failure_review.md) reviews the 35 SciFact residual no-hit@100 cases. This is retrieval-only diagnosis, not a new retrieval method or generation result. |
| Validated bounded query-representation result | On the official 102-query NFCorpus test/video subset, adding the bounded official description to the BEIR title raises BM25 Recall@100 from 0.2821 to 0.3700 under a fixed index and model configuration. | [`docs/reports/2026-07-28-nfcorpus-video-query-representation.md`](docs/reports/2026-07-28-nfcorpus-video-query-representation.md) reports the predeclared cohort, paired-bootstrap interval, no-hit transitions, and exact-output artifact. This is subset-specific retrieval evidence, not a full-test or generation result. |
| Implemented, evaluation pending | The T5-base generator-capacity sweep driver and configurable alternative-generator paths exist. | [`scripts/run_generator_capacity_sweep.py`](scripts/run_generator_capacity_sweep.py) and the Phase A protocol are implemented, but no T5-base or FLAN-T5 headline result is claimed until a complete versioned run lands. |
| Not established | Retrieval gains transfer to grounded generation on the external benchmarks; dense retrieval beats BM25 under a fair full-corpus candidate condition; or the findings generalize beyond the two evaluated BEIR collections. | These remain research questions, not conclusions of the checked artifacts. |

## Implemented components

| Area | What is included |
|---|---|
| Retrieval | BM25, dense SBERT/FAISS, BM25-on-sample comparisons, full-corpus BM25 runner support for the judged TREC-DL 2019/2020 passage tracks, and BEIR NFCorpus/SciFact adapters. |
| Reranking | Cross-encoder reranking with dataset-selectable TREC-DL and BEIR runners, aggregate lift, first-stage comparison, and query-level promoted/demoted/new-hit/lost-hit diagnostics. |
| Generation | Paired BM25-vs-reranked generation runs using the same generator, prompt format, query set, and top-k depth. |
| Evaluation | Paired-bootstrap confidence intervals, BERTScore proxy checks, grounding audit, RAG triad reporting, query-form slicing, and regression taxonomy. |
| Reproducibility | Config-driven runners, manifests, output hashes, metadata, CI, report artifacts, and optional experiment tracking. |

## Reports and notes

The repository includes runnable code plus written analysis artifacts:

- [`reports/repo_report/report.pdf`](reports/repo_report/report.pdf) / [`report.html`](reports/repo_report/report.html) — repository report with the current engineering surface and historical experiment results.
- [`reports/acl_findings/report.pdf`](reports/acl_findings/report.pdf) — compact ACL-style experimental findings report.
- [`RESULTS.md`](RESULTS.md) — headline metrics, statistical intervals, and interpretation limits.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — setup, checks, run artifacts, and reproduction commands.
- [`docs/experiments.md`](docs/experiments.md) — stage-by-stage experiment narrative and caveats.
- [`docs/architecture.md`](docs/architecture.md) — module boundaries and artifact flow.
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — reproducible evaluation contract.
- [`docs/failure_taxonomy.md`](docs/failure_taxonomy.md) — regression and grounding error taxonomy.
- [`docs/retrieval_quality_reporting.md`](docs/retrieval_quality_reporting.md) — run-level retrieval metrics and matched-qid comparison reports.
- [`docs/retrieval_lift_analysis.md`](docs/retrieval_lift_analysis.md) — query-level reranker lift analysis protocol.
- [`docs/nfcorpus_first_stage_error_analysis.md`](docs/nfcorpus_first_stage_error_analysis.md) — fixed-output NFCorpus candidate-set coverage diagnosis.
- [`docs/nfcorpus_first_stage_taxonomy_review.md`](docs/nfcorpus_first_stage_taxonomy_review.md) — complete 72-query review of NFCorpus source-context and first-stage failures.
- [`docs/scifact_first_stage_error_analysis.md`](docs/scifact_first_stage_error_analysis.md) — fixed-output SciFact candidate-set coverage diagnosis and NFCorpus comparison.
- [`docs/scifact_failure_review.md`](docs/scifact_failure_review.md) — bounded review of the 35 residual SciFact first-stage no-hit@100 cases.
- [`docs/cross_dataset_error_analysis.md`](docs/cross_dataset_error_analysis.md) — retrieval-only NFCorpus/SciFact comparison of candidate-set, ranking-depth, and dataset/query-form effects.
- [`docs/reports/2026-07-28-nfcorpus-video-query-representation.md`](docs/reports/2026-07-28-nfcorpus-video-query-representation.md) — controlled 102-query test of bounded NFCorpus source context.
- [`docs/context_packing.md`](docs/context_packing.md) — prompt compression, provenance, and packed-vs-plain generation comparison.
- [`docs/rag_triad_evaluation.md`](docs/rag_triad_evaluation.md) — context relevance, groundedness, and answer relevance report protocol.
- [`docs/input_validation.md`](docs/input_validation.md) — run-file, JSONL, prompt, and serving input validation contract.
- [`docs/artifact_versioning.md`](docs/artifact_versioning.md) — policy for source data, derived indexes, model outputs, and Git-tracked pointers.
- [`docs/artifact_registry.md`](docs/artifact_registry.md) — canonical result-to-commit, config, lockfile, and manifest evidence contract.
- [`docs/lockfile_reproduction.md`](docs/lockfile_reproduction.md) — dependency snapshot update and reproduction policy.
- [`docs/rag_observatory_exports.md`](docs/rag_observatory_exports.md) — trace and sweep exports for external RAG observability analysis.
- [`docs/trec_dl_external_validity.md`](docs/trec_dl_external_validity.md) — TREC-DL dataset contract, runner commands, metric boundaries, and benchmark follow-ups.
- [`docs/cross_domain_benchmarks.md`](docs/cross_domain_benchmarks.md) — NFCorpus and SciFact benchmark contract, commands, output isolation, and result boundary.
- [`notebooks/rag_eval_demo.ipynb`](notebooks/rag_eval_demo.ipynb) — lightweight evaluation workflow demo.

## Engineering surface

The repository includes engineering support for running, validating, and
auditing the experiments:

- **Config-driven pipeline.** `configs/pipeline.yaml` defines the BM25 → dense
  → hybrid RRF → rerank → retrieval-matrix → generation → paired-bootstrap
  → generator-capacity sequence.
  `python scripts/run_pipeline.py --dry-run` prints the executable plan without
  loading data or models.
- **Research evaluation workflow.** `rag-eval run --config configs/baseline.yaml`
  builds the end-to-end BM25, dense, rerank, paired generation, bootstrap, and
  grounding-plus-triad plan from the baseline config. Use `--dry-run` to inspect the
  command sequence before touching data or models.
- **Model-stack smoke.** `python scripts/smoke_model_stack.py --config
  configs/baseline.yaml` loads the pinned generator and dense encoder, runs one
  short CPU generation, and checks a normalized embedding shape. Use it before
  accepting torch / transformers / sentence-transformers upgrades.
- **CI and automation.** The GitHub Actions workflow runs unit tests, linting,
  deterministic fixture metric goldens, and manifest/reproduction checks; the
  local mirror is `make test`, `make lint`, and `make check-fixture-metrics`.
- **Run metadata.** Major runners write `manifest.json`, `resolved_config.yaml`,
  metrics, output hashes, config hashes, git commit, dependency fingerprints,
  and sampling metadata.
- **Retrieval quality reports.** `mgq-retrieval-report` evaluates any
  TREC-format `run.tsv`, compares two runs on matched qids, and builds
  multi-run matrices for BM25, dense, RRF, and reranked outputs. See
  [`docs/retrieval_quality_reporting.md`](docs/retrieval_quality_reporting.md).
- **Retrieval lift diagnostics.** `scripts/analyze_retrieval_lift.py` compares
  two `run.tsv` files per query, bucketizing reranker gains/losses into
  promoted, demoted, new-hit, and lost-hit cases. See
  [`docs/retrieval_lift_analysis.md`](docs/retrieval_lift_analysis.md).
- **TREC-DL benchmark runners.** The BM25 and cross-encoder runners accept the
  judged TREC-DL 2019/2020 passage tracks, reuse the shared full-corpus index,
  isolate outputs by year, and record track and topic-scope provenance. This is
  the execution surface for the graded evaluation in
  [#153](https://github.com/GioiaZheng/msmarco-genqa/issues/153) and the
  benchmark in [#154](https://github.com/GioiaZheng/msmarco-genqa/issues/154);
  it is not itself a new empirical result.
- **Cross-domain benchmark adapters.** NFCorpus and SciFact run through the same
  BM25 and cross-encoder entry points, but each dataset uses its own BEIR corpus,
  BM25 index, qrels, and output directory. This is the execution surface for
  [#165](https://github.com/GioiaZheng/msmarco-genqa/issues/165); it does not
  claim cross-domain results until complete checked artifacts are committed.
- **RAG triad reporting.** `mgq-rag-triad` joins prediction files with optional
  qrels and writes per-query context relevance, groundedness, and answer
  relevance diagnostics. See
  [`docs/rag_triad_evaluation.md`](docs/rag_triad_evaluation.md).
- **Trace export for observability.** `mgq-export-rag-observatory` exports one
  prediction row into the `msmarco-genqa.trace-export.v1` JSON shape consumed by
  `rag-observatory`. `mgq-export-rag-observatory-sweep` writes a small sweep
  manifest plus per-configuration trace files so failures can be compared across
  stable config ids. `make reproduce-small` builds public-safe single-trace and
  two-arm sweep fixture exports without downloading data or model weights. See
  [`docs/rag_observatory_exports.md`](docs/rag_observatory_exports.md).
- **Context packing.** `mgq-generate --context-packing` applies deterministic
  passage trimming, sentence selection, deduplication, and span provenance
  before generation. `mgq-context-packing-report` compares packed and plain
  prediction files on matched qids. See
  [`docs/context_packing.md`](docs/context_packing.md).
- **Input validation.** Shared validation rejects malformed `run.tsv` rows,
  corrupted JSONL records, empty queries, duplicate ids, invalid ranks,
  non-finite scores, and replacement-character-heavy text before expensive
  runners or serving calls proceed. See
  [`docs/input_validation.md`](docs/input_validation.md).
- **Experiment tracking.** `msmarco_genqa.util.tracking.ExperimentTracker`
  writes local JSONL events by default and can mirror runs to MLflow or
  Weights & Biases via `pip install -e ".[tracking]"`. `mgq-sweep-summary`
  reconstructs multi-arm comparison tables from those local events without
  requiring a hosted tracking service.
- **Model serving.** `mgq-serve` exposes a lightweight FastAPI wrapper around
  the generator (`pip install -e ".[serve]"`), with `/health` and `/generate`
  endpoints for local demos or integration tests. It binds to `127.0.0.1` by
  default and rejects non-loopback hosts unless `--allow-remote` is supplied.
  The opt-in does not add authentication, TLS, or rate limiting, so place the
  service behind appropriate controls rather than exposing it directly.
  Validation failures are returned as structured 422 payloads. See
  `examples/demo_payload.json` for a minimal request body:

  ```bash
  curl -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    --data @examples/demo_payload.json
  ```
- **Larger-generator sweep.** `scripts/run_generator_capacity_sweep.py` runs
  the same paired BM25/reranked comparison with `t5-base`; pass
  `--model-name google/flan-t5-base` to evaluate FLAN-T5 under the same
  pipeline and bootstrap protocol.

## Project layout

- **`experiments/`** — four pipeline-stage runners (BM25, dense, rerank, generation). Source of the benchmark numbers.
- **`scripts/`** — analyses, ablations, validation, integration smokes that read `experiments/` outputs.
- **`src/`** — importable library code backing both.
- **`docs/`** — experiment narratives and reproducibility notes.
- **`notebooks/`** — lightweight demos over package APIs and CLI dry runs; they are
  not required for metric reproduction.
- **`reports/acl_findings/`** — ACL-Findings-style experimental report draft.
- **`reports/repo_report/`** — repository report PDF, HTML, sources, and figures.
- **`reports/generated/artifacts/`** — checked machine-readable report table inputs.
- **`reports/generated/tables/`** — LaTeX table fragments plus source sidecars.
- **`metadata.json`** — project metadata summarising dataset scale, pipeline stages,
  headline metrics, CI, tracking, and serving support.

See [§2 Directory layout](#2-directory-layout) for the full breakdown.

User-facing experiment sections use compact W-stage aliases only as
chronological report labels. Filesystem artifacts use descriptive directory
names such as `outputs/bm25_baseline/`, `outputs/dense_retrieval/`, and
`outputs/cross_encoder_rerank/`.

Refresh report table fragments after metrics artifacts change:

```bash
python scripts/export_report_tables.py
```

CI runs the same exporter and fails if `reports/generated/tables/` drifts
from the checked artifacts.

Notebook demos are kept output-free and lightweight:

```bash
python scripts/check_notebooks.py
```

Use package entry points, scripts, and configs for reproducible experiments;
notebooks are only for interactive inspection.

## 1. Status

### Stage 1 — EDA

Dataset statistics + query/passage/answer-type distributions covered in §1 of
[`reports/repo_report/report.pdf`](reports/repo_report/report.pdf).
Source figures: `figures/{query_length,passage_length,query_type,answer_type_by_query_type}_distribution.png`.

### Stage 2 — BM25 retrieval

[`experiments/run_retrieval.py`](experiments/run_retrieval.py). On `dev/small`
(6 980 queries, full 8.8 M-passage corpus):
**MRR@10 = 0.1703**, Recall@100 = 0.6212, Recall@1000 = 0.8154.

### Stage 3 — RAG generation (200-query sample)

[`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py).
200-query sample of dev/small (seed 42), T5-small (no fine-tuning),
top-3 BM25 passages:
ROUGE-L = 0.1626, BLEU = 0.0574, EM = 0.0050, Token-F1 = 0.1756.
The full-dev comparison against a reranked upstream lives in
*Generation × retrieval source* below.

### Stage 4 — Dense retrieval (qrels-anchored 50 k sample)

[`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py).
`all-MiniLM-L6-v2`, FAISS `IndexFlatIP` over L2-normalised embeddings.

Dense MRR@10 = 0.8830, nDCG@10 = 0.9041, Recall@100 = 0.9946 — vs
BM25-on-sample MRR@10 = 0.6948, Recall@100 = 0.9338 (same 50 k pool).
Numbers are upper-bounded by the sampling — every dev relevant doc is
in the pool. Read the comparison as dense-vs-BM25 on the same sample,
not against the BM25 full-corpus number.

**Dense-retrieval follow-ups:**

- *Same-tier encoder comparison* on the identical 50 k sample
  ([`scripts/run_encoder_comparison.py`](scripts/run_encoder_comparison.py)):

  | Encoder | MRR@10 | nDCG@10 | Recall@100 | ms/passage |
  |---|---:|---:|---:|---:|
  | `all-MiniLM-L6-v2` (baseline) | 0.8830 | 0.9041 | 0.9946 | 16.2 |
  | `all-MiniLM-L12-v2` | 0.8933 | 0.9131 | 0.9955 | 31.3 |
  | **`BAAI/bge-small-en-v1.5`** | **0.9021** | **0.9196** | **0.9967** | 34.4 |

  `bge-small` lifts MRR@10 by +0.019 at ~2× the CPU encoding cost.
- *Density sensitivity* with `bge-small` at three sample sizes
  ([`scripts/run_density_sweep.py`](scripts/run_density_sweep.py)).
  Δ MRR@10 over BM25-on-sample grows monotonically with sample size:
  +0.166 (49.6 % density, 15 k) → +0.188 (24.8 %, 30 k) → +0.207 (14.9 %, 50 k).

### Stage 5 — Cross-encoder reranking

[`experiments/run_reranker.py`](experiments/run_reranker.py).
`cross-encoder/ms-marco-MiniLM-L-6-v2` over the dense top-100.
Full dev/small (6 980 queries):

| Metric      | Dense      | + CE rerank | Δ          |
|-------------|-----------:|------------:|-----------:|
| MRR@10      | 0.8830     | 0.9304      | **+0.0474** |
| nDCG@10     | 0.9041     | 0.9434      | +0.0393    |
| Recall@100  | 0.9946     | 0.9946      | +0.0000    |

Runtime ~4h37m on a 6-core MacBook (538 000 query-passage pairs at
~32 pairs/s, peak RSS ~3.3 GiB). Recall@100 is unchanged because
reranking only reorders the top-100. A 1 000-query pilot produced
MRR Δ +0.0435, nDCG Δ +0.0398 — close to the full-dev deltas, so the
gain is not specific to the subsample.

### TREC-DL full-corpus benchmark

BM25 and BM25-plus-cross-encoder were run separately on the judged TREC-DL
2019 and 2020 passage tracks against all 8,841,823 MS MARCO passages. Graded
nDCG keeps the original 0–3 labels; MRR and recall use `rel >= 2`.

| Track | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---|---:|---:|---:|---:|
| 2019 (43 topics) | BM25 | 0.5471 | 0.4239 | 0.4469 | 0.6983 |
| 2019 (43 topics) | BM25 + CE top-100 | **0.8787** | **0.7210** | 0.4469 | n/a |
| 2020 (54 topics) | BM25 | 0.6280 | 0.4773 | 0.5105 | 0.7521 |
| 2020 (54 topics) | BM25 + CE top-100 | **0.8256** | **0.6801** | 0.5105 | n/a |

The reranker improves ordering on both tracks without changing its fixed
top-100 candidate set. All four runs cover every judged topic, and independent
`ir-measures` 0.4.3 checks agree within `2.22e-16`. The CE run has only 100
documents per topic, so CE Recall@1000 is intentionally not reported. See the
[full protocol, runtime, provenance, and error review](docs/trec_dl_external_validity.md).

The four text-only ranked runs are also published as a checksummed
[GitHub Release bundle](https://github.com/GioiaZheng/msmarco-genqa/releases/tag/v2.1-trec-dl-baselines).
Recompute the table without rebuilding the 8.8M-passage index or rerunning the
cross-encoder:

```bash
make reproduce-trec-eval
```

The command follows the pinned
[`artifacts/trec_dl_baselines_v1.json`](artifacts/trec_dl_baselines_v1.json)
pointer, verifies the archive and every member, obtains public qrels through
`ir_datasets`, and fails if any reproduced metric differs by more than
`1e-12`. The release excludes passage/query text, qrels mirrors, model caches,
and machine-local manifests.

### Generation × retrieval source

Cross-stage comparison on **full dev/small (6 980 queries)**: same
T5-small (no fine-tuning), same top-3 passages, only the retrieval
source changes. Mutually restricted via `--restrict-to-run` so both
runs cover the same 6 980 qids (qid set-diff = ∅).

| Retrieval source → T5-small | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25     | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked | **0.3621** | **0.2922** | **0.0606** | **0.3677** |
| **Δ (rerank − BM25)** | **+0.1763** | **+0.2206** | **+0.0471** | **+0.1711** |

Reranking roughly doubles every generation metric. The retrieval-flag
rate (at least one relevance-judged passage in top-3) jumps from
20.8 % (BM25) to 96.9 % (reranked). Per-query token-F1 splits:
4 015 strict improvements / 1 766 regressions / 1 199 ties — net
+2 249 / 6 980 queries (32 %) strictly improved.

#### Paired-bootstrap 95 % CI on Δ (rerank − BM25)

Full 6 980 paired qids, 10 000 bootstrap resamples, seed 42 (per-query
ROUGE-L and BLEU from `rouge_score.RougeScorer` and NLTK `sentence_bleu`
smoothing-1):

| Metric          | Δ       | 95 % CI on Δ          | p₂      |
|-----------------|--------:|----------------------:|--------:|
| ROUGE-L         | +0.1742 | [+0.1663, +0.1820]    | < 0.001 |
| BLEU (sentence) | +0.1330 | [+0.1265, +0.1395]    | < 0.001 |
| Exact-Match     | +0.0471 | [+0.0417, +0.0527]    | < 0.001 |
| Token-F1        | +0.1711 | [+0.1632, +0.1789]    | < 0.001 |

#### Bucket analysis by query type (token-F1)

| query_type     |    n |   BM25 |  Reranked |        Δ |
|----------------|-----:|-------:|----------:|---------:|
| DESCRIPTION    | 3725 | 0.1889 |    0.3939 | **+0.2050** |
| ENTITY         |  631 | 0.1765 |    0.3186 | +0.1421  |
| LOCATION       |  498 | 0.2495 |    0.3928 | +0.1433  |
| NUMERIC        | 1665 | 0.1997 |    0.3235 | +0.1238  |
| PERSON         |  461 | 0.2186 |    0.3557 | +0.1371  |

`DESCRIPTION` (53 % of eval) benefits most; `NUMERIC` benefits least.

#### Semantic-proxy BERTScore (3 000-paired subsample)

DistilBERT-based BERTScore (rescaled, paired-bootstrap CI; seed 42):
BM25 0.2192 → Reranked 0.3920, **Δ +0.1728, 95 % CI [+0.1608, +0.1850],
p < 0.001**. Per-query: rerank strictly better 64.8 %, tie 5.7 %,
BM25 strictly better 29.5 %.

The semantic-proxy Δ sits within a hair of the surface-form Token-F1 Δ
(+0.1711) and ROUGE-L Δ (+0.1742). The rerank gain shows up in a
semantic-similarity scorer at the same magnitude, so it is not a
surface-form artefact. DistilBERT is a proxy for the conventional
`roberta-large` BERTScore — for cross-paper comparison re-run with
`--model-type microsoft/deberta-xlarge-mnli` or
`--model-type roberta-large`.

#### Regression failure taxonomy (40-query triage)

233 of 6 980 paired qids land in the `regression` bucket (reranker
brings a relevant passage into top-3 but token-F1 drops vs BM25). A
40-query seeded triage with deterministic heuristic labels
([`scripts/regression_failure_taxonomy.py`](scripts/regression_failure_taxonomy.py),
seed 42):

| label                    |  n | share |
|--------------------------|---:|------:|
| `truncation_midword`     | 22 | 55 %  |
| `truncation_short`       | 14 | 35 %  |
| `topic_drift`            |  2 |  5 %  |
| `extractive_passage_bias`|  2 |  5 %  |
| `semantic_mismatch`      |  0 |  0 %  |

~90 % of regressions are generation-side output style — the reranker
brings in richer passages, but T5-small extracts short or mid-sentence
fragments. Example (qid 49802, "belizean cuisine"): BM25 produces a
24-token description; reranked produces "Belizean cuisine" (the
passage title).

### Stage 6 — Evaluation layer follow-ups

- **Decoding-budget closure** (`max_new_tokens=64→128`, full dev/small).
  Same generator, prompts, retrieval inputs, seed; only `max_new_tokens`
  changes. Token-F1 Δ +0.1706, ROUGE-L Δ +0.1736, BLEU Δ +0.1325,
  EM Δ +0.0471 — all four CIs strictly above 0. Regression bucket
  shrinks 233 → 231; truncation share 90.0 % → 87.5 %. The
  budget-cap reading is falsified: T5-small hits EOS naturally, not a
  decode-budget wall. The mid-word output style is intrinsic to the
  model on this prompt format.
- **Question-form analyses** (no new generation; offline
  analyses over the existing per-query metrics):
  [`scripts/tag_query_forms.py`](scripts/tag_query_forms.py) tags all
  6 980 queries; [`scripts/analyze_rerank_by_query_form.py`](scripts/analyze_rerank_by_query_form.py)
  reports rerank Δ per form. `which` (n = 120) is the only form whose
  95 % CI on ΔToken-F1 includes zero (Δ = +0.0250, CI [−0.028, +0.077],
  p = 0.33) despite a retrieval-side ΔMRR@10 of +0.71 — the reranker
  improves retrieval but the gain does not transfer downstream on
  selection-style queries. Factual wh-forms (what / where / why /
  how / who) all see Δ in +0.12 to +0.23 with CIs strictly above 0.
  [`scripts/regression_query_profile.py`](scripts/regression_query_profile.py):
  Mann-Whitney on five structural features finds no meaningful
  query-side difference between regressions and the rest.

### Stage 7 — Grounding audit

Deterministic CPU pass over the existing full-dev paired predictions,
asking what T5-small is actually doing on this prompt format
([`scripts/grounding_audit.py`](scripts/grounding_audit.py); ~2 s
scoring + bootstrap, no model load):

| Metric                          | BM25    | Reranked | Δ        | 95 % CI on Δ         |
|---------------------------------|--------:|---------:|---------:|---------------------:|
| Lexical content-token grounding | 0.9972  | 0.9977   | +0.0005  | [−0.0003, +0.0014]   |
| 3-gram grounding                | 0.9873  | 0.9905   | +0.0032  | [+0.0015, +0.0050]   |
| NLI entailment (3 000-pair)     | 0.2270  | 0.0821   | **−0.1448** | [−0.1597, −0.1297] |

Both arms sit at the lexical ceiling — T5-small on
`question: ... context: ...` is doing extractive QA. The surface-form
rerank gain (ΔToken-F1 +0.171) is downstream of retrieval: better
passages → better extractive output.

The NLI Δ is the one metric in this project whose sign reverses vs
the surface-form story. Likely mechanism (to be confirmed by a
generator swap): reranked top-3 produces more fragmentary / mid-word
snippets that inflate word + 3-gram overlap but the sentence-level
NLI cross-encoder cannot entail a fragmentary hypothesis.
[`src/msmarco_genqa/evaluation/nli_grounding.py`](src/msmarco_genqa/evaluation/nli_grounding.py)
implements the score; driver is `grounding_audit.py --nli-n-pairs 3000`.

A 30-query seeded triage of low-grounding rerank outputs
([`scripts/low_grounding_case_study.py`](scripts/low_grounding_case_study.py)):
77 % `paraphrase_reorder` (content words present, order/phrasing
differs), 23 % `partial_external` (mostly tokeniser artefacts:
`350oF` vs `350°F`; `competed` vs `compete`), 0 %
`parametric_or_external`. No genuine hallucinations in the sample.

### Reference points

Published Anserini/Lucene BM25 on MS MARCO `dev/small`: MRR@10 ≈ 0.184.
Our `bm25s`-based 0.1703 is in the same ballpark; the gap is mostly
tokenizer-induced.

## 2. Directory layout

```
configs/             baseline.yaml — paths + retrieval/generation/reranker/eval knobs
experiments/         four pipeline-stage runners
scripts/             analysis, drivers, ablations, validation, smokes
src/                 importable library code backing experiments/ and scripts/
  data/                msmarco.py — ir_datasets loader for the official corpus
  retrieval/           bm25.py, dense.py, sampling.py
  reranking/           cross_encoder.py, io.py
  generation/          rag_generator.py — T5/BART RAG generator
  evaluation/          retrieval.py, generation.py, grounding.py, nli_grounding.py,
                       rag_triad.py, bertscore.py, bootstrap.py, query_form.py
  util/                manifest.py, environment.py — per-run provenance
tests/               pytest suite (no network, no models)
docs/                experiments.md — pipeline narrative (BM25 → dense → rerank → gen)
reports/
  repo_report/         report.tex + report.pdf + report.html + figures/
figures/             plots used in the report (committed)
outputs/             run.tsv, metrics.json, examples.jsonl, manifest.json per stage (gitignored)
data/                raw/, processed/, cache/ — all gitignored, .gitkeep tracked
```

### `experiments/` vs `scripts/`

- **`experiments/`** — the four pipeline-stage runners. Each consumes the
  official MS MARCO corpus (via `ir_datasets`) and produces a structured
  output directory under `outputs/`:

  | Runner | Stage |
  |---|---|
  | [`experiments/run_retrieval.py`](experiments/run_retrieval.py) | BM25 first-stage retrieval |
  | [`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py) | Dense first-stage retrieval (sampled) |
  | [`experiments/run_reranker.py`](experiments/run_reranker.py) | Cross-encoder reranking |
  | [`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py) | RAG generation |

  These runners produce the numbers cited in
  [`reports/repo_report/report.pdf`](reports/repo_report/report.pdf)
  and in [`docs/experiments.md`](docs/experiments.md).

- **`scripts/`** — everything that reads or analyses outputs of
  `experiments/`, plus ablation drivers, validation, and integration
  smokes. Examples:

  | Kind | Examples |
  |---|---|
  | Evaluation drivers | `bootstrap_generation_comparison.py`, `bertscore_paired_eval.py`, `grounding_audit.py`, `grounding_correlation.py`, `mgq-rag-triad` |
  | Failure / case analysis | `regression_failure_taxonomy.py`, `regression_query_profile.py`, `low_grounding_case_study.py` |
  | Slicing / tagging | `tag_query_forms.py`, `analyze_rerank_by_query_form.py`, `analyze_generation_rerank.py` |
  | Ablation drivers | `run_density_sweep.py`, `run_encoder_comparison.py`, `run_topk_sweep.py`, `run_generator_capacity_sweep.py` |
  | End-to-end driver | `run_full_generation_and_analysis.py` |
  | Validation / smoke | `validate_full_rerank.py`, `smoke_test_resume.py` |

  Scripts may change shape as analyses evolve; only the `experiments/`
  output schema is held fixed.

Everything runs from the project root after the editable install registers
the package; no `PYTHONPATH` needed.

## 3. Setup

Python 3.10+ is required. CI currently runs on Python 3.10.

```bash
pip install -r requirements.txt
pip install -e .                       # register `src` as a real package
```

For a pinned version of the current security-refreshed environment, install
the lockfile instead:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

CI dry-runs the lockfile resolver so pinned direct dependencies remain
installable together.

Or, equivalently:

```bash
make install
```

Model-stack dependency updates should also run the opt-in smoke after install.
It downloads the pinned HuggingFace checkpoints from `configs/baseline.yaml` and
does not touch MS MARCO data:

```bash
python scripts/smoke_model_stack.py --config configs/baseline.yaml --device cpu
```

Optional, only for PDF report generation:

```bash
brew install pandoc                    # macOS
brew install --cask basictex           # macOS LaTeX
# Linux: sudo apt-get install pandoc texlive-xetex
```

## 4. Run the official baselines

Each runner is exposed both as a Python script and as a console script
(see `pyproject.toml [project.scripts]`):

```bash
python experiments/run_retrieval.py    # script form
mgq-retrieve                            # console form
```

Console names: `mgq-transform-queries`, `mgq-query-transform-ablation`,
`mgq-retrieve`, `mgq-dense`, `mgq-fuse`, `mgq-retrieval-report`,
`mgq-context-packing-report`,
`mgq-rag-triad`,
`mgq-export-rag-observatory`,
`mgq-rerank`, `mgq-generate`, `mgq-trec-eval`, `mgq-trec-release`,
`mgq-beir-release`.
The examples below use whichever of the script or console forms makes the
dataset and stage boundary clearest.

### TREC-DL 2019/2020 full-corpus retrieval

The BM25 and cross-encoder runners accept the judged TREC-DL passage tracks
without changing the shared MS MARCO corpus index or model configuration:

```bash
mgq-retrieve --dataset msmarco-passage/trec-dl-2019/judged --resume
mgq-rerank --dataset msmarco-passage/trec-dl-2019/judged --resume

mgq-retrieve --dataset msmarco-passage/trec-dl-2020/judged --resume
mgq-rerank --dataset msmarco-passage/trec-dl-2020/judged --resume
```

The default outputs are separated under `outputs/trec_dl_2019/` and
`outputs/trec_dl_2020/`. For these datasets, runner `metrics.json` files use
the original graded labels for nDCG@10 and relevance threshold 2 for MRR@10,
Recall@100, and Recall@1000. Reportable runs still pass through the independent
cross-check below. See
[`docs/trec_dl_external_validity.md`](docs/trec_dl_external_validity.md).

### NFCorpus / SciFact cross-domain retrieval

NFCorpus and SciFact are the first BEIR targets for checking whether the
retrieval and reranking stack generalizes beyond MS MARCO-derived passage
benchmarks:

```bash
mgq-retrieve --dataset beir/nfcorpus/test --resume
mgq-rerank --dataset beir/nfcorpus/test --resume

mgq-retrieve --dataset beir/scifact/test --resume
mgq-rerank --dataset beir/scifact/test --resume
```

Unlike TREC-DL, these datasets do not reuse the MS MARCO passage corpus. Their
default outputs are isolated under `outputs/beir_nfcorpus_test/` and
`outputs/beir_scifact_test/`, and their BM25 indexes are isolated under
`data/processed/bm25_index_beir_nfcorpus_test/` and
`data/processed/bm25_index_beir_scifact_test/`. The BM25 runs report
Recall@1000; top-100 cross-encoder runs omit it because their candidate depth
cannot support a comparable Recall@1000 measurement. See
[`docs/cross_domain_benchmarks.md`](docs/cross_domain_benchmarks.md).

The complete checked runs cover all 323 NFCorpus and 300 SciFact test queries:

| Dataset | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---|---:|---:|---:|---:|
| NFCorpus | BM25 | 0.5186 | 0.3064 | 0.2378 | 0.4572 |
| NFCorpus | BM25 + CE | 0.5662 | 0.3411 | 0.2378 | n/a |
| SciFact | BM25 | 0.6312 | 0.6617 | 0.8759 | 0.9606 |
| SciFact | BM25 + CE | 0.6517 | 0.6787 | 0.8759 | n/a |

Here `n/a` means the reranked run has depth 100; reporting it as Recall@1000
would be misleading. The fixed candidate set also explains the unchanged
Recall@100. The cross-encoder improves early ranking on both collections, while
the low NFCorpus Recall@100 establishes a candidate-set ceiling. The
fixed-output follow-up finds that 72/323 queries have no relevant document in
the BM25 top-100 set. A complete review attributes 67/72 cases primarily to
source-page context missing from the exported title; 62/72 contain only
level-1 qrels and 58/72 come from topic pages. On the 144 non-topic queries,
MRR@10 moves from 0.5073 to 0.5827 after reranking while Recall@100 remains
0.2769. This makes benchmark representation a competing explanation to
retriever capacity rather than an architecture verdict.

The matching SciFact first-stage diagnostic shows a different failure shape:
BM25 retrieves at least one relevant top-100 candidate for 265/300 queries,
259/300 queries already have complete relevant-document coverage at depth 100,
and only 11 queries remain misses at depth 1000. This supports treating the
large NFCorpus gap as dataset- and representation-sensitive rather than a
general failure of the unchanged first stage. The residual 35-case review finds
that most SciFact misses are claim/evidence formulation mismatches rather than
the same missing source-context pattern observed in NFCorpus. See the
[`NFCorpus first-stage diagnostic`](docs/nfcorpus_first_stage_error_analysis.md),
[`SciFact first-stage diagnostic`](docs/scifact_first_stage_error_analysis.md),
[`cross-dataset first-stage comparison`](docs/cross_dataset_error_analysis.md),
[`SciFact residual failure review`](docs/scifact_failure_review.md),
and
[`72-case taxonomy review`](docs/nfcorpus_first_stage_taxonomy_review.md).

The predeclared follow-up changes only the query representation on the official
102-query NFCorpus test/video subset. Combining the BEIR title with the bounded
official description raises BM25 Recall@100 from 0.2821 to 0.3700
(`+0.0880`, paired 95% CI `[+0.0535, +0.1265]`) and reduces no-hit-at-100
queries from 11 to 4. Description alone has a positive point estimate, but its
primary interval crosses zero. This is evidence of missing source context on
the video subset, not an architecture or full-dataset generalization result.
See the
[`query-representation report`](docs/reports/2026-07-28-nfcorpus-video-query-representation.md).

The six exact runs behind this ablation (three query representations, before
and after reranking) are published as a checksummed, text-only
[GitHub Release bundle](https://github.com/GioiaZheng/msmarco-genqa/releases/tag/v2.3-nfcorpus-video-query-ablation).
Download the asset, verify the archive and member hashes, recover the public
qrels, and recompute all metrics and paired comparisons with:

```bash
make reproduce-nfcorpus-video-eval
```

The command follows
[`artifacts/nfcorpus_video_query_representation_v1.json`](artifacts/nfcorpus_video_query_representation_v1.json).
It does not rerun retrieval or the cross-encoder, and the bundle excludes query
and document text, qrels mirrors, model weights, caches, and local manifests.

Separately, the four cross-domain NFCorpus/SciFact ranked runs are published as
a checksummed, text-only
[GitHub Release bundle](https://github.com/GioiaZheng/msmarco-genqa/releases/tag/v2.2-beir-cross-domain-baselines).
Recompute all four rows without rebuilding either corpus index or rerunning
the cross-encoder:

```bash
make reproduce-beir-eval
```

The command follows
[`artifacts/beir_cross_domain_v1.json`](artifacts/beir_cross_domain_v1.json),
verifies the ZIP and every member digest, recovers public qrels through
`ir_datasets`, and rejects metric drift above `1e-12`. The release contains
document identifiers, ranks, and scores only; it excludes document/query text,
qrels mirrors, model weights, caches, and machine-local manifests.

### Independent TREC metric cross-check

Install the optional standard evaluator and export the MS MARCO dev/small
qrels once:

```bash
pip install -e ".[evaluation]"
ir_datasets export msmarco-passage/dev/small qrels --format trec \
    > data/processed/msmarco-dev-small.qrels
```

Then validate any retrieval artifact and compare the repository's MRR@10,
nDCG@10, Recall@100, and Recall@1000 calculations with `ir-measures`:

```bash
mgq-trec-eval --backend ir-measures --qrels-format trec \
    --qrels data/processed/msmarco-dev-small.qrels \
    --run outputs/bm25_baseline/run.tsv \
    --output-dir outputs/trec_eval/bm25

mgq-trec-eval --backend ir-measures --qrels-format trec \
    --qrels data/processed/msmarco-dev-small.qrels \
    --run outputs/dense_retrieval/run.tsv \
    --output-dir outputs/trec_eval/dense

mgq-trec-eval --backend ir-measures --qrels-format trec \
    --qrels data/processed/msmarco-dev-small.qrels \
    --run outputs/cross_encoder_rerank_full/run.tsv \
    --output-dir outputs/trec_eval/reranked

mgq-trec-eval --backend ir-measures --qrels-format trec \
    --qrels data/processed/msmarco-dev-small.qrels \
    --run outputs/hybrid_rrf/run.tsv \
    --output-dir outputs/trec_eval/hybrid_rrf
```

Each invocation writes validated `run.trec`, canonical `qrels.trec`, and a
machine-readable `metrics.json`. The canonical run uses rank-derived scores
to preserve the source rank order even when a retrieval backend emitted tied
model scores. Graded collections use raw relevance labels for nDCG and the
explicit `--rel-threshold` for MRR and recall. For TREC-DL passage tracks, use
`--rel-threshold 2`; every qrels topic remains in the evaluation denominator,
including topics absent from the run.
See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) for metric
scope and sampled-corpus interpretation rules.

### Optional pre-retrieval query transformation

Query transformation is disabled for canonical baselines, but the repository
has deterministic artifacts for normalization, lexical expansion, and
de-contextualization ablations:

```bash
mgq-transform-queries --config configs/baseline.yaml --method normalize \
    --output-dir outputs/query_transform/normalize

mgq-query-transform-ablation \
    --summary none=outputs/query_transform/none/summary.json \
    --summary normalize=outputs/query_transform/normalize/summary.json \
    --output-dir outputs/query_transform/ablation
```

Add `--metrics method=path/to/metrics.json` entries after evaluating matched
retrieval runs to report metric deltas alongside changed-query coverage.
The ablation command writes per-method tracking events under
`outputs/query_transform/ablation/tracking/<method>/events.jsonl` plus
`outputs/query_transform/ablation/tracking/summary/sweep_summary.{json,csv,md}`.
To rebuild the comparison table from local tracking events:

```bash
mgq-sweep-summary outputs/query_transform/ablation/tracking \
    --name query-transform-ablation \
    --output-dir outputs/query_transform/ablation/tracking/summary
```

### Stage 2 — BM25

```bash
python experiments/run_retrieval.py
```

First run: ~5 min download (~1 GB), ~15 min `ir_datasets` encoding fix,
~10 min `bm25s` index build, ~70 min retrieve. Total ≈ 1h40m on a
16 GB MacBook. Subsequent runs reuse
`data/processed/bm25_index_msmarco/` (2.1 GB) and only retrieve.

```bash
python experiments/run_retrieval.py --resume         # picks up at next chunk boundary
python experiments/run_retrieval.py --rebuild-index  # force fresh index
```

Outputs: `outputs/bm25_baseline/{metrics.json, run.tsv, examples.jsonl, manifest.json}`.

### Stage 3 — RAG generation

Requires Stage 2 output `outputs/bm25_baseline/run.tsv`.

```bash
python experiments/run_generation_baseline.py
```

Default: 200 dev queries, T5-small, top-3 BM25 passages. CPU runtime
~5–15 min. Tunable knobs in `configs/baseline.yaml`:
- `generation.model_name` (`t5-small`, `t5-base`, `facebook/bart-base`, …)
- `generation.num_eval_queries`
- `generation.top_k_passages`

The runner is retrieval-source agnostic — feed it any TREC-format
`run.tsv` via CLI flags:

```bash
python experiments/run_generation_baseline.py \
    --input-run outputs/cross_encoder_rerank/run.tsv \
    --output-dir outputs/generation_reranked \
    --retrieval-source reranked
```

Use `--restrict-to-run <other_run.tsv>` to force two runs to evaluate on
the same query subsample even when their upstream retrievers cover
different sets — used in *Generation × retrieval source* above.

To compare prompt compression under the same retrieval source, keep the
baseline output untouched and write a packed run to a separate directory:

```bash
mgq-generate \
    --config configs/baseline.yaml \
    --input-run outputs/cross_encoder_rerank_full/run.tsv \
    --output-dir outputs/generation_reranked_packed \
    --retrieval-source reranked_packed \
    --restrict-to-run outputs/bm25_baseline/run.tsv \
    --num-eval-queries 9999 \
    --context-packing \
    --context-max-chars 900 \
    --context-max-passage-chars 320 \
    --context-sentence-selection query_overlap \
    --context-ordering rank

mgq-context-packing-report \
    --baseline-predictions outputs/generation_reranked_full/predictions.jsonl \
    --compressed-predictions outputs/generation_reranked_packed/predictions.jsonl \
    --baseline-name reranked \
    --compressed-name reranked_packed \
    --output-dir outputs/context_packing
```

The packed `predictions.jsonl` keeps `context_packing` span metadata so each
prompt segment can be traced back to its source document id.

### Stage 4 — Dense retrieval

Requires Stage 2 to have produced
`data/processed/bm25_index_msmarco/doc_ids.json` (sampler input).

```bash
python experiments/run_dense_retrieval.py
```

First run: ~13.5 min to encode 50 k passages on a 6-core CPU
(`all-MiniLM-L12-v2` takes ~26 min); ~20 s FAISS search. Subsequent
runs reuse the cached index. Tunable knobs:
- `dense.model_name` (e.g. `sentence-transformers/msmarco-MiniLM-L6-cos-v5`)
- `dense.sample_size` (default 50 000)
- `dense.compare_bm25_on_sample`

### Stage 4B - Hybrid RRF fusion

Fuse two or more TREC-format first-stage runs with weighted Reciprocal
Rank Fusion (RRF). The sample-matched BM25 and dense outputs from Stage 4
are the recommended first comparison because both runs share the same
candidate pool and qrels caveat.

```bash
python experiments/run_hybrid_fusion.py \
    --input-run bm25_sample=outputs/dense_retrieval/run_bm25_sample.tsv \
    --input-run dense=outputs/dense_retrieval/run.tsv \
    --output-dir outputs/hybrid_rrf \
    --top-k 1000
```

Pass `--qrels <path>` to compute MRR, nDCG, and recall in `metrics.json`.
The runner always writes `run.tsv`, `provenance.jsonl`, `metrics.json`,
`resolved_config.yaml`, and `manifest.json`.

For a same-qid RRF comparison table, rerank the fused run and then build a
matrix over BM25-on-sample, dense, RRF, and RRF-plus-rerank:

```bash
python experiments/run_reranker.py \
    --input-run outputs/hybrid_rrf/run.tsv \
    --output-dir outputs/hybrid_rrf_rerank \
    --resume

mgq-retrieval-report matrix \
    --run bm25_sample=outputs/dense_retrieval/run_bm25_sample.tsv \
    --run dense=outputs/dense_retrieval/run.tsv \
    --run rrf=outputs/hybrid_rrf/run.tsv \
    --run rrf_reranked=outputs/hybrid_rrf_rerank/run.tsv \
    --baseline-name bm25_sample \
    --output-dir outputs/retrieval_reports/hybrid_matrix
```

The matrix report writes `matrix.json`, `pairwise_deltas.jsonl`, and
`report.md`; every row is restricted to the qids shared by all four runs.

### Stage 5 — Cross-encoder reranking

Requires Stage 4 output `outputs/dense_retrieval/run.tsv`.

```bash
python experiments/run_reranker.py
```

Reranks the dense top-100 per query. CPU runtime scales linearly with
the number of queries — full 6 980 × top-100 is ~6 h on a 6-core
laptop. Use `--num-eval-queries` for a deterministic subsample.

```bash
python experiments/run_reranker.py --num-eval-queries 50 --rerank-top-k 100   # smoke (~1 min)
OMP_NUM_THREADS=12 python experiments/run_reranker.py --num-eval-queries 1000  # ~50 min
```

Tunable knobs under `reranker:`:
- `reranker.model_name` (any HF cross-encoder)
- `reranker.rerank_top_k` (depth; cost is O(K))
- `reranker.batch_size`, `reranker.max_length`

### Full-dev BM25-vs-reranked comparison

The block reproduced in *Generation × retrieval source*:

```bash
# Generation on full dev/small (~1 h each on a 6-core CPU; mutually restricted):
python experiments/run_generation_baseline.py \
    --input-run outputs/bm25_baseline/run.tsv \
    --output-dir outputs/generation_bm25_full \
    --retrieval-source bm25 \
    --restrict-to-run outputs/cross_encoder_rerank_full/run.tsv \
    --num-eval-queries 9999

python experiments/run_generation_baseline.py \
    --input-run outputs/cross_encoder_rerank_full/run.tsv \
    --output-dir outputs/generation_reranked_full \
    --retrieval-source reranked \
    --restrict-to-run outputs/bm25_baseline/run.tsv \
    --num-eval-queries 9999

# Paired bootstrap + bucket analysis + grounding + triad:
python scripts/bootstrap_generation_comparison.py \
    --bm25-dir outputs/generation_bm25_full \
    --reranked-dir outputs/generation_reranked_full \
    --output-dir outputs/generation_bootstrap_full
python scripts/analyze_generation_rerank.py \
    --bm25-dir outputs/generation_bm25_full \
    --reranked-dir outputs/generation_reranked_full \
    --output-dir outputs/generation_analysis
python scripts/bertscore_paired_eval.py --n-pairs 3000
python scripts/regression_failure_taxonomy.py
python scripts/grounding_audit.py \
    --bm25-dir outputs/generation_bm25_full \
    --reranked-dir outputs/generation_reranked_full \
    --output-dir outputs/grounding
mgq-rag-triad \
    --predictions bm25=outputs/generation_bm25_full/predictions.jsonl \
    --predictions reranked=outputs/generation_reranked_full/predictions.jsonl \
    --baseline-config bm25 \
    --output-dir outputs/rag_triad
```

## 4.5. Single-query demo

The pipeline is batch-eval oriented: every entrypoint in `experiments/`
consumes a retrieval run (`run.tsv`) and emits metrics + paired
predictions. There is no `--question "..."` CLI. The blocks below are
the smallest in-Python composition for a single-query sanity check.

### A. Generator-only

```python
from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator

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

Passages are hand-written here, so the output reflects generator
behaviour, not the end-to-end pipeline.

### B. Retrieve + generate (sketch)

Loading a real index, then mapping returned `doc_ids` back to passage
text. Not a runnable copy-paste — the `doc_id → passage` map is the
missing piece, currently built inside the batch runners:

```python
from msmarco_genqa.retrieval.bm25 import BM25Retriever

bm25 = BM25Retriever.load("data/processed/bm25_index_msmarco")
scores, doc_ids = bm25.retrieve("what is bm25", k=3)
passages = [passage_by_id[d] for d in doc_ids]  # supply your own map
answer = gen.generate(query="what is bm25", passages=passages)
```

A thin `python -m msmarco_genqa.demo.ask "<question>"` wrapper that
bundles the mapping is listed under §8 *Next*.

## 6. Configuration

All knobs live in [`configs/baseline.yaml`](configs/baseline.yaml). Key ones:

| Key | Effect |
|-----|--------|
| `retrieval.k1`, `retrieval.b` | BM25 hyperparameters |
| `retrieval.top_k` | Depth of saved run (default 1000) |
| `retrieval.chunk_size` | Checkpoint cadence in queries (default 200). Smaller = more durable, more I/O. |
| `retrieval.n_threads` | `0` = sequential (matches the 0.1703 baseline). `-1` = all CPUs. |
| `data.corpus_limit` | Set to a small int for a smoke test; does **not** reproduce official numbers. Leave `null` for real runs. |
| `generation.num_eval_queries` | Size of the generation eval subset |

## 6.5. Reproducibility status

| Area | Status | Notes |
|---|---|---|
| Unit tests | works | `make test` / `pytest -q` — no network, no heavy deps. Slow tests excluded by `pytest.ini_options`. |
| Slow tests | works (skips gracefully) | `make test-slow` includes `@pytest.mark.slow`. HF metric scripts skip if unavailable. |
| Lockfile | basic | `requirements-lock.txt` is pip-freeze-style; sub-dep transitive closure + hash pinning are TODO. Model-stack pins are checked with `scripts/smoke_model_stack.py`. |
| Installable package | works | `pip install -e .` registers `src` via `pyproject.toml`; scripts import the installed package without local `sys.path` shims. |
| CI | basic | `.github/workflows/ci.yml`: pytest + ruff on push/PR to main. No slow tests, no data download. |
| Lint | minimal | `ruff` with `F` + `W` (pyflakes + whitespace). `E` / `I` / `UP` are off on the first pass. |
| Artifact manifest | wired | `src/msmarco_genqa/util/manifest.py` writes `outputs/<stage>/manifest.json` alongside `metrics.json`. Captures git commit + dirty flag, command, config hash, dependency-file hashes, per-output sha256 (truncated). |
| Historical experiment numbers in `reports/repo_report/report.pdf` | historical | Reflect the dev environment at tag `v1.0-first-report`. Current dependencies are security-refreshed; use the first-report tag for archival reproduction. |

**Artifact path naming.** Current snapshot anchors use descriptive stage names
such as `outputs/bm25_baseline/`, `outputs/dense_retrieval/`, and
`outputs/cross_encoder_rerank/`. Their committed `provenance.backfill.json`
files remain the archival reproduction anchors for tag `v1.0-first-report`.

Limitations to be aware of:

- The lockfile reflects a macOS CPU-only dev environment. Linux / CUDA may resolve different versions; install `torch` from the appropriate PyTorch index first.
- Corpus, encoder, and reranker checkpoints are downloaded by `ir_datasets` / HuggingFace at first run and are not checksummed by the project.

## 7. Known limitations

- **Tokenizer mismatch with Anserini.** Our 0.1703 vs reference 0.184 is mostly tokenizer-induced (`bm25s` default tokenizer ≠ Lucene `EnglishAnalyzer`).
- **CPU-only retrieve is slow at 8.8 M docs.** ~70 min for 6 980 queries. `n_threads=-1` may help; not yet benchmarked on this corpus.
- **Generation: pretrained T5-small, no fine-tuning.** Numbers will be low on overlap-based metrics. Fine-tuning is in scope for a later iteration.
- **NumPy 2.x runtime warning.** Some compiled deps (torch) were built against NumPy 1.x. Cosmetic; downgrade to `numpy<2` if it ever causes a real failure.
- **TREC-DL is retrieval evidence, not generation evidence.** The two judged
  tracks validate full-corpus BM25 and top-100 reranking. They do not establish
  downstream answer quality or cross-domain RAG generalization.
- **BEIR coverage is deliberately narrow.** NFCorpus and SciFact show that the
  same reranker improves early ranking on two non-MS-MARCO corpora. Two small
  retrieval collections do not establish broad domain generalization, and no
  generation stage was evaluated on either dataset.

## 8. Next

Longer-horizon research and engineering directions are tracked in
[`ROADMAP.md`](ROADMAP.md). The list below is the shorter technical queue
closest to the current experimental surface.

- **TREC-DL reference expansion.** Compare the frozen BM25-plus-MiniLM result
  against learned sparse or late-interaction retrieval only under the same
  per-track qrels, candidate-depth, and independent-evaluator contract.
- **Top-k Pareto.** K ∈ {50, 100, 200} perf-latency Pareto on
  both first stages (1 000-q subsample for K=50/200; K=100 reuses the reranker
  full-dev). Queued.
- **Generator capacity, not decode budget.** The generation-analysis closure
  (`max_new_tokens=64→128`) plus the grounding ceiling (~99 %
  extractiveness on both arms) together imply generator-side work
  should target richer prompt formats (multi-passage synthesis,
  citation-aware decoding) or a different model — not the decode
  budget. Driver
  [`scripts/run_generator_capacity_sweep.py`](scripts/run_generator_capacity_sweep.py)
  runs T5-base on both BM25 and reranked top-3 and re-scores every
  grounding metric, so the open question — does the NLI sign flip
  (Δ = −0.145) hold at higher capacity? — is one command away.
- **Try a MS-MARCO-tuned dense encoder.** Swap
  `sentence-transformers/all-MiniLM-L6-v2` for
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5` and re-run dense retrieval on
  the same 50 k sample. Measures how much of the current dense-vs-BM25
  gap is attributable to generic-encoder choice vs the retrieval setup.
- **Single-query demo CLI.** A thin
  `python -m msmarco_genqa.demo.ask "<question>"` wrapper around the
  composition shown in §4.5.

## 9. Contributing

Setup, the `make test` / `make lint` gates, and the branch, commit, and
pull-request conventions are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## 10. Citation

If you use this repository, cite the software metadata in
[`CITATION.cff`](CITATION.cff). The current citation target is
`v2.0-reproducibility-protocol`, which anchors the schema-v2 manifest contract
and reproduction checks.

## 11. License

See [LICENSE](LICENSE).

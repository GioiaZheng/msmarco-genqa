# Reproducibility

This repository is organized so that the code, configuration, output
manifests, and written reports can be audited together. The goal is not only
to run the pipeline, but to make each reported number traceable to a command,
configuration, dependency set, and output directory.

## Environment

Recommended setup:

```bash
python -m venv .venv
. .venv/bin/activate
make install
```

`make install` installs both `requirements.txt` and the editable package.
Pinned package versions used for the frozen report snapshot are recorded in
`requirements-lock.txt`.

## Data

The project uses MS MARCO Passage Ranking / QA data through the local dataset
cache. Raw corpus files, generated indexes, model weights, and experiment
outputs are not committed.

The repository policy for source data, derived indexes, model outputs, and
small pointer files is documented in `docs/artifact_versioning.md`. Use
`make check-artifacts` to verify that common large artifact formats and
generated run payloads have not entered Git.

The canonical headline-evidence index is `artifacts/registry.json`. It joins
the checked metric artifacts to commits, configuration and lockfile snapshots,
manifest availability, and explicit provenance limitations. Its contract is
documented in `docs/artifact_registry.md`; dependency snapshot changes follow
`docs/lockfile_reproduction.md`.

Important scale reference:

| Item | Value |
|---|---:|
| Dev queries | 6,980 |
| Passage corpus | about 8.8M passages |
| Dense sample | 50k qrels-anchored passages |

## Fast Checks

These commands are intended for local development and CI:

```bash
make test
make lint
make check-artifacts
make check-registry
python scripts/run_pipeline.py --dry-run
```

The dry run prints the pipeline plan without loading MS MARCO data or model
weights.

## Small Trace Export

The smallest end-to-end interop check is:

```bash
make reproduce-small
```

It uses the synthetic fixture under `tests/fixtures/rag_observatory_export/`
and writes a single-trace export plus a two-arm configuration sweep bundle:

| Artifact | Purpose |
|---|---|
| `outputs/reproduce_small/rag_observatory_export.json` | one standard trace export |
| `outputs/reproduce_small/rag_observatory_sweep/rag_observatory_sweep.json` | sweep manifest with stable config ids and comparison rows |
| `outputs/reproduce_small/rag_observatory_sweep/traces/*/*.json` | per-configuration trace files for `rag-observatory` ingestion |

This target does not download MS MARCO data or model weights. It only verifies
that the repository can produce the `msmarco-genqa.trace-export.v1` and
`msmarco-genqa.trace-sweep.v1` shapes used for observability interop.

When torch, transformers, or sentence-transformers changes, also run:

```bash
python scripts/smoke_model_stack.py --config configs/baseline.yaml --device cpu
```

That opt-in check downloads the pinned generator and dense encoder revisions,
runs one short generation, and verifies a normalized embedding shape. It is kept
outside default CI because it requires HuggingFace Hub access and model weights.

## Reproducing the BM25 Baseline

The canonical BM25 reproduction target is:

```bash
make reproduce-baseline
```

This target installs the project, runs BM25 retrieval on the full MS MARCO
dev/small setup, and verifies the resulting manifest and output hashes.

Expected headline value:

| Metric | Expected value |
|---|---:|
| MRR@10 | 0.1703 |

First-run runtime is about 30 minutes on a recent CPU laptop. Later runs can
reuse the cached BM25 index and finish faster.

## Reproducing the Published TREC-DL Evidence

The fast external-evidence target is:

```bash
make reproduce-trec-eval
```

On Windows or other environments without `make`, run the equivalent Python
entry point directly:

```bash
python -m msmarco_genqa.cli.trec_release reproduce
```

It downloads the pinned GitHub Release asset, verifies the ZIP size and
SHA-256 digest, validates every member against the bundle manifest, and
recomputes the BM25 and BM25-plus-cross-encoder metrics for TREC-DL 2019 and
2020. Public qrels are recovered through `ir_datasets`; no private credentials
are required. The command writes checked outputs under
`outputs/reproductions/trec_dl_baselines_v1/evaluation/`.

This is an evidence reproduction, not a new model run: it takes the published
rankings as input and avoids rebuilding the full 8.8M-passage index or
rerunning the cross-encoder. Use the full-corpus commands in
`docs/trec_dl_external_validity.md` when the retrieval pipeline itself must be
rerun.

The Git-tracked pointer is
`artifacts/trec_dl_baselines_v1.json`. It pins the immutable release tag, asset
name, byte size, archive hash, experiment commit, and compact source record.
The release contains document identifiers and scores only; it does not
redistribute MS MARCO passage/query text, qrels mirrors, or model weights.

## Reproducing the Published BEIR Evidence

The corresponding cross-domain evidence target is:

```bash
make reproduce-beir-eval
```

The equivalent direct Python command is:

```bash
python -m msmarco_genqa.cli.beir_release reproduce
```

It downloads the immutable NFCorpus/SciFact release asset, checks the pinned
size and SHA-256 digest, validates every archived run, obtains both public test
qrels sets through `ir_datasets`, and recomputes BM25 and BM25-plus-cross-
encoder MRR@10, nDCG@10, and recall. The output is written under
`outputs/reproductions/beir_cross_domain_v1/evaluation/`.

The Git-tracked pointer is `artifacts/beir_cross_domain_v1.json`. The archive
contains the exact four ranked run files behind the report table, but no
document/query text, qrels mirror, model weights, caches, or machine-local
manifests. This validates the published evidence without rebuilding the
NFCorpus/SciFact indexes or rerunning the cross-encoder.

The fixed-output first-stage diagnostics reuse the same release and public
qrels, then write query-level coverage reports under `outputs/analysis/`:

```bash
make analyze-nfcorpus-first-stage
make analyze-scifact-first-stage
make analyze-cross-dataset-errors
```

These targets do not change the retriever, reranker, or model configuration.
They only separate top-100 candidate-set misses, depth-1000 recoverable cases,
complete relevant-document coverage, and the NFCorpus/SciFact cross-dataset
failure partition for the published BM25 outputs. The cross-dataset target
also validates the compact NFCorpus manual taxonomy table before writing
`outputs/analysis/cross_dataset_errors/summary.json` and
`outputs/analysis/cross_dataset_errors/report.md`.

## Reproducing the NFCorpus Video Query Ablation

The six fixed runs from the 102-query query-representation experiment can be
recovered and checked with:

```bash
make reproduce-nfcorpus-video-eval
```

The equivalent direct Python command is:

```bash
python -m msmarco_genqa.cli.nfcorpus_video_release reproduce \
  --cache-dir outputs/reproductions/beir_irds_cache
```

The command follows
`artifacts/nfcorpus_video_query_representation_v1.json`, downloads the pinned
GitHub Release asset, checks its byte size and SHA-256 digest, and validates
every archived member. It then obtains the public NFCorpus test qrels through
`ir_datasets`, selects the frozen 102-query video cohort from the run qids, and
recomputes all six aggregate result rows. It also verifies that each reranked
candidate set equals the corresponding BM25 top 100 and reruns the published
10,000-resample paired bootstrap with seed `20260727`.

Checked `metrics.json` and `metrics.md` outputs are written under
`outputs/reproductions/nfcorpus_video_query_representation_v1/evaluation/`.
This is an exact-output evidence reproduction: it does not rerun BM25, the
cross-encoder, or corpus indexing. The archive contains ranked document
identifiers and scores only; query/document text, qrels mirrors, model weights,
caches, and machine-local manifests are excluded.

## Full Pipeline Plan

Print the executable plan:

```bash
make pipeline-dry-run
```

The configured stages are full-corpus BM25 retrieval, dense retrieval,
cross-encoder reranking, retrieval lift analysis, BM25-based generation,
reranked generation, paired-bootstrap confidence intervals, and the
generator-capacity sweep.

The source of truth for this plan is `configs/pipeline.yaml`.

## Run Artifacts

Major experiment runners write:

| Artifact | Purpose |
|---|---|
| `manifest.json` | schema, command, git state, dependencies, output hashes |
| `resolved_config.yaml` | final config after CLI overrides |
| `metrics.json` | metrics plus sampling metadata |
| task outputs | `run.tsv`, `predictions.jsonl`, `examples.jsonl`, or equivalent |

The detailed manifest contract is documented in
`docs/reproducibility_protocol.md`.

## Tracked Sweeps

Experiment tracking remains local-first. Every tracked run writes
`events.jsonl` with run metadata, parameters, metrics, and artifact
references. Optional MLflow or Weights & Biases integrations mirror the same
run when those packages and credentials are available; the local JSONL file is
still the reproducibility source of truth.

Use `mgq-sweep-summary` to rebuild comparison tables from local tracking
directories:

```bash
mgq-sweep-summary outputs/query_transform/ablation/tracking \
    --name query-transform-ablation \
    --output-dir outputs/query_transform/ablation/tracking/summary
```

The command writes `sweep_summary.json`, `sweep_summary.csv`, and
`sweep_summary.md`. The JSON output preserves nested tags, parameters,
metrics, and artifact references; the CSV and Markdown outputs are intended
for quick review and report-table plumbing.

## Sampling Boundary

The dense retrieval and reranker numbers are measured on a qrels-anchored
50k-passage sample. Every dev relevant document is included by construction,
with random distractors filling the remainder. This makes the dense and
BM25-on-sample comparison controlled, but optimistic relative to full-corpus
retrieval.

Use these comparisons as:

- valid: dense-on-sample vs BM25-on-sample,
- valid: dense top-100 vs cross-encoder reranked top-100,
- not valid: dense-sample MRR directly compared to full-corpus BM25 MRR.

## Reported Results

The main result summary is in `RESULTS.md`. Stage-by-stage experiment notes
are in `docs/experiments.md`, the repository report is in
`reports/repo_report/report.pdf`, and the compact paper-style findings write-up
is in `reports/acl_findings/report.pdf`.

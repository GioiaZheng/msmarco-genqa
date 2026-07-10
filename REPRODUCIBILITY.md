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

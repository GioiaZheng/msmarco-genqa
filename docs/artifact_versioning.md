# Data and artifact versioning policy

This project uses Git for source code, configuration, small fixtures,
reports, and compact provenance records. Large MS MARCO-derived data and
model artifacts stay outside normal commits.

The goal is to keep the repository reproducible without turning Git into a
storage backend for corpora, embedding matrices, FAISS indexes, or model
weights.

## What stays in Git

| Class | Examples | Reason |
|---|---|---|
| Source code | `src/`, `experiments/`, `scripts/`, tests | Required to review and reproduce behavior |
| Configuration | `configs/*.yaml`, `pyproject.toml`, lock files | Defines the run contract |
| Small fixtures | synthetic JSONL/TSV fixtures under `tests/fixtures/` | Enables CI without network or model downloads |
| Run metadata | manifests, resolved configs, provenance backfill JSON | Re-identifies a run without storing payloads |
| Report outputs | checked report PDFs/HTML, generated table fragments | Keeps reported evidence auditable |

## What does not stay in Git

| Class | Examples | Expected handling |
|---|---|---|
| Raw corpora | MS MARCO passage files, qrels mirrors, local dataset cache | Managed by `ir_datasets` or a local dataset cache |
| Derived indexes | BM25 indexes, FAISS indexes, HNSW/IVF files | Rebuilt from config and manifests, or referenced by external pointers |
| Embeddings | `.npy`, `.npz`, memory-mapped matrices, vector shards | Stored outside Git and tied back by manifest fingerprints |
| Model weights | HuggingFace cache, checkpoints, `.pt`, `.bin`, `.safetensors` | Resolved by model name/revision or an external artifact pointer |
| Large generated runs | full `run.tsv`, full prediction dumps, candidate pools | Written under ignored `outputs/` paths and summarized by metrics/manifests |

## Current repository contract

The current default is manifest-first and local-first:

1. Runners write `manifest.json`, `resolved_config.yaml`, `metrics.json`, and
   task-specific outputs under `outputs/<run>/`.
2. Git tracks compact provenance records and report table inputs, not full
   generated run payloads.
3. The manifest records code state, dependency files, config hashes, output
   hashes, sampling metadata, and data/environment fingerprints.
4. A reported number should be traceable to a command, config, dependency set,
   and output directory even when the large payload itself is not committed.

This is sufficient for small public fixtures and report evidence. It is not a
complete long-term artifact backend for large indexes or model outputs.

## When DVC is worth adding

DVC or a similar pointer-file workflow becomes worthwhile when the project
needs to version large artifacts across multiple comparable runs, for example:

- several dense indexes built from different encoders,
- repeated ablation sweeps with large candidate pools,
- model checkpoints or embedding shards that must be recovered exactly,
- shared storage across machines or collaborators.

Before adding DVC as a normal dependency, the project should first prototype a
local-only remote or dry-run workflow. The prototype should commit only small
pointer files and should not require cloud credentials in CI.

For small fixtures, report tables, and manifest JSON, the existing Git plus
manifest workflow remains simpler and more reviewable than DVC.

## Guardrail

Use:

```bash
make check-artifacts
```

The check scans tracked files and fails if common large artifact formats or
data/output payloads are staged for Git. It is intentionally conservative: it
does not inspect ignored local caches, and it does not require DVC to be
installed.

This guardrail is a repository hygiene check, not a storage system. A future
DVC or object-store workflow should still preserve the existing manifest
contract so code, data, and reported metrics remain connected.

# Reproducibility protocol

Single source of truth for the `msmarco-genqa.manifest.v2` contract:
what every experiment run records, what each field means, and how to
verify a recorded run is reproducible.

## TL;DR

Every manifest-writing experiment runner (`mgq-{retrieve, dense, fuse, rerank,
generate}` and the matching `experiments/run_*.py` entry points) writes:

- `outputs/<run>/manifest.json` — schema-v2 provenance contract
- `outputs/<run>/resolved_config.yaml` — config dict actually used (with CLI overrides applied)
- `outputs/<run>/metrics.json` — numbers + sampling caveat + env capture
- `outputs/<run>/{run.tsv, examples.jsonl, predictions.jsonl, ...}` — task-specific artefacts

To reproduce the headline BM25 baseline:

```bash
make reproduce-baseline
```

To audit a recorded run:

```bash
python scripts/verify_reproduction.py outputs/week02_bm25
```

## Manifest schema v2

Schema string: `msmarco-genqa.manifest.v2`. **Hard break from v1**:
writes always emit v2; v1 manifests on disk remain readable as plain
JSON but no migration is provided. New runs land on v2 only.

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema` | str | Always `"msmarco-genqa.manifest.v2"` |
| `timestamp_utc` | str | ISO-8601 UTC, second precision. **Not** part of any hash. |
| `git` | dict | `{"commit": <12-hex SHA>, "dirty": <bool>}` |
| `command` | list[str] | `sys.argv` of the run |
| `config` | list[dict] | Config-file records (`{path, size_bytes, sha256_16}`) |
| `dependencies` | list[dict] | `requirements*.txt` + `pyproject.toml` file records |
| `outputs` | list[dict] | Per-output file records (path + size + truncated digest) |
| `python` | dict | `{version, executable, platform}` |
| `extra` | dict | Task-specific fields — including the 4 required ones below |

### Required-fields contract (the six)

Six fields are **required**. Writing a manifest with any required field
missing or `None` raises `RequiredFieldMissingError` and refuses to
leave a partial file on disk. Pass `--allow-incomplete-manifest` at
the runner CLI to bypass during development.

| Required field | Source | What it records |
|---|---|---|
| `git.commit` | `git rev-parse --short=12 HEAD` | Code state at write time |
| `git.dirty` | `git status --porcelain` | Uncommitted-changes flag |
| `extra.seed` | `cfg["seed"]` | RNG seed (covers random / numpy / torch / transformers) |
| `extra.resolved_config_hash` | `compute_resolved_config_hash(cfg)` | 64-hex SHA256 of resolved cfg dict |
| `extra.data_fingerprint` | `compute_data_fingerprint(...)` | 64-hex SHA256 of run inputs |
| `extra.env_fingerprint` | `compute_env_fingerprint(env_dict)` | 64-hex SHA256 of `capture_environment()` |

The four `extra.*` fields are what makes "the manifest is sufficient
to re-identify and reproduce the run" load-bearing rather than
aspirational.

### Why each fingerprint exists

- **`resolved_config_hash`**: the file-level sha256 of
  `configs/baseline.yaml` (which already lives in `manifest.config[0]`)
  misses CLI overrides like `--sample-size`, `--model-name`,
  `--num-eval-queries`. The resolved hash captures the actual decision
  state.
- **`data_fingerprint`**: lean — `cache_dir` + `corpus_limit` + content
  hashes of any per-run extra inputs (e.g. `sample_doc_ids.json` for
  dense, upstream `run.tsv` for reranker / generation). The 8.8M-passage
  corpus body is NOT hashed — its identity is anchored by `cache_dir`
  + ir_datasets dataset name + `corpus_limit`; rehashing the body on
  every run would be wasteful and download-order-dependent.
- **`env_fingerprint`**: stable hash of `capture_environment()`.
  Sensitive to package version drift, python version, cpu brand,
  mem_gb. Insensitive to wall-clock noise — `capture_environment`
  has no timestamp field, so two consecutive runs on the same env
  hash identically.

### Adjacent artefact: `resolved_config.yaml`

YAML serialisation of the cfg dict that drove the run, with all CLI
overrides applied. Written with `sort_keys=True` so diffing two runs'
configs is order-stable. The `verify_reproduction.py` script loads
this file back to a dict and re-computes its hash — round-tripping
through YAML is part of the contract.

### Sampling block (peer of `metrics`)

Every `metrics.json` also carries `payload["sampling"]` (top-level peer
of `metrics`, **not** nested inside it). One of two shapes:

```json
{"is_sampled": false}
```

on full-corpus runs (BM25 baseline by default), or:

```json
{
  "is_sampled": true,
  "method": "qrels-anchored",
  "sample_size": 50000,
  "caveat": "Numbers are derived from a qrels-anchored sub-sample of the full MS MARCO passage corpus..."
}
```

on sub-corpus runs (dense, reranker, generation-when-upstream-is-sampled).
The canonical caveat asserts three load-bearing phrases — `qrels-anchored`,
`upper-bound`, `not comparable to full-corpus` — pinned by test against
silent wording drift.

## Dev-time bypasses

Two CLI flags recognised by the manifest-writing experiment runners:

- `--require-clean-tree` — refuse to write the manifest if the git
  tree is dirty. Default off. Use for canonical / headline runs.
- `--allow-incomplete-manifest` — bypass the required-fields contract.
  Default off. Use for development iterations that don't intend to
  produce reproducible artefacts.

The two are symmetric in role: both are dev-time bypasses for
contracts that production / headline runs must satisfy.

## Reproducing a recorded run

Two halves:

### 1. Re-run

```bash
make reproduce-baseline
```

This installs (`pip install -e .`), runs BM25 retrieval on the full
8.8M-passage MS MARCO corpus, and writes a v2-compliant manifest.
Expected wall-clock on a 2024-era 8-core CPU laptop: ~30 minutes
(first run pays ~20 min indexing + ~10 min retrieval; subsequent
runs hit the cached index in ~2 min).

Expected headline number: `MRR@10 = 0.1703` on dev/small (6,980
queries, full corpus).

### 2. Verify

```bash
python scripts/verify_reproduction.py outputs/week02_bm25
```

Five checks (any failure → exit 1):

1. Manifest `schema == "msmarco-genqa.manifest.v2"`.
2. All six `REQUIRED_FIELDS` populated (non-`None`).
3. `resolved_config.yaml` reloads to a dict whose
   `compute_resolved_config_hash` matches recorded
   `extra.resolved_config_hash`.
4. `metrics.json` on disk has the `sha256_16` recorded in
   `manifest.outputs`.
5. `manifest.git.commit` matches current HEAD (warning, not error,
   if reproducing on a different commit).

Headline metrics are also printed for visual inspection — but the
structural check is what guarantees "the recorded numbers are the
ones you would reproduce", not "the recorded numbers are the right
numbers".

## How fields propagate (call graph)

```
runner (experiments/run_*.py)
  ↓
  cfg = load_config(args.config)
  cfg = apply_cli_overrides(cfg, args)         # CLI overrides land here
  env_dict = capture_environment()
  ↓
  write_resolved_config(cfg, output_dir)       # writes resolved_config.yaml
  resolved_config_hash = compute_resolved_config_hash(cfg)
  data_fingerprint = compute_data_fingerprint(cache_dir=..., extra_files=...)
  env_fingerprint = compute_env_fingerprint(env_dict)
  ↓
  write_run_manifest(
    extra={
      "seed": seed,
      "resolved_config_hash": resolved_config_hash,
      "data_fingerprint":     data_fingerprint,
      "env_fingerprint":      env_fingerprint,
      ...task-specific fields...
    },
    allow_incomplete=args.allow_incomplete_manifest,
  )
  ↓
  _validate_required(manifest)                 # default strict
  ↓
  write JSON to output_dir/manifest.json
```

All four runners share this shape; per-task differences live in `extra`'s
free-form fields.

## What the contract does NOT capture

These remain out of scope (handled separately or accepted as gaps):

- **Corpus binary content** — anchored by `cache_dir` + `corpus_limit`
  + ir_datasets dataset name. Re-hashing 8.8M passages every run is
  wasteful and download-order-dependent.
- **HuggingFace model weights** — `revision` (40-hex commit SHA) is
  pinned in `configs/baseline.yaml` and recorded in `extra`. The
  weights themselves are downloaded by `transformers` /
  `sentence-transformers` from HF Hub at first use; the pinned
  revision is what guarantees the same weights. Dependency upgrades that
  touch torch / transformers / sentence-transformers should run
  `python scripts/smoke_model_stack.py --config configs/baseline.yaml --device cpu`
  so both the pinned generator and dense encoder are loaded under the new
  stack before the lockfile is accepted.
- **Hardware non-determinism** — captured at the descriptive level
  in `env.cpu.brand`, `env.mem_gb`. CUDA non-determinism would
  surface when the repo moves off CPU-laptop (24-month-roadmap
  Phase 2).
- **Wall-clock identity** — `timestamp_utc` is recorded for
  bookkeeping but is **not** part of any hash. Two runs at different
  wall-clock times with identical inputs / env / cfg hash to the
  same `data_fingerprint` + `env_fingerprint` + `resolved_config_hash`.

## Per-task profile extensions (forward-looking)

The schema-v2 contract is generic. Some experiments (notably
NLI-grounding evaluation) layer additional required fields on top
via per-task profiles. This is **not** in v2 baseline; it lands
when the experiment code does:

- **R5 `research/metric-robustness`** will extend `REQUIRED_FIELDS`
  with per-NLI-task fields: `extra.nli.backbone`,
  `extra.nli.score_formula`, `extra.nli.threshold`,
  `extra.nli.premise_hypothesis_direction`. These are the
  8-confounder-control parameters from the Axis C literature audit;
  recording them per run is what lets a paper Robustness Section
  argue "the reversal is intrinsic to NLI choice, not an artefact
  of one specific parameter setting".

The extension is per-task: a retrieval run does not need NLI fields;
a grounding run does. The validator gains a task-keyed required-
fields map rather than a single global tuple.

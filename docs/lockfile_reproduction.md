# Lockfile reproduction policy

The dependency files serve two different purposes:

- `requirements.txt` and `pyproject.toml` define the supported dependency
  ranges for development and installation;
- `requirements-lock.txt` records the pinned direct dependency snapshot used
  for reproducibility checks.

The current normalized lockfile hash and the commit of its most recent
dependency change are recorded in `artifacts/registry.json`. CI verifies the
current file against that hash and re-reads historical lockfiles from Git for
each canonical experiment entry.

## Change protocol

A lockfile update must be intentional and reviewable. The pull request should
state why the dependency changed, whether it can affect model or metric
behavior, which validation was run, and whether new experiment evidence is
needed.

| Change type | Minimum evidence |
|---|---|
| Tooling-only dependency | Resolver dry run, default tests, lint, artifact registry check. |
| Evaluation or numerical library | Above checks plus fixture metric goldens and affected evaluation tests. |
| Torch, Transformers, SentenceTransformers, tokenizer, or model revision | Above checks plus `scripts/smoke_model_stack.py`; rerun the affected headline experiment or open a linked follow-up before treating old and new results as comparable. |
| Security remediation with forced transitive changes | Record the vulnerability and constrained package set, run the affected smoke/evaluation checks, and document any unavoidable reproduction boundary. |

### Security refresh: 2026-07-21

The current snapshot updates `torch` from 2.12.1 to 2.13.0 and `nltk` from
3.9.4 to 3.10.0. The change removes the direct pins flagged for
CVE-2025-3000 and CVE-2026-12243 during the repository security audit while
keeping the rest of the model and evaluation stack fixed.

The two packages have different reproduction implications:

- the Torch update can affect model execution or numerical behavior, so the
  model-stack smoke test is required before this snapshot is accepted;
- NLTK is used here through `nltk.translate.bleu_score`, so the fixture metric
  goldens and bootstrap scorer tests are the relevant regression boundary.

This refresh does not rebaseline any published metric. Historical results stay
attached to the lockfile snapshots recorded by their artifact-registry entries;
new experiment manifests record the environment used for new runs.

At minimum, run:

```bash
python -m pip install --dry-run -r requirements-lock.txt
python scripts/check_fixture_headline_metrics.py
python scripts/check_artifact_registry.py
python scripts/export_report_tables.py
```

The registry hash must be updated in the same change as the lockfile. That
mechanical update does not by itself validate numerical equivalence.

## Historical boundary

A newer lockfile does not rebaseline an older result. Each canonical entry
records the lockfile available at its evidence or production commit:

- `repository_snapshot` means the file existed and its historical hash is
  verified from Git;
- `not_present` means the commit predates the lockfile;
- absence of the original installed environment remains a stated limitation,
  even when a repository lockfile is available.

Dependency vulnerability triage should prefer the smallest compatible change.
If a safe resolution requires a model-stack change, preserve the old record,
produce new evidence under the new environment, and compare the two rather
than silently replacing the headline number.

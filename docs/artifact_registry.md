# Canonical artifact registry

`artifacts/registry.json` is the machine-readable index for the experiment
evidence behind the repository's headline results. It does not duplicate run
payloads. Instead, it joins each checked report artifact to the strongest
provenance that the repository can actually support: commits, configuration
and lockfile snapshots, runtime-manifest locations, external pointers, metric
summaries, and explicit limitations.

Run the consistency check with:

```bash
make check-registry
```

The check validates schemas, repository-relative paths, normalized hashes,
artifact identifiers, provenance boundaries, and metric values. When a Git
executable is available in a full clone, it also verifies that commits exist,
report tags resolve to the recorded commits, and historical configuration and
lockfile hashes match the files at those commits. CI uses a full-history
checkout so these Git checks are always active there. On systems where Git is
not on `PATH`, set `GIT_EXECUTABLE` or pass `--git-executable` to the checker.

## Evidence levels

Every entry declares one of two provenance states:

| State | Meaning |
|---|---|
| `exact` | The production commit is known and agrees with the checked artifact record. |
| `historical_partial` | The reported result is retained, but the original production commit or runtime evidence is incomplete. `production_commit` must be `null`; evidence commits and a report anchor are recorded without presenting them as the run commit. |

Manifest availability is recorded separately:

| Availability | Meaning |
|---|---|
| `tracked` | The record is committed and the path must exist. |
| `local_only` | The record exists in the maintainer's local output tree but is intentionally outside Git. |
| `not_preserved` | The historical report names the path, but the original file is no longer available. |

This distinction matters for the pre-protocol baselines. Legacy local
backfills name `5a35de9c18ea` as a run commit, but that object is not present in
the retained repository history. The registry therefore does not promote that
value to a production commit. The `v1.0-first-report` tag and the earliest
verifiable evidence commits are recorded separately.

## Hash convention

Registry text hashes use `sha256_lf`: CRLF and bare CR newlines are normalized
to LF before SHA-256 is computed. This makes the same JSON, YAML, or lockfile
hash identically on Windows and Linux. Binary release payloads keep ordinary
byte-for-byte SHA-256 in their external pointer.

## Entry contract

Each canonical entry contains:

- a checked table artifact and its `sha256_lf`;
- exact or explicitly partial provenance;
- the relevant configuration snapshot(s);
- the lockfile state at the evidence commit(s);
- a compact numeric metric summary; and
- notes defining comparison or recovery limits.

External release assets add an immutable pointer containing the tag, asset
name, byte size, and byte-level SHA-256. Large run files, model weights, raw
datasets, indexes, and machine-specific manifests remain outside Git.

## Updating the registry

1. Add or update the small checked report artifact. Do not copy a large run
   payload into `artifacts/`.
2. Identify the production commit only from a runtime manifest or equivalent
   primary evidence. If it cannot be established, use `historical_partial`.
3. Hash the relevant artifact, configuration, and lockfile snapshots using the
   LF-normalized convention.
4. Add a concise metric summary and state sampling or comparison boundaries in
   `notes`.
5. Run `make check-registry`, `make check-report-tables`, and the relevant
   experiment verification before review.

Published metrics are never silently rewritten to match a new environment. A
new run receives a new evidence record; historical entries retain their
original status and limitations.

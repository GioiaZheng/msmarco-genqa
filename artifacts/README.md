# External artifact pointers

This directory contains small, reviewable experiment-evidence records and
pointers to immutable assets. Payloads remain outside Git; each external
pointer pins the release tag, asset name, byte size, and SHA-256 digest needed
to recover it without private credentials.

Published assets are never replaced in place. A changed payload receives a
new tag, asset name, and pointer so prior report evidence remains recoverable.

Current records:

- `registry.json` — canonical headline-evidence registry joining report
  artifacts to commits, configuration and lockfile snapshots, manifest
  availability, and explicit provenance limits.
- `trec_dl_baselines_v1.json` — text-only BM25 and BM25-plus-cross-encoder
  TREC-DL 2019/2020 ranked runs.

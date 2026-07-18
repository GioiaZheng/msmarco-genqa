# External artifact pointers

This directory contains small, reviewable pointers to immutable experiment
assets. Payloads remain outside Git; each pointer pins the release tag, asset
name, byte size, and SHA-256 digest needed to recover it without private
credentials.

Published assets are never replaced in place. A changed payload receives a
new tag, asset name, and pointer so prior report evidence remains recoverable.

Current pointer:

- `trec_dl_baselines_v1.json` — text-only BM25 and BM25-plus-cross-encoder
  TREC-DL 2019/2020 ranked runs.

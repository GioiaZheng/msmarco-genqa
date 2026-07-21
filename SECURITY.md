# Security policy

Last reviewed: 2026-07-21.

## Supported versions

Security fixes are applied to the current `main` branch and included in the next
release. Historical research releases remain immutable so their published
artifacts and environments can still be reproduced; they do not receive
backports.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/GioiaZheng/msmarco-genqa/security/advisories/new).
Include the affected entry point, reproduction steps, expected impact, and any
suggested mitigation. Do not open a public issue for an unpatched vulnerability
or include live credentials, private data, or exploit payloads in a report.

General hardening suggestions that do not disclose a vulnerability may use the
normal issue tracker.

## Security boundary

This repository is a research pipeline operated by a trusted user. Its primary
entry points are local CLI commands that process operator-selected datasets,
models, configurations, and filesystem paths. The project does not provide
multi-user isolation or sandbox arbitrary model and dataset artifacts.

The optional `mgq-serve` FastAPI wrapper is intended for local demos and
integration tests. It binds to `127.0.0.1` by default and rejects non-loopback
hosts unless the operator explicitly passes `--allow-remote`. That override does
not add authentication, TLS, authorization, request quotas, or rate limiting;
deployments outside a trusted host must supply those controls separately.

## External artifacts and network access

- Hugging Face libraries download model and metric artifacts selected by the
  configuration or CLI. Use pinned model revisions for controlled experiments.
- `ir_datasets` and dataset loaders retrieve benchmark corpora and judgments
  from their configured upstream sources.
- Release-reproduction commands download repository-owned result bundles and
  verify the archive and member hashes before consuming them.

Treat model weights, datasets, run files, JSONL inputs, and archives from
untrusted sources as untrusted data. Run third-party artifacts in an isolated
environment when their provenance is uncertain.

## Repository controls

- GitHub secret scanning and push protection cover repository history and
  incoming pushes.
- A repository-local high-confidence scanner checks tracked text files in CI.
- CodeQL analyzes Python changes and the default branch.
- GitHub Actions are pinned to full commit SHAs and receive read-only repository
  contents by default.
- Dependency updates are proposed by Dependabot. The direct-dependency snapshot
  is audited during security reviews and checked against the artifact registry.

## Known limitations

- `requirements-lock.txt` pins direct dependencies but is not a fully resolved,
  hash-locked transitive environment. Use isolated environments and review
  resolver output for sensitive deployments.
- Downloaded models and datasets inherit the trust and availability properties
  of their upstream registries.
- The local FastAPI wrapper is not a production-ready public API.

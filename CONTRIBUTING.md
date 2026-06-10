# Contributing

Thanks for your interest in improving this project. This guide covers the
local setup, the checks every change must pass, and the conventions used
for branches, commits, and pull requests.

## Setup

Python 3.10+ is required; CI runs on Python 3.10.

```bash
pip install -r requirements.txt
pip install -e .          # register `src` as a package
```

For a pinned environment, install the lockfile instead:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

`make install` performs the same steps.

## Checks

Two gates run locally and in CI. Both must pass before a change is merged:

```bash
make test    # pytest -q; slow tests are excluded by default
make lint    # ruff check over src tests experiments scripts
```

`make test-slow` additionally runs tests marked `@pytest.mark.slow`; these
download pinned model checkpoints and skip gracefully when unavailable.

If a change touches the model stack or its pins, also run the opt-in smoke,
which downloads the checkpoints declared in `configs/baseline.yaml` and does
not touch the dataset:

```bash
python scripts/smoke_model_stack.py --config configs/baseline.yaml --device cpu
```

## Branches

Development happens on a single trunk (`main`) with short-lived,
single-purpose branches that merge back when complete. Name a branch
`<type>/<short-topic>`, using one of:

- `infra/` — packaging, CI, tooling, code quality
- `research/` — experimental code and run sets
- `docs/` — documentation and protocol writeups
- `fix/` — bug fixes
- `feature/` — discrete dataset or model integrations

## Commits

- One logical change per commit. Each commit passes `make test` and
  `make lint` on its own, so the history stays bisectable.
- Subject line in the imperative mood, ~70 characters or fewer, prefixed
  with the scope it touches, e.g. `infra(manifest):`, `research(nli):`,
  `docs(readme):`.
- The body explains the *why* and is wrapped at ~72 columns.
- No work-in-progress or fixup commits on a branch that is up for review;
  amend or rebase locally before opening the pull request.

## Pull requests

- Open the pull request against `main`.
- CI (test + lint) must be green before merge.
- Merges preserve a non-linear history — use a merge commit rather than
  squashing, so each round remains visible in the graph.
- Reference the issue the change closes in the description.

## Issues

When filing an issue, prefix the title with the area it concerns in
brackets, matching the existing labels — for example `[Infra]`,
`[Research]`, `[Evaluation]`, or `[Cleanup]`. State the goal, the scope,
and the acceptance criteria so the work is self-contained.

## License

By contributing, you agree that your contributions are licensed under the
same terms as the project; see [LICENSE](LICENSE).

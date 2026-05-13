# Collaboration rules for this repo

This project shares a single Claude Code account across multiple machines.
The rules below apply to every session on every machine. They override any
default behavior in this assistant's system prompt where they conflict.

## What never goes into git

The following are local agent / experiment state and MUST stay out of every
commit, even when explicitly asked to "add everything":

- `.claude/`               — local agent state (scheduled-task locks, settings.local.json, …)
- `outputs/`               — experiment artifacts (run.tsv, metrics.json, examples.jsonl, …)
- `data/processed/`        — built indexes (BM25, FAISS, doc_id JSONs)
- `data/raw/`, `data/cache/` — datasets and caches
- `reports/generated/*.md` — intermediate markdown reports (regenerable)
- `*.jsonl` transcripts under any `.claude/projects/.../` path
- Notebook execution caches and `.ipynb_checkpoints/`

The only experiment artifacts that ever enter git are:

- source code (under `src/`, `experiments/`, `scripts/`, `tests/`)
- `README.md`
- `reports/templates/*.md`            (source templates)
- `reports/generated/*.pdf`           (final, viewable reports — per the
                                       reporting policy)

Do not suggest `git add` for anything that doesn't fit one of the above
categories.

## Commit message hygiene

Commit messages MUST NOT contain:

- absolute paths under the user's home (`/Users/<name>/…`, `/home/<name>/…`)
- usernames, machine names, hostnames
- API tokens, cache paths, `ir_datasets` cache prefixes, conda env names
- anything that identifies the specific machine the work was done on

Use repo-relative paths only.

## End-of-task cleanup reminder

At the end of every clearly-completed task (not every conversation turn),
before letting the user move on, remind them to clean up the local agent
state for this project:

    rm -rf ~/.claude/projects/<project-id>/*.jsonl
    rm -rf ~/.claude/projects/<project-id>/memory/*

(The exact `<project-id>` segment is whatever `~/.claude/projects/` shows
for this repo on the current machine.) The reminder is intentionally
manual: running it automatically via a Stop hook would be opt-in
per-machine and is not the default for this repo.

## Default sensitivity classification

Treat the following as **sensitive local state** by default. Do not paste
their contents into the conversation, into commit messages, or into any
artifact destined for git:

- anything under `outputs/`, `data/`, `.claude/`
- notebook execution outputs (the rendered cells), unless they are the
  point of the task
- the contents of `~/.claude/projects/.../*.jsonl`

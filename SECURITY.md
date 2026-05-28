# Security posture

This is a **single-user research CLI**: every entry point under
`experiments/` and `scripts/` is invoked by a trusted operator, and
there is no networked service, no web framework, no exposed port, and
no multi-tenancy. The threat surface is correspondingly narrow.

Last audit: 2026-05-20.

## What is in scope

1. **Secrets / credential leakage.** API keys, HF tokens, AWS keys, or
   private-key material in the source tree or git history.
2. **Untrusted deserialization.** `pickle.load` / `torch.load` /
   unsafe `yaml.load` against attacker-controllable bytes.
3. **Subprocess injection.** `shell=True`, `os.system`, or
   user-controlled strings concatenated into shell commands.
4. **Path traversal.** User-controlled filesystem paths going into
   `open()` without bounding.
5. **SSRF.** Outbound HTTP requests where the URL is derived from
   user input.

## What is out of scope (and why)

- **CSRF and security response headers.** Not applicable: the repo
  ships no HTTP server, no web framework, no cookies / sessions, no
  cross-origin endpoints. No browser ever loads anything this repo
  produces.
- **Multi-user authorization.** Single-operator CLI by construction.

## Audit results — 2026-05-20

| Axis | Result | Notes |
|---|---|---|
| Secrets in source | ✅ clean | `grep -rEn 'HF_TOKEN\|OPENAI_API_KEY\|AWS_SECRET\|sk-[A-Za-z0-9]{20,}\|hf_[A-Za-z0-9]{20,}\|BEGIN.*PRIVATE KEY'` over the tracked source paths (`src`, `tests`, `experiments`, `scripts`, `configs`, `README.md`, `pyproject.toml`, `requirements*.txt`) returns nothing. |
| `.env` / credential files | ✅ none | No `.env`, no `*.pem`, no `*.key`, no `credentials*`, no `secrets*` anywhere outside `.git/`, `outputs/`, `data/`, and the local-scratch directory. |
| Secrets in git history | ✅ clean | `git log --all -p` over the source paths above returns no plaintext-key matches. |
| `pickle.load` / `torch.load` | ✅ none in source | Zero direct calls. PyTorch is invoked via `transformers` / `sentence_transformers`, which fetch model weights from the HF Hub (trusted registry) — see *Outbound network surface* below. |
| `yaml.load` (unsafe) | ✅ none | All five YAML loads in the project use `yaml.safe_load` against `configs/baseline.yaml`. |
| `subprocess` with `shell=True` | ✅ none | All `subprocess` call sites pass argv lists, not shell strings. Sites: `src/msmarco_genqa/util/manifest.py` (`git rev-parse`, `git status`), `src/msmarco_genqa/util/environment.py` (`git rev-parse`), `tests/test_sampling.py` (test fixtures), `scripts/run_full_generation_and_analysis.py` (driver subprocess). All commands are fixed strings + Path values, not user-string-interpolated. |
| Path-into-`open()` | ✅ bounded | Every entry point uses `argparse` with `type=Path`; downstream `open()` calls receive these paths unchanged. There is no web-form path input. For a single-user CLI this is the expected pattern. |
| SSRF | ✅ none | Zero `requests.get` / `urllib.request.urlopen` / `httpx` calls in source. The only outbound traffic is library-internal (see below); URLs are not user-controlled. |

## Outbound network surface (audited explicitly)

The repo makes outbound HTTP calls only through library internals,
never through hand-written code:

1. **HuggingFace Hub** — pulled by `transformers` / `sentence_transformers`
   / `bert_score` for model weights. Targets are fixed registry URLs
   (`huggingface.co/<org>/<model>`); model ids are baked into
   `configs/baseline.yaml` or set as CLI defaults
   (`distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`,
   `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/nli-deberta-v3-small`,
   `t5-small`). A `--model` / `--nli-model` flag *can* override the id;
   in the single-operator threat model this is benign (the operator
   chooses what to download).
2. **ir_datasets** — pulled for the MS MARCO Passage corpus + qrels.
   URLs are fixed inside `ir_datasets`; cache lives under
   `data/raw/`, gitignored.

Both libraries cache their downloads, so the runtime network surface
collapses to zero after the first invocation.

## Gitignored sensitive paths

Per the project's local operating contract, the following directories
never enter git regardless of how `git add` is invoked:

- The local-scratch directory (operator workspace)
- `outputs/` (experiment artifacts)
- `data/raw/`, `data/processed/`, `data/cache/` (datasets, indexes,
  caches)
- `*.jsonl` transcripts under any local-scratch subdirectory

This is enforced both by the project `.gitignore` and by operator
convention.

## How to report a problem

This repo has no public deployment, no users, and no security contact.
If you believe you have found a real security issue in the code, open
a GitHub issue.

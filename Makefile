# Convenience targets for the MS MARCO GenQA project.
#
# All targets assume the project root as the working directory and that
# Python dependencies are installed (``pip install -r requirements.txt &&
# pip install -e .``).

.PHONY: help install test test-slow lint check-results check-notebooks export-report-tables check-report-tables pipeline-dry-run rag-eval-dry-run model-stack-smoke serve-dev clean-pycache reproduce-baseline

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

help:
	@echo "Common targets:"
	@echo "  make install              -- pip install -r requirements.txt + editable install"
	@echo "  make test                 -- run default unit tests (fast, no network, no HF Hub)"
	@echo "  make test-slow            -- include slow tests (HF metric scripts; skipped if offline)"
	@echo "  make lint                 -- ruff check on src/, tests/, experiments/, scripts/"
	@echo "  make check-results        -- verify metadata.json headline metrics against RESULTS.md"
	@echo "  make check-notebooks      -- verify notebooks stay lightweight demos"
	@echo "  make export-report-tables -- refresh checked LaTeX table fragments"
	@echo "  make check-report-tables  -- verify checked LaTeX table fragments are current"
	@echo "  make pipeline-dry-run     -- print the config-driven experiment plan"
	@echo "  make rag-eval-dry-run     -- print the research evaluation workflow plan"
	@echo "  make model-stack-smoke    -- load baseline HF models and run a short smoke"
	@echo "  make serve-dev            -- start the optional FastAPI generation service"
	@echo "  make reproduce-baseline   -- re-run + verify the W2 BM25 baseline (~30 min CPU laptop)"

# ----------------------------------------------------------------------------- #
# Install
# ----------------------------------------------------------------------------- #

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #

# Default unit-test target. Excludes ``slow`` and ``integration`` markers
# (see [tool.pytest.ini_options] in pyproject.toml). Should always pass
# without network access or downloaded models.
test:
	$(PYTEST) -q

# Run slow tests as well. Slow tests are expected to *skip* (not fail) when
# their external dependency is unavailable — e.g. ``test_evaluate_generation_smoke``
# skips when the HF ``evaluate`` metric scripts can't be loaded.
test-slow:
	$(PYTEST) -q -m "slow or not slow"

# ----------------------------------------------------------------------------- #
# Lint
# ----------------------------------------------------------------------------- #

# Conservative ruff ruleset (pyflakes + whitespace; see pyproject.toml).
# Does not auto-fix; run ``python -m ruff check --fix`` manually if you want fixes.
# Invokes ruff via ``python -m`` so it works regardless of where it's installed
# (``pip install --user`` puts it in a non-PATH location on some setups).
lint:
	$(RUFF) check src tests experiments scripts

check-results:
	$(PYTHON) scripts/check_headline_metrics.py

check-notebooks:
	$(PYTHON) scripts/check_notebooks.py

export-report-tables:
	$(PYTHON) scripts/export_report_tables.py

check-report-tables:
	$(PYTHON) scripts/export_report_tables.py
	git diff --exit-code reports/generated/tables

pipeline-dry-run:
	$(PYTHON) scripts/run_pipeline.py --dry-run

rag-eval-dry-run:
	rag-eval run --config configs/baseline.yaml --dry-run

model-stack-smoke:
	$(PYTHON) scripts/smoke_model_stack.py --config configs/baseline.yaml --device cpu --max-new-tokens 16

serve-dev:
	mgq-serve --host 127.0.0.1 --port 8000

# ----------------------------------------------------------------------------- #
# Housekeeping
# ----------------------------------------------------------------------------- #

clean-pycache:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete

# ----------------------------------------------------------------------------- #
# One-command reproducibility
# ----------------------------------------------------------------------------- #

# Reproduce the W2 BM25 headline baseline end-to-end:
#   1. Install (editable; `pip install -e .`).
#   2. Run BM25 retrieval on the full 8.8M-passage MS MARCO corpus
#      (dev/small, 6,980 queries). Expected MRR@10 = 0.1703.
#   3. Verify the resulting manifest is v2-compliant and re-identifies
#      the run (schema, 6 required fields, resolved_config.yaml hash,
#      metrics.json hash, git HEAD).
#
# First-run wall-clock on a 2024-era 8-core CPU laptop: ~30 min
# (~20 min indexing + ~10 min retrieval). Subsequent runs hit the
# cached BM25 index and finish in ~2 min.
#
# `--require-clean-tree` is intentional: this target is for canonical
# reproduction, not iterative dev. Commit your edits first, or use
# `mgq-retrieve` directly with `--allow-incomplete-manifest`.
reproduce-baseline: install
	mgq-retrieve --require-clean-tree
	$(PYTHON) scripts/verify_reproduction.py outputs/bm25_baseline

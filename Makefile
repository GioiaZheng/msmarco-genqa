# Convenience targets for the MS MARCO GenQA project.
#
# All targets assume the project root as the working directory and that
# Python dependencies are installed (``pip install -r requirements.txt &&
# pip install -e .``).

.PHONY: help install test test-slow lint check-results check-fixture-metrics check-lockfile check-notebooks check-artifacts check-registry export-report-tables check-report-tables pipeline-dry-run rag-eval-dry-run model-stack-smoke serve-dev clean-pycache reproduce-small reproduce-baseline reproduce-trec-eval build-trec-release reproduce-beir-eval build-beir-release reproduce-nfcorpus-video-eval build-nfcorpus-video-release analyze-nfcorpus-first-stage analyze-scifact-first-stage review-nfcorpus-first-stage review-scifact-first-stage analyze-cross-dataset-errors

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
	@echo "  make check-fixture-metrics -- verify deterministic fixture metric goldens"
	@echo "  make check-lockfile       -- dry-run the pinned lockfile resolver"
	@echo "  make check-notebooks      -- verify notebooks stay lightweight demos"
	@echo "  make check-artifacts      -- verify large generated artifacts stay out of Git"
	@echo "  make check-registry       -- validate canonical artifact evidence and history"
	@echo "  make export-report-tables -- refresh checked LaTeX table fragments"
	@echo "  make check-report-tables  -- verify checked LaTeX table fragments are current"
	@echo "  make pipeline-dry-run     -- print the config-driven experiment plan"
	@echo "  make rag-eval-dry-run     -- print the research evaluation workflow plan"
	@echo "  make model-stack-smoke    -- load baseline HF models and run a short smoke"
	@echo "  make serve-dev            -- start the optional FastAPI generation service"
	@echo "  make reproduce-small      -- build the tiny trace-export interop fixture"
	@echo "  make reproduce-baseline   -- re-run + verify the BM25 baseline (~30 min CPU laptop)"
	@echo "  make reproduce-trec-eval  -- fetch + verify published TREC-DL runs and metrics"
	@echo "  make build-trec-release   -- build the maintainer release ZIP from canonical runs"
	@echo "  make reproduce-beir-eval  -- fetch + verify published BEIR runs and metrics"
	@echo "  make build-beir-release   -- build the maintainer BEIR release ZIP"
	@echo "  make reproduce-nfcorpus-video-eval -- verify + evaluate six query-ablation runs"
	@echo "  make build-nfcorpus-video-release -- build the maintainer query-ablation ZIP"
	@echo "  make analyze-nfcorpus-first-stage -- reproduce inputs + diagnose BM25 coverage"
	@echo "  make analyze-scifact-first-stage -- reproduce inputs + diagnose BM25 coverage"
	@echo "  make review-nfcorpus-first-stage -- validate the 72-case taxonomy review"
	@echo "  make review-scifact-first-stage -- validate the 35-case residual failure review"
	@echo "  make analyze-cross-dataset-errors -- compare NFCorpus/SciFact failure regimes"

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

check-fixture-metrics:
	$(PYTHON) scripts/check_fixture_headline_metrics.py

check-lockfile:
	$(PYTHON) -m pip install --dry-run -r requirements-lock.txt

check-notebooks:
	$(PYTHON) scripts/check_notebooks.py

check-artifacts:
	$(PYTHON) scripts/check_artifact_boundaries.py

check-registry:
	$(PYTHON) scripts/check_artifact_registry.py

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

reproduce-small:
	$(PYTHON) scripts/export_rag_observatory_fixture.py
	$(PYTHON) scripts/export_rag_observatory_sweep_fixture.py

# Reproduce the BM25 headline baseline end-to-end:
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

# Fast evidence reproduction: download the public text-only run bundle,
# verify both archive and member hashes, recover the public judged qrels via
# ir_datasets, and recompute the four reported TREC-DL result rows. This does
# not rebuild the 8.8M-passage index or rerun the cross-encoder.
reproduce-trec-eval:
	$(PYTHON) -m msmarco_genqa.cli.trec_release reproduce

# Maintainer-only packaging target. The generated ZIP remains under ignored
# outputs/; its exact hash and size are pinned by artifacts/trec_dl_baselines_v1.json.
build-trec-release:
	$(PYTHON) -m msmarco_genqa.cli.trec_release build --output outputs/releases/trec-dl-baselines-v1.zip

# Fast external-domain evidence reproduction. The release contains only ranked
# identifiers and scores; public NFCorpus/SciFact qrels are recovered through
# ir_datasets before the four result rows are recomputed.
reproduce-beir-eval:
	$(PYTHON) -m msmarco_genqa.cli.beir_release reproduce --cache-dir outputs/reproductions/beir_irds_cache

# Maintainer-only packaging target. The generated ZIP stays under ignored
# outputs/ and is published only after its exact size and SHA-256 are pinned.
build-beir-release:
	$(PYTHON) -m msmarco_genqa.cli.beir_release build --output outputs/releases/beir-cross-domain-baselines-v1.zip

# Fast reproduction of the NFCorpus test/video query-representation evidence.
# The release contains six fixed text-only runs (three representations x two
# systems). Public qrels are recovered through ir_datasets before aggregate
# metrics, candidate-set invariants, and the pinned paired bootstrap are checked.
reproduce-nfcorpus-video-eval:
	$(PYTHON) -m msmarco_genqa.cli.nfcorpus_video_release reproduce --cache-dir outputs/reproductions/beir_irds_cache

# Maintainer-only deterministic packaging target. The generated ZIP remains
# ignored; the tracked pointer pins its exact byte size and SHA-256 digest.
build-nfcorpus-video-release:
	$(PYTHON) -m msmarco_genqa.cli.nfcorpus_video_release build --output outputs/releases/nfcorpus-video-query-ablation-v1.zip

# Query-level diagnosis over the immutable NFCorpus BM25 run. The dependency
# materializes and verifies the public release, qrels, and source archive; this
# target does not rebuild the index, rerun retrieval, or invoke the reranker.
analyze-nfcorpus-first-stage: reproduce-beir-eval
	$(PYTHON) scripts/analyze_first_stage_coverage.py

# Query-level diagnosis over the immutable SciFact BM25 run. This mirrors the
# NFCorpus diagnostic and does not rebuild the index, rerun retrieval, or invoke
# the reranker.
analyze-scifact-first-stage: reproduce-beir-eval
	$(PYTHON) scripts/check_first_stage_contract.py --contract configs/scifact_first_stage_contract.json --output outputs/analysis/scifact_first_stage/data_metric_contract.json --label SciFact
	$(PYTHON) scripts/analyze_first_stage_coverage.py --contract configs/scifact_first_stage_contract.json --output-dir outputs/analysis/scifact_first_stage --sample-seed scifact-first-stage-errors-v1 --label SciFact

# Validate the complete 24+48 NFCorpus review census and regenerate the ignored
# evidence guide and summary from the compact tracked annotations.
review-nfcorpus-first-stage: analyze-nfcorpus-first-stage
	$(PYTHON) scripts/export_nfcorpus_first_stage_review.py --annotations reports/annotations/nfcorpus_first_stage_review_v1.csv

# Validate the bounded SciFact residual review over the 35 no-hit-at-100 cases
# and regenerate the ignored evidence guide and summary from frozen evidence.
review-scifact-first-stage: analyze-scifact-first-stage
	$(PYTHON) scripts/analyze_scifact_failure_review.py

# Cross-dataset retrieval-only error analysis over the frozen BEIR evidence.
# This compares the two first-stage diagnostics and the complete NFCorpus
# failure-review census without rerunning retrieval, reranking, or generation.
analyze-cross-dataset-errors: review-nfcorpus-first-stage review-scifact-first-stage
	$(PYTHON) scripts/analyze_cross_dataset_errors.py

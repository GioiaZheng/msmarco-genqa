# Convenience targets for the MS MARCO GenQA project.
#
# All targets assume the project root as the working directory and that
# Python dependencies are installed (``pip install -r requirements.txt &&
# pip install -e .``).

.PHONY: help install test test-slow lint clean-pycache

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

help:
	@echo "Common targets:"
	@echo "  make install     -- pip install -r requirements.txt + editable install"
	@echo "  make test        -- run default unit tests (fast, no network, no HF Hub)"
	@echo "  make test-slow   -- include slow tests (HF metric scripts; skipped if offline)"
	@echo "  make lint        -- ruff check on src/, tests/, experiments/, scripts/"

# ----------------------------------------------------------------------------- #
# Install
# ----------------------------------------------------------------------------- #

install:
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

# ----------------------------------------------------------------------------- #
# Housekeeping
# ----------------------------------------------------------------------------- #

clean-pycache:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete

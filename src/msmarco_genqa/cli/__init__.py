"""Console-script shims.

Each module here re-exports the ``main`` function from the
corresponding ``experiments/run_*.py`` so the ``mgq-*`` console
entry points declared in ``pyproject.toml`` have something to
import. The actual runner logic stays under ``experiments/`` —
these shims are intentionally one-liners.
"""

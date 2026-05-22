"""Unified RNG seeding for reproducible experiment runs.

The legacy convention in this repo was to call ``random.seed(seed)`` at
the top of each ``experiments/run_*.py`` runner. That seeds Python's
stdlib RNG but NOT ``numpy.random``, NOT ``torch.manual_seed``, and NOT
``transformers.set_seed``. For greedy-decoding T5 and BM25 retrieval
the missing seeds are benign in practice (those code paths don't draw
from numpy/torch RNGs), but the moment anyone adds sampling, dropout,
or any code path that touches NumPy / torch randomness, the
``seed: 42`` promise in ``configs/baseline.yaml`` is silently broken.

``set_global_seed(seed)`` seeds all known RNG sinks at once:

    1. Python stdlib ``random``
    2. NumPy ``numpy.random``        (skipped if numpy not installed)
    3. PyTorch ``torch.manual_seed`` + ``torch.cuda.manual_seed_all`` +
       ``torch.backends.cudnn.deterministic = True`` +
       ``torch.backends.cudnn.benchmark = False``
                                     (skipped if torch not installed)
    4. HuggingFace ``transformers.set_seed``
                                     (skipped if transformers not installed)

It returns a coverage dict describing which sinks were successfully
seeded, so callers can log it (and so manifests can later record the
seeding coverage actually achieved by the run — see
``src/util/manifest.py``).

Design notes
------------

- We seed sinks 1/2/3 explicitly even though ``transformers.set_seed``
  internally calls all three. The per-sink calls are belt-and-suspenders:
  if transformers is not installed (e.g. in lightweight unit tests),
  numpy/torch still get seeded directly.
- ``torch.backends.cudnn.deterministic = True`` is a no-op on CPU but
  matters the moment anyone runs on CUDA. Setting it is cheap.
- All optional-dep imports are wrapped in ``try``/``except ImportError``.
  A missing optional library never prevents seeding the sinks that
  ARE available. The coverage dict reports each as ``"ok"`` or
  ``"skipped: <reason>"``.
"""

from __future__ import annotations

import logging
import random
from typing import Dict

logger = logging.getLogger(__name__)

SeedCoverage = Dict[str, str]


def _seed_random(seed: int) -> str:
    random.seed(seed)
    return "ok"


def _seed_numpy(seed: int) -> str:
    try:
        import numpy as np
    except ImportError:
        return "skipped: numpy not installed"
    np.random.seed(seed)
    return "ok"


def _seed_torch(seed: int) -> str:
    try:
        import torch
    except ImportError:
        return "skipped: torch not installed"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN flags are no-ops on CPU but expected by reviewers who skim
    # for "did you set determinism on GPU".
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return "ok"


def _seed_transformers(seed: int) -> str:
    try:
        from transformers import set_seed
    except ImportError:
        return "skipped: transformers not installed"
    set_seed(seed)
    return "ok"


def set_global_seed(seed: int, *, log: bool = True) -> SeedCoverage:
    """Seed all known RNG sinks; return a coverage dict.

    Parameters
    ----------
    seed :
        The seed value (typically ``cfg["seed"]`` from
        ``configs/baseline.yaml``).
    log :
        If true (default), emit a single INFO log line summarising
        coverage. Pass ``False`` in tests to keep output quiet.

    Returns
    -------
    dict mapping sink name (``random`` / ``numpy`` / ``torch`` /
    ``transformers``) to either ``"ok"`` or
    ``"skipped: <reason>"``. Callers can stash this dict into a
    manifest's ``extra`` field to record the seeding coverage
    actually achieved by a particular run.
    """
    coverage: SeedCoverage = {
        "random": _seed_random(seed),
        "numpy": _seed_numpy(seed),
        "torch": _seed_torch(seed),
        "transformers": _seed_transformers(seed),
    }
    if log:
        logger.info("set_global_seed(%d): %s", seed, coverage)
    return coverage

"""Capture lightweight runtime info for inclusion in metrics.json.

The goal is *audit-after-the-fact* (six months later, "which version of
bm25s produced this MRR@10?") not exhaustive provenance — we deliberately
keep the surface small and never let this raise.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - belt-and-braces
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def capture_environment(
    package_names: tuple[str, ...] = (
        "bm25s",
        "ir_datasets",
        "rank_bm25",
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "rouge_score",
        "sacrebleu",
    ),
) -> dict:
    """Return a small dict describing the current runtime.

    Always returns a dict; never raises. Unknown packages are silently dropped.
    """
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "git_commit": _git_commit(),
        "packages": {
            pkg: ver
            for pkg in package_names
            if (ver := _safe_version(pkg)) is not None
        },
    }

"""Capture lightweight runtime info for inclusion in metrics.json.

The goal is *audit-after-the-fact* (six months later, "which version of
bm25s produced this MRR@10?") not exhaustive provenance — we deliberately
keep the surface small and never let this raise.
"""

from __future__ import annotations

import os
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


def _cpu_brand() -> str | None:
    """Best-effort CPU brand string (e.g. "Apple M2 Pro"). Never raises."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            return out.strip() or None
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            return None
    return None


def _total_mem_bytes() -> int | None:
    """Best-effort total RAM in bytes. Never raises."""
    # Prefer psutil when available (cross-platform, well-tested); fall
    # back to platform-specific kernel queries otherwise.
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            return int(out.strip())
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Example line: "MemTotal:       16384000 kB"
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            return None
    return None


def capture_environment(
    package_names: tuple[str, ...] = (
        "bm25s",
        "ir_datasets",
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "rouge_score",
        "sacrebleu",
        # Added in infra/reproducibility-round1 (infra(deps) commit):
        # these are runtime-load-bearing for retrieval / generation /
        # BERTScore / arrow-backed dataset I/O and were previously
        # invisible to the manifest.
        "sentence-transformers",
        "bert-score",
        "faiss-cpu",
        "pyarrow",
    ),
) -> dict:
    """Return a small dict describing the current runtime.

    Always returns a dict; never raises. Unknown packages are silently dropped.
    """
    mem_bytes = _total_mem_bytes()
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "git_commit": _git_commit(),
        "cpu": {
            "brand": _cpu_brand(),
            "logical_count": os.cpu_count(),
        },
        "mem_gb": round(mem_bytes / 1024**3, 1) if mem_bytes is not None else None,
        "packages": {
            pkg: ver
            for pkg in package_names
            if (ver := _safe_version(pkg)) is not None
        },
    }

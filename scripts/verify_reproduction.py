"""Verify that a recorded run's manifest re-identifies the run.

Usage:
    python scripts/verify_reproduction.py outputs/bm25_baseline
    python scripts/verify_reproduction.py outputs/dense_retrieval outputs/cross_encoder_rerank

Checks (any failure → exit 1):

1. Manifest schema = msmarco-genqa.manifest.v2.
2. All six REQUIRED_FIELDS populated (non-None).
3. resolved_config.yaml roundtrips to the recorded resolved_config_hash.
4. metrics.json on disk has the sha256_16 recorded in manifest.outputs.
5. git.commit matches HEAD (warning, not error, on mismatch).

The captured headline metrics from metrics.json are also printed for
visual inspection.

Exit code: 0 on PASS (all checks across all output dirs); 1 if any
check failed in any dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from msmarco_genqa.util.manifest import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    compute_resolved_config_hash,
)


def _git_head_short() -> str | None:
    """Current HEAD as a 12-char short SHA, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _file_sha256_16(path: Path) -> str | None:
    """Recompute the 16-char truncated sha256 of a file, matching the
    manifest file-record digest convention."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_MISSING = object()


def _get_dotted(d: dict, dotted: str) -> Any:
    """Walk a dotted path into a nested dict; return _MISSING if any
    intermediate key is absent. Distinguishes 'absent' from
    'present-but-None' the same way the manifest module's validator does."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _truncate(value: Any, n: int = 28) -> str:
    """Truncate a value's string form for one-line display."""
    s = str(value)
    return s if len(s) <= n else s[: n - 3] + "..."


def verify_one(output_dir: Path) -> int:
    """Verify a single recorded run. Return the number of FAIL checks
    (0 = passed). WARN-level findings do not count toward failures."""
    failures = 0
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  FAIL: manifest not found at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())

    # 1. Schema
    schema = manifest.get("schema")
    if schema == SCHEMA_VERSION:
        print(f"  PASS: schema = {SCHEMA_VERSION}")
    else:
        print(f"  FAIL: schema = {schema!r}, expected {SCHEMA_VERSION!r}")
        failures += 1

    # 2. Required fields
    for field in REQUIRED_FIELDS:
        v = _get_dotted(manifest, field)
        if v is _MISSING:
            print(f"  FAIL: required field {field} is absent")
            failures += 1
        elif v is None:
            print(f"  FAIL: required field {field} is None")
            failures += 1
        else:
            print(f"  PASS: {field} = {_truncate(v)}")

    # 3. resolved_config.yaml roundtrip
    resolved_yaml_path = output_dir / "resolved_config.yaml"
    if not resolved_yaml_path.exists():
        print(f"  FAIL: resolved_config.yaml missing at {resolved_yaml_path}")
        failures += 1
    else:
        import yaml as _yaml

        reloaded = _yaml.safe_load(resolved_yaml_path.read_text())
        recomputed = compute_resolved_config_hash(reloaded)
        recorded = _get_dotted(manifest, "extra.resolved_config_hash")
        if recorded is _MISSING or recorded is None:
            print("  FAIL: extra.resolved_config_hash missing — cannot cross-check yaml")
            failures += 1
        elif recomputed == recorded:
            print(f"  PASS: resolved_config.yaml hash matches ({_truncate(recorded, 20)})")
        else:
            print("  FAIL: resolved_config.yaml hash drift")
            print(f"    recorded:   {recorded}")
            print(f"    recomputed: {recomputed}")
            failures += 1

    # 4. metrics.json sha256_16 vs recorded
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        actual = _file_sha256_16(metrics_path)
        recorded = None
        for rec in manifest.get("outputs", []):
            if rec.get("path", "").endswith("metrics.json"):
                recorded = rec.get("sha256_16")
                break
        if recorded is None:
            print("  WARN: metrics.json record not found in manifest.outputs")
        elif actual == recorded:
            print(f"  PASS: metrics.json sha256_16 matches ({recorded})")
        else:
            print(f"  FAIL: metrics.json hash drift (disk={actual}, recorded={recorded})")
            failures += 1
    else:
        print(f"  WARN: metrics.json not found at {metrics_path}")

    # 5. git.commit vs HEAD (warn-not-fail)
    head = _git_head_short()
    recorded_commit = _get_dotted(manifest, "git.commit")
    if head and recorded_commit not in (_MISSING, None):
        if head == recorded_commit:
            print(f"  PASS: git.commit matches current HEAD ({head})")
        else:
            print(f"  WARN: git.commit = {recorded_commit}, current HEAD = {head}")

    # Headline metrics for visual inspection.
    if metrics_path.exists():
        metrics_blob = json.loads(metrics_path.read_text()).get("metrics", {})
        headline = _extract_headline_metrics(metrics_blob)
        if headline:
            print()
            print("  Headline metrics (recorded):")
            for k, v in headline.items():
                if isinstance(v, (int, float)):
                    print(f"    {k:24s} = {v:.4f}")
                else:
                    print(f"    {k:24s} = {v}")

    return failures


def _extract_headline_metrics(metrics_blob: Any) -> dict[str, Any]:
    """Pull out scalar metric values whose key contains a known headline
    needle (mrr, ndcg, recall, f1, rouge, bleu, em, bert).

    Recurses one level into nested dicts so the dense runner's
    ``{"dense": {...}, "bm25_sample": {...}}`` shape is flattened with a
    namespaced key (``dense.mrr@10``) — without losing apples-to-apples
    comparability when both arms are reported on the same sample.
    """
    needles = ("mrr", "ndcg", "recall", "f1", "rouge", "bleu", "em", "bert")
    out: dict[str, Any] = {}
    if not isinstance(metrics_blob, dict):
        return out
    for k, v in metrics_blob.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if any(n in kk.lower() for n in needles) and isinstance(vv, (int, float)):
                    out[f"{k}.{kk}"] = vv
        elif any(n in k.lower() for n in needles) and isinstance(v, (int, float)):
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dirs",
        type=Path,
        nargs="+",
        help="One or more run output dirs (each containing manifest.json).",
    )
    args = parser.parse_args()

    total_failures = 0
    for out in args.output_dirs:
        print(f"Verifying {out} ...")
        n = verify_one(out)
        if n == 0:
            print(f"  → OK ({out} is reproducible: manifest + artefacts consistent)")
        else:
            print(f"  → FAIL ({n} check(s) failed)")
        total_failures += n
        print()

    if total_failures == 0:
        print(f"All checks PASSED across {len(args.output_dirs)} run(s).")
        return 0
    print(f"FAIL: {total_failures} check(s) failed in total.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

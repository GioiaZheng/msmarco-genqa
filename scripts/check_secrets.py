#!/usr/bin/env python3
"""Scan tracked files for high-confidence secret patterns."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 2_000_000

SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "vendor",
}

SKIP_EXTENSIONS = {
    ".7z",
    ".bin",
    ".bmp",
    ".dll",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mp4",
    ".onnx",
    ".pdf",
    ".png",
    ".pt",
    ".pyc",
    ".safetensors",
    ".so",
    ".sqlite",
    ".tar",
    ".tif",
    ".tiff",
    ".webp",
    ".zip",
}

ALLOWLIST_HINTS = re.compile(
    r"(dummy|example|fake|placeholder|redacted|sample|test[-_ ]?only|"
    r"your[-_ ]?(api[-_ ]?)?(key|secret|token)|xxxxx|<[^>]+>)",
    re.IGNORECASE,
)

PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{80,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("Stripe live secret key", re.compile(r"\bsk_live_[0-9A-Za-z]{24,}\b")),
    ("sk-prefixed API key", re.compile(r"\bsk-[A-Za-z0-9]{48,}\b")),
]


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [Path(name) for name in result.stdout.decode().split("\0") if name]


def should_skip(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    try:
        return path.stat().st_size > MAX_FILE_BYTES
    except OSError:
        return True


def scan_file(path: Path) -> list[tuple[int, str]]:
    data = path.read_bytes()
    if b"\0" in data:
        return []

    findings: list[tuple[int, str]] = []
    for line_no, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        if ALLOWLIST_HINTS.search(line):
            continue
        for label, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((line_no, label))
    return findings


def main() -> int:
    findings: list[tuple[Path, int, str]] = []
    for path in tracked_files():
        if should_skip(path):
            continue
        findings.extend((path, line_no, label) for line_no, label in scan_file(path))

    if not findings:
        print("No high-confidence secret patterns found.")
        return 0

    print("High-confidence secret patterns were found:")
    for path, line_no, label in findings:
        print(f"{path.as_posix()}:{line_no}: {label}")
    print("Remove the secret or replace it with an inert placeholder before committing.")
    return 1


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    sys.exit(main())

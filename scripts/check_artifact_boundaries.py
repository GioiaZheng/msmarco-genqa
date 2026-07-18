"""Check that large generated artifacts stay out of Git."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DISALLOWED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".faiss",
    ".feather",
    ".h5",
    ".hdf5",
    ".index",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}

DATA_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "data/cache/",
)
ARTIFACT_POINTER_PREFIX = "artifacts/"

ALLOWED_EXACT = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/cache/.gitkeep",
    "outputs/.gitkeep",
    "reports/generated/.gitkeep",
}


def tracked_files(project_root: Path, *, git_executable: str = "git") -> list[str]:
    """Return tracked repository paths using forward slashes."""
    result = subprocess.run(
        [git_executable, "ls-files", "-z"],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [
        entry.replace("\\", "/")
        for entry in result.stdout.decode("utf-8").split("\0")
        if entry
    ]


def is_allowed_output_pointer(path: str) -> bool:
    """Return whether a tracked outputs/ file is intentionally small metadata."""
    return path == "outputs/.gitkeep" or path.endswith("/provenance.backfill.json")


def check_paths(
    paths: list[str],
    *,
    project_root: Path,
    max_pointer_bytes: int,
) -> list[str]:
    """Return artifact-boundary violations for tracked paths."""
    errors: list[str] = []

    for path in paths:
        if path in ALLOWED_EXACT:
            continue

        file_path = project_root / path
        suffix = file_path.suffix.lower()

        if path.startswith(ARTIFACT_POINTER_PREFIX):
            if path != "artifacts/README.md" and suffix != ".json":
                errors.append(f"{path}: artifacts/ may contain only JSON pointers and README.md")
                continue
            if file_path.stat().st_size > max_pointer_bytes:
                errors.append(f"{path}: external artifact pointers must stay small")
            continue

        if path.startswith(DATA_PREFIXES):
            errors.append(f"{path}: data payloads must stay outside Git")
            continue

        if path.startswith("outputs/") and not is_allowed_output_pointer(path):
            errors.append(f"{path}: generated run payloads must stay outside Git")
            continue

        if suffix in DISALLOWED_SUFFIXES:
            errors.append(f"{path}: {suffix} artifacts must stay outside Git")
            continue

        if suffix == ".dvc" and file_path.stat().st_size > max_pointer_bytes:
            errors.append(f"{path}: DVC pointer files must stay small")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--max-pointer-bytes",
        type=int,
        default=250_000,
        help="Maximum size for pointer files such as .dvc records.",
    )
    parser.add_argument(
        "--git-executable",
        default=None,
        help="Git executable used to enumerate tracked files.",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    git_executable = (
        args.git_executable
        or os.environ.get("GIT_EXECUTABLE")
        or shutil.which("git")
        or "git"
    )
    errors = check_paths(
        tracked_files(project_root, git_executable=git_executable),
        project_root=project_root,
        max_pointer_bytes=args.max_pointer_bytes,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Artifact boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

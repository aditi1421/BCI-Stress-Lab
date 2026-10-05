"""Portable run artifacts and reproducibility records."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def source_metadata(project_root: Path, output: Path) -> dict:
    paths = sorted(project_root.glob("src/bci_stress/*.py"))
    paths += [project_root / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    fingerprints = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(project_root)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            fingerprints[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def git_value(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    repository_root = git_value("rev-parse", "--show-toplevel")
    own_repository = repository_root == str(project_root.resolve())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            package.metadata["Name"]: package.version
            for package in importlib.metadata.distributions()
            if package.metadata["Name"]
        },
        "git_commit": git_value("rev-parse", "HEAD") if own_repository else None,
        "git_dirty": bool(git_value("status", "--porcelain")) if own_repository else None,
        "source_sha256": fingerprints,
        "numeric_threads": 1,
        "random_generator": "numpy.random.PCG64",
    }

"""Download verified inputs or run the fixed clean/stress experiments."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .config import CleanConfig
from .experiment import run_experiment


def fetch_data(data_root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe path in dataset manifest")
        destination = data_root / relative
        if destination.exists():
            content = destination.read_bytes()
            if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Cached file fails pinned checksum: {relative}")
            print(f"Verified cached file: {relative}")
            continue
        url = f"https://s3.amazonaws.com/openneuro.org/ds004362/{relative.as_posix()}"
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Download fails pinned checksum: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(f"Downloaded and verified: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "run", "stress"):
        child = commands.add_parser(command)
        child.add_argument("--data-root", type=Path, default=Path("data/ds004362"))
        child.add_argument("--manifest", type=Path, default=Path("configs/sub-001-manifest.json"))
        if command in {"run", "stress"}:
            child.add_argument("--config", type=Path, default=Path("configs/clean.json"))
            child.add_argument("--output", type=Path, required=True)
        if command == "stress":
            child.add_argument("--stress-config", type=Path, default=Path("configs/stress.json"))
            child.add_argument("--baseline", type=Path, default=Path("reports/clean-sub001"))
    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch_data(arguments.data_root, arguments.manifest)
    elif arguments.command == "run":
        run_experiment(
            CleanConfig.load(arguments.config), arguments.data_root, arguments.manifest,
            arguments.output, Path(__file__).resolve().parents[2],
        )
    else:
        from .stress import run_stress
        from .stress_config import StressConfig

        run_stress(
            StressConfig.load(arguments.stress_config), CleanConfig.load(arguments.config),
            arguments.data_root, arguments.manifest, arguments.baseline,
            arguments.output, Path(__file__).resolve().parents[2],
        )


if __name__ == "__main__":
    main()

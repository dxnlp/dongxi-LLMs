#!/usr/bin/env python3
"""Capture a credential-free machine and Python environment manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COURSE_PACKAGES = (
    "accelerate",
    "datasets",
    "numpy",
    "peft",
    "torch",
    "transformers",
    "triton",
    "trl",
)


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def memory_snapshot() -> dict[str, float]:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    gib = 1024**2
    return {
        "total_gib": round(fields["MemTotal"] / gib, 2),
        "available_at_capture_gib": round(fields["MemAvailable"] / gib, 2),
        "swap_total_gib": round(fields["SwapTotal"] / gib, 2),
    }


def file_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Lock file does not exist: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {"path": str(resolved), "sha256": digest}


def git_identity(repo: Path) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"], cwd=repo)
    status = command_output(["git", "status", "--porcelain"], cwd=repo)
    return {
        "root": str(repo.resolve()),
        "commit": commit,
        "dirty": bool(status),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in COURSE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(repo: Path, lock_file: Path | None) -> dict[str, Any]:
    os_release = read_key_values(Path("/etc/os-release"))
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": git_identity(repo),
        "machine": {
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "os": os_release.get("PRETTY_NAME", platform.platform()),
            "gpu_and_driver": gpu,
            "cuda_toolkit": command_output(["nvcc", "--version"]),
            "memory": memory_snapshot(),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "packages": package_versions(),
        },
        "environment_lock": file_identity(lock_file),
        "context": {
            "hostname_recorded": False,
            "username_recorded": False,
            "credentials_recorded": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git repository whose code identity should be recorded.",
    )
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.repo, args.lock_file)
    rendered = json.dumps(manifest, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

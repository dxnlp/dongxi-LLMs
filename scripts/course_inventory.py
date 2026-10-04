#!/usr/bin/env python3
"""Print a source-hashed build inventory from an actual notebook verification run.

Read-only: no models, downloads, installations, or source/report writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook_manifest", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebooks = json.loads(args.notebook_manifest.read_text())
    paths = set()
    for pattern in ["book/**/*.md", "src/dongxi_llms/*.py", "scripts/*.py",
                    "tests/*.py", "notebooks/day-*/*.ipynb",
                    "notebooks/day-*/README.md", "experiments/configs/*",
                    "experiments/data/*", "fixtures/**/*"]:
        paths.update(p for p in root.glob(pattern) if p.is_file())
    for relative in ["README.md", "BOOK.md", "ROADMAP.md", "PROGRESS.md",
                     "LEARNING_MEMORY.md", "AGENTS.md", "pyproject.toml",
                     "notebooks/requirements-course.txt", "notebooks/README.md",
                     "docs/COURSE_BLUEPRINT.md", "docs/COURSE_SEQUENCE.md",
                     "docs/EXPERIMENT_MATRIX.md", "docs/NOTEBOOK_CURRICULUM.md",
                     "docs/RELEASE_CHECKLIST.md", "docs/handoffs/CURRENT.md",
                     "visuals/animations/COURSE_STORYBOARDS.md"]:
        paths.add(root / relative)
    versions = {}
    for name in ["torch", "numpy", "matplotlib", "nbformat", "nbclient",
                 "ipykernel", "jupyterlab", "PyYAML", "transformers",
                 "datasets", "peft", "safetensors"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    code_cells = sum(n["code_cells"] for n in notebooks["notebooks"])
    executed = sum(n["executed_code_cells"] for n in notebooks["notebooks"])
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete teaching material, not completed learner practice or Qwen campaign",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": versions},
        "inventory": {
            "chapters": len(list((root/"book/chapters").glob("*.md"))),
            "solutions": len(list((root/"book/solutions").glob("*.md"))),
            "labs": len(list((root/"book/labs").glob("*.md"))),
            "front_matter": len(list((root/"book/front-matter").glob("*.md"))),
            "appendices": len(list((root/"book/appendices").glob("*.md"))),
            "days": len(list((root/"notebooks").glob("day-*/README.md"))),
            "notebooks": len(notebooks["notebooks"]),
            "source_code_cells": code_cells, "executed_reference_cells": executed,
            "skipped_unfinished_exercise_cells": code_cells-executed,
            "image_outputs": sum(n["images"] for n in notebooks["notebooks"]),
            "chapter_words": sum(len(p.read_text().split()) for p in (root/"book/chapters").glob("*.md")),
        },
        "notebook_verification": notebooks,
        "notebook_manifest_sha256": sha256(args.notebook_manifest),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(paths)},
    }
    print(json.dumps(result, indent=None if args.compact else 2, allow_nan=False))


if __name__ == "__main__":
    main()

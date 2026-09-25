#!/usr/bin/env python3
"""Archive small verification evidence, leaving corpus and checkpoints untracked."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from dongxi_llms.stories_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH="src")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           capture_output=True, text=True, env=env, timeout=180)
    evidence = dict(schema=1, test_exit_code=tests.returncode,
                    test_output=tests.stdout+tests.stderr,
                    data=json.loads((Path(args.data)/"manifest.json").read_text()),
                    environment=json.loads(Path(args.environment).read_text()),
                    recovery=json.loads(Path(args.recovery).read_text()), runs={})
    evidence["additional_packages"] = {name: importlib.metadata.version(name)
                                       for name in ("tokenizers", "huggingface_hub", "requests")}
    for label, directory in (("uninterrupted", args.run), ("resumed", args.resume)):
        root = Path(directory)
        result = {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}
        result["metrics"] = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
        result["checkpoints"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                                  for p in root.glob("*.pt")}
        evidence["runs"][label] = result
    evidence["verification_sources"] = {str(p): digest(p) for p in (
        Path("scripts/train_stories.py"), Path("scripts/verify_stories_checkpoints.py"),
        Path("scripts/collect_stories_evidence.py"), Path("tests/test_stories_pipeline.py"))}
    with Path(args.output).open("x") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")
    print(f"Tests exit: {tests.returncode}; evidence: {args.output}")
    if tests.returncode:
        print(tests.stderr)
        raise SystemExit(tests.returncode)


if __name__ == "__main__":
    main()

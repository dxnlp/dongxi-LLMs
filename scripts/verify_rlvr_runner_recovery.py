#!/usr/bin/env python3
"""Independent fixed RLVR/identity CPU panel, not a pretrained-stage launcher."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from verify_runner_recovery import ROOT, memory_sample
from run_cpu_verification import execute
from dongxi_llms.run_identity import file_digest

PATTERNS = ("test_rlvr_runner_recovery.py", "test_run_identity.py")
SOURCE_NAMES = (
    "src/dongxi_llms/__init__.py", "src/dongxi_llms/training_snapshot.py",
    "src/dongxi_llms/run_identity.py", "src/dongxi_llms/qwen_rlvr_lab.py",
    "src/dongxi_llms/batched_cache_lab.py", "src/dongxi_llms/grpo_lab.py",
    "src/dongxi_llms/decoder_lab.py", "src/dongxi_llms/pretraining_lab.py",
    "src/dongxi_llms/course_manifest.py", "src/dongxi_llms/checkpoint_merge.py",
    "scripts/run_cpu_verification.py", "scripts/verify_runner_recovery.py",
    "scripts/verify_rlvr_runner_recovery.py", "uv.lock",
    *("tests/" + name for name in PATTERNS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    hashes = lambda: {name: file_digest(ROOT/name) for name in SOURCE_NAMES}
    report = dict(status="running", utc=datetime.now(timezone.utc).isoformat(),
        scope="Original random local HF CPU RLVR completed/pending recovery and actual identity/merge fixtures; not pretrained/CUDA/BF16, capability, Mac/hosted, full-course or learner evidence",
        executable=sys.executable, environment_prefix=sys.prefix,
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        packages={name: metadata.version(name) for name in ("torch", "transformers", "tokenizers", "peft")},
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes(), commands=[], memory_observations=[], reserve_bytes=25*1024**3,
        resource_scope="Actual MemAvailable before/after commands, not continuous interval minima; fixed60-second external timeouts",
        failure=None)
    path = output/"acceptance.json"

    def retain():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    started = time.perf_counter()
    retain()
    try:
        for index, pattern in enumerate(PATTERNS):
            observed = memory_sample()
            report["memory_observations"].append(observed)
            if observed["available_bytes"] < report["reserve_bytes"]:
                raise RuntimeError("Host reserve below25GiB; fixed command not launched")
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
            row = execute(command, output/f"command-{index+1}.txt", env=env, timeout=60)
            report["commands"].append(row)
            report["memory_observations"].append(memory_sample())
            retain()
            if row["exit_code"] != 0 or row["timed_out"]:
                raise RuntimeError(f"Fixed acceptance failed: {pattern}")
        report["source_hashes_after"] = hashes()
        report["sources_unchanged"] = report["source_hashes"] == report["source_hashes_after"]
        if not report["sources_unchanged"]:
            raise RuntimeError("Scoped source drift during fixed acceptance")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = dict(type=type(error).__name__, message=str(error))
    finally:
        report["seconds"] = time.perf_counter()-started
        report["minimum_observed_available_bytes"] = min(
            (row["available_bytes"] for row in report["memory_observations"]), default=None)
        retain()
    print(json.dumps(dict(status=report["status"], path=str(path), sha256=file_digest(path))))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

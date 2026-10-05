#!/usr/bin/env python3
"""Fixed original CPU runner-recovery acceptance; no pretrained/GPU launcher."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.run_identity import file_digest
from run_cpu_verification import execute

PATTERNS = ("test_training_snapshot.py", "test_training_snapshot_failure_paths.py",
            "test_sft_runner_recovery.py", "test_dpo_runner_recovery.py")
SOURCE_NAMES = (
    "src/dongxi_llms/__init__.py", "src/dongxi_llms/training_snapshot.py",
    "src/dongxi_llms/run_identity.py", "src/dongxi_llms/batched_cache_lab.py",
    "src/dongxi_llms/decoder_lab.py", "src/dongxi_llms/dpo_lab.py",
    "src/dongxi_llms/grpo_lab.py", "src/dongxi_llms/pretraining_lab.py",
    "src/dongxi_llms/course_manifest.py", "scripts/run_chapter09_spark_sft.py",
    "scripts/run_chapter11_spark_dpo.py", "scripts/run_cpu_verification.py",
    "scripts/verify_runner_recovery.py", *("tests/" + name for name in PATTERNS))


def memory_sample():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return {"utc": datetime.now(timezone.utc).isoformat(),
                    "available_bytes": int(line.split()[1]) * 1024,
                    "source": "actual-linux-MemAvailable"}
    raise RuntimeError("No actual MemAvailable observation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sources = [ROOT/name for name in SOURCE_NAMES]
    hashes = lambda: {str(p.relative_to(ROOT)): file_digest(p) for p in sources}
    report = {"status": "running", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Original tiny local HF/tensor CPU mechanisms only; no pretrained, CUDA/BF16 replay, capability, Mac/hosted or learner evidence",
              "executable": sys.executable, "environment_prefix": sys.prefix,
              "python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "packages": {name: metadata.version(name)
                  for name in ("torch", "transformers", "tokenizers", "peft")},
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
              "source_hashes": hashes(), "commands": [], "memory_observations": [],
              "source_scope": "Fixed core/SFT/DPO acceptance and imported local dependencies; excludes concurrent RLVR work and full-course acceptance",
              "resource_scope": "Actual host observations before/after each command, not continuous minima; each fixed command has a60second external timeout",
              "reserve_bytes": 25 * 1024**3, "failure": None}
    report_path = output / "acceptance.json"

    def retain():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    started = time.perf_counter()
    retain()
    try:
        for index, pattern in enumerate(PATTERNS):
            if not (ROOT / "tests" / pattern).is_file():
                raise FileNotFoundError(pattern)
            observation = memory_sample()
            report["memory_observations"].append(observation)
            if observation["available_bytes"] < report["reserve_bytes"]:
                raise RuntimeError("Host reserve below25GiB; test command not launched")
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
            row = execute(command, output / f"command-{index + 1}.txt", env=env, timeout=60)
            report["commands"].append(row)
            report["memory_observations"].append(memory_sample())
            retain()
            if row["exit_code"] != 0 or row["timed_out"]:
                raise RuntimeError(f"Fixed acceptance command failed: {pattern}")
        report["source_hashes_after"] = hashes()
        report["sources_unchanged"] = report["source_hashes"] == report["source_hashes_after"]
        if not report["sources_unchanged"]:
            raise RuntimeError("Executable source drifted during acceptance")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["seconds"] = time.perf_counter() - started
        report["minimum_observed_available_bytes"] = min(
            (row["available_bytes"] for row in report["memory_observations"]), default=None)
        retain()
    print(json.dumps({"status": report["status"], "path": str(report_path),
                      "sha256": file_digest(report_path), "commands": len(report["commands"])}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

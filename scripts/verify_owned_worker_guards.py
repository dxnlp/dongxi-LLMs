#!/usr/bin/env python3
"""Collect fixed CPU guard controls and retain actual exits, errors and hashes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms import owned_worker_guards as guards
from dongxi_llms.run_identity import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    sources = ["src/dongxi_llms/owned_worker_guards.py", "src/dongxi_llms/campaign_supervisor.py",
               "src/dongxi_llms/run_identity.py", "tests/test_owned_worker_guards.py",
               "tests/test_owned_worker_failure_paths.py", "scripts/verify_owned_worker_guards.py",
               "experiments/specs/2026-10-05-owned-workers-and-disk-guards.md"]
    before = {path: file_digest(ROOT / path) for path in sources}
    rows = []
    began = time.monotonic()
    report = {"schema": 1, "command": list(sys.orig_argv), "date_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "interpreter": sys.executable, "prefix": sys.prefix,
              "machine": platform.machine(), "platform": platform.platform(), "source_sha256": before,
              "scope": "fixed CPU resource/cleanup controls, not model jobs, cgroup containment or a production launcher",
              "status": "running", "controls": rows, "test_commands": [],
              "initial_diagnostic": {"path": "experiments/reports/2026-10-05-owned-worker-first-test-attempt.json",
                  "sha256": file_digest(ROOT / "experiments/reports/2026-10-05-owned-worker-first-test-attempt.json")}}
    def save():
        path = output / "acceptance.json"
        temporary = output / "acceptance.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    save()
    try:
        for mode in sorted(guards.MODES):
            kwargs = {}
            if mode in ("blocked-tree", "escaped-worker"):
                kwargs["limits"] = guards.Limits(seconds=.8)
            if mode == "aggregate-file-growth":
                kwargs["disk_limits"] = guards.DiskLimits(per_file_bytes=8192, aggregate_bytes=48 * 1024)
            result = guards.supervise_owned_fixture(mode, output / mode, **kwargs)
            rows.append({"id": mode, "result": result})
        def low_free(directory):
            return {"source": "injected-low-free-control-not-actual-shortage", "artifact_bytes": 0, "free_bytes": 0}
        rows.append({"id": "injected-preflight-free-stop", "result": guards.supervise_owned_fixture(
            "success", output / "injected-preflight-free-stop", disk_probe=low_free)})
        samples = 0
        def failed_observer(directory):
            nonlocal samples
            samples += 1
            if samples > 12: raise OSError("declared CPU disk-observer failure control")
            return guards.disk_sample(directory)
        rows.append({"id": "injected-runtime-observer-failure", "result": guards.supervise_owned_fixture(
            "escaped-worker", output / "injected-runtime-observer-failure", disk_probe=failed_observer)})
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for pattern in ("test_owned_worker*.py", "test_staged_campaign.py"):
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
            run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
            report["test_commands"].append({"command": command, "actual_exit_code": run.returncode,
                                            "stdout": run.stdout, "stderr": run.stderr})
        by_id = {row["id"]: row["result"] for row in rows}
        assertions = {
            "eight_fixed_controls_retained": len(rows) == 8,
            "all_known_owned_workers_stopped": all(not any(w["live"] for w in row["result"]["owned_final_states"]) for row in rows),
            "actual_success": by_id["success"]["actual_leader_exit_code"] == 0 and by_id["success"]["status"] == "completed",
            "deadline_worker_cleanup": all(by_id[mode]["stop_reason"] == "external-deadline"
                and by_id[mode]["actual_leader_exit_code"] == -signal.SIGKILL
                and len(by_id[mode]["owned_final_states"]) >= 2 for mode in ("blocked-tree", "escaped-worker")),
            "escaped_session_observed": any(w["observed_sid"] != by_id["escaped-worker"]["owned_final_states"][0]["pid"]
                for w in by_id["escaped-worker"]["owned_final_states"][1:]),
            "leader_success_not_safety_success": by_id["parent-exits-with-worker"]["actual_leader_exit_code"] == 0
                and by_id["parent-exits-with-worker"]["stop_reason"] == "leader-exited-with-owned-workers",
            "actual_hard_file_limit": by_id["single-file-overflow"]["actual_leader_exit_code"] == -signal.SIGXFSZ,
            "actual_sampled_aggregate_stop": by_id["aggregate-file-growth"]["stop_reason"] == "sampled-artifact-byte-cap",
            "injected_free_refusal_without_spawn": by_id["injected-preflight-free-stop"]["actual_leader_exit_code"] is None
                and by_id["injected-preflight-free-stop"]["stop_reason"] == "filesystem-free-below-threshold",
            "observer_failure_cleaned_up": by_id["injected-runtime-observer-failure"]["stop_reason"] == "supervisor-failure"
                and by_id["injected-runtime-observer-failure"]["actual_leader_exit_code"] is not None,
            "every_test_command_passes": all(row["actual_exit_code"] == 0 for row in report["test_commands"]),
        }
        report["assertions"] = assertions
        report["changed_sources_during_run"] = [path for path, old in before.items() if old != file_digest(ROOT / path)]
        report["status"] = "passed" if all(assertions.values()) and not report["changed_sources_during_run"] else "failed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:512]}
    report["elapsed_seconds"] = time.monotonic() - began
    save()
    print(json.dumps({"report": str(output / "acceptance.json"), "status": report["status"], "controls": len(rows)}))
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()

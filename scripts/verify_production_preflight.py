#!/usr/bin/env python3
"""Fixed injected preflight tests; no platform/model/backend launch."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dongxi_llms.run_identity import file_digest
from run_cpu_verification import execute

SOURCES = ("src/dongxi_llms/production_preflight.py", "tests/test_production_preflight.py",
           "src/dongxi_llms/run_identity.py", "src/dongxi_llms/staged_campaign.py",
           "scripts/verify_production_preflight.py", "scripts/run_cpu_verification.py",
           "experiments/specs/2026-10-05-production-preflight.md")


def memory_sample():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return {"utc":datetime.now(timezone.utc).isoformat(),"available_bytes":int(line.split()[1])*1024,
                    "source":"actual-linux-MemAvailable"}
    raise RuntimeError("No actual MemAvailable sample")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/"verification.json"
    hashes=lambda:{name:file_digest(ROOT/name) for name in SOURCES}
    report=dict(status="running",scope="Injected sanitized CPU backend observations only; no actual platform readiness or authorization",
                utc=datetime.now(timezone.utc).isoformat(),executable=sys.executable,python=platform.python_version(),
                platform=platform.platform(),source_hashes=hashes(),command=None,failure=None,memory_observations=[],
                resource_scope="Actual before/after test-command samples, not continuous; external command timeout60seconds")
    def retain():path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    started=time.perf_counter();retain()
    try:
        report["memory_observations"].append(memory_sample())
        if report["memory_observations"][-1]["available_bytes"] < 25*1024**3:
            raise RuntimeError("Host reserve below25GiB: command refused")
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        command=[sys.executable,"-m","unittest","discover","-s","tests","-p","test_production_preflight.py","-v"]
        report["command"]=execute(command,args.output/"command.txt",env=env,timeout=60)
        report["memory_observations"].append(memory_sample())
        report["source_hashes_after"]=hashes()
        report["sources_unchanged"]=report["source_hashes"]==report["source_hashes_after"]
        if report["command"]["exit_code"] != 0 or report["command"]["timed_out"] or not report["sources_unchanged"]:
            raise RuntimeError("Test command or source-stability gate failed")
        report["status"]="passed"
    except BaseException as error:
        report["status"]="failed";report["failure"]={"type":type(error).__name__,"message":str(error)}
    finally:
        report["seconds"]=time.perf_counter()-started
        report["minimum_observed_available_bytes"]=min((v["available_bytes"] for v in report["memory_observations"]),default=None)
        retain()
    print(json.dumps({"status":report["status"],"path":str(path),"sha256":file_digest(path)}))
    return 0 if report["status"]=="passed" else 1


if __name__ == "__main__":raise SystemExit(main())

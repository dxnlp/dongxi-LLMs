#!/usr/bin/env python3
"""Close the bounded guard-source evidence; no model or notebook launcher."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.run_identity import file_digest
from dongxi_llms.staged_campaign import prepare_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--selector-diagnostic", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report = args.report.absolute()
    if args.report.exists(): parser.error("Use a new report; retain prior evidence")
    cpu_path = ROOT / "experiments/reports/2026-10-05-guard-readiness-cpu/cpu-verification.json"
    cpu = json.loads(cpu_path.read_text())
    panel_paths = [ROOT / "experiments/reports/2026-10-05-guard-readiness-cpu/notebooks/manifest.json", args.targeted_manifest]
    panels = [json.loads(path.read_text()) for path in panel_paths]
    selected = [row for panel in panels for row in panel["notebooks"]]
    stale_sources = [path for path, digest in cpu["source_hashes"].items() if file_digest(ROOT / path) != digest]
    stale_notebooks = [row["path"] for row in selected if file_digest(ROOT / row["path"]) != row["sha256"]]
    ledger = json.loads((ROOT / "docs/course_improvements.json").read_text())
    counts = {name: sum(row["status"] == name for row in ledger["items"]) for name in ("complete", "in_progress", "planned")}
    missing_evidence = [path for row in ledger["items"] for path in row["evidence"]
                        if not (ROOT / path).exists() and path != str(args.report.relative_to(ROOT))]
    module_files = ["scripts/verify_guard_readiness_closure.py", "docs/course_improvements.json", "docs/PRODUCTION_RECOVERY_PLAN.md",
                    "book/chapters/06-pretraining-as-a-controlled-system.md", "book/solutions/06-pretraining-as-a-controlled-system.md",
                    "book/labs/06-reading-a-pretraining-run.md", "book/chapters/14-when-optimization-goes-wrong.md",
                    "book/solutions/14-when-optimization-goes-wrong.md", "book/labs/14-when-optimization-goes-wrong.md",
                    "learning_artifacts/day-09-pretraining-run-and-diagnosis/target-budget-is-a-boundary.md",
                    "experiments/reports/2026-10-05-course-guard-readiness.md"]
    closing_hashes = {path: file_digest(ROOT / path) for path in module_files}
    evidence_paths = ["experiments/reports/2026-10-05-story-valid-target-budget.json",
                      "experiments/reports/2026-10-05-owned-worker-final/acceptance.json",
                      "experiments/reports/2026-10-05-recovery-tensor-bytes-verification.json"]
    evidences = {path: json.loads((ROOT / path).read_text()) for path in evidence_paths}
    current_maps = [evidences[evidence_paths[0]]["source_sha256_after"],
                    evidences[evidence_paths[1]]["source_sha256"],
                    evidences[evidence_paths[2]]["current_artifact_sha256"]]
    stale_evidence = [path for mapping in current_maps for path, digest in mapping.items()
                      if file_digest(ROOT / path) != digest]
    lab_path = ROOT / "book/labs/06-reading-a-pretraining-run.md"
    blocks = re.findall(r"```python\n(.*?)```", lab_path.read_text(), flags=re.S)
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        for index, block in enumerate(blocks):
            exec(compile(block, f"{lab_path}:block{index + 1}", "exec"), namespace)
    # This source-only preparation must neither spawn jobs nor acquire weights.
    with patch("subprocess.Popen") as spawn, patch("subprocess.run") as run:
        campaign = prepare_campaign(ROOT)
    spawn.assert_not_called(); run.assert_not_called()
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    math = subprocess.run([sys.executable, "scripts/check_book_math.py"], cwd=ROOT,
                          capture_output=True, text=True, timeout=10)
    routes = subprocess.run([sys.executable, "scripts/check_course_integrity.py", "--json"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assertions = {"all_integrated_commands_passed": cpu["status"] == "passed" and all(row["exit_code"] == 0 for row in cpu["commands"]),
                  "all_panel_sources_current": not stale_sources,
                  "all_current_evidence_hashes_match": not stale_evidence,
                  "all_notebook_hashes_current": not stale_notebooks,
                  "nine_disjoint_cpu_references": len(selected) == len({row["path"] for row in selected}) == 9,
                  "both_panels_passed": all(not panel["failures"] for panel in panels),
                  "all_kernel_prefixes_and_cuda_hidden": all(row["kernel_identity"]["prefix"] == str(Path(sys.prefix))
                      and row["kernel_identity"]["cuda_available"] is False for row in selected),
                  "four_evidence_lab_blocks_executed": len(blocks) == 4,
                  "all_ledger_evidence_present": not missing_evidence,
                  "goal_and_learner_unchanged": ledger["status"] == "active" and counts == {"complete": 13, "in_progress": 5, "planned": 0}
                      and ledger["summary"]["learner_day"] == 9,
                  "all45_external_rows_pending": len(campaign["stage_rows"]) == 45 and campaign["jobs_started"] == 0
                      and campaign["actual_genealogy"] == []
                      and all(row["status"] == "pending-external-evidence" and all(value is None for value in row["actual"].values())
                              for row in campaign["stage_rows"]),
                  "closing_checks_passed": all(row.returncode == 0 for row in (diff, math, routes)),
                  "closing_prose_unchanged": closing_hashes == {path: file_digest(ROOT / path) for path in module_files}}
    report = {"schema": 1, "date_utc": datetime.now(timezone.utc).isoformat(), "command": list(sys.orig_argv),
              "status": "passed" if all(assertions.values()) else "failed", "assertions": assertions,
              "scope": "Actual current bounded CPU/source checks; no all76 rerun, pretrained/Mac/hosted/GPU/learner evidence",
              "integrated_panel": {"path": str(cpu_path), "sha256": file_digest(cpu_path), "actual_commands": cpu["commands"],
                                   "executable_sources": len(cpu["source_hashes"]), "stale_sources": stale_sources},
              "notebook_manifests": [{"path": str(path), "sha256": file_digest(path), "contents": panel}
                                     for path, panel in zip(panel_paths, panels)],
              "notebook_totals": {"notebooks": len(selected), "executed_cells": sum(row["executed_code_cells"] for row in selected),
                  "code_cells": sum(row["code_cells"] for row in selected), "images": sum(row["images"] for row in selected),
                  "skipped_unfinished_cells": sum(row["code_cells"] - row["executed_code_cells"] for row in selected)},
              "selector_diagnostic": {"sha256": file_digest(args.selector_diagnostic), "contents": json.loads(args.selector_diagnostic.read_text())},
              "closing_source_sha256": closing_hashes, "evidence_sha256": {path: file_digest(ROOT / path) for path in evidence_paths},
              "stale_evidence_hashes": stale_evidence, "ledger_counts": counts,
              "evidence_lab": {"blocks_executed": len(blocks), "captured_output": output.getvalue()},
              "closing_commands": [{"command": row.args, "actual_exit_code": row.returncode, "stdout": row.stdout, "stderr": row.stderr}
                                   for row in (diff, math, routes)]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle: handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "status": report["status"], "notebooks": report["notebook_totals"]}))
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()

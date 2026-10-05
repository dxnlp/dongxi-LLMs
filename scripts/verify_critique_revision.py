#!/usr/bin/env python3
"""Full offline replay plus source, budget, failure and fresh-kernel acceptance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.critique_revision_lab import digest, run_campaign, stable_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    parser.add_argument("--notebook-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Choose a new acceptance report; failed or historical evidence must remain")
    notebook = "notebooks/day-26/04_critique_revision_and_acceptance.ipynb"
    files = ["src/dongxi_llms/critique_revision_lab.py", "tests/test_critique_revision_lab.py",
        "scripts/verify_critique_revision.py", "fixtures/critique-revision/protocol.json",
        "fixtures/critique-revision/adversarial.json", "fixtures/critique-revision/README.md",
        "experiments/specs/2026-10-04-critique-revision.md", "notebooks/day-26/README.md", notebook,
        "experiments/specs/2026-10-04-critique-revision-hardening.md",
        "experiments/specs/2026-10-04-critique-revision-hardening-final.md",
        "src/dongxi_llms/inference_selection_lab.py", "src/dongxi_llms/reasoning_evaluation.py",
        "src/dongxi_llms/reasoning_generation.py", "src/dongxi_llms/evaluation_lab.py"]
    files += [f"notebooks/figures/chapter-15/day-26-04_critique_revision_and_acceptance-{i:02d}.png"
              for i in range(1, 7)]
    before = {p: digest(ROOT/p) for p in files}
    manifest = json.loads(args.notebook_manifest.read_text())
    nb = next((n for n in manifest["notebooks"] if n["path"] == notebook), None)
    if (manifest["failures"] or nb is None or nb["sha256"] != before[notebook] or nb["images"] != 6):
        parser.error("Matching passed fresh-kernel notebook and six figures required")
    reference = json.loads(args.reference.read_text())
    for path, expected in reference["source_identity"]["source_and_input_sha256"].items():
        if digest(ROOT/path) != expected:
            parser.error(f"Frozen campaign source/input differs: {path}")
    began = time.perf_counter()
    fresh = run_campaign(ROOT, ROOT/"fixtures/critique-revision/protocol.json")
    saved = {k: v for k, v in reference.items() if k not in ("source_identity", "inputs_unchanged", "content_sha256")}
    same_content = stable_payload(fresh) == stable_payload(saved)
    historical = json.loads(args.historical_reference.read_text())
    historical_saved = {k: v for k, v in historical.items() if k not in ("source_identity", "inputs_unchanged", "content_sha256")}
    same_historical = stable_payload(fresh) == stable_payload(historical_saved)
    source_path = ROOT/reference["protocol"]["source_responses"]
    original = [json.loads(line) for line in source_path.read_text().splitlines()]
    same_raw = fresh["source_candidates"] == reference["source_candidates"] == original
    exact = [c for c in fresh["cases"] if c["panel"] == "exact-programmatic-control"]
    actual = [c for c in fresh["cases"] if c["panel"] == "actual-candidate-replay"]
    endpoint = {r["fixture"]["id"]: r["loop"]["final_transition"]["category"] for r in fresh["adversarial"]}
    adversarial_counts = {
        "critique_failures": sum(r["callback_costs"][0]["error"] is not None
            for c in fresh["adversarial"] for r in c["loop"]["rounds"]),
        "revision_failures": sum(r["callback_costs"][1]["error"] is not None
            for c in fresh["adversarial"] for r in c["loop"]["rounds"]),
        "rejected_proposals": sum(not r["acceptance"]["accepted"]
            for c in fresh["adversarial"] for r in c["loop"]["rounds"])}
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checks = []
    for command in ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_critique_revision_lab.py", "-v"],
                    [sys.executable, "scripts/check_book_math.py"]):
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        checks.append({"command": command, "exit_code": completed.returncode,
                       "stdout": completed.stdout, "stderr": completed.stderr})
    after = {p: digest(ROOT/p) for p in files}
    passed = (same_content and same_historical and same_raw and before == after and len(original) == 864
        and len(exact) == 162 and len(actual) == 324 and len(fresh["adversarial"]) == 12
        and all(c["independent"]["exact_serialized_match"] for c in exact)
        and endpoint["wrong-to-right"] == "wrong_to_right" and endpoint["right-to-wrong"] == "right_to_wrong"
        and endpoint["double-flip"] == "unchanged" and adversarial_counts == {
            "critique_failures": 2, "revision_failures": 1, "rejected_proposals": 6}
        and fresh["new_model_forwards"] == 0 and all(c["exit_code"] == 0 for c in checks))
    evidence = {"schema_version": 1, "package": "DXI-06", "status": "passed" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(sys.orig_argv),
        "source_sha256": before, "source_changes_during_run": [p for p in before if before[p] != after[p]],
        "reference": {"path": str(args.reference), "sha256": digest(args.reference)},
        "historical_reference": {"path": str(args.historical_reference), "sha256": digest(args.historical_reference)},
        "historical_numeric_paths_and_errors_exactly_replayed": same_historical,
        "source_responses_sha256": digest(source_path), "source_candidates": len(original),
        "raw_source_candidates_and_historical_costs_identical": same_raw,
        "all_rounds_decisions_failures_budgets_and_counts_replayed": same_content,
        "exact_programmatic_cases": len(exact), "exact_programmatic_matches": sum(c["independent"]["exact_serialized_match"] for c in exact),
        "actual_replay_cases": len(actual), "actual_exact_serialized_matches": sum(c["independent"]["exact_serialized_match"] for c in actual),
        "actual_whole_attempt_overrun_tokens": sum(c["independent"]["overshoot"] for c in actual),
        "actual_underfill_tokens": sum(c["independent"]["underfill"] for c in actual),
        "authored_adversarial_counts": adversarial_counts, "authored_endpoints": endpoint,
        "actual_verification_wall_seconds": time.perf_counter() - began, "new_model_forwards": 0,
        "notebook_manifest_sha256": digest(args.notebook_manifest), "notebook_verification": nb, "checks": checks,
        "comparison_boundary": "New callback/campaign wall time and run identity excluded; all raw continuations, token IDs, accept/reject/stop decisions, grade states, errors, counts, budgets and source cost fields retained. Original source responses and historical costs compared exactly.",
        "limits": fresh["limits"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "status": evidence["status"], "cases": len(fresh["cases"]),
                      "seconds": evidence["actual_verification_wall_seconds"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

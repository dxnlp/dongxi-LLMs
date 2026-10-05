#!/usr/bin/env python3
"""Full bounded numerical replay with immutable raw programmatic teacher evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dongxi_llms.teacher_data_lab import run_reference, load_protocol, validate_digest, VOCAB, SPECIAL
from dongxi_llms.run_identity import file_digest


def measured_semantics(obj):
    """Ignore real durations and their content-derived digests, never counts/IDs."""
    excluded = {"elapsed_seconds","actual_elapsed_seconds","payload_sha256","attempt_sha256",
                "identity","pool_id","dataset_id"}
    if isinstance(obj,dict):
        return {k:("dongxi_llms.teacher_data_lab:programmatic_teacher"
                   if k == "actual_executor" and v == "__main__:programmatic_teacher" else measured_semantics(v))
                for k,v in obj.items() if k not in excluded}
    if isinstance(obj,list): return [measured_semantics(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--notebook-manifest",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists(): parser.error("Unused acceptance report required")
    notebook = "notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb"
    files = ["src/dongxi_llms/teacher_data_lab.py","tests/test_teacher_data_lab.py","scripts/verify_teacher_data.py",
        "src/dongxi_llms/sft_lab.py","src/dongxi_llms/instruction_data_lab.py","src/dongxi_llms/decoder_lab.py",
        "src/dongxi_llms/run_identity.py","fixtures/teacher-data/protocol.json","fixtures/teacher-data/README.md",
        "experiments/specs/2026-10-04-teacher-data.md",notebook,"notebooks/day-11/README.md",
        "book/chapters/08-instruction-data-as-an-interface.md","book/solutions/08-instruction-data-as-an-interface.md",
        "book/labs/08-instruction-data-as-an-interface.md"]
    files += [f"notebooks/figures/chapter-08/day-11-04_teacher_attempts_and_matched_rejection_sft-{i:02d}.png" for i in range(1,7)]
    before = {p:file_digest(ROOT/p) for p in files}
    manifest = json.loads(args.notebook_manifest.read_text())
    row = next((r for r in manifest["notebooks"] if r["path"] == notebook),None)
    if manifest["failures"] or row is None or row["sha256"] != before[notebook] or row["images"] != 6:
        parser.error("Matching passed fresh notebook with six measured/reference figures required")
    reference = json.loads(args.reference.read_text())
    previous = reference["results"]
    raw_path = ROOT/previous["journal"]["path"]/"attempts.jsonl"
    if file_digest(raw_path) != previous["journal"]["attempts_sha256"]:
        parser.error("Historical teacher attempt bytes changed")
    original_attempts = [json.loads(line) for line in raw_path.read_text().splitlines()]
    for attempt in original_attempts: validate_digest(attempt)
    workspace = Path(tempfile.mkdtemp(prefix="dongxi-teacher-replay-"))
    began = time.perf_counter()
    actual = run_reference(ROOT,workspace/"journal")
    alias_source_and_contract_match = (before["src/dongxi_llms/teacher_data_lab.py"] ==
        reference["source_sha256"]["src/dongxi_llms/teacher_data_lab.py"] and
        actual["contract_id"] == previous["contract_id"])
    new_attempts = [json.loads(line) for line in (workspace/"journal/attempts.jsonl").read_text().splitlines()]
    comparisons = []
    for current,old in zip(actual["runs"],previous["runs"]):
        comparisons.append({"arm":current["arm"],"seed":current["seed"],
            "coordinates_match":current["arm"] == old["arm"] and current["seed"] == old["seed"],
            "fit_metrics_and_states_match":measured_semantics(current["fit"]) == measured_semantics(old["fit"]),
            "all_baseline_outputs_and_cost_counts_match":measured_semantics(current["baseline"]) == measured_semantics(old["baseline"]),
            "all_student_outputs_and_cost_counts_match":measured_semantics(current["after"]) == measured_semantics(old["after"]),
            "final_state_sha256":current["fit"]["final_state_sha256"]})
    regrading = []
    for run in actual["runs"]:
        for record in run["after"]["rows"]:
            expected = [VOCAB[w] for w in record["reference"].split()]+[SPECIAL["end"]]
            regrading.append(record["correct"] == (record["error"] is None and record["generated_ids"] == expected))
    env = dict(os.environ,PYTHONPATH=str(ROOT/"src"),CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    checks = []
    for command in ([sys.executable,"-m","unittest","discover","-s","tests","-p","test_teacher_data_lab.py","-v"],
                    [sys.executable,"scripts/check_book_math.py"]):
        completed = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        checks.append({"command":command,"exit_code":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr})
    after = {p:file_digest(ROOT/p) for p in files}
    same_attempts = measured_semantics(new_attempts) == measured_semantics(original_attempts)
    same_pool = measured_semantics(actual["pool"]) == measured_semantics(previous["pool"])
    same_selection = measured_semantics(actual["selection"]) == measured_semantics(previous["selection"])
    passed = (len(comparisons) == 9 and len(actual["runs"]) == len(previous["runs"]) and not actual["failures"]
        and alias_source_and_contract_match and same_attempts and same_pool and same_selection and before == after and all(regrading)
        and all(all(c[k] for k in ("coordinates_match","fit_metrics_and_states_match",
            "all_baseline_outputs_and_cost_counts_match","all_student_outputs_and_cost_counts_match")) for c in comparisons)
        and all(c["exit_code"] == 0 for c in checks))
    evidence = {"schema_version":1,"package":"DXI-09","status":"passed" if passed else "failed",
        "date_utc":datetime.now(timezone.utc).isoformat(),"command":list(sys.orig_argv),"source_sha256":before,
        "source_changes_during_run":[p for p in before if before[p] != after[p]],
        "reference":{"path":str(args.reference),"sha256":file_digest(args.reference),"raw_attempts_sha256":file_digest(raw_path)},
        "actual_replay_seconds":time.perf_counter()-began,"teacher_attempt_semantics_match":same_attempts,
        "known_executor_alias_source_and_contract_match":alias_source_and_contract_match,
        "pool_semantics_match":same_pool,"selection_semantics_match":same_selection,"comparisons":comparisons,
        "regraded_after_training_records":len(regrading),"regrading_all_match":all(regrading),
        "unique_actual_baseline_records":len({(r["seed"],p["item_id"],p["mode"],p["sample"]) for r in actual["runs"] for p in r["baseline"]["rows"]}),
        "student_updates":sum(r["fit"]["updates"] for r in actual["runs"]),
        "student_valid_target_presentations":sum(r["fit"]["valid_supervised_token_presentations"] for r in actual["runs"]),
        "notebook_manifest_sha256":file_digest(args.notebook_manifest),"notebook_verification":row,"checks":checks,
        "replay_workspace":str(workspace),"new_raw_attempts_sha256":file_digest(workspace/"journal/attempts.jsonl"),
        "comparison_boundary":"Actual durations and resulting content digests differ. Original -m records __main__:programmatic_teacher; imported replay records dongxi_llms.teacher_data_lab:programmatic_teacher. Only this exact known Python alias is normalized, with identical frozen source digest. Raw provenance strings remain intact. All attempt coordinates, text, IDs, errors, stop/cost counts, selection, model states, update metrics and outputs compared",
        "limitations":actual["limits"]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle: handle.write(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":evidence["status"],"arms":len(comparisons),"seconds":evidence["actual_replay_seconds"]}))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()

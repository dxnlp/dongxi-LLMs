"""Collect a new evidence snapshot; never overwrite preparation or snapshot01.

No model, Torch, hardware job, Git mutation or publication is performed.
Actual outputs/weights were already streamed by the independent reviewers.
This collector identifies those receipts and preserves every unrun stage row.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def binding(relative):
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


old = read("experiments/reports/2026-10-05-native-base-profile/current-campaign-evidence-01.json")
preparation = old["immutable_preparation"]
assert binding(preparation["path"])["sha256"] == preparation["sha256"]
assert len(old["rows"]) == 45
profile_review = read("experiments/reports/2026-10-05-native-base-profile/independent-review-01.json")
snapshot = json.loads(json.dumps(old))
row = next(row for row in snapshot["rows"] if row["id"] == "assistant-profile-base06")
old_child_files = dict(row["actual"]["checkpoint_files"])
policy_prefix = "outputs/native-base-profile-20261005-run01/policy/"
child_files = {item["path"][len(policy_prefix):]: item["sha256"]
               for item in profile_review["output_inventory"]
               if item["path"].startswith(policy_prefix)}
assert child_files["model.safetensors"] == profile_review["exported_model"]["sha256"]
assert child_files["model.safetensors"] != row["actual"]["parent_checkpoint_files"]["model.safetensors"]
row["actual"]["checkpoint_files"] = child_files
for previous, current in zip(old["rows"], snapshot["rows"]):
    if previous["id"] != "assistant-profile-base06":
        assert previous == current and all(value is None for value in current["actual"].values())

invocations = [
    ("full-replay-original", "2026-10-05-native-sft-full-replay/returned-supervision-1.json"),
    ("full-replay-resume-attempt01", "2026-10-05-native-sft-full-replay/returned-supervision-2.json"),
    ("full-replay-resume-retry02", "2026-10-05-native-sft-full-replay/returned-supervision-retry02.json"),
    ("lora-replay-original", "2026-10-05-native-sft-lora-replay/returned-supervision-1.json"),
    ("lora-replay-resume", "2026-10-05-native-sft-lora-replay/returned-supervision-2.json"),
    ("common-development-base", "2026-10-05-pretrained-evaluation-replay/base-returned-supervision.json"),
    ("common-development-sft20", "2026-10-05-pretrained-evaluation-replay/sft20-returned-supervision.json"),
    ("bf16-merge-failed", "2026-10-05-native-lora-merge-reload/returned-supervision.json"),
    ("fp32-merge-reload-passed", "2026-10-05-native-lora-fp32-merge-reload/returned-supervision.json"),
]
snapshot["auxiliary_invocations"] = []
for role, tail in invocations:
    path = "experiments/reports/" + tail
    receipt = read(path)
    snapshot["auxiliary_invocations"].append({
        "role": role, "receipt": binding(path),
        "actual_exit_code": receipt["actual_exit_code"],
        "seconds": receipt["child_seconds"], "status": receipt["status"],
        "stop_reason": receipt["stop_reason"],
        "minimum_sampled_available_bytes": receipt["minimum_sampled_available_bytes"],
    })
assert len(snapshot["auxiliary_invocations"]) == 9
snapshot["jobs_started"] = 1 + len(snapshot["auxiliary_invocations"])
snapshot["utc"] = datetime.now(timezone.utc).isoformat()
snapshot["prior_snapshot"] = binding("experiments/reports/2026-10-05-native-base-profile/current-campaign-evidence-01.json")
snapshot["collection_correction"] = {
    "field": "assistant-profile-base06.actual.checkpoint_files",
    "reason": "Snapshot01 accidentally copied Base file digests beside the child checkpoint path; snapshot02 uses independently streamed exported-policy inventory. Parent map remains separate. No old evidence is overwritten.",
    "old_child_map": old_child_files, "corrected_child_map": child_files,
    "evidence": binding("experiments/reports/2026-10-05-native-base-profile/independent-review-01.json"),
}
snapshot["scope"] = (
    "10 actual native/model invocations:1 profile and9 auxiliary recovery/evaluation/merge jobs, including2 failed attempts. "
    "Only the original profile campaign row has actual fields;44 unrun stage rows remain untouched/null. "
    "Auxiliary20-update/15-item/FP32-validation evidence does not backfill400-update, DPO, RLVR, story or publication stages."
)
snapshot["auxiliary_genealogy_evidence"] = [
    binding("outputs/native-sft-full-replay-20261005-original/policy/course-genealogy.json"),
    binding("outputs/native-sft-lora-replay-20261005-original/policy/course-genealogy.json"),
    binding("outputs/native-lora20-fp32-merged-20261005/course-genealogy.json"),
    binding("experiments/reports/2026-10-05-native-acceptance-independent.json"),
]
destination = HERE / "current-campaign-evidence-02.json"
with destination.open("x") as output:
    output.write(json.dumps(snapshot, indent=2) + "\n")
print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "jobs_started": snapshot["jobs_started"], "rows": len(snapshot["rows"]),
                  "populated_campaign_rows": 1, "correction_retained": True}))

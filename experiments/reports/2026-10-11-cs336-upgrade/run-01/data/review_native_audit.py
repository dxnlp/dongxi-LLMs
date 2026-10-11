"""Read-only accounting and source checks; no model loading or new jobs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
REPORT = OUT.parent / "posttraining/native-audit-01/report.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads(REPORT.read_text())
rows = report["rows"]
items = json.loads((REPORT.parent / "frozen-items.json").read_text())
protocol = json.loads((REPORT.parent / "frozen-protocol.json").read_text())
for path, digest in report["source_sha256"].items():
    assert sha(ROOT / path) == digest
assert report["status"] == "completed"
assert report["completed_responses"] == len(rows) == report["requested_responses"] == 48
assert report["uncollected_responses"] == 0
assert protocol == report["protocol"]
assert items == json.loads((ROOT / "experiments/fixtures/cs336/posttraining_native_items.json").read_text())
lookup = {item["id"]: item for item in items}
seen = set()
for row in rows:
    item = lookup[row["item_id"]]
    key = row["item_id"], row["sample_index"]
    assert key not in seen
    seen.add(key)
    assert row["split"] == item["split"] and row["source_group"] == item["source_group"]
    expected = (item["a"] + item["b"]) * (item["c"] if item["operation"] == "multiply_sum" else 1)
    assert row["expected"] == item["expected"] == expected
    assert row["token_ids"] == [value for value, valid in zip(row["full_padded_response_ids"], row["response_mask"]) if valid]
    assert 1 <= sum(row["response_mask"]) <= 32
    stopped = row["original_stop"] == "eos"
    assert row["truncated"] == (not stopped)
    assert stopped == (row["token_ids"][-1] in report["interface"]["generation_stop_ids"])
    assert not any(token in report["interface"]["generation_stop_ids"] for token in row["token_ids"][:-1])
    assert row["strict_terminated_integer_reward"] is False

for attempt in report["attempts"]:
    group = [row for row in rows if row["item_id"] == attempt["item_id"]]
    assert len(group) == len(attempt["decode_attempts"]) == 4
    assert attempt["successful_multinomial_draws"] == 4 * len(attempt["sampled_steps"])
    assert attempt["successful_generation_forward_calls"] == len(attempt["sampled_steps"])
    assert attempt["successful_old_policy_forward_calls"] == 1
    gradient = next(g for g in report["gradients"] if g["item_id"] == attempt["item_id"])
    assert gradient["valid_scored_response_tokens"] == sum(sum(row["response_mask"]) for row in group)
    assert gradient["collection_work"]["valid_response_tokens"] == gradient["valid_scored_response_tokens"]
    assert gradient["collection_work"]["multinomial_draws"] == attempt["successful_multinomial_draws"]
    assert gradient["task_gradient_norm"] == gradient["advantage_norm"] == 0
    assert gradient["initial_log_ratio_max_abs"] == 0
    assert gradient["parent_state_sha256"] == report["parent_state_sha256_before"]

for key, summary in report["summary_by_split_tier"].items():
    split, tier = key.split("/")
    group = [row for row in rows if (row["split"], row["tier"]) == (split, tier)]
    assert summary["responses"] == len(group) == 8
    assert summary["natural_stops"] == sum(row["original_stop"] == "eos" for row in group)
    assert summary["truncated"] == sum(row["truncated"] for row in group)
    assert summary["statuses"] == dict(Counter(row["richer_status"] for row in group))
assert report["parent_state_sha256_before"] == report["parent_state_sha256_after"]
assert sum(sum(row["response_mask"]) for row in rows) == 1141 <= 12*4*32
assert report["runtime_seconds"] < 900 and report["peak_allocated_gib"] < 8
assert report["initial_host_available_gib"] >= 25

from dongxi_llms.corpus_curation_lab import load_fixture, duplicate_audit, split_audit, extract_text
fixture_path = ROOT / "experiments/fixtures/cs336/corpus.json"
fixture = load_fixture()
audit = duplicate_audit({row["id"]:extract_text(row["html"]) for row in fixture})
issues = split_audit(fixture, audit["components"])
assert not issues
splits = {row["id"]:row["split"] for row in fixture}
assert not any(splits[a] != splits[b] for a, b in audit["oracle_edges"])
learning = json.loads((OUT / "learning-reference.json").read_text())
assert learning["fixture_sha256"] == sha(fixture_path)

result = {
    "status":"accepted-with-explicit-evidence-boundaries",
    "scope":"Independent read-only review of native capability/reward audit source, original frozen tasks/protocol, tests and completed actual report; no model loading, training, new GPU job or source mutation",
    "report":str(REPORT.relative_to(ROOT)),"report_sha256":sha(REPORT),
    "source_sha256":report["source_sha256"],
    "test_source_sha256":sha(ROOT / "tests/test_posttraining_native_audit.py"),
    "mathematical_contract":{
        "loss":"-(1/G) sum_i A_i (sum_t m_it logpi(y_it|x,y_i,<t)/L_i), inclusive group mean/population std and detached advantages",
        "head_gradient":"For each token, A_i*m_it/(G*L_i) times (p_it−onehot(y_it)) outer detached hidden state; signs match minimization of negative reward-weighted log probability",
        "sampler":"Existing collector draws unfiltered full-vocabulary softmax at temperature1; no topk/topp/answer grammar. First natural stop enters mask and likelihood; forced padded EOS after termination is excluded",
        "precision":"Actual parent/head clone/hidden matmul is BF16; logits/logsoftmax use FP32, head-weight backward is represented in the clone's dtype, and returned gradient is cast to FP32 for aggregation. No full-FP32-gradient claim",
        "alignment":"Reported alignment checks head projection versus full model teacher-forced old-policy scoring. It does not independently attest bitwise equality to every generation-step kernel distribution",
        "reach":"Detached-hidden output-weight subspace only. Tied embedding/transformer pathways and KL gradients are excluded; no optimizer exists in this audit",
        "independent_tests":"CPU Qwen random fixture matches manually accumulated softmax outer products; constant0/1 rewards give exactzero; mismatched likelihood rejects; gate requires both splits and aggregate nonzero norm"},
    "ancestry":{
        "model_id":protocol["model_id"],"revision":protocol["revision"],"snapshot":report["snapshot"]["path"],
        "snapshot_file_hashes_recorded":True,"license_sha256":report["snapshot"]["license_sha256"],
        "parent_before":report["parent_state_sha256_before"],"parent_after":report["parent_state_sha256_after"],
        "all_twelve_collections_match_parent":True,"interface_sha256":report["interface"]["tokenizer"]["interface_sha256"],
        "thinking":False,"splits":["train","development"],"heldout_access":False,
        "boundary":"This cached native Qwen root is distinct from the compatible random-init CPU SFT/sibling chain and from historical trained native descendants; future same-parent SFT/RL remains unexecuted"},
    "accounting":{
        "requested_responses":48,"completed_responses":48,"uncollected":0,"groups":12,"strata":6,"responses_per_stratum":8,
        "valid_response_tokens":1141,"token_cap":1536,"multinomial_draws":1404,"generation_forward_calls":351,
        "old_policy_score_calls":12,"head_teacher_forced_input_positions":2604,
        "natural_stops":28,"token_limit":20,"decode_failures":0,"richer_statuses":{"UNSUPPORTED":48},
        "strict_successes":0,"mixed_groups":0,"per_group_task_head_gradient_norms":[g["task_gradient_norm"] for g in report["gradients"]],
        "aggregate_norms":report["aggregate_task_head_gradient_norm_by_split"],"signal_gate_passes":False,
        "runtime_seconds":report["runtime_seconds"],"peak_allocated_gib":report["peak_allocated_gib"],
        "initial_host_available_gib":report["initial_host_available_gib"],
        "failure_contract":"Decode exceptions yield strictreward0 and explicit ERROR rows, retained in requested/completed denominators. Unsupported/invalid parser outcomes remain rows. Collector/load exceptions preserve partial attempt counters through finally/atomic flush and stop; uncollected count remains separate. No retry or discarded-group selection"},
    "interpretation":{
        "accepted":"No usable task-relative update signal under this frozen strict reward/extraction interface; absence is the actual diagnostic result",
        "arithmetic_boundary":"All48 richer grades are UNSUPPORTED, so0 richer-correct is not established0 arithmetic capability. EOS texts visibly contain correct-looking equations/answers, such as2+3=5,−7+2=−5 and(3+4)*2=14, that these extraction contracts reject. Preserve strings/statuses; no silent parser relaxation or hindsight reward substitution",
        "future":"No same-parent SFT/RL pilot was executed or justified by this failed signal gate"},
    "corpus_preflight_applicability":{
        "fixture":str(fixture_path.relative_to(ROOT)),"fixture_sha256":sha(fixture_path),
        "learning_reference_matches_audited_fixture":True,"declared_source_and_verified_duplicate_split_issues":issues,
        "oracle_cross_split_duplicates":[],
        "decision":"Existing explicit whole-fixture split audit, immutable fixture hash, independent tests and notebook checkpoint4 before checkpoint6 are sufficient for this actual bounded lesson acceptance. No source edit or duplicate preflight infrastructure is necessary",
        "api_boundary":"learning_reference accepts a custom fixture path and does not itself promise to run the source/near-duplicate preflight. TrainingSession rejects IDs and normalized-exact text overlap only. A future changed fixture must rerun existing duplicate_audit+split_audit before interpreting training; an optional convenience call can reuse these functions rather than invent a second validator"},
    "open_source_defects":[]}
(OUT / "native-audit-peer-review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print("Native audit accepted:48responses/12groups/1141validtokens; hashes, masks, counts, ancestry and zero-gradient gate reconcile. Corpus audited hash matches learning reference; existing preflight sufficient.")

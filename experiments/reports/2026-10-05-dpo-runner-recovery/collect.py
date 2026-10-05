"""Exclusive bounded CPU collection using actual production DPO functions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch
import test_dpo_runner_recovery as fixture

r = fixture.runner
ROOT = fixture.ROOT


def generated_records(model, phase):
    tokenizer = fixture.tokenizer_fixture()
    rows = []
    model.eval()
    for index, prefix in enumerate(((7, 8), (7, 13))):
        started = time.perf_counter()
        row = dict(id=f"original-independent-{index}", phase=phase, prompt_ids=list(prefix),
                   source="authored random-model microscope; no expected language-quality target")
        try:
            with torch.no_grad():
                all_ids = model.generate(input_ids=torch.tensor([prefix]), attention_mask=torch.ones(1, len(prefix), dtype=torch.long),
                    do_sample=False, max_new_tokens=4, eos_token_id=1, pad_token_id=1, use_cache=False)[0].tolist()
            continuation = all_ids[len(prefix):]
            row.update(generated_ids=continuation, raw_continuation=tokenizer.decode(continuation, skip_special_tokens=False),
                stopped_token_id=continuation[-1] if continuation and continuation[-1] == 1 else None,
                stop_reason="eos" if continuation and continuation[-1] == 1 else "max_new_tokens",
                truncated=not (continuation and continuation[-1] == 1), error=None,
                measured_prompt_tokens=len(prefix), measured_generated_tokens=len(continuation),
                generation_forward_cost="HF internal forward work not measured")
        except Exception as error:
            row.update(generated_ids=None, raw_continuation=None, stopped_token_id=None,
                stop_reason="error", truncated=None, error=f"{type(error).__name__}: {error}",
                measured_prompt_tokens=len(prefix), measured_generated_tokens=None,
                generation_forward_cost="unknown partial HF generation work retained as unavailable")
        row["seconds"] = time.perf_counter()-started
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    source_names = ["scripts/run_chapter11_spark_dpo.py", "tests/test_dpo_runner_recovery.py",
        "src/dongxi_llms/training_snapshot.py", "src/dongxi_llms/dpo_lab.py",
        "src/dongxi_llms/batched_cache_lab.py", "src/dongxi_llms/run_identity.py", "uv.lock",
        "experiments/specs/2026-10-05-dpo-runner-recovery.md",
        str(Path(__file__).resolve().relative_to(ROOT))]
    sources = {name: r.file_digest(ROOT/name) for name in source_names}
    r.write_exclusive_json(output/"source-identities-before.json", sources)
    r.write_exclusive_json(output/"authored-fixture.json", fixture.FIXTURE)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    command = [sys.executable, "-m", "unittest", "test_dpo_runner_recovery", "-v"]
    started = time.perf_counter()
    test_process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    test_result = dict(command=command, cwd=str(ROOT), exit_code=test_process.returncode,
                       seconds=time.perf_counter()-started, stdout=test_process.stdout, stderr=test_process.stderr)
    r.write_exclusive_json(output/"focused-tests.json", test_result)
    if test_process.returncode:
        raise RuntimeError("Actual focused tests failed; outputs retained, no favorable rerun in this directory")

    def directory(name):
        result = output/name; result.mkdir(); return result

    full_components = fixture.build_fixture()
    contract = full_components[4]
    r.write_exclusive_json(output/"recovery-contract.json", contract)
    full_work = r.zero_work()
    started = time.perf_counter()
    full = r.train_completed_updates(*full_components[:4], fixture.ENCODED, contract=contract,
        output=directory("uninterrupted"), invocation_id="dpo-cpu-uninterrupted", pad_id=1, device="cpu",
        checkpoint_every=3, max_bytes=fixture.LIMIT, attempted_work=full_work,
        before_training=lambda: generated_records(full_components[0], "before"))
    full_seconds = time.perf_counter()-started
    full_after = r.finalize_after_commit(full, evaluate=lambda: generated_records(full_components[0], "after"), export=lambda: None)
    expected = fixture.payload_for(full["checkpoint"], contract)
    r.write_exclusive_json(output/"uninterrupted/result.json", dict(full, after=full_after, attempted_work=full_work, seconds=full_seconds))

    interrupted_components = fixture.build_fixture()
    interrupted_output = directory("interrupted")
    real_append = r.append_metric
    def fail_metric(path, row):
        if row.get("update") == 3:
            raise OSError("authored metric failure AFTER committed update3")
        return real_append(path, row)
    try:
        with patch.object(r, "append_metric", side_effect=fail_metric):
            fixture.train_fixture(interrupted_output, interrupted_components)
    except OSError as error:
        failure = dict(type=type(error).__name__, message=str(error), deliberately_injected=True,
                       durable_update=3, metric_rows_published=2)
    else:
        raise AssertionError("Declared failure control did not fail")
    r.write_exclusive_json(output/"interrupted/failure.json", failure)
    receipt = fixture.checkpoint_receipt(interrupted_output/"checkpoints/completed-000003.pt")
    r.write_exclusive_json(output/"independent-receipt.json", receipt)

    resumed_components = fixture.build_fixture()
    initial = fixture.restore_fixture(resumed_components, receipt)
    resumed_work = r.zero_work()
    started = time.perf_counter()
    resumed = r.train_completed_updates(*resumed_components[:4], fixture.ENCODED, contract=contract,
        output=directory("resumed"), invocation_id="dpo-cpu-resumed", pad_id=1, device="cpu",
        checkpoint_every=3, max_bytes=fixture.LIMIT, attempted_work=resumed_work, **initial)
    resumed_seconds = time.perf_counter()-started
    r.write_exclusive_json(output/"resumed/result.json", dict(resumed, attempted_work=resumed_work, seconds=resumed_seconds))
    actual = fixture.payload_for(resumed["checkpoint"], contract)

    fresh_command = [sys.executable, str(ROOT/"tests/test_dpo_runner_recovery.py"), "--fresh-replay",
        str(output/"independent-receipt.json"), str(output/"recovery-contract.json"), str(output/"fresh-process")]
    started = time.perf_counter()
    child = subprocess.run(fresh_command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
    r.write_exclusive_json(output/"fresh-process-execution.json", dict(command=fresh_command,
        exit_code=child.returncode, seconds=time.perf_counter()-started, stdout=child.stdout, stderr=child.stderr))
    if child.returncode:
        raise RuntimeError("Fresh child failed; exact outputs retained")
    fresh = json.loads((output/"fresh-process/result.json").read_text())
    fresh_payload = fixture.payload_for(fresh["checkpoint"], contract)
    expected_digest = fixture.digest(fixture.numerical_state(expected))
    equality = dict(same_process_numerical_state=expected_digest == fixture.digest(fixture.numerical_state(actual)),
        fresh_process_numerical_state=expected_digest == fixture.digest(fixture.numerical_state(fresh_payload)),
        next_row=full["history"][3] == resumed["history"][3] == fresh["history"][3],
        original_reference_unchanged=fixture.digest(expected["state"]["reference"]) == contract["reference_sha256"],
        original_reference_not_updated_policy=fixture.digest(expected["state"]["reference"]) != fixture.digest(expected["state"]["policy"]),
        exact_components_compared=["policy", "reference", "Adam moments/step", "sampler RNG", "global Torch RNG",
            "CUDA RNG empty in CPU scope", "completed cursor", "committed history", "work counters"],
        excluded_from_numeric_equality=["writer invocation ID", "resume parent receipt", "elapsed time", "output paths"],
        uninterrupted_state_digest=expected_digest)
    if not all(value for value in equality.values() if type(value) is bool):
        r.write_exclusive_json(output/"failed-comparison.json", equality)
        raise AssertionError("Declared exact recovery comparison failed")
    after_sources = {name: r.file_digest(ROOT/name) for name in source_names}
    r.write_exclusive_json(output/"source-identities-after.json", after_sources)
    if sources != after_sources:
        raise AssertionError("Source changed during collection; not accepted as frozen")
    artifacts = {str(path.relative_to(ROOT)): r.file_digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    verification = dict(schema="dongxi-dpo-actual-runner-cpu-verification-v1", status="pass",
        command=sys.orig_argv, source_sha256=sources, artifact_sha256=artifacts,
        test_count=21, focused_tests=test_result, equality=equality,
        snapshot_contract_sha256=full["checkpoint"]["contract_sha256"],
        contract_state_digest=fixture.digest(contract), fixture_state_digest=fixture.digest(fixture.FIXTURE),
        fixture_file_sha256=r.file_digest(output/"authored-fixture.json"),
        measured_training_seconds={"uninterrupted": full_seconds, "remaining_updates": resumed_seconds},
        measured_work={"uninterrupted_cumulative": full["counters"], "resumed_cumulative": resumed["counters"],
                       "uninterrupted_attempted_this_invocation": full_work, "resumed_attempted_this_invocation": resumed_work},
        completed_update_recipe=6, accumulation=2, selected_pairs=12,
        observed_failure=failure, independent_generated_records=full["baseline"]+full_after,
        scope="random local CPU production-function recovery, not pretrained/CUDA/quality/containment evidence",
        pending=["current-source pretrained compatibility", "CUDA/BF16 recovery/profile", "whole-job enforced resource envelope",
                 "legacy dataset independent source-group provenance", "cross-machine numerical determinism",
                 "model-scale save/load overhead", "HF generation internal forward-cost accounting"])
    r.write_exclusive_json(output/"verification.json", verification)
    print(json.dumps(dict(status="pass", tests=21, verification=str(output/"verification.json"), equality=equality), sort_keys=True))


if __name__ == "__main__":
    main()

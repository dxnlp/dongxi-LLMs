#!/usr/bin/env python3
"""Exclusive bounded CPU collection for the original matched chosen-SFT control."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"tests")]
PYTHON = "/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python"
NAMES = ("src/dongxi_llms/chosen_sft_control.py", "scripts/run_matched_chosen_sft.py",
    "scripts/verify_chosen_sft_control.py", "tests/test_chosen_sft_control.py",
    "fixtures/matched-chosen-sft/protocol.json",
    "experiments/specs/2026-10-05-matched-chosen-sft-control.md",
    "scripts/run_chapter09_spark_sft.py", "scripts/run_chapter11_spark_dpo.py",
    "src/dongxi_llms/dpo_lab.py", "src/dongxi_llms/sft_lab.py",
    "src/dongxi_llms/run_identity.py", "src/dongxi_llms/batched_cache_lab.py",
    "fixtures/chapter11/train.jsonl", "fixtures/chapter11/validation.jsonl",
    "fixtures/chapter11/evaluation.jsonl", "uv.lock")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def available():
    return next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def campaign(directory):
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM
    from test_chosen_sft_control import fixture, save_parent
    from dongxi_llms.chosen_sft_control import compare_results, encode_dataset, run_arm
    from dongxi_llms.run_identity import artifact_hashes, environment_identity
    from dongxi_llms.batched_cache_lab import digest
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    rows = []
    for seed in (1818, 1819):
        parent, tokenizer, dataset, settings = fixture(seed, updates=6)
        path = directory/f"parent-{seed}"
        save_parent(path, parent, tokenizer)
        parent_hashes = artifact_hashes(path)
        parent_digest = digest(parent.state_dict())
        # Every scientific arm starts from the same actual saved local bytes.
        loaded = Qwen3ForCausalLM.from_pretrained(path, local_files_only=True,
            dtype=torch.float32, attn_implementation="sdpa")
        observed_tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        if digest(loaded.state_dict()) != parent_digest:
            raise AssertionError("Actual save/reload changed the original random parent")
        dataset = encode_dataset(observed_tokenizer, *dataset["raw_splits"], max_length=64)
        results = {}
        for arm in ("unchanged", "chosen-sft", "dpo"):
            events = []
            policy, result = run_arm(loaded, dataset, observed_tokenizer, settings,
                arm=arm, row_sink=events.append)
            result["parent_artifacts"] = parent_hashes
            result["events"] = events
            export = directory/f"seed-{seed}-{arm}"
            policy.save_pretrained(export, safe_serialization=True)
            observed_tokenizer.save_pretrained(export)
            reloaded = Qwen3ForCausalLM.from_pretrained(export, local_files_only=True, dtype=torch.float32)
            if digest(reloaded.state_dict()) != result["final_policy_sha256"]:
                raise AssertionError("Actual final exported policy changed")
            result["final_export"] = artifact_hashes(export)
            results[arm] = result
        if artifact_hashes(path) != parent_hashes:
            raise AssertionError("Original saved parent was overwritten")
        rows.append(dict(seed=seed, arms=results, comparison=compare_results(results)))
    write(directory/"results.json", dict(schema="dongxi-matched-chosen-sft-observations-v1",
        origin="Original existing location literals, actual random local HF checkpoints, no pretrained weights",
        environment=environment_identity(ROOT/"uv.lock"), parameter_count=sum(p.numel() for p in loaded.parameters()),
        runs=rows, scope="Two fixed seeds/all three arms; no retuning or model-scale stage execution"))


def collect(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    before = {name:sha(ROOT/name) for name in NAMES}
    write(directory/"source-before.json", before)
    archive = directory/"original-input-archive"
    archive.mkdir()
    for name in NAMES:
        destination = archive/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    env = dict(os.environ, PYTHONPATH="src:tests", CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    commands = [
        ("focused-tests", [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_chosen_sft_control.py", "-v"]),
        ("campaign", [PYTHON, str(ROOT/"scripts/verify_chosen_sft_control.py"), "--campaign-child", str(directory/"campaign")])]
    entries = []
    for name, command in commands:
        started = time.monotonic()
        minimum = available()
        if minimum < 25*1024**3:
            raise RuntimeError("Sampled host reserve below25GiB before child")
        with (directory/f"{name}.stdout.txt").open("x") as stdout, (directory/f"{name}.stderr.txt").open("x") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            refusal = None
            while child.poll() is None:
                minimum = min(minimum, available())
                if minimum < 25*1024**3:
                    refusal = "sampled25GiB reserve breached"
                if time.monotonic()-started >= 60:
                    refusal = "external60-second child deadline"
                if refusal:
                    child.kill()
                    break
                time.sleep(.05)
            code = child.wait(timeout=5)
        entries.append(dict(name=name, command=command, pid=child.pid, actual_exit_code=code,
            elapsed_seconds=time.monotonic()-started, minimum_sampled_mem_available_bytes=minimum,
            memory_sampling_seconds=.05, refusal=refusal,
            stdout_sha256=sha(directory/f"{name}.stdout.txt"), stderr_sha256=sha(directory/f"{name}.stderr.txt")))
        write(directory/f"{name}-invocation.json", entries[-1])
        if code != 0 or refusal:
            break
    after = {name:sha(ROOT/name) for name in NAMES}
    write(directory/"source-after.json", after)
    artifacts = {str(path.relative_to(directory)):dict(sha256=sha(path), bytes=path.stat().st_size)
                 for path in directory.rglob("*") if path.is_file()}
    success = len(entries) == 2 and all(e["actual_exit_code"] == 0 and e["refusal"] is None for e in entries) and before == after
    result = dict(schema="dongxi-matched-chosen-sft-verification-v1", status="PASS" if success else "FAILED",
        collected_utc=datetime.now(timezone.utc).isoformat(), invocations=entries, source_sha256=before,
        source_hashes_unchanged=before == after, artifacts=artifacts,
        scope="Bounded CPU scientific source control; original45 model-scale rows remain unexecuted")
    write(directory/"verification.json", result)
    print(json.dumps(dict(status=result["status"], verification=str(directory/"verification.json"))))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--campaign-child":
        campaign(sys.argv[2])
    elif len(sys.argv) == 2:
        collect(sys.argv[1])
    else:
        raise SystemExit("Pass a new exclusive output directory")

#!/usr/bin/env python3
"""Bounded real random-HF CPU reference; writes a new evidence JSON, no acquisition."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_reasoning_generation import tiny_fixture, items, settings, ScriptedForward
from dongxi_llms.reasoning_generation import (generate_record, freeze_local_contract,
                                             ADAPTER_VERSION)
from dongxi_llms.reasoning_evaluation import replay_records
from dongxi_llms.run_identity import artifact_hashes, environment_identity, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="New evidence JSON; never overwrites")
    parser.add_argument("--notebook-manifest", type=Path, help="Optional actual fresh Day10 verification manifest")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Historical adapter evidence cannot be overwritten")
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix="dongxi-generation-reference-"))
    checkpoint = workspace / "random-hf"; checkpoint.mkdir()
    model, tokenizer = tiny_fixture(checkpoint)
    suite = workspace / "items.json"; suite.write_text(json.dumps(items(), indent=2) + "\n")
    commands, invocations = [], {}
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")

    def run(command):
        before = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=60)
        entry = {"command": command, "exit_code": result.returncode, "stdout": result.stdout,
                 "stderr": result.stderr, "seconds": time.perf_counter() - before}
        commands.append(entry)
        if result.returncode:
            raise RuntimeError(f"Verification failed: {entry}")
        return result.stdout

    for mode in ("raw", "chat"):
        config = workspace / f"{mode}-settings.json"; config.write_text(json.dumps(settings(mode=mode), indent=2) + "\n")
        contract = workspace / f"{mode}-contract.json"
        run([sys.executable, "scripts/generate_reasoning_records.py", "--items", str(suite),
             "--checkpoint", str(checkpoint), "--settings", str(config), "--freeze-contract", str(contract)])
        output = workspace / f"{mode}-run"
        run([sys.executable, "scripts/generate_reasoning_records.py", "--items", str(suite),
             "--checkpoint", str(checkpoint), "--contract", str(contract), "--output", str(output)])
        replay = json.loads(run([sys.executable, "scripts/evaluate_reasoning_records.py",
             "--items", str(suite), "--contract", str(contract), "--records", str(output / "responses.jsonl")]))
        saved = json.loads((output / "evaluation.json").read_text())
        if replay != saved:
            raise AssertionError("Offline CLI replay differs from actual generated ledger")
        invocations[mode] = {"summary": json.loads((output / "summary.json").read_text()),
            "identity": json.loads((output / "input-identity.json").read_text()),
            "interface": json.loads((output / "observed-interface.json").read_text()),
            "contract": json.loads(contract.read_text()), "evaluation": replay,
            "raw_records": [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()],
            "events_sha256": file_digest(output / "events.jsonl"),
            "files_sha256": {p.name: file_digest(p) for p in output.iterdir() if p.is_file()}}

    for test in ("test_reasoning_generation.py", "test_reasoning_evaluation.py"):
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", test, "-v"])
    run([sys.executable, "scripts/check_book_math.py"])
    control_contract = freeze_local_contract(items(), settings(decoding="greedy"), tokenizer)
    actual_identity = invocations["raw"]["identity"]
    controls = []
    for label, scripted, contract, deadline in (
            ("natural-eos", ScriptedForward([9, 1]), control_contract, None),
            ("natural-turn", ScriptedForward([3]), control_contract, None),
            ("context-before-output-cap", ScriptedForward([9]),
             freeze_local_contract(items(), settings(decoding="greedy", context=5), tokenizer), None),
            ("partial-forward-error", ScriptedForward([9], RuntimeError), control_contract, None),
            ("partial-interruption", ScriptedForward([9], KeyboardInterrupt), control_contract, None),
            ("expired-deadline", ScriptedForward([9]), control_contract, 0.)):
        record = generate_record(scripted, tokenizer, items()[0], contract,
            checkpoint_id="scripted-control-not-a-HF-checkpoint", identity=actual_identity, deadline=deadline)
        controls.append({"label": label, "origin": "Declared scripted stop/error control, NOT an HF model output",
                         "contract": contract, "record": record,
                         "evaluation": replay_records(items(), [record], contract)})
    sources = ["src/dongxi_llms/reasoning_generation.py", "src/dongxi_llms/reasoning_evaluation.py",
        "src/dongxi_llms/run_identity.py", "src/dongxi_llms/sampling_likelihood_lab.py",
        "scripts/generate_reasoning_records.py", "scripts/evaluate_reasoning_records.py",
        "scripts/verify_reasoning_generation.py", "tests/test_reasoning_generation.py",
        "tests/test_reasoning_evaluation.py", "book/chapters/07-evaluation-is-a-contract.md",
        "book/solutions/07-evaluation-is-a-contract.md", "book/labs/07-evaluation-is-a-contract.md",
        "experiments/specs/2026-10-04-reasoning-generation-adapter.md"]
    report = {"schema_version": 1, "package": "DXI-02", "component": ADAPTER_VERSION,
        "status": "passed bounded CPU random-checkpoint adapter", "date_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "Actual local randomly initialized HF CPU forwards; no pretrained/model acquisition",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "environment": environment_identity(), "device": "cpu", "dtype": "float32",
        "random_model": {"class": type(model).__name__, "seed": 1004,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": model.config.to_dict(), "checkpoint_sha256": artifact_hashes(checkpoint)},
        "fixture_items": items(), "fixture_origin": "Original course-authored tokenizer, suite and random weights; not upstream data/results",
        "reference_workspace": str(workspace), "commands": commands,
        "invocations": invocations, "scripted_controls": controls,
        "source_sha256": {p: file_digest(ROOT / p) for p in sources},
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_report_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_boundary": "Linux report-process RSS; excludes subprocess peaks and any continuous system/GPU maximum",
        "development_failures": ["Initial chat direct-tokenization check compared a Transformers BatchEncoding to IDs; explicit return_dict=False fixed it; one of initial13 tests failed"],
        "pending": ["Approved genuine pretrained checkpoint evaluation", "Independent behavioral review",
                    "CUDA/Spark model-scale profile and supervision", "Mac execution"],
        "limits": ["Eight raw records on two authored development items are instrument evidence, not capability estimates",
            "Scripted stop/error controls in tests are separate from unforced real random-model CLI outputs",
            "Full-prefix no-cache loop is not an optimized inference benchmark",
            "A forward-boundary deadline is not a hard external supervisor; no exact-resume implementation",
            "No installs, downloads, APIs, servers, GPU jobs, Git mutations or animation rendering"]}
    if args.notebook_manifest:
        manifest = json.loads(args.notebook_manifest.read_text())
        if manifest["failures"]:
            raise ValueError("Notebook evidence contains failures")
        if any(file_digest(ROOT / n["path"]) != n["sha256"] for n in manifest["notebooks"]):
            raise ValueError("Notebook sources changed after verification")
        report["notebook_verification"] = manifest
        report["notebook_manifest_sha256"] = file_digest(args.notebook_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "workspace": str(workspace),
        "parameters": report["random_model"]["parameters"], "records": sum(len(v["raw_records"]) for v in invocations.values()),
        "command_exits": [c["exit_code"] for c in commands], "status": report["status"]}))


if __name__ == "__main__":
    main()

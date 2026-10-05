#!/usr/bin/env python3
"""Local-only chosen-SFT/DPO scientific control; no downloads or implicit runs.

CPU controls establish the source path, not a production Spark launcher.
CUDA/pretrained commands require separate profile, recovery and authority.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.chosen_sft_control import encode_dataset, native_runners, recipe, run_arm
from dongxi_llms.run_identity import artifact_hashes, canonical_hash, environment_identity, file_digest

SOURCES = ("src/dongxi_llms/chosen_sft_control.py", "scripts/run_matched_chosen_sft.py",
           "scripts/run_chapter09_spark_sft.py", "scripts/run_chapter11_spark_dpo.py",
           "src/dongxi_llms/sft_lab.py", "src/dongxi_llms/dpo_lab.py",
           "src/dongxi_llms/run_identity.py", "src/dongxi_llms/batched_cache_lab.py")


def write_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "train", "validation", "evaluation", "groups", "output", "environment-lock"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--tokenizer-revision", required=True, help="Exact declared40-character source revision; local bytes separately observed")
    parser.add_argument("--tokenizer-id", required=True, help="Declared tokenizer source ID, separate from its local directory")
    parser.add_argument("--arm", required=True, choices=("unchanged", "chosen-sft", "dpo"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-parameters", type=int, required=True)
    parser.add_argument("--runtime-seconds", type=float, required=True)
    parser.add_argument("--reserve-gib", type=float, default=25.)
    args = parser.parse_args(argv)
    settings = recipe(seed=args.seed, updates=args.updates, accumulation=args.accumulation,
        learning_rate=args.lr, beta=args.beta, max_length=args.max_length, max_new_tokens=args.max_new_tokens)
    if (not math.isfinite(args.runtime_seconds) or not 0 < args.runtime_seconds <= 1800
            or not math.isfinite(args.reserve_gib) or args.reserve_gib < 25
            or not 1 <= args.max_parameters <= 10**9):
        parser.error("Explicit finite runtime<=1800s, reserve>=25GiB and bounded parameter ceiling required")
    if (len(args.tokenizer_revision) != 40 or any(c not in "0123456789abcdef" for c in args.tokenizer_revision)):
        parser.error("Immutable lowercase40-character tokenizer declaration required")
    if not args.checkpoint.is_dir() or not args.tokenizer.is_dir():
        parser.error("Checkpoint and tokenizer must already exist as local directories")
    args.output.mkdir(parents=True, exist_ok=False)
    started, minimum = time.monotonic(), None
    phase = "preflight"
    def guard():
        nonlocal minimum
        available = next((int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemAvailable:")), None)
        if available is None:
            raise RuntimeError("Actual Linux sampled MemAvailable is required")
        minimum = available if minimum is None else min(minimum, available)
        if available < args.reserve_gib * 1024**3:
            raise RuntimeError("Sampled25GiB host reserve breached")
        if time.monotonic()-started > args.runtime_seconds:
            raise TimeoutError("Sampled deadline breached; external hard deadline still required")
    try:
        guard()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA requested but unavailable; never silently fall back")
        inputs = {str(path.resolve()): file_digest(path) for path in (
            args.train, args.validation, args.evaluation, args.groups, args.environment_lock)}
        parent_files = artifact_hashes(args.checkpoint, guard)
        sources = {name: file_digest(ROOT/name) for name in SOURCES}
        identity = dict(schema="dongxi-matched-control-invocation-v1", created_utc=datetime.now(timezone.utc).isoformat(),
            command=list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
            parent_files=parent_files, tokenizer_files=artifact_hashes(args.tokenizer, guard),
            input_sha256=inputs, source_sha256=sources, environment=environment_identity(args.environment_lock),
            device=args.device, recipe=settings, arm=args.arm,
            readiness="prepared source command; no production recovery/ledger/whole-job containment claim")
        write_exclusive(args.output/"identity.json", identity)
        _, dpo = native_runners()
        saved = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        interface, adoption = dpo.verify_parent_tokenizer(args.checkpoint, saved, tokenizer,
            tokenizer_id=args.tokenizer_id, tokenizer_revision=args.tokenizer_revision)
        phase = "encoding"
        sidecar = json.loads(args.groups.read_text())
        groups = sidecar["groups"] if "groups" in sidecar else sidecar
        splits = []
        for path in (args.train, args.validation, args.evaluation):
            rows = dpo.load_rows(path)
            splits.append([dict(row, group=groups[row["id"]]) for row in rows])
        dataset = encode_dataset(tokenizer, *splits, max_length=args.max_length)
        if any(len(prefix)+args.max_new_tokens > args.max_length for prefix in dataset["prefixes"]):
            raise ValueError("Publication prefix plus whole output cap exceeds declared context; no truncation")
        write_exclusive(args.output/"contract.json", dict(parent_files=parent_files,
            interface=interface, legacy_adoption=adoption, recipe=settings,
            rows_sha256=dataset["rows_sha256"], encoded_sha256=dataset["encoded_sha256"],
            train_ids=dataset["train_ids"], train_groups=dataset["train_groups"], split=dataset["split"]))
        guard()
        config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        with torch.device("meta"):
            observed_parameters = sum(p.numel() for p in AutoModelForCausalLM.from_config(config).parameters())
        if observed_parameters > args.max_parameters:
            raise ValueError("Observed architecture exceeds explicitly declared parameter ceiling before weight loading")
        phase = "loading-local-parent"
        parent = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
            dtype=torch.float32, attn_implementation="sdpa").to(args.device)
        if sum(p.numel() for p in parent.parameters()) != observed_parameters:
            raise ValueError("Actual loaded parameter count differs from inspected architecture")
        if args.device == "cuda":
            parent.gradient_checkpointing_enable()
        guard()
        phase = "training-and-independent-evaluation"
        def row_sink(row):
            with (args.output/"events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
                handle.flush()
        autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.device == "cuda" else nullcontext
        policy, result = run_arm(parent, dataset, tokenizer, settings, arm=args.arm,
            guard=guard, autocast_factory=autocast, row_sink=row_sink)
        phase = "export"
        guard()
        policy.save_pretrained(args.output/"policy", safe_serialization=True)
        tokenizer.save_pretrained(args.output/"policy")
        write_exclusive(args.output/"policy/course-genealogy.json", dict(kind="full-HF-"+args.arm+"-control",
            parent_files=parent_files, parent_checkpoint=str(args.checkpoint.resolve()),
            checkpoint_interface=interface, template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            objective=result["objective"], encoded_sha256=dataset["encoded_sha256"], recipe=settings))
        guard()
        if any(file_digest(ROOT/name) != value for name, value in sources.items()):
            raise ValueError("Actual source bytes changed during the control")
        if artifact_hashes(args.checkpoint, guard) != parent_files:
            raise ValueError("Actual original parent artifacts changed")
        if artifact_hashes(args.tokenizer, guard) != identity["tokenizer_files"]:
            raise ValueError("Actual proposed tokenizer artifacts changed")
        if any(file_digest(path) != value for path, value in inputs.items()):
            raise ValueError("Actual input/lock bytes changed during the control")
        result.update(parameter_count=observed_parameters, elapsed_seconds=time.monotonic()-started,
            minimum_sampled_mem_available_bytes=minimum, checkpoint_files=artifact_hashes(args.output/"policy", guard),
            source_identity_sha256=canonical_hash(identity), production_recovery="pending; no implicit resume")
        write_exclusive(args.output/"result.json", result)
        print(json.dumps(dict(status="completed-source-control", arm=args.arm, output=str(args.output.resolve()),
                             independent_exact_match=result["independent_exact_match"])))
    except BaseException as error:
        write_exclusive(args.output/"failure.json", dict(phase=phase, type=type(error).__name__,
            message=str(error)[:512], elapsed_seconds=time.monotonic()-started,
            minimum_sampled_mem_available_bytes=minimum,
            scope="Raw prior events retained; unknown interrupted backend work not invented"))
        raise


if __name__ == "__main__":
    main()

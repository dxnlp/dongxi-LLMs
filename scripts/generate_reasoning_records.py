#!/usr/bin/env python3
"""Explicit local checkpoint generation; use evaluate_reasoning_records.py for replay."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.reasoning_generation import (freeze_local_contract,
    load_local_tokenizer, run_generation)
from dongxi_llms.run_identity import artifact_hashes, canonical_hash, file_digest, TOKENIZER_PATTERNS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--contract", type=Path)
    modes.add_argument("--freeze-contract", type=Path, metavar="NEW.json")
    parser.add_argument("--settings", type=Path, help="Required only when freezing a new contract")
    parser.add_argument("--output", type=Path, help="New directory required only for generation")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--allow-cuda", action="store_true", help="Explicit extra device gate; no automatic GPU selection")
    args = parser.parse_args(argv)
    if args.freeze_contract:
        if args.settings is None or args.output is not None or args.allow_cuda:
            parser.error("Freeze needs --settings, no --output and no CUDA allowance")
        if args.freeze_contract.exists():
            raise FileExistsError("Cannot overwrite an existing frozen contract")
        items = json.loads(args.items.read_text())
        settings = json.loads(args.settings.read_text())
        tokenizer = load_local_tokenizer(args.tokenizer or args.checkpoint)
        contract = freeze_local_contract(items, settings, tokenizer)
        contract["preparation_inputs"] = {"settings_sha256": file_digest(args.settings),
            "items_sha256": file_digest(args.items),
            "tokenizer_files": artifact_hashes(args.tokenizer or args.checkpoint, patterns=TOKENIZER_PATTERNS)}
        contract["identity"] = canonical_hash({k: v for k, v in contract.items() if k != "identity"})
        with args.freeze_contract.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"contract": str(args.freeze_contract), "identity": contract["identity"],
                          "mode": "tokenizer-only contract freeze; no model weights loaded"}))
        return contract
    if args.output is None or args.settings is not None:
        parser.error("Generation needs a frozen --contract and new --output, not mutable --settings")
    result = run_generation(root=ROOT, items_path=args.items, contract_path=args.contract,
        checkpoint=args.checkpoint, tokenizer_path=args.tokenizer, output=args.output,
        environment_lock=args.environment_lock, allow_cuda=args.allow_cuda)
    print(json.dumps(result, allow_nan=False))
    if result["status"] != "completed":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()

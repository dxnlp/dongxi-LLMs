#!/usr/bin/env python3
"""Replay a frozen response ledger offline; never loads or generates a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.reasoning_evaluation import (freeze_contract, replay_records,
                                            paired_group_bootstrap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixture = ROOT / "fixtures/reasoning-evaluation"
    parser.add_argument("--items", type=Path, default=fixture / "items.json")
    contracts = parser.add_mutually_exclusive_group()
    contracts.add_argument("--settings", type=Path, help="Legacy authored fixture settings; defaults when no contract is supplied")
    contracts.add_argument("--contract", type=Path, help="Existing frozen contract from local generation")
    parser.add_argument("--records", type=Path, default=fixture / "responses.jsonl")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1010)
    args = parser.parse_args()
    items = json.loads(args.items.read_text())
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if args.contract:
        contract = json.loads(args.contract.read_text())
    else:
        settings = json.loads((args.settings or fixture / "settings.json").read_text())
        contract = freeze_contract(items, settings)
    result = replay_records(items, records, contract)
    if args.compare:
        a, b = args.compare
        grouped = [[row for row in result["rows"] if row["checkpoint_id"] == key] for key in (a, b)]
        comparison = paired_group_bootstrap(*grouped, draws=args.draws, seed=args.seed)
        # The notebook uses the full resampling distribution; concise CLI keeps
        # its identity/settings and interval, without thousands of sample values.
        comparison.pop("samples")
        result["paired_comparison"] = dict(comparison, baseline=a, candidate=b)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

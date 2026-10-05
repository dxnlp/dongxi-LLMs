#!/usr/bin/env python3
"""Export an evidence-limited model card by replaying an existing response ledger."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dongxi_llms.evaluation_model_card import export_model_card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True, help="Existing frozen contract; never silently refrozen")
    parser.add_argument("--records", type=Path, required=True, help="Complete retained raw-response JSONL ledger")
    parser.add_argument("--run-identity", type=Path, action="append", default=[], help="Optional recorded input-identity.json; repeat for multiple invocations")
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--output", type=Path, required=True, help="New exclusive bundle directory; cannot overwrite evidence")
    args = parser.parse_args(argv)
    card = export_model_card(items_path=args.items, contract_path=args.contract,
        records_path=args.records, output=args.output, run_identity_paths=args.run_identity,
        comparisons=args.compare, draws=args.draws, seed=args.seed)
    print(json.dumps({"status": "exported offline saved-response card", "output": str(args.output.absolute()),
                      "card_sha256": card["card_sha256"],
                      "replayed_record_count": card["evaluation"]["replayed_record_count"]}, sort_keys=True))
    return card


if __name__ == "__main__":
    main()

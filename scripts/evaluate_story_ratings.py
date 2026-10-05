#!/usr/bin/env python3
"""Prepare blind story packets or replay supplied ratings offline; never score."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dongxi_llms.story_rubric import SCHEMA, prepare_packet, evaluate_ratings, read_json, write_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a public packet/private codebook; ratings remain empty")
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--checkpoints", type=Path, required=True)
    prepare.add_argument("--records", type=Path, required=True, help="Complete supplied response JSONL, including errors/caps")
    prepare.add_argument("--raters", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=None,
                         help="Private shuffle seed; authored controls default909, real packets use private randomness")
    prepare.add_argument("--compare", nargs=2, action="append", default=None,
                         help="Freeze comparison before ratings; two checkpoints default to their ordered pair")
    prepare.add_argument("--adjudication-policy", choices=("two-rater-mean", "explicit-disagreement"), default="two-rater-mean")
    prepare.add_argument("--output", type=Path, required=True, help="New exclusive bundle directory")
    evaluate = commands.add_parser("evaluate", help="Consume explicitly supplied ratings, retaining missing cells")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--ratings", type=Path, action="append", default=[])
    evaluate.add_argument("--adjudication", type=Path)
    evaluate.add_argument("--compare", nargs=2, action="append", default=[])
    evaluate.add_argument("--draws", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=1010)
    evaluate.add_argument("--output", type=Path, required=True, help="New exclusive result directory")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        paths = [args.contract, args.checkpoints, args.records, args.raters]
        packet, book = prepare_packet(read_json(args.contract), read_json(args.checkpoints),
            read_json(args.records, jsonl=True), read_json(args.raters), seed=args.seed,
            adjudication_policy=args.adjudication_policy, comparisons=args.compare)
        artifacts = {"packet.json": packet, "private-codebook.json": book}
        for i, rater in enumerate(book["raters"], 1):
            artifacts[f"ratings-{i}.json"] = {"schema_version": SCHEMA, "packet_sha256": packet["packet_sha256"],
                "rubric_sha256": book["rubric_sha256"], "rater_id": rater["rater_id"], "ratings": []}
        receipt = write_bundle(args.output, artifacts, input_paths=paths)
        print(json.dumps({"status": "prepared-empty-ratings", "packet_sha256": packet["packet_sha256"],
                          "codebook_sha256": book["codebook_sha256"], "receipt": receipt}, sort_keys=True))
    else:
        packet_path, book_path = args.bundle/"packet.json", args.bundle/"private-codebook.json"
        paths = [packet_path, book_path, *args.ratings]
        if args.adjudication:
            paths.append(args.adjudication)
        report = evaluate_ratings(read_json(packet_path), read_json(book_path),
            [read_json(p) for p in args.ratings], comparisons=args.compare,
            adjudication=read_json(args.adjudication) if args.adjudication else None,
            draws=args.draws, seed=args.seed)
        receipt = write_bundle(args.output, {"report.json": report}, input_paths=paths)
        print(json.dumps({"status": report["status"], "report_sha256": report["report_sha256"],
                          "receipt": receipt}, sort_keys=True))


if __name__ == "__main__":
    main()

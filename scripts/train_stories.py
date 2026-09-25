#!/usr/bin/env python3
"""Prepare TinyStories or run a bounded, explicit single-device training recipe."""
import argparse
from dongxi_llms.stories_data import prepare
from dongxi_llms.stories_training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--length", type=int, default=1024)
    prep.add_argument("--train-limit", type=int, default=1024, help="0 means full source")
    prep.add_argument("--valid-limit", type=int, default=128, help="0 means full source")
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name, default in (("total", 3), ("warmup", 1), ("microbatch", 1),
                          ("accumulation", 1), ("valid-windows", 2),
                          ("sample-tokens", 16), ("checkpoint-every", 2)):
        train.add_argument("--"+name, type=int, default=default)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--resume")
    train.add_argument("--max-seconds", type=float, default=600.)
    train.add_argument("--peak-lr", type=float, default=3e-4)
    train.add_argument("--floor-lr", type=float, default=3e-5)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else run)(**args)


if __name__ == "__main__":
    main()

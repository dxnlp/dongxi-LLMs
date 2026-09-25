#!/usr/bin/env python3
"""Compare trusted tensor-only recovery checkpoints and record measured differences."""
import argparse
import json
from pathlib import Path
import torch


def compare(a, b):
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError("Tensor shape/dtype mismatch")
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("State keys differ")
        return max((compare(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("State lengths differ")
        return max((compare(x, y) for x, y in zip(a, b)), default=0.)
    if a != b:
        raise AssertionError(f"State scalar differs: {a!r} vs {b!r}")
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    a = torch.load(args.first, map_location="cpu", weights_only=True)
    b = torch.load(args.second, map_location="cpu", weights_only=True)
    errors = {key: compare(value, b[key]) for key, value in a.items()}
    report = dict(first=args.first, second=args.second, maximum_absolute_errors=errors,
                  bitwise_identical=all(error == 0 for error in errors.values()))
    with Path(args.output).open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    if not report["bitwise_identical"]:
        raise SystemExit("Recovery not bitwise identical; inspect measured differences")


if __name__ == "__main__":
    main()

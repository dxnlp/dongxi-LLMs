#!/usr/bin/env python3
"""Explicit local CPU adapter merge; size the model before authorizing a run."""
import argparse
from dongxi_llms.checkpoint_merge import merge_local_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "output", "base-revision", "environment-lock"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allow-legacy-interface", action="store_true",
                        help="Explicitly adopt actual saved semantics; does not prove old ancestry")
    args = parser.parse_args()
    print(merge_local_adapter(args.base, args.adapter, args.output,
        base_revision=args.base_revision, environment_lock=args.environment_lock,
        allow_legacy=args.allow_legacy_interface))


if __name__ == "__main__":
    main()

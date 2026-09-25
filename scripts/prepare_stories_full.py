#!/usr/bin/env python3
"""Bounded full-corpus preparation using the declared validation-first policy."""
import argparse
from pathlib import Path
from dongxi_llms.stories_data import prepare_full
from dongxi_llms.stories_training import MemoryMonitor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--monitor-output', required=True)
args = parser.parse_args()
Path(args.monitor_output).mkdir(parents=True, exist_ok=False)
with MemoryMonitor(args.monitor_output, seconds=3600):
    prepare_full(args.output)

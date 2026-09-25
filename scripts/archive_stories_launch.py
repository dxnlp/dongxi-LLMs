#!/usr/bin/env python3
"""Collect small, portable launch evidence without copying data or model weights."""
import json
from pathlib import Path

root = Path('outputs')
profiles = {}
for name in ('day09-profile-b8', 'day09-profile-b16'):
    directory = root/name
    profiles[name] = {p.name: json.loads(p.read_text()) for p in directory.glob('*.json')}
    profiles[name]['metrics'] = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
data = Path('data/cache/day09-full-v2/manifest.json')
result = dict(
    launch=json.loads((root/'day09-learning-launch/status.json').read_text()),
    config=json.loads(Path('experiments/configs/tinystories-learning-01.json').read_text()),
    environment=json.loads((root/'day09-learning-environment.json').read_text()),
    recovery=json.loads((root/'day09-launch-recovery.json').read_text()),
    data=json.loads(data.read_text()) if data.exists() else None,
    profiles=profiles,
)
tests = root/'day09-learning-launch/tests.txt'
if tests.exists(): result['launch_regression_output'] = tests.read_text()
out = Path('experiments/reports/2026-09-13-tinystories-learning-launch.json')
out.write_text(json.dumps(result, indent=2)+'\n')
print(out)

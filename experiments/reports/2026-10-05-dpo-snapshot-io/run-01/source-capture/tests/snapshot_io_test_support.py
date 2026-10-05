"""Explicit authored CLI metadata, never a production default or launcher."""
import json
from pathlib import Path

from dongxi_llms.snapshot_io_budget import IO_KEYS, io_budget_contract


def explicit_snapshot_io_args(directory, *, payload_bound=16*1024**2):
    directory = Path(directory)
    limits = {key: (128 if key.endswith('_operations') else 100000000) for key in IO_KEYS}
    envelope = dict(max_payload_bytes=payload_bound, max_tree_nodes=100000,
                    max_tensor_elements=2000000, max_tensor_bytes=payload_bound, max_primitive_bytes=65536)
    contract = io_budget_contract(limits, envelope, 1024**2)
    path = directory/'authored-snapshot-io-limits.json'
    with path.open('x') as handle: handle.write(json.dumps(contract, sort_keys=True, allow_nan=False)+'\n')
    return ['--snapshot-io-limits', str(path), '--snapshot-io-ledger', str(directory/'authored-io.jsonl')]

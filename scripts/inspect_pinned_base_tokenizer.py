"""Approved fixed-pin acquisition and offline tokenizer sizing; never model work.

Each invocation creates a new receipt. The native tokenizer functions are
compiled individually from their source AST, without importing the Torch runner.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
CACHE = Path('/home/dongxi/.cache/huggingface/hub')
SNAPSHOT = CACHE / 'models--Qwen--Qwen3-0.6B-Base' / 'snapshots' / REVISION
MANIFEST = ROOT / 'experiments/reports/2026-10-05-checkpoint-inspection/upstream-manifest.json'
SOURCES = [Path(__file__), MANIFEST, ROOT/'scripts/run_chapter09_spark_sft.py',
    ROOT/'scripts/prepare_chapter09_instruction_fixture.py',
    ROOT/'experiments/data/instruction_interface_v1.jinja',
    ROOT/'experiments/data/instruction-interface-v1-data-card.json',
    ROOT/'experiments/configs/sft-0.6b-full-profile.json',
    ROOT/'src/dongxi_llms/run_identity.py',
    Path('/home/dongxi/dgx-spark-dongxi/uv.lock')]


def digest(path, algorithm='sha256', git_blob=False):
    path = Path(path)
    result = hashlib.new(algorithm)
    if git_blob:
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bindings():
    return {str(path): digest(path) for path in SOURCES}


def memory():
    with Path('/proc/meminfo').open() as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def source_function(path, name):
    source = Path(path).read_text()
    matches = [node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError('Need one undecorated native function')
    namespace = {}
    code = compile(ast.Module(body=matches, type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return namespace[name]


def tokenizer_file_bindings():
    receipt = ROOT/'experiments/reports/2026-10-05-base-tokenizer-inspection/acquisition-01.json'
    acquired = json.loads(receipt.read_text())
    if acquired['status'] != 'verified' or acquired['revision'] != REVISION:
        raise ValueError('Verified exact acquisition receipt required')
    names = ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')
    expected = {row['name']:row for row in acquired['result']['files']}
    observed = {}
    for name in names:
        path = SNAPSHOT/name
        observed[name] = dict(bytes=path.stat().st_size,sha256=digest(path))
        if observed[name] != {k:expected[name][k] for k in ('bytes','sha256')}:
            raise ValueError('Tokenizer input bytes differ from acquisition receipt')
    return dict(acquisition_receipt_sha256=digest(receipt),files=observed)


def forbid_torch_import():
    """Process-local backend exclusion; Transformers5 ignores USE_TORCH=0.

    The availability probe sees no Torch. An audit hook independently refuses
    any actual Torch import. No installed dependency or shared environment changes.
    """
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('Torch was already imported')
    original = importlib.util.find_spec
    def find_spec(name,*args,**kwargs):
        return None if name == 'torch' or name.startswith('torch.') else original(name,*args,**kwargs)
    def audit(event,args):
        if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
            raise RuntimeError('Torch import refused in tokenizer-only process')
    importlib.util.find_spec = find_spec
    sys.addaudithook(audit)


def acquire():
    from huggingface_hub import snapshot_download
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['declared_revision'] == REVISION and manifest['repository'] == MODEL
    names = [row['rfilename'] for row in manifest['files']]
    assert len(names) == len(set(names)) == 10
    if shutil.disk_usage(CACHE).free < 2*manifest['total_file_bytes']:
        raise RuntimeError('Insufficient download disk reserve')
    location = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE,
        allow_patterns=names, token=False, max_workers=2))
    if location != SNAPSHOT:
        raise ValueError('Unexpected snapshot location')
    if sorted(p.name for p in location.iterdir() if p.is_file()) != sorted(names):
        raise ValueError('Snapshot inventory differs from pinned manifest')
    files = []
    for expected in manifest['files']:
        path = location/expected['rfilename']
        size = path.stat().st_size
        sha256 = digest(path)
        upstream = expected.get('lfs')
        observed_upstream_digest = sha256 if upstream else digest(path, 'sha1', True)
        expected_upstream_digest = upstream['sha256'] if upstream else expected['blobId']
        if size != expected['size'] or observed_upstream_digest != expected_upstream_digest:
            raise ValueError(f'Pinned byte verification failed: {path.name}')
        files.append(dict(name=path.name, bytes=size, sha256=sha256,
            upstream_digest_kind='lfs-sha256' if upstream else 'git-blob-sha1',
            upstream_digest=observed_upstream_digest, upstream_verified=True))
    return dict(snapshot=str(location), file_count=len(files), files=files,
        payload_bytes=sum(row['bytes'] for row in files),
        scope='Verified local payload bytes; not measured network traffic. No tensor deserialization.')


def size_tokenizer():
    for key, value in {'CUDA_VISIBLE_DEVICES':'', 'USE_TORCH':'0', 'USE_TF':'0',
                       'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}.items():
        if os.environ.get(key) != value:
            raise ValueError(f'Required tokenizer-only environment: {key}')
    tokenizer_before = tokenizer_file_bindings()
    forbid_torch_import()
    from transformers import AutoTokenizer
    from dongxi_llms.run_identity import canonical_hash, tokenizer_interface
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = (ROOT/'experiments/data/instruction_interface_v1.jinja').read_text()
    markers = {}
    for marker in ('<|im_start|>', '<|im_end|>'):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError('Existing single-token message marker required')
        markers[marker] = ids[0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = sorted({v for v in (markers['<|im_end|>'], tokenizer.eos_token_id) if v is not None})
    interface = tokenizer_interface(tokenizer, template=tokenizer.chat_template,
        stop_ids=stop_ids, tokenizer_id=MODEL, tokenizer_revision=REVISION)
    records = source_function(ROOT/'scripts/prepare_chapter09_instruction_fixture.py', 'records')
    encode = source_function(ROOT/'scripts/run_chapter09_spark_sft.py', 'encode_record')
    generation = source_function(ROOT/'scripts/run_chapter09_spark_sft.py', 'generation_upper')
    raw = {split:records(split, start, count) for split,start,count in
           [('train',0,80), ('dev',80,20), ('test',100,40)]}
    card = json.loads((ROOT/'experiments/data/instruction-interface-v1-data-card.json').read_text())
    raw_hashes = {}
    for split, rows in raw.items():
        content = ''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows).encode()
        raw_hashes[split] = hashlib.sha256(content).hexdigest()
        if raw_hashes[split] != card['splits'][split]['sha256']:
            raise ValueError('Original raw fixture identity changed')
    encoded = {split:[encode(tokenizer, row, 256) for row in raw[split]] for split in ('train','dev')}
    measurements = {}
    for split, rows in encoded.items():
        measurements[split] = [dict(id=row['id'], tokens=len(row['ids']),
            shifted_targets=sum(v != -100 for v in row['labels'][1:]),
            masked_labels=sum(v == -100 for v in row['labels']), encoded=row) for row in rows]
    order = list(range(240)); random.Random(1212).shuffle(order)
    selected = [encoded['train'][i] for i in order[:80]]
    train_targets = sum(sum(v != -100 for v in row['labels'][1:]) for row in selected)
    dev_targets = sum(row['shifted_targets'] for row in measurements['dev'])
    train_positions = sum(len(row['ids']) for row in selected)
    dev_positions = sum(row['tokens'] for row in measurements['dev'])
    prefixes = [tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in raw['dev'][:8]]
    if any(not ids or len(ids)+64 > 256 for ids in prefixes):
        raise ValueError('Native fixed generation context does not fit')
    upper = generation(prefixes, 64)
    known = dict(train_updates=20, sampled_examples=80, selector_steps=80,
        valid_targets=train_targets+2*dev_targets,
        logical_sequence_tokens=train_positions+2*dev_positions+2*upper['logical_sequence_tokens'],
        policy_forward_calls=80+120+2*upper['policy_forward_calls'],
        policy_forward_positions=train_positions+2*dev_positions+2*upper['policy_forward_positions'],
        evaluation_calls=120+2*upper['evaluation_calls'],
        evaluation_positions=2*dev_positions+2*upper['evaluation_positions'],
        generation_calls=16, generation_position_upper_bound=2*upper['generation_position_upper_bound'],
        generation_tokens=1024, recovery_validation_operations=2, recovery_history_rows=20)
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('Torch imported during tokenizer-only sizing')
    tokenizer_after = tokenizer_file_bindings()
    if tokenizer_after != tokenizer_before:
        raise ValueError('Tokenizer input bytes changed during measurement')
    return dict(snapshot=str(SNAPSHOT), tokenizer_interface=interface, message_marker_ids=markers,
        tokenizer_files_before=tokenizer_before,tokenizer_files_after=tokenizer_after,
        backend_exclusion='process-local availability-probe exclusion plus fail-closed audit import hook; installed packages unchanged',
        raw_split_sha256=raw_hashes, encoded_split_sha256={s:canonical_hash(v) for s,v in encoded.items()},
        measurements=measurements, selected_train_indices=order[:80],
        selected_train_ids=[row['id'] for row in selected], train_targets=train_targets,
        dev_targets_per_panel=dev_targets, train_positions=train_positions, dev_positions_per_panel=dev_positions,
        generation_prompts=[dict(id=row['id'], ids=ids, prompt_tokens=len(ids),
            prompt_plus_cap=len(ids)+64) for row,ids in zip(raw['dev'][:8], prefixes)],
        one_generation_panel_upper=upper, schedule_requirements=known,
        unresolved=['recovery_tensor_elements', 'recovery_rng_states', 'snapshot payload/tree/tensor/primitive envelopes',
                    'I/O9 allowances', 'work journal/output allowances', 'actual GPU memory fit', 'native launch approval'],
        publication_test_tokenized=False, torch_imported=False, model_loaded=False,
        scope='Tokenizer measurements and unchanged schedule requirements; conservative generation reservations, not executed model work or admitted caps.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('acquire','size'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Reserve an exclusive receipt before any work; failures remain evidence.
    handle = args.receipt.open('x')
    started = time.monotonic()
    before = bindings()
    report = dict(action=args.action, repository=MODEL, revision=REVISION,
        started_at_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        platform=platform.platform(), packages={name:importlib.metadata.version(name) for name in
            ('transformers','tokenizers','huggingface-hub')}, source_before=before,
        mem_available_before_bytes=memory(), disk_free_before_bytes=shutil.disk_usage(CACHE).free)
    exit_code = 1
    try:
        report['result'] = acquire() if args.action == 'acquire' else size_tokenizer()
        report['source_after'] = bindings()
        if report['source_after'] != before:
            raise ValueError('Source bindings changed during collection')
        report['status'] = 'verified'; exit_code = 0
    except BaseException as error:
        # Avoid retaining signed HTTP URLs or credential-bearing exception text.
        report['status'] = 'failed'; report['error_type'] = type(error).__name__
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            mem_available_after_bytes=memory(), disk_free_after_bytes=shutil.disk_usage(CACHE).free)
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n'); handle.flush(); os.fsync(handle.fileno()); handle.close()
    print(json.dumps(dict(status=report['status'], action=args.action, receipt=str(args.receipt),
        elapsed_seconds=report['elapsed_seconds'])))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

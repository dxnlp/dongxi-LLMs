"""Independent read-only byte/journal review of actual fixed Spark acceptances."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path('/home/dongxi/dongxi_ai/Dongxi_LLMs')
REPORT = ROOT/'experiments/reports'
OUTPUT = REPORT/'2026-10-05-native-acceptance-independent.json'
ARCHIVE = REPORT/'2026-10-05-native-acceptance-independent'
spec = importlib.util.spec_from_file_location('own_readonly_journal_review',
    REPORT/'2026-10-05-native-base-profile/independent-review-01.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
canonical, digest, read, rows, journal = review.canonical, review.digest, review.read, review.rows, review.journal


def byte_entries(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            pieces = entry.filename.split('/')
            if len(pieces) == 3 and pieces[1] == 'data' and pieces[2].isdigit():
                assert pieces[2] not in result
                sha = hashlib.sha256()
                with archive.open(entry) as handle:
                    while block := handle.read(1024*1024):
                        sha.update(block)
                result[pieces[2]] = dict(bytes=entry.file_size, sha256=sha.hexdigest())
    assert result
    return result


def supervised(path, expected_success=True):
    value = read(path)
    assert value['final_record_retained'] is True
    assert not value['cleanup_errors'] and value['journal_error'] is None
    assert all(row['cleanup_complete'] and row['still_alive'] is False for row in value['helper_cleanup'])
    assert all(row['still_alive'] is False for row in value['queue_feeder_shutdown'])
    assert value['child_seconds'] < 900
    minimum = min(row['available_bytes'] for row in value['observations'])
    assert minimum == value['minimum_sampled_available_bytes'] and minimum > 25*1024**3
    assert value['limits']['external_seconds'] == 900 and value['limits']['reserve_bytes'] == 25*1024**3
    if expected_success:
        assert value['status'] == 'completed' and value['actual_exit_code'] == 0
        assert value['stop_reason'] is None
        assert all(not row['conflict_scan']['conflicts'] and row['conflict_scan']['unreadable'] == 0
                   for row in value['observations'])
    else:
        assert value['status'] == 'failed' and value['actual_exit_code'] != 0
    return dict(binding=digest(path), status=value['status'], exit_code=value['actual_exit_code'],
        child_seconds=value['child_seconds'], samples=len(value['observations']),
        minimum_sampled_available_bytes=minimum, stop_reason=value['stop_reason'],
        returned_final_ack=True, helpers_and_feeders_clean=True)


def source_bindings(values):
    verified = {}
    for key, value in values.items():
        current = digest(Path(value['actual_path']))
        assert current['sha256'] == value['sha256'] and current['bytes'] == value['bytes'], key
        verified[key] = current
    return verified


def native_pair(mode):
    folder = REPORT/f'2026-10-05-native-sft-{mode}-replay'
    accepted = read(folder/('acceptance-retry02.json' if mode == 'full' else 'acceptance.json'))
    prepared = read(folder/'preparation.json')
    assert accepted['status'] == 'passed' and all(accepted['checks'].values())
    orig = accepted['original_result']
    resumed = accepted['replayed_result' if mode == 'full' else 'resumed_result']
    orig_root = Path(orig['latest_completed_snapshot']['path']).parent
    resume_root = Path(resumed['latest_completed_snapshot']['path']).parent
    originals = rows(orig_root/'metrics.jsonl')
    replays = rows(resume_root/'metrics.jsonl')
    numerical_fields = ('update', 'answer_nll', 'targets', 'gradient_norm', 'cursor',
        'cumulative_supervised_targets', 'processed_positions', 'cumulative_processed_positions')
    project = lambda values: [{k: row[k] for k in numerical_fields} for row in values]
    assert [row['update'] for row in originals] == list(range(1, 21))
    assert [row['update'] for row in replays] == list(range(11, 21))
    assert project(originals[10:]) == project(replays)
    assert all(math.isfinite(row['answer_nll']) and math.isfinite(row['gradient_norm'])
               for row in originals+replays)
    assert sum(row['targets'] for row in originals) == 455
    assert sum(row['targets'] for row in replays) == 229
    assert orig['cumulative_supervised_targets'] == resumed['cumulative_supervised_targets'] == 455
    old_entries = byte_entries(orig_root/'checkpoint-000020.pt')
    new_entries = byte_entries(resume_root/'checkpoint-000020.pt')
    assert old_entries == new_entries == accepted['tensor_entries_original']
    assert new_entries == accepted['tensor_entries_replayed' if mode == 'full' else 'tensor_entries_resumed']
    sample_fields = ('id', 'prompt_ids', 'generated_ids', 'text', 'reference', 'exact_match',
                     'stopped_on_end', 'stop_reason', 'truncated', 'error')
    sample_project = lambda values: [{k: row[k] for k in sample_fields} for row in values]
    assert sample_project(orig['samples']) == sample_project(resumed['samples'])
    panels = []
    for label, directory, result in (('original', orig_root, orig), ('resumed', resume_root, resumed)):
        config = read(directory/'config.json')
        assert config['mode'] == mode and config['updates'] == 20 and config['rank'] == 8
        assert config['revision'] == config['tokenizer_revision'] == 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
        assert config['max_length'] == 256 and config['microbatch'] == 1 and config['accumulation'] == 4
        assert config['seed'] == 1212 and config['learning_rate'] == 2e-5
        assert config['allow_download'] is False
        for stage, nll_key in (('baseline', 'initial_dev_nll'), ('final', 'final_dev_nll')):
            nll = rows(directory/f'{stage}-nll-raw.jsonl')
            generated = rows(directory/f'{stage}-generation-raw.jsonl')
            assert len(nll) == 60 and sum(row['valid_targets'] for row in nll) == 360
            measured = math.fsum(row['loss_sum'] for row in nll)/360
            assert math.isclose(measured, result[nll_key], rel_tol=1e-14)
            assert len(generated) == 8 and all(row['error'] is None for row in generated)
            if stage == 'final': assert sample_project(generated) == sample_project(result['samples'])
            panels.append(dict(invocation=label, stage=stage, nll=measured, targets=360,
                reported_exact=sum(row['exact_match'] for row in generated),
                message_end_stops=sum(row['stopped_on_end'] for row in generated),
                truncated=sum(row['truncated'] for row in generated),
                generated_tokens=sum(row['generated_tokens'] for row in generated)))
    work, prefixes, tickets, work_binding = journal(Path(resumed['work_journal']))
    io, io_prefixes, io_tickets, io_binding = journal(Path(resumed['snapshot_io_journal']))
    assert work == resumed['cumulative_work_ledger'] and io == resumed['cumulative_snapshot_io_ledger']
    assert prefixes[orig['cumulative_work_ledger']['sequence']] == orig['cumulative_work_ledger']
    assert io_prefixes[orig['cumulative_snapshot_io_ledger']['sequence']] == orig['cumulative_snapshot_io_ledger']
    assert work['completed']['train_updates'] == 30 and work['completed']['sampled_examples'] == 120
    assert work['completed']['valid_targets'] == 684+1440 == 2124
    assert work['completed']['recovery_validation_operations'] == 6
    assert work['completed']['recovery_history_rows'] == 70
    assert work['completed']['recovery_rng_states'] == 18
    assert not work['open_tickets'] and not work['failed_tickets'] and not io['open_tickets'] and not io['failed_tickets']
    assert io['completed']['snapshot_save_operations'] == 5
    assert io['completed']['snapshot_inspect_operations'] == io['completed']['snapshot_load_operations'] == 1
    snapshots = []
    for directory, updates in ((orig_root, (0, 10, 20)), (resume_root, (10, 20))):
        contract = read(directory/'recovery-contract.json')
        assert canonical(contract) == work['contract_sha256']
        for update in updates:
            path = directory/f'checkpoint-{update:06d}.pt'
            header = read(Path(str(path)+'.commit.json'))
            receipt = read(Path(str(path)+'.work.json'))
            observed = digest(path)
            assert observed['bytes'] == header['payload_bytes'] and observed['sha256'] == header['payload_sha256']
            assert header == receipt['snapshot'] and header['contract_sha256'] == canonical(contract)
            assert receipt['runner_work_prefix'] == prefixes[receipt['runner_work_prefix']['sequence']]
            assert receipt['io_prefix'] == io_prefixes[receipt['io_prefix']['sequence']]
            snapshots.append(observed)
    verified_sources = source_bindings(prepared['source_bindings'])
    verified_inputs = source_bindings(prepared['input_bindings'])
    for number in (1, 2):
        launch = read(folder/f'launch-{number}.json')
        assert launch['source_bindings'] == prepared['source_bindings']
        assert launch['input_bindings'] == prepared['input_bindings']
    external = [supervised(folder/'returned-supervision-1.json')]
    if mode == 'full':
        retry = read(folder/'launch-retry02.json')
        assert retry['original_attempt'] == read(folder/'launch-2.json')
        source_bindings(dict(retry_collector=retry['collector'], retry_spec=retry['retry_spec']))
        failed = supervised(folder/'returned-supervision-2.json', False)
        assert failed['stop_reason'] == 'conflicting-process' and failed['exit_code'] == -15
        external.extend((failed, supervised(folder/'returned-supervision-retry02.json')))
        assert accepted['earlier_failed_attempt'] == read(folder/'failure.json')
    else:
        external.append(supervised(folder/'returned-supervision-2.json'))
    archived_journals = []
    for label, source in (('work', Path(resumed['work_journal'])), ('io', Path(resumed['snapshot_io_journal']))):
        target = ARCHIVE/f'{mode}-{label}.jsonl'
        with target.open('xb') as handle: handle.write(source.read_bytes())
        assert digest(target)['sha256'] == digest(source)['sha256']
        archived_journals.append(digest(target))
    return dict(status='passed', accepted_binding=digest(folder/('acceptance-retry02.json' if mode == 'full' else 'acceptance.json')),
        exact_ten_update_tail=True, exact_tensor_entries=True, tensor_entry_count=len(old_entries),
        tensor_entry_map_sha256=canonical(old_entries), exact_final_generated_records=True,
        original_training_targets=455, replay_training_targets=229, cumulative_final_model_targets=455,
        cumulative_likelihood_work_targets=2124, panels=panels, snapshots=snapshots,
        work=work, snapshot_io=io, journals=archived_journals, external_invocations=external,
        source_bindings=verified_sources, input_bindings=verified_inputs)


def merge_checks():
    values = {}
    for precision, prefix in (('BF16', ''), ('FP32', 'fp32-')):
        folder = REPORT/f'2026-10-05-native-lora-{prefix}merge-reload'
        launch = read(folder/'launch.json')
        source_bindings(launch['bindings'])
        identity = read(folder/'input-identity.json')
        assert identity['identity_sha256'] == canonical({k: v for k, v in identity.items() if k != 'identity_sha256'})
        for name, expected in identity['source_sha256'].items():
            assert digest(ROOT/name)['sha256'] == expected
        for name, expected in identity['input_sha256'].items():
            assert digest(ROOT/name)['sha256'] == expected
        for name, expected in identity['checkpoint_files'].items():
            assert digest(Path(identity['checkpoint_path'])/name)['sha256'] == expected
        adapter = ROOT/'outputs/native-sft-lora-replay-20261005-original/policy'
        for name, expected in identity['config']['adapter_files'].items():
            assert digest(adapter/name)['sha256'] == expected
        external = supervised(folder/'returned-supervision.json', precision == 'FP32')
        if precision == 'BF16':
            assert identity['config']['atol'] == .125 and identity['config']['rtol'] == .015625
            failure = read(folder/'failure.json')
            assert failure['type'] == 'AssertionError' and '0.67578125' in failure['message']
            values[precision] = dict(status='retained-failure', failure_binding=digest(folder/'failure.json'),
                atol=.125, rtol=.015625, first_prefix_absolute_discrepancy=.67578125, external=external)
        else:
            result = read(folder/'verification.json')
            assert result['status'] == 'passed' and result['atol'] == .002 and result['rtol'] == .001
            assert identity['config']['atol'] == .002 and identity['config']['rtol'] == .001
            assert result['exact_reloaded_logits'] is True and result['interface_compatible'] is True
            assert len(result['comparisons']) == 8
            assert all(row['last_position_argmax_equal'] for row in result['comparisons'])
            maximum = max(row['max_absolute'] for row in result['comparisons'])
            assert maximum == 5.364418029785156e-05
            assert result['forward_calls'] == 24 and result['prompt_tokens'] == 269 and result['full_prefix_positions'] == 807
            merged = ROOT/'outputs/native-lora20-fp32-merged-20261005'
            inventory = []
            for name, expected in result['merged_files'].items():
                observed = digest(merged/name)
                assert observed['sha256'] == expected
                inventory.append(observed)
            genealogy = read(merged/'course-genealogy.json')
            interface = read(folder/'merged-interface.json')
            assert interface['interface_sha256'] == genealogy['checkpoint_interface']['interface_sha256']
            assert interface['tokenizer'] == genealogy['checkpoint_interface']['tokenizer']
            assert interface['template_sha256'] == genealogy['checkpoint_interface']['template_sha256']
            assert interface['generation_stop_ids'] == genealogy['checkpoint_interface']['generation_stop_ids']
            assert sum(row['bytes'] for row in inventory) < 4*1024**3
            values[precision] = dict(status='passed-recorded-measurements', atol=.002, rtol=.001,
                reported_exact_reload_logits=True, reported_argmax_agreement=8,
                reported_max_absolute_discrepancy=maximum, diagnostic_forwards=24, full_prefix_positions=807,
                interface_verified=True, actual_merged_inventory=inventory, external=external,
                scope='Independent byte/receipt verification, not a new numerical-model rerun.')
        values[precision]['identity_binding'] = digest(folder/'input-identity.json')
    return values


def main():
    started = time.monotonic()
    ARCHIVE.mkdir(mode=0o700, exist_ok=False)
    pairs = {mode: native_pair(mode) for mode in ('full', 'lora')}
    merged = merge_checks()
    result = dict(schema='dongxi-independent-native-acceptance-review-v1', status='passed',
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        full_and_lora=pairs, merge=merged, torch_imported='torch' in sys.modules, model_loaded=False,
        collector=digest(Path(__file__)), journal_reviewer=digest(Path(review.__file__)),
        scope='Independent streamed checkpoint/ZIP-entry hashing, raw JSONL replay, source/input identity and journal/returned watchdog review. No tensor/pickle/model deserialization, inference, GPU work or shared-source edits.',
        limitations=[
            'Identical serialized tensor entry bytes do not establish arbitrary pickle metadata/Python RNG equivalence or cross-machine replay.',
            'BF16 merge failure remains failed. FP32 is a separately predeclared numerical representation, not BF16 policy equivalence.',
            'Merge numeric maxima/argmax/exact reload are checks of recorded original measurements; this review did not rerun model forwards.',
            'Returned watchdog ACK/helper/feeder fields are observed separately from stored records unknown-at-write. Memory remains sampled.',
            'No400-update pilot, selected downstream parent, DPO/RLVR campaign, publication behavioral claim, Mac/CI or learner advancement follows.'],
        original_dxi01_review='Original byte/interface/full checkpoint/explicit FP32 merge-reload identity criteria now have realistic bounded evidence. Retain precision limitations; campaign chain and later broad containment work must not silently redefine the original identity criteria.')
    assert result['torch_imported'] is False
    with OUTPUT.open('x') as handle: handle.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'],
        pairs={mode: dict(tensor_entries=value['tensor_entry_count'], targets=value['cumulative_likelihood_work_targets'],
            invocations=len(value['external_invocations']), final_panels=[p for p in value['panels'] if p['stage']=='final'])
            for mode, value in pairs.items()}, merge_status={k:v['status'] for k,v in merged.items()}), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

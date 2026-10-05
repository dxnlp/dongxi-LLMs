"""Read-only native-profile evidence review; never deserialize a model/tensor."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/home/dongxi/dongxi_ai/Dongxi_LLMs')
RUN = ROOT/'outputs/native-base-profile-20261005-run01'
EVIDENCE = ROOT/'experiments/reports/2026-10-05-native-base-profile'
JOURNALS = ROOT/'outputs/native-base-profile-20261005-journals'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024*1024):
            value.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino), str(path)
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return dict(path=label, bytes=after.st_size, sha256=value.hexdigest())


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def journal(path):
    """Independent replay of the saved record chain and all cumulative costs."""
    raw = path.read_bytes()
    assert raw.endswith(b'\n')
    events = [json.loads(line) for line in raw.splitlines()]
    header = events.pop(0)
    info = path.stat()
    assert header['file_identity'] == [info.st_dev, info.st_ino]
    assert len(raw) <= header['max_bytes']
    limits = header['limits']
    sequence = 0
    chain = canonical(header)
    tickets = {}
    prefixes = {}

    def summary():
        vectors = {name: dict.fromkeys(limits, 0) for name in (
            'reserved', 'completed', 'known_partial', 'attempted_upper', 'uncertain_upper')}
        opened, failed = [], []
        for number, ticket in tickets.items():
            for key, value in ticket['costs'].items():
                vectors['reserved'][key] += value
            if ticket['status'] == 'complete':
                for key, value in ticket['actual'].items():
                    vectors['completed'][key] += value
                    vectors['attempted_upper'][key] += value
            else:
                (opened if ticket['status'] == 'open' else failed).append(number)
                for key, value in ticket['actual'].items():
                    vectors['known_partial'][key] += value
                for key, value in ticket.get('attempted', ticket['costs']).items():
                    vectors['attempted_upper'][key] += value
                    vectors['uncertain_upper'][key] += value-ticket['actual'][key]
        return dict(schema=header['schema'], ledger_id=header['ledger_id'],
                    file_identity=header['file_identity'], contract_sha256=header['contract_sha256'],
                    limits=limits, max_bytes=header['max_bytes'], sequence=sequence,
                    chain_sha256=chain, open_tickets=opened, failed_tickets=failed, **vectors)

    prefixes[0] = summary()
    for envelope in events:
        assert set(envelope) == {'record', 'sha256'}
        record = envelope['record']
        assert envelope['sha256'] == canonical(record)
        assert record['sequence'] == sequence+1 and record['previous'] == chain
        if record['event'] == 'reserve':
            costs = record['costs']
            assert set(costs) == set(limits)
            assert all(type(v) is int and v >= 0 for v in costs.values())
            assert all(summary()['reserved'][k]+v <= limits[k] for k, v in costs.items())
            assert record['ticket'] == record['sequence']
            tickets[record['ticket']] = dict(costs=costs, actual=dict.fromkeys(limits, 0),
                                            status='open', operation=record['operation'])
        else:
            assert record['event'] in ('complete', 'fail')
            ticket = tickets[record['ticket']]
            assert ticket['status'] == 'open'
            assert set(record['actual']) == set(limits)
            assert all(type(v) is int and 0 <= v <= ticket['costs'][k]
                       for k, v in record['actual'].items())
            if record['event'] == 'fail':
                assert all(record['actual'][k] <= v <= ticket['costs'][k]
                           for k, v in record['attempted'].items())
                ticket['attempted'] = record['attempted']
            ticket.update(actual=record['actual'], status=record['event'])
        sequence, chain = record['sequence'], envelope['sha256']
        prefixes[sequence] = summary()
    return summary(), prefixes, tickets, digest(path)


def main():
    started = time.monotonic()
    prepared = read(EVIDENCE/'preparation-01.json')
    config = read(RUN/'config.json')
    result = read(RUN/'result.json')
    supervision = read(EVIDENCE/'supervision-01/result.json')
    contract = read(RUN/'recovery-contract.json')
    sizing = read(ROOT/'experiments/reports/2026-10-05-base-tokenizer-inspection/sizing-02.json')['result']
    expected = dict(mode='full', updates=20, microbatch=1, accumulation=4, max_length=256,
                    learning_rate=2e-5, runtime_seconds=900, reserve_gib=25, seed=1212, checkpoint_every=20)
    assert prepared['request']['recipe'] == expected
    assert all(config[k] == v for k, v in expected.items())
    assert config['revision'] == config['tokenizer_revision'] == 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
    assert config['model'] == 'Qwen/Qwen3-0.6B-Base'
    assert config['allow_download'] is False and config['resume_bytes'] is None
    assert config['checkpoint_interface'] == sizing['tokenizer_interface']
    assert supervision['child_command'] == prepared['argv']
    bindings = {}
    for section in ('input_bindings', 'source_bindings'):
        for label, value in prepared[section].items():
            current = digest(Path(value['actual_path']))
            assert current['sha256'] == value['sha256'] and current['bytes'] == value['bytes'], label
            bindings[label] = current
    metrics = rows(RUN/'metrics.jsonl')
    assert [r['update'] for r in metrics] == list(range(1, 21))
    assert all(math.isfinite(r['answer_nll']) and math.isfinite(r['gradient_norm']) and r['gradient_norm'] > 0
               for r in metrics)
    selected = [sizing['measurements']['train'][i] for i in sizing['selected_train_indices']]
    for i, record in enumerate(metrics):
        group = selected[i*4:(i+1)*4]
        assert record['targets'] == sum(r['shifted_targets'] for r in group)
        assert record['processed_positions'] == sum(r['tokens'] for r in group)
        assert record['cursor'] == 4*(i+1)
        assert record['cumulative_supervised_targets'] == sum(r['targets'] for r in metrics[:i+1])
    assert sum(r['targets'] for r in metrics) == result['cumulative_supervised_targets'] == 455
    assert sum(r['processed_positions'] for r in metrics) == result['cumulative_processed_positions'] == 3154
    panels = {}
    prompts = sizing['generation_prompts']
    for stage, key in (('baseline', 'initial_dev_nll'), ('final', 'final_dev_nll')):
        nll = rows(RUN/f'{stage}-nll-raw.jsonl')
        generations = rows(RUN/f'{stage}-generation-raw.jsonl')
        assert [r['id'] for r in nll] == [r['id'] for r in sizing['measurements']['dev']]
        assert len(nll) == 60 and sum(r['valid_targets'] for r in nll) == 360
        assert all(math.isfinite(r['loss_sum']) and r.get('error') is None for r in nll)
        recomputed = math.fsum(r['loss_sum'] for r in nll)/360
        assert math.isclose(recomputed, result[key], rel_tol=1e-14)
        assert [r['id'] for r in generations] == [r['id'] for r in prompts]
        for row, prompt in zip(generations, prompts):
            assert row['prompt_ids'] == prompt['ids'] and row['error'] is None
            assert len(row['generated_ids']) == row['generated_tokens'] == 64
            assert row['generated_ids'][-1] not in config['generation_stop_ids']
            assert row['stop_reason'] == 'max_new_tokens' and row['truncated'] is True
        panels[stage] = dict(nll_rows=len(nll), valid_targets=360, recomputed_nll=recomputed,
                             generation_rows=8, generated_tokens=512,
                             reported_exact_matches=sum(r['exact_match'] for r in generations),
                             observed_end_marker_stops=sum(r['stopped_on_end'] for r in generations),
                             truncated=sum(r['truncated'] for r in generations))
    work, work_prefixes, work_tickets, work_binding = journal(JOURNALS/'work-01.jsonl')
    io, io_prefixes, io_tickets, io_binding = journal(JOURNALS/'io-01.jsonl')
    assert work == result['cumulative_work_ledger'] and io == result['cumulative_snapshot_io_ledger']
    assert work['limits'] == prepared['sft_limits']
    assert io['limits'] == prepared['snapshot_io_contract']['limits']
    assert not work['failed_tickets'] and not work['open_tickets'] and not io['failed_tickets'] and not io['open_tickets']
    assert work['completed']['valid_targets'] == 1175
    assert work['contract_sha256'] == canonical(contract)
    assert io['contract_sha256'] == canonical(dict(io_contract=prepared['snapshot_io_contract'], science_sha256=canonical(contract)))
    snapshots = []
    for update in (0, 20):
        path = RUN/f'checkpoint-{update:06d}.pt'
        header = read(Path(str(path)+'.commit.json'))
        receipt = read(Path(str(path)+'.work.json'))
        observed = digest(path)
        assert observed['sha256'] == header['payload_sha256'] and observed['bytes'] == header['payload_bytes']
        assert header['completed_updates'] == update and header['phase'] == 'completed'
        assert header['contract_sha256'] == canonical(contract) and receipt['snapshot'] == header
        assert observed['bytes'] < prepared['request']['snapshot_max_bytes']
        assert receipt['io_contract_sha256'] == canonical(prepared['snapshot_io_contract'])
        assert receipt['io_prefix'] == io_prefixes[receipt['io_prefix']['sequence']]
        assert receipt['runner_work_prefix'] == work_prefixes[receipt['runner_work_prefix']['sequence']]
        snapshots.append(dict(update=update, **observed, work_prefix_sequence=receipt['runner_work_prefix']['sequence'],
                              io_prefix_sequence=receipt['io_prefix']['sequence']))
    assert sum(r['bytes'] for r in snapshots) == io['completed']['snapshot_hash_bytes']
    assert sum(r['bytes'] for r in snapshots) == io['completed']['snapshot_serialization_bytes']
    watchdog_events = rows(EVIDENCE/'supervision-01/events.jsonl')
    assert watchdog_events == supervision['in_memory_events']
    observations = supervision['observations']
    assert observations == [r['observation'] for r in watchdog_events if r['event'] == 'resource-sample']
    minimum = min(r['available_bytes'] for r in observations)
    assert minimum == supervision['minimum_sampled_available_bytes'] and minimum > 25*1024**3
    assert all(not r['conflict_scan']['conflicts'] and r['conflict_scan']['unreadable'] == 0 for r in observations)
    assert supervision['actual_exit_code'] == 0 and supervision['status'] == 'completed'
    assert supervision['child_seconds'] < 900 and supervision['stop_reason'] is None
    assert supervision['failure'] is None and supervision['journal_error'] is None and not supervision['cleanup_errors']
    assert all(r['cleanup_complete'] and r['still_alive'] is False for r in supervision['helper_cleanup'])
    assert [r['signal'] for r in watchdog_events if r['event'] == 'owned-signal'] == [15, 9]
    exported = digest(RUN/'policy/model.safetensors')
    assert exported['sha256'] != config['base_checkpoint_files']['model.safetensors']
    genealogy = read(RUN/'policy/course-genealogy.json')
    assert genealogy['checkpoint_interface'] == config['checkpoint_interface']
    assert genealogy['base_revision'] == config['revision'] and genealogy['data_sha256'] == config['train_sha256']
    cached = {r['path']: {k: r[k] for k in ('path', 'bytes', 'sha256')} for r in snapshots}
    cached[exported['path']] = exported
    inventory = []
    for path in sorted(RUN.rglob('*')):
        if path.is_file():
            label = str(path.relative_to(ROOT))
            inventory.append(cached[label] if label in cached else digest(path))
    retained_raw = []
    for name in ('baseline-nll-raw.jsonl', 'final-nll-raw.jsonl',
                 'baseline-generation-raw.jsonl', 'final-generation-raw.jsonl', 'metrics.jsonl'):
        destination = EVIDENCE/('retained-'+name)
        with destination.open('xb') as handle:
            handle.write((RUN/name).read_bytes())
        observed = digest(destination)
        assert observed['sha256'] == digest(RUN/name)['sha256']
        retained_raw.append(observed)
    compact_result = {k: v for k, v in result.items() if k not in ('samples', 'baseline_samples')}
    compact_result['samples_retained_in'] = 'retained-final-generation-raw.jsonl'
    compact_result['baseline_samples_retained_in'] = 'retained-baseline-generation-raw.jsonl'
    with (EVIDENCE/'retained-native-result.json').open('x') as handle:
        handle.write(json.dumps(compact_result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    observations_out = dict(
        schema='dongxi-independent-native-profile-review-v1', status='passed',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        scope='Read-only JSON/journal replay and streamed hashes; no model/tensor deserialization, inference or GPU work.',
        recipe=expected, training=dict(updates=20, targets=455, positions=3154,
            finite_loss_and_gradient_rows=20, gradient_norm_min=min(r['gradient_norm'] for r in metrics),
            gradient_norm_max=max(r['gradient_norm'] for r in metrics)),
        panels=panels, snapshots=snapshots, source_and_input_bindings=bindings,
        work_journal=dict(binding=work_binding, final=work), io_journal=dict(binding=io_binding, final=io),
        watchdog=dict(exit_code=0, child_seconds=supervision['child_seconds'], samples=len(observations),
            minimum_available_bytes=minimum, minimum_available_gib=minimum/1024**3,
            helpers_clean=True, owned_signals=[15, 9], observations_match_retained_events=True,
            final_record_acknowledgment=supervision['final_record_acknowledgment']),
        exported_model=exported, output_total_logical_bytes=sum(r['bytes'] for r in inventory), output_inventory=inventory,
        retained_raw=retained_raw, retained_native_result=digest(EVIDENCE/'retained-native-result.json'),
        evidence_boundaries=[
            'Development NLL decreased; both observed panels remain0/8 exact and0/8 end-marker stops. No assistant/general capability claim.',
            'Snapshot bytes/markers/receipt prefixes verified; recovery application and internal payload semantic replay not executed by this review.',
            'External memory is sampled, not a continuous minimum or physical cgroup/quota guarantee.',
            'Saved watchdog record cannot authenticate its own future writer ACK; returned controller receipt is a separate root observation.',
            'This profile does not establish downstream smoke/recovery/pilot, publication test, Mac/CI or final defense completion.'],
        torch_imported='torch' in sys.modules, model_loaded=False)
    assert observations_out['torch_imported'] is False
    with (EVIDENCE/'independent-review-01.json').open('x') as handle:
        handle.write(json.dumps(observations_out, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: observations_out[k] for k in ('status', 'elapsed_seconds', 'training', 'panels', 'watchdog', 'snapshots', 'output_total_logical_bytes')}, indent=2))


if __name__ == '__main__':
    main()

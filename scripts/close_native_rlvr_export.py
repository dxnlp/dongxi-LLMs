#!/usr/bin/env python3
"""Explicit model-free consistency closure; never reclassifies supervision."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'dongxi-native-rlvr-observed-export-v1'
STATUS = 'export-consistent-supervision-not-reclassified'
CHECKS = ('pilot_completed_horizon', 'original_prompt_and_source_contract',
          'no_open_or_failed_journal_tickets')
SOURCES = ('scripts/close_native_rlvr_export.py',
           'tests/test_native_rlvr_observed_export.py')


def load_stages():
    spec = importlib.util.spec_from_file_location('observed_export_stages',
                                                ROOT/'scripts/run_native_rlvr_stages.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def supervision_observation(value):
    fields = ('status', 'actual_exit_code', 'stop_reason', 'failure', 'cleanup_errors',
              'journal_error', 'final_record_retained')
    if (not set(fields) <= set(value) or type(value.get('actual_exit_code')) is not int
            or value.get('actual_exit_code') != 0 or value.get('stop_reason') is not None
            or value.get('failure') is not None or value.get('cleanup_errors') != []):
        raise ValueError('Actual native exit0 with no stop/failure/cleanup required')
    normal = (value.get('status') == 'completed'
              and value.get('journal_error') is None
              and value.get('final_record_retained') is True)
    logging_only = (value.get('status') == 'failed'
                    and value.get('final_record_retained') is False
                    and value.get('journal_error') == dict(
                        type='TimeoutError', message='Final logging timed out'))
    if not (normal or logging_only):
        raise ValueError('Only completed supervision or exact retained final-logging failure')
    return {key:value.get(key) for key in fields}


def metadata_bindings(stages, locations):
    evidence, output, journals = (locations[key] for key in ('evidence', 'pilot', 'journals'))
    required = dict(preparation=evidence/'preparation.json',
        native_report=output/'report.json', returned_supervision=evidence/'returned-supervision-1.json',
        work_journal=journals/'work.jsonl', io_journal=journals/'io.jsonl')
    optional = dict(original_failure=evidence/'failure.json', original_acceptance=evidence/'acceptance.json',
        original_closing=evidence/'closing-bindings.json',
        provisional_supervisor_result=evidence/'supervision-1/result.json')
    required.update({key:path for key,path in optional.items() if path.exists() or path.is_symlink()})
    return {key:stages._digest(str(path)) for key,path in required.items()}


def close_export(group, run_id='run-01'):
    if type(group) is not int or group not in (4,8) or run_id != 'run-01':
        raise ValueError('Only fixed G4/G8 pilot run-01 exports are in scope')
    stages = load_stages(); locations = stages.paths('pilot', group, run_id)
    destination = locations['evidence']/'export-observation.json'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Exclusive export observation already exists')
    before = metadata_bindings(stages, locations)
    consumer = {name:stages._digest(str(ROOT/name)) for name in SOURCES}
    record = json.loads((locations['evidence']/'preparation.json').read_text())
    if (record.get('stage') != 'pilot' or record.get('group') != group
            or record.get('run_id') != run_id):
        raise ValueError('Fixed pilot preparation selector mismatch')
    stages.verify_prepared(record)
    supervision = json.loads((locations['evidence']/'returned-supervision-1.json').read_text())
    observed = supervision_observation(supervision)
    results = stages.check_results(record)
    if set(results['checks']) != set(CHECKS) or any(results['checks'][key] is not True for key in CHECKS):
        raise ValueError('All three actual completed16/export consistency checks required')
    export = results['exports']['pilot']
    if export.get('completed_updates') != 16 or export.get('report_binding') != before['native_report']:
        raise ValueError('Actual completed16 export/report binding differs')
    stages.verify_prepared(record)
    after = metadata_bindings(stages, locations)
    if after != before or consumer != {name:stages._digest(str(ROOT/name)) for name in SOURCES}:
        raise ValueError('Original evidence or closure source changed during consistency closure')
    receipt = dict(schema=SCHEMA, status=STATUS, stage='pilot', group=group, run_id=run_id,
        preparation_binding=before['preparation'], native_report_binding=before['native_report'],
        returned_supervision_binding=before['returned_supervision'],
        supervision_value_sha256=stages.canonical_hash(supervision), supervision_observation=observed,
        checks=results['checks'], export_binding=export, recipe=record['recipe'],
        local_model_binding=record['local_model_binding'], observed_geometry=record['observed_geometry'],
        source_bindings=record['source_bindings'], input_bindings=record['input_bindings'],
        prerequisite_bindings=record['prerequisite_bindings'], consumer_source_bindings=consumer,
        before_bindings=before, after_bindings=after,
        scope='Explicit observed-export consistency only: original supervision/failure bytes and values unchanged; '
              'no pilot acceptance retrofit, provisional-result promotion, quality claim, refit or native launch; '
              'parent-side payload/export hashes are outside native work23/I/O9 ledgers')
    receipt['observation_sha256'] = stages.canonical_hash(receipt)
    stages.retain(destination, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-size', required=True, type=int, choices=(4,8))
    parser.add_argument('--run-id', default='run-01', choices=('run-01',))
    args = parser.parse_args(argv)
    receipt = close_export(args.group_size, args.run_id)
    print(json.dumps(dict(status=receipt['status'], group=receipt['group'],
                         observation_sha256=receipt['observation_sha256'])))
    return 0


if __name__ == '__main__': raise SystemExit(main())

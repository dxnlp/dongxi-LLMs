"""One fixed retry after the retained concurrent-CPU conflict; no generic argv."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_supervise
spec=importlib.util.spec_from_file_location('fixed_replay_review',ROOT/'scripts/run_native_sft_replay_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
EVIDENCE=Path(__file__).parent


def main():
    launch=json.loads((EVIDENCE/'launch-2.json').read_text())
    for collection in ('source_bindings','input_bindings'):
        for row in launch[collection].values():
            executable=row['path']==launch['argv'][0]
            if _digest(row['path'],executable=executable)!=row:
                raise ValueError('Retained binding changed; retry refused')
    original=ROOT/'outputs/native-sft-full-replay-20261005-original'
    resumed=ROOT/'outputs/native-sft-full-replay-20261005-resumed02'
    if resumed.exists():raise FileExistsError(resumed)
    argv=list(launch['argv']);argv[argv.index('--output')+1]=str(resumed)
    declaration='Goal-authorized fixed continuation retry after retained CPUCLI conflict; original horizon/caps/checkpoint/dualjournals unchanged; extra900s external invocation recorded separately.'
    runner.retain(EVIDENCE/'launch-retry02.json',dict(argv=argv,operator_declaration=declaration,
        original_attempt=launch,collector=_digest(str(Path(__file__))),
        retry_spec=_digest(str(ROOT/'experiments/specs/2026-10-05-native-sft-full-replay-retry.md'))))
    external=_supervise(argv,EVIDENCE/'supervision-retry02',native=True,seconds=900,
        probe=_native_probe,operator_declaration=declaration)
    runner.retain(EVIDENCE/'returned-supervision-retry02.json',external)
    if external['status']!='completed' or external['actual_exit_code']!=0:
        raise RuntimeError('Retry failed; retain actual receipt, no automatic cap change')
    rows=lambda path:[json.loads(line) for line in path.read_text().splitlines()]
    fields=('update','answer_nll','targets','gradient_norm','cursor',
        'cumulative_supervised_targets','processed_positions','cumulative_processed_positions')
    project=lambda values:[{k:row[k] for k in fields} for row in values]
    expected=project(rows(original/'metrics.jsonl')[10:]);actual=project(rows(resumed/'metrics.jsonl'))
    tensor_a=runner.tensor_bytes(original/'checkpoint-000020.pt')
    tensor_b=runner.tensor_bytes(resumed/'checkpoint-000020.pt')
    a=json.loads((original/'result.json').read_text());b=json.loads((resumed/'result.json').read_text())
    samplekeys=('id','prompt_ids','generated_ids','text','reference','exact_match',
        'stopped_on_end','stop_reason','truncated','error')
    samples=lambda values:[{k:r[k] for k in samplekeys} for r in values]
    checks=dict(ten_replayed_updates=len(actual)==10,exact_numerical_tail=expected==actual,
        exact_serialized_tensor_entries=tensor_a==tensor_b,
        exact_generated_token_stop_records=samples(a['samples'])==samples(b['samples']))
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        expected_numerical_tail=expected,actual_numerical_tail=actual,
        tensor_entries_original=tensor_a,tensor_entries_replayed=tensor_b,
        original_result=a,replayed_result=b,
        earlier_failed_attempt=json.loads((EVIDENCE/'failure.json').read_text()),
        replay_external={k:external[k] for k in ('actual_exit_code','child_seconds',
            'minimum_sampled_available_bytes','final_record_retained','queue_feeder_shutdown')},
        scope='actual same-Spark pretrained BF16 completed-update replay; retained failedattempt; notPythonpicklemetadata, cross-machine, mergedLoRA,400updatepilot or publicationbehavior')
    for label,directory in (('original',original),('replayed02',resumed)):
        for name in ('metrics.jsonl','result.json','recovery-contract.json',
            'baseline-generation-raw.jsonl','final-generation-raw.jsonl',
            'baseline-nll-raw.jsonl','final-nll-raw.jsonl'):
            with (EVIDENCE/f'retained-{label}-{name}').open('xb') as handle:
                handle.write((directory/name).read_bytes())
    runner.retain(EVIDENCE/'acceptance-retry02.json',report)
    print(json.dumps(dict(status=report['status'],checks=checks,evidence=str(EVIDENCE))))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

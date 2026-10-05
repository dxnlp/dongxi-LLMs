#!/usr/bin/env python3
"""Closed offline reasoning rows; tokenizer-only preparation is the default.

Each explicit execution supervises one fixed interface/output-budget pair for
900 seconds. Its five cells retain the original four sampled seeds and greedy
diagnostic on all twenty original prompts. There is no arbitrary argv, model
acquisition, prompt rewrite, cap increase, automatic retry or training route.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dongxi_llms.native_profile_supervisor import (
    MODEL, REVISION, _digest, _native_probe, _new_target, _supervise)
from dongxi_llms.reasoning_generation import (
    SOURCE_FILES as GENERATION_SOURCES, freeze_local_contract,
    load_local_tokenizer, run_generation, serialize_prompt)
from dongxi_llms.reasoning_evaluation import replay_records
from dongxi_llms.run_identity import artifact_hashes, cached_snapshot, canonical_hash
from dongxi_llms.staged_campaign import evaluation_contracts

ROWS = ('base-raw', 'base-chat', 'instruct-thinking-off', 'instruct-thinking-on')
BUDGETS = (32, 128)
SEEDS = (1009, 1019, 1029, 1039)
SECONDS = 900
CONTEXT = 512
GIB = 1024**3
INTERPRETER = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
INSTRUCT_MODEL = 'Qwen/Qwen3-0.6B'
INSTRUCT_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SOURCES = tuple(dict.fromkeys((*GENERATION_SOURCES,
    'src/dongxi_llms/staged_campaign.py',
    'src/dongxi_llms/native_profile_supervisor.py',
    'src/dongxi_llms/campaign_supervisor.py',
    'scripts/run_native_reasoning_baselines.py',
    'tests/test_native_reasoning_baselines.py')))


def retain(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def paths(row, budget, run_id):
    if (row not in ROWS or type(budget) is not int or budget not in BUDGETS
            or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}', run_id) is None):
        raise ValueError('Only fixed reasoning rows,32/128 budgets and run-NN are supported')
    stem = f'native-reasoning-{row}-cap{budget}-20261005-{run_id}'
    return dict(evidence=ROOT/'experiments/reports'/stem, output=ROOT/'outputs'/stem)


def source_bindings():
    return {name: _digest(str(ROOT/name)) for name in SOURCES}


def local_model_binding(row):
    if row not in ROWS:
        raise ValueError('Unknown fixed reasoning row')
    model, revision = ((MODEL, REVISION) if row.startswith('base-')
                       else (INSTRUCT_MODEL, INSTRUCT_REVISION))
    snapshot = cached_snapshot(model, revision)
    files = artifact_hashes(snapshot)
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Complete already cached full model weights are required')
    return dict(path=str(snapshot), model=model, revision=revision, files=files,
        revision_evidence='Declared immutable local cache ID and streamed actual artifact bytes; upstream correspondence is not remotely verified by this adapter')


def logical_panel():
    panel = evaluation_contracts(ROOT)['reasoning-panel-v1']
    if (len(panel['items']) != 20 or panel['decoding']['seeds'] != list(SEEDS)
            or panel['budgets'] != {'answer-only': 32, 'short-work': 128}
            or panel['greedy_diagnostic'] is not True):
        raise ValueError('Original twenty-item reasoning panel/decoding cells changed')
    # Copy the entire annotated source row, including source/task groups,
    # references, original prompt, problem, train overlap and campaign role.
    return deepcopy(panel)


def fixed_inputs():
    values = dict(math_items=ROOT/'fixtures/reasoning-controls/math_items.json',
        custom_template=ROOT/'experiments/data/instruction_interface_v1.jinja',
        interpreter=Path(INTERPRETER), environment_lock=Path(ENVIRONMENT_LOCK))
    return {key: _digest(str(path), executable=key == 'interpreter')
            for key, path in values.items()}


def fixed_argv(row, budget, run_id):
    paths(row, budget, run_id)
    return [INTERPRETER, str(ROOT/'scripts/run_native_reasoning_baselines.py'),
        '--row', row, '--budget', str(budget), '--run-id', run_id, '--child']


def settings(row, budget, tokenizer, decoding, seed):
    paths(row, budget, 'run-00')
    if decoding not in ('sample', 'greedy') or seed not in SEEDS:
        raise ValueError('Only original sampled seeds and greedy diagnostic are supported')
    raw = row == 'base-raw'
    template = (None if raw else
        (ROOT/'experiments/data/instruction_interface_v1.jinja').read_text()
        if row == 'base-chat' else tokenizer.chat_template)
    if not raw and (not isinstance(template, str) or not template):
        raise ValueError('Actual saved native/custom chat template required')
    if row.startswith('instruct-') and 'enable_thinking' not in template:
        raise ValueError('Native instruct template must expose enable_thinking')
    vocabulary = tokenizer.get_vocab()
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in ('<|endoftext|>', '<|im_end|>')]
    if (any(token not in vocabulary for token in ('<|endoftext|>', '<|im_end|>'))
            or stop_ids != [vocabulary['<|endoftext|>'], vocabulary['<|im_end|>']]
            or len(set(stop_ids)) != 2):
        raise ValueError('Observed distinct end-of-text/message-end token IDs required')
    thinking = ('not-applicable' if raw else 'template-default' if row == 'base-chat'
                else 'disabled' if row.endswith('-off') else 'enabled')
    return dict(template_id='reasoning-panel-v1/'+row, thinking_mode=thinking,
        decoding=dict(mode=decoding, seed=seed, temperature=1., top_k=None, top_p=1.),
        stopping=dict(eos_token_ids=[stop_ids[0]], turn_stop_token_ids=[stop_ids[1]],
            pad_token_id=tokenizer.pad_token_id), max_new_tokens=budget,
        generation=dict(input_mode='raw' if raw else 'chat', template=template,
            context_window=CONTEXT, samples=1, device='cuda', dtype='bfloat16',
            add_special_tokens=False, max_run_seconds=SECONDS,
            scoring_text='decode_without_terminal_stop'))


def prepare(row, budget, run_id):
    locations = paths(row, budget, run_id)
    for path in locations.values():
        _new_target(str(path))
    evidence = locations['evidence']
    evidence.mkdir(mode=0o700)
    try:
        sources, inputs, model, panel = source_bindings(), deepcopy(fixed_inputs()), local_model_binding(row), logical_panel()
        # The preparation route loads tokenizer assets only, never model weights.
        tokenizer = load_local_tokenizer(model['path'])
        items = panel['items']
        retain(evidence/'items.json', items)
        retain(evidence/'logical-panel.json', panel)
        cells = []
        for decoding, seed in [('sample', value) for value in SEEDS] + [('greedy', SEEDS[0])]:
            name = f'{decoding}-{seed}'
            contract = freeze_local_contract(items, settings(row, budget, tokenizer, decoding, seed), tokenizer)
            contract['panel_binding'] = dict(logical_contract_sha256=panel['logical_contract_sha256'],
                original_items_sha256=panel['items_sha256'], row=row,
                prompt_policy='Original prompt text unchanged; chat adds only the declared serialization')
            contract['identity'] = canonical_hash({k: v for k, v in contract.items() if k != 'identity'})
            encodings = []
            for item in items:
                text, ids = serialize_prompt(tokenizer, item, contract['settings'])
                if len(ids) + budget > CONTEXT:
                    raise ValueError('Original prompt plus output budget exceeds declared context')
                encodings.append(dict(item_id=item['id'], serialized_prompt=text, token_ids=ids))
            retain(evidence/(name+'-contract.json'), contract)
            cells.append(dict(id=name, mode=decoding, seed=seed, contract_id=contract['identity'],
                contract_path=str(evidence/(name+'-contract.json')),
                output=str(locations['output']/name), observed_prompt_encodings=encodings))
        for key, filename in [('items', 'items.json'), ('logical_panel', 'logical-panel.json')]:
            inputs[key] = _digest(str(evidence/filename))
        for cell in cells:
            inputs['contract/'+cell['id']] = _digest(cell['contract_path'])
        closing_inputs = fixed_inputs()
        if source_bindings() != sources or local_model_binding(row) != model or closing_inputs != {key: inputs[key] for key in closing_inputs}:
            raise ValueError('Source/local artifact/input changed during tokenizer preparation')
        record = dict(schema='dongxi-fixed-native-reasoning-baselines-v1', row=row,
            budget=budget, run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
            evidence=str(evidence), output=str(locations['output']),
            source_bindings=sources, input_bindings=inputs, local_model_binding=model,
            logical_panel_sha256=canonical_hash(panel), cells=cells,
            argv=fixed_argv(row, budget, run_id), execution_requested=False,
            limits=dict(external_seconds=SECONDS, reserve_bytes=25*GIB,
                context_window=CONTEXT, cells=5, items_per_cell=20,
                planned_responses=100, maximum_emitted_tokens=100*budget),
            limitations=['Tokenizer-only preparation is not model execution or a remotely authenticated upstream revision',
                'Four sampled seeds plus a separate greedy diagnostic; no best-of-N selection',
                'Both thinking modes use the same instruct checkpoint; interface toggles are not different weights',
                'Original train and RLVR overlap rows remain diagnostics, excluded from heldout summaries',
                'Raw output is graded unchanged; thinking text is not stripped or converted into a favorable answer',
                'Full-prefix generation without KV caching; sampled reserve is not a physical quota',
                '900 seconds covers all five cells; partial/errors remain evidence, no automatic retry'])
        record['preparation_sha256'] = canonical_hash(record)
        retain(evidence/'preparation.json', record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def verify_prepared(record):
    locations = paths(record['row'], record['budget'], record['run_id'])
    if (canonical_hash({k: v for k, v in record.items() if k != 'preparation_sha256'}) != record.get('preparation_sha256')
            or record['argv'] != fixed_argv(record['row'], record['budget'], record['run_id'])
            or any(record[key] != str(path) for key, path in locations.items())):
        raise ValueError('Closed prepared command/selector/locations changed')
    if source_bindings() != record['source_bindings']:
        raise ValueError('Bound executable source changed')
    if local_model_binding(record['row']) != record['local_model_binding']:
        raise ValueError('Bound local model artifacts changed')
    if canonical_hash(logical_panel()) != record['logical_panel_sha256']:
        raise ValueError('Original logical panel changed')
    expected_inputs = deepcopy(fixed_inputs())
    evidence = locations['evidence']
    for key, filename in [('items', 'items.json'), ('logical_panel', 'logical-panel.json')]:
        expected_inputs[key] = _digest(str(evidence/filename))
    expected_cells = [(f'sample-{seed}', 'sample', seed) for seed in SEEDS] + [(f'greedy-{SEEDS[0]}', 'greedy', SEEDS[0])]
    if [(cell['id'], cell['mode'], cell['seed']) for cell in record['cells']] != expected_cells:
        raise ValueError('Original decoding cell order/coverage changed')
    items = logical_panel()['items']
    if json.loads((evidence/'items.json').read_text()) != items:
        raise ValueError('Original annotated prompts/references changed')
    for cell in record['cells']:
        path = evidence/(cell['id']+'-contract.json')
        if cell['contract_path'] != str(path) or cell['output'] != str(locations['output']/cell['id']):
            raise ValueError('Fixed contract/output location changed')
        expected_inputs['contract/'+cell['id']] = _digest(str(path))
        contract = json.loads(path.read_text())
        replay_records(items, [], contract)
        if contract['identity'] != cell['contract_id']:
            raise ValueError('Frozen original contract identity changed')
    if expected_inputs != record['input_bindings']:
        raise ValueError('Bound original input/contract bytes changed')


def output_summary(record):
    items = logical_panel()['items']
    index = {item['id']: item for item in items}
    cells, all_rows, partial_rows, unreadable = [], [], [], []
    def read_jsonl(path):
        if not path.exists():
            return []
        values = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                unreadable.append(dict(path=str(path), line=number, error=str(error),
                    scope='Original bytes remain retained; this line is not a completed record'))
        return values
    for cell in record['cells']:
        output = Path(cell['output'])
        responses = output/'responses.jsonl'
        rows = read_jsonl(responses)
        contract = json.loads(Path(cell['contract_path']).read_text())
        graded = replay_records(items, rows, contract)['rows']
        for row in graded:
            item = index[row['item_id']]
            row['panel_annotation'] = {key: deepcopy(item[key]) for key in (
                'family', 'template_id', 'problem', 'rlvr_train_problem_overlap', 'campaign_role')}
            row['cell_id'] = cell['id']
        completed = {(row['item_id'], row['sample_index']) for row in rows}
        latest_partial = {}
        for event in read_jsonl(output/'events.jsonl'):
            if event.get('stage') == 'partial_response':
                partial = event['record']
                key = (partial['item_id'], partial['sample_index'])
                if key not in completed:
                    latest_partial[key] = dict(partial, cell_id=cell['id'])
        partial_rows.extend(latest_partial.values())
        summary_path = output/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        cells.append(dict(id=cell['id'], summary=summary, records=len(rows),
            missing_item_ids=sorted(set(index)-{row['item_id'] for row in rows})))
        all_rows.extend(graded)
    def metrics(rows):
        return dict(records=len(rows), correct=sum(row['correct'] for row in rows),
            natural_stops=sum(row['natural_termination'] for row in rows),
            truncated=sum(row['truncated'] for row in rows),
            statuses=dict(Counter(row['status'] for row in rows)),
            costs={key: sum(row['cost'][key] for row in rows) for key in (
                'generation_tokens', 'attempted_forward_tokens', 'model_forward_tokens',
                'attempted_forward_calls', 'forward_calls', 'wall_seconds')})
    heldout = [row for row in all_rows if row['panel_annotation']['campaign_role'] == 'heldout-controlled-slice']
    cost_keys = ('generation_tokens', 'attempted_forward_tokens', 'model_forward_tokens',
        'attempted_forward_calls', 'forward_calls', 'wall_seconds')
    return dict(cells=cells, rows=all_rows, all_original_rows=metrics(all_rows), heldout=metrics(heldout),
        incomplete_attempts=partial_rows, unreadable_jsonl_lines=unreadable,
        incomplete_observed_costs={key: sum(row['cost'][key] for row in partial_rows) for key in cost_keys},
        cost_boundary='Completed-record totals and latest retained nonfinal per-attempt costs are separate; a killed in-flight forward can have unobserved cost',
        heldout_item_ids=[item['id'] for item in items if item['campaign_role'] == 'heldout-controlled-slice'],
        planned_responses=100, missing_responses=100-len(all_rows),
        scope='Every observed response, including train overlap, failures and truncations; heldout subset is separately labeled')


def child(row, budget, run_id):
    evidence = paths(row, budget, run_id)['evidence']
    record = json.loads((evidence/'preparation.json').read_text())
    launch = json.loads((evidence/'launch.json').read_text())
    if launch.get('preparation_sha256') != record.get('preparation_sha256') or launch.get('argv') != record['argv']:
        raise ValueError('Parent-retained closed launch binding required')
    verify_prepared(record)
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[name] = '1'
    output = _new_target(record['output'])
    output.mkdir(mode=0o700)
    try:
        for cell in record['cells']:
            verify_prepared(record)
            result = run_generation(root=ROOT, items_path=evidence/'items.json',
                contract_path=cell['contract_path'], checkpoint=record['local_model_binding']['path'],
                output=cell['output'], environment_lock=ENVIRONMENT_LOCK, allow_cuda=True)
            verify_prepared(record)
            if result['status'] != 'completed' or result['record_count'] != 20:
                raise RuntimeError('Original cell did not complete all twenty attempts; retained outputs remain')
        retain(output/'child-summary.json', output_summary(record))
        return 0
    except BaseException as error:
        retain(output/'child-failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def execute(record, declaration):
    if type(declaration) is not str or not 12 <= len(declaration) <= 4096:
        raise ValueError('Explicit recorded goal-authorized reasoning declaration required')
    evidence = Path(record['evidence'])
    supervision = None
    try:
        verify_prepared(record)
        _new_target(record['output'])
        retain(evidence/'launch.json', dict(argv=record['argv'], operator_declaration=declaration,
            preparation_sha256=record['preparation_sha256']))
        supervision = _supervise(record['argv'], evidence/'supervision', native=True,
            seconds=SECONDS, probe=_native_probe, operator_declaration=declaration)
        retain(evidence/'returned-supervision.json', supervision)
        verify_prepared(record)
        retain(evidence/'closing-bindings.json', dict(status='unchanged',
            preparation_sha256=record['preparation_sha256'], source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'], local_model_binding=record['local_model_binding']))
        summary = output_summary(record)
        checks = dict(actual_exit0=supervision['status'] == 'completed' and supervision['actual_exit_code'] == 0,
            all_five_cells_completed=all(cell['summary'] is not None and cell['summary']['status'] == 'completed'
                and cell['records'] == 20 and not cell['missing_item_ids'] for cell in summary['cells']),
            all_original_responses=summary['missing_responses'] == 0,
            no_unreadable_jsonl=not summary['unreadable_jsonl_lines'])
        retain(evidence/'acceptance.json', dict(status='passed' if all(checks.values()) else 'failed',
            row=record['row'], budget=record['budget'], checks=checks,
            actual_exit_code=supervision['actual_exit_code'], child_seconds=supervision['child_seconds'],
            minimum_sampled_available_bytes=supervision['minimum_sampled_available_bytes'],
            local_model_binding=record['local_model_binding'], evaluation=summary,
            limitations=record['limitations'], scope='Actual baseline/interface cells; correctness may remain negative, not training or broad reasoning superiority'))
        if not all(checks.values()):
            raise RuntimeError('Owned reasoning baseline incomplete/failed; partial records and costs retained')
        print(json.dumps(dict(status='passed', row=record['row'], budget=record['budget'], evidence=str(evidence))))
        return 0
    except BaseException as error:
        retain(evidence/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            supervision_retained=supervision is not None, scope='No automatic retry or cap/recipe change'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', choices=ROWS, required=True)
    parser.add_argument('--budget', type=int, choices=BUDGETS, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-declaration')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    paths(args.row, args.budget, args.run_id)
    if args.child:
        if args.execute or args.operator_declaration is not None:
            parser.error('Internal owned child takes only the closed prepared selectors')
        return child(args.row, args.budget, args.run_id)
    if args.execute and (not args.operator_declaration or not 12 <= len(args.operator_declaration) <= 4096):
        parser.error('Explicit recorded goal-authorized reasoning declaration required')
    record = prepare(args.row, args.budget, args.run_id)
    if args.execute:
        return execute(record, args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed', row=args.row,
        budget=args.budget, evidence=record['evidence'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

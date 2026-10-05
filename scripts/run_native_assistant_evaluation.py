"""Fixed120-item publication evaluation; preparation never loads model weights."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _supervise
from dongxi_llms.reasoning_generation import freeze_local_contract, load_local_tokenizer
from dongxi_llms.run_identity import artifact_hashes

BASE = Path('/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
PYTHON = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'


def retain(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def publication_items():
    path=ROOT/'outputs/course-sft-interface-v1/test.jsonl'
    card=json.loads((ROOT/'experiments/data/instruction-interface-v1-data-card.json').read_text())
    if _digest(str(path))['sha256'] != card['splits']['test']['sha256']:
        raise ValueError('Publication bytes differ from the original data card')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 120 or len({row['group'] for row in rows}) != 40:
        raise ValueError('Original120-item/40-source-group publication panel required')
    return [dict(id=row['id'], source_group=row['group'], split='publication',
        task=row['family'], prompt=row['messages'][-2]['content'],
        messages=row['messages'][:-1], kind='text',
        reference=row['messages'][-1]['content'],
        format_policy='any', extraction='whole') for row in rows]


def checkpoint(label, run_id):
    if re.fullmatch(r'run-[0-9]{2}', run_id) is None:
        raise ValueError('Fixed run-NN identity required')
    if label == 'base': return BASE
    if label == 'full400':
        return ROOT/'outputs'/f'native-sft-full-pilot400-20261005-{run_id}'/'policy'
    if label == 'lora400-fp32':
        return ROOT/'outputs'/f'native-sft-lora-pilot400-merged-20261005-{run_id}'/'policy'
    raise ValueError('Only declared actual full policies are supported')


def strict_result(row, item):
    """Score decoded answer content, not its retained terminal special token."""
    return dict(item_id=row['item_id'],source_group=row['source_group'],
        exact=row['error'] is None and row['response_text'].strip()==item['reference'],
        natural_stop=row['stop_reason'] in ('eos','turn_stop'),truncated=row['truncated'],
        stop_reason=row['stop_reason'],generated_tokens=row['generated_tokens'])


def source_bindings(sources):
    return {str(path):_digest(str(path),executable=str(path)==PYTHON) for path in sources}


def parent_evidence(label, run_id, parent):
    """A directory name cannot certify a400-update artifact."""
    if label=='base':
        return {}
    mode='full' if label=='full400' else 'lora'
    receipt=ROOT/'experiments/reports'/f'native-sft-{mode}-pilot400-20261005-{run_id}'/'acceptance.json'
    accepted=json.loads(receipt.read_text())
    if (accepted.get('status')!='passed' or not accepted.get('checks')
            or any(value is not True for value in accepted['checks'].values())
            or accepted.get('result',{}).get('updates')!=400
            or accepted['result'].get('resume_checkpoint_update')!=0):
        raise ValueError('Actual fresh completed400 pilot acceptance required')
    exported=accepted['exported_policy']
    if artifact_hashes(exported['path'])!=exported['files']:
        raise ValueError('Accepted pilot export bytes changed')
    if label=='full400':
        if exported['path']!=str(parent) or artifact_hashes(parent)!=exported['files']:
            raise ValueError('Publication full400 differs from the accepted export')
    else:
        genealogy=json.loads((parent/'course-genealogy.json').read_text())
        if (genealogy.get('method')!='explicit-local-PEFT-merge'
                or genealogy.get('merge_precision')!='CPU FP32'
                or genealogy.get('parent_adapter')!=exported['path']
                or genealogy.get('adapter_files')!=exported['files']):
            raise ValueError('Publication merge differs from the accepted LoRA400 adapter')
        merged_receipt=ROOT/'experiments/reports'/f'native-sft-lora-pilot400-merged-20261005-{run_id}'/'acceptance.json'
        merged=json.loads(merged_receipt.read_text())
        if (merged.get('status')!='passed' or not merged.get('checks')
                or any(value is not True for value in merged['checks'].values())
                or merged.get('exported_policy',{}).get('path')!=str(parent)
                or merged['exported_policy'].get('files')!=artifact_hashes(parent)):
            raise ValueError('Actual passed byte-bound FP32 merge/reload acceptance required')
        return dict(acceptance=_digest(str(receipt)),merge_acceptance=_digest(str(merged_receipt)),
            invocation_id=accepted['result']['invocation_id'],updates=400,exported_policy=exported,
            scope='Actual selected final400 adapter and separately verified FP32 merge/reload, not quality-selected')
    return dict(acceptance=_digest(str(receipt)),invocation_id=accepted['result']['invocation_id'],
        updates=400,exported_policy=exported,scope='Actual selected final400 parent, not quality-selected')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('base','full400','lora400-fp32'), required=True)
    parser.add_argument('--run-id', default='run-01')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-declaration')
    args = parser.parse_args(argv)
    parent = checkpoint(args.checkpoint, args.run_id)
    evidence = ROOT/'experiments/reports'/f'native-assistant-publication-20261005-{args.checkpoint}-{args.run_id}'
    evidence.mkdir(mode=0o700, exist_ok=False)
    ancestry=parent_evidence(args.checkpoint,args.run_id,parent)
    items = publication_items()
    tokenizer = load_local_tokenizer(parent)
    settings = dict(template_id='original-instruction-interface-v1-publication',
        thinking_mode='template-default',
        decoding=dict(mode='greedy',seed=1010,temperature=1,top_k=None,top_p=1),
        stopping=dict(eos_token_ids=[151643],turn_stop_token_ids=[151645],pad_token_id=151643),
        max_new_tokens=64,
        generation=dict(input_mode='chat',template=(ROOT/'experiments/data/instruction_interface_v1.jinja').read_text(),
            context_window=512,samples=1,device='cuda',dtype='bfloat16',
            add_special_tokens=False,max_run_seconds=900,scoring_text='decode_without_terminal_stop'))
    contract = freeze_local_contract(items, settings, tokenizer)
    retain(evidence/'items.json', items)
    retain(evidence/'contract.json', contract)
    sources = (Path(__file__), ROOT/'src/dongxi_llms/native_profile_supervisor.py',
        ROOT/'src/dongxi_llms/reasoning_generation.py',ROOT/'src/dongxi_llms/reasoning_evaluation.py',
        ROOT/'src/dongxi_llms/run_identity.py',ROOT/'src/dongxi_llms/sampling_likelihood_lab.py',
        ROOT/'scripts/generate_reasoning_records.py',ROOT/'experiments/data/instruction_interface_v1.jinja',
        ROOT/'experiments/data/instruction-interface-v1-data-card.json',
        ROOT/'outputs/course-sft-interface-v1/test.jsonl',evidence/'items.json',evidence/'contract.json',
        Path(LOCK),Path(PYTHON))
    bindings = source_bindings(sources)
    model_files = artifact_hashes(parent)
    command = [PYTHON,str(ROOT/'scripts/generate_reasoning_records.py'),
        '--checkpoint',str(parent),'--items',str(evidence/'items.json'),
        '--contract',str(evidence/'contract.json'),'--output',str(evidence/'generation'),
        '--environment-lock',LOCK,'--allow-cuda']
    retain(evidence/'preparation.json',dict(command=command, bindings=bindings,
        checkpoint_files=model_files,parent_evidence=ancestry,contract=contract['identity'],items=120,source_groups=40,
        maximum_new_tokens=7680,external_seconds=900,reserve_bytes=25*1024**3,
        scoring='Retain generic text replay; strict case-sensitive stripped whole-response equality is separate.',
        precision='All generation BF16; LoRA label refers to FP32 merged storage, not FP32 evaluation or BF16 merge equivalence.',
        execution_requested=args.execute))
    if not args.execute:
        print(json.dumps(dict(status='prepared-not-executed',evidence=str(evidence))))
        return 0
    if not args.operator_declaration or len(args.operator_declaration)<12:
        raise ValueError('Explicit retained goal-authorized scope required')
    try:
        if (source_bindings(sources) != bindings or artifact_hashes(parent)!=model_files
                or parent_evidence(args.checkpoint,args.run_id,parent)!=ancestry):
            raise ValueError('Source/input/checkpoint drift before generation')
        retain(evidence/'launch.json',dict(command=command,operator_declaration=args.operator_declaration))
        result = _supervise(command,evidence/'supervision',native=True,seconds=900,
            probe=_native_probe,operator_declaration=args.operator_declaration)
        retain(evidence/'returned-supervision.json',result)
        if result['status']!='completed' or result['actual_exit_code']!=0:
            raise RuntimeError('Actual generation failed; partial attempts retained')
        if (source_bindings(sources) != bindings or artifact_hashes(parent)!=model_files
                or parent_evidence(args.checkpoint,args.run_id,parent)!=ancestry):
            raise ValueError('Source/input/checkpoint drift during generation')
        rows = [json.loads(line) for line in (evidence/'generation/responses.jsonl').read_text().splitlines()]
        index={item['id']:item for item in items}
        strict=[strict_result(row,index[row['item_id']]) for row in rows]
        if len(rows)!=120 or len({row['item_id'] for row in rows})!=120:
            raise ValueError('Publication attempts missing or duplicated')
        retain(evidence/'strict-results.json',dict(status='completed',checkpoint=args.checkpoint,
            contract=contract['identity'],attempts=120,source_groups=40,rows=strict,
            exact=sum(row['exact'] for row in strict),natural_stops=sum(row['natural_stop'] for row in strict),
            truncations=sum(row['truncated'] for row in strict),generated_tokens=sum(row['generated_tokens'] for row in strict),
            source_stable=True, scope='Original held-out lexical values/shared task templates; not a general assistant benchmark.'))
        print(json.dumps(dict(status='completed',evidence=str(evidence),exact=sum(row['exact'] for row in strict))),flush=True)
        return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__ == '__main__':
    raise SystemExit(main())

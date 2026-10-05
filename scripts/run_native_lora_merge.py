#!/usr/bin/env python3
"""Closed offline LoRA400 CPU-FP32 merge and CUDA-FP32 reload verification.

Preparation only reads bounded metadata and streams existing artifact hashes.
The owned child alone loads weights and executes the fixed24 forward calls.
No adapter selection, acquisition, tolerance override, arbitrary argv or retry.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _new_target, _supervise
from dongxi_llms.run_identity import (
    artifact_hashes, assert_compatible, canonical_hash, environment_identity,
    tokenizer_interface, validate_interface)

MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
BASE = Path('/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots')/REVISION
PYTHON = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
GIB = 1024**3
SECONDS = 900
ATOL, RTOL = .002, .001
PREFIX_IDS = ('dev-80-copy', 'dev-80-reverse', 'dev-80-extract',
    'dev-81-copy', 'dev-81-reverse', 'dev-81-extract', 'dev-82-copy', 'dev-82-reverse')
PREFIX_LENGTHS = (29, 36, 37, 29, 36, 37, 29, 36)
DEV_SHA256 = '7e615ca255e9d026534a47cddf82ff34fff5a9021c5132b6b536cf27080ac62e'
TRAIN_SHA256 = '50e0375220c97ee3a75509148f5024c3f3120185af3b5878ca5b8e8f7e01d234'
TEMPLATE_SHA256 = 'f26fd6284d6bc05d057a4b5800e2a268a85df6546edf484d979e23a51ed7886d'
INTERFACE_SHA256 = 'e869c7e93ad366530f30cb62e78122577d84f0f95fe2dad5a0e0923d4d031f63'
SOURCES = ('scripts/run_native_lora_merge.py', 'tests/test_native_lora_merge.py',
    'src/dongxi_llms/run_identity.py', 'src/dongxi_llms/native_profile_supervisor.py',
    'experiments/reports/2026-10-05-native-lora-fp32-merge-reload/merge-child.py')
LIMITS = dict(invocations=1, external_seconds=SECONDS, reserve_bytes=25*GIB,
    output_planning_bytes=4*GIB, verification_prefixes=8, forward_calls=24,
    full_prefix_positions=807, atol=ATOL, rtol=RTOL, exact_reloaded_logits=True,
    merge_device='cpu', merge_dtype='float32', storage_dtype='float32',
    forward_device='cuda', forward_dtype='float32', attention_backend='sdpa')


def retain(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def paths(run_id):
    if type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}', run_id) is None:
        raise ValueError('Fixed run-NN identity required')
    stem = f'native-sft-lora-pilot400-merged-20261005-{run_id}'
    return dict(evidence=ROOT/'experiments/reports'/stem, output=ROOT/'outputs'/stem,
        policy=ROOT/'outputs'/stem/'policy',
        adapter=ROOT/'outputs'/f'native-sft-lora-pilot400-20261005-{run_id}'/'policy',
        parent_report=ROOT/'experiments/reports'/f'native-sft-lora-pilot400-20261005-{run_id}')


def source_bindings():
    return {name: _digest(str(ROOT/name)) for name in SOURCES}


def input_paths(locations):
    pilot = locations['adapter'].parent
    report = locations['parent_report']
    return dict(acceptance=report/'acceptance.json', preparation=report/'preparation.json',
        closing=report/'closing-bindings.json', result=pilot/'result.json', config=pilot/'config.json',
        genealogy=locations['adapter']/'course-genealogy.json',
        adapter_config=locations['adapter']/'adapter_config.json',
        train=ROOT/'outputs/course-sft-interface-v1/train.jsonl',
        dev=ROOT/'outputs/course-sft-interface-v1/dev.jsonl',
        template=ROOT/'experiments/data/instruction_interface_v1.jinja',
        data_card=ROOT/'experiments/data/instruction-interface-v1-data-card.json',
        interpreter=Path(PYTHON), environment_lock=Path(LOCK))


def input_bindings(locations):
    return {role: _digest(str(path), executable=role=='interpreter')
        for role, path in input_paths(locations).items()}


def original_prefix_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows)!=60 or len({row['id'] for row in rows})!=60:
        raise ValueError('Original unique60-item development panel required')
    selected = rows[:8]
    if tuple(row['id'] for row in selected)!=PREFIX_IDS:
        raise ValueError('Only the original fixed first8 development prefixes are supported')
    if any(row['messages'][-1]['role']!='assistant' or row['messages'][-2]['role']!='user'
            for row in selected):
        raise ValueError('Original prompt/answer separation required')
    return selected


def validate_parent_metadata(accepted, result, config, genealogy, adapter_config,
        prepared, closing, *, adapter_path, adapter_files, base_files, inputs):
    if (accepted.get('status')!='passed' or accepted.get('actual_exit_code')!=0
            or not accepted.get('checks') or any(v is not True for v in accepted['checks'].values())
            or accepted.get('result')!=result or result.get('status')!='completed'
            or result.get('updates')!=400 or result.get('resume_checkpoint_update')!=0):
        raise ValueError('Actual fresh completed LoRA400 acceptance required')
    exported = accepted.get('exported_policy', {})
    if (exported.get('path')!=str(adapter_path) or exported.get('files')!=adapter_files
            or exported.get('genealogy')!=genealogy):
        raise ValueError('Accepted LoRA400 export bytes or genealogy changed')
    if (prepared.get('mode')!='lora' or prepared.get('local_base_binding',{}).get('files')!=base_files
            or closing.get('status')!='unchanged'
            or any(closing.get(key)!=prepared.get(key) for key in
                ('source_bindings','input_bindings','local_base_binding'))):
        raise ValueError('Actual closed pilot/base binding required')
    if (config.get('mode')!='lora' or config.get('updates')!=400 or config.get('rank')!=8
            or config.get('seed')!=1212 or config.get('resume_contract') is not None
            or config.get('resume_sha256') is not None
            or config.get('model')!=MODEL or config.get('revision')!=REVISION
            or config.get('tokenizer_revision')!=REVISION):
        raise ValueError('Original fresh pinned rank8 LoRA400 producer required')
    if (genealogy.get('kind')!='PEFT-adapter-requires-pinned-base'
            or genealogy.get('mode')!='lora' or genealogy.get('base_model')!=MODEL
            or genealogy.get('base_revision')!=REVISION
            or genealogy.get('base_checkpoint_files')!=base_files
            or genealogy.get('run_identity_file')!=f"identity-{result['invocation_id']}.json"):
        raise ValueError('Exact accepted LoRA400 Base/invocation genealogy required')
    if (adapter_config.get('peft_type')!='LORA' or adapter_config.get('r')!=8
            or adapter_config.get('lora_alpha')!=8
            or set(adapter_config.get('target_modules',[]))!={'q_proj','v_proj'}
            or adapter_config.get('bias')!='none' or adapter_config.get('lora_dropout')!=0
            or adapter_config.get('base_model_name_or_path')!=str(BASE)
            or adapter_config.get('revision')!=REVISION):
        raise ValueError('Exact original rank8 Q/V LoRA adapter required')
    for role, digest in (('train',TRAIN_SHA256),('dev',DEV_SHA256),('template',TEMPLATE_SHA256)):
        if inputs[role]['sha256']!=digest or config.get(role+'_sha256')!=digest:
            raise ValueError('Original data/template binding changed: '+role)
    if genealogy.get('data_sha256')!=TRAIN_SHA256 or genealogy.get('template_sha256')!=TEMPLATE_SHA256:
        raise ValueError('Original genealogy data/template binding changed')
    interface = validate_interface(genealogy['checkpoint_interface'])
    if (interface['interface_sha256']!=INTERFACE_SHA256
            or interface['source']!={'tokenizer_id':MODEL,'tokenizer_revision':REVISION}
            or interface['generation_stop_ids']!=[151643,151645]):
        raise ValueError('Original full tokenizer/template/stop interface required')
    return dict(invocation_id=result['invocation_id'], updates=400,
        acceptance_sha256=inputs['acceptance']['sha256'], parent_adapter=str(adapter_path),
        adapter_files=deepcopy(adapter_files), base_files=deepcopy(base_files),
        interface=deepcopy(interface), producer_source_bindings=deepcopy(prepared['source_bindings']),
        producer_source_scope='Retained historical producer; not a claim its sources remain current')


def parent_binding(locations, inputs):
    values = {role:json.loads(Path(inputs[role]['path']).read_text()) for role in
        ('acceptance','result','config','genealogy','adapter_config','preparation','closing')}
    adapter_files, base_files = artifact_hashes(locations['adapter']), artifact_hashes(BASE)
    if not any(name.endswith('.safetensors') for name in adapter_files) or 'model.safetensors' not in base_files:
        raise ValueError('Complete existing pinned adapter/Base weights required')
    card = json.loads(Path(inputs['data_card']['path']).read_text())
    if any(card['splits'][role]['sha256']!=inputs[role]['sha256'] for role in ('train','dev')):
        raise ValueError('Original data card population changed')
    original_prefix_rows(inputs['dev']['path'])
    return validate_parent_metadata(values['acceptance'],values['result'],values['config'],
        values['genealogy'],values['adapter_config'],values['preparation'],values['closing'],
        adapter_path=locations['adapter'],adapter_files=adapter_files,base_files=base_files,inputs=inputs)


def fixed_command(run_id):
    paths(run_id)
    return [PYTHON,str(ROOT/'scripts/run_native_lora_merge.py'),'--run-id',run_id,'--child']


def prepare(run_id):
    locations = paths(run_id)
    for role in ('evidence','output'):
        _new_target(str(locations[role]))
    evidence = locations['evidence']
    evidence.mkdir(mode=0o700)
    try:
        if shutil.disk_usage(ROOT/'outputs').free<4*GIB:
            raise RuntimeError('4GiB new output planning reserve required')
        sources, inputs = source_bindings(), input_bindings(locations)
        parent = parent_binding(locations, inputs)
        record = dict(schema='dongxi-fixed-native-lora400-merge-v1',run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),evidence=str(evidence),output=str(locations['output']),
            policy=str(locations['policy']),command=fixed_command(run_id),
            source_bindings=sources,input_bindings=inputs,parent_binding=parent,
            limits=deepcopy(LIMITS),prefix_ids=list(PREFIX_IDS),prefix_lengths=list(PREFIX_LENGTHS),
            execution_requested=False,scope='Actual final400 merge/reload when executed; no quality or BF16-equivalence claim',
            boundaries=['sampled900s/25GiB watchdog, not physical containment',
                'CPU FP32 merge arithmetic and weight storage; CUDA FP32 SDPA validation',
                '24forward calls/807positions outside completed SFT16/I/O9 work budgets',
                'Historical BF16 mismatch and disposable20-update report remain unchanged',
                'No acquisition, implicit retry, tolerance increase or publication launch'])
        record['preparation_sha256'] = canonical_hash(record)
        retain(evidence/'preparation.json', record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


def verify_prepared(record):
    if (record.get('schema')!='dongxi-fixed-native-lora400-merge-v1'
            or canonical_hash({k:v for k,v in record.items() if k!='preparation_sha256'})!=record.get('preparation_sha256')):
        raise ValueError('Prepared command/identity changed')
    locations = paths(record['run_id'])
    if (any(record[key]!=str(locations[key]) for key in ('evidence','output','policy'))
            or record['command']!=fixed_command(record['run_id']) or record['limits']!=LIMITS
            or record['prefix_ids']!=list(PREFIX_IDS) or record['prefix_lengths']!=list(PREFIX_LENGTHS)):
        raise ValueError('Only the closed merge/reload recipe is supported')
    inputs = input_bindings(locations)
    if (source_bindings()!=record['source_bindings'] or inputs!=record['input_bindings']
            or parent_binding(locations,inputs)!=record['parent_binding']):
        raise ValueError('Bound source/input/parent model bytes changed')
    return locations


def comparison(torch, row_id, before, after, *, exact=False):
    if (before.shape!=after.shape or before.dtype!=torch.float32 or after.dtype!=torch.float32
            or not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all())):
        raise ValueError('Finite same-shape FP32 logits required')
    difference = before-after
    report = dict(id=row_id,shape=list(before.shape),max_absolute=float(difference.abs().max()),
        rms=float(difference.square().mean().sqrt()),
        last_position_argmax_equal=bool(before[:,-1].argmax()==after[:,-1].argmax()),
        bitwise_equal=bool(torch.equal(before,after)),atol=ATOL,rtol=RTOL)
    # Return the measured row even when it fails; callers retain it before raising.
    if exact:
        report['passed'] = report['bitwise_equal']
    else:
        report['passed'] = bool(torch.allclose(before,after,atol=ATOL,rtol=RTOL))
    return report


def floating_parameters(torch, model, device):
    count = elements = 0
    for parameter in model.parameters():
        if parameter.is_floating_point():
            if parameter.dtype!=torch.float32 or parameter.device.type!=device:
                raise ValueError('All floating weights must be FP32 on declared device')
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError('Nonfinite model weights')
            count += 1
            elements += parameter.numel()
    if not count:
        raise ValueError('Actual floating model weights required')
    return dict(parameter_tensors=count,parameter_elements=elements,dtype='float32',device=device)


def genealogy(record):
    parent = record['parent_binding']
    return dict(kind='full-HF-model',method='explicit-local-PEFT-merge',merge_precision='CPU FP32',
        base_model=MODEL,base_revision=REVISION,base_checkpoint_files=parent['base_files'],
        parent_adapter=parent['parent_adapter'],adapter_files=parent['adapter_files'],
        parent_invocation_id=parent['invocation_id'],parent_updates=400,
        parent_acceptance_sha256=parent['acceptance_sha256'],
        checkpoint_interface=parent['interface'],template_sha256=TEMPLATE_SHA256,data_sha256=TRAIN_SHA256,
        verification=dict(atol=ATOL,rtol=RTOL,reload='bitwise FP32 logits',forward_precision='CUDA FP32 SDPA'),
        scope='Actual accepted final400 adapter merged; storage FP32, not a BF16-equivalence or quality claim')


def child(run_id):
    locations = paths(run_id)
    evidence = locations['evidence']
    record = json.loads((evidence/'preparation.json').read_text())
    started = time.monotonic()
    attempted_calls = completed_calls = attempted_positions = completed_positions = 0
    stage = 'initial-binding'
    trace = None
    try:
        verify_prepared(record)
        launch = json.loads((evidence/'launch.json').read_text())
        if launch.get('preparation_sha256')!=record['preparation_sha256']:
            raise ValueError('Owned launch must select the exact prepared merge')
        _new_target(str(locations['output']))
        locations['output'].mkdir(mode=0o700)
        trace = (evidence/'forward-events.jsonl').open('x',encoding='utf-8')
        def event(value):
            trace.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
            trace.flush(); os.fsync(trace.fileno())
        event(dict(event='child-start',preparation_sha256=record['preparation_sha256'],seconds=0))
        if shutil.disk_usage(ROOT/'outputs').free<4*GIB:
            raise RuntimeError('4GiB new output planning reserve required before weights')
        stage = 'imports'
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA execution environment required')
        torch.manual_seed(1212)
        torch.cuda.manual_seed_all(1212)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats()
        retain(evidence/'environment.json',dict(environment=environment_identity(LOCK),
            interpreter_binding=_digest(PYTHON,executable=True),lock_binding=_digest(LOCK),
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
            matmul_precision='highest',allow_tf32=False,attention_backend='sdpa'))
        stage = 'tokenizer-interface'
        tokenizer = AutoTokenizer.from_pretrained(locations['adapter'],local_files_only=True,trust_remote_code=False)
        interface = tokenizer_interface(tokenizer,template=tokenizer.chat_template,stop_ids=[151643,151645],
            tokenizer_id=MODEL,tokenizer_revision=REVISION)
        assert_compatible(record['parent_binding']['interface'],interface)
        rows = original_prefix_rows(record['input_bindings']['dev']['path'])
        prefixes = [tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False) for row in rows]
        if tuple(map(len,prefixes))!=PREFIX_LENGTHS:
            raise ValueError('Original fixed8-prefix token geometry changed')
        retain(evidence/'prefixes.json',dict(interface=interface,rows=[dict(id=row['id'],token_ids=ids,
            prompt_tokens=len(ids),messages=row['messages'][:-1]) for row,ids in zip(rows,prefixes)],
            prompt_tokens=sum(map(len,prefixes)),forward_calls=24,full_prefix_positions=807))
        stage = 'parent-load'
        base = AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base,locations['adapter'],local_files_only=True,
            autocast_adapter_dtype=True).float().eval().cuda()
        retain(evidence/'parent-weight-precision.json',floating_parameters(torch,model,'cuda'))
        def logits(value, phase):
            nonlocal attempted_calls,completed_calls,attempted_positions,completed_positions,stage
            answers = []
            with torch.inference_mode():
                for row, ids in zip(rows,prefixes):
                    if attempted_calls>=24 or attempted_positions+len(ids)>807:
                        raise ValueError('Fixed verification forward allowance exceeded')
                    stage = phase+'-forward-'+row['id']
                    attempted_calls += 1; attempted_positions += len(ids)
                    event(dict(event='entered',phase=phase,id=row['id'],call=attempted_calls,
                        input_positions=len(ids),seconds=time.monotonic()-started))
                    output = value(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).logits.detach()
                    if output.dtype!=torch.float32 or not bool(torch.isfinite(output).all()):
                        raise ValueError('Finite FP32 forward output required')
                    answers.append(output.cpu())
                    completed_calls += 1; completed_positions += len(ids)
                    event(dict(event='completed',phase=phase,id=row['id'],call=completed_calls,
                        input_positions=len(ids),shape=list(output.shape),seconds=time.monotonic()-started))
            return answers
        before = logits(model,'adapter')
        stage = 'cpu-merge'
        model = model.cpu()
        torch.cuda.empty_cache()
        retain(evidence/'merge-input-precision.json',floating_parameters(torch,model,'cpu'))
        merged = model.merge_and_unload(safe_merge=True).float().eval()
        retain(evidence/'merged-cpu-precision.json',floating_parameters(torch,merged,'cpu'))
        event(dict(event='cpu-merge-completed',seconds=time.monotonic()-started))
        merged = merged.cuda()
        after = logits(merged,'merged')
        stage = 'merge-comparison'
        comparisons = [comparison(torch,row['id'],a,b) for row,a,b in zip(rows,before,after)]
        retain(evidence/'merge-comparisons.json',dict(atol=ATOL,rtol=RTOL,rows=comparisons))
        if not all(row['passed'] for row in comparisons):
            raise ValueError('Actual LoRA400 FP32 merge differs beyond predeclared tolerance')
        stage = 'save-fp32'
        merged = merged.cpu()
        retain(evidence/'saved-weight-precision.json',floating_parameters(torch,merged,'cpu'))
        _new_target(str(locations['policy']))
        locations['policy'].mkdir(mode=0o700)
        merged.save_pretrained(locations['policy'],safe_serialization=True)
        tokenizer.save_pretrained(locations['policy'])
        retain(locations['policy']/'course-genealogy.json',genealogy(record))
        saved_files = artifact_hashes(locations['policy'])
        retain(evidence/'saved-policy.json',dict(path=str(locations['policy']),files=saved_files))
        del model,base,merged,before
        gc.collect(); torch.cuda.empty_cache()
        stage = 'reload-fp32'
        reloaded = AutoModelForCausalLM.from_pretrained(locations['policy'],local_files_only=True,
            trust_remote_code=False,dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
        retain(evidence/'reloaded-weight-precision.json',floating_parameters(torch,reloaded,'cuda'))
        reload_logits = logits(reloaded,'reloaded')
        stage = 'reload-comparison'
        reload_comparisons = [comparison(torch,row['id'],a,b,exact=True) for row,a,b in zip(rows,after,reload_logits)]
        retain(evidence/'reload-comparisons.json',dict(required='bitwise FP32 logits',rows=reload_comparisons))
        if not all(row['passed'] for row in reload_comparisons):
            raise ValueError('Actual merged reload logits differ bitwise')
        actual_tokenizer = AutoTokenizer.from_pretrained(locations['policy'],local_files_only=True,trust_remote_code=False)
        reconstructed = tokenizer_interface(actual_tokenizer,template=actual_tokenizer.chat_template,
            stop_ids=[151643,151645],tokenizer_id=MODEL,tokenizer_revision=REVISION)
        assert_compatible(interface,reconstructed)
        retain(evidence/'reloaded-interface.json',reconstructed)
        stage = 'closing-binding'
        verify_prepared(record)
        if artifact_hashes(locations['policy'])!=saved_files:
            raise ValueError('Merged policy bytes changed during reload')
        if (attempted_calls,completed_calls,attempted_positions,completed_positions)!=(24,24,807,807):
            raise ValueError('Actual verification cost differs from the fixed complete panel')
        retain(evidence/'verification.json',dict(status='passed',preparation_sha256=record['preparation_sha256'],
            atol=ATOL,rtol=RTOL,comparisons=comparisons,reload_comparisons=reload_comparisons,
            exact_reloaded_logits=True,interface_compatible=True,merged_files=saved_files,
            prompt_tokens=269,forward_calls=completed_calls,full_prefix_positions=completed_positions,
            attempted_forward_calls=attempted_calls,attempted_full_prefix_positions=attempted_positions,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
            scope='CPU FP32 merge/storage; actual CUDA FP32 SDPA same-prefix verification; no BF16/capability claim'))
        event(dict(event='child-completed',seconds=time.monotonic()-started))
        print(json.dumps(dict(status='passed',seconds=time.monotonic()-started)),flush=True)
        return 0
    except BaseException as error:
        retain(evidence/'child-failure.json',dict(type=type(error).__name__,message=str(error),stage=stage,
            attempted_forward_calls=attempted_calls,completed_forward_calls=completed_calls,
            attempted_full_prefix_positions=attempted_positions,completed_full_prefix_positions=completed_positions,
            seconds=time.monotonic()-started,scope='Actual partial attempt retained; no retry or relaxed tolerance'))
        raise
    finally:
        if trace is not None: trace.close()


def execute(record, declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:
        raise ValueError('Explicit retained goal-authorized scope required')
    evidence = Path(record['evidence'])
    supervision = None
    try:
        locations = verify_prepared(record)
        _new_target(str(locations['output']))
        retain(evidence/'launch.json',dict(command=record['command'],operator_declaration=declaration,
            preparation_sha256=record['preparation_sha256']))
        supervision = _supervise(record['command'],evidence/'supervision',native=True,seconds=SECONDS,
            probe=_native_probe,operator_declaration=declaration)
        retain(evidence/'returned-supervision.json',supervision)
        verify_prepared(record)
        retain(evidence/'closing-bindings.json',dict(status='unchanged',source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'],parent_binding=record['parent_binding']))
        if supervision['status']!='completed' or supervision['actual_exit_code']!=0:
            raise RuntimeError('Actual owned merge/reload failed; partial events retained')
        verification = json.loads((evidence/'verification.json').read_text())
        merged_files = artifact_hashes(locations['policy'])
        checks = dict(actual_child_completed=supervision['actual_exit_code']==0,
            fixed_complete_panel=verification['forward_calls']==24 and verification['full_prefix_positions']==807,
            merge_within_predeclared_tolerance=all(row['passed'] for row in verification['comparisons']),
            exact_reloaded_logits=verification['exact_reloaded_logits'] is True,
            full_interface_preserved=verification['interface_compatible'] is True,
            exported_bytes_unchanged=merged_files==verification['merged_files'],
            standard_genealogy=json.loads((locations['policy']/'course-genealogy.json').read_text())==genealogy(record))
        accepted = dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
            parent_binding=record['parent_binding'],verification=verification,
            exported_policy=dict(path=str(locations['policy']),files=merged_files,genealogy=genealogy(record)),
            actual_exit_code=supervision['actual_exit_code'],child_seconds=supervision['child_seconds'],
            scope='Accepted actual final400 CPU-FP32 merged full artifact/reload; publication generation remains separate')
        retain(evidence/'acceptance.json',accepted)
        if accepted['status']!='passed':
            raise RuntimeError('Merge/reload acceptance failed')
        print(json.dumps(dict(status='passed',evidence=str(evidence),policy=str(locations['policy']))),flush=True)
        return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_supervision_retained=supervision is not None,scope='Attempt retained; no implicit retry'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='run-01')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--operator-declaration')
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if args.execute or args.operator_declaration:
            parser.error('Closed child cannot accept execution or declaration overrides')
        return child(args.run_id)
    if args.execute and (not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096):
        parser.error('Explicit recorded goal-authorized declaration required')
    record = prepare(args.run_id)
    if args.execute: return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',evidence=record['evidence'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

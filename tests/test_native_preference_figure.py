"""Explicit AUTHORED CPU FIXTURE controls, not pretrained/actual comparison data."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preference_figure',ROOT/'scripts/plot_native_preference_comparison.py')
fig=importlib.util.module_from_spec(spec);spec.loader.exec_module(fig)
from dongxi_llms.reasoning_evaluation import freeze_contract,grade_response,SCHEMA_VERSION


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values))


def binding(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(root):
    """Every receipt, PID, score and inventory below is inert authored schema data."""
    evidence=root/'experiments/reports'/fig.STEM;output=root/'outputs'/fig.STEM;evidence.mkdir(parents=True)
    sources={}
    for name in (*fig.GENERATION_SOURCES,'scripts/run_native_preference_evaluation.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'# AUTHORED CPU FIXTURE source, not a native producer\n');sources[name]=binding(path)
    interpreter=root/'authored-python';interpreter.write_bytes(b'AUTHORED inert executable-byte receipt\n')
    lock=root/'authored.lock';lock.write_bytes(b'AUTHORED lock\n')
    validation=[dict(id=f'validation-{i}',chosen='answer',rejected='wrong',prompt=[dict(role='user',content='Authored prompt')]) for i in range(4)]
    validation_path=root/'fixtures/chapter11/validation.jsonl';rows(validation_path,validation)
    groups={row['id']:f'scenario-{i}' for i,row in enumerate(validation)}
    protocol=root/'fixtures/matched-chosen-sft/protocol.json';save(protocol,dict(groups=groups))
    template=root/'experiments/data/instruction_interface_v1.jinja';template.parent.mkdir(parents=True,exist_ok=True)
    template.write_text('AUTHORED CPU FIXTURE template, not a trained tokenizer')
    inputs=dict(interpreter=binding(interpreter),environment_lock=binding(lock),validation=binding(validation_path),
        protocol=binding(protocol),template=binding(template))
    fixed={name:root/'fixtures/chapter11'/(name+'.jsonl') for name in ('train','evaluation')}
    fixed.update(math_items=root/'fixtures/reasoning-controls/math_items.json',
        assistant_card=root/'experiments/data/instruction-interface-v1-data-card.json')
    fixed.update({f'assistant_{name}':root/'outputs/course-sft-interface-v1'/(name+'.jsonl') for name in ('train','dev','test')})
    for name,path in fixed.items():
        save(path,dict(scope='AUTHORED CPU FIXTURE fixed-role bytes, not corpus'))
        inputs[name]=binding(path)
    parent_path=root/'outputs/native-sft-full-pilot400-20261005-run-01/policy'
    parent_files={'model.safetensors':'a'*64,'tokenizer.json':'b'*64}
    full_receipt=root/'experiments/reports/native-sft-full-pilot400-20261005-run-01/acceptance.json'
    parent_genealogy=dict(kind='full-HF-model',scope='AUTHORED CPU FIXTURE genealogy')
    save(parent_path/'course-genealogy.json',parent_genealogy)
    full_checks={key:True for key in ('completed400','exactly400_metrics','original_encoded_geometry','no_open_or_failed_work',
        'fresh_pinned_base','five_completed_snapshots','export_kind','loadable_full_policy')}
    save(full_receipt,dict(status='passed',checks=full_checks,result={'status':'completed','updates':400},actual_exit_code=0,
        exported_policy=dict(path=str(parent_path),files=parent_files,genealogy=parent_genealogy)))
    parent=dict(path=str(parent_path),files=parent_files,genealogy=parent_genealogy,acceptance=binding(full_receipt),scope='AUTHORED CPU FIXTURE parent')
    producers={'unchanged':dict(path=str(parent_path),files=parent_files,parent=parent)}
    for role,kind,sha in (('chosen100','chosen','c'),('dpo100','dpo','d')):
        directory=root/'experiments/reports'/f'native-{kind}-pilot-20261005-run-01'
        policy=root/'outputs'/f'native-{kind}-pilot-20261005-run-01-pilot/policy'
        result=dict(completed_recovery={'completed_updates':100},reference_has_gradients=False)
        genealogy=dict(parent_checkpoint=str(parent_path),parent_checkpoint_sha256=parent_files,completed_recovery=result['completed_recovery'])
        save(policy/'course-genealogy.json',genealogy);files={'model.safetensors':sha*64,'tokenizer.json':'b'*64,
            'course-genealogy.json':binding(policy/'course-genealogy.json')['sha256']}
        prepared=dict(mode='pilot',run_id='run-01',limits={'updates':100},parent_binding=parent)
        checks={key:True for key in ('all_actual_children_completed','completed100','exactly100_metrics','clean_journals')}
        if kind=='chosen':checks['unchanged_reference']=True
        accepted=dict(status='passed',checks=checks,parent_binding=parent,
            invocations=[dict(role='pilot',result=dict(status='completed',actual_exit_code=0))],
            exports={'pilot':dict(path=str(policy),files=files)},results={'pilot':result})
        save(directory/'acceptance.json',accepted);save(directory/'preparation.json',prepared)
        save(policy.parent/'result.json',result);rows(policy.parent/'metrics.jsonl',[{'update':i} for i in range(1,101)])
        producers[role]=dict(path=str(policy),files=files,parent=parent,genealogy=genealogy,
            acceptance=binding(directory/'acceptance.json'),preparation=binding(directory/'preparation.json'),
            actual_result=binding(policy.parent/'result.json'),actual_metrics=binding(policy.parent/'metrics.jsonl'))
    interface={'interface_sha256':'f'*64}
    settings=dict(template_id='original-selected-full400-preference-retention',thinking_mode='template-default',
        decoding=dict(mode='greedy',seed=1010,temperature=1.,top_k=None,top_p=1.),
        stopping=dict(eos_token_ids=[9],turn_stop_token_ids=[],pad_token_id=9),max_new_tokens=64,
        generation=dict(input_mode='chat',template=template.read_text(),context_window=512,samples=1,device='cuda',
            dtype='bfloat16',add_special_tokens=False,max_run_seconds=900,scoring_text='decode_without_terminal_stop',interface=interface))
    commands={role:[str(interpreter),str(root/'scripts/run_native_preference_evaluation.py'),'--run-id','run-01','--child',role] for role in fig.ROLES}
    contracts={};items_by_panel={}
    for panel,count in fig.COUNTS.items():
        items=[dict(id=f'{panel}-{i}',source_group=f'{panel}-group-{i//3}',task='arithmetic' if panel=='reasoning' else 'copy',
            split='publication',prompt='AUTHORED CPU FIXTURE prompt',kind='math' if panel=='reasoning' else 'text',
            reference='3' if panel=='reasoning' else 'answer',extraction='whole',format_policy='any') for i in range(count)]
        contract=freeze_contract(items,settings);contracts[panel]=contract;items_by_panel[panel]=items
        save(evidence/(panel+'-items.json'),items);save(evidence/(panel+'-contract.json'),contract)
        for name in ('items','contract'):inputs[panel+'/'+name]=binding(evidence/(panel+'-'+name+'.json'))
    encoded=[[[[1,2,3],[False,True,True]],[[1,2,3],[False,True,True]]] for _ in range(4)]
    prepared=dict(schema='dongxi-fixed-native-preference-evaluation-v1',run_id='run-01',
        locations=dict(evidence=str(evidence),output=str(output)),source_bindings=sources,input_bindings=inputs,
        producers=producers,interfaces={role:interface for role in fig.ROLES},commands=commands,
        limits=dict(seconds_each=900,reserve_bytes=25*fig.GIB,context=512,output_cap=64,validation_pairs=4,
            validation_policy_calls=8,validation_reference_calls=8,validation_positions_each=16,
            planned_responses_each=144,maximum_emitted_tokens_each=144*64,
            retained_reader_bytes=dict(response_or_pair_jsonl=fig.JSON_BYTES,event_jsonl=fig.EVENT_BYTES,input_identity=fig.JSON_BYTES)),
        validation_rows=validation,encoded_validation=encoded,scenario_groups=groups,
        contracts={panel:contract['identity'] for panel,contract in contracts.items()},precision=fig.PRECISION,
        scope='AUTHORED CPU FIXTURE — no actual model results or GPU work')
    prepared['preparation_sha256']=fig.canonical_hash(prepared)
    save(evidence/'preparation.json',prepared);checks={'no_unreadable_jsonl':True};invocations=[]
    comparison=dict(validation_likelihood_and_reference_margin={},latest_pair_costs={},independent_panels={},unreadable_jsonl=[])
    for ordinal,role in enumerate(fig.ROLES):
        terminal=dict(schema='dongxi-native-profile-watchdog-result-v1',status='completed',actual_exit_code=0,
            child_pid=100+ordinal,actual_native_profile_executed=True,stop_reason=None,failure=None,journal_error=None,
            child_command=commands[role],cleanup_errors=[],limits=dict(external_seconds=900,reserve_bytes=25*fig.GIB),
            minimum_sampled_available_bytes=26*fig.GIB,helper_cleanup=[])
        returned=deepcopy(terminal);returned.update(final_record_retained=True,
            helper_cleanup=[dict(scope='AUTHORED later logger ACK/cleanup')],queue_feeder_shutdown=[])
        save(evidence/(role+'-returned-supervision.json'),returned);save(evidence/(role+'-supervision/result.json'),terminal)
        save(evidence/(role+'-launch.json'),dict(argv=commands[role],preparation_sha256=prepared['preparation_sha256']))
        invocations.append(dict(producer=role,result=returned));pair=[]
        chosen,rejected,margin=((-4.,-8.,0.),(-6.,-8.,-2.),(-3.,-9.,2.))[ordinal]
        for row in validation:pair.append(dict(id=row['id'],source_group=groups[row['id']],chosen_logp=chosen,
            rejected_logp=rejected,reference_relative_margin=.1*margin,reference_relative_logp_margin=margin,
            beta=.1,loss=.7,chosen_targets=2,rejected_targets=2))
        rows(output/role/'pair-scores.jsonl',pair);comparison['validation_likelihood_and_reference_margin'][role]=pair
        cost={network:dict(attempted_calls=8,forward_calls=8,attempted_positions=16,forward_positions=16) for network in ('policy','reference')}
        save(output/role/'pair-summary.json',dict(status='completed',pairs=4,cost=cost,reference_has_gradients=False))
        rows(output/role/'pair-cost-events.jsonl',[dict(cost=cost,wall_seconds=.01)])
        comparison['latest_pair_costs'][role]=dict(cost=cost,wall_seconds=.01)
        save(output/role/'child-result.json',dict(status='completed',producer=role,planned_responses=144))
        checks.update({role+'/'+key:True for key in ('all_validation_pairs','frozen_validation_groups','pair_reference_no_grad','pair_completed_and_costed','child_complete')})
        for panel,items in items_by_panel.items():
            contract=contracts[panel];folder=output/role/panel;producer=producers[role]
            tokenizer={'tokenizer.json':'b'*64}
            input_map={str(evidence.relative_to(root)/(panel+'-'+name+'.json')):inputs[panel+'/'+name]['sha256'] for name in ('items','contract')}
            identity=dict(schema_version=1,checkpoint_path=producer['path'],checkpoint_files=producer['files'],
                selected_tokenizer_path=producer['path'],selected_tokenizer_files=tokenizer,input_sha256=input_map,
                config=settings,checkpoint_interface=interface,source_sha256={name:sources[name]['sha256'] for name in fig.GENERATION_SOURCES},
                device=dict(mode='cuda'),command=commands[role],environment=dict(interpreter=str(interpreter),
                    environment_lock=dict(status='hashed',path=str(lock),sha256=inputs['environment_lock']['sha256'])))
            identity['identity_sha256']=fig.canonical_hash(identity);save(folder/'input-identity.json',identity)
            checkpoint='local-hf-sha256:'+fig.canonical_hash(dict(files=producer['files'],tokenizer=tokenizer,interface=interface['interface_sha256']))
            records=[];projected={}
            for i,item in enumerate(items):
                good=role=='chosen100' or role=='dpo100' and i%2==0;answer=item['reference'] if good else '4' if panel=='reasoning' else 'wrong'
                tokens=[9] if good else [1]*64
                cost=dict(wall_seconds=.01,generation_tokens=len(tokens),scoring_tokens=0,model_forward_tokens=len(tokens),
                    attempted_forward_tokens=len(tokens),forward_calls=len(tokens),attempted_forward_calls=len(tokens))
                row=dict(schema_version=SCHEMA_VERSION,adapter_version='local-hf-response-v1',contract_id=contract['identity'],
                    checkpoint_id=checkpoint,sample_id='authored-sample-'+str(i),sample_index=0,item_id=item['id'],
                    source_group=item['source_group'],task=item['task'],split=item['split'],raw_response=answer,response_text=answer,
                    raw_response_sha256=hashlib.sha256(answer.encode()).hexdigest(),response_text_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                    token_ids=tokens,prompt_tokens=1,prompt_token_ids=[1],generated_tokens=len(tokens),stop_reason='eos' if good else 'max_tokens',
                    truncated=not good,error=None,cost=cost,stop_token_id=9 if good else None,checkpoint_interface_sha256=interface['interface_sha256'],
                    settings_sha256=fig.canonical_hash(settings),input_identity_sha256=identity['identity_sha256'],input_sha256=input_map,
                    selected_behavior_log_probabilities=[-1.]*len(tokens),selected_raw_log_probabilities=[-1.]*len(tokens),retained_support_sizes=[2]*len(tokens))
                records.append(row);grade=grade_response(item,row)
                projected[item['id']]=dict(item_id=item['id'],source_group=item['source_group'],task=item['task'],correct=good,
                    status=grade['status'],natural_stop=good,format_valid=grade['format_valid'],truncated=not good,cost=cost)
            rows(folder/'responses.jsonl',records);rows(folder/'events.jsonl',[dict(stage='AUTHORED CPU FIXTURE',no_actual_model=True)])
            save(folder/'summary.json',dict(status='completed',record_count=len(items),contract_id=contract['identity'],
                checkpoint_id=checkpoint,identity_sha256=identity['identity_sha256'],inputs_unchanged=True))
            known=sum(row['correct'] for row in projected.values())
            observed=comparison['independent_panels'].setdefault(panel,dict(rows={},summaries={},incomplete_attempts={}))
            observed['rows'][role]=projected;observed['summaries'][role]=dict(planned=len(items),completed=len(items),missing=0,
                correct=known,natural_stops=known,truncated=len(items)-known);observed['incomplete_attempts'][role]=[]
            checks.update({role+'/'+panel+'/'+key:True for key in ('coverage','no_response_errors','partial_identity_join','generation_identity_complete')})
    comparison['checks']=checks;save(evidence/'comparison.json',comparison)
    save(evidence/'acceptance.json',dict(status='passed',checks=checks,invocations=invocations,producers=producers,precision=fig.PRECISION))
    return evidence,output,prepared


class NativePreferenceFigure(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dongxi-preference-figure-');self.root=Path(self.temp.name)
        self.stack=ExitStack();self.stack.enter_context(patch.object(fig,'ROOT',self.root))
        self.stack.enter_context(patch.object(fig,'PYTHON',str(self.root/'authored-python')))
        self.stack.enter_context(patch.object(fig,'LOCK',str(self.root/'authored.lock')))
        self.evidence,self.output,self.prepared=fixture(self.root)
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def collect(self):return fig.collect(fig.Inputs())
    def alter(self,path,change):
        value=json.loads(path.read_text());change(value);save(path,value)
    def reprepare(self,change):
        value=json.loads((self.evidence/'preparation.json').read_text());change(value)
        value['preparation_sha256']=fig.canonical_hash({k:v for k,v in value.items() if k!='preparation_sha256'})
        save(self.evidence/'preparation.json',value)
    def test_import_is_stdlib_and_bounded_grader_only(self):
        code="import sys,runpy;sys.modules['torch']=None;sys.modules['transformers']=None;runpy.run_path(sys.argv[1],run_name='not_main')"
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts/plot_native_preference_comparison.py')],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_exact_three_arms_negative_margin_and_separate_panels(self):
        data=self.collect();self.assertEqual(set(data['likelihood']),set(fig.ROLES))
        self.assertEqual(data['likelihood']['chosen100']['reference_relative_logp_margin'],-2.)
        self.assertEqual(data['likelihood']['dpo100']['reference_relative_logp_margin'],2.)
        self.assertEqual([data['retention']['assistant'][role]['correct'] for role in fig.ROLES],[0,120,60])
        self.assertEqual([data['retention']['reasoning'][role]['correct'] for role in fig.ROLES],[0,20,10])
        self.assertIn('not original32/128',data['scope']);self.assertIn('AUTHORED CPU FIXTURE',data['producer_declared_scope'])
        self.assertEqual(data['precision'],fig.PRECISION)
    def test_actual_schema_genealogy_and_post_write_supervision_fields_are_preserved(self):
        accepted=json.loads((self.root/'experiments/reports/native-sft-full-pilot400-20261005-run-01/acceptance.json').read_text())
        self.assertIn('genealogy',accepted['exported_policy'])
        terminal=json.loads((self.evidence/'unchanged-supervision/result.json').read_text())
        returned=json.loads((self.evidence/'unchanged-returned-supervision.json').read_text())
        self.assertNotEqual(terminal,returned);self.assertNotIn('final_record_retained',terminal)
        self.assertTrue(returned['final_record_retained']);self.collect()
    def test_likelihood_and_generation_precision_attention_labels_are_separate(self):
        labels=self.collect()['figure_label_scopes']
        self.assertIn('FP32 policy/reference weights',labels['likelihood'])
        self.assertIn('BF16 CUDA autocast',labels['likelihood']);self.assertIn('backend not forced',labels['likelihood'])
        self.assertIn('BF16-loaded CUDA policies',labels['retention']);self.assertIn('eager attention',labels['retention'])
        self.assertNotIn('FP32',labels['retention']);self.assertNotIn('autocast',labels['retention'])
        self.assertNotIn('MATH',str(labels))
    def test_actual_acceptance_failed_or_missing_checks_refuses(self):
        self.alter(self.evidence/'acceptance.json',lambda d:d.update(status='failed'))
        with self.assertRaisesRegex(ValueError,'passed'):self.collect()
    def test_missing_required_actual_check_is_not_an_optimistic_pass(self):
        for name in ('acceptance.json','comparison.json'):
            self.alter(self.evidence/name,lambda d:d['checks'].pop('dpo100/assistant/generation_identity_complete'))
        with self.assertRaisesRegex(ValueError,'original common-comparison checks'):self.collect()
    def test_missing_or_swapped_actual_arm_refuses(self):
        self.reprepare(lambda d:d['producers']['dpo100'].update(path=d['producers']['chosen100']['path']))
        self.alter(self.evidence/'acceptance.json',lambda d:d.update(producers=json.loads((self.evidence/'preparation.json').read_text())['producers']))
        with self.assertRaisesRegex(ValueError,'final100 export'):self.collect()
    def test_unaccepted_or_incomplete100_metadata_refuses(self):
        path=Path(self.prepared['producers']['dpo100']['actual_metrics']['path'])
        rows(path,[{'update':100}])
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.collect()
    def test_optimistic100_checks_do_not_replace_actual_native_acceptance_checks(self):
        prepared=deepcopy(self.prepared);role=prepared['producers']['dpo100'];path=Path(role['acceptance']['path'])
        self.alter(path,lambda d:d.update(checks={'authored_optimistic':True}));role['acceptance']=binding(path)
        with self.assertRaisesRegex(ValueError,'accepted complete100'):fig.producer_metadata(prepared,fig.Inputs())
    def test_failed_native_exit_and_returned_receipt_mismatch_refuse(self):
        self.alter(self.evidence/'dpo100-returned-supervision.json',lambda d:d.update(actual_exit_code=1))
        with self.assertRaisesRegex(ValueError,'exit0'):self.collect()
    def test_changed_terminal_exit_cannot_hide_behind_successful_returned_receipt(self):
        self.alter(self.evidence/'dpo100-supervision/result.json',lambda d:d.update(actual_exit_code=1))
        with self.assertRaisesRegex(ValueError,'exit0'):self.collect()
    def test_missing_actual_final_writer_ack_refuses(self):
        self.alter(self.evidence/'dpo100-returned-supervision.json',lambda d:d.update(final_record_retained=False))
        value=json.loads((self.evidence/'dpo100-returned-supervision.json').read_text())
        self.alter(self.evidence/'acceptance.json',lambda d:d['invocations'][2].update(result=value))
        with self.assertRaisesRegex(ValueError,'exit0'):self.collect()
    def test_missing_or_incomplete_raw_pair_population_refuses(self):
        (self.output/'chosen100/pair-scores.jsonl').unlink()
        with self.assertRaises(FileNotFoundError):self.collect()
    def test_pair_row_tamper_cannot_use_saved_comparison(self):
        path=self.output/'dpo100/pair-scores.jsonl';values=fig.Inputs().rows(path);values[0]['chosen_logp']=-100.;rows(path,values)
        with self.assertRaisesRegex(ValueError,'raw4'):self.collect()
    def test_ref_relative_margin_is_unscaled_and_matched_to_frozen_reference(self):
        for name in ('comparison.json',):
            self.alter(self.evidence/name,lambda d:d['validation_likelihood_and_reference_margin']['chosen100'][0].update(reference_relative_logp_margin=-20.))
        with self.assertRaisesRegex(ValueError,'raw4'):self.collect()
    def test_matching_copied_wrong_margin_still_refuses_common_reference_join(self):
        path=self.output/'chosen100/pair-scores.jsonl';values=fig.Inputs().rows(path)
        values[0].update(reference_relative_logp_margin=-20.,reference_relative_margin=-2.);rows(path,values)
        self.alter(self.evidence/'comparison.json',lambda d:d['validation_likelihood_and_reference_margin'].update(chosen100=values))
        with self.assertRaisesRegex(ValueError,'common frozen full400 reference'):self.collect()
    def test_missing_response_or_raw_contract_mismatch_refuses(self):
        path=self.output/'dpo100/reasoning/responses.jsonl';values=fig.Inputs().rows(path);values.pop();rows(path,values)
        with self.assertRaisesRegex(ValueError,'population'):self.collect()
    def test_different_response_contract_or_policy_cannot_be_copied(self):
        path=self.output/'dpo100/assistant/responses.jsonl';values=fig.Inputs().rows(path);values[0]['contract_id']='e'*64;rows(path,values)
        with self.assertRaisesRegex(ValueError,'contract identity'):self.collect()
    def test_wrong_identity_receipt_and_summary_are_not_complete_actual_evidence(self):
        self.alter(self.output/'chosen100/assistant/input-identity.json',lambda d:d.update(checkpoint_path='/authored/other-policy'))
        with self.assertRaisesRegex(ValueError,'identity/summary'):self.collect()
    def test_wrong_displayed_retention_count_refuses(self):
        self.alter(self.evidence/'comparison.json',lambda d:d['independent_panels']['assistant']['summaries']['unchanged'].update(correct=120))
        with self.assertRaisesRegex(ValueError,'retention counts'):self.collect()
    def test_source_and_original_input_drift_refuse(self):
        source=self.root/'scripts/run_native_preference_evaluation.py';source.write_bytes(b'# authored changed source\n')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.collect()
    def test_original_template_input_drift_refuses(self):
        Path(self.prepared['input_bindings']['template']['path']).write_text('Changed authored template')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.collect()
    def test_swapped_original_input_role_refuses_before_render(self):
        other=self.root/'authored-other-template';other.write_bytes(Path(self.prepared['input_bindings']['template']['path']).read_bytes())
        self.reprepare(lambda d:d['input_bindings'].update(template=binding(other)))
        with self.assertRaisesRegex(ValueError,'fixed data/template'):self.collect()
    def test_different_frozen_arm_interface_refuses(self):
        self.reprepare(lambda d:d['interfaces'].update(chosen100={'interface_sha256':'e'*64}))
        with self.assertRaisesRegex(ValueError,'Same frozen selected-parent interface'):self.collect()
    def test_precision_mislabel_and_new_output_cap_refuse(self):
        self.reprepare(lambda d:d.update(precision='BF16 policy/reference weights'))
        self.alter(self.evidence/'acceptance.json',lambda d:d.update(precision='BF16 policy/reference weights'))
        with self.assertRaisesRegex(ValueError,'precision'):self.collect()
    def test_new_output_cap_cannot_relabel_original_common_contract(self):
        self.reprepare(lambda d:d['limits'].update(output_cap=128))
        with self.assertRaisesRegex(ValueError,'panel/caps'):self.collect()
    def test_duplicate_nonfinite_partial_tail_and_symlink_refuse(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}'):
            with self.assertRaises(ValueError):fig.parse(raw)
        path=self.output/'unchanged/pair-scores.jsonl';path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaisesRegex(ValueError,'Complete nonempty'):fig.Inputs().rows(path)
        link=self.root/'authored-link';link.symlink_to(path)
        with self.assertRaises(OSError):fig.regular(link)
    def test_exclusive_render_has_two_cpu_pngs_and_before_after_receipt(self):
        target=self.root/'experiments/reports/authored-figures'
        with patch.dict(os.environ,{'MPLCONFIGDIR':str(self.root/'mpl-cache'),'CUDA_VISIBLE_DEVICES':''}),redirect_stdout(io.StringIO()):
            self.assertEqual(fig.main(['--output',str(target)]),0)
        receipt=json.loads((target/'inputs.json').read_text());acceptance=json.loads((target/'acceptance.json').read_text())
        self.assertEqual(acceptance['status'],'passed');self.assertEqual(receipt['input_bindings_before'],receipt['input_bindings_after'])
        self.assertIn('AUTHORED CPU FIXTURE',receipt['displayed_data']['producer_declared_scope'])
        for name in ('likelihood-and-margin.png','separate-retention-panels.png'):
            self.assertTrue((target/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertEqual(receipt['figures'][name]['sha256'],binding(target/name)['sha256'])
        before=(target/'inputs.json').read_bytes()
        with self.assertRaises(FileExistsError):fig.main(['--output',str(target)])
        self.assertEqual((target/'inputs.json').read_bytes(),before)
    def test_input_mutation_during_render_retains_failure_not_acceptance(self):
        target=self.root/'experiments/reports/authored-failure'
        def corrupt(data,directory):
            (directory/'likelihood-and-margin.png').write_bytes(b'Explicit authored failed image prefix')
            (self.evidence/'comparison.json').write_bytes(b'Explicit authored changed JSON bytes')
        with patch.object(fig,'render',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'changed during rendering'):fig.main(['--output',str(target)])
        self.assertTrue((target/'failure.json').exists());self.assertFalse((target/'acceptance.json').exists())
        self.assertFalse((target/'inputs.json').exists());self.assertTrue((target/'likelihood-and-margin.png').exists())


if __name__=='__main__':unittest.main()

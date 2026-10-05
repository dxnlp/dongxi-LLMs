"""Authored CPU controls for saved-record comparison; no model evaluation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_assistant_comparison',ROOT/'scripts/compare_native_assistant_evaluations.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
from dongxi_llms.reasoning_evaluation import freeze_contract
from dongxi_llms.run_identity import SPECIAL_IDS


def contract(items,source='authored-control'):
    template=(ROOT/'experiments/data/instruction_interface_v1.jinja').read_text()
    interface=dict(schema_version=1,tokenizer=dict(vocab_sha256='a'*64,encoding_sha256='b'*64,
        special_ids={key:None for key in SPECIAL_IDS},vocab_size=151669,max_token_id=151668,wrapper_settings={}),
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),generation_stop_ids=[151643,151645],
        source=dict(tokenizer_id=source,tokenizer_revision='d'*40),implementation_class='AuthoredControlTokenizer')
    interface['interface_sha256']=adapter.canonical_hash({key:interface[key] for key in
        ('schema_version','tokenizer','template_sha256','generation_stop_ids')})
    settings=dict(template_id='original-instruction-interface-v1-publication',thinking_mode='template-default',
        decoding=dict(mode='greedy',seed=1010,temperature=1,top_k=None,top_p=1),
        stopping=dict(eos_token_ids=[151643],turn_stop_token_ids=[151645],pad_token_id=151643),max_new_tokens=64,
        generation=dict(input_mode='chat',template=template,context_window=512,samples=1,device='cuda',dtype='bfloat16',
            add_special_tokens=False,max_run_seconds=900,scoring_text='decode_without_terminal_stop',interface=interface))
    return freeze_contract(items,settings)


def record(item,frozen,*,text=None,stop='turn_stop',checkpoint='authored-checkpoint'):
    text=item['reference'] if text is None else text
    token_ids=[7,151645] if stop=='turn_stop' else [7]*64 if stop=='max_tokens' else [7]
    raw=text+'<|im_end|>' if stop=='turn_stop' else text
    seed=adapter.attempt_seed(1010,item['id'],0)
    return dict(schema_version=frozen['schema_version'],adapter_version='local-hf-response-v1',
        contract_id=frozen['identity'],checkpoint_id=checkpoint,sample_id=f'sample-0-seed-{seed}',
        item_id=item['id'],source_group=item['source_group'],task=item['task'],split=item['split'],
        raw_response=raw,response_text=text,raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        response_text_sha256=hashlib.sha256(text.encode()).hexdigest(),token_ids=token_ids,prompt_tokens=3,
        prompt_token_ids=[1,2,3],generated_tokens=len(token_ids),stop_reason=stop,
        truncated=stop=='max_tokens',error='authored forward failure' if stop=='error' else None,
        attempt_seed=seed,sample_index=0,decoding=deepcopy(frozen['settings']['decoding']),
        checkpoint_interface_sha256=frozen['settings']['generation']['interface']['interface_sha256'],
        input_identity_sha256='c'*64,settings_sha256=adapter.canonical_hash(frozen['settings']),
        stop_token_id=151645 if stop=='turn_stop' else None,
        selected_behavior_log_probabilities=[0.]*len(token_ids),selected_raw_log_probabilities=[-.1]*len(token_ids),
        retained_support_sizes=[1]*len(token_ids),cost=dict(wall_seconds=.1,generation_tokens=len(token_ids),
            scoring_tokens=0,model_forward_tokens=3*len(token_ids),attempted_forward_tokens=3*len(token_ids),
            forward_calls=len(token_ids),attempted_forward_calls=len(token_ids)))


class SavedAssistantComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=adapter.publication.publication_items()

    def panel(self,label='base',source='authored-control',text=None):
        frozen=contract(self.items,source)
        records=[record(item,frozen,text=text,checkpoint=label) for item in self.items]
        return adapter.analyze_panel(label,self.items,frozen,records)

    def test_import_requires_no_torch_or_transformers(self):
        code="import sys,runpy;sys.modules['torch']=None;sys.modules['transformers']=None;runpy.run_path(sys.argv[1],run_name='not_main')"
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts/compare_native_assistant_evaluations.py')],
            capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_original120_40_is_not_replaced_by_diagnostics(self):
        adapter.validate_original_items(self.items,self.items)
        for changed in (self.items[:20],list(reversed(self.items))):
            with self.assertRaises(ValueError):adapter.validate_original_items(changed,self.items)

    def test_strict_case_sensitive_and_generic_parser_are_separate(self):
        frozen=contract(self.items);item=self.items[0]
        actual=record(item,frozen,text=item['reference'].upper())
        panel=adapter.analyze_panel('base',self.items,frozen,[actual])
        self.assertFalse(panel['strict']['rows'][0]['exact'])
        self.assertTrue(panel['generic']['replay']['rows'][0]['task_success'])
        self.assertEqual(panel['raw_records'],[actual])
        self.assertEqual(actual['token_ids'][-1],151645)

    def test_stop_cap_error_and_partial_costs_are_retained_per_family(self):
        frozen=contract(self.items)
        rows=[record(self.items[0],frozen),record(self.items[1],frozen,stop='max_tokens'),
            record(self.items[2],frozen,stop='error')]
        rows[-1]['cost']['wall_seconds']=None
        rows[-1]['cost']['attempted_forward_tokens']+=3
        panel=adapter.analyze_panel('base',self.items,frozen,rows)
        summary=panel['strict']['summary']['overall']
        self.assertEqual((summary['attempts'],summary['errors'],summary['natural_stops'],summary['caps']),(3,1,1,1))
        self.assertEqual(summary['costs']['wall_seconds'],dict(known_sum=.2,unknown_rows=1))
        self.assertEqual(panel['raw_records'],rows)
        self.assertEqual(sum(group['attempts'] for group in panel['strict']['summary']['by_task'].values()),3)
        self.assertFalse(panel['strict']['rows'][-1]['exact'])
        self.assertTrue(panel['strict']['rows'][1]['exact']) # equality is not a natural-stop claim

    def test_incomplete_panel_retains_missing_ids_without_interval(self):
        panels={label:self.panel(label) for label in adapter.LABELS}
        frozen=contract(self.items)
        panels['base']=adapter.analyze_panel('base',self.items,frozen,[record(self.items[0],frozen)])
        self.assertEqual(len(panels['base']['missing_item_ids']),119)
        with patch.object(adapter,'paired_group_bootstrap',return_value={}) as bootstrap:
            result=adapter.paired_comparisons(panels)
        self.assertEqual([row['status'] for row in result],['unaligned-no-interval','unaligned-no-interval','aligned-descriptive'])
        self.assertEqual(bootstrap.call_count,4)

    def test_physical_source_receipts_differ_without_faking_identity(self):
        panels={label:self.panel(label,source=label) for label in adapter.LABELS}
        originals={label:deepcopy(panel['raw_records']) for label,panel in panels.items()}
        self.assertEqual(len({panel['physical_contract_id'] for panel in panels.values()}),3)
        self.assertEqual(len({panel['logical_contract_sha256'] for panel in panels.values()}),1)
        with patch.object(adapter,'paired_group_bootstrap',return_value={}) as bootstrap:
            result=adapter.paired_comparisons(panels)
        self.assertEqual(bootstrap.call_count,12)
        for call in bootstrap.call_args_list:
            self.assertEqual(call.kwargs['draws'],2000);self.assertEqual(call.kwargs['seed'],1010)
            self.assertEqual(len(call.args[0]),120)
            self.assertEqual(len({row['source_group'] for row in call.args[0]}),40)
            self.assertTrue(all('physical_contract_id' in row for row in call.args[0]))
        self.assertTrue(all(row['status']=='aligned-descriptive' for row in result))
        for label in panels:self.assertEqual(panels[label]['raw_records'],originals[label])

    def test_actual_shared_bootstrap_is_fixed_descriptive_and_paired(self):
        panels={label:self.panel(label) for label in adapter.LABELS}
        panels['base']=self.panel('base',text='authored incorrect answer')
        result=adapter.paired_comparisons(panels)[0]['intervals']['strict']['correct']
        self.assertEqual((result['delta'],result['interval']),(1.,[1.,1.]))
        self.assertEqual((result['draws'],result['seed'],result['n_items_samples'],result['n_source_groups']),(2000,1010,120,40))

    def test_actual_prompt_token_mismatch_blocks_pairing(self):
        panels={label:self.panel(label) for label in adapter.LABELS}
        panel=panels['full400'];panel['encoded_contract']['prompt_token_ids'][self.items[0]['id']]=[4,2,3]
        panel['encoded_contract_sha256']=adapter.canonical_hash(panel['encoded_contract'])
        with self.assertRaisesRegex(ValueError,'prompt token'):adapter.paired_comparisons(panels)

    def test_generation_source_environment_mismatch_blocks_pairing(self):
        panels={label:self.panel(label) for label in adapter.LABELS}
        for panel in panels.values():panel['producer_execution_contract_sha256']='a'*64
        panels['lora400-fp32']['producer_execution_contract_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'source/environment'):adapter.paired_comparisons(panels)

    def test_forged_matching_model_event_and_rows_do_not_certify_selector(self):
        frozen=contract(self.items)
        expected=adapter.expected_checkpoint_identity({'model.safetensors':'a'*64},
            {'tokenizer.json':'b'*64},frozen['settings']['generation']['interface'])
        row=record(self.items[0],frozen,checkpoint='local-hf-sha256:'+'f'*64)
        loaded=[dict(checkpoint_id=row['checkpoint_id'],dtype='torch.bfloat16',device='cuda')]
        with self.assertRaisesRegex(ValueError,'accepted checkpoint identity'):
            adapter.validate_loaded_identity([row],loaded,expected)
        row['checkpoint_id']=expected;loaded[0]['checkpoint_id']=expected
        adapter.validate_loaded_identity([row],loaded,expected)

    def test_precision_or_stop_or_template_correction_is_not_silent(self):
        frozen=contract(self.items)
        for field,value in (('dtype','float32'),('template','different'),('context_window',1024)):
            changed=deepcopy(frozen['settings']);changed['generation'][field]=value
            with self.assertRaises(ValueError):adapter.logical_contract(freeze_contract(self.items,changed),self.items)
        changed=deepcopy(frozen['settings']);changed['stopping']['turn_stop_token_ids']=[151646]
        with self.assertRaises(ValueError):adapter.logical_contract(freeze_contract(self.items,changed),self.items)

    def test_unknown_nonfinite_duplicate_or_misaligned_records_are_not_dropped(self):
        frozen=contract(self.items);row=record(self.items[0],frozen)
        for key,value in (('source_group','other'),('attempt_seed',9),('sample_id','invented')):
            changed=deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):adapter.analyze_panel('base',self.items,frozen,[changed])
        changed=deepcopy(row);changed['cost']['wall_seconds']=float('nan')
        with self.assertRaises(ValueError):adapter.analyze_panel('base',self.items,frozen,[changed])
        with self.assertRaises(ValueError):adapter.analyze_panel('base',self.items,frozen,[row,row])

    def test_saved_json_has_no_duplicate_or_nonfinite_keys(self):
        for raw in (b'{"a":1,"a":2}',b'{"cost":NaN}'):
            with self.assertRaises(ValueError):adapter.parse_json(raw)

    def test_empty_and_incomplete_saved_journal_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'responses.jsonl';path.touch()
            rows,tail,binding=adapter.jsonl_input(path)
            self.assertEqual((rows,tail,binding['bytes']),([],b'',0))
            path.write_bytes(b'{"a":1}\n{"partial":')
            rows,tail,binding=adapter.jsonl_input(path)
            self.assertEqual(rows,[{'a':1}]);self.assertEqual(tail,b'{"partial":')
            self.assertEqual(binding['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            link=Path(tmp)/'link';link.symlink_to(path)
            with self.assertRaises(OSError):adapter.bytes_identity(link)

    def test_own_source_input_receipt_and_model_byte_drift_are_refused(self):
        with patch.object(adapter,'source_bindings',return_value={'source':'first'}):
            with self.assertRaisesRegex(ValueError,'source changed'):adapter.verify_bindings({'source':'second'},{})
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'receipt';path.write_bytes(b'first')
                expected=adapter.bytes_identity(path)[1];path.write_bytes(b'second')
                with self.assertRaisesRegex(ValueError,'receipt changed'):adapter.verify_bindings({'source':'first'},{str(path):expected})
            with patch.object(adapter,'artifact_hashes',return_value={'model.safetensors':'second'}):
                with self.assertRaisesRegex(ValueError,'model artifact'):adapter.verify_bindings({'source':'first'},
                    {'/authored/model':dict(kind='artifact-file-map',files={'model.safetensors':'first'})})

    def nll_fixture(self):
        card=json.loads((ROOT/'experiments/data/instruction-interface-v1-data-card.json').read_text())
        result=dict(updates=400,resume_checkpoint_update=0,initial_dev_nll=3.,final_dev_nll=2.,
            supervised_targets_this_invocation=10,cumulative_processed_positions=20,invocation_id='authored-training-control')
        values={}
        for mode in ('full','lora'):
            values[mode]=dict(result=deepcopy(result),acceptance=dict(status='passed',checks={'completed400':True},result=deepcopy(result)),
                config=dict(mode=mode,updates=400,dev=str(ROOT/'outputs/course-sft-interface-v1/dev.jsonl'),
                    dev_sha256=card['splits']['dev']['sha256'],
                    template_sha256=hashlib.sha256((ROOT/'experiments/data/instruction_interface_v1.jinja').read_bytes()).hexdigest(),
                    dtype='authored BF16 training control'))
        def read(path):
            path=Path(path)
            value=card if path.name.endswith('data-card.json') else values['lora' if 'native-sft-lora' in str(path) else 'full'][path.stem]
            return deepcopy(value),dict(path=str(path),bytes=1,sha256='d'*64)
        return values,read

    def test_development_nll_is_separate_original60_population_before_merge(self):
        _,read=self.nll_fixture()
        with patch.object(adapter,'json_input',side_effect=read):result=adapter.development_nll({})
        self.assertEqual(set(result['rows']),{'full','lora'})
        self.assertEqual(result['rows']['lora']['initial_to_final_delta'],-1.)
        self.assertIn('before FP32 merge',result['rows']['lora']['scope'])
        self.assertNotIn('interval',result)

    def test_nll_refuses_unaccepted_nonfinite_or_changed_population(self):
        for mutation in ('checks','initial_dev_nll','dev_sha256'):
            values,read=self.nll_fixture()
            if mutation=='checks':values['full']['acceptance']['checks']={}
            elif mutation=='dev_sha256':values['full']['config']['dev_sha256']='e'*64
            else:
                values['full']['result']['initial_dev_nll']=float('nan')
                values['full']['acceptance']['result']=deepcopy(values['full']['result'])
            with patch.object(adapter,'json_input',side_effect=read):
                with self.assertRaises(ValueError):adapter.development_nll({})

    def test_report_checks_before_and_after_and_retains_failed_closing(self):
        panels={label:self.panel(label) for label in adapter.LABELS}
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'experiments/reports';destination.mkdir(parents=True)
            with patch.object(adapter,'ROOT',Path(tmp)),patch.object(adapter,'_new_target'),\
                    patch.object(adapter,'source_bindings',return_value={'authored':'source'}),\
                    patch.object(adapter,'bytes_identity',side_effect=lambda path:(b'',dict(path=str(path),bytes=0,sha256='c'*64))),\
                    patch.object(adapter,'_digest',return_value={'authored':'executable'}),\
                    patch.object(adapter.publication,'publication_items',return_value=self.items),\
                    patch.object(adapter,'load_publication',side_effect=lambda label,*args:panels[label]),\
                    patch.object(adapter,'development_nll',return_value={'scope':'separate authored control'}),\
                    patch.object(adapter,'paired_comparisons',return_value=[]),\
                    patch.object(adapter,'environment_identity',return_value={'interpreter':'authored CPU control'}),\
                    patch.object(adapter,'verify_bindings',side_effect=[None,ValueError('closing drift')]) as verify:
                with self.assertRaisesRegex(ValueError,'closing drift'):adapter.run()
            self.assertEqual(verify.call_count,2)
            evidence=destination/('native-assistant-comparison-20261005-'+adapter.COMPARISON_RUN_ID)
            self.assertTrue((evidence/'comparison.json').is_file());self.assertTrue((evidence/'failure.json').is_file())
            self.assertFalse((evidence/'closing-bindings.json').exists())
            retained=json.loads((evidence/'comparison.json').read_text())
            self.assertEqual(retained['panels']['base']['raw_records'],panels['base']['raw_records'])

    def test_closed_selectors_output_and_cli_accept_no_arbitrary_paths(self):
        self.assertEqual(adapter.RUN_ID,'run-01')
        self.assertEqual(adapter.COMPARISON_RUN_ID,'run-02')
        self.assertTrue(str(adapter.report_paths('base')).endswith('base-run-01'))
        self.assertEqual(adapter.LABELS,('base','full400','lora400-fp32'))
        with self.assertRaises(ValueError):adapter.report_paths('latest-best')
        with self.assertRaises(SystemExit):adapter.main(['--output','/tmp/arbitrary'])
        with patch.object(adapter,'run',return_value=0) as run:
            self.assertEqual(adapter.main([]),0);run.assert_called_once_with()


if __name__=='__main__':unittest.main()

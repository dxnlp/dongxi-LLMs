"""CPU-only fixed final400 merge preparation, identity and comparison controls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_lora_merge',ROOT/'scripts/run_native_lora_merge.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def metadata():
    """Use actual retained metadata/bytes; no weight deserialization."""
    locations = adapter.paths('run-01')
    inputs = adapter.input_bindings(locations)
    import json
    values = {role:json.loads(Path(inputs[role]['path']).read_text()) for role in
        ('acceptance','result','config','genealogy','adapter_config','preparation','closing')}
    return locations,inputs,values,adapter.artifact_hashes(locations['adapter']),adapter.artifact_hashes(adapter.BASE)


class ClosedPreparation(unittest.TestCase):
    def test_exact_original_eight_prefixes(self):
        rows = adapter.original_prefix_rows(ROOT/'outputs/course-sft-interface-v1/dev.jsonl')
        self.assertEqual(tuple(row['id'] for row in rows),adapter.PREFIX_IDS)
        self.assertTrue(all(row['messages'][:-1][-1]['role']=='user' for row in rows))
        self.assertEqual(sum(adapter.PREFIX_LENGTHS),269)
        self.assertEqual(adapter.LIMITS['full_prefix_positions'],3*269)

    def test_closed_routing_and_no_arbitrary_child_argv(self):
        for value in ('../../outside','latest','run-1','run-001',None):
            with self.assertRaises(ValueError): adapter.paths(value)
        self.assertEqual(adapter.fixed_command('run-01'),[adapter.PYTHON,
            str(ROOT/'scripts/run_native_lora_merge.py'),'--run-id','run-01','--child'])
        locations = adapter.paths('run-01')
        self.assertEqual(locations['policy'],ROOT/'outputs/native-sft-lora-pilot400-merged-20261005-run-01/policy')
        self.assertNotEqual(locations['adapter'],locations['policy'])

    def test_fixed_fp32_cpu_merge_and_cuda_verification_limits(self):
        self.assertEqual(adapter.LIMITS['merge_device'],'cpu')
        self.assertEqual(adapter.LIMITS['merge_dtype'],'float32')
        self.assertEqual(adapter.LIMITS['storage_dtype'],'float32')
        self.assertEqual(adapter.LIMITS['forward_device'],'cuda')
        self.assertEqual(adapter.LIMITS['forward_dtype'],'float32')
        self.assertEqual(adapter.LIMITS['attention_backend'],'sdpa')
        self.assertEqual((adapter.ATOL,adapter.RTOL),(.002,.001))
        self.assertEqual(adapter.LIMITS['forward_calls'],24)
        self.assertEqual(adapter.LIMITS['external_seconds'],900)
        self.assertEqual(adapter.LIMITS['reserve_bytes'],25*1024**3)

    def test_preparation_does_not_load_model_or_execute_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'experiments/reports').mkdir(parents=True)
            (root/'outputs').mkdir()
            with patch.object(adapter,'ROOT',root), patch.object(adapter,'source_bindings',return_value={'source':'fixed'}), \
                    patch.object(adapter,'input_bindings',return_value={'inputs':'fixed'}), \
                    patch.object(adapter,'parent_binding',return_value={'parent':'fixed'}), \
                    patch.object(adapter,'_supervise',side_effect=AssertionError('no child')), \
                    patch.dict('sys.modules',{'torch':None,'transformers':None,'peft':None}):
                record = adapter.prepare('run-01')
                self.assertFalse(record['execution_requested'])
                self.assertFalse(Path(record['output']).exists())
                self.assertFalse(Path(record['policy']).exists())
                self.assertTrue((Path(record['evidence'])/'preparation.json').is_file())

    def test_prepared_tampering_is_rejected_before_inputs(self):
        with self.assertRaisesRegex(ValueError,'Prepared command'):
            adapter.verify_prepared({'schema':'dongxi-fixed-native-lora400-merge-v1',
                'command':['unselected'], 'preparation_sha256':'0'*64})


class ActualParentMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.locations,cls.inputs,cls.values,cls.adapter_files,cls.base_files = metadata()

    def validate(self,values):
        return adapter.validate_parent_metadata(values['acceptance'],values['result'],values['config'],
            values['genealogy'],values['adapter_config'],values['preparation'],values['closing'],
            adapter_path=self.locations['adapter'],adapter_files=self.adapter_files,
            base_files=self.base_files,inputs=self.inputs)

    def test_genuine_fresh400_accepted_bytes_interface(self):
        value = self.validate(self.values)
        self.assertEqual(value['updates'],400)
        self.assertEqual(value['adapter_files'],self.adapter_files)
        self.assertEqual(value['base_files'],self.base_files)
        self.assertEqual(value['interface']['interface_sha256'],adapter.INTERFACE_SHA256)

    def test_short_or_resumed_or_unaccepted_parent_is_rejected(self):
        for key,value in (('updates',399),('resume_checkpoint_update',1),('status','failed')):
            values = deepcopy(self.values)
            values['result'][key] = value
            values['acceptance']['result'] = deepcopy(values['result'])
            with self.assertRaises(ValueError): self.validate(values)
        values = deepcopy(self.values); values['acceptance']['checks']['completed400'] = False
        with self.assertRaises(ValueError): self.validate(values)

    def test_parent_bytes_rank_targets_and_full_interface_cannot_drift(self):
        cases = [('acceptance','exported_policy'),('adapter_config','r'),
            ('adapter_config','target_modules'),('genealogy','checkpoint_interface')]
        for branch,key in cases:
            values = deepcopy(self.values)
            if key=='exported_policy': values[branch][key]['files']['adapter_model.safetensors']='0'*64
            elif key=='r': values[branch][key]=16
            elif key=='target_modules': values[branch][key]=['q_proj','k_proj']
            else: values[branch][key]['generation_stop_ids']=[151643]
            with self.assertRaises(ValueError): self.validate(values)

    def test_merged_genealogy_standard_consumer_parent_binding(self):
        parent = self.validate(self.values)
        g = adapter.genealogy({'parent_binding':parent})
        self.assertEqual(g['kind'],'full-HF-model')
        self.assertEqual(g['method'],'explicit-local-PEFT-merge')
        self.assertEqual(g['merge_precision'],'CPU FP32')
        self.assertEqual(g['parent_adapter'],str(self.locations['adapter']))
        self.assertEqual(g['adapter_files'],self.adapter_files)
        self.assertEqual(g['checkpoint_interface'],parent['interface'])


class CPUComparison(unittest.TestCase):
    def test_merge_tolerance_and_bitwise_reload_are_separate(self):
        import torch
        before = torch.tensor([[[1.,2.,3.],[4.,5.,6.]]])
        after = before+.001
        self.assertTrue(adapter.comparison(torch,'control',before,after)['passed'])
        self.assertFalse(adapter.comparison(torch,'control',before,after,exact=True)['passed'])
        self.assertTrue(adapter.comparison(torch,'control',before,before.clone(),exact=True)['passed'])
        self.assertFalse(adapter.comparison(torch,'control',before,before+.1)['passed'])

    def test_nonfinite_wrong_shape_or_storage_precision_fails_closed(self):
        import torch
        before = torch.ones(1,2,3)
        for after in (torch.full_like(before,float('nan')),torch.ones(1,3,3),before.double()):
            with self.assertRaises(ValueError): adapter.comparison(torch,'control',before,after)
        model = torch.nn.Linear(2,2).float()
        self.assertEqual(adapter.floating_parameters(torch,model,'cpu')['parameter_elements'],6)
        with self.assertRaises(ValueError): adapter.floating_parameters(torch,model,'cuda')
        with self.assertRaises(ValueError): adapter.floating_parameters(torch,model.double(),'cpu')


if __name__=='__main__':
    unittest.main()

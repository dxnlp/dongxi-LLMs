"""Closed120-item evaluation preparation and parent routing, no model run."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assistant_eval',ROOT/'scripts/run_native_assistant_evaluation.py')
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PublicationPreparation(unittest.TestCase):
    def test_original_panel_separates_references_from_model_messages(self):
        items=adapter.publication_items()
        self.assertEqual(len(items),120)
        self.assertEqual(len({item['source_group'] for item in items}),40)
        self.assertTrue(all(item['split']=='publication' for item in items))
        self.assertTrue(all(item['messages'][-1]['role']=='user' for item in items))
        self.assertTrue(all(not any(message['role']=='assistant' for message in item['messages']) for item in items))
        self.assertEqual({item['task'] for item in items},{'copy','reverse','extract'})

    def test_unselected_parent_or_escaping_run_identity_is_refused(self):
        for label,run in [('adapter','run-01'),('base','../../outside'),('full400','latest')]:
            with self.assertRaises(ValueError): adapter.checkpoint(label,run)

    def test_full_and_explicit_merged_lora_are_distinct(self):
        full=adapter.checkpoint('full400','run-01')
        merged=adapter.checkpoint('lora400-fp32','run-01')
        self.assertNotEqual(full,merged)
        self.assertEqual(full.name,'policy')
        self.assertIn('merged',str(merged))

    def test_natural_stop_is_not_part_of_the_answer_content(self):
        row=dict(item_id='copy-100',source_group='100',error=None,
            raw_response='item100<|im_end|>',response_text='item100',
            stop_reason='turn_stop',truncated=False,generated_tokens=4)
        result=adapter.strict_result(row,{'reference':'item100'})
        self.assertTrue(result['exact'])
        self.assertTrue(result['natural_stop'])
        self.assertEqual(row['raw_response'],'item100<|im_end|>')
        row['response_text']='Item100'
        self.assertFalse(adapter.strict_result(row,{'reference':'item100'})['exact'])

    def test_score_does_not_turn_error_or_cap_into_natural_termination(self):
        row=dict(item_id='copy-100',source_group='100',error='failed',
            raw_response='item100',response_text='item100',
            stop_reason='max_new_tokens',truncated=True,generated_tokens=64)
        result=adapter.strict_result(row,{'reference':'item100'})
        self.assertFalse(result['exact'])
        self.assertFalse(result['natural_stop'])
        self.assertTrue(result['truncated'])


if __name__=='__main__':unittest.main()

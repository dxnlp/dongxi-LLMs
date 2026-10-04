"""CPU contract tests for the optional CUDA runner; never load a checkpoint."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('course_spark_sft',ROOT/'scripts/run_chapter09_spark_sft.py')
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
GEN_SPEC=importlib.util.spec_from_file_location('course_instruction_gen',ROOT/'scripts/prepare_chapter09_instruction_fixture.py')
generator=importlib.util.module_from_spec(GEN_SPEC)
GEN_SPEC.loader.exec_module(generator)


class SymbolicTemplate:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False,**kwargs):
        text=''.join('<'+m['role']+'>'+m['content']+'<end>\n' for m in messages)
        if add_generation_prompt: text+='<assistant>'
        return [ord(char) for char in text]


class SparkContractTests(unittest.TestCase):
    def test_prefix_mask_and_ending(self):
        record=dict(id='example',messages=[dict(role='user',content='copy red'),dict(role='assistant',content='red')])
        result=runner.encode_record(SymbolicTemplate(),record,256)
        supervised=''.join(chr(token) for token in result['labels'] if token!=-100)
        self.assertEqual(supervised,'red<end>\n')
        self.assertTrue(all(token==-100 for token in result['labels'][:len(result['labels'])-len(supervised)]))

    def test_multi_turn_ownership_and_length_rejection(self):
        record=dict(id='multi',messages=[dict(role='user',content='one'),dict(role='assistant',content='two'),
                                         dict(role='user',content='three'),dict(role='assistant',content='four')])
        result=runner.encode_record(SymbolicTemplate(),record,256)
        self.assertEqual(''.join(chr(token) for token in result['labels'] if token!=-100),'two<end>\nfour<end>\n')
        with self.assertRaises(ValueError): runner.encode_record(SymbolicTemplate(),record,2)

    def test_reject_incompatible_prefix(self):
        class Broken(SymbolicTemplate):
            def apply_chat_template(self,*args,**kwargs):
                ids=super().apply_chat_template(*args,**kwargs)
                return ids+[99] if kwargs.get('add_generation_prompt') else ids
        record=dict(id='example',messages=[dict(role='user',content='a'),dict(role='assistant',content='b')])
        with self.assertRaises(ValueError): runner.encode_record(Broken(),record,256)

    def test_original_fixture_group_separation(self):
        train=generator.records('train',0,80)
        dev=generator.records('dev',80,20)
        test=generator.records('test',100,40)
        self.assertEqual([len(train),len(dev),len(test)],[240,60,120])
        self.assertFalse({row['group'] for row in train}&{row['group'] for row in dev+test})
        self.assertFalse({row['group'] for row in dev}&{row['group'] for row in test})


if __name__=='__main__': unittest.main()

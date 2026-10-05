"""CPU contract tests for the optional CUDA runner; never load a checkpoint."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch
from snapshot_io_test_support import explicit_snapshot_io_args

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('course_spark_sft',ROOT/'scripts/run_chapter09_spark_sft.py')
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
GEN_SPEC=importlib.util.spec_from_file_location('course_instruction_gen',ROOT/'scripts/prepare_chapter09_instruction_fixture.py')
generator=importlib.util.module_from_spec(GEN_SPEC)
GEN_SPEC.loader.exec_module(generator)


def explicit_work_args(path):
    limits=path/'work-limits.json'
    runner.write_json(limits,{key:10000 for key in runner.WORK_KEYS})
    return ['--work-limits',str(limits),'--work-journal-max-bytes',str(1024**2)]+explicit_snapshot_io_args(path)


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

    def test_hardware_failure_keeps_source_lock_and_stage_without_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            # Contents are never decoded: this test stops at the hardware gate.
            inputs=[]
            for name in ('train.jsonl','dev.jsonl','template.jinja','environment.lock'):
                target=path/name; target.write_text('original preflight-only fixture\n'); inputs.append(target)
            args=['--revision','1'*40,'--tokenizer-revision','1'*40,
                  '--train',str(inputs[0]),'--dev',str(inputs[1]),'--template',str(inputs[2]),
                  '--environment-lock',str(inputs[3]),'--output',str(path/'run')]+explicit_work_args(path)
            with patch.object(runner.torch.cuda,'is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'CUDA'):
                    runner.main(args)
            saved=json.loads(next((path/'run').glob('identity-*.json')).read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['failure']['stage'],'hardware_preflight')
            self.assertEqual(saved['identity']['environment']['environment_lock']['status'],'hashed')
            self.assertTrue(saved['identity']['source_sha256'])

    def test_resume_cannot_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); output=path/'prior-run'; output.mkdir()
            preserved=output/'config.json'; preserved.write_text('original user run')
            args=['--revision','1'*40,'--tokenizer-revision','1'*40,
                  '--train',str(path/'train'),'--dev',str(path/'dev'),'--template',str(path/'template'),
                  '--environment-lock',str(path/'lock'),'--resume',str(path/'checkpoint.pt'),
                  '--output',str(output)]+explicit_work_args(path)
            with self.assertRaisesRegex(FileExistsError,'every invocation'):
                runner.main(args)
            self.assertEqual(preserved.read_text(),'original user run')
            self.assertEqual(len(list(output.iterdir())),1)

    def test_keyboard_interrupt_finalizes_partial_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ('train','dev','template','lock'):
                (path/name).write_text('preflight-only fixture')
            args=['--revision','1'*40,'--tokenizer-revision','1'*40,
                  '--train',str(path/'train'),'--dev',str(path/'dev'),'--template',str(path/'template'),
                  '--environment-lock',str(path/'lock'),'--output',str(path/'run')]+explicit_work_args(path)
            with patch.object(runner.torch.cuda,'is_available',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.main(args)
            saved=json.loads(next((path/'run').glob('identity-*.json')).read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['failure']['type'],'KeyboardInterrupt')
            self.assertEqual(saved['failure']['stage'],'hardware_preflight')

    def test_distinct_tokenizer_revision_needs_actual_base_semantics(self):
        from test_run_identity import local_tokenizer, TEMPLATE, REVISION
        options=dict(template=TEMPLATE,stop_ids=[1],tokenizer_id='local-fixture',
                     base_revision=REVISION,tokenizer_revision='2'*40)
        result=runner.verify_base_tokenizer(local_tokenizer(),local_tokenizer(),**options)
        self.assertEqual(result['source']['tokenizer_revision'],'2'*40)
        for changed in (local_tokenizer(permuted=True),local_tokenizer(lowercase=True)):
            with self.assertRaises(ValueError):
                runner.verify_base_tokenizer(local_tokenizer(),changed,**options)


if __name__=='__main__': unittest.main()

"""Standard-library checks only; no Hub request or tokenizer/model load."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pinned_inspection', ROOT/'scripts/inspect_pinned_base_tokenizer.py')
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


class PinnedTokenizerInspectionTests(unittest.TestCase):
    def test_manifest_and_recipe_pin(self):
        manifest = json.loads(inspection.MANIFEST.read_text())
        recipe = json.loads((ROOT/'experiments/configs/sft-0.6b-full-profile.json').read_text())
        self.assertEqual(manifest['declared_revision'], inspection.REVISION)
        self.assertEqual(recipe['revision'], inspection.REVISION)
        self.assertEqual(recipe['tokenizer_revision'], inspection.REVISION)
        self.assertEqual(manifest['repository'], inspection.MODEL)
        self.assertEqual(len(manifest['files']), 10)
        self.assertEqual(sum(row['size'] for row in manifest['files']), 1203641805)

    def test_original_generator_hashes(self):
        generator = inspection.source_function(ROOT/'scripts/prepare_chapter09_instruction_fixture.py', 'records')
        card = json.loads((ROOT/'experiments/data/instruction-interface-v1-data-card.json').read_text())
        for split,start,count in [('train',0,80),('dev',80,20),('test',100,40)]:
            rows = generator(split,start,count)
            data = ''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows).encode()
            self.assertEqual(hashlib.sha256(data).hexdigest(), card['splits'][split]['sha256'])

    def test_profile_selected_order(self):
        order = list(range(240)); random.Random(1212).shuffle(order)
        self.assertEqual(order[:8], [135,147,24,166,212,10,216,2])
        self.assertEqual(len(set(order[:80])), 80)
        self.assertEqual([sum(i%3==j for i in order[:80]) for j in range(3)], [28,22,30])

    def test_native_generation_reservation(self):
        upper = inspection.source_function(ROOT/'scripts/run_chapter09_spark_sft.py', 'generation_upper')
        prefixes = [[1]*30, [1]*40]
        actual = upper(prefixes,64)
        self.assertEqual(actual['policy_forward_positions'], 64*70+2*2016)
        self.assertEqual(actual['logical_sequence_tokens'], 70+128)
        self.assertEqual(actual['generation_calls'], 2)
        self.assertEqual(actual['policy_forward_calls'], 128)
        with self.assertRaises(ValueError): upper([[]],64)

    def test_shifted_target_rule_is_native(self):
        encoder = inspection.source_function(ROOT/'scripts/run_chapter09_spark_sft.py','encode_record')
        class Tokenizer:
            def apply_chat_template(self,messages,*,add_generation_prompt,**kwargs):
                ids = []
                for row in messages:
                    ids += [10,11]+row['content']+[12,13]
                if add_generation_prompt: ids += [10,11]
                return ids
        row = {'id':'one','messages':[{'role':'user','content':[20]},
                                     {'role':'assistant','content':[21,22]}]}
        encoded = encoder(Tokenizer(),row,256)
        self.assertEqual(encoded['labels'], [-100]*7+[21,22,12,13])
        self.assertEqual(sum(v!=-100 for v in encoded['labels'][1:]),4)
        with self.assertRaises(ValueError): encoder(Tokenizer(),row,10)

    def test_streamed_git_blob_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'file'; path.write_bytes(b'hello\n')
            expected = hashlib.sha1(b'blob 6\0hello\n').hexdigest()
            self.assertEqual(inspection.digest(path,'sha1',True),expected)
            self.assertEqual(inspection.digest(path),hashlib.sha256(b'hello\n').hexdigest())

    def test_ast_extraction_rejects_duplicate_and_decorated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.py'
            for source in ['def f(): pass\ndef f(): pass\n', '@something\ndef f(): pass\n']:
                path.write_text(source)
                with self.assertRaises(ValueError): inspection.source_function(path,'f')


if __name__ == '__main__':
    unittest.main()

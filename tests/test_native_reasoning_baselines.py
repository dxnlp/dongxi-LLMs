"""CPU-only closed-selector/interface controls; never launches pretrained CUDA."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.reasoning_generation import generate_record, serialize_prompt
from dongxi_llms.run_identity import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('native_reasoning_baselines',
    ROOT/'scripts/run_native_reasoning_baselines.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

NATIVE_TEMPLATE = ("{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}"
    "assistant {% if enable_thinking is defined and enable_thinking is false %}<think> </think> {% endif %}")


def tokenizer_fixture(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocabulary = {'[UNK]': 0, '<|endoftext|>': 1, '<|im_end|>': 2,
        '<|im_start|>': 3, 'user': 4, 'assistant': 5, '<think>': 6,
        '</think>': 7, 'Compute': 8, '1': 9, '+': 10, '2.': 11}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    tokenizer.chat_template = NATIVE_TEMPLATE
    tokenizer.save_pretrained(directory)
    return tokenizer


class ImmediateEOS:
    """A declared one-forward stop control, not a measured pretrained model."""
    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        logits = torch.full((1, input_ids.shape[1], 12), -100.)
        logits[0, -1, 1] = 100.
        return SimpleNamespace(logits=logits)


class NativeReasoningBaselineControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dongxi-reasoning-controls-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'experiments/reports').mkdir(parents=True)
        (self.root/'outputs').mkdir()
        (self.root/'experiments/data').mkdir()
        (self.root/'experiments/data/instruction_interface_v1.jinja').write_bytes(
            (ROOT/'experiments/data/instruction_interface_v1.jinja').read_bytes())
        self.tokenizer = tokenizer_fixture(self.root/'tiny-tokenizer')
        self.panel = runner.logical_panel()
        self.model = dict(path=str(self.root/'tiny-tokenizer'), model='authored-tiny-control',
            revision='a'*40, files={'config.json': 'b'*64, 'model.safetensors': 'c'*64},
            revision_evidence='Injected fixture identity, not pretrained evidence')
        self.sources = {'source.py': {'path': '/fixture/source.py', 'sha256': 'd'*64, 'bytes': 10}}
        self.inputs = {}
        for key, content in [('math_items', b'authored source binding'),
                             ('custom_template', b'authored template binding'),
                             ('interpreter', b'authored interpreter binding'),
                             ('environment_lock', b'authored lock binding')]:
            path = self.root/(key+'.txt')
            path.write_bytes(content)
            self.inputs[key] = runner._digest(str(path))
        for patcher in (patch.object(runner, 'ROOT', self.root),
                patch.object(runner, 'logical_panel', return_value=self.panel),
                patch.object(runner, 'source_bindings', return_value=self.sources),
                patch.object(runner, 'fixed_inputs', side_effect=lambda: deepcopy(self.inputs)),
                patch.object(runner, 'local_model_binding', return_value=self.model),
                patch.object(runner, 'load_local_tokenizer', return_value=self.tokenizer)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, row='base-raw', budget=32, run_id='run-01'):
        return runner.prepare(row, budget, run_id)

    def test_default_preparation_never_launches_or_loads_weights(self):
        with (patch.object(runner, '_supervise') as supervise, patch.object(runner, 'run_generation') as generation,
                patch('subprocess.Popen') as spawn, patch('transformers.AutoModelForCausalLM.from_pretrained') as weights):
            self.assertEqual(runner.main(['--row', 'base-raw', '--budget', '32', '--run-id', 'run-01']), 0)
        supervise.assert_not_called(); generation.assert_not_called(); spawn.assert_not_called(); weights.assert_not_called()
        evidence = runner.paths('base-raw', 32, 'run-01')['evidence']
        self.assertEqual(evidence.stat().st_mode & 0o777, 0o700)
        self.assertFalse(runner.paths('base-raw', 32, 'run-01')['output'].exists())

    def test_full_original_panel_annotations_prompts_and_references_survive_freeze(self):
        record = self.prepare()
        actual = json.loads((Path(record['evidence'])/'items.json').read_text())
        self.assertEqual(actual, self.panel['items'])
        self.assertEqual(len(actual), 20)
        self.assertTrue(any(item['rlvr_train_problem_overlap'] for item in actual))
        self.assertTrue(any(item['campaign_role'] == 'heldout-controlled-slice' for item in actual))
        self.assertEqual([cell['id'] for cell in record['cells']],
            ['sample-1009', 'sample-1019', 'sample-1029', 'sample-1039', 'greedy-1009'])
        self.assertEqual(record['limits']['planned_responses'], 100)
        runner.verify_prepared(record)

    def test_both_budgets_and_all_four_interfaces_are_explicit(self):
        for row in runner.ROWS:
            for budget in runner.BUDGETS:
                with self.subTest(row=row, budget=budget):
                    record = self.prepare(row, budget)
                    contract = json.loads(Path(record['cells'][0]['contract_path']).read_text())
                    settings = contract['settings']
                    self.assertEqual(settings['max_new_tokens'], budget)
                    self.assertEqual(settings['decoding'], dict(mode='sample', seed=1009,
                        temperature=1., top_k=None, top_p=1.))
                    self.assertEqual(settings['generation']['samples'], 1)
                    self.assertEqual(settings['generation']['dtype'], 'bfloat16')
                    self.assertEqual(settings['stopping']['eos_token_ids'], [1])
                    self.assertEqual(settings['stopping']['turn_stop_token_ids'], [2])
                    self.assertEqual(record['limits']['external_seconds'], 900)

    def test_raw_is_unmodified_custom_chat_and_native_thinking_toggle_are_distinct(self):
        item = self.panel['items'][0]
        raw, _ = serialize_prompt(self.tokenizer, item,
            runner.settings('base-raw', 32, self.tokenizer, 'greedy', 1009))
        chat, _ = serialize_prompt(self.tokenizer, item,
            runner.settings('base-chat', 32, self.tokenizer, 'greedy', 1009))
        off, _ = serialize_prompt(self.tokenizer, item,
            runner.settings('instruct-thinking-off', 128, self.tokenizer, 'sample', 1009))
        on, _ = serialize_prompt(self.tokenizer, item,
            runner.settings('instruct-thinking-on', 128, self.tokenizer, 'sample', 1009))
        self.assertEqual(raw, item['prompt'])
        self.assertEqual(chat, '<|im_start|>user\n'+item['prompt']+'<|im_end|>\n<|im_start|>assistant\n')
        self.assertIn('<think> </think>', off)
        self.assertNotIn('<think> </think>', on)
        self.assertIn(item['prompt'], off); self.assertIn(item['prompt'], on)

    def test_unsupported_toggle_or_missing_stop_refuses_before_launch(self):
        self.tokenizer.chat_template = 'user {{ messages[0]["content"] }} assistant'
        with self.assertRaisesRegex(ValueError, 'enable_thinking'):
            runner.settings('instruct-thinking-on', 32, self.tokenizer, 'sample', 1009)
        with patch.object(self.tokenizer, 'get_vocab', return_value={'[UNK]': 0}):
            with self.assertRaisesRegex(ValueError, 'token IDs'):
                runner.settings('base-raw', 32, self.tokenizer, 'sample', 1009)

    def test_unknown_selectors_never_create_artifacts(self):
        for arguments in [('remote-model', 32, 'run-01'), ('base-raw', 256, 'run-01'),
                          ('base-raw', True, 'run-01'), ('base-raw', 32, '../escape'),
                          ('base-raw', 32, 'run-001')]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                runner.prepare(*arguments)
        self.assertEqual(list((self.root/'experiments/reports').iterdir()), [])

    def test_existing_evidence_is_exclusive_and_symlink_target_refuses(self):
        record = self.prepare()
        original = (Path(record['evidence'])/'preparation.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((Path(record['evidence'])/'preparation.json').read_bytes(), original)
        paths = runner.paths('base-raw', 128, 'run-01')
        paths['output'].symlink_to(self.root/'absent')
        with self.assertRaises(FileExistsError):
            self.prepare(budget=128)
        self.assertFalse(paths['evidence'].exists())

    def test_source_model_and_contract_changes_refuse(self):
        record = self.prepare()
        for target, replacement, message in (
                ('source_bindings', {'changed': 1}, 'source'),
                ('local_model_binding', dict(self.model, files={'changed': 'e'*64}), 'model artifacts')):
            with patch.object(runner, target, return_value=replacement), self.assertRaisesRegex(ValueError, message):
                runner.verify_prepared(record)
        contract_path = Path(record['cells'][0]['contract_path'])
        original = contract_path.read_text()
        contract_path.write_text(original+' ')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            runner.verify_prepared(record)

    def test_even_rehashed_arbitrary_argv_or_output_redirection_refuses(self):
        record = self.prepare()
        for field, value in [('argv', ['/usr/bin/true']), ('output', str(self.root/'other-output'))]:
            changed = deepcopy(record); changed[field] = value
            changed['preparation_sha256'] = canonical_hash({k: v for k, v in changed.items() if k != 'preparation_sha256'})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Closed prepared'):
                runner.verify_prepared(changed)

    def test_prepare_changed_model_is_failed_receipt_not_execution(self):
        with (patch.object(runner, 'local_model_binding', side_effect=[self.model, dict(self.model, files={'changed': 'e'*64})]),
                patch.object(runner, '_supervise') as supervise):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.prepare()
        supervise.assert_not_called()
        failure = runner.paths('base-raw', 32, 'run-01')['evidence']/'preparation-failure.json'
        self.assertEqual(json.loads(failure.read_text())['status'], 'failed')

    def test_internal_child_needs_parent_launch_receipt(self):
        self.prepare()
        with patch.object(runner, 'run_generation') as generation, self.assertRaises(FileNotFoundError):
            runner.child('base-raw', 32, 'run-01')
        generation.assert_not_called()

    def test_missing_model_weights_refuse_without_any_download(self):
        # The outer per-test mock is bypassed explicitly to exercise cache
        # inventory admission without touching the pretrained cache.
        source_spec = importlib.util.spec_from_file_location('reasoning_admission_control',
            ROOT/'scripts/run_native_reasoning_baselines.py')
        admission = importlib.util.module_from_spec(source_spec); source_spec.loader.exec_module(admission)
        with (patch.object(admission, 'cached_snapshot', return_value=self.root/'tiny-tokenizer'),
                patch('transformers.AutoModelForCausalLM.from_pretrained') as weights):
            with self.assertRaisesRegex(ValueError, 'cached full model weights'):
                admission.local_model_binding('base-raw')
        weights.assert_not_called()

    def test_partial_summary_keeps_train_labels_and_nonfinal_costs_without_double_counting(self):
        record = self.prepare()
        cell = record['cells'][0]
        contract = json.loads(Path(cell['contract_path']).read_text())
        contract['settings']['generation']['device'] = 'cpu'
        contract['settings']['generation']['dtype'] = 'float32'
        contract['identity'] = canonical_hash({k: v for k, v in contract.items() if k != 'identity'})
        Path(cell['contract_path']).write_text(json.dumps(contract))
        output = Path(cell['output']); output.mkdir(parents=True)
        identity = {'identity_sha256': 'a'*64, 'input_sha256': {}}
        completed = generate_record(ImmediateEOS(), self.tokenizer, self.panel['items'][0], contract,
            checkpoint_id='authored-cpu-stop-control', identity=identity)
        unfinished = generate_record(ImmediateEOS(), self.tokenizer, self.panel['items'][1], contract,
            checkpoint_id='authored-cpu-stop-control', identity=identity)
        (output/'responses.jsonl').write_text(json.dumps(completed)+'\n'+'{"interrupted-write":')
        (output/'events.jsonl').write_text('\n'.join(json.dumps({'stage': 'partial_response', 'record': row})
            for row in (completed, unfinished, unfinished))+'\n')
        summary = runner.output_summary(record)
        self.assertEqual(summary['all_original_rows']['records'], 1)
        self.assertEqual(summary['heldout']['records'], 0)
        self.assertEqual(summary['missing_responses'], 99)
        self.assertEqual(len(summary['incomplete_attempts']), 1)
        self.assertEqual(summary['incomplete_observed_costs']['generation_tokens'], 1)
        self.assertEqual(len(summary['unreadable_jsonl_lines']), 1)
        self.assertEqual(summary['rows'][0]['panel_annotation']['campaign_role'], 'development-or-seen-diagnostic')


if __name__ == '__main__':
    unittest.main()

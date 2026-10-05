"""Original, local-only checkpoint-interface and evidence regression tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.run_identity import (IdentityJournal, artifact_hashes,
    assert_compatible, canonical_hash, cached_snapshot, collect_run_identity, parent_interface,
    prepare_model_snapshot, safe_command,
    safe_config, tokenizer_interface, validate_interface, write_json)

REVISION = "1" * 40
TEMPLATE = "{% for message in messages %}{{ message.content }} {% endfor %}"


def local_tokenizer(permuted=False, lowercase=False):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "cat": 2, "dog": 3,
                  "red": 4, "blue": 5, "hello": 6, "[BOS]": 7}
    if permuted:
        vocabulary["cat"], vocabulary["dog"] = vocabulary["dog"], vocabulary["cat"]
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if lowercase:
        backend.normalizer = normalizers.Lowercase()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                    eos_token="[EOS]", pad_token="[EOS]", bos_token="[BOS]")
    result.chat_template = TEMPLATE
    return result


def interface(tokenizer, **overrides):
    config = dict(template=TEMPLATE, stop_ids=[1], tokenizer_id="original-local-fixture",
                  tokenizer_revision=REVISION)
    config.update(overrides)
    return tokenizer_interface(tokenizer, **config)


class InterfaceTests(unittest.TestCase):
    def test_actual_hf_save_reload_preserves_interface(self):
        from transformers import AutoTokenizer
        tokenizer = local_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            reloaded = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.assertTrue(assert_compatible(interface(tokenizer), interface(reloaded)))
            self.assertEqual(tokenizer.encode("cat dog"), reloaded.encode("cat dog"))

    def test_same_size_permutation_and_encoding_change_rejected(self):
        original = interface(local_tokenizer())
        for changed in (local_tokenizer(permuted=True), local_tokenizer(lowercase=True)):
            self.assertEqual(original["tokenizer"]["vocab_size"], len(changed.get_vocab()))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                assert_compatible(original, interface(changed))

    def test_special_stop_template_and_revision_are_contracts(self):
        tokenizer = local_tokenizer()
        original = interface(tokenizer)
        other = local_tokenizer()
        other.eos_token = "[BOS]"
        changes = [interface(other), interface(tokenizer, stop_ids=[7]),
                   interface(tokenizer, template=TEMPLATE + "changed"),
                   interface(tokenizer, tokenizer_revision="2" * 40)]
        for changed in changes:
            with self.assertRaises(ValueError):
                assert_compatible(original, changed)
        # Source declarations are evidence separate from semantically equal mappings.
        self.assertTrue(assert_compatible(original, changes[-1], compare_source=False))

    def test_runtime_padding_and_truncation_do_not_change_semantics(self):
        tokenizer = local_tokenizer()
        original = interface(tokenizer)
        tokenizer.backend_tokenizer.enable_padding(pad_id=1, pad_token="[EOS]")
        tokenizer.backend_tokenizer.enable_truncation(5)
        self.assertTrue(assert_compatible(original, interface(tokenizer)))

    def test_tampering_bad_schema_and_slow_tokenizer_fail_closed(self):
        value = interface(local_tokenizer())
        changed = copy.deepcopy(value)
        changed["tokenizer"]["vocab_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_interface(changed)
        for revision in (1, "main", "f" * 39):
            with self.assertRaises(ValueError):
                interface(local_tokenizer(), tokenizer_revision=revision)
        tokenizer = type("Unsafe", (), {"get_vocab": lambda self: {"a": 0}})()
        with self.assertRaisesRegex(ValueError, "serialization"):
            interface(tokenizer)

    def test_legacy_adoption_is_explicit_and_parent_files_rechecked(self):
        tokenizer = local_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Legacy"):
                parent_interface(directory, tokenizer, template=TEMPLATE, stop_ids=[1])
            observed, metadata = parent_interface(directory, tokenizer, template=TEMPLATE,
                stop_ids=[1], allow_legacy=True, tokenizer_id="fixture", tokenizer_revision=REVISION)
            self.assertTrue(metadata["legacy_adoption"])
            write_json(Path(directory) / "course-genealogy.json", {"checkpoint_interface": observed})
            verified, metadata = parent_interface(directory, tokenizer, template=TEMPLATE, stop_ids=[1])
            self.assertFalse(metadata["legacy_adoption"])
            self.assertEqual(verified, observed)
            with self.assertRaises(ValueError):
                parent_interface(directory, local_tokenizer(permuted=True), template=TEMPLATE, stop_ids=[1])


class EvidenceTests(unittest.TestCase):
    def test_hashes_changes_guards_and_unrecognized_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            (path / "credentials.env").write_text("not collected")
            first = artifact_hashes(path)
            self.assertEqual(list(first), ["config.json"])
            (path / "config.json").write_text('{"changed": true}')
            self.assertNotEqual(first, artifact_hashes(path))
            def fail():
                raise TimeoutError("cooperative guard")
            with self.assertRaises(TimeoutError):
                artifact_hashes(path, fail)

    def test_environment_source_identity_and_no_cpu_driver_query(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source, lock = path / "source.py", path / "fixture.lock"
            source.write_text("# original test fixture\n")
            lock.write_text("fixed fixture lock, not an installed-environment assertion\n")
            calls = []
            def command(args, cwd=None):
                calls.append(args)
                return {"status": "measured", "value": "" if "status" in args else REVISION}
            with patch("dongxi_llms.run_identity._command_output", side_effect=command):
                identity = collect_run_identity(path, source_files=[source], input_files=[lock],
                    environment_lock=lock, interface=interface(local_tokenizer()),
                    device={"mode": "cpu"}, command=["runner", "--token", "secret"])
            self.assertFalse(any("nvidia-smi" in args for args in calls))
            self.assertEqual(identity["git"]["changed_paths"], [])
            self.assertEqual(identity["environment"]["environment_lock"]["status"], "hashed")
            self.assertIn("source.py", identity["source_sha256"])
            self.assertEqual(identity["command"][-1], "[REDACTED]")
            self.assertEqual(identity['identity_sha256'],canonical_hash({k:v for k,v in identity.items()
                if k!='identity_sha256'}))

    def test_pinned_snapshot_path_and_explicit_acquisition_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/REVISION; path.mkdir()
            (path/'config.json').write_text('{}')
            with patch('huggingface_hub.try_to_load_from_cache',return_value=str(path/'config.json')):
                self.assertEqual(cached_snapshot('fixture',REVISION),path)
                self.assertEqual(prepare_model_snapshot('fixture',REVISION),path)
                with self.assertRaises(ValueError):
                    cached_snapshot('fixture','2'*40)
            with patch('huggingface_hub.try_to_load_from_cache',return_value=None):
                with self.assertRaises(ValueError):
                    cached_snapshot('fixture',REVISION)
            with patch('huggingface_hub.snapshot_download',return_value=str(path)) as acquire:
                self.assertEqual(prepare_model_snapshot('fixture',REVISION,allow_download=True),path)
                self.assertEqual(acquire.call_count,1)
            with self.assertRaises(ValueError):
                cached_snapshot('fixture','main')

    def test_redaction_does_not_remove_tokenizer_or_token_ids(self):
        value = safe_config({"hf_token": "secret", "tokenizer": "fixture", "eos_token_id": 1,
                             "nested": {"access-token": "secret"}})
        self.assertEqual(value["hf_token"], "[REDACTED]")
        self.assertEqual(value["nested"]["access-token"], "[REDACTED]")
        self.assertEqual(value["eos_token_id"], 1)
        self.assertEqual(safe_command(["run", "--api-key=secret", "--token", "secret"]),
                         ["run", "--api-key=[REDACTED]", "--token", "[REDACTED]"])

    def test_original_module_invocation_is_not_rewritten_as_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            original = ['python','-m','dongxi_llms.qwen_rlvr_lab','--hf-token=secret']
            rewritten = ['/course/qwen_rlvr_lab.py','--hf-token=secret']
            with patch('dongxi_llms.run_identity.sys.orig_argv', original), \
                 patch('dongxi_llms.run_identity.sys.argv', rewritten), \
                 patch('dongxi_llms.run_identity._command_output', return_value={'status':'measured','value':''}):
                value = collect_run_identity(directory, source_files=[])
            self.assertEqual(value['command'][:3], original[:3])
            self.assertEqual(value['command'][-1], '--hf-token=[REDACTED]')
            self.assertEqual(value['python_argv'][0], rewritten[0])
            self.assertEqual(value['python_argv'][-1], '--hf-token=[REDACTED]')
            self.assertEqual(value['command_origin'], 'original-interpreter-argv')

    def test_actual_module_process_preserves_original_launch(self):
        root = Path(__file__).resolve().parents[1]
        setup = ("import json; from dongxi_llms.run_identity import collect_run_identity; "
                 f"print(json.dumps(collect_run_identity({str(root)!r}, source_files=[])))")
        result = subprocess.run([sys.executable,'-m','timeit','-n','1','-r','1','-s',setup,'pass'],
            cwd=root, text=True, capture_output=True, timeout=30,
            env=dict(os.environ, PYTHONPATH=str(root/'src'), CUDA_VISIBLE_DEVICES='',
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{')))
        self.assertEqual(identity['command'][1:3], ['-m','timeit'])
        self.assertTrue(identity['python_argv'][0].endswith('timeit.py'))
        self.assertEqual(identity['command_origin'], 'original-interpreter-argv')

    def test_partial_failure_and_invocation_identity_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = IdentityJournal(directory, {"token": "secret"}, invocation_id="fixture")
            journal.attach({"source": "verified before model load"})
            journal.stage("loading_model")
            journal.fail(RuntimeError("load failed"))
            saved = json.loads(journal.path.read_text())
            self.assertEqual(saved["failure"]["stage"], "loading_model")
            self.assertEqual(saved["identity"]["source"], "verified before model load")
            self.assertEqual(saved["status"], "failed")
            with self.assertRaises(FileExistsError):
                IdentityJournal(directory, {}, invocation_id="fixture")


class RealLocalMergeTests(unittest.TestCase):
    def test_tiny_random_hf_and_nonzero_peft_merge_reload(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from dongxi_llms.checkpoint_merge import merge_local_adapter
        torch.manual_seed(61)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            base, adapter = path / "base", path / "adapter"
            lock = path / "fixture.lock"
            lock.write_text("tiny random CPU fixture, not model quality evidence\n")
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                bos_token_id=7, eos_token_id=1, pad_token_id=1)).eval()
            tokenizer = local_tokenizer()
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            reloaded = AutoModelForCausalLM.from_pretrained(base, local_files_only=True).eval()
            ids = torch.tensor([[2, 3, 4]])
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, reloaded(ids).logits, rtol=0, atol=0)
            network = get_peft_model(reloaded, LoraConfig(r=2, lora_alpha=2, lora_dropout=0,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", revision=REVISION)).eval()
            with torch.no_grad():
                for name, parameter in network.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(.03)
                expected = network(ids).logits.clone()
            network.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            write_json(adapter / "course-genealogy.json", {"base_model": "local-random-fixture",
                "base_revision": REVISION, "checkpoint_interface": interface(tokenizer),
                "template_sha256": interface(tokenizer)["template_sha256"],
                "base_checkpoint_files": artifact_hashes(base)})
            exported = merge_local_adapter(base, adapter, path / "merge", base_revision=REVISION,
                                           environment_lock=lock)
            merged = AutoModelForCausalLM.from_pretrained(exported, local_files_only=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, merged(ids).logits, rtol=1e-5, atol=1e-6)
            saved_tokenizer = AutoTokenizer.from_pretrained(exported, local_files_only=True)
            self.assertTrue(assert_compatible(interface(tokenizer), interface(saved_tokenizer)))
            self.assertFalse((exported / "adapter_config.json").exists())
            manifest=json.loads(next((path/'merge').glob('identity-*.json')).read_text())['identity']
            self.assertEqual(manifest['identity_sha256'],canonical_hash({k:v for k,v in manifest.items()
                if k!='identity_sha256'}))
            # Actual local HF RLVR entry point, random fixture only; no pretrained model.
            from dongxi_llms.qwen_rlvr_lab import main as rlvr_main, BUDGET_KEYS
            from snapshot_io_test_support import explicit_snapshot_io_args
            work_limits=path/'rlvr-work-limits.json'
            write_json(work_limits,{key:100000 for key in BUDGET_KEYS})
            io_args = explicit_snapshot_io_args(path)
            with patch("dongxi_llms.qwen_rlvr_lab.guard_memory", return_value=100.):
                # The eight-word merge fixture aliases all arithmetic prompts.
                # Keep it as a negative encoding control; do not bless leakage.
                with self.assertRaisesRegex(ValueError, 'encoded.*collide'):
                    rlvr_main(['--model-dir',str(base),'--revision',REVISION,
                        '--output',str(path/'rlvr'),'--device','cpu','--prompt-mode','raw',
                        '--updates','1','--group-size','2','--max-new-tokens','2',
                        '--snapshot-max-bytes','16777216',
                        '--work-limits',str(work_limits),'--work-journal-max-bytes','1048576',
                        '--environment-lock',str(lock), *io_args])
            rlvr_report=json.loads((path/'rlvr/report.json').read_text())
            self.assertEqual(rlvr_report['status'],'failed')
            self.assertEqual(rlvr_report['run_identity']['environment']['environment_lock']['status'],'hashed')
            self.assertEqual(rlvr_report['records'],[])
            # Different actual base bytes are rejected before model loading.
            (base / "config.json").write_text((base / "config.json").read_text() + " ")
            with self.assertRaisesRegex(ValueError, "artifact bytes"):
                merge_local_adapter(base, adapter, path / "bad-merge", base_revision=REVISION,
                                    environment_lock=lock)
            failure = json.loads(next((path / "bad-merge").glob("identity-*.json")).read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIsNotNone(failure['identity'])

    def test_separate_adequate_random_hf_rlvr_identity(self):
        """Separate fixture, not a repaired response or pretrained capability test."""
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from dongxi_llms.qwen_rlvr_lab import main as rlvr_main, BUDGET_KEYS
        from snapshot_io_test_support import explicit_snapshot_io_args
        words = ['[UNK]', '[EOS]', '[BOS]', 'Return', 'only', 'the', 'integer',
                 'answer.', '+', '=', *map(str, range(11))]
        backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                           eos_token='[EOS]', pad_token='[EOS]', bos_token='[BOS]')
        tokenizer.chat_template = TEMPLATE
        torch.manual_seed(71)
        with tempfile.TemporaryDirectory(prefix='dongxi-adequate-identity-') as directory:
            path = Path(directory)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(words), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                bos_token_id=2, eos_token_id=1, pad_token_id=1, attention_dropout=0.))
            model.save_pretrained(path/'base')
            tokenizer.save_pretrained(path/'base')
            (path/'fixture.lock').write_text('original local random arithmetic identity fixture\n')
            write_json(path/'work-limits.json',{key:100000 for key in BUDGET_KEYS})
            io_args = explicit_snapshot_io_args(path)
            with patch('dongxi_llms.qwen_rlvr_lab.guard_memory', return_value=100.):
                rlvr_main(['--model-dir',str(path/'base'),'--revision',REVISION,
                    '--output',str(path/'run'),'--device','cpu','--prompt-mode','raw',
                    '--updates','1','--group-size','2','--max-new-tokens','2',
                    '--snapshot-max-bytes','16777216','--environment-lock',str(path/'fixture.lock'),
                    '--work-limits',str(path/'work-limits.json'),'--work-journal-max-bytes','1048576', *io_args])
            report = json.loads((path/'run/report.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['checkpoint_interface']['tokenizer']['vocab_size'], len(words))
            self.assertEqual(len(report['records']), 1)
            self.assertEqual(report['run_identity']['environment']['environment_lock']['status'], 'hashed')


if __name__ == "__main__":
    unittest.main()

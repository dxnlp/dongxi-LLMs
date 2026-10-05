"""Original local-only CPU generation controls; no pretrained model or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.reasoning_evaluation import (grade_response, replay_records,
    validate_record)
from dongxi_llms.reasoning_generation import (ADAPTER_VERSION, attempt_seed,
    choose_token, freeze_local_contract, generate_record, load_local_tokenizer,
    run_generation, serialize_prompt, validate_settings)
from dongxi_llms.run_identity import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if enable_thinking %}THINK {% else %}NO_THINK {% endif %}assistant "


def tiny_fixture(directory, *, permuted=False):
    """Actual random HF decoder and independently authored fast WordLevel tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[BOS]": 2, "[TURN]": 3,
        "THINK": 4, "NO_THINK": 5, "user": 6, "assistant": 7, "hello": 8,
        "4": 9, "2": 10, "+": 11, "?": 12, "blue": 13, "red": 14, "world": 15}
    if permuted:
        vocabulary["blue"], vocabulary["red"] = vocabulary["red"], vocabulary["blue"]
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
        eos_token="[EOS]", bos_token="[BOS]", pad_token="[EOS]",
        additional_special_tokens=["[TURN]"])
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(directory)
    with torch.random.fork_rng():
        torch.manual_seed(1004)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=32,
            n_ctx=32, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0.,
            embd_pdrop=0., attn_pdrop=0., bos_token_id=2, eos_token_id=1,
            pad_token_id=1))
    model.save_pretrained(directory, safe_serialization=True)
    return model.eval(), tokenizer


def settings(*, mode="raw", decoding="sample", context=24, cap=4):
    return {"template_id": "original-tiny-raw-v1" if mode == "raw" else "original-tiny-chat-v1",
        "thinking_mode": "not-applicable" if mode == "raw" else "disabled",
        "decoding": {"mode": decoding, "seed": 1004, "temperature": 1., "top_k": None, "top_p": 1.},
        "stopping": {"eos_token_ids": [1], "turn_stop_token_ids": [3], "pad_token_id": 1},
        "max_new_tokens": cap,
        "generation": {"input_mode": mode, "template": None if mode == "raw" else TEMPLATE,
            "context_window": context, "samples": 2, "device": "cpu", "dtype": "float32",
            "add_special_tokens": False, "max_run_seconds": 30,
            "scoring_text": "decode_without_terminal_stop"}}


def items():
    return [{"id": "arithmetic-a", "source_group": "original-arithmetic", "split": "dev",
             "task": "arithmetic", "prompt": "2 + 2 ?", "kind": "math", "reference": "4",
             "extraction": "whole", "format_policy": "single_integer"},
            {"id": "greeting-a", "source_group": "original-greeting", "split": "dev",
             "task": "greeting", "prompt": "hello world", "kind": "text", "reference": "hello",
             "extraction": "whole", "format_policy": "any"}]


def identity():
    return {"identity_sha256": "a" * 64, "input_sha256": {"items.json": "b" * 64, "contract.json": "c" * 64}}


class ScriptedForward:
    """Declared stop/error controls; not measured model quality or a real HF model."""
    def __init__(self, actions, failure=None):
        self.actions, self.failure, self.calls = actions, failure, 0

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        index = self.calls
        self.calls += 1
        if self.failure and index == 1:
            raise self.failure("Declared control failure after one token")
        logits = torch.full((1, input_ids.shape[1], 16), -10.)
        logits[0, -1, self.actions[min(index, len(self.actions) - 1)]] = 10.
        return SimpleNamespace(logits=logits)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="dongxi-random-hf-")
        cls.checkpoint = Path(cls.directory.name)
        cls.model, cls.tokenizer = tiny_fixture(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def contract(self, **kwargs):
        return freeze_local_contract(items(), settings(**kwargs), self.tokenizer)

    def record(self, model=None, contract=None, item=None, **kwargs):
        return generate_record(model or self.model, self.tokenizer, item or items()[0],
            contract or self.contract(), checkpoint_id="original-random-control", identity=identity(), **kwargs)

    def test_real_random_hf_forward_and_save_reload_replay(self):
        contract = self.contract()
        a, b = self.record(contract=contract), self.record(contract=contract)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertIsNone(a["error"])
        self.assertEqual(a["generated_tokens"], len(a["token_ids"]))
        self.assertGreater(a["cost"]["forward_seconds"], 0)
        self.assertEqual(a["cost"]["forward_calls"], a["generated_tokens"])
        n = a["generated_tokens"]
        self.assertEqual(a["cost"]["model_forward_tokens"], sum(a["prompt_tokens"] + i for i in range(n)))
        report = replay_records(items(), [a], contract)
        self.assertEqual(report["rows"][0]["token_ids"], a["token_ids"])
        self.assertEqual(report["checkpoints"][a["checkpoint_id"]]["missing_item_ids"], ["greeting-a"])

    def test_attempt_seeds_do_not_depend_on_item_iteration_order(self):
        ids = [i["id"] for i in items()]
        forward = {i: attempt_seed(1004, i, 0) for i in ids}
        reverse = {i: attempt_seed(1004, i, 0) for i in reversed(ids)}
        self.assertEqual(forward, reverse)
        self.assertNotEqual(forward[ids[0]], attempt_seed(1004, ids[0], 1))
        self.assertNotEqual(self.record(sample_index=0)["sample_id"], self.record(sample_index=1)["sample_id"])

    def test_raw_chat_and_thinking_are_explicit_no_gold_in_prompt(self):
        item = dict(items()[0], reference="DO_NOT_PROMPT_THIS_REFERENCE")
        raw = serialize_prompt(self.tokenizer, item, settings())
        self.assertEqual(raw[0], item["prompt"])
        chat_settings = settings(mode="chat")
        chat = serialize_prompt(self.tokenizer, item, chat_settings)
        self.assertIn("NO_THINK", chat[0])
        self.assertNotIn(item["reference"], chat[0])
        chat_settings["thinking_mode"] = "enabled"
        self.assertIn("THINK", serialize_prompt(self.tokenizer, item, chat_settings)[0])
        actual = self.record(contract=self.contract(mode="chat"))
        self.assertIsNone(actual["error"])
        self.assertIn("NO_THINK", actual["serialized_prompt"])
        item["messages"] = [{"role": "assistant", "content": "leaked answer"}]
        with self.assertRaises(ValueError):
            serialize_prompt(self.tokenizer, item, chat_settings)

    def test_greedy_ties_and_behavior_transformations_match_contract(self):
        d = settings(decoding="greedy")["decoding"]
        choice = choose_token(torch.tensor([1., 1., 0.]), d, torch.Generator().manual_seed(1))
        self.assertEqual(choice[0], 0)
        self.assertEqual(choice[1], 0.)
        d = dict(d, mode="sample", temperature=.5, top_k=2, top_p=.9)
        choice = choose_token(torch.log(torch.tensor([.55, .30, .15])), d, torch.Generator().manual_seed(1))
        self.assertIn(choice[0], (0, 1))
        expected = [.55**2 / (.55**2 + .3**2), .3**2 / (.55**2 + .3**2)][choice[0]]
        self.assertAlmostEqual(math_exp(choice[1]), expected, places=7)
        self.assertEqual(choice[3], 2)

    def test_unknown_invalid_and_silently_ignored_settings_rejected(self):
        for change in ({"temperature": 0}, {"seed": True}, {"top_p": 1.1}, {"top_k": 0}):
            s = settings(); s["decoding"].update(change)
            with self.assertRaises(ValueError): validate_settings(s, require_interface=False)
        for key, value in (("device", "auto"), ("dtype", "float16"), ("samples", True), ("unknown_flag", 1)):
            s = settings(); s["generation"][key] = value
            with self.assertRaises(ValueError): validate_settings(s, require_interface=False)
        s = settings(decoding="greedy"); s["decoding"]["temperature"] = .7
        with self.assertRaisesRegex(ValueError, "silently ignore"):
            validate_settings(s, require_interface=False)
        s = settings(); s["thinking_mode"] = "enabled"
        with self.assertRaises(ValueError): validate_settings(s, require_interface=False)
        s = settings(mode="chat"); s["generation"]["template"] = "{{ messages }}"
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            validate_settings(s, require_interface=False)

    def test_stop_action_retained_but_only_terminal_stop_excluded_from_scoring(self):
        contract = self.contract(decoding="greedy")
        row = self.record(model=ScriptedForward([9, 1]), contract=contract)
        self.assertEqual(row["token_ids"], [9, 1])
        self.assertIn("[EOS]", row["raw_response"])
        self.assertEqual(row["response_text"], "4")
        self.assertEqual(row["cost"]["generation_tokens"], 2)
        grade = grade_response(items()[0], row)
        self.assertTrue(grade["task_success"])
        self.assertTrue(grade["natural_termination"])
        turn = self.record(model=ScriptedForward([3]), contract=contract)
        self.assertEqual(turn["stop_reason"], "turn_stop")
        self.assertEqual(turn["token_ids"], [3])
        self.assertTrue(grade_response(items()[0], turn)["natural_termination"])

    def test_prompt_eos_padding_and_context_cap_do_not_fabricate_stopping(self):
        contract = self.contract(decoding="greedy", cap=2)
        item = dict(items()[0], prompt="[EOS] hello")
        row = self.record(model=ScriptedForward([8]), contract=contract, item=item)
        self.assertEqual(row["stop_reason"], "max_tokens")
        self.assertEqual(row["token_ids"], [8, 8])
        self.assertEqual(row["response_text"], row["raw_response"])
        context = self.contract(decoding="greedy", context=5)
        row = self.record(model=ScriptedForward([9]), contract=context)
        self.assertEqual(row["stop_reason"], "context_limit")
        self.assertEqual(row["generated_tokens"], 1)
        self.assertTrue(row["truncated"])
        exhausted = self.contract(context=3)
        row = self.record(contract=exhausted)
        self.assertEqual(row["generated_tokens"], 0)
        self.assertEqual(row["cost"]["forward_calls"], 0)
        self.assertEqual(row["stop_reason"], "context_limit")

    def test_partial_failure_interrupt_and_nonfinite_logits_are_retained(self):
        contract = self.contract(decoding="greedy")
        for failure in (RuntimeError, KeyboardInterrupt):
            partial = []
            row = self.record(model=ScriptedForward([9], failure), contract=contract, progress=partial.append)
            self.assertEqual(row["stop_reason"], "error")
            self.assertEqual(row["token_ids"], [9])
            self.assertEqual(row["cost"]["forward_calls"], 1)
            self.assertEqual(row["cost"]["attempted_forward_calls"], 2)
            self.assertEqual(len(partial), 1)
            self.assertEqual(row["interrupted"], failure is KeyboardInterrupt)
        with self.assertRaises(ValueError):
            choose_token(torch.tensor([0., float("nan")]), settings()["decoding"], torch.Generator())

    def test_oversized_raw_output_is_kept_but_grader_is_bounded(self):
        row = self.record(model=ScriptedForward([9]), contract=self.contract(decoding="greedy"))
        row["raw_response"] = row["response_text"] = "x" * 16385
        for key in ("raw_response", "response_text"):
            row[key + "_sha256"] = hashlib.sha256(row[key].encode()).hexdigest()
        validate_record(row, self.contract(decoding="greedy"), items()[0])
        self.assertEqual(grade_response(items()[0], row)["status"], "INVALID")

    def _inputs(self, directory, *, contract=None):
        path = Path(directory)
        suite, frozen = path / "items.json", path / "contract.json"
        suite.write_text(json.dumps(items()))
        frozen.write_text(json.dumps(contract or self.contract()))
        return suite, frozen

    def test_actual_local_cli_pipeline_new_only_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, contract = self._inputs(directory)
            output = Path(directory) / "run"
            summary = run_generation(root=ROOT, items_path=suite, contract_path=contract,
                checkpoint=self.checkpoint, output=output)
            self.assertEqual(summary["status"], "completed")
            rows = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["input_sha256"] for row in rows))
            self.assertEqual(json.loads((output / "evaluation.json").read_text())["mode"], "offline saved-response replay")
            with self.assertRaises(FileExistsError):
                run_generation(root=ROOT, items_path=suite, contract_path=contract, checkpoint=self.checkpoint, output=output)
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 4)

    def test_same_size_tokenizer_permutation_and_template_mismatch_before_weight_load(self):
        from transformers import AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            altered = path / "altered"; altered.mkdir()
            _, tokenizer = tiny_fixture(altered, permuted=True)
            suite, contract = self._inputs(path)
            with patch.object(AutoModelForCausalLM, "from_pretrained") as loading:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, tokenizer_path=altered, output=path / "bad-tokenizer")
                loading.assert_not_called()
            # Freezing the wrong external tokenizer cannot bypass the saved
            # model tokenizer check by making expected==observed externally.
            external_contract = freeze_local_contract(items(), settings(), tokenizer)
            _, contract = self._inputs(path, contract=external_contract)
            with patch.object(AutoModelForCausalLM, "from_pretrained") as loading:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, tokenizer_path=altered, output=path / "bad-saved-mapping")
                loading.assert_not_called()
            changed = self.contract(mode="chat")
            changed["settings"]["generation"]["template"] += " changed"
            changed["identity"] = canonical_hash({k: v for k, v in changed.items() if k != "identity"})
            _, contract = self._inputs(path, contract=changed)
            with patch.object(AutoModelForCausalLM, "from_pretrained") as loading:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, output=path / "bad-template")
                loading.assert_not_called()

    def test_load_failure_and_interrupt_journal_keep_actual_stages_and_partial_rows(self):
        from transformers import AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); suite, contract = self._inputs(path, contract=self.contract(decoding="greedy"))
            with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=RuntimeError("declared load failure")):
                with self.assertRaises(RuntimeError):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, output=path / "load-failure")
            failure = json.loads((path / "load-failure/failure.json").read_text())
            self.assertEqual(failure["stage"], "load_model")
            self.assertTrue((path / "load-failure/input-identity.json").is_file())
            class Interrupted(ScriptedForward):
                config = self.model.config
                def to(self, device): return self
                def parameters(self): return self_outer.model.parameters()
                def get_input_embeddings(self): return self_outer.model.get_input_embeddings()
                def get_output_embeddings(self): return self_outer.model.get_output_embeddings()
            self_outer = self
            with patch.object(AutoModelForCausalLM, "from_pretrained", return_value=Interrupted([9], KeyboardInterrupt)):
                with self.assertRaises(KeyboardInterrupt):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, output=path / "interrupt")
            events = [json.loads(line) for line in (path / "interrupt/events.jsonl").read_text().splitlines()]
            rows = [json.loads(line) for line in (path / "interrupt/responses.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["stage"], "interrupted")
            self.assertTrue(any(event["stage"] == "partial_response" for event in events))
            self.assertEqual(rows[0]["token_ids"], [9])
            self.assertEqual(rows[0]["stop_reason"], "error")
            self.assertEqual(len(json.loads((path / "interrupt/failure.json").read_text())["not_attempted"]), 3)

    def test_cli_freeze_is_tokenizer_only_and_existing_contract_is_protected(self):
        spec = importlib.util.spec_from_file_location("generation_cli", ROOT / "scripts/generate_reasoning_records.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        from transformers import AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); suite = path / "items.json"; suite.write_text(json.dumps(items()))
            config = path / "settings.json"; config.write_text(json.dumps(settings()))
            contract = path / "contract.json"
            args = ["--items", str(suite), "--checkpoint", str(self.checkpoint), "--settings", str(config), "--freeze-contract", str(contract)]
            with patch.object(AutoModelForCausalLM, "from_pretrained") as loading:
                cli.main(args); loading.assert_not_called()
            frozen = json.loads(contract.read_text())
            self.assertEqual(frozen["settings"], self.contract()["settings"])
            self.assertEqual(frozen["preparation_inputs"]["settings_sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError): cli.main(args)

    def test_input_mutation_and_declared_model_context_are_rejected(self):
        from dongxi_llms import reasoning_generation as adapter
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); suite, contract = self._inputs(path)
            original = adapter.generate_record
            changed = False
            def mutate(*args, **kwargs):
                nonlocal changed
                row = original(*args, **kwargs)
                if not changed:
                    suite.write_text(suite.read_text() + "\n")
                    changed = True
                return row
            with patch.object(adapter, "generate_record", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                    run_generation(root=ROOT, items_path=suite, contract_path=contract,
                        checkpoint=self.checkpoint, output=path / "mutated")
            self.assertEqual(len((path / "mutated/responses.jsonl").read_text().splitlines()), 4)
            self.assertEqual(json.loads((path / "mutated/failure.json").read_text())["stage"], "verify_unchanged_inputs")
            suite, contract = self._inputs(path, contract=self.contract(context=33))
            with self.assertRaisesRegex(ValueError, "actual model capacity"):
                run_generation(root=ROOT, items_path=suite, contract_path=contract,
                    checkpoint=self.checkpoint, output=path / "too-long")

    def test_deadline_and_record_tampering_do_not_become_valid_evidence(self):
        row = self.record(deadline=0.)
        self.assertEqual(row["error_stage"], "deadline_boundary")
        self.assertEqual(row["generated_tokens"], 0)
        self.assertEqual(row["cost"]["attempted_forward_tokens"], 0)
        contract = self.contract(decoding="greedy")
        row = self.record(model=ScriptedForward([9, 1]), contract=contract)
        for change in ({"raw_response": "invented raw"}, {"response_text": "invented answer"},
                       {"stop_token_id": 3}, {"checkpoint_interface_sha256": "0" * 64},
                       {"settings_sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_record(dict(row, **change), contract, items()[0])


def math_exp(value):
    import math
    return math.exp(value)


if __name__ == "__main__":
    unittest.main()

"""Original random local HF controls for the production DPO update function.

No Hub weights, CUDA, quality assertion, installation or external acquisition.
The fixed fixture was authored before its first update. Helpers also serve the
dated evidence collector and a separately launched clean process.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.training_snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dongxi_production_dpo", ROOT/"scripts/run_chapter11_spark_dpo.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SEED, LIMIT = 1818, 16*1024*1024
CONFIG = dict(vocab_size=16, hidden_size=16, intermediate_size=32,
              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
              head_dim=8, max_position_embeddings=32, attention_dropout=0.,
              bos_token_id=7, eos_token_id=1, pad_token_id=1)
# Masks refer to token positions, before the single causal shift. EOS1 is a
# supervised termination when real, but never a target when collate adds padding.
ENCODED = [
    [([7, 3, 9, 1], [False, False, True, True]),
     ([7, 3, 10, 11, 1], [False, False, True, True, True])],
    [([7, 4, 12, 9, 1], [False, False, True, True, True]),
     ([7, 4, 10, 1], [False, False, True, True])],
    [([7, 5, 9, 12, 1], [False, False, True, True, True]),
     ([7, 5, 11, 10, 1], [False, False, True, True, True])],
]
VALID = [[([7, 6, 9, 1], [False, False, True, True]),
          ([7, 6, 10, 1], [False, False, True, True])]]
FIXTURE = dict(schema="dongxi-dpo-recovery-fixture-v1", config=CONFIG, encoded_train=ENCODED,
               encoded_validation=VALID, evaluation_prefixes=[[7, 8]],
               seed=SEED, updates=6, accumulation=2, beta=.2, lr=.008,
               checkpoint_every=3, pad_id=1, max_bytes=LIMIT)


def tokenizer_fixture():
    vocabulary = {token: number for number, token in enumerate(
        ["<UNK>", "<END>", "<USER>", "<ASSISTANT>", "red", "blue", "cup", "key",
         "bag", "good", "bad", "long", "answer", "book", "box", "yes"])}
    core = Tokenizer(WordLevel(vocabulary, unk_token="<UNK>"))
    core.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<UNK>",
        eos_token="<END>", pad_token="<END>", additional_special_tokens=["<USER>", "<ASSISTANT>"])
    tokenizer.chat_template = "{% for m in messages %}{{ '<USER> ' if m['role']=='user' else '<ASSISTANT> ' }}{{ m['content'] }}{{ ' <END> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<ASSISTANT> ' }}{% endif %}"
    return tokenizer


def build_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(Qwen3Config(**CONFIG))
    reference = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.01)
    sampler = torch.Generator().manual_seed(SEED)
    source_names = ["scripts/run_chapter11_spark_dpo.py", "src/dongxi_llms/training_snapshot.py",
                    "src/dongxi_llms/dpo_lab.py", "src/dongxi_llms/batched_cache_lab.py"]
    tokenizer = tokenizer_fixture()
    options = dict(parent_files={"authored-random-parent": digest(reference.state_dict())},
        inputs={"authored_fixture": digest(FIXTURE)},
        sources={name: runner.file_digest(ROOT/name) for name in source_names},
        environment={"python": sys.version, "torch": str(torch.__version__),
                     "transformers": __import__("transformers").__version__,
                     "lock_sha256": runner.file_digest(ROOT/"uv.lock")},
        interface={"tokenizer_json_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                   "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                   "eos": 1, "pad": 1, "stops": [1]},
        encoded_train=ENCODED, encoded_valid=VALID, evaluation_prefixes=[[7, 8]],
        seed=SEED, updates=6, accumulation=2, beta=.2, lr=.008,
        max_length=32, max_new_tokens=4, device={"mode": "cpu", "dtype": "float32"})
    contract = runner.make_recovery_contract(model=model, reference=reference, optimizer=optimizer, **options)
    return model, reference, optimizer, sampler, contract, options


def train_fixture(output, components, *, initial=None, before_training=None):
    model, reference, optimizer, sampler, contract, _ = components
    initial = initial or {}
    return runner.train_completed_updates(model, reference, optimizer, sampler, ENCODED,
        contract=contract, output=output, invocation_id="original-cpu-fixture",
        pad_id=1, device="cpu", checkpoint_every=3, max_bytes=LIMIT,
        before_training=before_training, **initial)


def checkpoint_receipt(path):
    return dict(path=str(Path(path).resolve()), **json.loads(Path(str(path)+".commit.json").read_text()))


def payload_for(receipt, contract):
    return load_snapshot(receipt["path"], expected_sha256=receipt["payload_sha256"],
        expected_bytes=receipt["payload_bytes"], expected_contract=contract,
        max_bytes=LIMIT, validate_payload=lambda value: runner.validate_dpo_payload(value, contract))


def restore_fixture(components, receipt):
    model, reference, optimizer, sampler, contract, _ = components
    completed, counters, history = runner.restore_dpo_payload(payload_for(receipt, contract),
        contract=contract, model=model, reference=reference, optimizer=optimizer, sampler=sampler)
    return dict(completed=completed, counters=counters, history=history, parent_checkpoint=receipt)


def numerical_state(payload):
    return {key: value for key, value in payload["state"].items() if key != "resume_parent"}


def fresh_child(receipt_file, contract_file, output):
    """Run only when explicitly requested by the bounded parent test/collector."""
    components = build_fixture()
    retained = json.loads(Path(contract_file).read_text())
    if components[4] != retained:
        raise ValueError("Fresh process actual science contract changed")
    receipt = json.loads(Path(receipt_file).read_text())
    initial = restore_fixture(components, receipt)
    Path(output).mkdir()
    result = train_fixture(output, components, initial=initial)
    runner.write_exclusive_json(Path(output)/"result.json", result)


class ActualDPORecovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dongxi-dpo-recovery-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def directory(self, name):
        result = self.root/name
        result.mkdir()
        return result

    def trained(self, name="full"):
        components = build_fixture()
        result = train_fixture(self.directory(name), components)
        return components, result, payload_for(result["checkpoint"], components[4])

    def interrupted(self):
        components, output = build_fixture(), self.directory("interrupted")
        real_append = runner.append_metric
        def fail_after_commit(path, row):
            if row.get("update") == 3:
                raise OSError("authored metric interruption after committed update3")
            return real_append(path, row)
        with patch.object(runner, "append_metric", side_effect=fail_after_commit), self.assertRaises(OSError):
            train_fixture(output, components)
        return components, checkpoint_receipt(output/"checkpoints/completed-000003.pt")

    def test_actual_six_updates_same_process_recovery_exact(self):
        _, full, expected = self.trained()
        original, receipt = self.interrupted()
        resumed = build_fixture()
        self.assertEqual(original[4], resumed[4])
        result = train_fixture(self.directory("resumed"), resumed, initial=restore_fixture(resumed, receipt))
        actual = payload_for(result["checkpoint"], resumed[4])
        self.assertEqual(digest(numerical_state(expected)), digest(numerical_state(actual)))
        self.assertEqual(full["history"][3], result["history"][3])  # next draws AND next loss
        self.assertEqual(result["counters"]["sampled_pairs"], 12)
        self.assertEqual(result["counters"]["policy_forward_calls"], 24)
        self.assertEqual(result["counters"]["reference_forward_calls"], 24)
        self.assertNotEqual(digest(actual["state"]["policy"]), digest(actual["state"]["reference"]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in resumed[1].parameters()))
        recovered = (self.root/"resumed/recovered-metrics.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(row)["update"] for row in recovered], [1, 2, 3])

    def test_fresh_process_recovery_exact(self):
        _, _, expected = self.trained()
        components, receipt = self.interrupted()
        runner.write_exclusive_json(self.root/"receipt.json", receipt)
        runner.write_exclusive_json(self.root/"contract.json", components[4])
        command = [sys.executable, str(Path(__file__).resolve()), "--fresh-replay",
                   str(self.root/"receipt.json"), str(self.root/"contract.json"), str(self.root/"fresh")]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        result = json.loads((self.root/"fresh/result.json").read_text())
        actual = payload_for(result["checkpoint"], components[4])
        self.assertEqual(digest(numerical_state(expected)), digest(numerical_state(actual)))

    def test_unchanged_objective_against_independent_one_update(self):
        actual = build_fixture()
        row = runner.completed_dpo_update(*actual[:4], ENCODED, pad_id=1, accumulation=2,
            beta=.2, device="cpu", update=1)
        independent = build_fixture()
        model, reference, optimizer, sampler = independent[:4]
        optimizer.zero_grad(set_to_none=True)
        losses, indices = [], []
        for _ in range(2):
            index = int(torch.randint(3, (), generator=sampler)); indices.append(index)
            values = []
            for network in (model, reference):
                scores = []
                for ids, mask in ENCODED[index]:
                    tokens = torch.tensor([ids])
                    logits = network(input_ids=tokens[:, :-1], attention_mask=torch.ones_like(tokens[:, :-1]), use_cache=False).logits
                    selected = logits.float().log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
                    scores.append(selected[:, torch.tensor(mask[1:])].sum())
                values.append(scores)
            margin = .2*(values[0][0]-values[0][1]-values[1][0].detach()+values[1][1].detach())
            loss = torch.nn.functional.softplus(-margin)
            (loss/2).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        self.assertEqual(row["indices"], indices)
        self.assertEqual(row["loss"], sum(losses)/2)
        self.assertEqual(digest(actual[0].state_dict()), digest(model.state_dict()))
        self.assertEqual(digest(actual[2].state_dict()), digest(optimizer.state_dict()))

    def test_eos_prompt_and_padding_masks_one_shift(self):
        ids, attention, mask = runner.collate(ENCODED[:2], 0, 1, "cpu")
        self.assertEqual(ids[0, -1].item(), 1)
        self.assertFalse(mask[0, -1])  # EOS-valued padding
        self.assertTrue(mask[0, -2])   # actual EOS
        self.assertFalse(mask[:, :2].any())
        logits = torch.zeros(2, ids.shape[1]-1, 16)
        from dongxi_llms.dpo_lab import sequence_logps
        score = sequence_logps(logits, ids[:, 1:], mask[:, 1:] & attention[:, 1:].bool())
        self.assertTrue(torch.allclose(score, -torch.tensor([2., 3.])*torch.log(torch.tensor(16.))))

    def test_baseline_randomness_cannot_change_training(self):
        _, _, expected = self.trained()
        components = build_fixture()
        actual = train_fixture(self.directory("baseline"), components, before_training=lambda: torch.rand(17).tolist())
        self.assertEqual(digest(numerical_state(expected)), digest(numerical_state(payload_for(actual["checkpoint"], components[4]))))

    def test_preapply_semantic_tampering_matrix(self):
        components, _, original = self.trained()
        mutations = {
            "bool completed": lambda p: p["state"].update(completed=True),
            "bool update": lambda p: p["state"]["history"][0].update(update=True),
            "bool index": lambda p: p["state"]["history"][0]["indices"].__setitem__(0, True),
            "bool row work": lambda p: p["state"]["history"][0]["work"].update(sampled_pairs=True),
            "bool counter": lambda p: p["state"]["counters"].update(sampled_pairs=True),
            "negative loss": lambda p: p["state"]["history"][0].update(loss=-1.),
            "negative norm": lambda p: p["state"]["history"][0].update(gradient_norm=-1.),
            "wrong draw": lambda p: p["state"]["history"][0]["indices"].__setitem__(0, (p["state"]["history"][0]["indices"][0]+1)%3),
            "wrong work": lambda p: p["state"]["counters"].update(chosen_targets=999),
            "wrong original reference": lambda p: next(iter(p["state"]["reference"].values())).add_(.01),
            "wrong sampler": lambda p: p["state"].update(sampler_rng=torch.Generator().manual_seed(999).get_state()),
            "wrong CPU RNG bytes": lambda p: p["state"].update(torch_rng=torch.zeros(9, dtype=torch.uint8)),
            "bool parameter id": lambda p: p["state"]["optimizer"]["param_groups"][0]["params"].__setitem__(0, False),
            "step rank": lambda p: p["state"]["optimizer"]["state"][0].update(step=torch.tensor([6.])),
            "step dtype": lambda p: p["state"]["optimizer"]["state"][0].update(step=torch.tensor(6, dtype=torch.int64)),
            "negative second moment": lambda p: p["state"]["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1.),
            "shape": lambda p: p["state"]["policy"].update({next(iter(p["state"]["policy"])): torch.zeros(1)}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = deepcopy(original); mutate(payload)
                with patch.object(components[0], "load_state_dict") as apply, self.assertRaises(ValueError):
                    runner.restore_dpo_payload(payload, contract=components[4], model=components[0],
                        reference=components[1], optimizer=components[2], sampler=components[3])
                apply.assert_not_called()

    def test_cuda_rng_layout_rejected_before_any_application_without_gpu(self):
        components, _, payload = self.trained()
        contract = deepcopy(components[4]); contract.update(cuda_rng_count=1, cuda_rng_bytes=[5056])
        payload["state"]["cuda_rng"] = [torch.zeros(8, dtype=torch.uint8)]
        with patch.object(components[0], "load_state_dict") as apply, self.assertRaises(ValueError):
            runner.restore_dpo_payload(payload, contract=contract, model=components[0], reference=components[1],
                optimizer=components[2], sampler=components[3])
        apply.assert_not_called()

    def test_observable_pre_model_identity_and_latent_effective_config(self):
        *_, contract, options = build_fixture()
        observed = runner.observable_recovery_contract(**options)
        runner.verify_observable_contract(contract, observed)
        for key in ("input_hashes", "source_hashes", "environment", "parent_files", "checkpoint_interface", "updates", "encoded_train_sha256"):
            modified = deepcopy(observed); modified[key] = "authored mismatch"
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify_observable_contract(contract, modified)
        components = build_fixture(); components[0].config.max_position_embeddings *= 2
        changed = runner.make_recovery_contract(model=components[0], reference=components[1], optimizer=components[2], **components[5])
        self.assertNotEqual(contract, changed)
        self.assertEqual(contract["policy_shapes"], changed["policy_shapes"])
        self.assertTrue(contract["policy_model"]["trainable_parameters"])
        self.assertEqual(contract["reference_model"]["trainable_parameters"], [])

    def test_non_completed_phase_cannot_apply_state(self):
        components, _, payload = self.trained()
        payload["phase"] = "pending-rollout"
        with patch.object(components[0], "load_state_dict") as apply, self.assertRaises(ValueError):
            runner.restore_dpo_payload(payload, contract=components[4], model=components[0], reference=components[1],
                optimizer=components[2], sampler=components[3])
        apply.assert_not_called()

    def test_live_reference_cannot_enter_actual_update(self):
        components = build_fixture(); components[1].requires_grad_(True)
        with self.assertRaises(ValueError):
            runner.completed_dpo_update(*components[:4], ENCODED, pad_id=1, accumulation=2,
                beta=.2, device="cpu", update=1)
        self.assertEqual(components[2].state_dict()["state"], {})

    def test_functional_attention_dropout_not_hidden_by_tensor_shapes(self):
        components = build_fixture()
        components[0].model.layers[0].self_attn.attention_dropout = .1
        with self.assertRaises(ValueError):
            runner.effective_model_contract(components[0])
        components = build_fixture(); components[0].config.attention_dropout = .1
        with self.assertRaises(ValueError):
            runner.effective_model_contract(components[0])

    def test_failed_next_save_preserves_previous_boundary(self):
        components, output = build_fixture(), self.directory("save-failure")
        real_save = runner.save_snapshot
        def fail_final(path, **kwargs):
            if kwargs["completed_updates"] == 6:
                raise OSError("authored final save failure")
            return real_save(path, **kwargs)
        with patch.object(runner, "save_snapshot", side_effect=fail_final), self.assertRaises(OSError):
            train_fixture(output, components)
        prior = payload_for(checkpoint_receipt(output/"checkpoints/completed-000003.pt"), components[4])
        self.assertEqual(prior["completed_updates"], 3)
        self.assertFalse((output/"checkpoints/completed-000006.pt.commit.json").exists())

    def test_initial_commit_precedes_failed_baseline(self):
        components, output = build_fixture(), self.directory("baseline-failure")
        def failure():
            raise RuntimeError("authored baseline failure")
        with self.assertRaises(RuntimeError):
            train_fixture(output, components, before_training=failure)
        initial = payload_for(checkpoint_receipt(output/"checkpoints/completed-000000.pt"), components[4])
        self.assertEqual(initial["completed_updates"], 0)
        self.assertEqual(initial["state"]["optimizer"]["state"], {})

    def test_evaluation_export_failure_keeps_final_recovery(self):
        components, result, _ = self.trained()
        for stage in ("evaluation", "export"):
            calls = []
            def evaluate():
                calls.append("evaluation")
                if stage == "evaluation":
                    raise RuntimeError("authored evaluation failure")
                return {}
            def export():
                calls.append("export")
                raise RuntimeError("authored export failure")
            with self.subTest(stage=stage), self.assertRaises(RuntimeError):
                runner.finalize_after_commit(result, evaluate=evaluate, export=export)
            self.assertEqual(calls, ["evaluation"] if stage == "evaluation" else ["evaluation", "export"])
            self.assertEqual(payload_for(result["checkpoint"], components[4])["completed_updates"], 6)
        with self.assertRaises(ValueError):
            runner.finalize_after_commit(dict(result, checkpoint={"completed_updates": 3}), evaluate=lambda: None, export=lambda: None)

    def test_metric_failure_snapshot_history_authoritative(self):
        components, receipt = self.interrupted()
        payload = payload_for(receipt, components[4])
        self.assertEqual([row["update"] for row in payload["state"]["history"]], [1, 2, 3])
        metrics = (self.root/"interrupted/metrics.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(row)["update"] for row in metrics], [1, 2])

    def test_attempted_forward_failure_work_not_claimed_completed(self):
        components = build_fixture(); attempted = runner.zero_work()
        with patch.object(components[0], "forward", side_effect=RuntimeError("authored model failure")), self.assertRaises(RuntimeError):
            runner.completed_dpo_update(*components[:4], ENCODED, pad_id=1, accumulation=2, beta=.2,
                device="cpu", update=1, attempted_work=attempted)
        self.assertEqual(attempted["sampled_pairs"], 1)
        self.assertEqual(attempted["policy_forward_calls"], 1)
        self.assertEqual(attempted["reference_forward_calls"], 0)
        self.assertEqual(components[2].state_dict()["state"], {})

    def test_existing_checkpoint_output_not_overwritten(self):
        components, result, _ = self.trained()
        original = runner.file_digest(result["checkpoint"]["path"])
        with self.assertRaises(FileExistsError):
            train_fixture(self.root/"full", components)
        self.assertEqual(original, runner.file_digest(result["checkpoint"]["path"]))
        with self.assertRaises(FileExistsError):
            runner.write_exclusive_json(self.root/"full/metrics.jsonl", {})

    def test_external_digest_size_contract_tamper_rejected(self):
        components, result, _ = self.trained()
        receipt, contract = result["checkpoint"], components[4]
        for changes in ({"payload_sha256": "0"*64}, {"payload_bytes": receipt["payload_bytes"]+1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                payload_for(dict(receipt, **changes), contract)
        changed = dict(contract, beta=.3)
        with self.assertRaises(ValueError):
            payload_for(receipt, changed)

    def test_actual_local_tokenizer_masks_and_hf_return_dict_false(self):
        tokenizer = tokenizer_fixture()
        row = dict(id="authored", prompt=[{"role": "user", "content": "red"}], chosen="good", rejected="bad long")
        prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False)
        encoded = runner.encode_pair(tokenizer, row, 32)
        self.assertTrue(all(type(token) is int for token in prefix))
        self.assertEqual(encoded[0][0][:len(prefix)], prefix)
        self.assertEqual(encoded[0][0][-1], 1)
        self.assertFalse(any(encoded[0][1][:len(prefix)]))
        self.assertTrue(all(encoded[0][1][len(prefix):]))
        self.assertNotEqual(len(encoded[0][0]), len(encoded[1][0]))
        for altered in (dict(row, prompt=[]), dict(row, chosen="")):
            with self.assertRaises(ValueError):
                runner.encode_pair(tokenizer, altered, 32)

    def test_source_groups_encoded_collisions_and_honest_legacy_gap(self):
        tokenizer = tokenizer_fixture()
        splits = [[dict(id=str(i), prompt=[{"role": "user", "content": text}], group=group)]
                  for i, (text, group) in enumerate((("red", "train-source"), ("blue", "validation-source"), ("cup", "test-source")))]
        self.assertEqual(runner.check_split_interfaces(tokenizer, splits, [[], [], []])["source_groups"], "fully-provided-disjoint")
        missing = deepcopy(splits); del missing[0][0]["group"]
        self.assertIn("pending", runner.check_split_interfaces(tokenizer, missing, [[], [], []])["source_groups"])
        duplicate_group = deepcopy(splits); duplicate_group[1][0]["group"] = "train-source"
        with self.assertRaises(ValueError):
            runner.check_split_interfaces(tokenizer, duplicate_group, [[], [], []])
        alias = deepcopy(splits); alias[1][0]["prompt"][0]["content"] = "  red  "
        with self.assertRaises(ValueError):
            runner.check_split_interfaces(tokenizer, alias, [[], [], []])

    def test_closure_actual_source_input_lock_and_parent_rehash(self):
        from dongxi_llms.run_identity import artifact_hashes
        checkpoint = self.directory("parent"); (checkpoint/"config.json").write_text('{"original":true}')
        for name in ("source.py", "input.jsonl", "lock.txt"):
            (self.root/name).write_text("authored bytes")
        identity = dict(source_sha256={"source.py": runner.file_digest(self.root/"source.py")},
            input_sha256={"input.jsonl": runner.file_digest(self.root/"input.jsonl")},
            environment={"environment_lock": {"path": str(self.root/"lock.txt"), "sha256": runner.file_digest(self.root/"lock.txt")}},
            checkpoint_path=str(checkpoint), checkpoint_files=artifact_hashes(checkpoint))
        runner.verify_identity_files(identity, self.root)
        for path in (self.root/"source.py", self.root/"input.jsonl", self.root/"lock.txt", checkpoint/"config.json"):
            saved = path.read_bytes(); path.write_bytes(saved+b" changed")
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.verify_identity_files(identity, self.root)
            path.write_bytes(saved)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--fresh-replay":
        fresh_child(*sys.argv[2:])
    else:
        unittest.main()

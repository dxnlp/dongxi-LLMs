"""Original tiny local HF controls; no pretrained acquisition or GPU work."""
from copy import deepcopy
import hashlib
import json
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
from dongxi_llms.chosen_sft_control import (ChosenSFTLoop, chosen_record, compare_results,
    encode_dataset, native_runners, recipe, run_arm)
from dongxi_llms.run_identity import artifact_hashes, tokenizer_interface

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"fixtures/matched-chosen-sft/protocol.json"
TOKENIZER_ID = "course-authored-location-wordlevel"
REVISION = "0"*40


def fixture_rows():
    protocol = json.loads(PROTOCOL.read_text())
    splits = []
    for kind in ("train", "validation", "evaluation"):
        rows = [json.loads(line) for line in (ROOT/protocol["inputs"][kind]).read_text().splitlines() if line.strip()]
        splits.append([dict(row, group=protocol["groups"][row["id"]]) for row in rows])
    return protocol, splits


def tiny_tokenizer(splits):
    # Fixed inventory of predeclared literal fixture atoms, not a trained BPE.
    atoms = set()
    for rows in splits:
        for row in rows:
            for message in row["prompt"]:
                atoms.update(message["content"].split())
            for key in ("chosen", "rejected", "expected"):
                if key in row:
                    atoms.update(row[key].split())
    tokens = ["<UNK>", "<END>", "<USER>", "<ASSISTANT>", *sorted(atoms)]
    core = Tokenizer(WordLevel({token:i for i,token in enumerate(tokens)}, unk_token="<UNK>"))
    core.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<UNK>",
        eos_token="<END>", pad_token="<END>", additional_special_tokens=["<USER>", "<ASSISTANT>"])
    tokenizer.chat_template = "{% for m in messages %}{{ '<USER> ' if m['role']=='user' else '<ASSISTANT> ' }}{{ m['content'] }}{{ ' <END> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<ASSISTANT> ' }}{% endif %}"
    return tokenizer


def fixture(seed=1818, updates=1):
    torch.set_num_threads(1)
    protocol, splits = fixture_rows()
    tokenizer = tiny_tokenizer(splits)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),
            **protocol["model"], bos_token_id=2, eos_token_id=1, pad_token_id=1))
    model.config._attn_implementation = "sdpa"
    dataset = encode_dataset(tokenizer, *splits, max_length=64)
    settings = recipe(seed=seed, updates=updates, accumulation=2,
        learning_rate=.008, beta=.2, max_length=64, max_new_tokens=8)
    return model, tokenizer, dataset, settings


def save_parent(path, model, tokenizer):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    interface = tokenizer_interface(tokenizer, template=tokenizer.chat_template,
        stop_ids=[1], tokenizer_id=TOKENIZER_ID, tokenizer_revision=REVISION)
    (Path(path)/"course-genealogy.json").write_text(json.dumps(dict(
        kind="original-random-local-fixture", checkpoint_interface=interface,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())))
    return interface


class ChosenControlTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.tokenizer, self.dataset, self.settings = fixture()

    def loop(self, model=None, encoded=None):
        model = model or deepcopy(self.parent)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.01)
        sampler = torch.Generator().manual_seed(1818)
        loop = ChosenSFTLoop(model, optimizer, sampler, encoded or self.dataset["train"],
            pad_id=1, stop_ids=[1], accumulation=2, updates=1)
        return loop

    def test_original_records_and_native_encoding(self):
        protocol, splits = fixture_rows()
        self.assertEqual([len(rows) for rows in splits], [8, 4, 4])
        _, dpo = native_runners()
        for row, pair in zip(splits[0], self.dataset["train"]):
            self.assertEqual(pair, dpo.encode_pair(self.tokenizer, row, 64))
            record = chosen_record(pair, row["id"])
            self.assertEqual(record["ids"], pair[0][0])
            self.assertEqual(record["labels"], [i if valid else -100 for i,valid in zip(*pair[0])])
        self.assertEqual(self.dataset["split"]["source_groups"], "fully-provided-disjoint")

    def test_manual_ce_gradient_and_adam_parity(self):
        actual = self.loop()
        manual = deepcopy(self.parent)
        optimizer = torch.optim.AdamW(manual.parameters(), lr=.008, weight_decay=.01)
        generator = torch.Generator().manual_seed(1818)
        indices = [int(torch.randint(len(self.dataset["train"]), (), generator=generator)) for _ in range(2)]
        count = sum(sum(self.dataset["train"][i][0][1][1:]) for i in indices)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in indices:
            tokens, mask = self.dataset["train"][index][0]
            ids = torch.tensor([tokens])
            logits = manual(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :-1].float()
            labels = torch.tensor([tokens[1:]])
            targets = torch.tensor([mask[1:]])
            loss = torch.nn.functional.cross_entropy(logits[targets], labels[targets], reduction="sum")
            (loss/count).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(manual.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = actual.completed_update()
        self.assertEqual(row["indices"], indices)
        self.assertAlmostEqual(row["loss"], sum(losses)/count, places=6)
        self.assertAlmostEqual(row["gradient_norm"], float(norm), places=6)
        for key,value in manual.state_dict().items():
            torch.testing.assert_close(actual.model.state_dict()[key], value, atol=1e-6, rtol=1e-6)
        self.assertEqual(digest(actual.optimizer.state_dict()), digest(optimizer.state_dict()))
        self.assertNotEqual(digest(actual.model.state_dict()), digest(self.parent.state_dict()))

    def test_native_dpo_update_exact_parity(self):
        model, reference = deepcopy(self.parent), deepcopy(self.parent).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.01)
        sampler = torch.Generator().manual_seed(1818)
        _, dpo = native_runners()
        native = dpo.completed_dpo_update(model, reference, optimizer, sampler, self.dataset["train"],
            pad_id=1, accumulation=2, beta=.2, device="cpu", update=1)
        trained, row = run_arm(self.parent, self.dataset, self.tokenizer, self.settings, arm="dpo")
        self.assertEqual(row["history"], [native])
        self.assertEqual(digest(trained.state_dict()), digest(model.state_dict()))
        self.assertEqual(row["sampler_sha256"], digest(sampler.get_state()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_same_draws_chosen_exposure_and_unequal_compute(self):
        results = {}
        for arm in ("unchanged", "chosen-sft", "dpo"):
            _, results[arm] = run_arm(self.parent, self.dataset, self.tokenizer, self.settings, arm=arm)
        comparison = compare_results(results)
        a, b = results["chosen-sft"]["training_work"], results["dpo"]["training_work"]
        self.assertEqual(a["chosen_targets"], b["chosen_targets"])
        self.assertEqual(a["reference_forward_calls"], 0)
        self.assertGreater(b["reference_forward_calls"], 0)
        self.assertGreater(b["policy_forward_positions"], a["policy_forward_positions"])
        self.assertEqual(results["unchanged"]["final_policy_sha256"], digest(self.parent.state_dict()))
        self.assertIn("objective reduction", comparison["unmatched"])

    def test_comparison_rejects_equal_length_different_chosen_identity(self):
        results = {}
        for arm in ("unchanged", "chosen-sft", "dpo"):
            _, results[arm] = run_arm(self.parent, self.dataset, self.tokenizer, self.settings, arm=arm)
        results["chosen-sft"]["dataset"]["chosen_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "same parent"):
            compare_results(results)

    def test_ragged_real_eos_vs_pad_mask(self):
        pair = deepcopy(self.dataset["train"][0])
        pair[0][0].insert(-1, pair[0][0][-2])
        pair[0][1].insert(-1, True)
        sft, dpo = native_runners()
        rows = [chosen_record(self.dataset["train"][0], "a"), chosen_record(pair, "b")]
        ids, labels, attention = sft.collate(rows, 1, "cpu")
        self.assertEqual(ids[0, -1].item(), 1)
        self.assertEqual(labels[0, -1].item(), -100)
        self.assertEqual(labels[0, -2].item(), 1)
        self.assertEqual(attention[0, -1].item(), 0)
        _, _, mask = dpo.collate([self.dataset["train"][0], pair], 0, 1, "cpu")
        self.assertFalse(mask[0, -1])
        self.assertTrue(mask[0, -2])

    def test_partial_forward_failure_is_poisoned_and_retained(self):
        loop = self.loop()
        original = loop.model.forward
        calls = []
        def failing(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("authored second-forward failure")
            return original(*args, **kwargs)
        with patch.object(loop.model, "forward", side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, "second-forward"):
                loop.completed_update()
        self.assertEqual(loop.completed, 0)
        self.assertEqual(loop.attempts[-1]["policy_forward_calls_entered"], 2)
        self.assertEqual(loop.attempts[-1]["policy_forward_calls_completed"], 1)
        self.assertTrue(any(p.grad is not None for p in loop.model.parameters()))
        with self.assertRaisesRegex(RuntimeError, "original parent"):
            loop.completed_update()

    def test_completed_history_precedes_evaluation_failure(self):
        events = []
        _, dpo = native_runners()
        with patch.object(dpo, "generate_dpo_panel", side_effect=RuntimeError("authored eval failure")):
            with self.assertRaisesRegex(RuntimeError, "eval failure"):
                run_arm(self.parent, self.dataset, self.tokenizer, self.settings,
                    arm="chosen-sft", row_sink=events.append)
        self.assertEqual(events[0]["phase"], "completed-update")
        self.assertEqual(events[0]["update"], 1)

    def test_failed_update_callback_contains_partial_attempt(self):
        events = []
        sft, _ = native_runners()
        with patch.object(sft, "summed_loss", side_effect=RuntimeError("authored loss failure")):
            with self.assertRaisesRegex(RuntimeError, "loss failure"):
                run_arm(self.parent, self.dataset, self.tokenizer, self.settings,
                    arm="chosen-sft", row_sink=events.append)
        self.assertEqual(events[-1]["phase"], "failed-update")
        self.assertEqual(events[-1]["policy_forward_calls_entered"], 1)
        self.assertEqual(events[-1]["policy_forward_calls_completed"], 0)

    def test_global_rng_is_not_changed(self):
        before = torch.get_rng_state().clone()
        run_arm(self.parent, self.dataset, self.tokenizer, self.settings, arm="chosen-sft")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mutated_consumed_encoding_refuses_before_model(self):
        data = deepcopy(self.dataset)
        data["train"][0][0][0][-2] += 1
        with patch.object(self.parent, "forward", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "contract changed"):
                run_arm(self.parent, data, self.tokenizer, self.settings, arm="chosen-sft")

    def test_source_group_and_raw_prompt_overlap_refuse(self):
        _, splits = fixture_rows()
        splits[1][0]["group"] = splits[0][0]["group"]
        with self.assertRaisesRegex(ValueError, "split-disjoint"):
            encode_dataset(self.tokenizer, *splits, max_length=64)
        _, splits = fixture_rows()
        splits[1][0]["prompt"] = deepcopy(splits[0][0]["prompt"])
        with self.assertRaisesRegex(ValueError, "split-disjoint"):
            encode_dataset(self.tokenizer, *splits, max_length=64)

    def test_duplicate_ids_and_missing_groups_refuse(self):
        _, splits = fixture_rows()
        splits[2][0]["id"] = splits[0][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            encode_dataset(self.tokenizer, *splits, max_length=64)
        _, splits = fixture_rows()
        del splits[0][0]["group"]
        with self.assertRaisesRegex(ValueError, "source group"):
            encode_dataset(self.tokenizer, *splits, max_length=64)

    def test_unknown_and_overlength_refuse(self):
        _, splits = fixture_rows()
        splits[0][0]["chosen"] = "literal-not-in-fixed-inventory"
        with self.assertRaisesRegex(ValueError, "Unknown-token"):
            encode_dataset(self.tokenizer, *splits, max_length=64)
        _, splits = fixture_rows()
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            encode_dataset(self.tokenizer, *splits, max_length=4)

    def test_invalid_recipe_bool_alias_and_unused_settings_refuse(self):
        for key in ("seed", "updates", "accumulation", "max_length", "max_new_tokens", "learning_rate", "beta"):
            values = {k:self.settings[k] for k in ("seed", "updates", "accumulation", "learning_rate", "beta", "max_length", "max_new_tokens")}
            values[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                recipe(**values)
        with self.assertRaisesRegex(ValueError, "ignored"):
            run_arm(self.parent, self.dataset, self.tokenizer, dict(self.settings, new_setting=True), arm="chosen-sft")

    def test_real_stop_and_boolean_masks_required(self):
        malformed = deepcopy(self.dataset["train"])
        malformed[0][0][1][-1] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.loop(encoded=malformed)
        malformed = deepcopy(self.dataset["train"])
        malformed[0][0][0][-1] = 7
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.loop(encoded=malformed)

    def test_local_parent_full_export_reload_and_interface(self):
        with tempfile.TemporaryDirectory(prefix="dongxi-chosen-local-") as temporary:
            path = Path(temporary)/"parent"
            interface = save_parent(path, self.parent, self.tokenizer)
            loaded = Qwen3ForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
            self.assertEqual(digest(loaded.state_dict()), digest(self.parent.state_dict()))
            saved = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
            _, dpo = native_runners()
            observed, adoption = dpo.verify_parent_tokenizer(path, saved, saved,
                tokenizer_id=TOKENIZER_ID, tokenizer_revision=REVISION)
            self.assertEqual(interface, observed)
            self.assertFalse(adoption["legacy_adoption"])
            (path/"chat_template.jinja").write_text(saved.chat_template+"wrong")
            with self.assertRaises(ValueError):
                dpo.verify_parent_tokenizer(path, saved, saved,
                    tokenizer_id=TOKENIZER_ID, tokenizer_revision=REVISION)

    def test_cli_real_local_control_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory(prefix="dongxi-chosen-cli-") as temporary:
            parent = Path(temporary)/"parent"
            save_parent(parent, self.parent, self.tokenizer)
            protocol, _ = fixture_rows()
            output = Path(temporary)/"control"
            command = [sys.executable, str(ROOT/"scripts/run_matched_chosen_sft.py"),
                "--checkpoint", str(parent), "--tokenizer", str(parent), "--tokenizer-id", TOKENIZER_ID,
                "--tokenizer-revision", REVISION, "--groups", str(PROTOCOL),
                "--train", str(ROOT/protocol["inputs"]["train"]),
                "--validation", str(ROOT/protocol["inputs"]["validation"]),
                "--evaluation", str(ROOT/protocol["inputs"]["evaluation"]),
                "--output", str(output), "--environment-lock", str(ROOT/"uv.lock"),
                "--arm", "chosen-sft", "--seed", "1818", "--updates", "1",
                "--accumulation", "2", "--lr", ".008", "--beta", ".2", "--max-length", "64",
                "--max-new-tokens", "8", "--max-parameters", "100000", "--runtime-seconds", "40"]
            child = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout+child.stderr)
            result = json.loads((output/"result.json").read_text())
            self.assertEqual(len(result["history"]), 1)
            self.assertEqual(len(result["publication"]), 4)
            reloaded = Qwen3ForCausalLM.from_pretrained(output/"policy", local_files_only=True, dtype=torch.float32)
            self.assertEqual(digest(reloaded.state_dict()), result["final_policy_sha256"])
            before = artifact_hashes(output/"policy")
            child = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(child.returncode, 0)
            self.assertEqual(before, artifact_hashes(output/"policy"))


if __name__ == "__main__":
    unittest.main()

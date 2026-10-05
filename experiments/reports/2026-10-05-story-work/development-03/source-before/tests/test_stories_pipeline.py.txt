import copy
import json
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from dongxi_llms.decoder_lab import DecoderConfig, TinyDecoder
from dongxi_llms.pretraining_lab import fixture
from dongxi_llms.stories_data import documents, write_split, EOS, IGNORE, Windows, digest
from dongxi_llms.stories_training import StoriesDecoder, Recipe, Session


class Fixture:
    def __init__(self):
        self.data, _ = fixture(length=8)
        self.identity = "authored-byte-fixture-v1"

    def __len__(self):
        return len(self.data["x"])

    def batch(self, indices):
        return self.data["x"][indices], self.data["y"][indices]


def config():
    return DecoderConfig(vocab=258, width=16, heads=4, kv_heads=4, head_dim=4,
                         layers=2, hidden=32, max_length=8, modern=True)


class StoriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_delimiters_and_incomplete_response(self):
        self.assertEqual(list(documents(["First\n", "line.\n", "<|endoftext|>\n",
                                         "Second.\n", "<|endoftext|>\n"])),
                         ["First\nline.", "Second."])
        with self.assertRaises(ValueError):
            list(documents(["truncated document"]))
        self.assertEqual(list(documents(["complete source final story"], verified_eof=True)),
                         ["complete source final story"])

    def test_windows_shift_eos_and_no_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            stats, _ = write_split(root, "train", ["abcdef", "g"],
                                    lambda s: [ord(c) for c in s], 4, 0)
            tokens = np.fromfile(Path(root)/"train.bin", dtype="<u4")
            windows = np.load(Path(root)/"train.windows.npy")
            labels = [tokens[start+1:start+n+1].tolist() for start, n in windows]
            self.assertEqual(labels, [[97, 98, 99, 100], [101, 102, EOS], [103, EOS]])
            self.assertEqual(stats["valid_targets"], 9)
            self.assertEqual(stats["long_documents"], 1)
            self.assertEqual(int(tokens[windows[2, 0]]), EOS)

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            _, hashes = write_split(root, "train", ["A cat"], lambda s: [1, 2], 4, 0)
            with self.assertRaises(ValueError):
                write_split(root, "valid", [" a CAT  "], lambda s: [1, 2], 4, 0, hashes)

    def test_validation_first_filter_and_dedup(self):
        with tempfile.TemporaryDirectory() as root:
            _, heldout = write_split(root, "valid", ["reserved"], lambda s: [1], 4, 0)
            stats, _ = write_split(root, "train", ["Reserved", "a", "A", "b"],
                lambda s: [2], 4, 0, heldout, filter_overlap=True, deduplicate=True)
            self.assertEqual(stats["documents"], 2)
            self.assertEqual(stats["excluded_overlap"], 1)
            self.assertEqual(stats["duplicate_documents"], 1)

    def test_loader_padding_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats, _ = write_split(root, "train", ["abc"], lambda s: [1, 2, 3], 8, 0)
            (root/"tokenizer.json").write_text("fixture tokenizer identity")
            manifest = dict(context=8, train=stats, tokenizer_sha256=digest(root/"tokenizer.json"))
            (root/"manifest.json").write_text(json.dumps(manifest))
            data = Windows(root, "train")
            x, y = data.batch([0])
            self.assertEqual(x.tolist(), [[EOS, 1, 2, 3, EOS, EOS, EOS, EOS]])
            self.assertEqual(y.tolist(), [[1, 2, 3, EOS, IGNORE, IGNORE, IGNORE, IGNORE]])
            with (root/"train.bin").open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                Windows(root, "train")

    def test_reference_forward_and_gradients(self):
        torch.manual_seed(1)
        reference = TinyDecoder(config()).double()
        actual = StoriesDecoder(config()).double()
        actual.load_state_dict(reference.state_dict())
        ids = torch.randint(0, 258, (2, 8))
        a, b = reference(ids), actual(ids)
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(reference.parameters(), actual.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-9)

    def test_causality_and_tied_weights(self):
        model = StoriesDecoder(config())
        ids = torch.randint(0, 258, (1, 8))
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:]+1) % 258
        torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4], atol=0, rtol=0)
        self.assertIs(model.token.weight, model.lm_head.weight)

    def test_checkpointing_gradients(self):
        a, b = StoriesDecoder(config()), StoriesDecoder(config(), True)
        b.load_state_dict(a.state_dict())
        ids = torch.randint(0, 258, (2, 8))
        a(ids).sum().backward()
        b(ids).sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_weighted_accumulation(self):
        data = Fixture()
        data.data["y"][0, 5:] = IGNORE
        a = Session(config(), Recipe(microbatch=2, activation_checkpointing=False), data)
        b = Session(config(), Recipe(accumulation=2, activation_checkpointing=False), data)
        a.stream.order = torch.arange(len(data))
        b.stream.order = torch.arange(len(data))
        ra, rb = a.update(), b.update()
        self.assertEqual(ra["valid_targets"], 13)
        self.assertEqual(ra["valid_targets"], rb["valid_targets"])
        self.assertAlmostEqual(ra["loss"], rb["loss"], places=5)
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_exact_resume_and_contract_rejection(self):
        data, recipe = Fixture(), Recipe()
        session = Session(config(), recipe, data)
        session.update()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"state.pt"
            session.save(path)
            expected = session.update()
            restored = Session(config(), recipe, data)
            restored.restore(path)
            actual = restored.update()
            for field in ("loss", "lr", "valid_targets", "cumulative_targets", "gradient_norm"):
                self.assertEqual(expected[field], actual[field])
            for p, q in zip(session.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            self.assertEqual(session.stream.cursor, restored.stream.cursor)
            changed = Session(config(), replace(recipe, peak_lr=.002), data)
            with self.assertRaises(ValueError):
                changed.restore(path)
            with self.assertRaises(FileExistsError):
                session.save(path)

    def test_validation_preserves_state_and_grouping(self):
        data = Fixture()
        a = Session(config(), Recipe(), data)
        b = Session(config(), Recipe(microbatch=3), data)
        before = copy.deepcopy(a.stream.state_dict())
        one, three = a.evaluate(data), b.evaluate(data)
        self.assertAlmostEqual(one["loss"], three["loss"], places=6)
        self.assertEqual(one["valid_targets"], three["valid_targets"])
        self.assertEqual(a.stream.cursor, before["cursor"])
        self.assertTrue(a.model.training)
        self.assertTrue(all(p.grad is None for p in a.model.parameters()))

    def test_invalid_recipe_and_bounds(self):
        with self.assertRaises(ValueError):
            Recipe(accumulation=0)
        with self.assertRaises(ValueError):
            Recipe(peak_lr=float("nan"))
        with self.assertRaises(ValueError):
            StoriesDecoder(config())(torch.ones(1, 9, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

"""Original CPU fixtures for the story update-boundary target cap.

No corpus, tokenizer, pretrained weights or full-size baseline is loaded.
Direct invocation with --report writes a new, non-overwriting evidence record.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.decoder_lab import DecoderConfig
from dongxi_llms.pretraining_lab import learning_rate
from dongxi_llms.stories_data import IGNORE
from dongxi_llms.stories_training import (Recipe, Session, ValidTargetBudgetExceeded,
                                         validate_target_budget, run)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return DecoderConfig(vocab=16, width=8, heads=2, kv_heads=2, head_dim=4,
                         layers=1, hidden=16, max_length=8, modern=True,
                         qk_norm=False, tied=True)


def recipe(**kwargs):
    return Recipe(total_updates=5, warmup=1, peak_lr=.003, floor_lr=.0003,
                  activation_checkpointing=False, **kwargs)


class AuthoredWindows:
    """ID-only variable-length windows; EOS 15 is real once and pad thereafter."""
    identity = "original-story-budget-ids-lengths-3-6-1-7-v1"
    length = 8
    manifest = {"fixture": identity, "context": length}

    def __init__(self, empty=False):
        self.x = torch.full((4, 8), 15, dtype=torch.long)
        self.y = torch.full((4, 8), IGNORE, dtype=torch.long)
        for i, count in enumerate((3, 6, 1, 7)):
            self.x[i, :count] = torch.tensor([14, *range(1, count)])
            if not empty:
                self.y[i, :count] = torch.tensor([*range(1, count), 15])

    def __len__(self):
        return 4

    def batch(self, indices):
        return self.x[indices], self.y[indices]


def session(cap=None, **kwargs):
    result = Session(config(), recipe(**kwargs), AuthoredWindows(),
                     valid_target_budget=cap)
    result.stream.order = torch.arange(4)
    return result


def snapshot(value):
    return copy.deepcopy(dict(model=value.model.state_dict(),
                              optimizer=value.optimizer.state_dict(),
                              gradients=[p.grad for p in value.model.parameters()],
                              stream=value.stream.state_dict(), rng=torch.get_rng_state(),
                              training=value.model.training, step=value.step,
                              targets=value.tokens, positions=value.positions))


def assert_tree_equal(test, left, right):
    test.assertIs(type(left), type(right))
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def historical_update(value):
    """Independent original no-cap update equations, without new budget branches."""
    value.model.train()
    batches = [value.train_data.batch(value.stream.take(value.recipe.microbatch))
               for _ in range(value.recipe.accumulation)]
    count = sum(int((y != IGNORE).sum()) for _, y in batches)
    lr = learning_rate(value.step, value.recipe.total_updates, value.recipe.warmup,
                       value.recipe.peak_lr, value.recipe.floor_lr)
    for group in value.optimizer.param_groups:
        group["lr"] = lr
    value.optimizer.zero_grad(set_to_none=True)
    total = 0.
    for x, y in batches:
        loss = value.summed_loss(x, y) / count
        loss.backward()
        total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(value.model.parameters(), value.recipe.clip,
                                         error_if_nonfinite=True)
    value.optimizer.step()
    value.step += 1
    value.tokens += count
    value.positions += sum(x.numel() for x, _ in batches)
    return dict(loss=total, lr=lr, gradient_norm=float(norm), valid_targets=count,
                cumulative_targets=value.tokens)


class StoryTargetBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_positive_integer_cap_only(self):
        self.assertIsNone(validate_target_budget(None))
        self.assertEqual(validate_target_budget(17), 17)
        for value in (0, -1, True, False, 3., 1.5, float("inf"), float("nan"),
                      "17", torch.tensor(17)):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                Session(config(), recipe(), AuthoredWindows(), valid_target_budget=value)

    def test_first_crossing_has_no_forward_or_state_change(self):
        value = session(8, accumulation=2)
        value.model.eval()
        before = snapshot(value)
        with patch.object(value, "summed_loss", side_effect=AssertionError("forward forbidden")), \
                self.assertRaises(ValidTargetBudgetExceeded) as caught:
            value.update()
        self.assertEqual(caught.exception.record(), dict(valid_target_budget=8,
            cumulative_targets=0, next_update_valid_targets=9, remaining_valid_targets=8))
        assert_tree_equal(self, before, snapshot(value))

    def test_equality_accepts_and_then_preserves_completed_gradients(self):
        value = session(9, accumulation=2)
        row = value.update()
        self.assertEqual((row["valid_targets"], value.tokens, value.positions), (9, 9, 16))
        before = snapshot(value)
        with patch.object(value.optimizer, "step", side_effect=AssertionError("step forbidden")), \
                self.assertRaises(ValidTargetBudgetExceeded) as caught:
            value.update()
        self.assertEqual((caught.exception.next_update, caught.exception.remaining), (8, 0))
        assert_tree_equal(self, before, snapshot(value))

    def test_positive_remainder_cannot_reweight_next_update(self):
        value = session(16, accumulation=2)
        value.update()
        with self.assertRaises(ValidTargetBudgetExceeded) as caught:
            value.update()
        self.assertEqual((value.step, value.tokens, caught.exception.remaining), (1, 9, 7))
        self.assertEqual(value.stream.cursor, 2)

    def test_two_steps_exactly_fit_and_eos_is_not_padding(self):
        value = session(17, accumulation=2)
        rows = [value.update(), value.update()]
        self.assertEqual([r["valid_targets"] for r in rows], [9, 8])
        self.assertEqual([r["cumulative_processed_positions"] for r in rows], [16, 32])
        self.assertEqual(value.tokens, 17)
        self.assertEqual(int((value.train_data.y == 15).sum()), 4)
        self.assertEqual(int((value.train_data.x == 15).sum()), 15)

    def test_epoch_crossing_restores_shuffle_rng_and_pending_ids(self):
        value = session(1, microbatch=3, accumulation=2)
        before = snapshot(value)
        with self.assertRaises(ValidTargetBudgetExceeded):
            value.update()
        assert_tree_equal(self, before, snapshot(value))
        expected = value.stream.take(6)
        value.stream.load_state_dict(before["stream"])
        with self.assertRaises(ValidTargetBudgetExceeded):
            value.update()
        self.assertEqual(value.stream.take(6), expected)

    def test_one_microbatch_uses_its_exact_valid_length(self):
        value = session(3)
        value.update()
        self.assertEqual((value.step, value.tokens, value.positions), (1, 3, 8))
        with self.assertRaises(ValidTargetBudgetExceeded):
            value.update()

    def test_empty_target_batch_restores_active_cap_cursor(self):
        value = Session(config(), recipe(), AuthoredWindows(empty=True), valid_target_budget=3)
        before = snapshot(value)
        with self.assertRaisesRegex(ValueError, "No valid"):
            value.update()
        assert_tree_equal(self, before, snapshot(value))

    def test_no_cap_matches_original_update_equations_bitwise(self):
        actual, original = session(None, accumulation=2), session(None, accumulation=2)
        self.assertNotIn("valid_target_budget", actual.contract)
        self.assertNotIn("valid_target_budget", asdict(actual.recipe))
        for _ in range(5):
            row, reference = actual.update(), historical_update(original)
            for key, expected in reference.items():
                self.assertEqual(row[key], expected)
            assert_tree_equal(self, snapshot(actual), snapshot(original))

    def test_large_cap_matches_uncapped_updates_bitwise(self):
        uncapped, capped = session(None, accumulation=2), session(1000, accumulation=2)
        for _ in range(5):
            a, b = uncapped.update(), capped.update()
            for field in ("loss", "lr", "gradient_norm", "valid_targets", "cumulative_targets",
                          "processed_positions", "cumulative_processed_positions"):
                self.assertEqual(a[field], b[field])
            assert_tree_equal(self, snapshot(uncapped), snapshot(capped))

    def test_schedule_remains_exhausted_before_budget_collection(self):
        value = session(1000)
        value.step = value.recipe.total_updates
        before = snapshot(value)
        with self.assertRaisesRegex(ValueError, "schedule exhausted"):
            value.update()
        assert_tree_equal(self, before, snapshot(value))

    def test_completed_boundary_resume_is_bitwise_and_cumulative(self):
        value = session(17, accumulation=2)
        value.update()
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "first.pt"
            value.save(checkpoint)
            expected = value.update()
            expected_state = snapshot(value)
            resumed = session(17, accumulation=2)
            resumed.restore(checkpoint)
            self.assertEqual((resumed.step, resumed.tokens, resumed.positions), (1, 9, 16))
            actual = resumed.update()
            for field in ("loss", "lr", "gradient_norm", "cumulative_targets",
                          "cumulative_processed_positions"):
                self.assertEqual(actual[field], expected[field])
            assert_tree_equal(self, expected_state, snapshot(resumed))

    def test_rejected_boundary_checkpoint_reconstructs_same_pending_batch(self):
        value = session(16, accumulation=2)
        value.update()
        with self.assertRaises(ValidTargetBudgetExceeded) as first:
            value.update()
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "blocked.pt"
            value.save(checkpoint)
            restored = session(16, accumulation=2)
            restored.restore(checkpoint)
            before = snapshot(restored)
            with self.assertRaises(ValidTargetBudgetExceeded) as second:
                restored.update()
            self.assertEqual(first.exception.record(), second.exception.record())
            assert_tree_equal(self, before, snapshot(restored))
            self.assertEqual(value.stream.take(2), restored.stream.take(2))

    def test_changed_added_or_removed_cap_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as root:
            for source, target in ((16, 17), (16, None), (None, 16)):
                with self.subTest(source=source, target=target):
                    path = Path(root) / f"{source}-{target}.pt"
                    session(source).save(path)
                    restored = session(target)
                    with patch.object(restored.model, "load_state_dict", side_effect=AssertionError("load forbidden")), \
                            self.assertRaisesRegex(ValueError, "contract changed"):
                        restored.restore(path)

    def test_invalid_resume_counters_fail_before_model_load(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            value = session(17, accumulation=2)
            value.update()
            value.save(root / "source.pt")
            initial = torch.load(root / "source.pt", weights_only=True)
            variants = ({"tokens": -1}, {"tokens": 9.}, {"tokens": True}, {"tokens": 18},
                        {"step": True}, {"step": 1.}, {"step": 0},
                        {"processed_positions": 8}, {"processed_positions": float("inf")},
                        {"processed_positions_accounting": "pretend-measured"})
            for i, change in enumerate(variants):
                with self.subTest(change=change):
                    state = copy.deepcopy(initial)
                    state.update(change)
                    path = root / f"invalid-{i}.pt"
                    torch.save(state, path)
                    restored = session(17, accumulation=2)
                    with patch.object(restored.model, "load_state_dict", side_effect=AssertionError("load forbidden")), \
                            self.assertRaises(ValueError):
                        restored.restore(path)

    def test_missing_position_field_is_explicitly_derived_not_measured(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            value = session(None, accumulation=2)
            value.update()
            value.save(root / "current.pt")
            state = torch.load(root / "current.pt", weights_only=True)
            del state["processed_positions"]
            del state["processed_positions_accounting"]
            torch.save(state, root / "same-source-legacy-shaped.pt")
            restored = session(None, accumulation=2)
            restored.restore(root / "same-source-legacy-shaped.pt")
            self.assertEqual(restored.positions, 16)
            self.assertEqual(restored.positions_accounting,
                             "derived-legacy-fixed-geometry-plus-measured-continuation")
            row = restored.update()
            self.assertEqual(row["cumulative_processed_positions"], 32)
            self.assertTrue(row["processed_positions_accounting"].startswith("derived-legacy"))

    def test_run_clean_budget_stop_and_matching_resume_have_correct_completion(self):
        import dongxi_llms.stories_training as module
        class NoMonitor:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        class FakeTokenizer:
            @classmethod
            def from_file(cls, _):
                return object()
        actual_session = module.Session
        def make_session(cfg, rec, data, device, cap):
            value = actual_session(cfg, rec, data, device, cap)
            value.stream.order = torch.arange(4)
            return value
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(module, "baseline", side_effect=lambda _: config()), \
                patch.object(module, "Windows", side_effect=lambda *_: AuthoredWindows()), \
                patch.object(module, "Session", side_effect=make_session), \
                patch.object(module, "MemoryMonitor", NoMonitor), \
                patch.object(module, "samples", return_value=[]), \
                patch.object(module, "activation_summary", return_value=[]), \
                patch("tokenizers.Tokenizer", FakeTokenizer):
            root = Path(temporary) / "first"
            with patch("builtins.print"):
                result = run("unused", root, device="cpu", total=5, warmup=1,
                             accumulation=2, valid_target_budget=16, max_seconds=600)
            self.assertEqual(result, 1)
            completion = json.loads((root / "completion.json").read_text())
            self.assertTrue(completion["stopped_for_target_budget"])
            self.assertFalse(completion["stopped_for_time_budget"])
            self.assertFalse(completion["requested_stop_reached"])
            self.assertEqual((completion["cumulative_targets"], completion["remaining_valid_targets"],
                              completion["cumulative_processed_positions"]), (9, 7, 16))
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 1)
            resumed = Path(temporary) / "resumed"
            with patch("builtins.print"):
                run("unused", resumed, device="cpu", total=5, warmup=1,
                    accumulation=2, valid_target_budget=16, max_seconds=600,
                    resume=root / "update-000001.pt")
            repeated = json.loads((resumed / "completion.json").read_text())
            self.assertEqual(repeated["target_budget_stop"], completion["target_budget_stop"])
            self.assertEqual((resumed / "metrics.jsonl").read_text(), "")

    def test_run_invalid_cap_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "absent"
            with self.assertRaises(ValueError):
                run("unused", output, device="cpu", valid_target_budget=True)
            self.assertFalse(output.exists())

    def test_cli_passes_optional_cap_without_executing_runner(self):
        spec = importlib.util.spec_from_file_location("story_budget_cli", ROOT / "scripts/train_stories.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for extra, expected in (([], None), (["--valid-target-budget", "17"], 17)):
            with self.subTest(expected=expected), patch.object(cli, "run") as dispatch, \
                    patch.object(sys, "argv", ["train_stories.py", "train", "--data", "unused",
                                              "--output", "unused", *extra]):
                cli.main()
            self.assertEqual(dispatch.call_args.kwargs["valid_target_budget"], expected)


def state_digest(value):
    """Length-framed data identity for report comparisons, not a checkpoint format."""
    h = hashlib.sha256()
    def frame(tag, data):
        h.update(tag + len(data).to_bytes(8, "big") + data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            frame(b"tensor", str((str(item.dtype), tuple(item.shape))).encode())
            frame(b"bytes", item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            frame(b"dict", str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            frame(type(item).__name__.encode(), str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            frame(type(item).__name__.encode(), repr(item).encode())
    visit(value)
    return h.hexdigest()


def collect_evidence():
    torch.set_num_threads(1)
    outcomes = []
    for cap in (8, 9, 16, 17):
        value = session(cap, accumulation=2)
        rows = []
        while value.step < value.recipe.total_updates:
            before = state_digest(snapshot(value))
            with patch.object(value, "summed_loss", wraps=value.summed_loss) as loss_call:
                try:
                    rows.append(value.update())
                except ValidTargetBudgetExceeded as refusal:
                    outcomes.append(dict(cap=cap, accepted_updates=rows, refusal=refusal.record(),
                        rejected_update_forward_calls=loss_call.call_count,
                        before_refusal_sha256=before, after_refusal_sha256=state_digest(snapshot(value)),
                        refusal_left_state_bitwise_equal=before == state_digest(snapshot(value)),
                        final_completed_updates=value.step, final_cumulative_targets=value.tokens,
                        final_cumulative_physical_positions=value.positions))
                    break
    a, b = session(None, accumulation=2), session(1000, accumulation=2)
    paired = []
    for _ in range(5):
        left, right = a.update(), b.update()
        paired.append(dict(uncapped=left, sufficiently_capped=right,
                           full_state_equal=state_digest(snapshot(a)) == state_digest(snapshot(b))))
    with tempfile.TemporaryDirectory() as root:
        value = session(17, accumulation=2)
        value.update()
        checkpoint = Path(root) / "completed-first-update.pt"
        value.save(checkpoint)
        expected = value.update()
        expected_digest = state_digest(snapshot(value))
        resumed = session(17, accumulation=2)
        resumed.restore(checkpoint)
        loaded = dict(completed_updates=resumed.step, cumulative_targets=resumed.tokens,
                      cumulative_physical_positions=resumed.positions)
        actual = resumed.update()
        replay = dict(loaded=loaded, uninterrupted=expected, resumed=actual,
                      uninterrupted_state_sha256=expected_digest,
                      resumed_state_sha256=state_digest(snapshot(resumed)),
                      bitwise_full_state_equal=expected_digest == state_digest(snapshot(resumed)))
    return dict(boundary_controls=outcomes, default_vs_large_cap=paired, completed_boundary_replay=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is None:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
        return
    if args.report.exists():
        raise FileExistsError(args.report)
    paths = ["src/dongxi_llms/stories_training.py", "src/dongxi_llms/stories_data.py",
             "src/dongxi_llms/pretraining_lab.py", "src/dongxi_llms/decoder_lab.py",
             "scripts/train_stories.py", "tests/test_stories_valid_target_budget.py",
             "tests/test_stories_pipeline.py", "experiments/specs/2026-10-05-story-valid-target-budget.md",
             "experiments/specs/2026-10-04-staged-spark-campaign.md", "uv.lock"]
    def hashes():
        return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    before = hashes()
    start = time.monotonic()
    result = collect_evidence()
    tests = []
    for pattern in ("test_stories_valid_target_budget.py", "test_stories_pipeline.py"):
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
        began = time.monotonic()
        outcome = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        match = re.search(r"Ran (\d+) tests? in", outcome.stderr)
        tests.append(dict(command=command, actual_exit_code=outcome.returncode,
                          tests_run=int(match.group(1)) if match else None,
                          stdout=outcome.stdout, stderr=outcome.stderr,
                          seconds=time.monotonic()-began))
    result.update(schema=1, scope="original tiny CPU update-boundary guard; no model-scale campaign",
                  invocation=[sys.executable, "tests/test_stories_valid_target_budget.py", "--report", str(args.report)],
                  actual_argv=sys.argv, python=sys.version, torch=str(torch.__version__),
                  executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
                  cuda_available=torch.cuda.is_available(), recipe=asdict(recipe(accumulation=2)),
                  model=asdict(config()), source_sha256_before=before, source_sha256_after=hashes(),
                  test_runs=tests,
                  seconds=time.monotonic()-start,
                  limitations=["No GPU, pretrained weights, TinyStories data, tokenizer load or story generation.",
                               "No Spark profile/pilot, production supervisor/tree containment or disk limit proof.",
                               "Historical checkpoint implementation hashes remain incompatible.",
                               "Deterministic single-process data only; no partial-update recovery guarantee.",
                               "Temporary checkpoint bytes are not retained; replay identities and outcomes are."])
    result["checks"] = dict(
        source_hashes_unchanged=result["source_sha256_before"] == result["source_sha256_after"],
        every_refused_update_unchanged=all(row["refusal_left_state_bitwise_equal"] for row in result["boundary_controls"]),
        every_refused_update_has_zero_forward_calls=all(row["rejected_update_forward_calls"] == 0 for row in result["boundary_controls"]),
        no_target_cap_overshoot=all(row["final_cumulative_targets"] <= row["cap"] for row in result["boundary_controls"]),
        all_focused_and_existing_pipeline_tests_pass=all(row["actual_exit_code"] == 0 for row in tests),
        all_default_large_cap_states_match=all(row["full_state_equal"] for row in result["default_vs_large_cap"]),
        completed_boundary_resume_matches=result["completed_boundary_replay"]["bitwise_full_state_equal"])
    with args.report.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(report=str(args.report), checks=result["checks"], seconds=result["seconds"])))
    if not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

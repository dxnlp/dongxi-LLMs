"""Original independent state/endpoint/budget/failure contracts; CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.critique_revision_lab import (ACTION_IDS, EOS, MODES, NUMBER_START,
    acceptance, callback_views, critique, digest, emission, independent_control,
    load_inputs, main, programmatic_pool, revise, run_loop, stable_payload, transition)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "fixtures/critique-revision/protocol.json"


def item(reference="0"):
    return dict(id="original-i", source_group="original-g", split="test", slice="fixture",
                task="binary", prompt="Original fixture", reference=reference,
                kind="math", format_policy="single_integer", extraction="whole")


def answer(value):
    return emission([NUMBER_START + value, EOS])


class StateMachineTests(unittest.TestCase):
    def test_callback_whitelist_physically_removes_nested_gold(self):
        i = item(); i.update(problem={"gold": 1}, rubric={"answer": "0"})
        d = answer(1); d.update(reference="0", correct=False, grade={"gold": 0}, cost={"reference": 0})
        iv, dv = callback_views(i, d)
        self.assertFalse({"reference", "problem", "rubric", "correct", "grade", "cost"} & (iv.keys() | dv.keys()))
        dv["token_ids"][0] = 99
        self.assertEqual(d["token_ids"], [7, 2])

    def test_gold_changes_grading_but_not_proposals_or_decisions(self):
        a = run_loop(item("0"), answer(0), mode="contrarian")
        b = run_loop(item("1"), answer(0), mode="contrarian")
        for x, y in zip(a["rounds"], b["rounds"]):
            for k in ("critique", "revision", "acceptance", "delivered", "stopping"):
                self.assertEqual(x[k], y[k])
        self.assertNotEqual(a["delivered_grade"], b["delivered_grade"])

    def test_callbacks_observe_only_prior_state_and_are_detached(self):
        seen = []
        def critic(iv, dv, r):
            self.assertNotIn("reference", iv); self.assertNotIn("correct", dv)
            seen.append(dv["token_ids"][:]); dv["token_ids"].clear()
            return emission([ACTION_IDS["flip"], EOS])
        a = run_loop(item(), answer(0), critique_fn=critic)
        self.assertEqual(seen, [[6, 2], [7, 2]])
        self.assertEqual(a["initial"]["token_ids"], [6, 2])

    def test_valid_tie_accepts_wrong_answer_without_correctness(self):
        d = acceptance(answer(0), answer(1))
        self.assertTrue(d["accepted"]); self.assertTrue(d["tie"]); self.assertFalse(d["gold_used"])
        a = run_loop(item(), answer(0), mode="contrarian", max_rounds=1)
        self.assertEqual(a["final_transition"]["category"], "right_to_wrong")

    def test_flip_has_both_directions_and_double_flip_endpoint(self):
        b = run_loop(item(), answer(1), mode="contrarian", max_rounds=1)
        self.assertEqual(b["final_transition"]["category"], "wrong_to_right")
        a = run_loop(item(), answer(0), mode="contrarian")
        self.assertEqual([r["attempted_transition"]["category"] for r in a["rounds"]],
                         ["right_to_wrong", "wrong_to_right"])
        self.assertEqual(a["final_transition"]["category"], "unchanged")
        self.assertEqual(a["serialized_tokens"], 10)

    def test_unchanged_stop_is_not_success_stop(self):
        a = run_loop(item(), answer(1), mode="identity")
        self.assertFalse(a["delivered_grade"]["complete_success"])
        self.assertEqual(len(a["rounds"]), 1)
        self.assertEqual(a["rounds"][0]["stopping"], "unchanged-accepted-path")
        self.assertEqual(a["serialized_tokens"], 6)

    def test_repair_uses_observed_token_or_zero_not_gold(self):
        d = emission([7, 7, 6], "max_tokens")
        a = run_loop(item("0"), d, mode="repair")
        self.assertEqual(a["delivered"]["token_ids"], [7, 2])
        self.assertFalse(a["delivered_grade"]["complete_success"])
        b = run_loop(item("1"), emission([EOS]), mode="repair")
        self.assertEqual(b["delivered"]["token_ids"], [6, 2])
        self.assertFalse(b["delivered_grade"]["complete_success"])

    def test_invalid_proposal_is_retained_rejected_and_charged(self):
        def invalid(*args): return emission([6, 7, EOS])
        a = run_loop(item(), answer(0), revision_fn=invalid)
        self.assertEqual(len(a["rounds"]), 2)
        self.assertTrue(a["delivered_grade"]["complete_success"])
        self.assertEqual(a["serialized_tokens"], 12)
        self.assertTrue(all(not r["acceptance"]["accepted"] for r in a["rounds"]))
        self.assertTrue(all(r["revision"]["token_ids"] == [6, 7, 2] for r in a["rounds"]))

    def test_empty_and_cap_proposals_are_not_natural_valid_answers(self):
        for row in (emission([EOS]), emission([6, 6, 6], "max_tokens")):
            a = run_loop(item(), answer(0), revision_fn=lambda *args, v=row: deepcopy(v))
            self.assertTrue(a["delivered_grade"]["complete_success"])
            self.assertFalse(a["rounds"][0]["acceptance"]["accepted"])
            self.assertEqual(a["rounds"][0]["revision"], row)

    def test_critique_exception_records_failure_and_skipped_revision(self):
        def broken(*args): raise RuntimeError("Original test failure")
        a = run_loop(item(), answer(1), critique_fn=broken)
        r = a["rounds"][0]
        self.assertIn("Original test failure", r["critique"]["error"])
        self.assertIsNone(r["revision"])
        self.assertEqual(r["callback_costs"][1]["attempted_calls"], 0)
        self.assertEqual(len(a["rounds"]), 2)

    def test_revision_exception_and_bad_serialization_retained(self):
        def broken(*args): raise RuntimeError("Original revision failure")
        a = run_loop(item(), answer(0), revision_fn=broken)
        self.assertIn("Original revision failure", a["rounds"][0]["revision"]["error"])
        invalid = answer(1); invalid["raw_response"] = "fabricated text"
        b = run_loop(item(), answer(0), revision_fn=lambda *args: deepcopy(invalid))
        self.assertEqual(b["rounds"][0]["revision"]["raw_response"], "fabricated text")
        self.assertIn("serialization", b["rounds"][0]["revision"]["error"])
        self.assertFalse(b["rounds"][0]["acceptance"]["accepted"])

    def test_unknown_and_unserializable_critique_are_failures(self):
        a = run_loop(item(), answer(0), critique_fn=lambda *args: emission([99, EOS]))
        self.assertEqual(a["rounds"][0]["critique"]["token_ids"], [99, 2])
        self.assertIsNone(a["rounds"][0]["revision"])
        b = run_loop(item(), answer(0), critique_fn=lambda *args: {"token_ids": [float("nan")]})
        self.assertTrue(b["rounds"][0]["critique"]["unserializable_return"])
        json.dumps(b, allow_nan=False)

    def test_capped_critique_is_retained_but_cannot_trigger_revision(self):
        capped = emission([ACTION_IDS["keep"], EOS], "max_tokens")
        a = run_loop(item(), answer(0), critique_fn=lambda *args: deepcopy(capped))
        for row in a["rounds"]:
            self.assertEqual(row["critique"]["token_ids"], [20, EOS])
            self.assertEqual(row["critique"]["stop_reason"], "max_tokens")
            self.assertTrue(row["critique"]["truncated"])
            self.assertIn("natural EOS", row["critique"]["error"])
            self.assertIsNone(row["revision"])
            self.assertEqual(row["callback_costs"][1]["attempted_calls"], 0)
            self.assertFalse(row["acceptance"]["accepted"])
        self.assertEqual(a["serialized_tokens"], 6)
        self.assertTrue(a["delivered_grade"]["complete_success"])

    def test_failing_deepcopy_does_not_erase_the_failed_round(self):
        class Uncopyable:
            def __deepcopy__(self, memo): raise TypeError("Original deepcopy failure")
        invalid = emission([99, EOS]); invalid["diagnostic"] = Uncopyable()
        a = run_loop(item(), answer(0), critique_fn=lambda *args: invalid)
        retained = a["rounds"][0]["critique"]
        self.assertTrue(retained["unserializable_return"])
        self.assertIn("known action", retained["error"])
        self.assertIn("diagnostic", retained["raw_return_repr"])
        self.assertEqual(retained["token_ids"], [99, EOS])
        self.assertEqual(retained["generated_tokens"], 2)
        self.assertFalse(retained["emitted_token_count_unknown"])
        self.assertIsNone(a["rounds"][0]["revision"])
        json.dumps(a, allow_nan=False)

    def test_gate_checks_tokens_not_just_decoded_string(self):
        a = answer(1); a["response_text"] = "0"
        self.assertFalse(acceptance(answer(0), a)["accepted"])
        for row in (dict(answer(0), truncated=True), dict(answer(0), error="failure")):
            self.assertFalse(acceptance(answer(1), row)["accepted"])

    def test_mode_round_and_action_bounds(self):
        for kwargs in ({"mode": "oracle"}, {"max_rounds": True}, {"max_rounds": 3}, {"max_rounds": 0}):
            with self.assertRaises(ValueError): run_loop(item(), answer(0), **kwargs)
        with self.assertRaises(ValueError): revise({}, emission([EOS]), emission([21, EOS]), 0)
        with self.assertRaises(ValueError): critique({}, answer(0), 0, mode="oracle")


class ControlAndIdentityTests(unittest.TestCase):
    def test_programmatic_draws_are_gold_and_order_independent(self):
        a = programmatic_pool(item("0"), 26061)
        b = programmatic_pool(item("1"), 26061)
        self.assertEqual(a, b)
        self.assertEqual(a, programmatic_pool(item(), 26061))
        self.assertEqual([r["generated_tokens"] for r in a], [2] * 8)
        self.assertTrue(all(r["model_forward_calls"] == 0 for r in a))

    def test_exact_serialized_match_for_all_fixed_modes(self):
        pool = programmatic_pool(item(), 26061)
        for mode in MODES:
            loop = run_loop(item(), pool[0], mode=mode)
            d = independent_control(item(), pool, loop["serialized_tokens"])
            self.assertTrue(d["exact_serialized_match"])
            self.assertEqual(d["overshoot"], 0); self.assertEqual(d["underfill"], 0)
            self.assertEqual(d["historical_source_costs"]["generation_tokens"]["known_total"], 0)

    def test_actual_pool_whole_attempt_overrun_is_preserved(self):
        pool = programmatic_pool(item(), 26061)
        pool[1].update(emission([6, 7, EOS]))
        d = independent_control(item(), pool, 4)
        self.assertEqual(d["eligible_tokens"], 2)
        self.assertEqual(d["attempted_tokens"], 5)
        self.assertEqual(d["underfill"], 2); self.assertEqual(d["overshoot"], 1)
        self.assertEqual(d["rejections"][0]["remaining"], 2)
        self.assertFalse(d["exact_serialized_match"])

    def test_independent_ties_and_gold_injection_are_stable(self):
        pool = programmatic_pool(item(), 26061)
        pool[0].update(answer(1)); pool[1].update(answer(0))
        a = independent_control(item(), pool, 4)
        self.assertEqual(a["selected"]["response_text"], "1")
        injected = deepcopy(pool)
        for r in injected: r.update(reference="0", correct=True)
        b = independent_control(item(), injected, 4)
        self.assertEqual(a["decision"], b["decision"])

    def test_all_actual_inputs_are_hash_gated_and_complete(self):
        protocol, items, raw, pools, adversarial, audit = load_inputs(ROOT, PROTOCOL)
        self.assertEqual(len(raw), 864); self.assertEqual(len(pools), 108)
        self.assertEqual(len(items), 18); self.assertEqual(len(adversarial), 12)
        self.assertEqual(audit["items"], 18)
        self.assertEqual(digest(ROOT/protocol["source_responses"]), protocol["source_responses_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"protocol.json"; changed = deepcopy(protocol)
            changed["source_responses_sha256"] = "0" * 64
            p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): load_inputs(ROOT, p)
            changed = deepcopy(protocol); changed["acceptance"] = "oracle"
            p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): load_inputs(ROOT, p)

    def test_stable_payload_omits_only_timing_not_decisions_or_source_costs(self):
        row = {"wall_seconds": .1, "raw_response": "0", "cost": {"generation_tokens": 2, "wall_seconds": .2},
               "callback_costs": [{"wall_seconds": .3, "attempted_calls": 1}]}
        self.assertEqual(stable_payload(row), {"raw_response": "0", "cost": {"generation_tokens": 2},
                                              "callback_costs": [{"attempted_calls": 1}]})

    def test_cli_failure_and_interrupt_preserve_partial_rounds(self):
        for failure in (RuntimeError("Original campaign failure"), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/"new-run"
                def partial(*args, journal=None):
                    journal.record({"round": 1, "raw_response": "retained partial"})
                    raise failure
                argv = ["--protocol", str(PROTOCOL), "--spec", str(ROOT/"experiments/specs/2026-10-04-critique-revision.md"), "--output", str(output)]
                with patch("dongxi_llms.critique_revision_lab.run_campaign", side_effect=partial):
                    with self.assertRaises(type(failure)): main(argv)
                self.assertIn("retained partial", (output/"responses.jsonl").read_text())
                self.assertTrue((output/"failure.json").exists())
                self.assertIn("failure", (output/"events.jsonl").read_text())
                with self.assertRaises(SystemExit): main(argv)


if __name__ == "__main__":
    unittest.main()

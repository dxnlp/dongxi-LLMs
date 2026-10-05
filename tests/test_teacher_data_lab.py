import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dongxi_llms.teacher_data_lab import (
    load_protocol, validate_protocol, freeze_contract, collect, TeacherJournal,
    execute_attempt, attach_digest, canonical_hash, filter_pool, select_datasets,
    verifier_score, make_student, fit_student, state_digest, evaluate_student,
    training_batch, prefix_ids, task_answer, SPECIAL, VOCAB)
from dongxi_llms.sft_lab import token_loss_sum

ROOT = Path(__file__).resolve().parents[1]


class TeacherDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"journal"
        self.protocol = load_protocol(ROOT)
        self.contract = freeze_contract(ROOT,self.protocol)

    def collected(self):
        journal = collect(self.contract,self.path)
        pool = filter_pool(journal.records,self.protocol)
        return journal,pool,select_datasets(pool,self.protocol)

    def test_original_protocol_split_ids_and_sources(self):
        self.assertEqual(len(self.protocol["items"]),24)
        self.assertEqual(len({tuple(prefix_ids(i)) for i in self.protocol["items"]}),24)
        self.assertEqual(len({i["source_group"] for i in self.protocol["items"]}),24)
        self.assertEqual(task_answer("please reverse red blue green"),"green blue red")

    def test_actual_id_whitespace_alias_rejected(self):
        document = copy.deepcopy(self.protocol)
        document["items"][12]["prompt"] = "copy  red blue"
        document["items"][12]["reference"] = "red blue"
        with self.assertRaisesRegex(ValueError,"collision"): validate_protocol(document)

    def test_underlying_group_polite_prefix_leakage_rejected(self):
        document = copy.deepcopy(self.protocol)
        document["items"][20]["prompt"] = "please copy red blue"
        document["items"][20]["reference"] = "red blue"
        with self.assertRaisesRegex(ValueError,"source group"): validate_protocol(document)

    def test_format_filter_cannot_become_correctness_filter(self):
        document = copy.deepcopy(self.protocol)
        document["filter"]["require_correctness"] = True
        with self.assertRaises(ValueError): validate_protocol(document)

    def test_partial_and_complete_resume_no_duplicate(self):
        first = collect(self.contract,self.path,max_new_records=5)
        first_ids = [r["attempt_id"] for r in first.records]
        resumed = collect(self.contract,self.path)
        complete = collect(self.contract,self.path)
        self.assertEqual(len(complete.records),108)
        self.assertEqual(len(complete.by_id),108)
        self.assertEqual(first_ids,[r["attempt_id"] for r in complete.records[:5]])
        self.assertEqual(resumed.records,complete.records)
        self.assertEqual(sum(r["retry_index"] for r in complete.records),12)
        self.assertEqual(sum(e["event"] == "execution_started" for e in complete.events),108)

    def test_retry_parent_error_is_preserved(self):
        journal,_,_ = self.collected()
        retries = [r for r in journal.records if r["retry_index"]]
        self.assertEqual(len(retries),12)
        for retry in retries:
            parent = next(r for r in journal.records if r["item_id"] == retry["item_id"] and r["sample_index"] == 7 and r["retry_index"] == 0)
            self.assertIsNotNone(parent["error"])
            self.assertEqual(parent["stop"],"error")
            self.assertEqual(retry["stop"],"END")

    def test_changed_contract_and_payload_rejected(self):
        collect(self.contract,self.path,max_new_records=1)
        changed = copy.deepcopy(self.contract); changed["content_terms"] = "changed"
        with self.assertRaisesRegex(ValueError,"contract"): TeacherJournal(self.path,changed)
        changed["identity"] = canonical_hash({k:v for k,v in changed.items() if k != "identity"})
        with self.assertRaisesRegex(ValueError,"different"): TeacherJournal(self.path,changed)
        path = self.path/"attempts.jsonl"
        row = json.loads(path.read_text()); row["final_answer"] = "blue"
        path.write_text(json.dumps(row)+"\n")
        with self.assertRaisesRegex(ValueError,"payload"): TeacherJournal(self.path,self.contract)

    def test_duplicate_records_and_concurrent_writer_rejected(self):
        collect(self.contract,self.path,max_new_records=1)
        writer = TeacherJournal(self.path,self.contract)
        self.addCleanup(writer.close)
        with self.assertRaisesRegex(ValueError,"writer"): TeacherJournal(self.path,self.contract)
        with self.assertRaisesRegex(ValueError,"already"): writer.commit(writer.records[0])
        writer.close()
        path = self.path/"attempts.jsonl"
        with path.open("ab") as handle: handle.write(path.read_bytes())
        with self.assertRaisesRegex(ValueError,"Duplicate"): TeacherJournal(self.path,self.contract)

    def test_partial_tail_preserved_before_explicit_recovery(self):
        collect(self.contract,self.path,max_new_records=2)
        raw = b'{"broken":\xff'
        with (self.path/"attempts.jsonl").open("ab") as handle: handle.write(raw)
        with self.assertRaisesRegex(ValueError,"partial tail"): TeacherJournal(self.path,self.contract)
        resumed = collect(self.contract,self.path,recover_tail=True)
        self.assertEqual(len(resumed.records),108)
        backup = next(self.path.glob("*.partial-tail-*.bin"))
        self.assertEqual(backup.read_bytes(),raw)
        metadata = json.loads(Path(str(backup)+".json").read_text())
        self.assertEqual(metadata["bytes"],len(raw))
        self.assertIn("Uncommitted",metadata["reason"])

    def test_complete_corrupt_line_is_not_repaired(self):
        collect(self.contract,self.path,max_new_records=1)
        with (self.path/"attempts.jsonl").open("ab") as handle: handle.write(b"corrupt\n")
        before = (self.path/"attempts.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError,"Complete corrupt"): TeacherJournal(self.path,self.contract,recover_tail=True)
        self.assertEqual(before,(self.path/"attempts.jsonl").read_bytes())

    def test_keyboard_interrupt_retains_start_and_visible_reexecution(self):
        def interrupted(*args): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): collect(self.contract,self.path,teacher=interrupted)
        journal = collect(self.contract,self.path)
        self.assertEqual(len(journal.records),108)
        self.assertEqual(sum(e["event"] == "execution_started" for e in journal.events),109)
        interrupted_event = next(e for e in journal.events if e["event"] == "execution_interrupted")
        self.assertIn("unknown",interrupted_event["cost_boundary"])

    def test_malformed_teacher_return_is_recorded_error(self):
        row = execute_attempt(self.contract,self.protocol["items"][0],0,0,lambda *args:("trace",42,"END"))
        self.assertEqual(row["error"]["type"],"ValueError")
        self.assertEqual(row["cost"]["api_calls"],0)

    def test_all_filter_reasons_wrong_candidates_retained(self):
        _,pool,_ = self.collected()
        self.assertEqual(len(pool["candidates"]),36)
        self.assertEqual(pool["candidate_correctness"],{"1":12,"0":24})
        for reason in ("empty_answer","overlength_answer","unsupported_answer","missing_END","teacher_error","duplicate_candidate"):
            self.assertIn(reason,pool["reason_counts"])
        self.assertEqual(pool["reason_counts"]["teacher_error"],12)
        self.assertEqual(pool["reason_counts"]["duplicate_candidate"],12)
        self.assertEqual(pool["cost"]["model_forward_calls"],0)

    def test_malformed_ids_and_nontrain_source_fail_audit(self):
        row = execute_attempt(self.contract,self.protocol["items"][0],0,0)
        row.pop("payload_sha256"); row["final_token_ids"][0] = SPECIAL["pad"]
        malformed = attach_digest(row)
        audit = filter_pool([malformed],self.protocol)["audit"][0]
        self.assertIn("malformed_answer_ids",audit["reasons"])
        row = execute_attempt(self.contract,self.protocol["items"][12],0,0)
        self.assertIn("source_group_or_split_leakage",filter_pool([row],self.protocol)["audit"][0]["reasons"])

    def test_verifier_cannot_read_reference_or_mode_labels(self):
        row = {"final_answer":"red blue","teacher_mode":"wrong","score":0}
        item = copy.deepcopy(self.protocol["items"][0]); item["reference"] = "green yellow"
        self.assertEqual(verifier_score(row,item),1)
        row["final_answer"] = "green yellow"
        self.assertEqual(verifier_score(row,item),0)

    def test_selection_pool_score_and_content_fingerprints(self):
        _,pool,selection = self.collected()
        for arm in selection["arms"].values(): self.assertEqual(len(arm),12)
        self.assertEqual(selection["summaries"]["top"]["correct"],12)
        self.assertEqual(selection["summaries"]["top"]["supervised_tokens"],42)
        self.assertEqual(selection["summaries"]["length_random"]["supervised_tokens"],42)
        self.assertEqual(selection["length_stratum_unavailable"],[])
        self.assertTrue(all(s["eligible"] == 2 and s["score_values"] == [0,1] for s in selection["length_strata"].values()))
        tampered = copy.deepcopy(pool); tampered["candidates"][0]["score"] = 0
        with self.assertRaisesRegex(ValueError,"pool changed"): select_datasets(tampered,self.protocol)
        tampered["identity"] = canonical_hash({k:v for k,v in tampered.items() if k != "identity"})
        with self.assertRaisesRegex(ValueError,"scorer"): select_datasets(tampered,self.protocol)

    def test_seeded_selection_order_invariance_and_global_coverage(self):
        _,pool,selection = self.collected()
        permuted = copy.deepcopy(pool); permuted["candidates"].reverse()
        permuted["identity"] = canonical_hash({k:v for k,v in permuted.items() if k != "identity"})
        second = select_datasets(permuted,self.protocol)
        for arm in ("top","random","length_random"):
            self.assertEqual(selection["arms"][arm],second["arms"][arm])
        self.assertEqual(selection["global"]["6"]["prompt_coverage"],.5)
        self.assertEqual(selection["global"]["6"]["difficulty"],{"two-word":6})
        self.assertEqual(selection["global"]["12"]["prompt_coverage"],1.)

    def test_missing_prompt_pool_rejects_instead_of_fallback(self):
        _,pool,_ = self.collected()
        item = pool["candidates"][0]["item_id"]
        pool["candidates"] = [r for r in pool["candidates"] if r["item_id"] != item]
        pool["identity"] = canonical_hash({k:v for k,v in pool.items() if k != "identity"})
        with self.assertRaisesRegex(ValueError,"No accepted"): select_datasets(pool,self.protocol)

    def test_ownership_shift_termination_and_padding(self):
        _,_,selection = self.collected()
        batch = training_batch(selection["arms"]["random"])
        count = int((batch["labels"][:,1:] != -100).sum())
        self.assertEqual(count,selection["summaries"]["random"]["supervised_tokens"])
        self.assertTrue((batch["labels"][~batch["attention_mask"]] == -100).all())
        self.assertEqual(int((batch["labels"] == SPECIAL["end"]).sum()),12)
        logits = torch.randn(*batch["input_ids"].shape,22,dtype=torch.float64,requires_grad=True)
        loss,_ = token_loss_sum(logits,batch["labels"]); loss.backward()
        ignored = batch["labels"][:,1:] == -100
        self.assertEqual(float(logits.grad[:,:-1][ignored].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,:-1][~ignored].abs().sum()),0.)

    def test_actual_sequence_student_changes_and_gradient_reaches_backbone(self):
        _,_,selection = self.collected()
        model,fit = fit_student(selection["arms"]["top"],self.protocol,1101,steps=2)
        self.assertNotEqual(fit["initial_state_sha256"],fit["final_state_sha256"])
        self.assertEqual(fit["valid_supervised_token_presentations"],84)
        for name in ("token.weight","blocks.0.attn.q.weight","blocks.0.attn.v.weight"):
            self.assertGreater(fit["first_backward"][name],0.)
        self.assertEqual(state_digest(model),fit["final_state_sha256"])

    def test_actual_generation_seeded_streams_and_independent_regrading(self):
        model = make_student(1101,self.protocol)
        first = evaluate_student(model,self.protocol,1101,"baseline")
        second = evaluate_student(model,self.protocol,1101,"other-label")
        self.assertEqual([r["generated_ids"] for r in first["rows"]],[r["generated_ids"] for r in second["rows"]])
        self.assertEqual(len(first["rows"]),120)
        for row in first["rows"]:
            expected = [VOCAB[w] for w in row["reference"].split()]+[SPECIAL["end"]]
            self.assertEqual(row["correct"],row["generated_ids"] == expected)
            self.assertEqual(row["cost"]["valid_generated_tokens"],len(row["generated_ids"]))

    def test_evaluation_failures_are_retained_in_denominators(self):
        class Broken:
            def __call__(self,ids): raise RuntimeError("local forward failure")
        result = evaluate_student(Broken(),self.protocol,1101,"broken")
        self.assertEqual(len(result["rows"]),120)
        self.assertTrue(all(r["error"] and r["stop"] == "error" and not r["correct"] for r in result["rows"]))
        self.assertEqual(result["summaries"]["test/greedy"]["n"],4)
        self.assertEqual(result["summaries"]["test/greedy"]["correct"],0.)

    def test_replay_normalizes_only_known_executor_alias(self):
        from scripts.verify_teacher_data import measured_semantics
        imported = {"actual_executor":"dongxi_llms.teacher_data_lab:programmatic_teacher","tokens":[1,2]}
        module_run = {"actual_executor":"__main__:programmatic_teacher","tokens":[1,2]}
        unrelated = {"actual_executor":"other_teacher:programmatic_teacher","tokens":[1,2]}
        self.assertEqual(measured_semantics(imported),measured_semantics(module_run))
        self.assertNotEqual(measured_semantics(imported),measured_semantics(unrelated))
        self.assertNotEqual(measured_semantics(imported),measured_semantics({**module_run,"tokens":[1,3]}))


if __name__ == "__main__": unittest.main()

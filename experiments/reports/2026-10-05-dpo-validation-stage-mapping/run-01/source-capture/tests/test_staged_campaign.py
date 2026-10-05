"""Independent non-launching campaign and bounded leaf supervision contracts."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms import campaign_supervisor as supervisor
from dongxi_llms import staged_campaign as campaign
from dongxi_llms.run_identity import canonical_hash


def clear_scan(**kwargs):
    return {"source":"injected-independent-test","conflicts":[],"unreadable":0,"scanned":1,
            "raw_command_lines_retained":False,"environments_read":False}


def real_or_large_memory():
    return {"available_bytes":100*supervisor.GIB,"source":"injected-independent-test-not-actual-host"}


class CampaignTests(unittest.TestCase):
    def test_preparation_does_not_spawn_or_load_and_every_model_outcome_is_null(self):
        with patch("subprocess.Popen") as popen, patch("subprocess.run") as run:
            value=campaign.prepare_campaign()
        popen.assert_not_called();run.assert_not_called()
        self.assertEqual(value["jobs_started"],0);self.assertEqual(value["actual_genealogy"],[])
        for row in value["stage_rows"]:
            self.assertEqual(row["status"],"pending-external-evidence")
            self.assertTrue(all(v is None for v in row["actual"].values()))
            self.assertIsNone(row["launch_command"])
            self.assertTrue(all(g["status"]=="pending" and g["evidence"] is None for g in row["gates"]))

    def test_all_branches_dependencies_and_parent_starts_are_distinct(self):
        rows=campaign.stage_rows();index={r["id"]:r for r in rows}
        self.assertEqual(len(index),len(rows));self.assertGreaterEqual(len(rows),40)
        visited=set()
        for row in rows:
            self.assertTrue(set(row["prerequisite_stage_ids"])<=visited)
            visited.add(row["id"])
        for name in ("assistant-dpo-pilot","assistant-chosen-sft-pilot"):
            self.assertEqual(index[name]["proposed_weight_start"],"assistant-selected-sft-parent")
        for group in (4,8):
            self.assertEqual(index[f"reasoning-rlvr-g{group}-pilot"]["proposed_weight_start"],"pinned-instruct06")
        for arm in ("control","half-lr"):
            self.assertIn("fresh-random",index[f"story-{arm}-pilot"]["proposed_weight_start"])
        self.assertEqual(index["assistant-selected-sft-parent"]["proposed_weight_start"],"assistant-sft-full-pilot")

    def test_valid_targets_are_not_inferred_from_padded_or_attempted_token_geometry(self):
        rows={r["id"]:r for r in campaign.stage_rows()}
        b=rows["story-control-pilot"]["proposed_budget"]
        self.assertEqual(b["maximum_padded_training_positions"],229376000)
        self.assertEqual(b["valid_training_target_cap"],50000000)
        self.assertIsNone(rows["story-control-pilot"]["actual"]["valid_training_targets"])
        for group in (4,8):
            b=rows[f"reasoning-rlvr-g{group}-pilot"]["proposed_budget"]
            self.assertEqual(b["maximum_attempted_response_tokens"],16*group*64)
            self.assertIsNone(b["maximum_padded_training_positions"])

    def test_story_rubric_and_logical_identity_are_frozen_but_model_interface_is_pending(self):
        contracts=campaign.evaluation_contracts(campaign.ROOT)
        for c in contracts.values():
            self.assertEqual(c["logical_contract_sha256"],canonical_hash({k:v for k,v in c.items() if k!="logical_contract_sha256"}))
        s=contracts["story-publication-v1"]
        self.assertEqual(len(s["items"]),12);self.assertEqual(len(s["decoding"]),4)
        self.assertEqual(s["predetermined_updates"],[0,400,4000,8000,14000])
        self.assertEqual(set(s["rubric"]),{"grammar","entity_object_consistency","causal_continuity","repetition","ending"})
        self.assertTrue(all(len(v)==3 for v in s["rubric"].values()))
        self.assertIsNone(s["tokenizer_declaration"]["observed_encoding_sha256"])

    def test_reasoning_train_problem_overlap_is_not_heldout_and_toggles_are_not_weights(self):
        c=campaign.evaluation_contracts(campaign.ROOT)["reasoning-panel-v1"]
        overlapping=[x for x in c["items"] if x["rlvr_train_problem_overlap"]]
        self.assertTrue(overlapping)
        self.assertTrue(all(x["campaign_role"]=="development-or-seen-diagnostic" for x in overlapping))
        self.assertEqual(len(c["items"]),20)
        self.assertIn("same-instruct/chat/thinking-enabled",c["input_rows"])
        self.assertIn("faithfulness",c["limits"])

    def test_history_is_linked_not_relabelled_a_new_training_comparison(self):
        value=campaign.prepare_campaign();a=value["historical_anchor"]
        self.assertEqual(a["completion"]["completed_updates"],14000)
        self.assertEqual(a["completion"]["cumulative_targets"],48839975)
        self.assertEqual(a["launch_exit"],0)
        self.assertIn("pending",a["claim"])


class SupervisorTests(unittest.TestCase):
    def run_child(self,mode,**kwargs):
        with tempfile.TemporaryDirectory() as directory:
            result=supervisor.supervise_fixture(mode,Path(directory)/"child",memory_probe=real_or_large_memory,
                                                conflict_probe=clear_scan,**kwargs)
            events=[json.loads(x) for x in Path(result["journal"]["path"]).read_text().splitlines()]
            return result,events

    def test_success_nonzero_and_actual_signal_exit_are_retained(self):
        for mode,expected in (("success",0),("exit7",7),("self-signal",-signal.SIGUSR1)):
            result,events=self.run_child(mode)
            self.assertEqual(result["actual_exit_code"],expected)
            self.assertEqual(events[-1]["event"],"actual-exit")
            self.assertEqual(events[-1]["exit_code"],expected)
            self.assertTrue(any(x["event"]=="child-started" for x in events))

    def test_external_deadline_kills_only_the_owned_ignoring_child_and_reaps(self):
        result,events=self.run_child("ignore-term",limits=supervisor.Limits(seconds=.2))
        self.assertEqual(result["stop_reason"],"external-deadline")
        self.assertEqual(result["actual_exit_code"],-signal.SIGKILL)
        requests=[x for x in events if x["event"]=="signal-requested"]
        self.assertEqual([x["signal"] for x in requests],[signal.SIGTERM,signal.SIGKILL])
        self.assertEqual({x["child_pid"] for x in requests},{result["child_pid"]})
        self.assertLess(result["supervised_seconds"],1.5)

    def test_limits_and_unknown_children_cannot_downgrade_or_launch(self):
        for limits in (supervisor.Limits(reserve_bytes=24*supervisor.GIB),supervisor.Limits(seconds=float("nan")),
                       supervisor.Limits(seconds=0),supervisor.Limits(seconds=30),supervisor.Limits(sampling_seconds=.5)):
            with self.assertRaises(ValueError):limits.validate()
        with patch("subprocess.Popen") as spawn:
            with self.assertRaises(ValueError):supervisor.supervise_fixture("train",Path("unused-child"))
        spawn.assert_not_called()

    def test_preflight_memory_or_conflict_refusal_never_signals_discovered_processes(self):
        scan=clear_scan();scan["conflicts"]=[{"pid":987654321,"rss_bytes":5*supervisor.GIB,"reasons":["large-resident-process"]}]
        for memory,inspect,reason in ((lambda:{"available_bytes":24*supervisor.GIB,"source":"injected"},clear_scan,"host-reserve-below-threshold"),
                                     (real_or_large_memory,lambda **kw:deepcopy(scan),"conflicting-or-uninspectable-process")):
            with tempfile.TemporaryDirectory() as td,patch("subprocess.Popen") as spawn,patch("os.kill") as kill:
                r=supervisor.supervise_fixture("success",Path(td)/"refused",memory_probe=memory,conflict_probe=inspect)
                self.assertIsNone(r["child_pid"]);self.assertIsNone(r["actual_exit_code"])
                self.assertEqual(r["stop_reason"],reason)
            spawn.assert_not_called();kill.assert_not_called()

    def test_running_low_memory_observer_error_and_interrupt_cleanup_own_child(self):
        for control in ("low","error","interrupt"):
            count=0
            def probe():
                nonlocal count
                count+=1
                if count==1:return real_or_large_memory()
                if control=="low":return {"available_bytes":24*supervisor.GIB,"source":"injected"}
                if control=="error":raise OSError("deliberate probe failure")
                raise KeyboardInterrupt("deliberate observer interrupt")
            with tempfile.TemporaryDirectory() as td:
                r=supervisor.supervise_fixture("ignore-term",Path(td)/control,memory_probe=probe,conflict_probe=clear_scan)
                self.assertIsNotNone(r["child_pid"])
                self.assertIn(r["actual_exit_code"],(-signal.SIGTERM,-signal.SIGKILL))
                self.assertIn(r["stop_reason"],("host-reserve-below-threshold","supervisor-failure","observer-interrupted"))
                if control!="low":self.assertIsNotNone(r["failure"])

    def test_signal_journal_error_cannot_strand_a_child(self):
        original=supervisor.Journal.event
        def event(journal,kind,**fields):
            if kind=="signal-requested":raise OSError("deliberate signal journal failure")
            return original(journal,kind,**fields)
        with patch.object(supervisor.Journal,"event",event):
            r,_=self.run_child("ignore-term",limits=supervisor.Limits(seconds=.2))
        self.assertEqual(r["actual_exit_code"],-signal.SIGKILL)
        self.assertEqual(len(r["signal_journal_errors"]),2)

    def test_existing_output_is_preserved_and_raw_argument_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"existing";output.mkdir()
            with self.assertRaises(FileExistsError):supervisor.supervise_fixture("success",output)
            scan=clear_scan();scan["cmdline"]="secret text must not enter journal"
            r=supervisor.supervise_fixture("success",Path(td)/"invalid",memory_probe=real_or_large_memory,conflict_probe=lambda **kw:scan)
            self.assertIsNone(r["child_pid"])
            self.assertNotIn("secret text",Path(r["journal"]["path"]).read_text())

    def test_process_scan_is_sanitized_and_reports_unrecognized_large_rss(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for pid,rss,args in (("98765",5*1024**2,["python","private-argument"]),
                                 ("98766",1024,["python","-m","vllm","--api-key","private-secret"]),
                                 ("98767",1024,["python","unrelated-small-process"])):
                p=root/pid;p.mkdir();(p/"status").write_text(f"VmRSS:\t{rss} kB\n")
                (p/"cmdline").write_bytes(b"\0".join(a.encode() for a in args))
            r=supervisor.conflicts(proc_root=root)
            self.assertEqual({x["pid"] for x in r["conflicts"]},{98765,98766})
            self.assertNotIn("private",json.dumps(r));self.assertFalse(r["environments_read"])


if __name__=="__main__":unittest.main()

"""Original bounded CPU controls for actual DPO semantic-validation charges."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_dpo_checkpoint_mode as mode
import test_dpo_runner_recovery as base
import test_dpo_work_budget as prior
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded

r=base.runner;ROOT=base.ROOT
SPEC="experiments/specs/2026-10-05-dpo-recovery-validation-budget.md"
COMPAT_SPEC="experiments/specs/2026-10-05-validation-budget-fixture-compatibility.md"
CAPS=dict(prior.CAPS)
BOUND=1024*1024
SOURCES=("scripts/run_chapter11_spark_dpo.py",SPEC,COMPAT_SPEC,"pyproject.toml","uv.lock",
    "src/dongxi_llms/training_snapshot.py","src/dongxi_llms/artifact_budget.py",
    "src/dongxi_llms/work_budget.py","src/dongxi_llms/run_identity.py",
    "src/dongxi_llms/batched_cache_lab.py","src/dongxi_llms/dpo_lab.py",
    "src/dongxi_llms/decoder_lab.py","src/dongxi_llms/grpo_lab.py",
    "src/dongxi_llms/pretraining_lab.py","src/dongxi_llms/sampling_likelihood_lab.py",
    "tests/test_dpo_validation_budget.py","tests/test_dpo_runner_recovery.py",
    "tests/test_dpo_checkpoint_mode.py","tests/test_dpo_work_budget.py",
    "tests/test_dpo_snapshot_artifacts.py","tests/test_training_snapshot.py",
    "tests/test_training_snapshot_failure_paths.py","tests/test_snapshot_artifact_budget.py")


def fixture(limits=CAPS):
    values=list(mode.make_mode(True));options=deepcopy(values[5])
    for name in ("tests/test_dpo_validation_budget.py",SPEC,"tests/test_dpo_work_budget.py",COMPAT_SPEC,
                 "src/dongxi_llms/work_budget.py","src/dongxi_llms/run_identity.py"):
        options["sources"][name]=r.file_digest(ROOT/name)
    options["work_budget"]=r.work_budget_contract(limits,BOUND)
    values[4]=r.make_recovery_contract(model=values[0],reference=values[1],optimizer=values[2],**options)
    values[5]=options
    return values


def create(path,values,invocation="original"):
    return WorkLedger.create(path,limits=values[4]["work_budget"]["limits"],
        contract_sha256=canonical_hash(values[4]),max_bytes=BOUND,invocation_id=invocation)


def reopen(path,values,invocation="fresh"):
    return WorkLedger.open(path,limits=values[4]["work_budget"]["limits"],
        contract_sha256=canonical_hash(values[4]),max_bytes=BOUND,invocation_id=invocation)


def train(directory,values,work,initial=None):
    return r.train_completed_updates(*values[:4],base.ENCODED,contract=values[4],output=directory,
        invocation_id="validation-cpu-fixture",pad_id=1,device="cpu",checkpoint_every=3,
        max_bytes=base.LIMIT,work_ledger=work,**(initial or {}))


def load(receipt,values,work):
    return r.load_snapshot(receipt["path"],expected_sha256=receipt["payload_sha256"],
        expected_bytes=receipt["payload_bytes"],expected_contract=values[4],max_bytes=base.LIMIT,
        validate_payload=lambda payload:r.validate_dpo_payload(payload,values[4],work))


def numeric(payload):
    return {key:value for key,value in payload["state"].items() if key not in
            ("resume_parent","work_ledger","snapshot_artifact_ledger")}


def one_update_payload(values,work):
    row=r.completed_dpo_update(*values[:4],base.ENCODED,pad_id=1,accumulation=2,
        beta=.2,device="cpu",update=1,work_ledger=work)
    state=r.snapshot_state(*values[:4],completed=1,counters=row["work"],history=[row],work_ledger=work)
    return dict(state=state,phase="completed",completed_updates=1)


def interrupted(directory,values,work):
    original=r.append_metric
    def fail(path,row):
        if row.get("update")==3:raise OSError("authored metric failure after durable update3")
        return original(path,row)
    try:
        with patch.object(r,"append_metric",side_effect=fail):train(directory,values,work)
    except OSError as error:
        return base.checkpoint_receipt(Path(directory)/"checkpoints/completed-000003.pt"),dict(
            type=type(error).__name__,message=str(error),deliberately_injected=True)
    raise AssertionError("Predeclared interrupted commit absent")


def failed_later_validation(receipt,values,work):
    retained=load(receipt,values,work)
    bad=deepcopy(retained);bad["state"]["sampler_rng"]=bad["state"]["sampler_rng"].float()
    before=work.snapshot()
    try:r.validate_dpo_payload(bad,values[4],work)
    except ValueError as error:
        after=work.snapshot()
        return dict(type=type(error).__name__,message=str(error),deliberately_injected=True,
                    before=before,after=after)
    raise AssertionError("Predeclared malformed RNG accepted")


def child(receipt_path,contract_path,journal_path,output):
    values=fixture();expected=json.loads(Path(contract_path).read_text())
    if values[4]!=expected:raise ValueError("Fresh actual source/numerical contract differs")
    receipt=json.loads(Path(receipt_path).read_text());output=Path(output);output.mkdir(mode=0o700)
    with reopen(journal_path,values) as work:
        retained=load(receipt,values,work);after_callback=work.snapshot()
        completed,counters,history=r.restore_dpo_payload(retained,contract=values[4],model=values[0],
            reference=values[1],optimizer=values[2],sampler=values[3],work_ledger=work)
        after_restore=work.snapshot()
        result=train(output,values,work,dict(completed=completed,counters=counters,history=history,parent_checkpoint=receipt))
        final=load(result["checkpoint"],values,work)
        for name,value in (("result.json",result),("after-callback.json",after_callback),
                ("after-restore.json",after_restore),("after-final-load.json",work.snapshot()),
                ("numerical.json",dict(sha256=digest(numeric(final)),restored_update=completed,
                    next_history_row=result["history"][3],checkpointing=values[0].is_gradient_checkpointing))):
            r.write_exclusive_json(output/name,value)


class DPOValidationBudgetTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(prefix="dongxi-dpo-validation-")
        self.addCleanup(directory.cleanup);self.root=Path(directory.name)

    def directory(self,name):
        path=self.root/name;path.mkdir(mode=0o700);return path

    def work(self,values,name="work"):
        work=create(self.directory(name)/"work.jsonl",values);self.addCleanup(work.close);return work

    def test_v2_exact19_dimensions_and_old14_refusal(self):
        self.assertEqual(len(r.BUDGET_KEYS),19)
        self.assertEqual(r.RECOVERY_BUDGET_KEYS,("recovery_validation_operations","recovery_history_rows",
            "recovery_tensor_elements","recovery_rng_states","recovery_sampler_draws"))
        self.assertEqual(r.work_budget_contract(CAPS,BOUND)["schema"],"dongxi-dpo-logical-work-v2")
        with self.assertRaises(ValueError):r.work_budget_contract({key:CAPS[key] for key in r.BUDGET_KEYS[:14]},BOUND)
        values=fixture();work=self.work(values);payload=one_update_payload(values,work)
        wrong=deepcopy(values[4]);wrong["work_budget"]["schema"]="dongxi-dpo-logical-work-v1"
        with patch.object(r,"state_digest") as hashed,patch.object(torch,"randint") as drawn,self.assertRaises(ValueError):
            r.validate_dpo_payload(payload,wrong,work)
        hashed.assert_not_called();drawn.assert_not_called()

    def test_estimator_matches_independent_layout_arithmetic_and_no_tensor_use(self):
        values=fixture();contract=values[4]
        p=sum(math.prod(item["shape"]) for item in contract["policy_shapes"].values())
        o=sum(math.prod(item["shape"]) for item in contract["optimizer_shapes"])
        for completed in (0,1,3,6):
            with patch.object(r,"state_digest") as hashed,patch.object(torch,"isfinite") as scanned,patch.object(torch,"randint") as drawn:
                actual=r.recovery_validation_upper(contract,completed)
            self.assertEqual(actual,dict(recovery_validation_operations=1,recovery_history_rows=completed,
                recovery_tensor_elements=3*p+(3*o+len(contract["optimizer_shapes"]) if completed else 0)+4*contract["torch_rng_bytes"],
                recovery_rng_states=2,recovery_sampler_draws=completed*2))
            hashed.assert_not_called();scanned.assert_not_called();drawn.assert_not_called()

    def test_each_zero_cap_refuses_before_tensor_replay_rng_or_application(self):
        for dimension in r.RECOVERY_BUDGET_KEYS:
            with self.subTest(dimension=dimension):
                values=fixture(dict(CAPS,**{dimension:0}));work=self.work(values,dimension)
                payload=one_update_payload(values,work);before=work.snapshot()
                with (patch.object(r,"state_digest") as hashed,patch.object(torch,"isfinite") as scanned,
                        patch.object(torch,"randint") as drawn,patch.object(torch,"Generator") as rng,
                        patch.object(values[0],"load_state_dict") as applied,self.assertRaises(WorkBudgetExceeded)):
                    r.restore_dpo_payload(payload,contract=values[4],model=values[0],reference=values[1],
                        optimizer=values[2],sampler=values[3],work_ledger=work)
                for spy in (hashed,scanned,drawn,rng,applied):spy.assert_not_called()
                self.assertEqual(work.snapshot()["reserved"],before["reserved"])

    def test_malformed_history_tensor_and_rng_reserve_and_retain_failure(self):
        for name in ("history","tensor","rng"):
            with self.subTest(name=name):
                values=fixture();work=self.work(values,name);payload=one_update_payload(values,work)
                if name=="history":payload["state"]["history"][0]["indices"][0]=True
                elif name=="tensor":
                    key=next(iter(payload["state"]["reference"]))
                    payload["state"]["reference"][key]=torch.zeros(1)
                else:payload["state"]["sampler_rng"]=payload["state"]["sampler_rng"].float()
                with patch.object(values[0],"load_state_dict") as applied,self.assertRaises(ValueError):
                    r.restore_dpo_payload(payload,contract=values[4],model=values[0],reference=values[1],
                        optimizer=values[2],sampler=values[3],work_ledger=work)
                applied.assert_not_called();summary=work.snapshot()
                for key,amount in r.recovery_validation_upper(values[4],1).items():
                    self.assertEqual(summary["reserved"][key],amount)
                    self.assertEqual(summary["completed"][key],0)
                    self.assertEqual(summary["uncertain_upper"][key],amount)
                self.assertEqual(len(summary["failed_tickets"]),1)

    def test_bad_tensor_metadata_refuses_before_reference_hash(self):
        values=fixture();work=self.work(values);payload=one_update_payload(values,work)
        key=next(iter(payload["state"]["reference"]));payload["state"]["reference"][key]=torch.zeros(100000)
        with patch.object(r,"state_digest") as hashed,self.assertRaises(ValueError):r.validate_dpo_payload(payload,values[4],work)
        hashed.assert_not_called();self.assertEqual(work.snapshot()["reserved"]["recovery_validation_operations"],1)

    def test_nonfinite_tensor_and_negative_Adam_preserve_panel_charge(self):
        for name in ("nonfinite-policy","negative-Adam","malformed-RNG-bytes"):
            with self.subTest(name=name):
                values=fixture();work=self.work(values,name);payload=one_update_payload(values,work)
                if name=="nonfinite-policy":
                    key=next(iter(payload["state"]["policy"]))
                    payload["state"]["policy"][key]=payload["state"]["policy"][key].clone()
                    payload["state"]["policy"][key].flatten()[0]=float("nan")
                elif name=="negative-Adam":
                    moments=next(iter(payload["state"]["optimizer"]["state"].values()))
                    moments["exp_avg_sq"]=moments["exp_avg_sq"].clone()
                    moments["exp_avg_sq"].flatten()[0]=-1.
                else:payload["state"]["sampler_rng"]=torch.zeros_like(payload["state"]["sampler_rng"])
                with patch.object(values[0],"load_state_dict") as applied,self.assertRaises(ValueError):
                    r.restore_dpo_payload(payload,contract=values[4],model=values[0],reference=values[1],
                        optimizer=values[2],sampler=values[3],work_ledger=work)
                applied.assert_not_called()
                self.assertEqual(work.snapshot()["reserved"]["recovery_validation_operations"],1)
                self.assertEqual(work.snapshot()["completed"]["recovery_validation_operations"],0)
                self.assertTrue(work.snapshot()["failed_tickets"])

    def test_real_history_replay_draws_and_reference_hash_are_counted(self):
        values=fixture();work=self.work(values);payload=one_update_payload(values,work)
        original=torch.randint;original_hash=r.state_digest
        with patch.object(torch,"randint",wraps=original) as draws,patch.object(r,"state_digest",wraps=original_hash) as hashes:
            r.validate_dpo_payload(payload,values[4],work)
        self.assertEqual(draws.call_count,2);self.assertEqual(hashes.call_count,1)
        summary=work.snapshot()
        for key,amount in r.recovery_validation_upper(values[4],1).items():
            self.assertEqual(summary["reserved"][key],amount)
            self.assertEqual(summary["completed"][key],amount)

    def test_validation_fsync_failure_precedes_all_semantic_work(self):
        values=fixture();work=self.work(values);payload=one_update_payload(values,work)
        with (patch("dongxi_llms.work_budget.os.fsync",side_effect=OSError("authored validation-reservation fsync failure")),
                patch.object(r,"state_digest") as hashes,patch.object(torch,"isfinite") as finite,
                patch.object(torch,"randint") as draws,patch.object(torch,"Generator") as rng,self.assertRaises(Exception)):
            r.validate_dpo_payload(payload,values[4],work)
        for spy in (hashes,finite,draws,rng):spy.assert_not_called()
        self.assertTrue(work.poisoned)

    def test_structural_and_hostile_layout_bounds_before_tensor_work(self):
        values=fixture();contract=values[4]
        invalid=[]
        for key,value in (("updates",1001),("accumulation",17),("sample_work",[{}]*4097),
                          ("cuda_rng_count",17),("torch_rng_bytes",True)):
            bad=deepcopy(contract);bad[key]=value;invalid.append(bad)
        bad=deepcopy(contract);bad["policy_shapes"]={"x":{"shape":[2]*9,"dtype":"torch.float32"}};invalid.append(bad)
        bad=deepcopy(contract);bad["policy_shapes"]={"x":{"shape":[2**31-1]*8,"dtype":"torch.float32"}};invalid.append(bad)
        for bad in invalid:
            with patch.object(torch,"randint") as drawn,patch.object(r,"state_digest") as hashed,self.assertRaises(ValueError):r.recovery_validation_upper(bad,0)
            drawn.assert_not_called();hashed.assert_not_called()
        work=self.work(values);payload=one_update_payload(values,work);payload["state"]["history"]*=1001
        with patch.object(r,"state_digest") as hashed,self.assertRaises(ValueError):r.validate_dpo_payload(payload,contract,work)
        hashed.assert_not_called();self.assertEqual(work.snapshot()["reserved"]["recovery_validation_operations"],0)

    def test_unbudgeted_checkpoint_on_api_preserves_numerical_history(self):
        values=mode.make_mode(True);result=base.train_fixture(self.directory("unbudgeted"),values)
        retained=base.payload_for(result["checkpoint"],values[4])
        self.assertEqual(retained["state"]["completed"],6)
        self.assertNotIn("work_ledger",retained["state"])
        self.assertTrue(values[0].is_gradient_checkpointing)

    def test_actual_budgeted_unbudgeted_endpoint_parity(self):
        ordinary=mode.make_mode(True);first=base.train_fixture(self.directory("ordinary"),ordinary)
        expected=base.payload_for(first["checkpoint"],ordinary[4])
        values=fixture();work=self.work(values);result=train(self.directory("budgeted"),values,work)
        actual=load(result["checkpoint"],values,work)
        self.assertEqual(digest(numeric(expected)),digest(numeric(actual)))
        self.assertEqual(work.snapshot()["reserved"]["recovery_validation_operations"],4)
        self.assertEqual(work.snapshot()["reserved"]["recovery_history_rows"],15)
        self.assertEqual(work.snapshot()["reserved"]["recovery_sampler_draws"],30)

    def test_commit_snapshot_carries_its_own_validation_and_duplicate_restore_costs(self):
        values=fixture();work=self.work(values);result=train(self.directory("full"),values,work)
        committed=work.snapshot();self.assertEqual(committed["reserved"]["recovery_validation_operations"],3)
        retained=load(result["checkpoint"],values,work)
        self.assertEqual(retained["state"]["work_ledger"]["reserved"]["recovery_validation_operations"],3)
        after_load=work.snapshot()
        r.restore_dpo_payload(retained,contract=values[4],model=values[0],reference=values[1],
            optimizer=values[2],sampler=values[3],work_ledger=work)
        after_restore=work.snapshot()
        self.assertEqual(after_load["reserved"]["recovery_validation_operations"],4)
        self.assertEqual(after_restore["reserved"]["recovery_validation_operations"],5)
        self.assertEqual(after_restore["reserved"]["recovery_history_rows"],21)

    def test_exhausted_callback_or_restore_refuses_without_replaying_state(self):
        values=fixture(dict(CAPS,recovery_validation_operations=3));work=self.work(values)
        result=train(self.directory("full"),values,work)
        with (patch.object(r,"state_digest") as hashed,patch.object(torch,"randint") as drawn,
                patch.object(values[0],"load_state_dict") as applied,self.assertRaises(WorkBudgetExceeded)):
            load(result["checkpoint"],values,work)
        hashed.assert_not_called();drawn.assert_not_called();applied.assert_not_called()

    def test_fresh_process_three_to_six_retains_later_validation_failure(self):
        ordinary=mode.make_mode(True);full=base.train_fixture(self.directory("full"),ordinary)
        expected=base.payload_for(full["checkpoint"],ordinary[4])
        values=fixture();workpath=self.directory("journal")/"work.jsonl"
        with create(workpath,values) as work:
            receipt,error=interrupted(self.directory("interrupted"),values,work)
            failed=failed_later_validation(receipt,values,work)
        r.write_exclusive_json(self.root/"receipt.json",receipt)
        r.write_exclusive_json(self.root/"contract.json",values[4])
        command=[sys.executable,str(Path(__file__).resolve()),"--child",str(self.root/"receipt.json"),
                 str(self.root/"contract.json"),str(workpath),str(self.root/"fresh")]
        process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH="src:tests",CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1"),capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        after=json.loads((self.root/"fresh/after-final-load.json").read_text())
        result=json.loads((self.root/"fresh/numerical.json").read_text())
        self.assertEqual(result["sha256"],digest(numeric(expected)))
        self.assertEqual(result["restored_update"],3);self.assertTrue(result["checkpointing"])
        self.assertEqual(result["next_history_row"],full["history"][3])
        self.assertTrue(set(failed["after"]["failed_tickets"])<=set(after["failed_tickets"]))
        self.assertGreater(after["reserved"]["recovery_validation_operations"],failed["after"]["reserved"]["recovery_validation_operations"])
        self.assertEqual(after["reserved"]["train_updates"],6)


def collect_reference(destination):
    """Exclusive original CPU evidence; only fixed self-owned fixture children."""
    from datetime import datetime,timezone
    import hashlib
    import importlib.metadata
    import platform
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    before={name:r.file_digest(ROOT/name) for name in SOURCES};started=time.monotonic()
    result=dict(schema="dongxi-dpo-validation-budget-cpu-v1",status="running",
        created_utc=datetime.now(timezone.utc).isoformat(),command=list(sys.orig_argv),python_argv=list(sys.argv),
        source_sha256=before,commands=[],model_scale_jobs_started=0,production_ready=False,launch_authorized=False,
        environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
            executable=sys.executable,packages={name:importlib.metadata.version(name) for name in
            ("torch","transformers","tokenizers","peft")},cuda_visible_devices="",hf_offline=True,omp_num_threads=1))
    def save(name,value):r.write_exclusive_json(destination/name,value)
    def raw(name,value):
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        return dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
    def directory(name):
        path=destination/name;path.mkdir(mode=0o700);return path
    env=dict(os.environ,PYTHONPATH="src:tests",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",
             TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1")
    try:
        modules=("test_dpo_runner_recovery","test_dpo_checkpoint_mode","test_dpo_work_budget",
            "test_dpo_snapshot_artifacts","test_training_snapshot","test_training_snapshot_failure_paths",
            "test_snapshot_artifact_budget")
        for label,names in (("new-focused",("test_dpo_validation_budget",)),("existing-regressions",modules)):
            command=[sys.executable,"-m","unittest",*names,"-v"];measured=time.monotonic()
            process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            output=process.stdout+process.stderr;count=re.search(r"Ran (\d+) tests? in ([\d.]+)s",output)
            result["commands"].append(dict(command=command,exit_code=process.returncode,
                seconds=time.monotonic()-measured,tests_ran=int(count.group(1)) if count else None,
                unittest_seconds=float(count.group(2)) if count else None,raw_log=raw(label+".log",output.encode())))
        for name in SOURCES:
            path=ROOT/name
            if path.stat().st_size>4*1024**2:raise ValueError("Explicit captured-source file envelope exceeded")
            value=path.read_bytes()
            if hashlib.sha256(value).hexdigest()!=before[name]:raise ValueError("Source changed before archive capture")
            raw("source-capture/"+name,value)
        save("declared-fixture.json",dict(original=base.FIXTURE,limits=CAPS,mode="policy checkpointing ON; reference frozen/cache OFF",
            recovery_units="semantic verifier envelope, not generic loader/serialization/physical work"))
        ordinary=mode.make_mode(True);original=base.train_fixture(directory("unbudgeted-on"),ordinary)
        expected=base.payload_for(original["checkpoint"],ordinary[4]);save("unbudgeted-on-training.json",original)
        off=mode.make_mode(False);off_training=base.train_fixture(directory("unbudgeted-off"),off)
        off_saved=base.payload_for(off_training["checkpoint"],off[4]);save("unbudgeted-off-training.json",off_training)
        result["checkpoint_off_on_parity"]=mode.comparison(off_saved,expected)
        values=fixture();save("recovery-contract.json",values[4]);save("estimator-cursors.json",
            {str(number):r.recovery_validation_upper(values[4],number) for number in (0,1,3,6)})
        journal_directory=directory("complete-journal");journal=journal_directory/"work.jsonl"
        with create(journal,values) as work:
            complete=train(directory("budgeted-complete"),values,work);before_load=work.snapshot()
            saved=load(complete["checkpoint"],values,work);after_load=work.snapshot()
            expected_digest=digest(numeric(expected));actual_digest=digest(numeric(saved))
            if expected_digest!=actual_digest:raise AssertionError("Budgeted validation changed numerical endpoint")
            if saved["state"]["work_ledger"]["reserved"]["recovery_validation_operations"]!=3:
                raise AssertionError("Committed snapshot omitted its own validation charge")
        save("complete-training.json",complete);save("complete-work-before-final-load.json",before_load)
        save("complete-work-after-final-load.json",after_load)
        result["budgeted_unbudgeted_parity"]=dict(expected_numerical_sha256=expected_digest,
            actual_numerical_sha256=actual_digest,exact=True,checkpointing=True)
        later_values=fixture();later_journal=directory("recovery-journal")/"work.jsonl"
        with create(later_journal,later_values) as work:
            receipt,interruption=interrupted(directory("interrupted"),later_values,work)
            failure=failed_later_validation(receipt,later_values,work)
        save("interrupted-receipt.json",receipt);save("metric-interruption.json",interruption)
        save("later-failed-validation.json",failure)
        command=[sys.executable,str(Path(__file__).resolve()),"--child",str(destination/"interrupted-receipt.json"),
                 str(destination/"recovery-contract.json"),str(later_journal),str(destination/"fresh-replay")]
        measured=time.monotonic();process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
        result["fresh_replay_command"]=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-measured,
            raw_log=raw("fresh-replay.log",(process.stdout+process.stderr).encode()),deadline_seconds=30)
        if process.returncode:raise AssertionError("Fresh bounded replay child failed")
        fresh=json.loads((destination/"fresh-replay/numerical.json").read_text())
        after=json.loads((destination/"fresh-replay/after-final-load.json").read_text())
        callback=json.loads((destination/"fresh-replay/after-callback.json").read_text())
        restored=json.loads((destination/"fresh-replay/after-restore.json").read_text())
        if fresh["sha256"]!=expected_digest or fresh["restored_update"]!=3 or fresh["next_history_row"]!=original["history"][3]:
            raise AssertionError("Fresh completed3-to6 numerical state/next action changed")
        if not set(failure["after"]["failed_tickets"])<=set(after["failed_tickets"]):
            raise AssertionError("Fresh recovery refunded later validation failure")
        for key in r.BUDGET_KEYS:
            if callback["reserved"][key]<failure["after"]["reserved"][key]:
                raise AssertionError("Recovered work cursor rewound attempted spending")
        panel=r.recovery_validation_upper(values[4],3)
        if any(restored["reserved"][key]-callback["reserved"][key]!=amount for key,amount in panel.items()):
            raise AssertionError("Duplicate restore panel was not charged")
        result["fresh_process_replay"]=dict(expected_numerical_sha256=expected_digest,actual_numerical_sha256=fresh["sha256"],
            exact=True,restored_update=3,completed_updates=6,next_history_row_exact=True,checkpointing=fresh["checkpointing"],
            later_failed_tickets=failure["after"]["failed_tickets"],final_failed_tickets=after["failed_tickets"],
            reserved_before_child=failure["after"]["reserved"],reserved_after_child=after["reserved"],
            callback_restore_duplicate_costs=panel)
        result["status"]="pass" if all(item["exit_code"]==0 for item in result["commands"]) else "fail"
    except BaseException as error:
        result["status"]="fail";result["failure"]=dict(type=type(error).__name__,message=str(error))
    result["source_sha256_after"]={name:r.file_digest(ROOT/name) for name in SOURCES}
    result["sources_unchanged"]=before==result["source_sha256_after"]
    if not result["sources_unchanged"]:result["status"]="fail"
    result["seconds"]=time.monotonic()-started
    result["limitations"]=dict(units="logical semantic verifier panel envelopes, not measured CPU instructions/time",
        generic_shared_pre_callback_loader_checks="excluded/pending",serialization_hashing_journal_integrity_and_application_overhead="excluded",
        production_backend_authority_gpu_mac_hosted="not executed",all45_external_outcomes="null")
    save("verification.json",result)
    return result


if __name__=="__main__":
    if len(sys.argv)==6 and sys.argv[1]=="--child":child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=="--collect":
        result=collect_reference(sys.argv[2]);print(json.dumps({key:result[key] for key in ("status","sources_unchanged","seconds")}))
        raise SystemExit(0 if result["status"]=="pass" else 1)
    else:unittest.main()

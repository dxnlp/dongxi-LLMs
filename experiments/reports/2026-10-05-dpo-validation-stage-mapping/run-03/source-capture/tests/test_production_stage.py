"""Original authored receipt controls; no model, service, probe or process launch."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch

from dongxi_llms import production_stage as stage
from dongxi_llms.production_preflight import ADAPTERS, MAX_RESPONSE_BYTES, PreflightPolicy, verify_preflight
from dongxi_llms.run_identity import SPECIAL_IDS, canonical_hash


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def interface():
    value = {"schema_version": 1, "tokenizer": {"vocab_sha256": "1"*64, "encoding_sha256": "2"*64,
        "special_ids": {key: (2 if key == "eos_token_id" else None) for key in SPECIAL_IDS},
        "vocab_size": 3, "max_token_id": 2, "wrapper_settings": {}},
        "template_sha256": None, "generation_stop_ids": [2],
        "source": {"tokenizer_id": "authored-three-token-interface-not-gpt2-or-qwen", "tokenizer_revision": None},
        "implementation_class": "AuthoredFixtureNotTokenizer"}
    value["interface_sha256"] = canonical_hash({k: value[k] for k in
        ("schema_version", "tokenizer", "template_sha256", "generation_stop_ids")})
    return value


class Fixture:
    def __init__(self, directory, identifier="story-profile", root=stage.staged_campaign.ROOT):
        self.base = Path(directory); self.artifacts = self.base/"artifacts"; self.receipts = self.base/"receipts"
        self.artifacts.mkdir(); self.receipts.mkdir(); self.root = Path(root); self.identifier = identifier
        self.nonce = "b"*64; self.now = 100.2
        iface = interface()
        files = {"data": write_json(self.artifacts/"data.json", {"scope": stage.FIXTURE_SCOPE, "items": ["original tiny item"]}),
                 "interface": write_json(self.artifacts/"interface.json", iface)}
        row = stage._row(identifier)
        slots = row["proposed_weight_start"]
        slots = slots if type(slots) is list else [slots]
        parents = {}
        for index, slot in enumerate(slots):
            role = "parent" if index == 0 else f"parent{index}"
            files[role] = write_json(self.artifacts/f"{role}.json", {"schema": "authored-parent-v1",
                "scope": stage.FIXTURE_SCOPE, "stage_slot": slot, "interface_sha256": iface["interface_sha256"]})
            parents[slot] = role
        limits = {key: 10**12 for key in stage.LIMIT_KEYS}
        limits.update(external_seconds=60, reserve_bytes=25*1024**3, artifact_bytes=16*1024**2,
            artifact_entries=32, per_file_bytes=1024**2, snapshot_max_bytes=1024**2,
            hard_memory_bytes=2*1024**3, hard_pid_limit=8, max_valid_training_targets=50_000_000,
            max_prompt_positions=7, private_instance_id="dongxi-stage-"+"c"*32, owner_uid=1000)
        self.bindings = {"scope": stage.FIXTURE_SCOPE, "files": files, "parent_slots": parents,
            "interface_role": "interface", "limits": limits}
        self.preparation = stage.prepare_stage(identifier, bindings=self.bindings, artifact_root=self.artifacts, root=self.root)
        self.prerequisites = {}
        parent_manifest = canonical_hash({slot: self.preparation["bindings"]["files"][role]
            for slot, role in parents.items()})
        for index, predecessor in enumerate(row["prerequisite_stage_ids"]):
            body = {"schema": "dongxi-stage-prerequisite-v1", "scope": stage.FIXTURE_SCOPE,
                "stage_id": predecessor, "status": "fixture-complete", "preparation_sha256": "d"*64,
                "result_sha256": "e"*64, "parent_manifest_sha256": parent_manifest}
            ref = self.receipt(f"prerequisite{index}", body)
            self.prerequisites[predecessor] = stage.PrerequisiteRef(ref, "d"*64, "e"*64, parent_manifest)
        policy = PreflightPolicy(identifier, self.preparation["preparation_sha256"], self.nonce,
            limits["private_instance_id"], limits["owner_uid"], limits["hard_memory_bytes"], limits["hard_pid_limit"],
            limits["artifact_bytes"], limits["artifact_entries"], limits["external_seconds"])
        observations = [self.observation(policy, 1, 100.), self.observation(policy, 2, 100.1)]
        self.preflight = verify_preflight(policy, [json.dumps(v).encode() for v in observations], now=self.now)
        self.preflight_ref = self.receipt("preflight", self.preflight)
        self.refresh_approval()

    def observation(self, policy, sequence, time):
        return {"schema": "dongxi-preflight-observation-v1", "scope": stage.FIXTURE_SCOPE,
            "stage_id": policy.stage_id, "preparation_sha256": policy.preparation_sha256, "nonce": policy.observation_nonce,
            "sequence": sequence, "observed_at": time, "observer_seconds": .01,
            "host": {"available_bytes": 40*1024**3, "complete": True, "measurement": "injected-sample-not-continuous"},
            "boundary": {"adapter": ADAPTERS["containment"], "instance_id": policy.instance_id,
                "owner_uid": policy.owner_uid, "exclusive": True, "membership_complete": True, "members": [],
                "hard_memory_bytes": policy.hard_memory_bytes, "hard_pid_limit": policy.hard_pid_limit},
            "quota": {"adapter": ADAPTERS["artifact_quota"], "instance_id": policy.instance_id,
                "owner_uid": policy.owner_uid, "exclusive": True, "hard_bytes": policy.artifact_bytes, "hard_entries": policy.artifact_entries},
            "observer": {"adapter": ADAPTERS["bounded_observers"], "timeout_enforced": True,
                "maximum_seconds": .5, "maximum_bytes": MAX_RESPONSE_BYTES},
            "watchdog": {"adapter": ADAPTERS["external_watchdog"], "instance_id": policy.instance_id,
                "independent": True, "armed": True, "deadline_seconds": policy.deadline_seconds, "cleanup_bounded": True},
            "gpu": {"adapter": ADAPTERS["gpu_clearance"], "ownership_complete": True, "unreadable": 0,
                "conflicts": [], "active_contexts": [], "scope": "injected-no-real-device-inspected"}}

    def receipt(self, name, body):
        identity = write_json(self.receipts/(name+".json"), body)
        return stage.ReceiptRef(name, identity["sha256"], identity["bytes"])

    def refresh_approval(self):
        self.approval = {"schema": "dongxi-stage-approval-v1", "scope": stage.FIXTURE_SCOPE,
            "stage_id": self.identifier, "preparation_sha256": self.preparation["preparation_sha256"],
            "decision": "fixture-validation-only", "limits_sha256": canonical_hash(self.bindings["limits"]),
            "prerequisite_receipts_sha256": {k: v.receipt.sha256 for k, v in self.prerequisites.items()},
            "preflight_receipt_sha256": self.preflight_ref.sha256}
        self.approval_ref = self.receipt("approval", self.approval)

    def compile(self, **overrides):
        kwargs = dict(bindings=self.bindings, artifact_root=self.artifacts, receipt_root=self.receipts,
            approval_ref=self.approval_ref, prerequisite_refs=self.prerequisites, preflight_ref=self.preflight_ref,
            observation_nonce=self.nonce, now=self.now, root=self.root)
        request = overrides.pop("request", {"stage_id": self.identifier})
        kwargs.update(overrides)
        return stage.compile_stage(request, **kwargs)


class ProductionStageTests(unittest.TestCase):
    def fixture(self, identifier="story-profile"):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        return Fixture(directory.name, identifier)

    def test_all45_fixed_recipes_prepare_without_launch_or_model_import(self):
        modules_before=set(sys.modules)
        with patch("subprocess.Popen") as spawn, patch("subprocess.run") as run, patch("os.kill") as kill:
            records = [stage.prepare_stage(identifier) for identifier in stage.STAGE_IDS]
        self.assertEqual(len(records), 45); self.assertEqual(len({r["stage_id"] for r in records}),45)
        for result in records:
            self.assertEqual(result["jobs_started"],0); self.assertFalse(result["launch_authorized"])
            self.assertFalse(result["production_ready"]); self.assertIsNone(result["launch_command"])
            self.assertIsNone(result["runner"]["argv"]); self.assertIsNone(result["bindings"])
        spawn.assert_not_called(); run.assert_not_called(); kill.assert_not_called()
        self.assertFalse((set(sys.modules)-modules_before)&{"torch","transformers","tokenizers","peft"})

    def test_unknown_ID_and_campaign_allowlist_drift_fail_before_file_reads(self):
        for identifier in ("train-anything", "../story-profile", True, None):
            with self.subTest(identifier=identifier), patch.object(stage,"_read") as read:
                with self.assertRaises(ValueError): stage.prepare_stage(identifier)
                read.assert_not_called()
        rows = stage.staged_campaign.stage_rows(); rows.append(deepcopy(rows[0]))
        with patch.object(stage.staged_campaign,"stage_rows",return_value=rows), self.assertRaises(ValueError):
            stage.prepare_stage("story-profile")

    def test_actual_DPO_and_RLVR_envelopes_are_not_padded_or_response_geometry(self):
        dpo = stage.prepare_stage("assistant-dpo-pilot")["logical_work_envelope"]
        self.assertEqual(dpo["padded_geometry"],204800)
        self.assertEqual(dpo["policy_scored_positions"],408800)
        self.assertEqual(dpo["reference_scored_positions"],408800)
        row = stage._row("reasoning-rlvr-g4-pilot")
        r = stage.work_envelope(row,prompt_positions=7)
        self.assertEqual(r["response_slots"],4096)
        self.assertEqual(r["generation_prefix_positions"],16*4*(64*7+64*63//2))
        self.assertGreater(r["policy_scored_positions"],r["response_slots"])
        self.assertIn("backward",r["scope"])

    def test_deterministic_preparation_binds_sources_lock_inputs_parent_and_interface(self):
        f = self.fixture()
        repeated = stage.prepare_stage(f.identifier,bindings=f.bindings,artifact_root=f.artifacts)
        self.assertEqual(repeated,f.preparation)
        self.assertEqual(repeated["bindings"]["files"],f.bindings["files"])
        self.assertEqual(repeated["lock_sha256"],repeated["source_files"]["uv.lock"]["sha256"])
        self.assertIn("src/dongxi_llms/training_snapshot.py",repeated["source_files"])
        self.assertIn("src/dongxi_llms/production_preflight.py",repeated["source_files"])
        for dependency in ("decoder_lab","pretraining_lab","dpo_lab","grpo_lab","batched_cache_lab","evaluation_lab","sampling_likelihood_lab","artifact_budget","work_budget"):
            self.assertIn(f"src/dongxi_llms/{dependency}.py",repeated["source_files"])
        self.assertIsNone(repeated["bindings"]["checkpoint_genealogy"])

    def test_positive_fixture_never_grants_real_authority_or_execution(self):
        f = self.fixture()
        with patch("subprocess.Popen") as spawn,patch("subprocess.run") as run,patch("os.kill") as kill:
            result = f.compile()
        self.assertEqual(result["status"],"validated-authored-fixture-preparation")
        self.assertEqual(result["jobs_started"],0); self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["production_ready"]); self.assertIsNone(result["actual_external_result"])
        spawn.assert_not_called(); run.assert_not_called(); kill.assert_not_called()

    def test_prerequisite_stage_result_and_parent_identities_are_independently_pinned(self):
        f = self.fixture("story-control-smoke"); self.assertEqual(len(f.compile()["receipt_identities"]["prerequisites"]),1)
        for field in ("preparation_sha256","result_sha256","parent_manifest_sha256"):
            changed = deepcopy(f.prerequisites); key = next(iter(changed))
            changed[key] = replace(changed[key],**{field:"0"*64})
            with self.subTest(field=field),self.assertRaises(ValueError):f.compile(prerequisite_refs=changed)
        with self.assertRaises(ValueError):f.compile(prerequisite_refs={})

    def test_request_cannot_inject_args_shell_paths_or_self_approval(self):
        f = self.fixture()
        for field in ("argv","shell","command","path","output","approved","science","limits"):
            with self.subTest(field=field),self.assertRaises(ValueError):
                f.compile(request={"stage_id":f.identifier,field:True})
        with self.assertRaises(ValueError):f.compile(request={"stage_id":"unknown"})

    def test_absent_receipts_and_uncommitted_wrong_lengths_or_hashes_fail_closed(self):
        f = self.fixture()
        for key in ("approval_ref","preflight_ref"):
            for ref in (None,replace(getattr(f,key),sha256="0"*64),replace(getattr(f,key),bytes=True),
                        replace(getattr(f,key),bytes=getattr(f,key).bytes+1)):
                with self.subTest(key=key,ref=ref),self.assertRaises(ValueError):f.compile(**{key:ref})
        (f.receipts/"approval.json").unlink()
        with self.assertRaises(FileNotFoundError):f.compile()

    def test_receipt_paths_symlinks_and_artifact_root_self_approval_refuse(self):
        f = self.fixture()
        for name in ("../approval","/approval","approval.json","a/b","a\\b"):
            with self.subTest(name=name),self.assertRaises(ValueError):f.compile(approval_ref=replace(f.approval_ref,name=name))
        (f.receipts/"approval.json").unlink();(f.receipts/"approval.json").symlink_to(f.artifacts/"data.json")
        with self.assertRaises(OSError):f.compile()
        with self.assertRaises(ValueError):f.compile(receipt_root=f.artifacts)

    def test_inventory_escape_symlinks_directories_and_byte_mutation_refuse(self):
        f = self.fixture()
        for name in ("../data.json","/data.json","a//b","a/./b","a\\b"):
            b = deepcopy(f.bindings);b["files"]["data"]["path"]=name
            with self.subTest(name=name),self.assertRaises(ValueError):f.compile(bindings=b)
        b=deepcopy(f.bindings);b["files"]["data"]["path"]="directory";(f.artifacts/"directory").mkdir()
        with self.assertRaises(ValueError):f.compile(bindings=b)
        os.mkfifo(f.artifacts/"pipe");b["files"]["data"]["path"]="pipe"
        with self.assertRaises(ValueError):f.compile(bindings=b)
        path=f.artifacts/"data.json";raw=path.read_bytes();path.write_bytes(raw+b" ")
        with self.assertRaises(ValueError):f.compile()
        path.unlink();path.symlink_to(f.artifacts/"parent.json")
        with self.assertRaises(OSError):f.compile()
        alias=f.base/"alias";alias.symlink_to(f.artifacts,target_is_directory=True)
        with self.assertRaises(OSError):f.compile(artifact_root=alias)
        ancestor=f.base/"aliased-parent";ancestor.symlink_to(f.base,target_is_directory=True)
        with self.assertRaises(OSError):f.compile(artifact_root=ancestor/"artifacts")

    def test_parent_slot_and_interface_change_refuse_even_rehashed_inventory(self):
        f=self.fixture();parent=json.loads((f.artifacts/"parent.json").read_text())
        for key,value in (("stage_slot","another-parent"),("interface_sha256","0"*64)):
            bad=deepcopy(parent);bad[key]=value;b=deepcopy(f.bindings)
            b["files"]["parent"]=write_json(f.artifacts/"parent.json",bad)
            with self.subTest(key=key),self.assertRaises(ValueError):f.compile(bindings=b)
        b=deepcopy(f.bindings);b["parent_slots"]={"unrelated":"parent"}
        with self.assertRaises(ValueError):f.compile(bindings=b)

    def test_limits_are_exact_typed_finite_and_do_not_relax_frozen_ceilings(self):
        f=self.fixture()
        for key,value in (("reserve_bytes",24*1024**3),("owner_uid",True),("artifact_entries",False),
                          ("external_seconds",float("inf")),("external_seconds",601),
                          ("max_policy_forward_positions",-1),("snapshot_max_bytes",2*1024**2),
                          ("private_instance_id","/user.slice"),("private_instance_id",False),
                          ("max_response_slots",2**63)):
            b=deepcopy(f.bindings);b["limits"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):f.compile(bindings=b)
        pilot=stage._row("story-control-pilot");limits=deepcopy(f.bindings["limits"])
        limits["max_valid_training_targets"]=50_000_001
        with self.assertRaises(ValueError):stage._limits(limits,pilot)

    def test_actual_production_scope_cannot_be_promoted_from_fixture(self):
        f=self.fixture();b=deepcopy(f.bindings);b["scope"]="actual-production"
        with self.assertRaises(RuntimeError):f.compile(bindings=b)
        f.approval["scope"]="actual-production";f.approval_ref=f.receipt("approval",f.approval)
        with self.assertRaises(ValueError):f.compile()

    def test_missing_unimplemented_stage_adapters_are_not_aliases_for_DPO_or_SFT(self):
        for identifier in ("assistant-chosen-sft-smoke","assistant-dpo-chosen-nll-extension","assistant-lora-merge"):
            f=self.fixture(identifier)
            self.assertEqual(f.preparation["runner"]["adapter_status"],"stage-adapter-unimplemented")
            with self.subTest(identifier=identifier),self.assertRaises(RuntimeError):f.compile()

    def test_unknown_duplicate_nonfinite_approval_JSON_and_boolean_approval_refuse(self):
        f=self.fixture()
        for field in ("approved","command","argv"):
            bad=deepcopy(f.approval);bad[field]=True;ref=f.receipt("bad",bad)
            with self.subTest(field=field),self.assertRaises(ValueError):f.compile(approval_ref=ref)
        for raw in (b'{"schema":1,"schema":2}',b'{"approved":NaN}',b'"not a receipt"'):
            (f.receipts/"bad.json").write_bytes(raw)
            ref=stage.ReceiptRef("bad",hashlib.sha256(raw).hexdigest(),len(raw))
            with self.subTest(raw=raw),self.assertRaises(ValueError):f.compile(approval_ref=ref)

    def test_preflight_nonce_binding_freshness_future_time_and_summary_refuse(self):
        f=self.fixture()
        for fields in ({"now":103.},{"now":99.},{"now":True},{"observation_nonce":"0"*64}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):f.compile(**fields)
        for key,value in (("stage_id","story-control-smoke"),("preparation_sha256","0"*64),
                          ("production_ready",True),("observed_utc","2099-01-01T00:00:00+00:00")):
            bad=deepcopy(f.preflight);bad[key]=value;ref=f.receipt("bad",bad)
            with self.subTest(key=key),self.assertRaises(ValueError):f.compile(preflight_ref=ref)

    def test_missing_backend_gates_and_forged_ready_summaries_refuse(self):
        f=self.fixture()
        for key in f.preflight["backend_gates"]:
            bad=deepcopy(f.preflight);bad["backend_gates"][key]="missing"
            with self.subTest(key=key),self.assertRaises(ValueError):f.compile(preflight_ref=f.receipt("bad",bad))
        bad=deepcopy(f.preflight);bad["evidence"]["observations"][1]["quota"]["hard_bytes"]=1
        with self.assertRaises(ValueError):f.compile(preflight_ref=f.receipt("bad",bad))

    def test_source_and_lock_revisions_change_preparation_and_reject_old_receipts(self):
        f=self.fixture();directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);root=Path(directory.name)
        for name in stage.SOURCE_FILES:
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f.root/name,dest)
        for target in ("uv.lock","src/dongxi_llms/production_stage.py"):
            original=(root/target).read_bytes();(root/target).write_bytes(original+b"\n")
            changed=stage.prepare_stage(f.identifier,bindings=f.bindings,artifact_root=f.artifacts,root=root)
            self.assertNotEqual(changed["preparation_sha256"],f.preparation["preparation_sha256"])
            with self.subTest(target=target),self.assertRaises(ValueError):f.compile(root=root)
            (root/target).write_bytes(original)

    def test_malformed_inventory_structure_and_bounded_reads_fail_before_large_parse(self):
        f=self.fixture()
        for key,value in (("parent_slots",False),("interface_role",None),("files",[])):
            b=deepcopy(f.bindings);b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):f.compile(bindings=b)
        b=deepcopy(f.bindings);b["files"]["data"]["bytes"]=stage.MAX_ARTIFACT_BYTES+1
        with self.assertRaises(ValueError):f.compile(bindings=b)
        with self.assertRaises(ValueError):f.compile(approval_ref=replace(f.approval_ref,bytes=stage.MAX_RECEIPT_BYTES+1))


def collect_reference(destination):
    """Fixed CPU unittest/reference collector; never a production runner launcher."""
    from datetime import datetime, timezone
    import importlib.metadata
    import os
    import platform
    import subprocess
    import sys
    import time
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    before={name:hashlib.sha256((stage.staged_campaign.ROOT/name).read_bytes()).hexdigest() for name in stage.SOURCE_FILES}
    started=time.monotonic()
    env=dict(os.environ,PYTHONPATH="src:tests",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",
             TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1")
    commands=[]
    for pattern in ("test_production_stage.py","test_staged_campaign.py","test_production_preflight.py"):
        command=[sys.executable,"-m","unittest","discover","-s","tests","-p",pattern,"-v"]
        child_started=time.monotonic()
        child=subprocess.run(command,cwd=stage.staged_campaign.ROOT,env=env,capture_output=True,text=True,timeout=60)
        log=child.stdout+child.stderr
        log_path=destination/(pattern+".log");log_path.write_text(log)
        commands.append({"command":command,"exit_code":child.returncode,"seconds":time.monotonic()-child_started,
            "raw_log":str(log_path),"raw_log_sha256":hashlib.sha256(log.encode()).hexdigest()})
    catalogs=[]
    for identifier in stage.STAGE_IDS:
        value=stage.prepare_stage(identifier)
        catalogs.append({"stage_id":identifier,"preparation_sha256":value["preparation_sha256"],
            "runner":value["runner"],"logical_work_envelope":value["logical_work_envelope"],
            "jobs_started":value["jobs_started"],"launch_authorized":value["launch_authorized"]})
    references=[]
    for identifier in ("story-profile","story-control-smoke","assistant-dpo-smoke"):
        fixture_directory=destination/identifier;fixture_directory.mkdir()
        f=Fixture(fixture_directory,identifier)
        references.append(f.compile())
    refused=[]
    for label,fields in (("unknown-stage",{"request":{"stage_id":"arbitrary-training"}}),
                         ("self-approved",{"request":{"stage_id":f.identifier,"approved":True}}),
                         ("missing-approval",{"approval_ref":None}),("stale-observations",{"now":103.})):
        try:
            f.compile(**fields)
        except (ValueError,RuntimeError) as error:
            refused.append({"control":label,"status":"refused","exception":type(error).__name__,
                "message":str(error),"model_jobs_started":0})
        else:
            refused.append({"control":label,"status":"unexpectedly-accepted","model_jobs_started":0})
    after={name:hashlib.sha256((stage.staged_campaign.ROOT/name).read_bytes()).hexdigest() for name in stage.SOURCE_FILES}
    result={"schema":"dongxi-production-stage-cpu-verification-v1","status":"pass" if before==after and all(c["exit_code"]==0 for c in commands) and all(r["status"]=="refused" for r in refused) else "fail",
        "created_utc":datetime.now(timezone.utc).isoformat(),"command":list(sys.orig_argv),"python_argv":list(sys.argv),
        "environment":{"python":platform.python_version(),"executable":sys.executable,"platform":platform.platform(),
            "packages":{name:importlib.metadata.version(name) for name in ("torch","transformers","tokenizers","peft")},
            "cuda_visible_devices":"","hf_offline":True},"commands":commands,"seconds":time.monotonic()-started,
        "source_sha256":before,"source_sha256_after":after,"sources_unchanged":before==after,
        "catalog":catalogs,"authored_fixture_preparations":references,"retained_refusals":refused,"model_jobs_started":0,
        "production_ready":False,"launch_authorized":False,
        "boundary":"Fixed CPU unittest children, original bounded self-owned leaf controls, and JSON fixture validation only; compiler starts no process and has no real backend/approval/model/checkpoint."}
    (destination/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv)==3 and sys.argv[1]=="--collect":
        result=collect_reference(sys.argv[2])
        print(json.dumps({"status":result["status"],"catalog_stages":len(result["catalog"]),"authored_fixtures":len(result["authored_fixture_preparations"])}))
        raise SystemExit(0 if result["status"]=="pass" else 1)
    unittest.main()

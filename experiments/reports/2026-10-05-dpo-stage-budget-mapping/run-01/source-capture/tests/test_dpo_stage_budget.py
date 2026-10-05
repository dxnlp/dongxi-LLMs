"""Original actual-tokenizer/cap/loop controls; no pretrained model or launcher."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from tokenizers import Tokenizer

import test_dpo_runner_recovery as base
import test_production_stage as preparation_controls
from dongxi_llms import dpo_stage_budget as budget
from dongxi_llms import production_stage as stage
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.run_identity import canonical_hash, tokenizer_interface
from dongxi_llms.work_budget import WorkLedger, WorkBudgetExceeded

r=base.runner;ROOT=base.ROOT
SEED=1818;BOUND=1024*1024;SNAPSHOT_BOUND=16*1024*1024
ATOL=RTOL=1e-6
SPEC="experiments/specs/2026-10-05-dpo-stage-budget-mapping.md"
SOURCES=("src/dongxi_llms/dpo_stage_budget.py","tests/test_dpo_stage_budget.py",SPEC,
    "scripts/run_chapter11_spark_dpo.py","src/dongxi_llms/work_budget.py",
    "src/dongxi_llms/training_snapshot.py","src/dongxi_llms/artifact_budget.py",
    "src/dongxi_llms/run_identity.py","src/dongxi_llms/dpo_lab.py",
    "src/dongxi_llms/batched_cache_lab.py","tests/test_dpo_runner_recovery.py",
    "tests/test_production_stage.py","src/dongxi_llms/production_stage.py",
    "src/dongxi_llms/staged_campaign.py","uv.lock")


def write(path,raw):
    path=Path(path);path.write_bytes(raw)
    return dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def json_bytes(value):return json.dumps(value,sort_keys=True,allow_nan=False).encode()+b"\n"


def assert_tree(test,first,second,tolerance=False):
    if isinstance(first,torch.Tensor):
        if tolerance:torch.testing.assert_close(first,second,atol=ATOL,rtol=RTOL)
        else:test.assertTrue(torch.equal(first,second))
    elif isinstance(first,dict):
        test.assertEqual(set(first),set(second))
        for key in first:assert_tree(test,first[key],second[key],tolerance)
    elif type(first) in (list,tuple):
        test.assertEqual(len(first),len(second))
        for a,b in zip(first,second):assert_tree(test,a,b,tolerance)
    else:test.assertEqual(first,second)


class Fixture:
    def __init__(self,directory,stage_id="assistant-dpo-smoke",long=False):
        self.base=Path(directory);self.prepared=preparation_controls.Fixture(self.base,stage_id)
        self.artifacts=self.prepared.artifacts;self.receipts=self.prepared.receipts
        self.tokenizer=base.tokenizer_fixture()
        self.interface=tokenizer_interface(self.tokenizer,template=self.tokenizer.chat_template,stop_ids=[1],
            tokenizer_id="authored-local-WordLevel-sixteen",tokenizer_revision=None)
        self.train=[dict(id="train-red-cup",group="source-red-cup",prompt=[dict(role="user",content="red cup")],chosen="box",rejected="long bag"),
                    dict(id="train-red-key",group="source-red-key",prompt=[dict(role="user",content="red key")],chosen="bag",rejected="long box")]
        if long:
            for row in self.train:row.update(chosen=" ".join(["box"]*506),rejected=" ".join(["bag"]*506))
        self.valid=[dict(id="validation-blue-key",group="source-blue-key",prompt=[dict(role="user",content="blue key")],chosen="book",rejected="long bag")]
        self.evaluation=[dict(id="generation-blue-cup",group="source-blue-cup",prompt=[dict(role="user",content="blue cup")],expected="box")]
        self.refresh()

    def refresh(self):
        files=self.prepared.bindings["files"]
        for kind,role,rows in (("train","data",self.train),("validation","validation",self.valid),("evaluation","evaluation",self.evaluation)):
            files[role]=write(self.artifacts/f"{kind}.jsonl",b"".join(json_bytes(row) for row in rows))
        files["tokenizer"]=write(self.artifacts/"tokenizer.json",self.tokenizer.backend_tokenizer.to_str().encode())
        files["template"]=write(self.artifacts/"template.jinja",self.tokenizer.chat_template.encode())
        files["interface"]=write(self.artifacts/"interface.json",json_bytes(self.interface))
        slot=stage._row(self.prepared.identifier)["proposed_weight_start"]
        files["parent"]=write(self.artifacts/"parent.json",json_bytes(dict(schema="authored-parent-v1",scope=stage.FIXTURE_SCOPE,
            stage_slot=slot,interface_sha256=self.interface["interface_sha256"])))
        self.preparation=stage.prepare_stage(self.prepared.identifier,bindings=self.prepared.bindings,artifact_root=self.artifacts)
        self.encoded_train=[r.encode_pair(self.tokenizer,row,512) for row in self.train]
        self.encoded_valid=[r.encode_pair(self.tokenizer,row,512) for row in self.valid]
        self.prefixes=[self.tokenizer.apply_chat_template(row["prompt"],tokenize=True,add_generation_prompt=True,
                          enable_thinking=False,return_dict=False) for row in self.evaluation]

    def requirements(self,attempts=1,**changes):
        options=dict(artifact_root=self.artifacts,encoded_train=self.encoded_train,encoded_valid=self.encoded_valid,
            evaluation_prefixes=self.prefixes,interface_sha256=self.interface["interface_sha256"],
            encoder_source_sha256=self.preparation["source_files"]["scripts/run_chapter11_spark_dpo.py"]["sha256"],
            attempt_allowance=attempts)
        options.update(changes)
        return budget.build_requirements(self.preparation,**options)

    def approve(self,requirements,limits=None,body_change=None,name="work-approval"):
        limits=deepcopy(requirements["required_limits"] if limits is None else limits)
        record=dict(schema="dongxi-dpo-work-approval-v1",scope=stage.FIXTURE_SCOPE,stage_id=requirements["stage_id"],
            preparation_sha256=requirements["preparation_sha256"],requirements_sha256=requirements["requirements_sha256"],
            limits_sha256=canonical_hash(limits),decision="fixture-work-cap-validation-only")
        if body_change:record.update(body_change)
        identity=write(self.receipts/f"{name}.json",json_bytes(record))
        reference=stage.ReceiptRef(name,identity["sha256"],identity["bytes"])
        return budget.approve_fixture(requirements,limits=limits,receipt_root=self.receipts,receipt_ref=reference)

    def reencode(self):
        core=Tokenizer.from_file(str(self.artifacts/"tokenizer.json"))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=core,unk_token="<UNK>",eos_token="<END>",pad_token="<END>",
                    additional_special_tokens=["<USER>","<ASSISTANT>"])
        tokenizer.chat_template=(self.artifacts/"template.jinja").read_text()
        iface=tokenizer_interface(tokenizer,template=tokenizer.chat_template,stop_ids=[1],
            tokenizer_id="authored-local-WordLevel-sixteen",tokenizer_revision=None)
        train=[r.encode_pair(tokenizer,row,512) for row in self.train]
        valid=[r.encode_pair(tokenizer,row,512) for row in self.valid]
        prefixes=[tokenizer.apply_chat_template(row["prompt"],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                  for row in self.evaluation]
        return dict(encoded_train=train,encoded_valid=valid,evaluation_prefixes=prefixes,interface_sha256=iface["interface_sha256"])


def components(f,mapping):
    torch.set_num_threads(1);torch.manual_seed(SEED)
    config=Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,max_position_embeddings=512,attention_dropout=0.,bos_token_id=7,eos_token_id=1,pad_token_id=1)
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32);model=Qwen3ForCausalLM(config)
    finally:torch.set_default_dtype(previous_dtype)
    reference=deepcopy(model).eval().requires_grad_(False)
    model.config.use_cache=False;reference.config.use_cache=False;model.gradient_checkpointing_enable()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7,weight_decay=.01);sampler=torch.Generator().manual_seed(SEED)
    cap_path=f.base/"consumable-work-limits.json";write(cap_path,budget.work_limits_bytes(mapping))
    from dongxi_llms.work_budget import _decode
    actual_limits=r.work_budget_contract(_decode(r.read_work_limits_file(cap_path)),BOUND)
    sources={name:r.file_digest(ROOT/name) for name in SOURCES}
    sources.update({name:identity["sha256"] for name,identity in f.preparation["source_files"].items()})
    options=dict(parent_files={"authored-random-microscope-parent":digest(reference.state_dict())},
        inputs={name:identity["sha256"] for name,identity in f.preparation["bindings"]["files"].items()},
        sources=sources,environment=dict(observed=deepcopy(f.preparation["interpreter"]),lock_sha256=r.file_digest(ROOT/"uv.lock")),
        interface=f.interface,encoded_train=f.encoded_train,encoded_valid=f.encoded_valid,evaluation_prefixes=f.prefixes,
        seed=SEED,updates=2,accumulation=4,beta=.1,lr=5e-7,max_length=512,max_new_tokens=64,
        device=dict(mode="cpu",dtype="float32"),work_budget=actual_limits)
    contract=r.make_recovery_contract(model=model,reference=reference,optimizer=optimizer,**options)
    return [model,reference,optimizer,sampler,contract]


def generate(f,v,ledger):
    return r.generate_dpo_panel(v[0],f.tokenizer,f.evaluation,f.prefixes,cap=64,stop_ids=[1],device="cpu",work_ledger=ledger)


def train(f,v,ledger,output,initial=None,baseline_sink=None):
    def baseline():
        rows=generate(f,v,ledger)
        if baseline_sink is not None:baseline_sink(rows)
        return rows
    return r.train_completed_updates(*v[:4],f.encoded_train,contract=v[4],output=output,invocation_id="authored-mapping-cpu",
        pad_id=1,device="cpu",checkpoint_every=2,max_bytes=SNAPSHOT_BOUND,work_ledger=ledger,
        before_training=baseline,**(initial or {}))


def finish(f,v,ledger,training):
    validation=r.score_dpo_panel(v[0],v[1],f.encoded_valid,pad_id=1,device="cpu",beta=.1,work_ledger=ledger)
    return dict(validation=validation,generation=generate(f,v,ledger))


def independent_updates(f,v):
    model,reference,optimizer,sampler=v[:4];history=[]
    for update in (1,2):
        model.train();reference.eval();optimizer.zero_grad(set_to_none=True);indices=[];losses=[]
        for _ in range(4):
            index=int(torch.randint(len(f.encoded_train),(),generator=sampler));indices.append(index);scores=[]
            for network in (model,reference):
                branches=[]
                for ids,mask in f.encoded_train[index]:
                    tokens=torch.tensor([ids]);logits=network(input_ids=tokens[:,:-1],attention_mask=torch.ones_like(tokens[:,:-1]),use_cache=False).logits.float()
                    selected=logits.log_softmax(-1).gather(-1,tokens[:,1:,None]).squeeze(-1)
                    branches.append(selected[:,torch.tensor(mask[1:])].sum())
                scores.append(branches)
            margin=.1*(scores[0][0]-scores[0][1]-scores[1][0].detach()+scores[1][1].detach())
            loss=torch.nn.functional.softplus(-margin);(loss/4).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append(dict(update=update,indices=indices,loss=sum(losses)/4,gradient_norm=float(norm)))
    return history


def ledger(path,v,opening=False):
    return (WorkLedger.open if opening else WorkLedger.create)(path,limits=v[4]["work_budget"]["limits"],
        contract_sha256=canonical_hash(v[4]),max_bytes=BOUND,invocation_id="authored-mapping-recovery" if opening else "authored-mapping-original")


def payload(receipt,v,work):
    return r.load_snapshot(receipt["path"],expected_sha256=receipt["payload_sha256"],expected_bytes=receipt["payload_bytes"],
        expected_contract=v[4],max_bytes=SNAPSHOT_BOUND,validate_payload=lambda saved:r.validate_dpo_payload(saved,v[4],work))


class DPOStageBudgetTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix="dongxi-dpo-mapping-");self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def fixture(self,stage_id="assistant-dpo-smoke",long=False):
        directory=self.root/f"fixture-{len(list(self.root.iterdir()))}";directory.mkdir(mode=0o700)
        return Fixture(directory,stage_id,long)

    def test_real_encoded_geometry_matches_existing_runner_reservations(self):
        f=self.fixture();q=f.requirements();ops=q["operations"]
        self.assertEqual(ops["one_update"],dict(budget.zero(),**r.update_upper(f.encoded_train,4)))
        self.assertEqual(ops["one_generation_panel"],dict(budget.zero(),**r.generation_upper(f.prefixes,64)))
        self.assertEqual(q["required_limits"]["valid_targets"],ops["training_valid_target_upper"]+ops["validation_target_upper"])
        self.assertEqual(q["schedule"]["baseline_generation_panels"],1);self.assertEqual(q["schedule"]["final_generation_panels"],1)

    def test_saved_real_tokenizer_reproduces_actual_input_order_masks_and_interface(self):
        f=self.fixture();self.assertEqual(f.encoded_train,f.reencode()["encoded_train"])
        self.assertTrue(budget.assert_reencoded(f.requirements(),f.requirements(**f.reencode())))
        self.assertEqual(f.interface["tokenizer"]["vocab_size"],16)

    def test_only_three_fixed_DPO_recipes_are_supported(self):
        for identifier in budget.STAGES:
            f=self.fixture(identifier);self.assertEqual(f.requirements()["science"]["intervention"]["seed"],1818)
        f=self.fixture();p=deepcopy(f.preparation);p["stage_id"]="assistant-chosen-sft-smoke"
        with self.assertRaises(ValueError):budget.build_requirements(p,artifact_root=f.artifacts,encoded_train=[],encoded_valid=[],evaluation_prefixes=[],
            interface_sha256="a"*64,encoder_source_sha256="b"*64,attempt_allowance=1)

    def test_actual_scope_missing_files_stale_sources_and_self_modified_preparation_refuse(self):
        f=self.fixture()
        for key,changed in (("interface_sha256","a"*64),("encoder_source_sha256","b"*64)):
            with self.subTest(key=key),self.assertRaises(ValueError):f.requirements(**{key:changed})
        f.preparation["bindings"]["scope"]="actual-production"
        with self.assertRaises(RuntimeError):f.requirements()
        f.refresh();f.preparation["science"]["proposed_budget"]["maximum_updates"]=3
        with self.assertRaises(ValueError):f.requirements()

    def test_input_byte_mutation_and_template_mismatch_refuse(self):
        f=self.fixture();(f.artifacts/"train.jsonl").write_bytes(b"changed")
        with self.assertRaises(ValueError):f.requirements()
        f=self.fixture();f.prepared.bindings["files"]["template"]=write(f.artifacts/"template.jinja",b"different template")
        f.preparation=stage.prepare_stage(f.prepared.identifier,bindings=f.prepared.bindings,artifact_root=f.artifacts)
        with self.assertRaises(ValueError):f.requirements()

    def test_masks_ID_types_prefixes_lengths_and_support_are_strict(self):
        f=self.fixture()
        for label,modify in (("bool-id",lambda x:x[0][0][0].__setitem__(0,True)),("negative",lambda x:x[0][0][0].__setitem__(0,-1)),
            ("out-of-vocab",lambda x:x[0][0][0].__setitem__(0,16)),("int-mask",lambda x:x[0][0][1].__setitem__(0,0)),
            ("no-target",lambda x:x[0][0].__setitem__(1,[False]*len(x[0][0][0]))),
            ("nonmonotone",lambda x:x[0][0][1].__setitem__(-1,False)),("prefix",lambda x:x[0][1][0].__setitem__(0,0)),
            ("overlength",lambda x:x[0][0].__setitem__(0,[2]*513))):
            rows=json.loads(json.dumps(f.encoded_train));modify(rows)
            with self.subTest(label=label),self.assertRaises(ValueError):f.requirements(encoded_train=rows)

    def test_split_collisions_and_changed_encoding_order_refuse_or_invalidate_receipt(self):
        f=self.fixture();f.valid[0]["id"]=f.train[0]["id"];f.refresh()
        with self.assertRaises(ValueError):f.requirements()
        f=self.fixture();q=f.requirements();swapped=f.requirements(encoded_train=list(reversed(f.encoded_train)))
        with self.assertRaises(ValueError):budget.assert_reencoded(q,swapped)
        mapping=f.approve(q)
        with self.assertRaises(ValueError):budget.approve_fixture(swapped,limits=swapped["required_limits"],receipt_root=f.receipts,
            receipt_ref=stage.ReceiptRef("work-approval",mapping["approval_receipt"]["sha256"],mapping["approval_receipt"]["bytes"]))

    def test_bounded_geometry_and_actual_context_refuse_before_work(self):
        f=self.fixture()
        with patch.object(budget,"MAX_POSITIONS",49),self.assertRaises(ValueError):f.requirements()
        with self.assertRaises(ValueError):f.requirements(evaluation_prefixes=[[2]*512])

    def test_schedule_is_explicit_typed_and_not_retry_permission(self):
        f=self.fixture()
        for attempts in (0,True,1.,5):
            with self.subTest(attempts=attempts),self.assertRaises(ValueError):f.requirements(attempts)
        with self.assertRaises(ValueError):f.requirements(generation_cap=4)
        q1,q2=f.requirements(),f.requirements(2)
        self.assertEqual(q2["required_limits"],{key:2*value for key,value in q1["required_limits"].items()})
        self.assertFalse(q2["schedule"]["retry_permission"]);self.assertFalse(q2["schedule"]["invocation_counter_enforced"])

    def test_each_of14_insufficient_caps_refuses_without_sampler_or_forward(self):
        f=self.fixture();q=f.requirements()
        for key in budget.KEYS:
            limits=deepcopy(q["required_limits"]);limits[key]-=1
            with self.subTest(key=key),patch("torch.randint") as draw,patch.object(Qwen3ForCausalLM,"forward") as forward:
                with self.assertRaises(ValueError):f.approve(q,limits)
                draw.assert_not_called();forward.assert_not_called()

    def test_extra_capacity_requires_new_schedule_not_individual_override(self):
        f=self.fixture();q=f.requirements();limits=deepcopy(q["required_limits"]);limits["train_updates"]+=1
        with self.assertRaises(ValueError):f.approve(q,limits)

    def test_missing_old_coarse_and_real_production_approvals_refuse(self):
        f=self.fixture();q=f.requirements()
        with self.assertRaises(ValueError):budget.approve_fixture(q,limits=q["required_limits"],receipt_root=f.receipts,receipt_ref=None)
        with self.assertRaises(ValueError):budget.approve_fixture(q,limits=q["required_limits"],receipt_root=f.receipts,receipt_ref=f.prepared.approval_ref)
        with self.assertRaises(ValueError):f.approve(q,body_change={"scope":"actual-production","decision":"approved"})

    def test_204800_geometry_cannot_approve_actual_four_branch_pilot(self):
        f=self.fixture("assistant-dpo-pilot",long=True);q=f.requirements();ops=q["operations"]
        self.assertTrue(all(len(branch[0])==512 for pair in f.encoded_train for branch in pair))
        self.assertEqual(ops["historical_single_branch_geometry"],204800)
        self.assertEqual(ops["actual_training_policy_upper"]+ops["actual_training_reference_upper"],817600)
        limits=deepcopy(q["required_limits"]);limits["policy_forward_positions"]=limits["reference_forward_positions"]=204800
        with self.assertRaises(ValueError):f.approve(q,limits)
        self.assertFalse(f.approve(q)["production_ready"])

    def test_exact_consumable_reader_schema_and_mapping_tamper_controls(self):
        f=self.fixture();mapping=f.approve(f.requirements());raw=budget.work_limits_bytes(mapping)
        path=f.base/"actual-work-limits.json";write(path,raw)
        from dongxi_llms.work_budget import _decode
        self.assertEqual(r.work_budget_contract(_decode(r.read_work_limits_file(path)),BOUND)["limits"],mapping["limits"])
        for key,value in (("production_ready",True),("jobs_started",False),("launch_command","python training.py")):
            changed=deepcopy(mapping);changed[key]=value;changed["mapping_sha256"]=canonical_hash({k:v for k,v in changed.items() if k!="mapping_sha256"})
            with self.subTest(key=key),self.assertRaises(ValueError):budget.work_limits_bytes(changed)

    def test_actual_full_schedule_fits_and_original_equation_replays(self):
        f=self.fixture();mapping=f.approve(f.requirements());v=components(f,mapping);out=f.base/"complete";out.mkdir()
        with ledger(f.base/"work.jsonl",v) as work:
            result=train(f,v,work,out);final=finish(f,v,work,result);summary=work.snapshot()
        self.assertEqual(summary["reserved"],mapping["limits"]);self.assertEqual(len(final["generation"]),1)
        other=components(f,mapping);reference_history=independent_updates(f,other)
        for expected,observed in zip(reference_history,result["history"]):
            self.assertEqual(expected["indices"],observed["indices"])
            for key in ("loss","gradient_norm"):
                torch.testing.assert_close(torch.tensor(expected[key],dtype=torch.float64),torch.tensor(observed[key],dtype=torch.float64),atol=ATOL,rtol=RTOL)
        for key in (0,1,2):
            first=v[key].state_dict();second=other[key].state_dict()
            assert_tree(self,first,second,tolerance=True)
        self.assertTrue(torch.equal(v[3].get_state(),other[3].get_state()))

    def test_interrupted_replay_retains_failed_attempts_and_refuses_extra_update(self):
        f=self.fixture();mapping=f.approve(f.requirements(2));clean=components(f,mapping);out=f.base/"clean";out.mkdir()
        with ledger(f.base/"clean-work.jsonl",clean) as work:
            full=train(f,clean,work,out);expected=payload(full["checkpoint"],clean,work)
        v=components(f,mapping);partial=f.base/"partial";partial.mkdir();journal=f.base/"retained-work.jsonl"
        original=r.completed_dpo_update
        def failing(*args,**kwargs):
            if kwargs["update"]==2:
                with patch.object(args[0],"forward",side_effect=OSError("authored second-update entered-forward interruption")):
                    return original(*args,**kwargs)
            return original(*args,**kwargs)
        with ledger(journal,v) as work:
            with patch.object(r,"completed_dpo_update",side_effect=failing),self.assertRaises(OSError):train(f,v,work,partial)
            spent=work.snapshot();receipt=base.checkpoint_receipt(partial/"checkpoints/completed-000000.pt")
        resumed=components(f,mapping);target=f.base/"resumed";target.mkdir()
        with ledger(journal,resumed,opening=True) as work:
            saved=payload(receipt,resumed,work)
            completed,counters,history=r.restore_dpo_payload(saved,contract=resumed[4],model=resumed[0],reference=resumed[1],optimizer=resumed[2],sampler=resumed[3],work_ledger=work)
            self.assertEqual(work.snapshot()["reserved"],spent["reserved"])
            result=train(f,resumed,work,target,dict(completed=completed,counters=counters,history=history,parent_checkpoint=receipt))
            actual=payload(result["checkpoint"],resumed,work);finish(f,resumed,work,result)
            for key in ("policy","reference","optimizer","sampler_rng","torch_rng","counters","history"):
                assert_tree(self,expected["state"][key],actual["state"][key])
            self.assertEqual(work.snapshot()["reserved"]["train_updates"],4)
            self.assertTrue(work.snapshot()["failed_tickets"])
            with patch("torch.randint") as draw,patch.object(resumed[0],"forward") as forward:
                with self.assertRaises(WorkBudgetExceeded):r.completed_dpo_update(*resumed[:4],f.encoded_train,pad_id=1,accumulation=4,beta=.1,device="cpu",update=3,work_ledger=work)
                draw.assert_not_called();forward.assert_not_called()


def collect_reference(destination):
    """Retain original bounded observations; only fixed CPU unittest children."""
    from datetime import datetime,timezone
    import importlib.metadata
    import os
    import platform
    import subprocess
    import sys
    import time
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    names=tuple(dict.fromkeys((*stage.SOURCE_FILES,*SOURCES)))
    before={name:r.file_digest(ROOT/name) for name in names};started=time.monotonic()
    result=dict(schema="dongxi-dpo-stage-mapping-cpu-reference-v1",status="running",created_utc=datetime.now(timezone.utc).isoformat(),
        command=list(sys.orig_argv),python_argv=list(sys.argv),source_sha256=before,
        environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),executable=sys.executable,
            packages={name:importlib.metadata.version(name) for name in ("torch","transformers","tokenizers","peft")},
            cuda_visible_devices="",hf_offline=True,omp_num_threads=1),commands=[],retained_refusals=[],model_scale_jobs_started=0,
        production_ready=False,launch_authorized=False,external_outcomes_filled=0)
    def save(name,value):write(destination/name,json_bytes(value))
    try:
        env=dict(os.environ,PYTHONPATH="src:tests",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1")
        for pattern in ("test_dpo_stage_budget.py","test_production_stage.py"):
            command=[sys.executable,"-m","unittest","discover","-s","tests","-p",pattern,"-v"]
            measured=time.monotonic()
            child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            log=child.stdout+child.stderr;identity=write(destination/(pattern+".log"),log.encode())
            result["commands"].append(dict(command=command,exit_code=child.returncode,seconds=time.monotonic()-measured,raw_log=identity))
        for name in names:
            raw,_=stage._read(ROOT,name,cap=4*1024*1024,expected_sha256=before[name])
            path=destination/"source-capture"/name;path.parent.mkdir(parents=True,exist_ok=True);write(path,raw)
        fdir=destination/"actual-smoke";fdir.mkdir(mode=0o700);f=Fixture(fdir)
        q=f.requirements();mapped=f.approve(q);save("single-attempt-requirements.json",q);save("single-attempt-mapping.json",mapped)
        save("actual-encodings.json",dict(train=f.encoded_train,validation=f.encoded_valid,prefixes=f.prefixes,interface=f.interface))
        fresh=f.requirements(**f.reencode());budget.assert_reencoded(q,fresh);save("fresh-reencoded-requirements.json",fresh)
        v=components(f,mapped);out=f.base/"complete";out.mkdir(mode=0o700)
        with ledger(f.base/"work.jsonl",v) as work:
            complete=train(f,v,work,out,baseline_sink=lambda rows:save("baseline-single-attempt.json",rows))
            final=finish(f,v,work,complete);summary=work.snapshot();saved=payload(complete["checkpoint"],v,work)
        save("complete-training.json",complete);save("complete-final-evaluation.json",final);save("complete-work-summary.json",summary)
        if summary["reserved"]!=mapped["limits"]:raise AssertionError("Actual whole schedule reservations differ from mapped caps")
        other=components(f,mapped);original_history=independent_updates(f,other)
        save("independent-equation-history.json",original_history)
        maxima=dict(policy=0.,reference=0.,adam=0.,loss=0.,gradient_norm=0.)
        def compare(a,b,label):
            if isinstance(a,torch.Tensor):
                torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)
                maxima[label]=max(maxima[label],float((a-b).abs().max()))
            elif isinstance(a,dict):
                if set(a)!=set(b):raise AssertionError("Original-equation state keys differ")
                for key in a:compare(a[key],b[key],label)
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise AssertionError("Original-equation state lengths differ")
                for x,y in zip(a,b):compare(x,y,label)
            elif a!=b:raise AssertionError("Original-equation scalar differs")
        for index,label in ((0,"policy"),(1,"reference"),(2,"adam")):compare(v[index].state_dict(),other[index].state_dict(),label)
        for a,b in zip(complete["history"],original_history):
            if a["indices"]!=b["indices"]:raise AssertionError("Original-equation sampler choices differ")
            for key in ("loss","gradient_norm"):
                torch.testing.assert_close(torch.tensor(a[key],dtype=torch.float64),torch.tensor(b[key],dtype=torch.float64),atol=ATOL,rtol=RTOL)
                maxima[key]=max(maxima[key],abs(a[key]-b[key]))
        if not torch.equal(v[3].get_state(),other[3].get_state()):raise AssertionError("Original-equation sampler cursor differs")
        result["original_equation_parity"]=dict(atol=ATOL,rtol=RTOL,max_absolute_difference=maxima,sampler_exact=True)
        # Separate full-attempt capacity does not change the numerical endpoint.
        q2=f.requirements(2);mapped2=f.approve(q2,name="two-attempt-work-approval")
        save("two-attempt-requirements.json",q2);save("two-attempt-mapping.json",mapped2)
        interrupted=components(f,mapped2);partial=f.base/"interrupted";partial.mkdir(mode=0o700);journal=f.base/"retained-work.jsonl"
        original=r.completed_dpo_update
        def fail_second(*args,**kwargs):
            if kwargs["update"]==2:
                with patch.object(args[0],"forward",side_effect=OSError("authored second-update entered-forward interruption")):
                    return original(*args,**kwargs)
            return original(*args,**kwargs)
        with ledger(journal,interrupted) as work:
            try:
                with patch.object(r,"completed_dpo_update",side_effect=fail_second):
                    train(f,interrupted,work,partial,baseline_sink=lambda rows:save("baseline-interrupted.json",rows))
            except OSError as error:
                failure=dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
            else:raise AssertionError("Predeclared interruption absent")
            spent=work.snapshot();receipt=base.checkpoint_receipt(partial/"checkpoints/completed-000000.pt")
        save("interruption.json",failure);save("work-after-interruption.json",spent)
        resumed=components(f,mapped2);target=f.base/"resumed";target.mkdir(mode=0o700)
        with ledger(journal,resumed,opening=True) as work:
            retained=payload(receipt,resumed,work)
            completed,counters,history=r.restore_dpo_payload(retained,contract=resumed[4],model=resumed[0],reference=resumed[1],
                optimizer=resumed[2],sampler=resumed[3],work_ledger=work)
            reopened=work.snapshot()
            if reopened["reserved"]!=spent["reserved"]:raise AssertionError("Resume refunded retained spending")
            continued=train(f,resumed,work,target,dict(completed=completed,counters=counters,history=history,parent_checkpoint=receipt),
                baseline_sink=lambda rows:save("baseline-resumed.json",rows))
            actual=payload(continued["checkpoint"],resumed,work);after=finish(f,resumed,work,continued);summary2=work.snapshot()
            numeric_keys=("policy","reference","optimizer","sampler_rng","torch_rng","counters","history")
            expected_digest=digest({key:saved["state"][key] for key in numeric_keys})
            actual_digest=digest({key:actual["state"][key] for key in numeric_keys})
            if expected_digest!=actual_digest:raise AssertionError("Same-environment numerical replay differs")
            with patch("torch.randint") as draw,patch.object(resumed[0],"forward") as forward:
                try:r.completed_dpo_update(*resumed[:4],f.encoded_train,pad_id=1,accumulation=4,beta=.1,device="cpu",update=3,work_ledger=work)
                except WorkBudgetExceeded as error:extra=dict(type=type(error).__name__,message=str(error),sampler_called=draw.called,forward_called=forward.called)
                else:raise AssertionError("Extra whole update unexpectedly fit exhausted capacity")
        save("resumed-training.json",continued);save("resumed-final-evaluation.json",after);save("work-after-resume.json",summary2)
        result["interrupted_replay"]=dict(expected_numerical_sha256=expected_digest,actual_numerical_sha256=actual_digest,exact=True,
            restored_update=completed,retained_failed_tickets=spent["failed_tickets"],reserved_before_resume=reopened["reserved"],
            reserved_after_resume=summary2["reserved"],extra_update_refusal=extra)
        # Real long tokenization gives the four-branch boundary without a pilot fit.
        pdir=destination/"encoded-pilot-boundary";pdir.mkdir(mode=0o700);pilot=Fixture(pdir,"assistant-dpo-pilot",True)
        pq=pilot.requirements();save("pilot-requirements-not-authorized.json",pq)
        save("actual-long-encodings.json",dict(train=pilot.encoded_train,validation=pilot.encoded_valid,prefixes=pilot.prefixes))
        rejected=[]
        for key in budget.KEYS:
            limits=deepcopy(q["required_limits"]);limits[key]-=1
            try:f.approve(q,limits,name="insufficient-"+key)
            except ValueError as error:rejected.append(dict(control="insufficient-"+key,type=type(error).__name__,message=str(error)))
            else:raise AssertionError("Insufficient dimension accepted")
        coarse=deepcopy(pq["required_limits"]);coarse["policy_forward_positions"]=coarse["reference_forward_positions"]=204800
        try:pilot.approve(pq,coarse,name="coarse-geometry-is-not-work")
        except ValueError as error:rejected.append(dict(control="coarse204800",type=type(error).__name__,message=str(error)))
        else:raise AssertionError("Coarse geometry was promoted to actual work approval")
        result["retained_refusals"]=rejected
        result["status"]="pass" if all(command["exit_code"]==0 for command in result["commands"]) else "fail"
    except BaseException as error:
        result["status"]="fail";result["failure"]=dict(type=type(error).__name__,message=str(error))
    result["source_sha256_after"]={name:r.file_digest(ROOT/name) for name in names}
    result["sources_unchanged"]=before==result["source_sha256_after"]
    if not result["sources_unchanged"]:result["status"]="fail"
    result["seconds"]=time.monotonic()-started
    result["limits"]=dict(supplier_provenance_authenticated=False,live_source_reencoding_gate="pending for real tokenizer/inputs",
        invocation_count_enforced=False,logical_work_not_flops=True,pretrained_gpu_mac_hosted="not executed")
    save("verification.json",result)
    return result


if __name__=="__main__":
    import sys
    if len(sys.argv)==3 and sys.argv[1]=="--collect":
        result=collect_reference(sys.argv[2]);print(json.dumps({key:result[key] for key in ("status","sources_unchanged","seconds")}))
        raise SystemExit(0 if result["status"]=="pass" else 1)
    unittest.main()

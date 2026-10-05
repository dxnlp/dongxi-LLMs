"""Prepare original comparison contracts; default execution launches no child.

Optional verification only runs fixed standard-library CPU supervisor fixtures.
There is no acquisition, model-loading, profile, inference or training command.
Planned parent slots are not checkpoint genealogy or model-scale permission.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import signal
import sys
import time

from .run_identity import canonical_hash, file_digest

ROOT = Path(__file__).resolve().parents[2]
SPEC = "experiments/specs/2026-10-04-staged-spark-campaign.md"
HISTORY = "experiments/reports/2026-09-14-tinystories-learning-result.json"
SOURCE_FILES = ("src/dongxi_llms/staged_campaign.py", "src/dongxi_llms/campaign_supervisor.py",
    "src/dongxi_llms/owned_worker_guards.py", "src/dongxi_llms/training_snapshot.py",
    "src/dongxi_llms/artifact_budget.py", "src/dongxi_llms/work_budget.py",
    "src/dongxi_llms/decoder_lab.py", "src/dongxi_llms/pretraining_lab.py",
    "src/dongxi_llms/dpo_lab.py", "src/dongxi_llms/grpo_lab.py",
    "src/dongxi_llms/batched_cache_lab.py", "src/dongxi_llms/evaluation_lab.py",
    "src/dongxi_llms/sampling_likelihood_lab.py",
    "src/dongxi_llms/dpo_stage_budget.py", "tests/test_dpo_stage_budget.py",
    "experiments/specs/2026-10-05-dpo-stage-budget-mapping.md",
    "src/dongxi_llms/production_stage.py", "src/dongxi_llms/production_preflight.py",
    "tests/test_production_stage.py", "tests/test_production_preflight.py",
    "experiments/specs/2026-10-05-production-stage-preparation.md",
    "tests/test_staged_campaign.py", SPEC, "src/dongxi_llms/run_identity.py",
    "src/dongxi_llms/stories_training.py", "src/dongxi_llms/stories_data.py",
    "scripts/launch_stories_learning.py", "scripts/train_stories.py", "scripts/run_chapter09_spark_sft.py",
    "scripts/run_chapter11_spark_dpo.py", "src/dongxi_llms/qwen_rlvr_lab.py",
    "src/dongxi_llms/reasoning_generation.py", "src/dongxi_llms/reasoning_evaluation.py",
    "fixtures/reasoning-controls/math_items.json", "fixtures/reasoning-controls/protocol.json",
    "scripts/prepare_chapter09_instruction_fixture.py", "experiments/data/instruction_interface_v1.jinja",
    "fixtures/chapter11/train.jsonl", "fixtures/chapter11/validation.jsonl", "fixtures/chapter11/evaluation.jsonl",
    HISTORY, "pyproject.toml", "uv.lock")

STORY_OPENINGS = (
    "Mara lent her blue umbrella to Ivo before the rain began.",
    "A small fox buried an apple beside the old stone gate.",
    "Nina saved two coins so she could repair her toy boat.",
    "When the bridge closed, Ben needed another way to visit Grandma.",
    "The farmer asked Ada to keep the tiny chick warm until morning.",
    "Oli left a red ribbon on the empty chair to remember his friend.",
    "A paper kite caught in a tree, and the children could not reach it.",
    "The baker gave Lila one loaf to carry safely across town.",
    "Sam promised to water the seed while his sister was away.",
    "A lost puppy followed the sound of a familiar little bell.",
    "Two mice disagreed about how to share the last piece of bread.",
    "After the snow melted, Eva found her missing mitten near the shed.")

ACTUAL_KEYS = ("checkpoint_files", "checkpoint_path", "parent_checkpoint_files", "interface_sha256",
    "upstream_revision_verified", "authorization_receipt", "profile", "recovery", "evaluation",
    "exit_code", "elapsed_seconds", "updates", "valid_training_targets", "padded_training_positions",
    "valid_prompt_tokens", "valid_response_tokens", "forwarded_generation_positions", "attempted_responses",
    "minimum_sampled_host_available_bytes", "cuda_peak_allocated_bytes", "failure")


def _contract(value):
    value = deepcopy(value)
    value["logical_contract_sha256"] = canonical_hash(value)
    return value


def evaluation_contracts(root):
    """Freeze authored logical panels, never pretend to observe model semantics."""
    story = _contract({"id": "story-publication-v1", "status": "logical-design-frozen-interface-pending",
        "items": [{"id": f"story-{i:02d}", "source_group": f"original-opening-{i:02d}", "prompt": prompt}
                  for i, prompt in enumerate(STORY_OPENINGS, 1)],
        "input_mode": "raw-story-prefix", "template": None, "thinking_mode": "not-applicable",
        "tokenizer_declaration": {"repository": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
             "declared_eos_and_bos_id": 50256, "observed_encoding_sha256": None},
        "decoding": [{"mode": "greedy", "temperature": 1., "top_k": None, "top_p": 1., "seed": None}]
            + [{"mode": "sample", "temperature": .8, "top_k": None, "top_p": 1., "seed": s} for s in (909,1909,2909)],
        "context_window": 1024, "max_new_tokens": 256, "predetermined_updates": [0,400,4000,8000,14000],
        "rubric": {
            "grammar": ["frequent broken constructions", "mostly readable with local errors", "consistently readable"],
            "entity_object_consistency": ["contradicts tracked entities or objects", "minor unclear reference", "preserves identities and locations"],
            "causal_continuity": ["events contradict the setup", "weak or partially unexplained links", "events follow intelligibly"],
            "repetition": ["persistent looping obstructs the story", "limited unnecessary repetition", "no obstructive repetition"],
            "ending": ["no resolution or incoherent ending", "partial resolution", "clear resolution consistent with setup"]},
        "rubric_scores": [0,1,2], "raters": 2, "blind_arm_checkpoint_labels": True,
        "uncertainty_unit": "source opening; paired group resampling, not individual sampled responses",
        "required_raw": ["token_ids", "text", "selected_likelihoods", "stop_reason", "raw_ratings", "rater_disagreement", "all_attempt_costs"],
        "stop_categories": ["natural-eos", "token-cap", "context-cap", "deadline", "resource-stop", "failure"],
        "contamination_audit": "pending exact/near-duplicate review against actual training bytes before publication use",
        "development_separation": "three historical inspected prompts excluded; publication never selects recipe/checkpoint/decoding",
        "nll": "fixed validation and matched frozen-checkpoint train/dev targets; not online-batch mean or story-rubric score"})
    assistant = _contract({"id": "assistant-interface-v1", "status": "logical-design-frozen-interface-pending",
        "source": "scripts/prepare_chapter09_instruction_fixture.py", "tasks": ["copy", "reverse", "extract"],
        "splits": {"train": {"value_groups": [0,79], "items": 240}, "development": {"value_groups": [80,99], "items": 60},
                   "publication": {"value_groups": [100,139], "items": 120}},
        "group_rule": "all task templates for one value stay together; task-template transfer is not established",
        "template_path": "experiments/data/instruction_interface_v1.jinja",
        "template_sha256": file_digest(root / "experiments/data/instruction_interface_v1.jinja"),
        "mask": "assistant body plus terminal marker/separator; prompt/header zero; one shift; right padding; reject overlength",
        "decoding": {"mode": "greedy", "temperature": 1., "top_k": None, "top_p": 1., "max_new_tokens": 64},
        "metrics": ["exact-answer by task", "termination", "format", "full-development target-weighted NLL", "source-group paired uncertainty"],
        "regression_panel": "reasoning panel and preference/location retention slices reported separately, not one capability percentage",
        "publication_policy": "fixed initial/final after development-only selection; no test-tuned coefficients",
        "observed_tokenizer_stop_interface": None})
    math_items = json.loads((root / "fixtures/reasoning-controls/math_items.json").read_text())
    seen = {tuple(sorted(p)) for p in ((2,3),(4,5),(3,6),(1,7))}
    annotated = []
    for item in math_items:
        row = deepcopy(item); p = row["problem"]
        overlap = p.get("operation") == "add" and tuple(sorted((p["a"],p["b"]))) in seen
        row["rlvr_train_problem_overlap"] = overlap
        row["campaign_role"] = "development-or-seen-diagnostic" if overlap or row["split"] == "train" else "heldout-controlled-slice"
        annotated.append(row)
    reasoning = _contract({"id": "reasoning-panel-v1", "status": "logical-design-frozen-interface-pending",
        "items": annotated, "items_sha256": file_digest(root / "fixtures/reasoning-controls/math_items.json"),
        "slices": ["arithmetic", "algebra", "multi-step", "heldout-source", "heldout-template", "heldout-family"],
        "parser": "bounded-rational-set-interval-v1; explicit extraction; INVALID/UNSUPPORTED/AMBIGUOUS retained",
        "decoding": {"mode": "sample", "temperature": 1., "top_k": None, "top_p": 1., "seeds": [1009,1019,1029,1039]},
        "budgets": {"answer-only": 32, "short-work": 128}, "greedy_diagnostic": True,
        "input_rows": ["base/raw", "base/explicit-chat", "instruct/chat/thinking-disabled", "same-instruct/chat/thinking-enabled"],
        "stop": "bind observed EOS/turn-stop/pad IDs before weights; natural-stop versus cap and unfinished work separately",
        "rlvr_reward": "existing strict whole-integer plus emitted EOS; training four pairs is distinct from this evaluation",
        "no_tuning": "publication/source-family slices never choose reward, budget, coefficients or checkpoints",
        "limits": "known train-problem overlaps excluded from heldout headline; common templates remain; final answer does not verify rationale faithfulness",
        "observed_checkpoint_interface": None})
    preference = _contract({"id": "preference-location-v1", "status": "logical-design-frozen-interface-pending",
        "input_sha256": {p: file_digest(root / p) for p in ("fixtures/chapter11/train.jsonl", "fixtures/chapter11/validation.jsonl", "fixtures/chapter11/evaluation.jsonl")},
        "slices": ["chosen/rejected absolute sequence likelihood", "reference-relative margin", "independent location extraction", "unrelated assistant/reasoning retention"],
        "scope": "small authored location fixture, not broad human preferences; source groups/template/masks must be audited before model use",
        "reference": "exact selected SFT parent, frozen bytes; no substituting a differently templated assistant",
        "generation": "same fixed initial/final greedy settings and publication source groups; no preference-margin selection"})
    return {c["id"]: c for c in (story,assistant,reasoning,preference)}


def budget(updates=None, *, microbatch=None, accumulation=None, length=None, seconds=None, targets=None, group=None, output_cap=None):
    physical = updates * microbatch * accumulation * length if all(x is not None for x in (updates,microbatch,accumulation,length)) else None
    return {"status": "proposed-unapproved-ceilings-profile-required", "maximum_updates": updates,
        "microbatch": microbatch, "accumulation": accumulation, "sequence_positions_per_example_max": length,
        "valid_training_target_cap": targets, "maximum_padded_training_positions": physical,
        "external_seconds": seconds, "group_size": group, "max_new_tokens": output_cap,
        "maximum_attempted_response_tokens": updates * group * output_cap if all(x is not None for x in (updates,group,output_cap)) else None,
        "target_guard": "must resolve before pilot approval; null exposure is not inferred from padded geometry",
        "memory": {"minimum_host_available_bytes": 25 * 1024**3, "cuda_peak": None, "profile_measured": False}}


def stage_rows():
    rows = []
    def add(identifier, branch, stage, parents, contract, limits, *, intervention=None, dependencies=(), blocker=None, optional=False):
        if stage == "frozen-evaluation":
            weight_start = list(parents)
        elif branch == "story":
            weight_start = identifier.replace("-recovery", "-smoke") if stage == "recovery" else "fresh-random-seed909-no-profile-transfer"
        elif branch == "assistant":
            if "-recovery" in identifier: weight_start = identifier.replace("-recovery", "-smoke")
            elif "dpo" in identifier or "chosen-sft" in identifier: weight_start = "assistant-selected-sft-parent"
            elif identifier == "assistant-lora-merge": weight_start = "assistant-sft-lora-pilot-with-exact-base06"
            elif stage == "frozen-evaluation": weight_start = list(parents)
            else: weight_start = "pinned-base06-no-profile-or-smoke-transfer"
        elif branch == "reasoning":
            weight_start = identifier.replace("-recovery", "-smoke") if stage == "recovery" else (
                "pinned-instruct06" if "rlvr" in identifier or "instruct" in identifier else "pinned-base06")
        else: weight_start = "separate-actually-obtained-branch-checkpoints"
        rows.append({"id": identifier, "branch": branch, "stage": stage, "prerequisite_stage_ids": list(parents),
            "proposed_weight_start": weight_start,
            "status": "pending-external-evidence", "actual": {k: None for k in ACTUAL_KEYS},
            "evaluation_contract": contract, "proposed_budget": limits, "intervention": intervention,
            "dependencies": ["DXI-01", "DXI-02", "DXI-17", *dependencies], "optional": optional,
            "gates": [{"id": name, "status": "pending", "evidence": None,
                       "scope": "campaign pilot approval; stage-specific required subset must be declared before its launch"} for name in
                ("stage-specific-authorization", "acquisition-and-data-rights", "local-bytes-and-interface", "resource-and-external-supervisor",
                 "profile", "finite-smoke-and-mask", "completed-and-pending-recovery", "frozen-evaluation-and-raw-records")],
            "known_blocker": blocker, "launch_command": None})
    add("story-profile", "story", "profile", (), "story-publication-v1", budget(40,microbatch=16,accumulation=1,length=1024,seconds=600), dependencies=("DXI-14",))
    for arm, rate in (("control",3e-4),("half-lr",1.5e-4)):
        previous = "story-profile"
        for name, updates, seconds in (("smoke",3,600),("recovery",3,600),("pilot",14000,14400)):
            ident = f"story-{arm}-{name}"
            add(ident,"story",name,(previous,),"story-publication-v1",
                budget(updates,microbatch=16,accumulation=1,length=1024,seconds=seconds,targets=50_000_000 if name == "pilot" else None),
                intervention={"peak_lr": rate, "floor_lr": rate/10, "seed": 909, "warmup": 200 if name == "pilot" else 1},
                blocker="whole-update target cap has CPU source proof; current-source Spark profile/recovery and matched fresh initialization remain pending", dependencies=("DXI-14",))
            previous = ident
    add("story-paired-comparison","story","frozen-evaluation",("story-control-pilot","story-half-lr-pilot"),"story-publication-v1",budget(output_cap=256),
        blocker="new paired training and two blinded raters unexecuted")
    add("assistant-profile-base06","assistant","profile",(),"assistant-interface-v1",budget(20,microbatch=1,accumulation=4,length=256,seconds=900),dependencies=("DXI-07",))
    add("assistant-frozen-base","assistant","baseline",("assistant-profile-base06",),"assistant-interface-v1",budget(output_cap=64))
    for arm in ("full","lora"):
        previous = "assistant-frozen-base"
        for name, updates, seconds in (("smoke",20,900),("recovery",20,900),("pilot",400,3600)):
            ident = f"assistant-sft-{arm}-{name}"
            add(ident,"assistant",name,(previous,),"assistant-interface-v1",budget(updates,microbatch=1,accumulation=4,length=256,seconds=seconds),
                intervention={"mode": arm,"rank": 8 if arm == "lora" else None,"learning_rate": 2e-5,"seed": 1212},
                dependencies=("DXI-07",),blocker="actual pretrained completed-update replay remains pending")
            previous = ident
        if arm == "lora":
            add("assistant-lora-merge","assistant","derived-checkpoint",(previous,),"assistant-interface-v1",budget(),blocker="merge exact recorded base; no adapter-as-full-policy handoff")
            previous = "assistant-lora-merge"
        add(f"assistant-sft-{arm}-evaluation","assistant","frozen-evaluation",(previous,),"assistant-interface-v1",budget(output_cap=64))
    add("assistant-selected-sft-parent","assistant","predeclared-full-sft-parent",("assistant-sft-full-evaluation","assistant-sft-lora-evaluation"),
        "assistant-interface-v1",budget(),intervention={"parent_rule":"full-SFT final checkpoint fixed before evaluation; failed full branch blocks downstream; no automatic LoRA substitution"},
        blocker="actual full-SFT parent remains null; changing parent requires a new declared comparison")
    rows[-1]["proposed_weight_start"]="assistant-sft-full-pilot"
    for arm in ("dpo","chosen-sft"):
        previous = "assistant-selected-sft-parent"
        for name, updates, seconds in (("smoke",2,600),("recovery",2,600),("pilot",100,1800)):
            ident=f"assistant-{arm}-{name}"
            add(ident,"assistant",name,(previous,),"preference-location-v1",budget(updates,microbatch=1,accumulation=4,length=512,seconds=seconds),
                intervention={"beta": .1 if arm == "dpo" else None,"learning_rate": 5e-7,"seed": 1818,"objective": arm},
                dependencies=("DXI-11","DXI-13","DXI-14"),
                blocker="DPO completed-update and checkpoint-mode recovery have CPU proof, not pretrained/CUDA/BF16 replay; chosen-SFT matched runner/control not yet wired")
            previous=ident
    add("assistant-preference-comparison","assistant","frozen-evaluation",("assistant-selected-sft-parent","assistant-dpo-pilot","assistant-chosen-sft-pilot"),
        "preference-location-v1",budget(output_cap=64))
    for arm in ("dpo-chosen-nll","dpo-rehearsal"):
        add(f"assistant-{arm}-extension","assistant","optional-pilot",("assistant-selected-sft-parent",),"preference-location-v1",budget(),
            dependencies=("DXI-11",),optional=True,blocker="new exact coefficients/data/token costs/runner and separate approval required")
    for role in ("base","instruct"):
        add(f"reasoning-profile-{role}","reasoning","profile",(),"reasoning-panel-v1",budget(output_cap=128),dependencies=("DXI-04","DXI-13"))
    for name,parent,thinking in (("base-raw","base","not-applicable"),("base-chat","base","explicit-template-no-unverified-toggle"),
                                 ("instruct-thinking-off","instruct","disabled"),("instruct-thinking-on","instruct","enabled")):
        add(f"reasoning-baseline-{name}","reasoning","baseline",(f"reasoning-profile-{parent}",),"reasoning-panel-v1",budget(output_cap=128),
            intervention={"checkpoint_role": parent,"thinking_mode": thinking,"budget_rows": [32,128]},dependencies=("DXI-04",),
            blocker="exact instruct revision/local bytes/template support and per-row authorization pending")
    for group in (4,8):
        previous="reasoning-baseline-instruct-thinking-off"
        for name,updates,cap,seconds in (("smoke",2,16,600),("recovery",2,16,600),("pilot",16,64,1800)):
            ident=f"reasoning-rlvr-g{group}-{name}"
            add(ident,"reasoning",name,(previous,),"reasoning-panel-v1",budget(updates,group=group,output_cap=cap,seconds=seconds),
                intervention={"seed":2323,"temperature":1.,"top_k":None,"top_p":1.,"lr":1e-6,"beta":.02,"epsilon":.2},
                dependencies=("DXI-04","DXI-12","DXI-13","DXI-14"),blocker="actual pretrained pending-rollout replay absent; primary runner diagnostic is not the common reasoning panel")
            previous=ident
    add("reasoning-g4-g8-comparison","reasoning","frozen-evaluation",("reasoning-baseline-instruct-thinking-off","reasoning-rlvr-g4-pilot","reasoning-rlvr-g8-pilot"),
        "reasoning-panel-v1",budget(output_cap=128),intervention={"equal_updates":16,"unequal_attempted_response_token_caps":[4096,8192]})
    add("assistant-1p7-confirmation","assistant","optional-confirmation",("assistant-preference-comparison",),"assistant-interface-v1",budget(),
        optional=True,blocker="new profile/uncertainty/scale budget and explicit acquisition/stage approval")
    add("reasoning-multi-seed-confirmation","reasoning","optional-confirmation",("reasoning-g4-g8-comparison",),"reasoning-panel-v1",budget(),
        optional=True,blocker="new justified seed/scale budget; no favorable single-seed selection")
    add("capstone-branch-comparison","capstone","frozen-evaluation",("story-paired-comparison","assistant-preference-comparison","reasoning-g4-g8-comparison"),
        "all-contracts-separately",budget(),blocker="only actually obtained compatible checkpoints; no universal pooled capability score")
    return rows


def prepare_campaign(root=ROOT):
    root=Path(root)
    history=json.loads((root / HISTORY).read_text())
    contracts=evaluation_contracts(root)
    rows=stage_rows()
    return {"schema":"dongxi-staged-campaign-v1","status":"prepared-model-scale-unexecuted",
        "jobs_started":0,"actual_genealogy":[],"stage_rows":rows,"evaluation_contracts":contracts,
        "models":{"base06":{"repository":"Qwen/Qwen3-0.6B-Base","declared_revision":"da87bfb608c14b7cf20ba1ce41287e8de496c0cd","actual_local_files":None},
                  "base17":{"repository":"Qwen/Qwen3-1.7B-Base","declared_revision":"ea980cb0a6c2ae4b936e82123acc929f1cec04c1","actual_local_files":None},
                  "instruct06":{"repository":"Qwen/Qwen3-0.6B","declared_revision":None,"actual_local_files":None}},
        "story_frozen_design":{"model":history["contract"]["model"],"control_recipe":history["contract"]["recipe"],
            "expected_data_manifest":history["data_manifest"],"expected_data_identity":history["contract"]["data"],
            "boundary":"historical declarations/bytes to revalidate locally; no current model/data loaded or authenticated"},
        "historical_anchor":{"report":HISTORY,"sha256":file_digest(root/HISTORY),"run":history["run"],
            "completion":history["completion"],"launch_exit":history["process_status"]["exit_code"],
            "claim":"existing measured baseline only; new matched training and systematic coherence scoring remain pending"},
        "authority":"no model/data acquisition, inference, profile, training, service or production supervisor integration authorized by preparation",
        "learner_position":"Day9; prepared future source is not study completion"}


def collect_cpu_fixtures(workspace):
    from .campaign_supervisor import GIB,Limits,host_memory,supervise_fixture
    workspace=Path(workspace);workspace.mkdir(parents=True,exist_ok=False)
    def synthetic_scan(*,exclude=(),found=False):
        return {"source":"injected-fixture-control","conflicts":[{"pid":987654321,"rss_bytes":5*GIB,"reasons":["large-resident-process"]}] if found else [],
            "unreadable":0,"scanned":1,"raw_command_lines_retained":False,"environments_read":False}
    rows=[]
    for name,mode in (("success","success"),("nonzero","exit7"),("self-signal","self-signal"),("deadline","ignore-term")):
        rows.append({"control":name,"result":supervise_fixture(mode,workspace/name,limits=Limits(seconds=.2 if name=="deadline" else 2.))})
    low=lambda:{"available_bytes":24*GIB,"source":"injected-fixture-control-not-host-shortage"}
    rows.append({"control":"preflight-reserve-rejection","result":supervise_fixture("success",workspace/"preflight-reserve",memory_probe=low)})
    count=0
    def later_low():
        nonlocal count
        count+=1
        return host_memory() if count==1 else low()
    rows.append({"control":"running-reserve-stop","result":supervise_fixture("ignore-term",workspace/"running-reserve",memory_probe=later_low)})
    rows.append({"control":"conflict-rejection","result":supervise_fixture("success",workspace/"conflict",conflict_probe=lambda **kw:synthetic_scan(found=True,**kw))})
    count=0
    def broken_probe():
        nonlocal count
        count+=1
        if count==1:return host_memory()
        raise OSError("deliberate injected observer failure")
    rows.append({"control":"observer-failure","result":supervise_fixture("ignore-term",workspace/"observer-failure",memory_probe=broken_probe)})
    expectations={"success":(0,None),"nonzero":(7,None),"self-signal":(-signal.SIGUSR1,None),
        "deadline":(-signal.SIGKILL,"external-deadline"),"preflight-reserve-rejection":(None,"host-reserve-below-threshold"),
        "running-reserve-stop":(None,"host-reserve-below-threshold"),"conflict-rejection":(None,"conflicting-or-uninspectable-process"),
        "observer-failure":(None,"supervisor-failure")}
    for row in rows:
        result=row["result"];expected_exit,expected_reason=expectations[row["control"]]
        exit_ok=result["actual_exit_code"]==expected_exit
        if row["control"] in ("running-reserve-stop","observer-failure"):
            exit_ok=result["child_pid"] is not None and result["actual_exit_code"] in (-signal.SIGTERM,-signal.SIGKILL)
        row["acceptance_passed"]=exit_ok and result["stop_reason"]==expected_reason
    return {"status":"passed" if all(x["acceptance_passed"] for x in rows) else "failed",
        "rows":rows,"boundary":"CPU fixtures only; injections are not real host/service failures; every external model stage remains pending"}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--verify-cpu-fixtures",action="store_true")
    parser.add_argument("--fixture-workspace",type=Path)
    args=parser.parse_args()
    if args.report.exists():parser.error("New report path required; preserve previous artifacts")
    if args.verify_cpu_fixtures != (args.fixture_workspace is not None):parser.error("Explicit fixture verification and a new fixture workspace are required together")
    before={p:file_digest(ROOT/p) for p in SOURCE_FILES};started=time.monotonic()
    campaign=prepare_campaign()
    fixtures=collect_cpu_fixtures(args.fixture_workspace) if args.verify_cpu_fixtures else None
    after={p:file_digest(ROOT/p) for p in SOURCE_FILES}
    report={"package":"DXI-03-preparation","status":"passed-preparation-external-evidence-pending" if before==after and (fixtures is None or fixtures["status"]=="passed") else "failed",
        "created_utc":datetime.now(timezone.utc).isoformat(),"command":list(sys.orig_argv),"python_argv":list(sys.argv),
        "environment":{"python":platform.python_version(),"executable":sys.executable,"platform":platform.platform(),"no_model_import_or_environment_install":True},
        "source_sha256":before,"source_changes":[p for p in before if before[p]!=after[p]],"seconds":time.monotonic()-started,
        "campaign":campaign,"cpu_fixture_verification":fixtures,"model_scale_execution":"unexecuted; preparation has no model launcher"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle:handle.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":report["status"],"planned_external_rows":len(campaign["stage_rows"]),"cpu_fixtures":0 if fixtures is None else len(fixtures["rows"])}))
    if report["status"]=="failed":raise SystemExit(1)


if __name__=="__main__":main()

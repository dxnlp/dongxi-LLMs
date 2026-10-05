"""Original complete-response distillation with actual teacher/student generation.

Printed arithmetic is checkable output, never a claim of faithful internal
reasoning. Shared09 digests and SFT loss are reused, not its color-task schema.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import re
import time

import torch

from .decoder_lab import DecoderConfig, TinyDecoder
from .evaluation_lab import canonical_hash
from .sft_lab import token_loss_sum
from .teacher_data_lab import attach_digest, validate_digest, derived_seed, state_digest
from .reasoning_generation import GenerationJournal
from .run_identity import collect_run_identity

SEEDS = (26011,26012,26013)
PAD,BOS,EOS,STEP,ANS,PARITY,ALIAS,POSITIVE,DIGIT = range(9)
VOCAB = {"PAD":PAD,"BOS":BOS,"EOS":EOS,"STEP":STEP,"ANS":ANS,"PARITY":PARITY,
         "ALIAS":ALIAS,"POSITIVE":POSITIVE,**{str(i):DIGIT+i for i in range(7)}}
TEMPLATE = "BOS instruction a b -> [STEP total] ANS binary EOS"
INTERFACE = canonical_hash({"vocabulary":VOCAB,"template":TEMPLATE,"eos":EOS,"context":12})
TEACHER_UPDATES,STUDENT_UPDATES,SAMPLES,CAP = 120,80,8,5


def prefix(item):
    return [BOS,ALIAS if item["template_id"]=="alias" else
            PARITY if item["family"]=="parity" else POSITIVE,
            DIGIT+item["problem"]["a"],DIGIT+item["problem"]["b"]]


def truth(item):
    total=item["problem"]["a"]+item["problem"]["b"]
    return total,total%2 if item["family"]=="parity" else int(total>0)


def validate_fixture(fixture):
    if not isinstance(fixture,dict) or fixture.get("schema")!="dongxi-response-distillation-fixture-v1":
        raise ValueError("Original response fixture schema required")
    items=fixture.get("items")
    if not isinstance(items,list) or not items:raise ValueError("Nonempty original suite required")
    ids,sources,problems,encodings=set(),{},{},{}
    for item in items:
        if not isinstance(item,dict):raise ValueError("Item must be an object")
        for key in ("id","source_group","split","slice","family","template_id","prompt"):
            if not isinstance(item.get(key),str) or not item[key].strip():raise ValueError("Nonempty identities required")
        if item["id"] in ids or item["split"] not in ("train","test") or item["slice"] not in ("train","heldout-source","heldout-template","heldout-family"):
            raise ValueError("Repeated ID or undeclared split/slice")
        ids.add(item["id"])
        if (item["split"]=="train")!=(item["slice"]=="train"):raise ValueError("Train split/slice mismatch")
        if item["family"] not in ("parity","sum-positive") or item["template_id"] not in ("direct","alias"):
            raise ValueError("Outside symbolic task family/template")
        p=item.get("problem")
        if not isinstance(p,dict) or set(p)!={"a","b"} or any(type(v)is not int or not 0<=v<=3 for v in p.values()):
            raise ValueError("Two integer operands0..3 required")
        if item.get("reference")!=str(truth(item)[1]):raise ValueError("Authored reference disagrees with arithmetic")
        for mapping,key in ((sources,item["source_group"]),(problems,canonical_hash([item["family"],p])),(encodings,canonical_hash(prefix(item)))):
            if mapping.setdefault(key,item["split"])!=item["split"]:raise ValueError("Source/problem/encoding leakage")
    train=[i for i in items if i["split"]=="train"]
    if not train or any(i["family"]!="parity" or i["template_id"]!="direct" for i in train):raise ValueError("Train is direct parity only")
    for probe in fixture.get("authored_probes",[]):
        total=probe["a"]+probe["b"]
        if probe["printed_step_valid"]!=(probe["step_total"]==total) or probe["answer_correct"]!=(probe["final_answer"]==total%2):
            raise ValueError("Authored probe labels disagree with arithmetic")
    return {"items":len(items),"sources":len(sources),"slices":dict(Counter(i["slice"] for i in items)),
            "train_item_ids":[i["id"] for i in train],"suite_sha256":canonical_hash(fixture)}


def make_model(seed,teacher=False):
    width,heads,hidden=(24,3,48) if teacher else (16,4,32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(DecoderConfig(vocab=15,width=width,heads=heads,kv_heads=heads,
            head_dim=width//heads,layers=1,hidden=hidden,max_length=12,modern=True,tied=False)).double().eval()


def body_shape(tokens):
    if not isinstance(tokens,list) or any(type(t)is not int or not 0<=t<15 for t in tokens):return None
    if len(tokens)==5 and tokens[0]==STEP and DIGIT<=tokens[1]<=DIGIT+6 and tokens[2]==ANS and tokens[3]in(DIGIT,DIGIT+1) and tokens[4]==EOS:
        return {"kind":"fulltrace","printed_total":tokens[1]-DIGIT,"answer":tokens[3]-DIGIT}
    if len(tokens)==3 and tokens[0]==ANS and tokens[1]in(DIGIT,DIGIT+1) and tokens[2]==EOS:
        return {"kind":"answer-only","printed_total":None,"answer":tokens[1]-DIGIT}
    return None


def batch_examples(examples):
    if not examples:raise ValueError("No supervised teacher examples")
    lengths=[len(e["prompt_ids"])+len(e["response_ids"]) for e in examples]
    if max(lengths)>12:raise ValueError("No silent truncation: sequence exceeds context")
    inputs=torch.zeros((len(examples),max(lengths)),dtype=torch.long)
    labels=torch.full_like(inputs,-100)
    for row,e in enumerate(examples):
        prompt,response=e["prompt_ids"],e["response_ids"]
        if not prompt or not response or response[-1]!=EOS or EOS in response[:-1] or any(type(t)is not int or not 0<=t<15 for t in prompt+response):
            raise ValueError("Malformed IDs or absent/embedded response EOS")
        ids=prompt+response;inputs[row,:len(ids)]=torch.tensor(ids)
        labels[row,len(prompt):len(ids)]=torch.tensor(response)
    return {"input_ids":inputs,"labels":labels,"supervised_targets":int((labels[:,1:]!=-100).sum())}


def fit_model(model,examples,*,updates,lr,progress=None):
    if type(updates)is not int or not 1<=updates<=120:raise ValueError("Bounded exact update count required")
    batch=batch_examples(examples);initial=state_digest(model);history=[]
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    began=time.perf_counter()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        total,count=token_loss_sum(model(batch["input_ids"]),batch["labels"])
        loss=total/count;loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise ValueError("Nonfinite response fit")
        reach={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.grad is not None} if update==0 else None
        optimizer.step();row={"update":update+1,"loss":float(loss.detach()),"gradient_norm_before_clip":float(norm),"first_gradient_reach":reach}
        history.append(row)
        if progress:progress(deepcopy(row))
    model.eval()
    return {"updates":updates,"initial_state_sha256":initial,"final_state_sha256":state_digest(model),
            "parameters":sum(p.numel() for p in model.parameters()),"supervised_targets_per_update":count,
            "supervised_target_presentations":count*updates,"padded_forward_positions":batch["input_ids"].numel()*updates,
            "example_ids":[e["item_id"] for e in examples],"history":history,"wall_seconds":time.perf_counter()-began}


def decode(tokens):
    if not isinstance(tokens,list) or any(type(t)is not int or not 0<=t<15 for t in tokens):
        raise ValueError("Text decoding requires declared exact integer token IDs")
    words={v:k for k,v in VOCAB.items()}
    return " ".join(words[t] for t in tokens)


def observed_available_memory():
    """One Linux /proc observation, never a continuously measured minimum."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])
    except (OSError,ValueError,IndexError):pass
    return None


@torch.no_grad()
def generate(model,item,contract,*,campaign,phase,ordinal):
    checkpoint=state_digest(model);rng_seed=derived_seed(campaign,checkpoint,item["id"],phase,ordinal)
    rng=torch.Generator().manual_seed(rng_seed);prompt=prefix(item);tokens=[];logps=[];error=None;stop="cap"
    cost={"generated_actions":0,"attempted_forward_calls":0,"completed_forward_calls":0,
          "attempted_forward_positions":0,"completed_forward_positions":0,"wall_seconds":0.}
    begin=time.perf_counter()
    try:
        for _ in range(CAP):
            ids=prompt+tokens;cost["attempted_forward_calls"]+=1;cost["attempted_forward_positions"]+=len(ids)
            logits=model(torch.tensor([ids]))[0,-1]
            cost["completed_forward_calls"]+=1;cost["completed_forward_positions"]+=len(ids)
            if not bool(torch.isfinite(logits).all()):raise ValueError("Nonfinite neural logits")
            lp=logits.log_softmax(-1);chosen=int(torch.multinomial(lp.exp(),1,generator=rng))
            tokens.append(chosen);logps.append(float(lp[chosen]))
            if chosen==EOS:stop="EOS";break
    except Exception as caught:error={"type":type(caught).__name__,"message":str(caught)};stop="error"
    cost.update(generated_actions=len(tokens),wall_seconds=time.perf_counter()-begin)
    return attach_digest({"schema":"dongxi-reasoning-teacher-response-v1","contract_id":contract["identity"],
        "interface_sha256":INTERFACE,"teacher_or_student_state_sha256":checkpoint,"campaign":campaign,
        "item_id":item["id"],"item_sha256":canonical_hash(item),"source_group":item["source_group"],"split":item["split"],
        "phase":phase,"ordinal":ordinal,"sample_id":canonical_hash([campaign,checkpoint,item["id"],phase,ordinal]),
        "attempt_seed":rng_seed,"prompt_ids":prompt,"token_ids":tokens,"raw_response":decode(tokens),
        "selected_model_log_probabilities":logps,"stop":stop,"error":error,"cost":cost,
        "provenance":"actual original tiny neural autoregressive response; no forced syntax or answer"})


def adapt_teacher(records,items,contract,teacher_state):
    if (contract.get("identity")!=canonical_hash({k:v for k,v in contract.items() if k!="identity"})
            or contract.get("vocabulary")!=VOCAB or contract.get("template")!=TEMPLATE
            or contract.get("interface_sha256")!=INTERFACE):
        raise ValueError("Teacher adapter contract/interface changed")
    mapping={i["id"]:i for i in items};seen=set();accepted={};audit=[]
    last_ordinal={}
    for row in records:
        validate_digest(row)
        item=mapping.get(row.get("item_id"));reasons=[]
        if row.get("sample_id") in seen:raise ValueError("Duplicate teacher attempt identity")
        seen.add(row.get("sample_id"))
        if item is None or item["split"]!="train" or row.get("split")!="train" or row.get("source_group")!=item["source_group"]:
            raise ValueError("Teacher data must retain exact train-only source provenance")
        if row.get("item_sha256")!=canonical_hash(item) or row.get("prompt_ids")!=prefix(item):raise ValueError("Changed teacher source or prompt encoding")
        if row.get("contract_id")!=contract["identity"] or row.get("interface_sha256")!=INTERFACE or row.get("teacher_or_student_state_sha256")!=teacher_state:
            raise ValueError("Teacher/template/tokenizer/state compatibility mismatch")
        if row.get("phase")!="teacher-data" or type(row.get("ordinal"))is not int or not 0<=row["ordinal"]<SAMPLES:
            raise ValueError("Teacher data coordinate outside frozen phase/sample range")
        if row["ordinal"]<=last_ordinal.get(item["id"],-1):raise ValueError("Teacher source coordinates must retain increasing collection order")
        last_ordinal[item["id"]]=row["ordinal"]
        coordinate=[row["campaign"],teacher_state,item["id"],row["phase"],row["ordinal"]]
        if row["attempt_seed"]!=derived_seed(*coordinate) or row["sample_id"]!=canonical_hash(coordinate):
            raise ValueError("Teacher attempt coordinate changed")
        if row.get("raw_response")!=decode(row["token_ids"]):raise ValueError("Teacher text/token serialization changed")
        shape=body_shape(row.get("token_ids"))
        if row.get("error") is not None:reasons.append("teacher_error")
        if row.get("stop")!="EOS":reasons.append("missing-natural-EOS")
        if shape is None or shape["kind"]!="fulltrace":reasons.append("malformed-or-nontrace")
        if not reasons and item["id"] not in accepted:
            accepted[item["id"]]={"item_id":item["id"],"source_group":item["source_group"],"split":"train",
                "prompt_ids":row["prompt_ids"],"response_ids":row["token_ids"],
                "parent_sample_id":row["sample_id"],"parent_payload_sha256":row["payload_sha256"],
                "teacher_state_sha256":teacher_state,"interface_sha256":INTERFACE,"transform":"full actual teacher response"}
        audit.append({"sample_id":row["sample_id"],"item_id":item["id"],"eligible":not reasons,
                      "selected":not reasons and accepted[item["id"]]["parent_sample_id"]==row["sample_id"],"reasons":reasons})
    missing=[i["id"] for i in items if i["split"]=="train" and i["id"] not in accepted]
    full=[accepted[i["id"]] for i in items if i["id"] in accepted]
    answer=[]
    for row in full:
        short=deepcopy(row);short["response_ids"]=[ANS,row["response_ids"][3],EOS]
        short["transform"]="derived answer-only subsequence from same parent teacher response";answer.append(short)
    return {"full":full,"answer_only":answer,"audit":audit,"missing_train_ids":missing,
            "selection_rule":"first eligible full trace, no semantic correctness filter"}


def grade(item,row):
    validate_digest(row);shape=body_shape(row["token_ids"]);total,expected=truth(item)
    formatted=shape is not None and row["stop"]=="EOS" and row["error"] is None
    answer_correct=bool(formatted and shape["answer"]==expected)
    step_valid=shape["printed_total"]==total if formatted and shape["printed_total"] is not None else None
    return {"sample_id":row["sample_id"],"item_id":item["id"],"source_group":item["source_group"],"slice":item["slice"],
        "answer_correct":answer_correct,"printed_step_valid":step_valid,"format_valid":formatted,
        "natural_EOS":row["stop"]=="EOS","joint_trace_correct":bool(answer_correct and step_valid is True),
        "wrong_step_right_answer":bool(answer_correct and step_valid is False),
        "valid_step_wrong_answer":bool(not answer_correct and step_valid is True),
        "step_status":"valid" if step_valid is True else "wrong" if step_valid is False else "not-provided-or-invalid"}


def select(pool,method):
    seen=set();eligible=[]
    views=[]
    contexts=set()
    for original in pool:
        validate_digest(original)
        row={k:deepcopy(original[k]) for k in ("sample_id","item_id","contract_id","teacher_or_student_state_sha256",
            "token_ids","stop","error","selected_model_log_probabilities","ordinal")}
        views.append(row)
        contexts.add((row["item_id"],row["contract_id"],row["teacher_or_student_state_sha256"]))
        if len(contexts)>1:raise ValueError("Cannot select from mixed item/model/contract pools")
        if row["sample_id"] in seen:raise ValueError("Repeated selection attempt")
        seen.add(row["sample_id"]);shape=body_shape(row["token_ids"])
        if shape is not None and row["stop"]=="EOS" and row["error"] is None:
            eligible.append((row["sample_id"],shape["answer"],row["selected_model_log_probabilities"]))
    if method=="first":return views[0]["sample_id"] if views else None
    if method not in ("majority","mean_logp"):raise ValueError("Undeclared selector")
    if not eligible:return None
    if method=="majority":
        counts=Counter(a for _,a,_ in eligible);maximum=max(counts.values())
        return next(i for i,a,_ in eligible if counts[a]==maximum)
    lengths={r["sample_id"]:len(r["token_ids"]) for r in views}
    if any(not scores or len(scores)!=lengths[i] or any(type(v)not in(int,float) or not math.isfinite(v) for v in scores) for i,_,scores in eligible):
        raise ValueError("Ranker needs finite actual model likelihoods")
    means=[sum(s)/len(s) for _,_,s in eligible];return eligible[means.index(max(means))][0]


def evaluate(model,items,contract,*,campaign,label,journal=None):
    pools=[]
    for item in items:
        pool=[generate(model,item,contract,campaign=campaign,phase="evaluation-"+label,ordinal=o) for o in range(SAMPLES)]
        if journal:
            for r in pool:journal.record(r)
        grades=[grade(item,r) for r in pool];mapping={g["sample_id"]:g for g in grades};decisions=[]
        for method in ("first","majority","mean_logp"):
            begin=time.perf_counter();selected=select(pool,method);g=mapping.get(selected)
            decisions.append({"method":method,"sample_id":selected,"selected_grade":g,
                "answer_success":bool(g and g["answer_correct"]),"joint_trace_success":bool(g and g["joint_trace_correct"]),
                "oracle_any_answer":any(g["answer_correct"] for g in grades),
                "selection_wall_seconds":time.perf_counter()-begin})
        pools.append({"item_id":item["id"],"source_group":item["source_group"],"slice":item["slice"],"responses":pool,"grades":grades,"decisions":decisions})
    summaries=[]
    for slice_name in ("all","train","heldout-source","heldout-template","heldout-family"):
        subset=[p for p in pools if slice_name=="all" or p["slice"]==slice_name]
        for index,method in enumerate(("first","majority","mean_logp")):
            rows=[p["decisions"][index] for p in subset]
            summaries.append({"slice":slice_name,"method":method,"items":len(rows),
                "answer_success":sum(r["answer_success"] for r in rows)/len(rows),
                "joint_trace_success":sum(r["joint_trace_success"] for r in rows)/len(rows),
                "oracle_any_answer":sum(r["oracle_any_answer"] for r in rows)/len(rows),
                "abstentions":sum(r["sample_id"] is None for r in rows)})
    return {"label":label,"state_sha256":state_digest(model),"parameters":sum(p.numel() for p in model.parameters()),
        "pools":pools,"summary":summaries,
        "cost":{k:sum(r["cost"][k] for p in pools for r in p["responses"]) for k in ("generated_actions","attempted_forward_positions","completed_forward_positions","wall_seconds")}}


def run_reference(fixture,contract,*,seeds=SEEDS,journal=None):
    audit=validate_fixture(fixture);validate_contract(contract,fixture)
    began=time.perf_counter();available_before=observed_available_memory()
    items=fixture["items"];train=[i for i in items if i["split"]=="train"]
    result={"protocol":"actual-response-distillation-v1","suite_audit":audit,"interface_sha256":INTERFACE,"runs":[],"authored_probes":fixture["authored_probes"]}
    for seed in seeds:
        teacher=make_model(seed+1000,True)
        gold=[{"item_id":i["id"],"prompt_ids":prefix(i),"response_ids":[STEP,DIGIT+truth(i)[0],ANS,DIGIT+truth(i)[1],EOS]} for i in train]
        run={"campaign":seed,"teacher_seed":seed+1000,"student_seed":seed};result["runs"].append(run)
        run["teacher_fit"]=fit_model(teacher,gold,updates=TEACHER_UPDATES,lr=.01,
            progress=(lambda r:journal.event("teacher_fit_update",campaign=seed,evidence=r)) if journal else None)
        teacher_records=[generate(teacher,i,contract,campaign=seed,phase="teacher-data",ordinal=o) for i in train for o in range(SAMPLES)]
        if journal:
            for r in teacher_records:journal.record(r)
        selection=adapt_teacher(teacher_records,items,contract,state_digest(teacher))
        run.update(teacher_records=teacher_records,teacher_data_grades=[grade(next(i for i in items if i["id"]==r["item_id"]),r) for r in teacher_records],selection=selection)
        initial=make_model(seed);models=[("original-student",initial),("teacher",teacher)];run["student_fits"]={}
        if not selection["missing_train_ids"]:
            for arm,examples in (("complete-response",selection["full"]),("answer-only",selection["answer_only"])):
                student=deepcopy(initial)
                fit=fit_model(student,examples,updates=STUDENT_UPDATES,lr=.015,
                    progress=(lambda r:journal.event("student_fit_update",campaign=seed,arm=arm,evidence=r)) if journal else None)
                run["student_fits"][arm]=fit;models.append((arm,student))
        else:run["blocked_reason"]="Missing eligible teacher coverage: no gold substitution or recipe repair"
        run["evaluations"]=[evaluate(m,items,contract,campaign=seed,label=label,journal=journal) for label,m in models]
        run["teacher_data_cost"]={k:sum(r["cost"][k] for r in teacher_records) for k in ("generated_actions","attempted_forward_positions","completed_forward_positions","wall_seconds")}
    result["wall_seconds"]=time.perf_counter()-began
    result["observed_mem_available_kib"]={"before":available_before,"after":observed_available_memory()}
    result["memory_boundary"]="Linux lifetime peakRSSKiB and two /proc MemAvailable observations, not tensor allocator or monitored available-memory minimum"
    result["peak_process_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["limits"]=["symbolic printed-step checks not faithful internal reasoning","actual tiny neural teacher, not pretrained/human rationale review",
        "responseNLL is not soft teacherKL/T²","equal80updates has unequal supervised token/forward budgets","teacher data/fit costs separate from student and evaluation","no model-scale transfer execution"]
    return result


def freeze_contract(fixture,spec_sha256):
    validate_fixture(fixture)
    if not isinstance(spec_sha256,str) or re.fullmatch(r"[0-9a-f]{64}",spec_sha256)is None:
        raise ValueError("Actual specification digest required")
    value={"schema":"dongxi-sequence-distillation-contract-v1","fixture_sha256":canonical_hash(fixture),
           "spec_sha256":spec_sha256,"interface_sha256":INTERFACE,"vocabulary":VOCAB,"template":TEMPLATE,
           "seeds":list(SEEDS),"teacher_updates":TEACHER_UPDATES,"student_updates":STUDENT_UPDATES,"samples":SAMPLES,"cap":CAP,
           "decoding":"full15way-temperature1-unforced-EOS","selection":"firsteligiblefulltrace-no-semantic-filter"}
    value["identity"]=canonical_hash(value);return value


def validate_contract(contract,fixture):
    if not isinstance(contract,dict) or contract.get("identity")!=canonical_hash({k:v for k,v in contract.items() if k!="identity"}):
        raise ValueError("Contract identity changed")
    if contract!=freeze_contract(fixture,contract.get("spec_sha256")):
        raise ValueError("Template/tokenizer/recipe compatibility mismatch")


def prepare_local_transfer(teacher,student,train_groups,test_groups,*,teacher_attempts=4,
                           teacher_cap=64,student_updates=80,reserve_gib=25,execute=False):
    """Validate a bounded local-only future plan. Never load/train/run any model.

    Each checkpoint's tokenizer/template must later be checked against its own
    recorded interface. Response text is re-encoded for the student; teacher ID
    arrays cannot be copied across vocabularies. No acquired artifact assumed.
    """
    if execute is not False:raise ValueError("This source prepares a protocol only; no execution authority")
    paths=[Path(teacher).expanduser().resolve(),Path(student).expanduser().resolve()]
    if paths[0]==paths[1] or any(not p.is_dir() for p in paths):raise ValueError("Two distinct existing local checkpoint directories required")
    for groups in (train_groups,test_groups):
        if not isinstance(groups,list) or not groups or any(not isinstance(g,str) or not g.strip() for g in groups) or len(set(groups))!=len(groups):
            raise ValueError("Nonempty unique frozen source groups required")
    if set(train_groups)&set(test_groups):raise ValueError("Transfer sources cross train/test")
    for value,maximum in ((teacher_attempts,8),(teacher_cap,256),(student_updates,200)):
        if type(value)is not int or not 1<=value<=maximum:raise ValueError("Transfer budget outside bounded protocol")
    if type(reserve_gib)not in(int,float) or not math.isfinite(reserve_gib) or reserve_gib<25:
        raise ValueError("At least25GiB host reserve required")
    descriptions=[]
    for path in paths:
        if (path/'adapter_config.json').exists():raise ValueError("Need local full or explicitly merged checkpoint, not an unmerged adapter")
        config=path/'config.json';tokenizer=path/'tokenizer.json'
        if not config.is_file() or not tokenizer.is_file() or not list(path.glob('*.safetensors')):
            raise ValueError("Local config/tokenizer/safetensors identities required; no download fallback")
        descriptions.append({'path':str(path),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            'weight_files':[p.name for p in sorted(path.glob('*.safetensors'))],
            'weight_identity_status':'Names inspected only; actual digests/interface/hardware verification required before execution'})
    plan={'schema':'dongxi-local-response-transfer-plan-v1','status':'prepared unexecuted local-only protocol',
        'teacher':descriptions[0],'student':descriptions[1],'train_source_groups':train_groups,'test_source_groups':test_groups,
        'teacher_attempts':teacher_attempts,'teacher_cap':teacher_cap,'student_updates':student_updates,'host_reserve_gib':reserve_gib,
        'execution_permitted':False,'teacher_model_loading':'not performed','student_model_loading':'not performed',
        'required_before_execution':['independently freeze actual saved tokenizer/template interfaces for each checkpoint',
            'hash all local weights and raw teacher records; re-encode teacher response text using student tokenizer',
            'preserve source splits/malformed/no-truncation/EOS gates and prompt-excluded labels',
            'review natural-language rationale claims separately; symbolic STEP checker is not a verifier',
            'approve Spark hardware profile, timed supervision and25GiB memory reserve',
            'retain attempted teacher and student work separately and publish a new actual report']}
    plan['identity']=canonical_hash(plan);return plan


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--spec",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args(argv)
    if args.output.exists():parser.error("New output directory required; preserve prior evidence")
    root=Path(__file__).resolve().parents[2];journal=GenerationJournal(args.output)
    sources=["src/dongxi_llms/response_distillation_lab.py","src/dongxi_llms/teacher_data_lab.py","src/dongxi_llms/sft_lab.py","src/dongxi_llms/decoder_lab.py","src/dongxi_llms/reasoning_generation.py","src/dongxi_llms/run_identity.py","tests/test_response_distillation_lab.py"]
    try:
        identity=collect_run_identity(root,source_files=sources,input_files=[args.fixture,args.spec]);journal.write_new("input-identity.json",identity)
        hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [*(root/s for s in sources),args.fixture,args.spec]}
        fixture=json.loads(args.fixture.read_text());contract=freeze_contract(fixture,hashes[str(args.spec)]);journal.write_new("contract.json",contract)
        torch.set_num_threads(1);result=run_reference(fixture,contract,journal=journal)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise RuntimeError("Source/input drift; records retained but invocation invalid")
        result.update(run_identity=identity,source_input_sha256=hashes,inputs_unchanged=True);journal.write_new("results.json",result)
        journal.event("completed",campaigns=len(result["runs"]));print(f"Saved complete response distillation: {args.output}/results.json")
    except (Exception,KeyboardInterrupt) as error:
        journal.event("failure",error_type=type(error).__name__,error=str(error));journal.write_new("failure.json",{"error_type":type(error).__name__,"error":str(error),"partial_evidence":"append-only attempts/fit events retained"});raise
    finally:journal.close()


if __name__=="__main__":main()

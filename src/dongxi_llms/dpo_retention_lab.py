"""Original tiny shared-decoder DPO/NLL/rehearsal experiment, CPU only.

Symbolic first-slot copying and independent parity are not language benchmarks.
The JSON contract freezes the recipe before fitting. All likelihoods score
response tokens including EOS; causal shifting happens exactly once.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .decoder_lab import DecoderConfig, TinyDecoder, parameter_count
from .dpo_lab import dpo_loss, sequence_logps
from .sft_lab import token_loss_sum

PAD, BOS, EOS, FIRST, PARITY, STYLE = range(6)
COLORS = range(6, 10)
NUMERALS = range(10, 14)
EVEN, ODD = 14, 15
CONDITIONS = ("clean", "noisy", "chosen-longer", "matched-long")


def load_contract(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2]/"fixtures/dpo-retention/contract.json"
    contract = json.loads(path.read_text())
    validate_contract(contract)
    return contract


def prompt(item):
    if item["task"] == "first":
        return [BOS, FIRST, 6+item["a"], 6+item["b"]]
    if item["task"] == "parity":
        return [BOS, PARITY, 10+item["a"], 10+item["b"]]
    raise ValueError("Unknown task")


def oracle(item):
    """Independent fixed copy rule or integer arithmetic, never a learned judge."""
    if item["task"] == "first":
        return 6+item["a"]
    if item["task"] == "parity":
        return EVEN + ((item["a"]+item["b"]) % 2)
    raise ValueError("Unknown task")


def validate_contract(c):
    if c.get("schema_version") != "dongxi-dpo-retention-v1":
        raise ValueError("Unknown contract schema")
    if len(set(c["seeds"])) != len(c["seeds"]) or not c["seeds"]:
        raise ValueError("Need distinct predeclared seeds")
    if c["conditions"] != list(CONDITIONS) or c["noise_indices"] != [1, 3]:
        raise ValueError("Unexpected preference intervention")
    if c["token_table"] != {"PAD":0,"BOS":1,"EOS":2,"FIRST":3,"PARITY":4,"STYLE":5,
                             "amber":6,"birch":7,"coral":8,"dune":9,"n0":10,"n1":11,
                             "n2":12,"n3":13,"even":14,"odd":15}:
        raise ValueError("Atomic token semantics changed")
    cfg = DecoderConfig(**c["model"])
    if cfg.vocab != 16 or cfg.max_length < 8:
        raise ValueError("Need the declared vocabulary and adequate context")
    if c["generation"] != {"decoding":"full-vocabulary-greedy","max_new_tokens":4,"eos":2}:
        raise ValueError("Generation contract changed")
    for recipe in (c["warmup"], c["fit"]):
        if not isinstance(recipe["updates"], int) or not 1 <= recipe["updates"] <= 400:
            raise ValueError("Invalid bounded update budget")
        if any(not math.isfinite(recipe[k]) or recipe[k] <= 0 for k in ("lr", "gradient_clip")):
            raise ValueError("Invalid optimization coefficient")
        if recipe["weight_decay"] != 0:
            raise ValueError("This contract declares zero weight decay")
    if not math.isfinite(c["fit"]["beta"]) or c["fit"]["beta"] <= 0:
        raise ValueError("Need positive beta")
    if len(c["arms"]) != 4 or len({a["name"] for a in c["arms"]}) != 4:
        raise ValueError("Need all four identified arms")
    for arm in c["arms"]:
        if any(not math.isfinite(arm[k]) or arm[k] < 0 for k in ("alpha", "gamma")):
            raise ValueError("Invalid auxiliary coefficient")
    ids, groups, problems, encoded = set(), {}, {}, {}
    counts = {}
    for item in c["items"]:
        if item["id"] in ids or item["task"] not in ("first", "parity") or item["split"] not in ("train", "heldout"):
            raise ValueError("Duplicate identity or unsupported task/split")
        ids.add(item["id"])
        if any(isinstance(item[k], bool) or not isinstance(item[k], int) or not 0 <= item[k] < 4 for k in ("a", "b")):
            raise ValueError("Operands must be known atomic symbols")
        if item["task"] == "first" and item["a"] == item["b"]:
            raise ValueError("Preference pair must compare distinct answers")
        key = (item["task"], item["a"], item["b"])
        code = tuple(prompt(item))
        if key in problems or code in encoded:
            raise ValueError("Duplicate underlying problem or encoded prompt")
        problems[key] = encoded[code] = item["split"]
        group = item["source_group"]
        if not isinstance(group, str) or not group or (group in groups and groups[group] != item["split"]):
            raise ValueError("Source group crosses a split")
        groups[group] = item["split"]
        slice_key = item["task"]+"/"+item["split"]
        counts[slice_key] = counts.get(slice_key, 0)+1
    if counts != {"first/train":4,"parity/train":4,"first/heldout":4,"parity/heldout":4}:
        raise ValueError("Need four fixed rows in every slice")


def branch(items, responses):
    """Right-pad only. Scored causal states precede all pad positions.

    TinyDecoder has no padding API. With right padding, no scored response can
    attend to a future pad. This does not justify left padding or packed rows.
    """
    if not items or len(items) != len(responses) or any(not y or y[-1] != EOS or
            EOS in y[:-1] or PAD in y for y in responses):
        raise ValueError("Need one nonempty EOS-terminated response per item")
    sequences = [prompt(x)+list(y) for x,y in zip(items,responses)]
    length = max(map(len,sequences))
    ids = torch.full((len(items),length), PAD, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    completion = torch.zeros_like(attention)
    for i, (x,y,seq) in enumerate(zip(items,responses,sequences)):
        ids[i,:len(seq)] = torch.tensor(seq)
        attention[i,:len(seq)] = True
        completion[i,len(prompt(x)):len(seq)] = True
    return ids, attention, completion


def pair_batches(c, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown data condition")
    items = [x for x in c["items"] if x["task"] == "first" and x["split"] == "train"]
    chosen = [[oracle(x),EOS] for x in items]
    rejected = [[6+x["b"],EOS] for x in items]
    if condition == "noisy":
        for index in c["noise_indices"]:
            chosen[index],rejected[index] = rejected[index],chosen[index]
    if condition in ("chosen-longer", "matched-long"):
        chosen = [[y[0],STYLE,STYLE,EOS] for y in chosen]
    if condition == "matched-long":
        rejected = [[y[0],STYLE,STYLE,EOS] for y in rejected]
    return branch(items,chosen), branch(items,rejected), items


def validate_branch(data):
    ids, attention, completion = data
    if ids.ndim != 2 or attention.shape != ids.shape or completion.shape != ids.shape:
        raise ValueError("Invalid branch shapes")
    if attention.dtype != torch.bool or completion.dtype != torch.bool or (completion & ~attention).any():
        raise ValueError("Invalid boolean response/attention masks")
    if (attention[:,1:] & ~attention[:,:-1]).any():
        raise ValueError("Only right padding is supported")
    if (completion[:,1:].sum(1)==0).any():
        raise ValueError("Every sequence needs response targets after shifting")


def scores(model, data):
    validate_branch(data)
    ids, attention, completion = data
    return sequence_logps(model(ids[:,:-1]),ids[:,1:],completion[:,1:] & attention[:,1:])


def response_nll(model, data):
    validate_branch(data)
    ids, attention, completion = data
    labels = ids.masked_fill(~(attention & completion), -100)
    total,count = token_loss_sum(model(ids),labels)
    return total/count, count


def combined_loss(pc, pr, rc, rr, lengths, *, beta=.5, alpha=0., gamma=0., rehearsal_nll=None):
    """Mean pair DPO + alpha global chosen-token NLL + gamma rehearsal NLL.

    Reference scores are detached by the existing DPO utility. Zero coefficients
    omit their branches, avoiding both an extra gradient path and 0*inf hazards.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Beta must be finite and positive")
    if pc.ndim != 1 or not pc.numel() or any(v.shape != pc.shape or not torch.isfinite(v).all() for v in (pc,pr,rc,rr)):
        raise ValueError("Need finite, nonempty, aligned score vectors")
    if lengths.shape != pc.shape or lengths.dtype not in (torch.int32,torch.int64) or (lengths <= 0).any():
        raise ValueError("Need positive integer response lengths")
    if any(not math.isfinite(x) or x < 0 for x in (alpha,gamma)):
        raise ValueError("Auxiliary coefficients must be finite and nonnegative")
    dpo,margin = dpo_loss(pc,pr,rc,rr,beta)
    loss = dpo
    chosen_nll = -pc.sum()/lengths.sum() if alpha else None
    if alpha:
        loss = loss + alpha*chosen_nll
    if gamma:
        if rehearsal_nll is None or rehearsal_nll.ndim != 0 or not torch.isfinite(rehearsal_nll):
            raise ValueError("Need a finite scalar rehearsal loss")
        loss = loss + gamma*rehearsal_nll
    return loss, {"dpo":dpo,"margin":margin,"chosen_nll":chosen_nll,
                  "rehearsal_nll":rehearsal_nll if gamma else None}


def make_model(c,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(DecoderConfig(**c["model"])).float().cpu().eval()


def state_hash(model):
    h = hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pair_cache_hash(reference,c,chosen,rejected,rc,rr):
    """Actual weights, atomic encoding/template, boundaries, dtype and scores."""
    h = hashlib.sha256(state_hash(reference).encode())
    h.update(json.dumps({"template":c["template"],"token_table":c["token_table"],
                         "normalization":"summed response log-likelihood including EOS"},sort_keys=True).encode())
    for value in (*chosen,*rejected,rc,rr):
        h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def render(tokens,c):
    names = {value:key for key,value in c["token_table"].items()}
    return " ".join(names.get(int(t),f"unknown:{t}") for t in tokens)


@torch.no_grad()
def generate(model,prefix,cap=4):
    if not prefix or not 1 <= cap <= model.cfg.max_length-len(prefix):
        raise ValueError("Generation cap exceeds context or is empty")
    ids = torch.tensor([prefix],dtype=torch.long)
    result = []
    for _ in range(cap):
        token = int(model(ids)[0,-1].argmax())
        result.append(token)
        if token == EOS:
            break
        ids = torch.cat((ids,torch.tensor([[token]])),1)
    return result


@torch.no_grad()
def evaluate(model,c,checkpoint):
    identity = hashlib.sha256(json.dumps({"generation":c["generation"],"template":c["template"],
             "token_table":c["token_table"],"items":c["items"]},sort_keys=True).encode()).hexdigest()
    rows = []
    for item in c["items"]:
        start = time.perf_counter()
        tokens = generate(model,prompt(item),c["generation"]["max_new_tokens"])
        elapsed = time.perf_counter()-start
        ended = tokens[-1] == EOS
        body = tokens[:-1] if ended else tokens
        expected = [oracle(item),EOS]
        expected_logp = float(scores(model,branch([item],[expected]))[0])
        rows.append({"id":item["id"],"source_group":item["source_group"],"task":item["task"],"split":item["split"],
            "checkpoint":checkpoint,"contract_id":identity,"prompt_tokens":prompt(item),"token_ids":tokens,
            "raw_response":render(tokens,c),"expected":expected,"expected_log_likelihood":expected_logp,
            "first_answer_correct":bool(body and body[0] == oracle(item)),
            "canonical_complete_success":tokens == expected,
            "format_valid":bool(ended and body and body[0] in (list(COLORS)+[EVEN,ODD]) and body[1:] in ([],[STYLE,STYLE])),
            "stop_reason":"eos" if ended else "max_tokens","truncated":not ended,
            "cost":{"generation_tokens":len(tokens),"generation_forwards":len(tokens),
                    "scoring_tokens":2,"scoring_forwards":1,"wall_seconds":elapsed}})
    summaries = {}
    for task in ("first","parity"):
        for split in ("train","heldout"):
            selected = [x for x in rows if x["task"] == task and x["split"] == split]
            summaries[task+"/"+split] = {"n":len(selected),
                **{key:sum(x[key] for x in selected)/len(selected) for key in
                   ("first_answer_correct","canonical_complete_success","format_valid","truncated")},
                "expected_log_likelihood":sum(x["expected_log_likelihood"] for x in selected)/len(selected)}
    return {"contract_id":identity,"rows":rows,"summaries":summaries}


def warm_start(c,seed,updates=None):
    model = make_model(c,seed)
    initial_hash = state_hash(model)
    initial = evaluate(model,c,f"seed{seed}-random")
    train = [x for x in c["items"] if x["split"] == "train"]
    data = branch(train,[[oracle(x),EOS] for x in train])
    recipe = c["warmup"]
    updates = recipe["updates"] if updates is None else updates
    if not 1 <= updates <= 400:
        raise ValueError("Invalid warm-up budget")
    optimizer = torch.optim.AdamW(model.parameters(),lr=recipe["lr"],weight_decay=recipe["weight_decay"])
    history = []
    for step in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        loss,count = response_nll(model,data)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip"])
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Nonfinite warm-up state")
        optimizer.step()
        history.append({"update":step,"loss":float(loss.detach()),"gradient_norm":float(norm)})
    return model,{"seed":seed,"updates":updates,"initial_state_sha256":initial_hash,"warm_state_sha256":state_hash(model),
                  "parameters":parameter_count(model),"initial":initial,"warm":evaluate(model,c,f"seed{seed}-warm{updates}"),
                  "history":history,"supervised_response_tokens":count*updates,"policy_forwards":updates}


def fit_arm(warm,c,seed,condition,arm,updates=None):
    model = deepcopy(warm).eval()
    reference = deepcopy(warm).eval().requires_grad_(False)
    ref_hash = state_hash(reference)
    chosen,rejected,items = pair_batches(c,condition)
    rehearsal_items = [x for x in c["items"] if x["task"] == "parity" and x["split"] == "train"]
    rehearsal = branch(rehearsal_items,[[oracle(x),EOS] for x in rehearsal_items])
    with torch.no_grad():
        rc,rr = scores(reference,chosen),scores(reference,rejected)
    cache_hash = pair_cache_hash(reference,c,chosen,rejected,rc,rr)
    lengths = chosen[2].sum(1)
    recipe = c["fit"]
    updates = recipe["updates"] if updates is None else updates
    if not 1 <= updates <= 400:
        raise ValueError("Invalid preference budget")
    optimizer = torch.optim.AdamW(model.parameters(),lr=recipe["lr"],weight_decay=recipe["weight_decay"])
    history = []
    first_gradient = None
    for step in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        pc,pr = scores(model,chosen),scores(model,rejected)
        replay,count = response_nll(model,rehearsal) if arm["gamma"] else (None,0)
        loss,parts = combined_loss(pc,pr,rc,rr,lengths,beta=recipe["beta"],
                                alpha=arm["alpha"],gamma=arm["gamma"],rehearsal_nll=replay)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip"])
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Nonfinite preference update")
        if step == 1:
            first_gradient = {name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        with torch.no_grad():
            post_c,post_r = scores(model,chosen),scores(model,rejected)
            _,post_margin = dpo_loss(post_c,post_r,rc,rr,recipe["beta"])
        history.append({"update":step,"loss":float(loss.detach()),"dpo":float(parts["dpo"].detach()),
            "chosen_nll":float(parts["chosen_nll"].detach()) if parts["chosen_nll"] is not None else None,
            "rehearsal_nll":float(replay.detach()) if replay is not None else None,
            "chosen_log_likelihood":float(post_c.mean()),"rejected_log_likelihood":float(post_r.mean()),
            "reference_relative_margin":float(post_margin.mean()),"gradient_norm":float(norm)})
    with torch.no_grad():
        cache_equal = torch.equal(scores(reference,chosen),rc) and torch.equal(scores(reference,rejected),rr)
    return {"seed":seed,"condition":condition,"arm":dict(arm),"updates":updates,
        "initial_state_sha256":state_hash(warm),"final_state_sha256":state_hash(model),
        "reference_state_sha256":ref_hash,"reference_unchanged":state_hash(reference)==ref_hash,
        "reference_has_gradients":any(x.grad is not None for x in reference.parameters()),"cache_equal":cache_equal,
        "cache_sha256":cache_hash,"first_gradient_reach":first_gradient,"history":history,
        "initial_pair":{"chosen_log_likelihood":float(rc.mean()),"rejected_log_likelihood":float(rr.mean()),
                        "reference_relative_margin":0.},
        "pair_rows":[{"id":x["id"],"source_group":x["source_group"],"chosen":chosen[0][i][chosen[2][i]].tolist(),
                      "rejected":rejected[0][i][rejected[2][i]].tolist(),"recorded_winner_matches_oracle":
                      int(chosen[0][i][chosen[2][i]][0]) == oracle(x)} for i,x in enumerate(items)],
        "final":evaluate(model,c,f"seed{seed}-{condition}-{arm['name']}-update{updates}"),
        "costs":{"pair_examples_per_update":len(items),"pair_score_response_tokens":int(chosen[2].sum()+rejected[2].sum())*updates,
                 "chosen_aux_supervised_tokens":int(chosen[2].sum())*updates if arm["alpha"] else 0,
                 "rehearsal_supervised_tokens":count*updates,"training_policy_forwards":(2+bool(arm["gamma"]))*updates,
                 "diagnostic_policy_forwards":2*updates,"reference_cache_forwards":2,"reference_check_forwards":2,
                 "boundary":"Chosen NLL reuses pair scores; its supervision tokens are not extra processed tokens. Rehearsal adds new targets/forwards. Generation costs are per record."}}


def run_retention(c=None,*,seeds=None,updates=None,warmup_updates=None):
    c = load_contract() if c is None else deepcopy(c)
    validate_contract(c)
    fixture_recipe = c == load_contract()
    seeds = c["seeds"] if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in c["seeds"] for seed in seeds):
        raise ValueError("Only distinct predeclared seeds are permitted")
    start = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    warm_records,runs = [],[]
    try:
        for seed in seeds:
            warm,record = warm_start(c,seed,warmup_updates)
            warm_records.append(record)
            for condition in c["conditions"]:
                for arm in c["arms"]:
                    runs.append(fit_arm(warm,c,seed,condition,arm,updates))
    finally:
        torch.set_num_threads(previous_threads)
    return {"mode":"actual tiny shared-decoder offline DPO with frozen SFT control","contract":c,
        "seeds":seeds,"warmups":warm_records,"runs":runs,"seconds":time.perf_counter()-start,
        "cpu_threads_during_fitting":1,"device":"cpu","dtype":"float32",
        "all_predeclared_rows_retained":len(runs)==len(seeds)*len(c["conditions"])*len(c["arms"]),
        "primary_recipe":fixture_recipe and seeds==c["seeds"] and updates in (None,c["fit"]["updates"]) and warmup_updates in (None,c["warmup"]["updates"]),
        "limits":["Symbolic tasks, not natural-language/human preference evidence", "No coefficient/checkpoint/seed chosen using held-out rows",
                  "Held-out parity uses previously unseen numeral tokens", "Extra chosen and rehearsal supervision are not equal-information controls",
                  "No universal retention repair or pretrained/Spark claim"]}


def main(argv=None):
    """Collect a new report without overwriting historical evidence."""
    import argparse
    import base64
    from datetime import datetime,timezone
    import os
    import platform
    import resource
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--notebook-manifest",type=Path)
    parser.add_argument("--export-previews",action="store_true")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Use an unused report path")
    root = Path(__file__).resolve().parents[2]
    files = ["src/dongxi_llms/dpo_retention_lab.py","tests/test_dpo_retention_lab.py","fixtures/dpo-retention/contract.json",
             "fixtures/dpo-retention/README.md","experiments/specs/2026-10-04-dpo-retention.md",
             "notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb","notebooks/day-18/README.md",
             "book/chapters/11-direct-preference-optimization.md","book/solutions/11-direct-preference-optimization.md",
             "book/labs/11-direct-preference-optimization.md","src/dongxi_llms/dpo_lab.py","src/dongxi_llms/sft_lab.py",
             "src/dongxi_llms/decoder_lab.py","pyproject.toml","uv.lock"]
    hashed = lambda: {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    before = hashed()
    historical = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                 ("experiments/reports/2026-10-04-preference-policy-cpu.json","experiments/reports/2026-10-04-preference-policy-cpu.md")}
    notebook,previews = None,[]
    if args.notebook_manifest:
        manifest = json.loads(args.notebook_manifest.read_text())
        notebook = next((x for x in manifest["notebooks"] if x["path"]==files[5]),None)
        if not notebook or manifest["failures"] or notebook["sha256"] != before[files[5]]:
            parser.error("Need passed fresh notebook evidence matching source")
    if args.export_previews:
        if not notebook:
            parser.error("Preview export requires a matching fresh notebook")
        import nbformat
        executed = nbformat.read(notebook["output"],as_version=4)
        images = [o.data["image/png"] for cell in executed.cells if cell.cell_type=="code" for o in cell.outputs
                  if o.output_type in ("display_data","execute_result") and "image/png" in o.data]
        if len(images) != 5:
            parser.error("Expected five original figures")
        for i,data in enumerate(images,1):
            p = root/f"notebooks/figures/chapter-11/day-18-02_dpo_retention_and_preference_controls-{i:02d}.png"
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(base64.b64decode(data))
            previews.append({"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    result = run_retention()
    env = dict(os.environ,PYTHONPATH=str(root/"src"),CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    checks = []
    for command in ([sys.executable,"-m","unittest","discover","-s","tests","-p","test_dpo_retention_lab.py","-v"],
                    [sys.executable,"scripts/check_book_math.py"],[sys.executable,"-m","unittest","discover","-s","tests","-v"]):
        child = subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=90)
        checks.append({"command":command,"exit_code":child.returncode,"stdout":child.stdout,"stderr":child.stderr})
    if hashed() != before or any(hashlib.sha256((root/p).read_bytes()).hexdigest()!=h for p,h in historical.items()):
        raise RuntimeError("Measured sources or historical evidence changed; preserve a coherent rerun")
    passed = result["primary_recipe"] and result["all_predeclared_rows_retained"] and all(x["reference_unchanged"] and
             not x["reference_has_gradients"] and x["cache_equal"] for x in result["runs"]) and all(x["exit_code"]==0 for x in checks)
    report = {"package":"DXI-11","status":"passed" if passed else "check_failed","date_utc":datetime.now(timezone.utc).isoformat(),
        "command":list(sys.orig_argv),"python_argv":list(sys.argv),
        "reproduction_command":[sys.executable,"-m","dongxi_llms.dpo_retention_lab",*sys.argv[1:]],
        "environment":{"python":platform.python_version(),"executable":sys.executable,"prefix":sys.prefix,
                       "platform":platform.platform(),"machine":platform.machine(),"torch":torch.__version__,
                       "cuda_available":torch.cuda.is_available(),"device":"cpu","dtype":"float32",
                       "thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
                       "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "maximum_rss_process_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "dirty_state":subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).splitlines(),
        "source_sha256":before,"historical_evidence_sha256":historical,"results":result,"checks":checks,
        "notebook_verification":notebook,"previews":previews,
        "notebook_manifest_sha256":hashlib.sha256(args.notebook_manifest.read_bytes()).hexdigest() if args.notebook_manifest else None,
        "acceptance_boundary":"Passing checks and all retained fits is instrument acceptance, not proof that an auxiliary repairs every regression",
        "resource_boundary":"Tiny CPU models only; Linux process RSS is not system/GPU peak; no pretrained/Spark or speed ranking"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":report["status"],"runs":len(result["runs"]),"seconds":result["seconds"]}))


if __name__ == "__main__":
    main()

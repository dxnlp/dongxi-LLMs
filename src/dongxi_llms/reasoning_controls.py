"""Original bounded constructive controls; no pretrained work or import effects.

The primary arm learns from sampled autoregressive rewards in a shared decoder.
Its declared position-specific syntax grammar is part of the objective. Exact
expected reward and a lookup arm are diagnostics, never substitutes for those
sampled updates. English fixture prompts are NOT parsed by this tiny decoder.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from .decoder_lab import DecoderConfig, TinyDecoder, parameter_count
from .evaluation_lab import canonical_hash
from .grpo_lab import clipped_objective, exact_kl, group_advantages
from .reasoning_evaluation import (SCHEMA_VERSION, freeze_contract, grade_response,
                                  canonicalize_answer, validate_record)

SEEDS = (2301, 2302, 2303)
UPDATES = 120
GROUP_SIZE = 16
LEARNING_RATE = .02
BETA = .01
BOS, EOS, PAD = 1, 2, 0
SUM, ALIAS, ODD = 3, 4, 5
NUMBER_START = 6
VOCAB = 10
FIRST = (NUMBER_START, NUMBER_START + 1)
LATER = (EOS, *FIRST)
SPLITS = ("train", "heldout-source", "heldout-template", "heldout-family")


def oracle_answer(item):
    """Independent integer/Fraction arithmetic, never model text or eval.

    This is a declared task oracle, not a neural parser or learned reasoning.
    Its structured inputs are separately authored from the response reference.
    """
    problem, family = item["problem"], item["family"]
    if any(type(value) is not int for key, value in problem.items() if key != "operation"):
        raise ValueError("Oracle operands must be exact integers")
    if family == "sum-positive":
        return Fraction(int(problem["a"] + problem["b"] > 0))
    if family == "odd-sum":
        return Fraction((problem["a"] + problem["b"]) % 2)
    if family == "arithmetic":
        a, b = Fraction(problem["a"]), Fraction(problem["b"])
        if problem["operation"] == "add":
            return a + b
        if problem["operation"] == "divide" and b:
            return a / b
    if family == "linear-algebra" and problem["a"]:
        return Fraction(problem["c"] - problem["b"], problem["a"])
    if family == "multi-step" and problem["groups"] > 0:
        return Fraction(problem["start"] + problem["receive"], problem["groups"])
    raise ValueError("Outside this independently fixed oracle's task grammar")


def answer_text(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def validate_items(items):
    """Reject ID, underlying problem and source leakage, not only prompt overlap."""
    if not items:
        raise ValueError("Nonempty original fixture required")
    ids, problems, groups = set(), {}, {}
    for item in items:
        for key in ("id", "source_group", "split", "task", "family", "template_id",
                    "problem", "prompt", "kind", "reference"):
            if key not in item:
                raise ValueError(f"Missing fixture field {key}")
        if item["id"] in ids or item["split"] not in SPLITS:
            raise ValueError("Duplicate ID or unknown split")
        ids.add(item["id"])
        for key in ("source_group", "template_id", "family", "prompt"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError("Nonempty source/template/family/prompt identities required")
        problem_id = canonical_hash({"family": item["family"], "problem": item["problem"]})
        if problems.setdefault(problem_id, item["split"]) != item["split"]:
            raise ValueError("Underlying mathematical problem crosses splits")
        if groups.setdefault(item["source_group"], item["split"]) != item["split"]:
            raise ValueError("Source group crosses splits")
        expected = canonicalize_answer(answer_text(oracle_answer(item)))
        actual = canonicalize_answer(item["reference"])
        if expected.get("canonical") != actual.get("canonical") or actual["status"] != "SUPPORTED":
            raise ValueError("Authored reference disagrees with independent arithmetic oracle")
    train = [item for item in items if item["split"] == "train"]
    if not train:
        raise ValueError("Need a frozen training split")
    templates = {item["template_id"] for item in train}
    families = {item["family"] for item in train}
    if any(item["template_id"] in templates for item in items if item["split"] == "heldout-template"):
        raise ValueError("Unseen-template slice contains a training template")
    if any(item["family"] in families for item in items if item["split"] == "heldout-family"):
        raise ValueError("Unseen-family slice contains a training task family")
    return {"count": len(items), "by_split": dict(Counter(item["split"] for item in items)),
            "training_templates": sorted(templates), "training_families": sorted(families),
            "suite_sha256": canonical_hash(items),
            "boundary": "unseen-template rows also have unseen sources; not a pure template ablation"}


def load_fixtures(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = root / "fixtures/reasoning-controls"
    control = json.loads((directory / "control_items.json").read_text())
    math_items = json.loads((directory / "math_items.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    validate_items(control)
    validate_items(math_items)
    return control, math_items, protocol


def prompt_ids(items):
    """Explicit symbolic interface, not English parsing or answer injection."""
    rows = []
    for item in items:
        if item["family"] not in ("sum-positive", "odd-sum"):
            raise ValueError("Tiny control vocabulary does not encode natural-language math panel")
        a, b = item["problem"]["a"], item["problem"]["b"]
        if not (type(a) is int and type(b) is int and 0 <= a <= 3 and 0 <= b <= 3):
            raise ValueError("Control operands must be0..3")
        instruction = ODD if item["family"] == "odd-sum" else (
            ALIAS if item["template_id"] == "predicate-worded" else SUM)
        rows.append([BOS, instruction, NUMBER_START+a, NUMBER_START+b])
    return torch.tensor(rows, dtype=torch.long)


def make_decoder(seed):
    if type(seed) is not int:
        raise ValueError("Integer initialization seed required")
    config = DecoderConfig(vocab=VOCAB, width=16, heads=4, kv_heads=2,
        head_dim=4, layers=1, hidden=32, max_length=8, modern=True, tied=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(config).eval()


def frozen_copy(model):
    return deepcopy(model).eval().requires_grad_(False)


def state_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decode(tokens):
    words = []
    for token in tokens:
        if token == EOS:
            break
        words.append(str(token-NUMBER_START) if NUMBER_START <= token < VOCAB else
                     {PAD:"<pad>", BOS:"<bos>", SUM:"<sum>", ALIAS:"<alias>", ODD:"<odd>"}[token])
    return " ".join(words)


@torch.no_grad()
def generate(model, prompts, *, cap=2, grammar=True, generator=None, greedy=False):
    """Actual autoregressive sampling with declared conditional syntax support."""
    if prompts.ndim != 2 or cap not in (1, 2) or prompts.shape[1]+cap > model.cfg.max_length:
        raise ValueError("Bounded nonempty prompts and cap1 or2 required")
    if not greedy and generator is None:
        raise ValueError("Sampled generation needs an explicit local RNG")
    current = prompts.clone()
    finished = torch.zeros(len(prompts), dtype=torch.bool)
    outputs, masks = [], []
    for position in range(cap):
        allowed = torch.tensor(FIRST if position == 0 else LATER) if grammar else torch.arange(VOCAB)
        logits = model(current)[:, -1, allowed]
        selection = logits.argmax(-1) if greedy else torch.multinomial(
            logits.softmax(-1), 1, generator=generator).squeeze(-1)
        token = allowed[selection]
        token = torch.where(finished, torch.full_like(token, PAD), token)
        outputs.append(token)
        masks.append(~finished)
        finished |= token == EOS
        current = torch.cat((current, token[:, None]), -1)
    return torch.stack(outputs, -1), torch.stack(masks, -1)


def path_logits(model, prompts, responses):
    if prompts.shape[0] != responses.shape[0] or responses.shape[1] != 2:
        raise ValueError("Training uses two-token paths and aligned prompt batches")
    inputs = torch.cat((prompts, responses), -1)[:, :-1]
    start = prompts.shape[1]-1
    logits = model(inputs)[:, start:start+2]
    return logits[:, 0, FIRST], logits[:, 1, LATER]


def path_log_probabilities(model, prompts, responses):
    first, later = path_logits(model, prompts, responses)
    first_action = responses[:, 0] - NUMBER_START
    # The second grammar's order is EOS,number0,number1.
    later_action = torch.where(responses[:, 1] == EOS, 0, responses[:, 1]-NUMBER_START+1)
    if bool(((first_action < 0) | (first_action > 1) | (later_action < 0) | (later_action > 2)).any()):
        raise ValueError("Recorded path violates its declared sampling grammar")
    return torch.stack((first.log_softmax(-1).gather(-1, first_action[:, None]).squeeze(-1),
                        later.log_softmax(-1).gather(-1, later_action[:, None]).squeeze(-1)), -1)


def strict_path_reward(tokens, mask, expected):
    active = tokens[mask].tolist()
    return float(len(active) == 2 and active == [NUMBER_START+int(expected), EOS])


@torch.no_grad()
def expected_success(model, items):
    """Diagnostic exact probability of the one correct complete path per item."""
    prompts = prompt_ids(items)
    answers = torch.tensor([int(oracle_answer(item)) for item in items])
    correct_paths = torch.stack((answers+NUMBER_START, torch.full_like(answers, EOS)), -1)
    return path_log_probabilities(model, prompts, correct_paths).sum(-1).exp()


@torch.no_grad()
def expected_answer_success(model, items):
    """Answer-only diagnostic; do not confuse it with learned termination."""
    target = torch.tensor([int(oracle_answer(item)) for item in items])
    return model(prompt_ids(items))[:, -1, FIRST].softmax(-1).gather(-1,target[:,None]).squeeze(-1)


def evaluate_policy(model, items, *, checkpoint_id, seed, cap=2, grammar=True,
                    sampled_per_item=0):
    """Every generated row is graded; no invalid/truncated filtering of averages."""
    if type(sampled_per_item) is not int or sampled_per_item < 0:
        raise ValueError("Nonnegative sampled repetition count required")
    settings = {"template_id":"tiny-symbolic-grammar" if grammar else "tiny-symbolic-full-vocabulary",
                "thinking_mode":"not-supported", "decoding":"greedy-plus-fixed-seeded-samples",
                "stopping":[EOS], "max_new_tokens":cap}
    contract = freeze_contract(items, settings)
    rows = []
    rng = torch.Generator().manual_seed(seed)
    for item in items:
        prompt = prompt_ids([item])
        for sample in range(sampled_per_item+1):
            started = time.perf_counter()
            output, mask = generate(model, prompt, cap=cap, grammar=grammar,
                generator=rng, greedy=sample == 0)
            active = output[0, mask[0]].tolist()
            ended = bool(active and active[-1] == EOS)
            record = {"schema_version":SCHEMA_VERSION,"contract_id":contract["identity"],
                "checkpoint_id":checkpoint_id, "sample_id":f"{item['id']}-{'greedy' if sample == 0 else 'sample-'+str(sample)}",
                "item_id":item["id"], "source_group":item["source_group"], "task":item["task"],
                "split":item["split"], "raw_response":decode(active), "token_ids":active,
                "prompt_tokens":prompt.shape[1],"generated_tokens":len(active),
                "stop_reason":"eos" if ended else "max_tokens", "truncated":not ended,"error":None,
                "cost":{"wall_seconds":time.perf_counter()-started,"generation_tokens":len(active),"scoring_tokens":0}}
            validate_record(record, contract, item)
            graded = grade_response(item, record)
            graded.update(template_id=item["template_id"], family=item["family"],
                          decoding="greedy" if sample == 0 else "sampled",
                          # Common grader reports answer/format separately from stop.
                          complete_path_success=bool(graded["task_success"] and ended),
                          strict_reward=bool(strict_path_reward(output[0],mask[0],oracle_answer(item))))
            rows.append(graded)
    summaries = {}
    for split in SPLITS:
        for decoding in ("greedy", "sampled"):
            selected = [r for r in rows if r["split"] == split and r["decoding"] == decoding]
            if selected:
                n = len(selected)
                summaries[f"{split}/{decoding}"] = {"n":n,
                    "answer_accuracy":sum(r["correct"] for r in selected)/n,
                    "format_valid":sum(r["format_valid"] for r in selected)/n,
                    "complete_path_success":sum(r["complete_path_success"] for r in selected)/n,
                    "invalid_or_unsupported":sum(r["status"] not in ("CORRECT","INCORRECT") for r in selected)/n,
                    "truncation":sum(r["truncated"] for r in selected)/n,
                    "generated_tokens":sum(r["generated_tokens"] for r in selected),
                    "statuses":dict(Counter(r["status"] for r in selected))}
    return {"contract":contract,"summaries":summaries,"rows":rows,
            "exact_success_by_item":dict(zip((i["id"] for i in items),expected_success(model,items).tolist())) if grammar and cap == 2 else None}


def decoder_control(items, seed, *, updates=UPDATES, group_size=GROUP_SIZE):
    """Sampled fresh-rollout GRPO on train only; no exact-reward training path."""
    validate_items(items)
    if type(updates) is not int or not 1 <= updates <= UPDATES or type(group_size) is not int or not 2 <= group_size <= GROUP_SIZE:
        raise ValueError("Use bounded positive update/group budgets")
    train = [i for i in items if i["split"] == "train"]
    policy = make_decoder(seed)
    reference = frozen_copy(policy)
    frozen_hash = state_hash(reference)
    initial_hash = state_hash(policy)
    initial = evaluate_policy(policy,items,checkpoint_id=f"decoder-seed{seed}-initial",seed=seed+100,
                              sampled_per_item=4)
    initial_probabilities = expected_success(policy,train).tolist()
    initial_answer_probabilities = expected_answer_success(policy,train).tolist()
    if not all(0 < value < .9 for value in initial_probabilities):
        raise RuntimeError("Constructive control must begin strictly unsaturated")
    optimizer = torch.optim.AdamW(policy.parameters(),lr=LEARNING_RATE,weight_decay=0.)
    generator = torch.Generator().manual_seed(seed+1)
    repeated = [item for item in train for _ in range(group_size)]
    prompts = prompt_ids(repeated)
    history, detailed = [], []
    for update in range(updates):
        responses, mask = generate(policy,prompts,generator=generator)
        reward = torch.tensor([strict_path_reward(row,valid,oracle_answer(item))
                               for row,valid,item in zip(responses,mask,repeated)])
        advantages = group_advantages(reward.reshape(len(train),group_size)).flatten()
        with torch.no_grad():
            old_logp = path_log_probabilities(policy,prompts,responses).detach()
            ref_first, ref_later = path_logits(reference,prompts,responses)
        current = path_log_probabilities(policy,prompts,responses)
        first, later = path_logits(policy,prompts,responses)
        kl = torch.stack((exact_kl(first,ref_first),exact_kl(later,ref_later)),-1)
        alignment = float((current.detach()-old_logp).abs().max())
        loss = clipped_objective(current,old_logp,advantages,mask,kl=kl,beta=BETA)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(policy.parameters(),1.))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
            raise RuntimeError("Nonfinite constructive-control loss/gradient")
        reached = {name:float(parameter.grad.abs().sum()) for name,parameter in policy.named_parameters()
                   if parameter.grad is not None} if update == 0 else None
        eos_gradient = float(policy.lm_head.weight.grad[EOS].abs().sum()) if update == 0 else None
        optimizer.step()
        diagnostic = float(expected_success(policy,train).mean())
        history.append({"update":update+1,"behavior_version":update,"policy_version":update+1,
            "sampled_reward":float(reward.mean()),"diagnostic_exact_train_success":diagnostic,
            "gradient_norm_before_clip":gradient,"loss":float(loss.detach()),
            "zero_variance_fraction":float((reward.reshape(len(train),group_size).std(-1,correction=0)==0).float().mean()),
            "valid_response_tokens":int(mask.sum()),"truncated_responses":int((responses[:,-1]!=EOS).sum()),
            "initial_log_ratio_max_abs":alignment,"first_gradient_reach":reached,
            "first_eos_output_gradient_abs_sum":eos_gradient})
        if update in (0,updates-1):
            detailed.append({"update":update+1,"prompt_item_ids":[i["id"] for i in repeated],
                "responses":responses.tolist(),"response_mask":mask.tolist(),"raw_responses":[decode(r.tolist()) for r in responses],
                "old_behavior_logp":old_logp.tolist(),"rewards":reward.tolist(),"advantages":advantages.tolist(),
                "stops":["eos" if r[-1]==EOS else "max_tokens" for r in responses]})
    if state_hash(reference) != frozen_hash:
        raise AssertionError("Frozen paired/reference policy changed")
    final = evaluate_policy(policy,items,checkpoint_id=f"decoder-seed{seed}-update{updates}",seed=seed+100,
                            sampled_per_item=4)
    frozen = evaluate_policy(reference,items,checkpoint_id=f"frozen-seed{seed}",seed=seed+100,
                             sampled_per_item=4)
    interventions = {"cap1":evaluate_policy(policy,items,checkpoint_id=f"decoder-seed{seed}-cap1",seed=seed+100,cap=1),
                    "full_vocabulary":evaluate_policy(policy,items,checkpoint_id=f"decoder-seed{seed}-full-vocab",seed=seed+100,grammar=False)}
    return {"seed":seed,"initial":initial,"final":final,"frozen":frozen,"interventions":interventions,
            "architecture":policy.cfg.__dict__,"parameters":parameter_count(policy),
            "initial_state_sha256":initial_hash,"final_state_sha256":state_hash(policy),
            "frozen_state_sha256":frozen_hash,"updates":updates,"group_size":group_size,
            "history":history,"first_and_last_rollout_groups":detailed,
            "initial_train_complete_path_probabilities":initial_probabilities,
            "final_train_complete_path_probabilities":expected_success(policy,train).tolist(),
            "initial_train_answer_only_probabilities":initial_answer_probabilities,
            "final_train_answer_only_probabilities":expected_answer_success(policy,train).tolist(),
            "training_response_tokens":sum(r["valid_response_tokens"] for r in history),
            "scope":"sampled autoregressive conditional-grammar GRPO; no SFT warm-start or English parsing"}


class LookupPolicy(nn.Module):
    """Separate exact-fit counterexample: unseen problem keys use a frozen row."""
    def __init__(self, train):
        super().__init__()
        self.keys = {canonical_hash({"family":i["family"],"problem":i["problem"]}):index+1
                     for index,i in enumerate(train)}
        self.logits = nn.Parameter(torch.zeros(len(self.keys)+1,2))

    def forward(self, items):
        indices = [self.keys.get(canonical_hash({"family":i["family"],"problem":i["problem"]}),0) for i in items]
        return self.logits[indices].softmax(-1)


def lookup_control(items, updates=40):
    """Exact one-answer expected reward; independent EOS oracle, not decoder RL."""
    train = [i for i in items if i["split"] == "train"]
    model = LookupPolicy(train)
    targets = torch.tensor([int(oracle_answer(i)) for i in train])
    optimizer = torch.optim.SGD(model.parameters(),lr=2.)
    before = model(items).detach()
    for _ in range(updates):
        objective = model(train).gather(-1,targets[:,None]).mean()
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        optimizer.step()
    after = model(items).detach()
    rows = [{"item_id":item["id"],"split":item["split"],"initial_correct_probability":float(p[int(oracle_answer(item))]),
             "final_correct_probability":float(q[int(oracle_answer(item))]),
             "final_greedy_answer":int(q.argmax()),"final_greedy_correct":int(q.argmax())==int(oracle_answer(item))}
            for item,p,q in zip(items,before,after)]
    return {"updates":updates,"objective":"exact expected correct-answer reward, EOS supplied by oracle",
            "rows":rows,"unknown_key_probability":model.logits[0].detach().softmax(-1).tolist(),
            "boundary":"Key memorization, not sampled decoder training or reasoning transfer"}


def math_panel_audit(items):
    """Author/oracle/parser checks only; these strings are not model generations."""
    validate_items(items)
    rows = []
    for item in items:
        text = answer_text(oracle_answer(item))
        for variant,raw,stop in (("oracle",text,"eos"),("wrong",answer_text(oracle_answer(item)+1),"eos"),
                                 ("invalid","not a numeric answer","eos"),("truncated",text,"max_tokens")):
            record = {"raw_response":raw,"stop_reason":stop,"truncated":stop=="max_tokens","error":None}
            grade = grade_response(item,record)
            rows.append({"item_id":item["id"],"source_group":item["source_group"],"split":item["split"],
                "family":item["family"],"template_id":item["template_id"],"variant":variant,
                "raw_response":raw,"reference":item["reference"],"status":grade["status"],
                "correct":grade["correct"],"truncated":record["truncated"],
                "complete_success":bool(grade["task_success"] and stop=="eos")})
    return {"mode":"hand-authored oracle/parser probes, NOT language-model generations","rows":rows}


def constant_baselines(items):
    """Fixed0/1 plus oracle EOS reveal class imbalance, not trained behavior."""
    baselines = []
    for answer in (0,1):
        rows = [{"item_id":item["id"],"split":item["split"],"answer":answer,
                 "response_tokens":[NUMBER_START+answer,EOS],
                 "complete_success":answer==oracle_answer(item)} for item in items]
        baselines.append({"answer":answer,"mode":"constructed constant answer + EOS, no model",
                          "rows":rows,"by_split":{split:sum(row["complete_success"] for row in rows if row["split"]==split)/sum(row["split"]==split for row in rows) for split in SPLITS}})
    return baselines


def run_controls(seeds=SEEDS, updates=UPDATES):
    control, math_items, protocol = load_fixtures()
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Unique predeclared initialization seeds required")
    torch.set_num_threads(1)
    began = time.perf_counter()
    runs = [decoder_control(control,seed,updates=updates) for seed in seeds]
    for run in runs:
        run["positive_control_improved"] = (sum(run["final_train_complete_path_probabilities"]) >
                                           sum(run["initial_train_complete_path_probabilities"]))
        run["all_training_greedy_correct"] = run["final"]["summaries"]["train/greedy"]["complete_path_success"] == 1.
        run["task_result"] = "all_train_rows_solved" if run["all_training_greedy_correct"] else "partial_training_failure_retained"
    return {"mode":"bounded CPU sampled sequence control","seeds":list(seeds),"updates":updates,
            "all_declared_seeds_retained":True,"decoder_runs":runs,"lookup":lookup_control(control),
            "control_fixture":validate_items(control),"math_fixture":validate_items(math_items),
            "constant_answer_baselines":constant_baselines(control),
            "math_panel_audit":math_panel_audit(math_items),"pretrained_protocol":protocol,
            "oracle_rows":[{"item_id":i["id"],"answer":answer_text(oracle_answer(i)),
                            "response_tokens":[NUMBER_START+int(oracle_answer(i)),EOS]} for i in control],
            "seconds":time.perf_counter()-began,
            "limits":["Constructive conditional-grammar task, not natural-language reasoning",
                      "English panel is oracle/parser validation, not decoder benchmark output",
                      "No pretrained/base/instruct/thinking campaign executed",
                      "Final answers do not establish rationale faithfulness",
                      "Original negative G4/G8 GRPO report preserved"]}


def main(argv=None):
    """Record a new CPU report; optionally attach a passed fresh notebook run."""
    import argparse
    import base64
    from datetime import datetime, timezone
    import platform
    import resource
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--notebook-manifest",type=Path)
    parser.add_argument("--export-previews",action="store_true")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Unused output path required; preserve historical evidence")
    root = Path(__file__).resolve().parents[2]
    files = ["src/dongxi_llms/reasoning_controls.py","tests/test_reasoning_controls.py",
        "fixtures/reasoning-controls/control_items.json","fixtures/reasoning-controls/math_items.json",
        "fixtures/reasoning-controls/protocol.json","fixtures/reasoning-controls/README.md",
        "experiments/specs/2026-10-04-reasoning-controls.md",
        "notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb","notebooks/day-23/README.md",
        "book/chapters/13-group-relative-policy-optimization.md",
        "book/solutions/13-group-relative-policy-optimization.md","book/labs/13-group-relative-policy-optimization.md",
        "book/chapters/15-distill-evaluate-and-defend.md","src/dongxi_llms/grpo_lab.py",
        "src/dongxi_llms/decoder_lab.py","src/dongxi_llms/reasoning_evaluation.py"]
    before = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    notebook = None
    previews = []
    if args.notebook_manifest is not None:
        manifest = json.loads(args.notebook_manifest.read_text())
        notebook = next((r for r in manifest["notebooks"] if r["path"]==files[7]),None)
        if notebook is None or manifest["failures"] or notebook["sha256"] != before[files[7]]:
            parser.error("Need a passed fresh-kernel notebook manifest matching current source")
    if args.export_previews:
        if notebook is None:
            parser.error("Preview export requires fresh notebook evidence")
        import nbformat
        executed = nbformat.read(notebook["output"],as_version=4)
        images = [output.data["image/png"] for cell in executed.cells if cell.cell_type=="code"
                  for output in cell.outputs if output.output_type in ("display_data","execute_result")
                  and "image/png" in output.data]
        if len(images) != 5:
            parser.error("Expected five original explanatory figures")
        for index,encoded in enumerate(images,1):
            path=root/f"notebooks/figures/chapter-13/day-23-03_reasoning_tasks_and_positive_controls-{index:02d}.png"
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(base64.b64decode(encoded))
            previews.append({"path":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    results = run_controls()
    import os
    checks = []
    env = dict(os.environ,PYTHONPATH=str(root/"src"),CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    for command in ([sys.executable,"-m","unittest","discover","-s","tests","-p","test_reasoning_controls.py","-v"],
                    [sys.executable,"scripts/check_book_math.py"],
                    [sys.executable,"-m","unittest","discover","-s","tests","-v"]):
        completed=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=60)
        checks.append({"command":command,"exit_code":completed.returncode,
                       "stdout":completed.stdout,"stderr":completed.stderr})
    after = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    if after != before:
        raise RuntimeError("Sources changed during this measurement; preserve a coherent rerun instead")
    report = {"package":"DXI-04","status":"passed" if all(r["positive_control_improved"] for r in results["decoder_runs"]) and all(c["exit_code"]==0 for c in checks) else "control_or_check_failed",
        "date_utc":datetime.now(timezone.utc).isoformat(),
        "command":list(sys.orig_argv),"python_argv":list(sys.argv),
        "reproduction_command":[sys.executable,"-m","dongxi_llms.reasoning_controls",*sys.argv[1:]],
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "dirty_state":subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).splitlines(),
        "python":platform.python_version(),"interpreter":sys.executable,"platform":platform.platform(),
        "machine":platform.machine(),"torch":torch.__version__,"device":"cpu","dtype":"float32",
        "cpu_threads":torch.get_num_threads(),"maximum_rss_report_process_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_sha256":before,"notebook_verification":notebook,"results":results,"checks":checks,"previews":previews,
        "notebook_manifest_sha256":hashlib.sha256(args.notebook_manifest.read_bytes()).hexdigest() if args.notebook_manifest else None,
        "notebook_kernel":manifest["kernel"] if notebook is not None else None,
        "acceptance_boundary":"passed instrument means all seeded mean training-path probabilities improve; it does NOT mean every known-solvable item is solved",
        "negative_report_sha256":hashlib.sha256((root/"experiments/reports/2026-10-04-grpo-diagnostics-distillation.json").read_bytes()).hexdigest(),
        "resource_boundary":"bounded tiny CPU models; no pretrained/GPU/acquisition; elapsed time is whole control execution; Linux report-process RSS is not a system/GPU peak"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":report["status"],"seconds":results["seconds"],
                      "seeds":results["seeds"]}))


if __name__ == "__main__":
    main()

"""Original bounded autoregressive actor, neural critic and frozen text-reward loop.

Finite syntax support is explicit. Collector caps are not EOS. No work on import,
no pretrained model or human-feedback claim, and no quality label in RL targets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn

from .run_identity import canonical_hash, file_digest
from .text_reward_lab import RewardConfig, TinyTextReward, last_valid_indices

PAD, BOS, EOS, RED, BLUE, PLAIN, FANCY = 0, 1, 4, 12, 13, 18, 19
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def load_protocol(root=DEFAULT_ROOT):
    document = json.loads((Path(root)/"fixtures/critic-policy/protocol.json").read_text())
    return validate_protocol(document)


def validate_protocol(document):
    items = document["items"]
    if document["schema"] != "dongxi-critic-policy-v1" or len(items) != 10:
        raise ValueError("Frozen ten-source protocol required")
    if (document["collector_cap"]!=3 or document["environment_horizon"]!=4
            or document["gamma"]!=1 or document["noise_amplitude"]!=2):
        raise ValueError("Declared cap3/horizon4/gamma1/noise2 environment required; no silently ignored control")
    vocabulary=document["actor_vocabulary"]
    if (not isinstance(vocabulary,list) or len(vocabulary)!=20
            or any(not isinstance(word,str) or not word.strip() for word in vocabulary)
            or len(set(vocabulary))!=len(vocabulary)):
        raise ValueError("Unique nonempty actor vocabulary tokens required")
    if any(vocabulary[i]!=word for i,word in
           ((PAD,"<pad>"),(BOS,"<bos>"),(EOS,"<eos>"),(RED,"red"),(BLUE,"blue"),(PLAIN,"plain"),(FANCY,"fancy"))):
        raise ValueError("Actor grammar token meanings changed")
    ids, groups, encodings = set(), set(), set()
    for item in items:
        if any(not isinstance(item[key],str) or not item[key].strip() for key in ("id","source_group_id","prompt")):
            raise ValueError("Nonempty original input/source identities required")
        try: encoded=tuple(prompt_ids(item,document))
        except KeyError as error:raise ValueError("Prompt outside declared actor vocabulary") from error
        if item["id"] in ids or item["source_group_id"] in groups or encoded in encodings:
            raise ValueError("ID/source/input collision")
        if item["color"] not in ("red", "blue") or item["split"] not in ("train", "calibration", "test", "control"):
            raise ValueError("Unsupported authored task")
        ids.add(item["id"]); groups.add(item["source_group_id"]); encodings.add(encoded)
    return document


def allowed(position):
    if position == 0: return (RED, BLUE)
    if position in (1, 2): return (EOS, PLAIN, FANCY)
    if position == 3: return (EOS,)
    raise ValueError("Four-action environment only")


def terminal_paths(prefix=()):
    if prefix and prefix[-1] == EOS: return [prefix]
    if len(prefix) >= 4: raise ValueError("Nonterminal path exceeds environment horizon")
    return [p for action in allowed(len(prefix)) for p in terminal_paths((*prefix, action))]


def state_prefixes():
    return sorted({p[:t] for p in terminal_paths() for t in range(len(p))}, key=lambda p:(len(p),p))


def render(tokens):
    labels = {RED:"red", BLUE:"blue", PLAIN:"plain", FANCY:"fancy"}
    active = []
    for token in tokens:
        if token in (EOS, PAD): break
        if token not in labels: raise ValueError("Not a response grammar token")
        active.append(labels[token])
    return " ".join(active)


def independent_quality(item, tokens):
    tokens = tuple(tokens)
    return bool(tokens and tokens[-1] == EOS and len(tokens) <= 3
                and tokens[0] == (RED if item["color"] == "red" else BLUE)
                and all(t in (PLAIN,FANCY) for t in tokens[1:-1]))


def prompt_ids(item, protocol):
    mapping = {word:i for i,word in enumerate(protocol["actor_vocabulary"])}
    # Actual words, not item/source identity or the independent quality answer.
    return [BOS, *[mapping[word] for word in item["prompt"].split()]]


def pad_states(items, prefixes, protocol):
    if len(items) != len(prefixes) or not items: raise ValueError("Aligned nonempty states required")
    rows = [prompt_ids(item,protocol)+list(prefix) for item,prefix in zip(items,prefixes)]
    width = max(map(len,rows))
    ids = torch.zeros(len(rows),width,dtype=torch.long)
    mask = torch.zeros_like(ids,dtype=torch.bool)
    for index,row in enumerate(rows):
        ids[index,:len(row)] = torch.tensor(row); mask[index,:len(row)] = True
    return ids,mask


class SequenceActor(TinyTextReward):
    """A genuine causal token decoder; the scalar head becomes a vocabulary head."""
    def __init__(self, vocab):
        super().__init__(RewardConfig(vocab,width=16,heads=2,max_positions=32))
        self.head = nn.Linear(16,vocab).double()

    def forward(self, ids, mask):
        hidden = self.hidden(ids,mask)
        endpoint = last_valid_indices(mask)
        return self.head(hidden[torch.arange(len(ids)),endpoint])


def make_models(seed, protocol):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = SequenceActor(len(protocol["actor_vocabulary"])).eval()
        torch.manual_seed(seed+10000)
        critic = TinyTextReward(RewardConfig(len(protocol["actor_vocabulary"]),width=16,heads=2,max_positions=32)).eval()
    return actor,critic


def state_hash(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def gae_targets(rewards, values, mask, terminated, bootstrap, *, gamma=1., lam=.95):
    """Return detached TD residuals, GAE and lambda-return critic targets [B,T].

    `values` are BEFORE each action. Bootstrap is AFTER a nonterminal last action.
    EOS sets continuation to zero; padding neither continues nor earns reward.
    """
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != mask.shape or mask.shape != terminated.shape:
        raise ValueError("Aligned [B,T] rewards/values/masks required")
    if mask.dtype != torch.bool or terminated.dtype != torch.bool or bootstrap.shape != rewards.shape[:1]:
        raise ValueError("Boolean masks and one bootstrap per response required")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1 or not mask.any(1).all():
        raise ValueError("Nonempty paths and bounded gamma/lambda required")
    if any(not torch.isfinite(x).all() for x in (rewards,values,bootstrap)):
        raise ValueError("Finite rewards/values/bootstraps required")
    if bool((mask[:,1:] & ~mask[:,:-1]).any()) or bool((terminated & ~mask).any()):
        raise ValueError("Valid positions must be contiguous and terminal positions valid")
    if bool((terminated[:,:-1] & mask[:,1:]).any()):
        raise ValueError("No valid action after EOS")
    residual = torch.zeros_like(values)
    advantage = torch.zeros_like(values)
    tail = torch.zeros_like(bootstrap)
    for t in reversed(range(rewards.shape[1])):
        following = mask[:,t+1] if t+1 < rewards.shape[1] else torch.zeros_like(mask[:,t])
        next_value = torch.where(following,values[:,t+1],bootstrap) if t+1 < rewards.shape[1] else bootstrap
        alive = (~terminated[:,t]).to(values.dtype)
        delta = rewards[:,t] + gamma*alive*next_value - values[:,t]
        tail = (delta + gamma*lam*alive*following.to(values.dtype)*tail)*mask[:,t]
        residual[:,t] = delta*mask[:,t]
        advantage[:,t] = tail
    return {"td":residual,"advantages":advantage,"targets":(advantage+values)*mask}


def actor_loss(logp, old_logp, advantage, mask, kl=None, *, beta=.02, epsilon=.2):
    if not (logp.shape == old_logp.shape == advantage.shape == mask.shape):
        raise ValueError("Aligned policy tensors required")
    ratio = (logp-old_logp.detach()).exp()
    a = advantage.detach()
    surrogate = torch.minimum(ratio*a,ratio.clamp(1-epsilon,1+epsilon)*a)
    loss = -(surrogate*mask).sum()/len(logp)
    if kl is not None: loss = loss+beta*(kl*mask).sum()/mask.sum()
    return loss


def critic_loss(values, targets, mask):
    return ((values-targets.detach()).square()*mask).sum()/mask.sum()


def fit_rewards(root, export_directory, protocol):
    from .char_reward_lab import (ASCIICharacterVocabulary, encoding_audit, fit_char_model,
                                  export_char_reward, load_char_reward, char_text_batch)
    export_directory = Path(export_directory)
    records,bundles = {},{}
    vocabulary = ASCIICharacterVocabulary()
    for arm in ("balanced","confounded"):
        path = Path(root)/f"fixtures/critic-policy/preferences-{arm}.json"
        obj = json.loads(path.read_text()); audit = encoding_audit(obj,vocabulary)
        if not audit["passed"]: raise ValueError(audit["issues"])
        model,fit = fit_char_model(obj,vocabulary,seed=protocol["reward_seed"],steps=protocol["reward_fit_updates"])
        export = export_char_reward(model,vocabulary,export_directory/f"frozen-{arm}.json",fixture_path=path,
            source_groups=audit["source_splits"],seed=protocol["reward_seed"],
            protocol_path=Path(root)/"experiments/specs/2026-10-04-critic-policy.md")
        frozen = load_char_reward(export["path"],expected_interface=vocabulary.interface(),expected_file_sha256=export["file_sha256"])
        calibration = [(i["prompt"],f"{color} {style}") for i in protocol["items"] if i["split"]=="train"
                       for color in ("red","blue") for style in ("plain","fancy")]
        train_scores = frozen.score_many(calibration)
        center,scale = float(train_scores.mean()),max(float(train_scores.std(correction=0)),.1)
        b = char_text_batch(vocabulary,calibration)
        with torch.no_grad(): reload_exact = torch.equal(model(b["ids"],b["mask"]),train_scores)
        prefixes = sorted({p[:t] for p in terminal_paths() for t in range(1,len(p)+1)})
        inputs = [(i["prompt"],render(p)) for i in protocol["items"] for p in prefixes]
        scores = frozen.score_many(inputs).tolist()
        table = {(i["id"],p):{"raw":score,"proxy":math.tanh((score-center)/scale)}
                 for (i,p),score in zip(((i,p) for i in protocol["items"] for p in prefixes),scores)}
        rows = [{"item_id":i["id"],"split":i["split"],"tokens":list(p),"text":render(p),
                 **table[(i["id"],p)],"quality":independent_quality(i,p)} for i in protocol["items"] for p in prefixes]
        bundles[arm] = {"frozen":frozen,"table":table,"center":center,"scale":scale}
        records[arm] = {"fit":fit,"encoding_audit":audit,"export":export,"reload_exact":reload_exact,
            "state_sha256":state_hash(frozen.model),"center":center,"scale":scale,
            "scaling_train_inputs":calibration,"scaling_train_scores":train_scores.tolist(),"actual_reward_rows":rows,
            "scoring_response_count":len(inputs)+len(calibration),
            "source_boundary":"No quality label/source ID supplied to model; train-only scale; all finite text scores from saved model"}
    return bundles,records


@torch.no_grad()
def oracle_tree(actor, items, protocol, bundle):
    prefixes = state_prefixes()
    owners = [i for i in items for p in prefixes]
    states = [p for i in items for p in prefixes]
    ids,mask = pad_states(owners,states,protocol)
    logits = actor(ids,mask)
    probabilities = {(i["id"],p):logits[k,list(allowed(len(p)))].softmax(-1).tolist()
                     for k,(i,p) in enumerate(zip(owners,states))}
    values = {}
    for item in items:
        for prefix in reversed(prefixes):
            total = 0.
            for action,p in zip(allowed(len(prefix)),probabilities[(item["id"],prefix)]):
                child = (*prefix,action)
                total += p*(bundle["table"][(item["id"],child)]["proxy"] if action==EOS else values[(item["id"],child)])
            values[(item["id"],prefix)] = total
    return values,probabilities


def exact_diagnostics(items,probabilities,bundle,cap):
    rows=[]
    for item in items:
        reward=quality=termination=0.
        for path in terminal_paths():
            mass=1.
            for t,action in enumerate(path):mass*=probabilities[(item["id"],path[:t])][allowed(t).index(action)]
            delivered=path[:cap]
            reward+=mass*bundle["table"][(item["id"],path)]["proxy"]
            quality+=mass*independent_quality(item,delivered)
            termination+=mass*(delivered[-1]==EOS)
        rows.append({"item_id":item["id"],"split":item["split"],"expected_terminal_proxy":reward,
                     "expected_delivered_quality":quality,"termination_probability":termination})
    return rows


@torch.no_grad()
def rollout(actor, items, protocol, *, generator, cap=3, greedy=False):
    if cap not in (3,4) or not items: raise ValueError("Declared collector cap3 or diagnostic cap4 required")
    responses = torch.zeros(len(items),cap,dtype=torch.long)
    mask = torch.zeros_like(responses,dtype=torch.bool)
    logp = torch.zeros(len(items),cap,dtype=torch.float64)
    prefixes = [() for _ in items]; finished = torch.zeros(len(items),dtype=torch.bool)
    forward_positions = 0
    for t in range(cap):
        ids,valid = pad_states(items,prefixes,protocol)
        forward_positions += int(valid.sum())
        distribution = actor(ids,valid)[:,list(allowed(t))].log_softmax(-1)
        pick = distribution.argmax(-1) if greedy else torch.multinomial(distribution.exp(),1,generator=generator).squeeze(-1)
        token = torch.tensor(allowed(t))[pick]
        responses[:,t] = torch.where(finished,PAD,token); mask[:,t] = ~finished
        logp[:,t] = distribution.gather(-1,pick[:,None]).squeeze(-1)*mask[:,t]
        prefixes = [p if finished[k] else (*p,int(token[k])) for k,p in enumerate(prefixes)]
        finished |= token==EOS
    return {"tokens":responses,"mask":mask,"old_logp":logp,"forward_positions":forward_positions}


def path_statistics(actor,reference,items,responses,mask,protocol):
    logps,kls = [],[]
    for t in range(responses.shape[1]):
        prefixes = [tuple(row[:t].tolist()) for row in responses]
        # Post-stop states are computed but masked; PAD is not a sampled action.
        prefixes = [tuple(x for x in p if x!=PAD) for p in prefixes]
        ids,valid = pad_states(items,prefixes,protocol)
        support = allowed(t)
        current = actor(ids,valid)[:,list(support)].log_softmax(-1)
        with torch.no_grad(): ref = reference(ids,valid)[:,list(support)].log_softmax(-1)
        index = torch.tensor([support.index(int(action)) if ok else 0 for action,ok in zip(responses[:,t],mask[:,t])])
        logps.append(current.gather(-1,index[:,None]).squeeze(-1))
        kls.append((current.exp()*(current-ref)).sum(-1))
    return torch.stack(logps,-1),torch.stack(kls,-1)


def neural_values(critic,items,responses,protocol):
    values = []
    for t in range(responses.shape[1]):
        prefixes = [tuple(x for x in row[:t].tolist() if x!=PAD) for row in responses]
        ids,valid = pad_states(items,prefixes,protocol); values.append(critic(ids,valid))
    return torch.stack(values,-1)


def state_noise(seed,item,prefix):
    digest = hashlib.sha256(f"{seed}:{item['id']}:{prefix}".encode()).digest()
    return 2.*(2.*int.from_bytes(digest[:8],"big")/(2**64-1)-1.)


@torch.no_grad()
def evaluate(actor,items,protocol,bundle,seed,*,cap=3):
    rows = []; batches=[]
    began=time.perf_counter()
    for item in items:
        for mode,count in (("greedy",1),("sampled",protocol["sampled_evaluation_paths_per_item"])):
            generator = torch.Generator().manual_seed(int(canonical_hash({"seed":seed,"item":item["id"],"mode":mode})[:15],16))
            started=time.perf_counter()
            batch=rollout(actor,[item]*count,protocol,generator=generator,cap=cap,greedy=mode=="greedy")
            batches.append({"item_id":item["id"],"mode":mode,"responses":count,
                "attempted_action_positions":count*cap,"valid_generated_tokens":int(batch["mask"].sum()),
                "full_prefix_forward_positions":batch["forward_positions"],"model_forward_calls":cap,
                "wall_seconds":time.perf_counter()-started,
                "cost_boundary":"Actual full-prefix batch, including post-stop dummy rows; not per-sequence latency or billed API units"})
            for k in range(count):
                tokens=tuple(batch["tokens"][k,batch["mask"][k]].tolist()); ended=tokens[-1]==EOS
                rows.append({"item_id":item["id"],"source_group_id":item["source_group_id"],"split":item["split"],
                    "mode":mode,"sample":k,"tokens":list(tokens),"mask":batch["mask"][k].tolist(),"text":render(tokens),
                    "stop":"eos" if ended else "collector-cap","truncated":not ended,"quality":independent_quality(item,tokens),
                    **bundle["table"][(item["id"],tokens)],"generated_tokens":len(tokens),
                    "prompt_tokens":len(prompt_ids(item,protocol)),"seed":generator.initial_seed()})
    summaries={}
    for split in ("train","calibration","test","control"):
        for mode in ("greedy","sampled"):
            selected=[r for r in rows if r["split"]==split and r["mode"]==mode]
            summaries[f"{split}/{mode}"]={"n":len(selected),"quality":sum(r["quality"] for r in selected)/len(selected),
                "proxy":sum(r["proxy"] for r in selected)/len(selected),"termination":sum(not r["truncated"] for r in selected)/len(selected),
                "generated_tokens":sum(r["generated_tokens"] for r in selected)}
    _,probabilities=oracle_tree(actor,items,protocol,bundle)
    return {"rows":rows,"summaries":summaries,"batches":batches,"cap":cap,
            "exact_finite_diagnostics":exact_diagnostics(items,probabilities,bundle,cap),
            "score_boundary":"Row proxy also scores unfinished prefixes; training only rewards emitted EOS. Do not interpret prefix proxy as an observed terminal reward.",
            "seconds":time.perf_counter()-began}


def run_arm(protocol,bundle,seed,kind,*,updates=None):
    updates=protocol["rollout_updates"] if updates is None else updates
    if kind not in ("oracle","learned","noisy") or not 1<=updates<=protocol["rollout_updates"]:
        raise ValueError("Declared critic kind and bounded updates required")
    actor,critic=make_models(seed,protocol); reference=deepcopy(actor).requires_grad_(False)
    initial_hash=state_hash(actor); critic_initial_hash=state_hash(critic); reward_hash=state_hash(bundle["frozen"].model)
    train=[i for i in protocol["items"] if i["split"]=="train"]
    owners=[i for i in train for _ in range(protocol["paths_per_prompt"])]
    optimizer=torch.optim.AdamW(actor.parameters(),lr=protocol["actor_learning_rate"],weight_decay=0)
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=protocol["critic_learning_rate"],weight_decay=0)
    generator=torch.Generator().manual_seed(seed+1)
    history=[];ledger=[];panels=[{"update":0,**evaluate(actor,protocol["items"],protocol,bundle,seed)}]
    began=time.perf_counter()
    for update in range(updates):
        sampled=rollout(actor,owners,protocol,generator=generator)
        tokens,mask=sampled["tokens"],sampled["mask"]
        exact,_=oracle_tree(actor,train,protocol,bundle)
        oracle=torch.zeros_like(sampled["old_logp"]);bootstrap=torch.zeros(len(owners),dtype=torch.float64)
        reward=torch.zeros_like(oracle);terminated=(tokens==EOS)&mask
        for k,item in enumerate(owners):
            path=tuple(tokens[k,mask[k]].tolist())
            for t in range(len(path)):
                oracle[k,t]=exact[(item["id"],path[:t])]
                if kind=="noisy":oracle[k,t]+=state_noise(seed,item,path[:t])
            if path[-1]==EOS: reward[k,len(path)-1]=bundle["table"][(item["id"],path)]["proxy"]
            else:
                bootstrap[k]=exact[(item["id"],path)]
                if kind=="noisy":bootstrap[k]+=state_noise(seed,item,path)
        with torch.no_grad():
            learned=neural_values(critic,owners,tokens,protocol)
            if kind=="learned":
                values=learned
                prefixes=[tuple(row[valid].tolist()) for row,valid in zip(tokens,mask)]
                ids,valid=pad_states(owners,prefixes,protocol)
                bootstrap=torch.where(terminated.any(-1),0.,critic(ids,valid))
            else: values=oracle
        target=gae_targets(reward,values,mask,terminated,bootstrap,gamma=protocol["gamma"],lam=protocol["lambda"])
        current,kl=path_statistics(actor,reference,owners,tokens,mask,protocol)
        alignment=float(((current.detach()-sampled["old_logp"])*mask).abs().max())
        loss=actor_loss(current,sampled["old_logp"],target["advantages"],mask,kl,beta=protocol["kl_beta"])
        optimizer.zero_grad(set_to_none=True);loss.backward()
        actor_reach={name:float(p.grad.abs().sum()) for name,p in actor.named_parameters() if p.grad is not None} if update==0 else None
        actor_gradient=float(nn.utils.clip_grad_norm_(actor.parameters(),1))
        critic_actor_leak=any(p.grad is not None for p in critic.parameters()) if update==0 else None
        if not math.isfinite(float(loss.detach())) or not math.isfinite(actor_gradient):raise RuntimeError("Nonfinite actor update")
        optimizer.step()
        value_loss=None;critic_gradient=None
        if kind=="learned":
            prediction=neural_values(critic,owners,tokens,protocol)
            value_loss=critic_loss(prediction,target["targets"],mask)
            critic_optimizer.zero_grad(set_to_none=True);value_loss.backward()
            critic_reach={name:float(p.grad.abs().sum()) for name,p in critic.named_parameters() if p.grad is not None} if update==0 else None
            critic_gradient=float(nn.utils.clip_grad_norm_(critic.parameters(),1))
            if not math.isfinite(float(value_loss.detach())) or not math.isfinite(critic_gradient):raise RuntimeError("Nonfinite critic update")
            critic_optimizer.step()
        metric={"update":update+1,"actor_loss":float(loss.detach()),"critic_loss":float(value_loss.detach()) if value_loss is not None else None,
            "first_actor_gradient_reach":actor_reach,"first_critic_gradient_reach":critic_reach if kind=="learned" and update==0 else None,
            "actor_gradient":actor_gradient,"critic_gradient":critic_gradient,"alignment_max":alignment,
            "valid_response_tokens":int(mask.sum()),"full_prefix_forward_positions":sampled["forward_positions"],
            "truncated_paths":int((~terminated.any(-1)).sum()),"sampled_terminal_reward":float(reward.sum(-1).mean()),
            "critic_oracle_mse_on_valid_states":float(((learned-oracle).square()*mask).sum()/mask.sum()) if kind=="learned" else None,
            "actor_loss_reaches_critic_parameters":critic_actor_leak}
        history.append(metric)
        ledger.append({"update":update+1,"item_ids":[i["id"] for i in owners],"prompt_token_ids":[prompt_ids(i,protocol) for i in owners],
            "tokens":tokens.tolist(),"mask":mask.tolist(),
            "old_logp":sampled["old_logp"].tolist(),"rewards":reward.tolist(),"values":values.tolist(),"bootstrap":bootstrap.tolist(),
            "terminated":terminated.tolist(),**{k:v.tolist() for k,v in target.items()}})
        if update+1 in protocol["checkpoint_updates"] or update+1==updates:
            panels.append({"update":update+1,**evaluate(actor,protocol["items"],protocol,bundle,seed)})
    if state_hash(reference)!=initial_hash or state_hash(bundle["frozen"].model)!=reward_hash:
        raise RuntimeError("Frozen actor reference or reward changed")
    if any(p.requires_grad or p.grad is not None for p in bundle["frozen"].model.parameters()):
        raise RuntimeError("Reward acquired a gradient path")
    return {"seed":seed,"critic":kind,"actor_initial_sha256":initial_hash,"actor_final_sha256":state_hash(actor),
        "critic_initial_sha256":critic_initial_hash,"critic_final_sha256":state_hash(critic),"reference_sha256":state_hash(reference),"reward_state_sha256":reward_hash,
        "actor_parameters":sum(p.numel() for p in actor.parameters()),"critic_parameters":sum(p.numel() for p in critic.parameters()),
        "history":history,"training_ledger":ledger,"panels":panels,
        "frozen_control":evaluate(reference,protocol["items"],protocol,bundle,seed),
        "cap4_intervention":evaluate(actor,protocol["items"],protocol,bundle,seed,cap=4),
        "actor_updates":updates,"critic_updates":updates if kind=="learned" else 0,
        "sampled_paths":updates*len(owners),"generated_response_tokens":sum(h["valid_response_tokens"] for h in history),
        "full_prefix_generation_forward_positions":sum(h["full_prefix_forward_positions"] for h in history),
        "seconds":time.perf_counter()-began,"checkpoint_boundary":"In-memory states hashed; no actor/critic checkpoint export claimed"}


def run_reference(root=DEFAULT_ROOT,export_directory=None):
    if export_directory is None:raise ValueError("A new explicit frozen export directory is required")
    export_directory=Path(export_directory)
    if export_directory.exists():raise FileExistsError("Choose new outputs; preserve earlier reward states")
    torch.set_num_threads(1)
    protocol=load_protocol(root);bundles,fits=fit_rewards(root,export_directory,protocol)
    rows=[];failures=[]
    for arm,kind in protocol["arms"]:
        for seed in protocol["actor_seeds"]:
            try: rows.append({"reward_arm":arm,**run_arm(protocol,bundles[arm],seed,kind)})
            except Exception as error:failures.append({"reward_arm":arm,"critic":kind,"seed":seed,"error":repr(error)})
    return {"protocol":protocol,"reward_fits":fits,"runs":rows,"failures":failures,
            "status":"completed" if not failures else "failed","limits":["Finite conditional grammar, not pretrained/open-language PPO",
            "Authored preferences and independent programmatic quality, not human feedback","Cap continuation is bootstrapped, not generated EOS",
            "Separate critics; equal sampled-path budget is not equal optimizer work","No best-seed/checkpoint selection or quality-gradient path"]}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--exports",type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists() or args.exports.exists():parser.error("New report and export paths required")
    sources=["src/dongxi_llms/critic_policy_lab.py","src/dongxi_llms/char_reward_lab.py","src/dongxi_llms/text_reward_lab.py",
             "src/dongxi_llms/run_identity.py","tests/test_critic_policy_lab.py","fixtures/critic-policy/protocol.json",
             "fixtures/critic-policy/preferences-balanced.json","fixtures/critic-policy/preferences-confounded.json",
             "experiments/specs/2026-10-04-critic-policy.md"]
    before={p:file_digest(DEFAULT_ROOT/p) for p in sources}
    began=time.perf_counter();result=run_reference(DEFAULT_ROOT,args.exports)
    after={p:file_digest(DEFAULT_ROOT/p) for p in sources}
    if before!=after:result["failures"].append({"stage":"source-immutability","error":"Measured source changed during run"});result["status"]="failed"
    report={"schema_version":1,"package":"DXI-10","date_utc":datetime.now(timezone.utc).isoformat(),
        "command":list(sys.orig_argv),"python_argv":list(sys.argv),"source_sha256":before,
        "environment":{"executable":sys.executable,"prefix":sys.prefix,"python":platform.python_version(),
                       "platform":platform.platform(),"torch":torch.__version__,"device":"cpu","dtype":"float64","threads":torch.get_num_threads()},
        "results":result,"seconds":time.perf_counter()-began}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle:handle.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":result["status"],"runs":len(result["runs"]),"seconds":report["seconds"]}))
    if result["status"]!="completed":raise SystemExit(1)


if __name__=="__main__":main()

"""A bounded compatible post-training assignment, assembled from course kernels.

This is symbolic CPU evidence, not English parsing or pretrained alignment.
The full-vocabulary SFT and grammar-conditional RL distributions are explicitly
different. Historical native rewards are replayed, never replaced by new grades.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import torch

from . import reasoning_controls as rc
from .decoder_lab import DecoderConfig, TinyDecoder
from .grpo_lab import clipped_objective, exact_kl, group_advantages, verify_integer
from .reasoning_evaluation import grade_response
from .sft_lab import token_loss_sum

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "experiments/fixtures/cs336"
ARCHIVES = ("outputs/native-rlvr-g4-pilot-20261005-run-01/report.json",
            "outputs/native-rlvr-g8-pilot-20261005-run-01/report.json")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(item):
    """Authored structured arithmetic; independent of labels and model output."""
    a, b = item["a"], item["b"]
    if type(a) is not int or type(b) is not int or not (0 <= a <= 3 and 0 <= b <= 3):
        raise ValueError("Operands must be exact integers in0..3")
    if item["family"] == "parity":
        return (a+b) % 2
    if item["family"] == "equality":
        return int(a == b)
    raise ValueError("Unknown independently fixed task family")


def validate_fixture(items):
    seen, sources, problems = set(), {}, {}
    for item in items:
        if item["id"] in seen:
            raise ValueError("Duplicate item ID")
        seen.add(item["id"])
        if item["answer"] != oracle(item):
            raise ValueError("Authored answer disagrees with arithmetic oracle")
        problem = (item["family"], item["a"], item["b"])
        if sources.setdefault(item["source_group"], item["split"]) != item["split"]:
            raise ValueError("Underlying source crosses splits")
        if problems.setdefault(problem, item["split"]) != item["split"]:
            raise ValueError("Underlying problem crosses splits")
    for split in ("train", "development", "heldout-source", "heldout-family"):
        rows = [i for i in items if i["split"] == split]
        for answer in (0, 1):
            if len({i["source_group"] for i in rows if oracle(i) == answer}) < 2:
                raise ValueError("Each split needs multiple sources in both answer classes")
        if split != "heldout-family":
            for source in {i["source_group"] for i in rows}:
                templates = [i["template"] for i in rows if i["source_group"] == source]
                if sorted(templates) != ["alias", "direct"]:
                    raise ValueError("Paired direct/alias forms required on identical sources")
    if any(i["family"] != "parity" for i in items if i["split"] == "train"):
        raise ValueError("Withheld task family must never enter training")
    return {"items": len(items), "by_split": dict(Counter(i["split"] for i in items)),
            "unique_sources": dict(Counter(split for split in sources.values())),
            "items_sha256": identity(items),
            "family_boundary": "Equality is an unseen operation; operands may overlap parity, but task identities differ."}


def load_fixture(directory=FIXTURE):
    directory = Path(directory)
    items = json.loads((directory / "posttraining_items.json").read_text())
    protocol = json.loads((directory / "posttraining_protocol.json").read_text())
    validate_fixture(items)
    expected = {"PAD": rc.PAD, "BOS": rc.BOS, "EOS": rc.EOS, "EQUALITY": 3,
                "PARITY_ALIAS": 4, "PARITY_DIRECT": 5, "NUMBER_START": rc.NUMBER_START}
    interface = protocol["interface"]
    if (interface["vocab_size"] != rc.VOCAB or interface["tokens"] != expected or
            interface["first_sampling_support"] != list(rc.FIRST) or
            interface["second_sampling_support"] != list(rc.LATER) or interface["cap"] != 2):
        raise ValueError("Fixture does not match reused sampler/scorer interface")
    return items, protocol


def prompt_ids(items):
    rows = []
    for item in items:
        oracle(item)
        operation = 3 if item["family"] == "equality" else {"alias": 4, "direct": 5}[item["template"]]
        rows.append([rc.BOS, operation, rc.NUMBER_START+item["a"], rc.NUMBER_START+item["b"]])
    return torch.tensor(rows, dtype=torch.long)


def supervised_batch(items, responses=None):
    prompts = prompt_ids(items)
    if responses is None:
        responses = torch.tensor([[rc.NUMBER_START+oracle(i), rc.EOS] for i in items])
    if tuple(responses.shape) != (len(items), 2):
        raise ValueError("SFT needs two-token response paths")
    ids = torch.cat((prompts, responses), -1)
    labels = ids.clone()
    labels[:, :prompts.shape[1]] = -100
    return {"input_ids": ids, "labels": labels}


def text_from_ids(tokens):
    names = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<equality>",
             4: "<parity-alias>", 5: "<parity-direct>"}
    return " ".join(str(t-rc.NUMBER_START) if rc.NUMBER_START <= t < rc.VOCAB
                    else names[t] for t in tokens if t != rc.EOS)


def load_checkpoint(path, protocol):
    """Reject foreign microscope weights instead of silently changing interfaces."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["interface"] != protocol["interface"] or
            checkpoint["interface_sha256"] != identity(protocol["interface"]) or
            checkpoint["protocol_sha256"] != identity(protocol)):
        raise ValueError("Checkpoint interface/protocol does not match this assignment")
    model = TinyDecoder(DecoderConfig(**checkpoint["architecture"])).double().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if rc.state_hash(model) != checkpoint["state_sha256"]:
        raise ValueError("Checkpoint state hash mismatch")
    return model, checkpoint


def path_grade(tokens, mask, item):
    active = tokens[mask].tolist()
    answer = bool(active and active[0] == rc.NUMBER_START+oracle(item))
    terminated = bool(active and active[-1] == rc.EOS)
    complete = answer and active == [rc.NUMBER_START+oracle(item), rc.EOS]
    return {"answer_correct": answer, "natural_termination": terminated,
            "complete_path_success": complete, "strict_reward": float(complete),
            "token_ids": active, "raw_response": text_from_ids(active),
            "stop_reason": "eos" if terminated else "max_tokens",
            "truncated": not terminated}


@torch.no_grad()
def exact_success(model, items):
    responses = torch.tensor([[rc.NUMBER_START+oracle(i), rc.EOS] for i in items])
    return rc.path_log_probabilities(model, prompt_ids(items), responses).sum(-1).exp()


@torch.no_grad()
def evaluate(model, items, seed, samples=4, grammar=True, cap=2):
    repeated = [item for item in items for _ in range(samples+1)]
    prompts = prompt_ids(repeated)
    sampled, valid = rc.generate(model, prompts, generator=torch.Generator().manual_seed(seed),
                                 grammar=grammar, cap=cap)
    greedy, greedy_valid = rc.generate(model, prompt_ids(items), greedy=True, grammar=grammar, cap=cap)
    rows = []
    for index, (item, tokens, mask) in enumerate(zip(repeated, sampled, valid)):
        position = index % (samples+1)
        if position == 0:
            tokens, mask = greedy[index//(samples+1)], greedy_valid[index//(samples+1)]
        rows.append(dict(path_grade(tokens, mask, item), item_id=item["id"],
                         source_group=item["source_group"], split=item["split"],
                         template=item["template"], family=item["family"],
                         decoding="greedy" if position == 0 else "sampled",
                         sample_index=position, prompt_ids=prompts[index].tolist()))
    summaries = {}
    for split, template, decoding in sorted({(r["split"], r["template"], r["decoding"]) for r in rows}):
        selected = [r for r in rows if (r["split"], r["template"], r["decoding"]) == (split, template, decoding)]
        summaries[f"{split}/{template}/{decoding}"] = {"n": len(selected), **{
            key: sum(r[key] for r in selected)/len(selected) for key in
            ("answer_correct", "natural_termination", "complete_path_success", "truncated")},
            "generated_tokens": sum(len(r["token_ids"]) for r in selected)}
    return {"grammar": grammar, "cap": cap, "summaries": summaries, "rows": rows,
            "exact_complete_path_probability": dict(zip((i["id"] for i in items),
                                                        exact_success(model, items).tolist())) if grammar and cap == 2 else None,
            "cost": {"attempted_responses": len(repeated)+len(items),
                     "sampling_batch_includes_replaced_greedy_slots": len(items),
                     "physical_autoregressive_positions": cap*(len(repeated)+len(items)),
                     "physical_generation_prefix_tokens": sum(4+position for position in range(cap))*(len(repeated)+len(items)),
                     "exact_diagnostic_scoring_input_tokens": 5*len(items) if grammar and cap == 2 else 0}}


def baselines(items):
    train = [i for i in items if i["split"] == "train"]
    counts = Counter(oracle(i) for i in train)
    majority = min(counts, key=lambda a: (-counts[a], a))
    return {split: {"sources": len({i["source_group"] for i in items if i["split"] == split}),
                    "constant0": sum(oracle(i) == 0 for i in items if i["split"] == split)/sum(i["split"] == split for i in items),
                    "constant1": sum(oracle(i) == 1 for i in items if i["split"] == split)/sum(i["split"] == split for i in items),
                    "train_majority_answer": majority,
                    "majority_accuracy": sum(oracle(i) == majority for i in items if i["split"] == split)/sum(i["split"] == split for i in items),
                    "boundary": "These baselines always emit answer plus EOS. They are arithmetic baselines, not trained policies."}
            for split in sorted({i["split"] for i in items})}


def _deadline(started, cap):
    if time.perf_counter()-started > cap:
        raise TimeoutError("Predeclared CPU cap reached; no automatic extension")


def _gradient(loss, model, retain_graph=True):
    parts = torch.autograd.grad(loss, list(model.parameters()), retain_graph=retain_graph, allow_unused=True)
    flat = torch.cat([p.reshape(-1) for p in parts if p is not None])
    return flat


def _sft(model, rows, updates, protocol, started, cap):
    batch = supervised_batch(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["sft_lr"], weight_decay=0.)
    history = []
    for update in range(updates):
        _deadline(started, cap)
        total, count = token_loss_sum(model(batch["input_ids"]), batch["labels"])
        loss = total/count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Nonfinite SFT state")
        optimizer.step()
        history.append({"update": update+1, "loss": float(loss.detach()),
                        "target_tokens": count, "gradient_norm": float(norm)})
    return {"history": history, "unique_sources": len({i["source_group"] for i in rows}),
            "rows_per_update": len(rows), "target_tokens": sum(h["target_tokens"] for h in history),
            "source_ids_with_repetitions": [i["source_group"] for i in rows]}


def signal_audit(model, items, seed, group_size):
    repeated = [i for i in items for _ in range(group_size)]
    prompts = prompt_ids(repeated)
    tokens, masks = rc.generate(model, prompts, generator=torch.Generator().manual_seed(seed))
    rows = [dict(path_grade(t, m, i), item_id=i["id"], source_group=i["source_group"],
                 split=i["split"], sample_index=j % group_size)
            for j, (i, t, m) in enumerate(zip(repeated, tokens, masks))]
    rewards = torch.tensor([r["strict_reward"] for r in rows], dtype=torch.float64).reshape(len(items), group_size)
    counts = rewards.sum(-1)
    return {"rows": rows, "sampled_successes": int(rewards.sum()),
            "sampled_failures": int(rewards.numel()-rewards.sum()),
            "groups": len(items), "mixed_groups": int(((counts > 0) & (counts < group_size)).sum()),
            "all_wrong_groups": int((counts == 0).sum()), "all_right_groups": int((counts == group_size).sum()),
            "group_success_counts": counts.tolist(), "physical_generated_positions": int(masks.sum()),
            "physical_generation_prefix_tokens": 9*len(repeated),
            "boundary": "This frozen-parent collection uses train/development only; it is not an evaluation-driven recipe choice."}


def _sibling(model, parent, train, updates, protocol, seed, arm, started, cap):
    parent_hash = rc.state_hash(parent)
    group_size = protocol["group_size"]
    repeated = [i for i in train for _ in range(group_size)]
    if not repeated:
        return {"status": "no-eligible-training-prompts", "history": [], "rollouts": []}
    prompts = prompt_ids(repeated)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["sibling_lr"], weight_decay=0.)
    generator = torch.Generator().manual_seed(seed+protocol["sibling_initial_rng_offset"])
    history, rollouts = [], []
    for update in range(updates):
        _deadline(started, cap)
        behavior_hash = rc.state_hash(model)
        tokens, mask = rc.generate(model, prompts, generator=generator)
        grades = [path_grade(t, m, i) for t, m, i in zip(tokens, mask, repeated)]
        rewards = torch.tensor([g["strict_reward"] for g in grades], dtype=torch.float64)
        groups = rewards.reshape(len(train), group_size)
        advantages = group_advantages(groups, normalize=arm != "grpo-center").flatten()
        with torch.no_grad():
            old_logp = rc.path_log_probabilities(model, prompts, tokens)
            ref_first, ref_later = rc.path_logits(parent, prompts, tokens)
        logp = rc.path_log_probabilities(model, prompts, tokens)
        first, later = rc.path_logits(model, prompts, tokens)
        kl = torch.stack((exact_kl(first, ref_first), exact_kl(later, ref_later)), -1)
        kl_loss = protocol["beta"]*((kl*mask).sum(-1)/mask.sum(-1)).mean()
        task_loss = clipped_objective(logp, old_logp, advantages, mask, epsilon=protocol["epsilon"])
        task_grad = _gradient(task_loss, model)
        kl_grad = _gradient(kl_loss, model)
        alignment = float((logp.detach()-old_logp).abs().max())
        if alignment > 1e-12:
            raise AssertionError("Fresh behavior/current conditional likelihood mismatch")
        selected = rewards.bool()
        accepted_tokens = int(selected.sum())*2
        if arm == "accepted-sft":
            if accepted_tokens:
                batch = supervised_batch([i for i, keep in zip(repeated, selected) if keep], tokens[selected])
                summed, count = token_loss_sum(model(batch["input_ids"]), batch["labels"])
                loss = summed/count
            else:
                loss = None
        else:
            loss = task_loss+kl_loss
        optimizer.zero_grad(set_to_none=True)
        gradient = 0.
        if loss is not None:
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"]))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
                raise RuntimeError("Nonfinite sibling objective/gradient")
            optimizer.step()
        counts = groups.sum(-1)
        mixed = (counts > 0) & (counts < group_size)
        first_entropy = -(first.log_softmax(-1)*first.softmax(-1)).sum(-1).mean()
        later_entropy = -(later.log_softmax(-1)*later.softmax(-1)).sum(-1).mean()
        history.append({"update": update+1, "behavior_version": update, "policy_version": update+1,
            "behavior_state_sha256": behavior_hash, "policy_state_sha256_after": rc.state_hash(model),
            "attempted_responses": len(repeated), "response_tokens": int(mask.sum()),
            "physical_generation_positions": int(mask.sum()), "accepted_supervised_tokens": accepted_tokens if arm == "accepted-sft" else 0,
            "physical_generation_prefix_tokens": 9*len(repeated),
            "physical_objective_scoring_input_tokens": 4*5*len(repeated),
            "accepted_sft_forward_input_tokens": 6*int(selected.sum()) if arm == "accepted-sft" else 0,
            "sampled_reward": float(rewards.mean()), "completion_rate": sum(g["natural_termination"] for g in grades)/len(grades),
            "mixed_group_fraction": float(mixed.double().mean()), "all_wrong_groups": int((counts == 0).sum()),
            "all_right_groups": int((counts == group_size).sum()), "task_advantage_norm": float(advantages.norm()),
            "task_reward_gradient_norm": float(task_grad.norm()), "weighted_kl_gradient_norm": float(kl_grad.norm()),
            "task_kl_gradient_dot": float(task_grad.dot(kl_grad)),
            "combined_rl_gradient_norm_before_clip": float((task_grad+kl_grad).norm()),
            "actual_objective": "accepted-token-mean full-vocabulary CE" if arm == "accepted-sft" else arm,
            "actual_objective_gradient_norm_before_clip": gradient, "optimizer_step": loss is not None,
            "loss": float(loss.detach()) if loss is not None else None,
            "exact_kl_nats_on_sampled_prefixes": float(kl.detach().mean()),
            "conditional_first_entropy_nats": float(first_entropy.detach()),
            "conditional_later_entropy_nats": float(later_entropy.detach()),
            "initial_log_ratio_max_abs": alignment,
            "mixed_filter_counterfactual": {"retained_groups": int(mixed.sum()), "discarded_groups": int((~mixed).sum()),
                "attempted_groups": len(train), "attempted_response_tokens": int(mask.sum()),
                "population_source_counts": dict(Counter(i["source_group"] for i, keep in zip(train, mixed) if keep)),
                "applied_to_update": False}})
        rollouts.append({"update": update+1, "behavior_state_sha256": behavior_hash,
            "item_ids": [i["id"] for i in repeated], "prompt_ids": prompts.tolist(),
            "response_ids": tokens.tolist(), "response_mask": mask.tolist(), "grades": grades,
            "old_conditional_logp": old_logp.tolist(), "rewards": rewards.tolist(),
            "advantages": advantages.tolist(), "all_attempts_retained": True})
        if rc.state_hash(parent) != parent_hash:
            raise AssertionError("Frozen SFT parent changed")
    return {"status": "completed", "history": history, "rollouts": rollouts,
            "attempted_response_tokens": sum(h["response_tokens"] for h in history),
            "accepted_supervised_tokens": sum(h["accepted_supervised_tokens"] for h in history),
            "reference_state_sha256_before_and_after": parent_hash,
            "diagnostic_gradient_boundary": "Accepted-SFT records a counterfactual GRPO task/KL gradient; its actual update uses CE alone."}


def paired_source_deltas(parent_evaluation, child_evaluation):
    """Pair source-level complete scores; both template forms share each source."""
    def means(evaluation):
        groups = defaultdict(list)
        for row in evaluation["rows"]:
            if row["decoding"] == "sampled":
                groups[(row["split"], row["source_group"], row["template"])].append(row["complete_path_success"])
        return {key: sum(values)/len(values) for key, values in groups.items()}
    before, after = means(parent_evaluation), means(child_evaluation)
    rows = [{"split": key[0], "source_group": key[1], "template": key[2],
             "parent_complete": before[key], "child_complete": after[key], "delta": after[key]-before[key]}
            for key in sorted(before)]
    summaries = {}
    for split in sorted({row["split"] for row in rows}):
        by_source = defaultdict(list)
        for row in rows:
            if row["split"] == split:
                by_source[row["source_group"]].append(row["delta"])
        values = [sum(v)/len(v) for v in by_source.values()]
        n = len(values)
        mean = sum(values)/n
        se = math.sqrt(sum((v-mean)**2 for v in values)/(n-1)/n) if n > 1 else None
        summaries[split] = {"sources": n, "mean_delta": mean, "paired_source_standard_error": se,
                            "source_deltas": values, "regressions": sum(v < 0 for v in values),
                            "boundary": "Descriptive source pairing; four heldout sources do not give a population guarantee."}
    return {"rows": rows, "summaries": summaries}


def run_seed(items, protocol, seed, *, sft_updates, sibling_updates, started, cap, artifact_dir=None):
    if seed not in protocol["seeds"] or not 1 <= sft_updates <= 24 or not 1 <= sibling_updates <= 16:
        raise ValueError("Use predeclared bounded seeds/updates")
    initial = rc.make_decoder(seed).double().eval()
    frozen_initial = rc.frozen_copy(initial)
    initial_hash = rc.state_hash(initial)
    train = [i for i in items if i["split"] == "train"]
    narrow_rows = [i for i in train if (i["a"], i["b"]) in ((0, 0), (0, 1))]*4
    evaluation_seed = seed+protocol["evaluation_rng_offset"]
    result = {"seed": seed, "interface_sha256": identity(protocol["interface"]),
              "architecture": asdict(initial.cfg), "ancestry": {}, "stages": {},
              "initial": evaluate(initial, items, evaluation_seed)}
    probabilities = exact_success(initial, train).tolist()
    result["initial_unsaturated_train_probabilities"] = probabilities
    if not all(0 < p < .9 for p in probabilities):
        raise RuntimeError("Initial compatible fixture must be measured unsaturated")
    models = {"initial": initial}
    for arm, rows in (("coverage-sft", train), ("repetition-sft", narrow_rows)):
        model = deepcopy(initial)
        statistics = _sft(model, rows, sft_updates, protocol, started, cap)
        statistics.update(evaluation=evaluate(model, items, evaluation_seed),
                          full_vocabulary=evaluate(model, items, evaluation_seed, samples=0, grammar=False),
                          cap1=evaluate(model, items, evaluation_seed, samples=0, cap=1))
        models[arm] = model
        result["stages"][arm] = statistics
        result["ancestry"][arm] = {"parent": "initial", "parent_state_sha256": initial_hash,
                                    "initial_state_sha256": initial_hash, "final_state_sha256": rc.state_hash(model)}
    if rc.state_hash(frozen_initial) != initial_hash or rc.state_hash(initial) != initial_hash:
        raise AssertionError("Random parent mutated")
    parent = rc.frozen_copy(models["coverage-sft"])
    parent_hash = rc.state_hash(parent)
    parent_probabilities = exact_success(parent, train).tolist()
    difficulty = protocol["difficulty"]
    eligible = [i for i, p in zip(train, parent_probabilities)
                if difficulty["min_probability"] <= p <= difficulty["max_probability"]]
    result["difficulty_selection"] = {"protocol": difficulty,
        "training_probabilities": dict(zip((i["id"] for i in train), parent_probabilities)),
        "all_training_prompt_ids": [i["id"] for i in train], "eligible_prompt_ids": [i["id"] for i in eligible],
        "discarded_prompt_ids": [i["id"] for i in train if i not in eligible],
        "original_prompt_probability": 1/len(train),
        "selected_prompt_probability": 1/len(eligible) if eligible else None,
        "selected_source_counts": dict(Counter(i["source_group"] for i in eligible)),
        "development_audit": signal_audit(parent, [i for i in items if i["split"] == "development"], seed+701, protocol["group_size"]),
        "training_audit": signal_audit(parent, eligible, seed+700, protocol["group_size"]) if eligible else None}
    for arm in protocol["sibling_arms"]:
        model = deepcopy(models["coverage-sft"])
        statistics = _sibling(model, parent, eligible, sibling_updates, protocol, seed, arm, started, cap)
        statistics.update(evaluation=evaluate(model, items, evaluation_seed),
                          full_vocabulary=evaluate(model, items, evaluation_seed, samples=0, grammar=False))
        statistics["paired_vs_parent"] = paired_source_deltas(result["stages"]["coverage-sft"]["evaluation"], statistics["evaluation"])
        models[arm] = model
        result["stages"][arm] = statistics
        result["ancestry"][arm] = {"parent": "coverage-sft", "parent_state_sha256": parent_hash,
                                    "initial_state_sha256": parent_hash, "final_state_sha256": rc.state_hash(model)}
    result["coverage_vs_repetition"] = paired_source_deltas(result["stages"]["repetition-sft"]["evaluation"],
                                                            result["stages"]["coverage-sft"]["evaluation"])
    if rc.state_hash(parent) != parent_hash:
        raise AssertionError("SFT reference mutated")
    result["frozen_reference_checks"] = {"initial": initial_hash, "sft_parent": parent_hash, "passed": True}
    reward_histories = [h for arm in ("grpo-std", "grpo-center") for h in result["stages"][arm]["history"]]
    train_signal = result["difficulty_selection"]["training_audit"]
    dev_signal = result["difficulty_selection"]["development_audit"]
    result["signal_gate"] = {"training_success_and_failure": bool(train_signal and train_signal["sampled_successes"] and train_signal["sampled_failures"]),
        "development_success_and_failure": bool(dev_signal["sampled_successes"] and dev_signal["sampled_failures"]),
        "mixed_training_groups": bool(train_signal and train_signal["mixed_groups"]),
        "independent_task_gradient": any(h["task_reward_gradient_norm"] > 1e-10 for h in reward_histories),
        "boundary": "Mechanism signal on symbolic train/dev; no real-model or heldout improvement guarantee."}
    if artifact_dir is not None:
        directory = Path(artifact_dir)/f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = {}
        for stage, model in models.items():
            path = directory/f"{stage}.pt"
            torch.save({"state_dict": model.state_dict(), "architecture": asdict(model.cfg),
                        "seed": seed, "stage": stage, "interface": protocol["interface"],
                        "interface_sha256": identity(protocol["interface"]),
                        "protocol_sha256": identity(protocol), "state_sha256": rc.state_hash(model),
                        "parent": result["ancestry"].get(stage)}, path)
            checkpoints[stage] = {"file": str(path.relative_to(Path(artifact_dir))), "file_sha256": file_hash(path),
                                   "state_sha256": rc.state_hash(model)}
        result["checkpoints"] = checkpoints
    return result


def finite_group_audit(probabilities=(.1, .5, .9), sizes=(2, 4, 8)):
    """Exact joint enumeration: action and random group normalizer stay coupled."""
    rows = []
    for p in probabilities:
        if not 0 < p < 1:
            raise ValueError("Interior Bernoulli probabilities required")
        for size in sizes:
            if type(size) is not int or not 2 <= size <= 12:
                raise ValueError("Finite teaching groups must fit enumeration cap")
            outcomes = list(product((0., 1.), repeat=size))
            rewards = torch.tensor(outcomes, dtype=torch.float64)
            successes = rewards.sum(-1)
            masses = p**successes*(1-p)**(size-successes)
            mixed = (successes > 0) & (successes < size)
            for normalized in (False, True):
                advantages = group_advantages(rewards, normalize=normalized)
                estimate = (advantages*(rewards-p)).mean(-1)
                mean = float((masses*estimate).sum())
                variance = float((masses*(estimate-mean)**2).sum())
                mixed_mass = float(masses[mixed].sum())
                selected_mean = float((masses[mixed]*estimate[mixed]).sum()/mixed_mass)
                rows.append({"success_probability": p, "group_size": size,
                    "normalization": "population-std" if normalized else "center-only",
                    "exact_expected_logit_gradient": mean, "exact_variance": variance,
                    "true_unbaselined_logit_gradient": p*(1-p),
                    "center_only_analytic_expectation": (size-1)/size*p*(1-p),
                    "large_group_std_approximation": math.sqrt(p*(1-p)) if normalized else p*(1-p),
                    "mixed_probability": mixed_mass, "conditional_mixed_expected_gradient": selected_mean,
                    "expected_collection_groups_per_mixed": 1/mixed_mass,
                    "all_constant_probability": 1-mixed_mass, "enumerated_outcomes": len(outcomes)})
    populations = []
    for size in sizes:
        masses = [1-p**size-(1-p)**size for p in probabilities]
        total = sum(masses)
        populations.append({"group_size": size, "original_prompt_probabilities": [1/len(probabilities)]*len(probabilities),
            "success_probabilities": list(probabilities), "mixed_selected_prompt_probabilities": [m/total for m in masses],
            "discarded_group_fraction": 1-total/len(probabilities),
            "attempts_per_selected_group": len(probabilities)/total})
    return {"rows": rows, "selection_populations": populations,
        "scope": "One-action Bernoulli score gradient, inclusive group mean, detached population std+1e-8. Exact finite joint enumeration, not an autoregressive quality experiment.",
        "direction": "Positive rows are ascent estimates; training minimizes their negative.",
        "approximation_boundary": "Large-group values are limits; finite std normalization is correlated with each reward. Mixed filtering conditions the estimator and changes prompt population and physical cost."}


def _instrument(text, stop, expected):
    original = bool(stop == "eos" and verify_integer(text, expected))
    item = {"kind": "math", "reference": str(expected), "format_policy": "any", "extraction": "boxed_or_final"}
    new = grade_response(item, {"raw_response": text, "stop_reason": "eos" if stop == "eos" else "max_tokens", "error": None})
    richer_complete = bool(new["task_success"] and new["natural_termination"])
    return {"strict_terminated_integer_reward": original, "richer_answer_correct": new["correct"],
        "richer_format_valid": new["format_valid"], "richer_natural_termination": new["natural_termination"],
        "richer_complete_reward": richer_complete, "richer_status": new["status"],
        "richer_reason": new.get("reason"), "richer_extraction": new["extraction"],
        "disagreement": original != richer_complete,
        "reason_code": "reward-instruments-disagree" if original != richer_complete else
                       "same-correct" if original else "same-zero",
        "protocol": "cs336-bounded-answer-audit-v1; diagnostic only"}


def native_reward_audit(root=ROOT):
    rows, sources = [], []
    for relative in ARCHIVES:
        path = Path(root)/relative
        report = json.loads(path.read_text())
        for record in report["records"]:
            match = re.fullmatch(r"Return only the integer answer\. ([0-9]+) \+ ([0-9]+) =", record["prompt"])
            if match is None:
                raise ValueError("Unknown archive prompt; no inferred answer")
            expected = int(match[1])+int(match[2])
            if not len(record["texts"]) == len(record["responses"]) == len(record["rewards"]) == len(record["stops"]):
                raise ValueError("Archive collection fields are not aligned")
            for sample, (text, ids, saved, stop) in enumerate(zip(record["texts"], record["responses"], record["rewards"], record["stops"])):
                grade = _instrument(text, stop, expected)
                if float(grade["strict_terminated_integer_reward"]) != saved:
                    raise AssertionError("Historical native reward did not replay exactly")
                rows.append(dict(grade, archive=relative, update=record["update"], source_id=record["source_id"],
                    sample_index=sample, prompt=record["prompt"], expected=expected, raw_response=text,
                    token_ids=ids, original_stop=stop, saved_reward=saved))
        sources.append({"path": relative, "sha256": file_hash(path), "updates": len(report["records"]),
                        "ancestry": report.get("parent_genealogy"), "status": report["status"]})
    probes = []
    for text, stop, expected, why in (("5", "eos", 5, "plain terminated integer"),
        ("Final answer: 5", "eos", 5, "extracted final answer with prose"),
        (r"\boxed{5}", "eos", 5, "boxed expression"), ("5", "max_tokens", 5, "correct but capped"),
        ("5 6", "eos", 5, "ambiguous two answers"), ("6", "eos", 5, "incorrect integer"),
        ("5+__import__('os').system('echo unsafe')", "eos", 5, "unsafe expression never executed"),
        ("５", "eos", 5, "Unicode digit outside strict ASCII contract")):
        probes.append(dict(_instrument(text, stop, expected), raw_response=text, original_stop=stop,
                           expected=expected, independent_reference="exact integer addition2+3", purpose=why))
    return {"name": "cs336-bounded-answer-audit-v1", "sources": sources, "rows": rows, "probes": probes,
        "summary": {"archived_responses": len(rows), "saved_positive_rewards": sum(r["saved_reward"] for r in rows),
                    "strict_replay_positive_rewards": sum(r["strict_terminated_integer_reward"] for r in rows),
                    "richer_complete_positive_rewards": sum(r["richer_complete_reward"] for r in rows),
                    "disagreements": sum(r["disagreement"] for r in rows),
                    "richer_statuses": dict(Counter(r["richer_status"] for r in rows))},
        "boundary": "Same retained text and original IDs/stops, no generation. Richer mathematical extraction is diagnostic and does not rewrite old rewards or become this fixture's training reward."}


def run_assignment(*, smoke=False, output=None, items=None, protocol=None):
    if items is None or protocol is None:
        items, protocol = load_fixture()
    validation = validate_fixture(items)
    specification = protocol["smoke" if smoke else "full"]
    seeds = protocol["smoke"]["seeds"] if smoke else protocol["seeds"]
    directory = Path(output) if output is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/"result.json").exists():
            raise FileExistsError("Retain existing result; choose a new report directory")
        (directory/"frozen-protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
        (directory/"frozen-items.json").write_text(json.dumps(items, indent=2)+"\n")
    started = time.perf_counter()
    result = {"status": "running", "mode": "smoke" if smoke else "reference", "protocol": protocol,
              "protocol_sha256": identity(protocol), "fixture": validation,
              "baselines": baselines(items), "seeds": [], "finite_group_audit": finite_group_audit(),
              "native_reward_audit": native_reward_audit(),
              "execution": {"python": sys.executable, "python_version": sys.version, "torch": torch.__version__,
                            "device": "cpu", "threads": torch.get_num_threads(), "dtype": protocol["dtype"]}}
    try:
        for seed in seeds:
            result["seeds"].append(run_seed(items, protocol, seed,
                sft_updates=specification["sft_updates"], sibling_updates=specification["sibling_updates"],
                started=started, cap=specification["runtime_cap_seconds"], artifact_dir=directory))
            if directory is not None:
                (directory/f"seed-{seed}/result.json").write_text(json.dumps(result["seeds"][-1], indent=2, allow_nan=False)+"\n")
        _deadline(started, specification["runtime_cap_seconds"])
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["runtime_seconds"] = time.perf_counter()-started
        result["source_sha256"] = {str(path.relative_to(ROOT)): file_hash(path) for path in
            (Path(__file__), ROOT/"src/dongxi_llms/reasoning_controls.py", ROOT/"src/dongxi_llms/grpo_lab.py",
             ROOT/"src/dongxi_llms/sft_lab.py", ROOT/"src/dongxi_llms/decoder_lab.py", ROOT/"src/dongxi_llms/reasoning_evaluation.py",
             FIXTURE/"posttraining_items.json", FIXTURE/"posttraining_protocol.json")}
        if directory is not None:
            (directory/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run_assignment(smoke=arguments.smoke, output=arguments.output)
    print(json.dumps({"status": result["status"], "mode": result["mode"],
                      "seeds": [s["seed"] for s in result["seeds"]],
                      "runtime_seconds": result["runtime_seconds"],
                      "signal_gates": [s["signal_gate"] for s in result["seeds"]],
                      "native_reward_audit": result["native_reward_audit"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()

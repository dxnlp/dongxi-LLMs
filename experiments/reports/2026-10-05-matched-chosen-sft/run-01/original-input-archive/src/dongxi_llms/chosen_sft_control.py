"""Matched chosen-response control using the existing SFT and DPO runner APIs.

No acquisition or execution on import. Equal chosen draws are not equal rejected
supervision, reductions or compute. This helper has no accounted resume API.
"""
from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache
import importlib.util
import math
from pathlib import Path

import torch

from .batched_cache_lab import digest
from .dpo_lab import sequence_logps
from .run_identity import canonical_hash, tokenizer_interface

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dongxi-matched-chosen-sft-v1"


@lru_cache(maxsize=1)
def native_runners():
    """Use the repository's actual functions; never execute their CLI on import."""
    modules = []
    for chapter, name in ((9, "run_chapter09_spark_sft.py"),
                          (11, "run_chapter11_spark_dpo.py")):
        path = ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(f"dongxi_chosen_control_{chapter}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("The paired control requires its original repository runner sources")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _integer(value, name, minimum=1, maximum=1000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} requires an exact integer in {minimum}..{maximum}")
    return value


def recipe(*, seed, updates, accumulation, learning_rate, beta, max_length, max_new_tokens):
    _integer(seed, "seed", 0, 2**63-1)
    _integer(updates, "updates")
    _integer(accumulation, "accumulation", 1, 16)
    _integer(max_length, "max length", 2, 4096)
    _integer(max_new_tokens, "generation cap", 1, 256)
    for key, value in (("learning rate", learning_rate), ("beta", beta)):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Finite positive {key} required")
    return dict(seed=seed, updates=updates, accumulation=accumulation,
                learning_rate=float(learning_rate), beta=float(beta),
                weight_decay=.01, clip_norm=1., max_length=max_length,
                max_new_tokens=max_new_tokens, decoding="greedy-uncached")


def _rows(rows, kind):
    if type(rows) is not list or not 1 <= len(rows) <= 4096:
        raise ValueError("Nonempty bounded explicit records required")
    required = {"id", "group", "prompt", "expected"} if kind == "evaluation" else {
        "id", "group", "prompt", "chosen", "rejected"}
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise ValueError("Exact pair/evaluation schema with explicit source group required")
        for key in ("id", "group"):
            if type(row[key]) is not str or not 1 <= len(row[key]) <= 256:
                raise ValueError("Nonempty bounded IDs and source groups required")
        prompt = row["prompt"]
        if type(prompt) is not list or not 1 <= len(prompt) <= 32:
            raise ValueError("Explicit bounded message prompt required")
        for message in prompt:
            if (type(message) is not dict or set(message) != {"role", "content"}
                    or message["role"] not in ("user", "assistant", "system")
                    or type(message["content"]) is not str
                    or not 1 <= len(message["content"]) <= 8192):
                raise ValueError("Known message roles and nonempty bounded content required")
        if prompt[-1]["role"] == "assistant":
            raise ValueError("Prompt must end before the chosen assistant response")
        for key in ("expected",) if kind == "evaluation" else ("chosen", "rejected"):
            if type(row[key]) is not str or not row[key].strip() or len(row[key]) > 8192:
                raise ValueError("Nonempty bounded literal answer required")


def _encoded_pair(pair, vocab_size, stop_ids):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("Both preference branches required")
    result = []
    for branch in pair:
        if not isinstance(branch, (tuple, list)) or len(branch) != 2:
            raise ValueError("IDs and exact target-token mask required")
        ids, mask = branch
        if (type(ids) is not list or type(mask) is not list or len(ids) < 2
                or len(ids) != len(mask) or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
                or any(type(value) is not bool for value in mask)):
            raise ValueError("Valid integer token IDs and boolean mask required")
        if mask[0] or not any(mask[1:]) or ids[-1] not in stop_ids or not mask[-1]:
            raise ValueError("Prompt masked and real terminal token supervised exactly once")
        first = mask.index(True)
        if any(mask[:first]) or not all(mask[first:]):
            raise ValueError("Final response span must be contiguous, including termination")
        result.append((list(ids), list(mask)))
    left, right = result
    lp, rp = left[1].index(True), right[1].index(True)
    if left[0][:lp] != right[0][:rp]:
        raise ValueError("Preference branches must share the exact generation prefix")
    return result


def encode_dataset(tokenizer, train, validation, evaluation, *, max_length):
    """Native encodings, one complete final-answer span and explicit split gates."""
    _integer(max_length, "max length", 2, 4096)
    for rows, kind in ((train, "pairs"), (validation, "pairs"), (evaluation, "evaluation")):
        _rows(rows, kind)
    all_rows = train + validation + evaluation
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("IDs must be unique across all splits")
    prompt_sets = [{canonical_hash(row["prompt"]) for row in rows}
                   for rows in (train, validation, evaluation)]
    group_sets = [{row["group"] for row in rows} for rows in (train, validation, evaluation)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if prompt_sets[a] & prompt_sets[b] or group_sets[a] & group_sets[b]:
            raise ValueError("Raw prompts and source groups must be split-disjoint")
    _, dpo = native_runners()
    stops = [tokenizer.eos_token_id]
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end is not None and end != tokenizer.unk_token_id and end not in stops:
        stops.append(end)
    if tokenizer.pad_token_id is None or any(type(i) is not int or i < 0 for i in stops):
        raise ValueError("Explicit pad and real EOS/turn-ending IDs required")
    encoded = [[_encoded_pair(dpo.encode_pair(tokenizer, row, max_length), len(tokenizer), stops)
                for row in rows] for rows in (train, validation)]
    prefixes = [tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in evaluation]
    if any(not prefix or len(prefix) >= max_length for prefix in prefixes):
        raise ValueError("Complete publication prefix must fit without truncation")
    all_ids = [ids for rows in encoded for pair in rows for ids, _ in pair] + prefixes
    if any(any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids) for ids in all_ids):
        raise ValueError("Invalid actual tokenizer IDs")
    if tokenizer.unk_token_id is not None and any(tokenizer.unk_token_id in ids for ids in all_ids):
        raise ValueError("Unknown-token collision refuses; no silent text substitution")
    split = dpo.check_split_interfaces(tokenizer, [train, validation, evaluation], [*encoded, prefixes])
    return dict(train=encoded[0], validation=encoded[1], prefixes=prefixes,
                evaluation=deepcopy(evaluation), stop_ids=stops, pad_id=tokenizer.pad_token_id,
                split=split, rows_sha256=canonical_hash([train, validation, evaluation]),
                raw_splits=deepcopy([train, validation, evaluation]),
                interface_sha256=canonical_hash(tokenizer_interface(tokenizer,
                    template=tokenizer.chat_template, stop_ids=stops)),
                encoded_sha256=digest([*encoded, prefixes]),
                train_ids=[row["id"] for row in train], train_groups=[row["group"] for row in train],
                validation_ids=[row["id"] for row in validation],
                evaluation_ids=[row["id"] for row in evaluation])


def chosen_record(pair, identifier):
    """Convert the exact native DPO chosen target mask to native SFT labels."""
    ids, mask = pair[0]
    return dict(id=identifier, ids=list(ids),
                labels=[token if valid else -100 for token, valid in zip(ids, mask)])


class ChosenSFTLoop:
    """Actual chosen-only updates with DPO's private replacement draw schedule.

    Partial updates are poisoned. There is no resume/ledger/production-safety
    promise; the original parent is the externally retained restart boundary.
    """
    def __init__(self, model, optimizer, sampler, encoded_train, *, pad_id, stop_ids,
                 accumulation, updates, device="cpu", guard=lambda: None,
                 autocast_factory=nullcontext):
        _integer(accumulation, "accumulation", 1, 16)
        _integer(updates, "updates")
        _integer(pad_id, "pad ID", 0, 2**31-1)
        if not encoded_train or not isinstance(sampler, torch.Generator) or sampler.device.type != "cpu":
            raise ValueError("Nonempty pairs and a private CPU Torch generator required")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (not parameters or len(parameters) != len(optimizer_parameters)
                or {id(p) for p in parameters} != {id(p) for p in optimizer_parameters}):
            raise ValueError("Optimizer must cover each actual trainable policy parameter exactly once")
        vocab = model.get_input_embeddings().weight.shape[0]
        if (type(stop_ids) is not list or not stop_ids or len(set(stop_ids)) != len(stop_ids)
                or any(type(value) is not int or not 0 <= value < vocab for value in stop_ids)):
            raise ValueError("Explicit real parent stop IDs required")
        self.train = [_encoded_pair(pair, vocab, stop_ids) for pair in encoded_train]
        self.model, self.optimizer, self.sampler = model, optimizer, sampler
        self.parameters, self.pad_id = parameters, pad_id
        self.accumulation, self.updates, self.device = accumulation, updates, torch.device(device)
        self.guard, self.autocast_factory = guard, autocast_factory
        self.completed, self.poisoned, self.history, self.attempts = 0, False, [], []

    def completed_update(self):
        if self.poisoned:
            raise RuntimeError("Interrupted chosen-SFT update requires restart from the original parent")
        if self.completed >= self.updates:
            raise ValueError("Fixed chosen-SFT update horizon exhausted")
        sft, _ = native_runners()
        attempted = dict(update=self.completed+1, indices=[], policy_forward_calls_entered=0,
                         policy_forward_calls_completed=0, policy_forward_positions_entered=0,
                         policy_forward_positions_completed=0, completed=False, error=None)
        self.attempts.append(attempted)
        try:
            self.guard()
            self.poisoned = True
            window = []
            for slot in range(self.accumulation):
                self.guard()
                index = int(torch.randint(len(self.train), (), generator=self.sampler))
                attempted["indices"].append(index)
                row = chosen_record(self.train[index], f"draw-{self.completed}-{slot}-{index}")
                window.append(sft.collate([row], self.pad_id, self.device))
            targets = sum(int((batch[1][:, 1:] != -100).sum()) for batch in window)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for batch in window:
                self.guard()
                positions = batch[0].numel()
                attempted["policy_forward_calls_entered"] += 1
                attempted["policy_forward_positions_entered"] += positions
                with self.autocast_factory():
                    loss = sft.summed_loss(self.model, batch)
                attempted["policy_forward_calls_completed"] += 1
                attempted["policy_forward_positions_completed"] += positions
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite chosen-token NLL")
                (loss / targets).backward()
                total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
            self.guard()
            self.optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters):
                raise RuntimeError("Nonfinite chosen-SFT updated policy")
            work = dict(sampled_pairs=self.accumulation, sampler_draws=self.accumulation,
                        chosen_targets=targets, rejected_targets=0,
                        logical_sequence_tokens=sum(batch[0].numel() for batch in window),
                        policy_forward_calls=self.accumulation, reference_forward_calls=0,
                        policy_forward_positions=sum(batch[0].numel() for batch in window),
                        reference_forward_positions=0)
            self.completed += 1
            row = dict(update=self.completed, loss=total_loss/targets, gradient_norm=float(norm),
                       indices=list(attempted["indices"]), work=work)
            self.history.append(deepcopy(row))
            attempted["completed"] = True
            self.poisoned = False
            return deepcopy(row)
        except BaseException as error:
            self.poisoned = True
            attempted["error"] = dict(type=type(error).__name__, message=str(error)[:512],
                scope="known entered/successful forward geometry; unknown backend partial work is not invented")
            raise


def score_pairs(model, reference, encoded, *, pad_id, device, beta,
                guard=lambda: None, autocast_factory=nullcontext):
    """Actual native pair diagnostics plus absolute complete-response likelihoods."""
    _, dpo = native_runners()
    diagnostics = dpo.score_dpo_panel(model, reference, encoded, pad_id=pad_id,
        device=device, beta=beta, guard=guard, autocast_factory=autocast_factory)
    model.eval()
    with torch.no_grad(), autocast_factory():
        for pair, row in zip(encoded, diagnostics):
            for branch, name in ((0, "chosen"), (1, "rejected")):
                guard()
                ids, attention, mask = dpo.collate([pair], branch, pad_id, device)
                output = model(input_ids=ids[:, :-1], attention_mask=attention[:, :-1], use_cache=False)
                value = sequence_logps(output.logits.float(), ids[:, 1:], mask[:, 1:] & attention[:, 1:].bool())
                row[name+"_logp"] = float(value[0])
                row[name+"_targets"] = int(mask[:, 1:].sum())
    return diagnostics


def run_arm(parent, dataset, tokenizer, settings, *, arm, guard=lambda: None,
            autocast_factory=nullcontext, row_sink=None):
    """Callable real model control; supplied parent is never trained in place.

    The caller owns local model loading/export and external deadline. Publication
    gold is used only after generation. There is no coefficient/checkpoint search.
    """
    if arm not in ("unchanged", "chosen-sft", "dpo"):
        raise ValueError("Known independent arm required")
    checked = recipe(**{key: settings[key] for key in ("seed", "updates", "accumulation",
        "learning_rate", "beta", "max_length", "max_new_tokens")})
    if settings != checked:
        raise ValueError("Exact frozen recipe required; no silently ignored settings")
    dataset = deepcopy(dataset)
    if (digest([dataset["train"], dataset["validation"], dataset["prefixes"]]) != dataset["encoded_sha256"]
            or canonical_hash(dataset["raw_splits"]) != dataset["rows_sha256"]
            or canonical_hash(tokenizer_interface(tokenizer, template=tokenizer.chat_template,
                stop_ids=dataset["stop_ids"])) != dataset["interface_sha256"]):
        raise ValueError("Actual data/encoding/tokenizer contract changed before training")
    _, dpo = native_runners()
    guard()
    parent_sha = digest(parent.state_dict())
    device = next(parent.parameters()).device
    model = deepcopy(parent)
    reference = deepcopy(parent).eval().requires_grad_(False)
    model.config.use_cache = reference.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    for module in reference.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    reference_sha = digest(reference.state_dict())
    initial_policy_sha = digest(model.state_dict())
    if initial_policy_sha != parent_sha or reference_sha != parent_sha:
        raise AssertionError("Actual initial policy/reference tensors differ from the parent")
    optimizer = torch.optim.AdamW(model.parameters(), lr=checked["learning_rate"], weight_decay=.01)
    sampler = torch.Generator().manual_seed(checked["seed"])
    histories, attempts = [], []
    # Fork the global RNG so comparison/evaluation cannot change caller training.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(checked["seed"])
        if arm == "chosen-sft":
            loop = ChosenSFTLoop(model, optimizer, sampler, dataset["train"],
                pad_id=dataset["pad_id"], stop_ids=dataset["stop_ids"], accumulation=checked["accumulation"],
                updates=checked["updates"], device=device, guard=guard,
                autocast_factory=autocast_factory)
            for _ in range(checked["updates"]):
                try:
                    row = loop.completed_update()
                except BaseException as error:
                    if row_sink is not None:
                        row_sink(dict(arm=arm, phase="failed-update", **deepcopy(loop.attempts[-1])))
                    raise
                histories.append(row)
                if row_sink is not None:
                    row_sink(dict(arm=arm, phase="completed-update", **deepcopy(row)))
            attempts = loop.attempts
        elif arm == "dpo":
            for update in range(1, checked["updates"]+1):
                attempted = dpo.zero_work()
                try:
                    row = dpo.completed_dpo_update(model, reference, optimizer, sampler, dataset["train"],
                        pad_id=dataset["pad_id"], accumulation=checked["accumulation"], beta=checked["beta"],
                        device=device, update=update, guard=guard, autocast_factory=autocast_factory,
                        attempted_work=attempted)
                except BaseException as error:
                    if row_sink is not None:
                        row_sink(dict(arm=arm, phase="failed-update", update=update,
                            attempted_work=attempted, error=dict(type=type(error).__name__, message=str(error)[:512])))
                    raise
                histories.append(row)
                attempts.append(dict(update=update, completed=True, attempted_work=attempted))
                if row_sink is not None:
                    row_sink(dict(arm=arm, phase="completed-update", **deepcopy(row)))
        # Never use this heldout panel to choose updates, coefficients or arm.
        diagnostics = score_pairs(model, reference, dataset["validation"],
            pad_id=dataset["pad_id"], device=device, beta=checked["beta"],
            guard=guard, autocast_factory=autocast_factory)
        generated = dpo.generate_dpo_panel(model, tokenizer, dataset["evaluation"], dataset["prefixes"],
            cap=checked["max_new_tokens"], stop_ids=dataset["stop_ids"], device=device,
            guard=guard, autocast_factory=autocast_factory,
            row_sink=(lambda row: row_sink(dict(arm=arm, phase="publication-generation", **row))) if row_sink else None)
    if digest(parent.state_dict()) != parent_sha or digest(reference.state_dict()) != reference_sha:
        raise AssertionError("Original parent/reference changed during the control")
    if any(parameter.grad is not None for parameter in reference.parameters()):
        raise AssertionError("Frozen original reference received gradients")
    work = {key: sum(row["work"][key] for row in histories) for key in dpo.WORK_KEYS}
    result = dict(schema=SCHEMA, arm=arm, recipe=checked,
        dataset=dict(rows_sha256=dataset["rows_sha256"], encoded_sha256=dataset["encoded_sha256"],
                     chosen_sha256=digest([pair[0] for pair in dataset["train"]]),
                     interface_sha256=dataset["interface_sha256"],
                     train_ids=dataset["train_ids"], train_groups=dataset["train_groups"],
                     validation_ids=dataset["validation_ids"], evaluation_ids=dataset["evaluation_ids"]),
        objective="none" if arm == "unchanged" else "mean valid chosen tokens per update" if arm == "chosen-sft" else "native mean pairs; summed completion log probabilities",
        parent_sha256=parent_sha, initial_policy_sha256=initial_policy_sha,
        final_policy_sha256=digest(model.state_dict()), reference_sha256=reference_sha,
        sampler_sha256=digest(sampler.get_state()), history=histories, attempts=attempts,
        training_work=work, validation=diagnostics, publication=generated,
        independent_exact_match=sum(row["exact_match"] is True for row in generated)/len(generated),
        scope="Original location fixture and local model only; no equal-compute, pretrained quality or accounted recovery claim")
    return model, result


def compare_results(results):
    """Check the actual matching claim, never match rejected/reference work away."""
    if set(results) != {"unchanged", "chosen-sft", "dpo"}:
        raise ValueError("All three predetermined controls required")
    baseline, chosen, dpo = (results[key] for key in ("unchanged", "chosen-sft", "dpo"))
    if (len({row["parent_sha256"] for row in results.values()}) != 1
            or len({canonical_hash(row["recipe"]) for row in results.values()}) != 1
            or len({canonical_hash(row["dataset"]) for row in results.values()}) != 1):
        raise ValueError("Controls must start at the same parent and declared recipe")
    if baseline["history"] or baseline["final_policy_sha256"] != baseline["parent_sha256"]:
        raise ValueError("Unchanged-parent control actually changed")
    for key in ("indices",):
        if [row[key] for row in chosen["history"]] != [row[key] for row in dpo["history"]]:
            raise ValueError("Chosen exposure cannot be called matched with different replacement draws")
    for key in ("sampled_pairs", "sampler_draws", "chosen_targets"):
        if chosen["training_work"][key] != dpo["training_work"][key]:
            raise ValueError("Chosen response exposure differs")
    if chosen["sampler_sha256"] != dpo["sampler_sha256"]:
        raise ValueError("Actual private sampler endpoints differ")
    return dict(matching="same actual parent, replacement draws, chosen targets, updates and recipe",
        unmatched="rejected supervision, objective reduction, reference work, policy input geometry and compute",
        independent_exact_match={key: row["independent_exact_match"] for key, row in results.items()},
        training_work={key: row["training_work"] for key, row in results.items()})

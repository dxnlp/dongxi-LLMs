"""Matched chosen-response control using the existing SFT and DPO runner APIs.

No acquisition or execution on import. Equal chosen draws are not equal rejected
supervision, reductions or compute. Accounted recovery is opt-in and restores
completed numerical boundaries without refunding later failed logical work.
"""
from contextlib import nullcontext
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import re

import torch

from .batched_cache_lab import digest
from .dpo_lab import sequence_logps
from .run_identity import canonical_hash, tokenizer_interface
from .run_identity import file_digest, environment_identity
from .work_budget import WorkLedger
from .snapshot_io_budget import (SnapshotIOBudget, validate_io_contract,
    validate_work_receipt, io_ledger_contract_sha256)
from .training_snapshot import load_snapshot, save_snapshot

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dongxi-matched-chosen-sft-v1"
RECOVERY_SCHEMA = "dongxi-chosen-sft-completed-v1"
RECOVERY_SOURCES = ("src/dongxi_llms/chosen_sft_control.py",
    "scripts/run_chapter09_spark_sft.py", "scripts/run_chapter11_spark_dpo.py",
    "src/dongxi_llms/work_budget.py", "src/dongxi_llms/training_snapshot.py",
    "src/dongxi_llms/snapshot_io_budget.py", "src/dongxi_llms/run_identity.py",
    "src/dongxi_llms/batched_cache_lab.py")


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


def chosen_work_budget_contract(limits, max_bytes):
    """Same nineteen dimensions as DPO; chosen-only scientific schema is distinct."""
    _, dpo = native_runners()
    result = dpo.work_budget_contract(limits, max_bytes)
    result['schema'] = 'dongxi-chosen-sft-logical-work-v1'
    return result


def _verify_recovery_files(contract):
    for role in ('parent_files', 'inputs', 'sources'):
        files = contract[role]
        if type(files) is not dict or not 1 <= len(files) <= 256:
            raise ValueError("Nonempty bounded actual chosen file identities required")
        for name, expected in files.items():
            if (type(name) is not str or not 1 <= len(name) <= 4096
                    or type(expected) is not str or re.fullmatch('[0-9a-f]{64}', expected) is None):
                raise ValueError("Explicit chosen file paths and byte hashes required")
            path = Path(name)
            if file_digest(path if path.is_absolute() else ROOT/path) != expected:
                raise ValueError("Actual chosen parent/input/source bytes changed: "+name)
    if not set(RECOVERY_SOURCES) <= set(contract['sources']):
        raise ValueError("Original chosen/native/shared recovery source identities required")


def _verify_recovery_environment(contract):
    environment = contract['environment']
    if (type(environment) is not dict or type(environment.get('environment_lock')) is not dict
            or environment['environment_lock'].get('status') != 'hashed'
            or environment != environment_identity(environment['environment_lock'].get('path'))):
        raise ValueError("Actual chosen numerical environment/selected lock differs from retained identity")


def _validate_recovery_contract(contract):
    required = {'schema', 'loop', 'parent_sha256', 'parent_files', 'inputs', 'sources',
                'environment', 'interface_sha256', 'dataset', 'recipe', 'work_budget', 'snapshot_io_budget'}
    if type(contract) is not dict or set(contract) != required or contract['schema'] != RECOVERY_SCHEMA:
        raise ValueError("Exact independently retained chosen recovery contract required")
    for key in ('parent_sha256', 'interface_sha256'):
        if type(contract[key]) is not str or re.fullmatch('[0-9a-f]{64}', contract[key]) is None:
            raise ValueError("Exact chosen recovery digest required")
    budget = contract['work_budget']
    if (type(budget) is not dict or budget != chosen_work_budget_contract(budget.get('limits'), budget.get('max_bytes'))
            or contract['snapshot_io_budget'] != validate_io_contract(contract['snapshot_io_budget'])):
        raise ValueError("Chosen scientific/work/I/O caps cannot be silently migrated")


def make_chosen_recovery_contract(loop, *, dataset, tokenizer, settings, parent_files,
                                 inputs, sources, environment, work_budget, snapshot_io_budget):
    """Observe a fresh original parent; caller retains this contract independently.

    Parent/input/source paths are actually hashed, not revision declarations.
    Environment is the caller's observed numerical package/lock identity; this
    API does not acquire files, initialize a GPU or change the environment.
    """
    if not isinstance(loop, ChosenSFTLoop) or loop.completed or loop.history or loop.poisoned:
        raise ValueError("Chosen recovery contract requires the fresh original parent")
    checked = recipe(**{key: settings[key] for key in ('seed', 'updates', 'accumulation',
        'learning_rate', 'beta', 'max_length', 'max_new_tokens')})
    if settings != checked:
        raise ValueError("Exact frozen chosen matching recipe required")
    suffix = dataset.get('terminal_suffix_ids', [])
    reencoded = encode_dataset(tokenizer, *dataset['raw_splits'], max_length=checked['max_length'],
                               terminal_suffix_ids=suffix)
    if (loop._sampler_seed != checked['seed'] or loop.updates != checked['updates']
            or loop.accumulation != checked['accumulation']
            or loop.terminal_suffix_ids != suffix
            or loop.optimizer.state or any(parameter.grad is not None for parameter in loop.parameters)
            or not torch.equal(loop._initial_sampler_state, torch.Generator().manual_seed(checked['seed']).get_state())
            or digest(loop.train) != digest(dataset['train'])
            or reencoded['encoded_sha256'] != dataset['encoded_sha256']
            or digest([dataset['train'], dataset['validation'], dataset['prefixes']]) != dataset['encoded_sha256']
            or canonical_hash(dataset['raw_splits']) != dataset['rows_sha256']
            or dataset['interface_sha256'] != canonical_hash(tokenizer_interface(tokenizer,
                template=tokenizer.chat_template, stop_ids=loop.stop_ids))
            or loop.optimizer.param_groups[0]['lr'] != checked['learning_rate']
            or loop.optimizer.param_groups[0]['weight_decay'] != checked['weight_decay']):
        raise ValueError("Actual chosen data/interface/seed/recipe changed")
    _, dpo = native_runners()
    result = dpo.json_plain(dict(schema=RECOVERY_SCHEMA, loop=loop._recovery_layout(),
        parent_sha256=digest(loop.model.state_dict()), parent_files=parent_files,
        inputs=inputs, sources=sources, environment=environment,
        interface_sha256=dataset['interface_sha256'],
        dataset={key: deepcopy(dataset[key]) for key in ('rows_sha256', 'encoded_sha256',
            'train_ids', 'train_groups', 'validation_ids', 'evaluation_ids')
            + (('terminal_suffix_ids',) if suffix else ())},
        recipe=checked, work_budget=work_budget, snapshot_io_budget=snapshot_io_budget))
    _validate_recovery_contract(result); _verify_recovery_files(result); _verify_recovery_environment(result)
    return result


def chosen_validation_upper(contract, completed):
    """Expected semantic replay envelope; excludes shared hash/tree/I/O work."""
    _validate_recovery_contract(contract)
    layout = contract['loop']
    _integer(completed, 'chosen completed cursor', 0, layout['updates'])
    p = sum(math.prod(shape['shape']) for shape in layout['policy_shapes'].values())
    o = sum(math.prod(shape['shape']) for shape in layout['optimizer_shapes'])
    return dict(recovery_validation_operations=1, recovery_history_rows=completed,
        recovery_tensor_elements=p+(3*o+len(layout['optimizer_shapes']) if completed else 0)
            +4*layout['cpu_rng_bytes']+sum(layout['cuda_rng_bytes']),
        recovery_rng_states=2+len(layout['cuda_rng_bytes']),
        recovery_sampler_draws=completed*layout['accumulation'])


def open_chosen_resume_budgets(*, contract, receipt, work_path, io_path, invocation_id,
                              expected_sha256, expected_bytes):
    """Bind both physical prefixes to independent bytes BEFORE payload loading."""
    _validate_recovery_contract(contract)
    receipt = validate_work_receipt(receipt)
    header = receipt['snapshot']; science = canonical_hash(contract)
    if (header['phase'] != 'completed' or header['completed_updates'] > contract['loop']['updates']
            or header['contract_sha256'] != science or header['payload_sha256'] != expected_sha256
            or type(expected_bytes) is not int or header['payload_bytes'] != expected_bytes
            or receipt['runner_work_prefix'] is None):
        raise ValueError("Independent chosen snapshot receipt/payload/science expectation changed")
    work = io = None
    try:
        budget, io_contract = contract['work_budget'], contract['snapshot_io_budget']
        work = WorkLedger.open(work_path, limits=budget['limits'], contract_sha256=science,
            max_bytes=budget['max_bytes'], invocation_id=invocation_id,
            expected_snapshot=receipt['runner_work_prefix'])
        io = WorkLedger.open(io_path, limits=io_contract['limits'],
            contract_sha256=io_ledger_contract_sha256(io_contract, science),
            max_bytes=io_contract['max_journal_bytes'], invocation_id=invocation_id,
            expected_snapshot=receipt['io_prefix'])
        hook = SnapshotIOBudget(io, contract=io_contract, scientific_contract_sha256=science,
            expected_receipt=receipt)
        return work, io, hook
    except BaseException:
        if io is not None: io.close()
        if work is not None: work.close()
        raise


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


def _terminal_suffix(value, vocab_size, stop_ids):
    """Opt-in exact template separators, never replacement stop IDs or trimming.

    The accepted original pretrained template ends a supervised assistant turn
    with its real stop token then newline198. The empty default retains the
    historical strict-final-EOS path. At most eight explicitly supplied IDs are
    allowed; every actual branch must end in exactly that audited suffix.
    """
    suffix = [] if value is None else value
    if (type(suffix) is not list or len(suffix) > 8
            or any(type(token) is not int or not 0 <= token < vocab_size or token in stop_ids
                   for token in suffix)):
        raise ValueError("Bounded explicit non-stop template suffix IDs required")
    return list(suffix)


def _encoded_pair(pair, vocab_size, stop_ids, terminal_suffix_ids=None):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("Both preference branches required")
    result = []
    suffix = _terminal_suffix(terminal_suffix_ids, vocab_size, stop_ids)
    for branch in pair:
        if not isinstance(branch, (tuple, list)) or len(branch) != 2:
            raise ValueError("IDs and exact target-token mask required")
        ids, mask = branch
        if (type(ids) is not list or type(mask) is not list or len(ids) < 2
                or len(ids) != len(mask) or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
                or any(type(value) is not bool for value in mask)):
            raise ValueError("Valid integer token IDs and boolean mask required")
        terminal = len(ids)-len(suffix)-1
        if (mask[0] or not any(mask[1:]) or terminal < 1 or ids[terminal] not in stop_ids
                or ids[terminal+1:] != suffix or not mask[terminal] or not mask[-1]):
            raise ValueError("Prompt masked and real terminal token supervised exactly once")
        first = mask.index(True)
        if any(mask[:first]) or not all(mask[first:]):
            raise ValueError("Final response span must be contiguous, including termination")
        if suffix and [i for i in range(first, len(ids)) if ids[i] in stop_ids] != [terminal]:
            raise ValueError("Exactly one real supervised stop before the audited template suffix required")
        result.append((list(ids), list(mask)))
    left, right = result
    lp, rp = left[1].index(True), right[1].index(True)
    if left[0][:lp] != right[0][:rp]:
        raise ValueError("Preference branches must share the exact generation prefix")
    return result


def encode_dataset(tokenizer, train, validation, evaluation, *, max_length, terminal_suffix_ids=None):
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
    suffix = _terminal_suffix(terminal_suffix_ids, len(tokenizer), stops)
    encoded = [[_encoded_pair(dpo.encode_pair(tokenizer, row, max_length), len(tokenizer), stops, suffix)
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
    result = dict(train=encoded[0], validation=encoded[1], prefixes=prefixes,
                evaluation=deepcopy(evaluation), stop_ids=stops, pad_id=tokenizer.pad_token_id,
                split=split, rows_sha256=canonical_hash([train, validation, evaluation]),
                raw_splits=deepcopy([train, validation, evaluation]),
                interface_sha256=canonical_hash(tokenizer_interface(tokenizer,
                    template=tokenizer.chat_template, stop_ids=stops)),
                encoded_sha256=digest([*encoded, prefixes]),
                train_ids=[row["id"] for row in train], train_groups=[row["group"] for row in train],
                validation_ids=[row["id"] for row in validation],
                evaluation_ids=[row["id"] for row in evaluation])
    if suffix:
        result['terminal_suffix_ids'] = suffix
    return result


def chosen_record(pair, identifier):
    """Convert the exact native DPO chosen target mask to native SFT labels."""
    ids, mask = pair[0]
    return dict(id=identifier, ids=list(ids),
                labels=[token if valid else -100 for token, valid in zip(ids, mask)])


class ChosenSFTLoop:
    """Actual chosen-only updates with DPO's private replacement draw schedule.

    Partial updates are poisoned. Opt-in recovery requires a separately retained
    contract, the same physical work/I/O journals and an independent byte receipt.
    The unaccounted comparison path remains unchanged; neither path is a physical
    resource quota or an external process supervisor.
    """
    def __init__(self, model, optimizer, sampler, encoded_train, *, pad_id, stop_ids,
                 accumulation, updates, device="cpu", guard=lambda: None,
                 autocast_factory=nullcontext, autocast_dtype=None, terminal_suffix_ids=None):
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
        self.terminal_suffix_ids = _terminal_suffix(terminal_suffix_ids, vocab, stop_ids)
        self.train = [_encoded_pair(pair, vocab, stop_ids, self.terminal_suffix_ids) for pair in encoded_train]
        self.model, self.optimizer, self.sampler = model, optimizer, sampler
        self.parameters, self.pad_id = parameters, pad_id
        self.accumulation, self.updates, self.device = accumulation, updates, torch.device(device)
        if autocast_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValueError("Explicit supported managed chosen autocast dtype required")
        if autocast_dtype is not None and autocast_factory is not nullcontext:
            raise ValueError("Do not combine an opaque factory with managed chosen autocast")
        self.autocast_dtype = autocast_dtype
        self.guard = guard
        self.autocast_factory = (autocast_factory if autocast_dtype is None else
            lambda: torch.autocast(self.device.type, dtype=self.autocast_dtype))
        self._managed_autocast_factory = self.autocast_factory if (
            autocast_factory is nullcontext or autocast_dtype is not None) else None
        self.completed, self.poisoned, self.history, self.attempts = 0, False, [], []
        self.work_ledger = self.io_budget = self.recovery_contract = None
        self._initial_sampler_state = sampler.get_state().clone()
        self._sampler_seed = sampler.initial_seed()
        self._recovery_hash = None
        self._recovery_tokenizer = None
        self.stop_ids = list(stop_ids)

    def _sample_work(self):
        return [dict(sampled_pairs=1, sampler_draws=1, chosen_targets=sum(pair[0][1][1:]),
            rejected_targets=0, logical_sequence_tokens=len(pair[0][0]),
            policy_forward_calls=1, reference_forward_calls=0,
            policy_forward_positions=len(pair[0][0]), reference_forward_positions=0)
            for pair in self.train]

    def _recovery_layout(self):
        _, dpo = native_runners()
        if (self._managed_autocast_factory is None or self.autocast_factory is not self._managed_autocast_factory
                or any(parameter.device != self.device for parameter in self.parameters)):
            raise ValueError("Chosen recovery requires managed precision and actual declared model device")
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        groups = self.optimizer.param_groups
        if (type(self.optimizer) is not torch.optim.AdamW or len(groups) != 1
                or len(groups[0]['params']) != len(self.parameters)
                or {id(p) for p in groups[0]['params']} != {id(p) for p in self.parameters}
                or any(id(p) not in names for p in groups[0]['params'])):
            raise ValueError("Chosen recovery requires exact single-group AdamW parameter coverage")
        return dpo.json_plain(dict(seed=self._sampler_seed, updates=self.updates,
            accumulation=self.accumulation, pad_id=self.pad_id, stop_ids=self.stop_ids,
            device=str(self.device), autocast_dtype=str(self.autocast_dtype), encoded_train_sha256=digest(self.train),
            sampler_origin_sha256=digest(self._initial_sampler_state),
            sample_work=self._sample_work(), model=dpo.effective_model_contract(self.model),
            policy_shapes=dpo.parameter_contract(self.model),
            optimizer_parameter_names=[names[id(p)] for p in groups[0]['params']],
            optimizer_shapes=[dict(shape=list(p.shape), dtype=str(p.dtype)) for p in groups[0]['params']],
            optimizer_group={key: value for key, value in groups[0].items() if key != 'params'},
            cpu_rng_bytes=torch.get_rng_state().numel(),
            cuda_rng_bytes=[value.numel() for value in torch.cuda.get_rng_state_all()]
                if self.device.type == 'cuda' else [],
            objective="mean valid chosen tokens per accumulation window; native full-input forward; one causal shift",
            **({'terminal_suffix_ids': list(self.terminal_suffix_ids)} if self.terminal_suffix_ids else {})))

    def bind_recovery(self, contract, *, work_ledger, io_budget, tokenizer):
        """Bind independently retained expectations before update or state load."""
        _validate_recovery_contract(contract)
        if self.completed or self.poisoned or self.history or self.recovery_contract is not None:
            raise ValueError("Bind a fresh chosen loop before updates or restore")
        if (contract['loop'] != self._recovery_layout()
                or contract['parent_sha256'] != digest(self.model.state_dict())
                or contract['interface_sha256'] != canonical_hash(tokenizer_interface(tokenizer,
                    template=tokenizer.chat_template, stop_ids=self.stop_ids))):
            raise ValueError("Actual parent/data/interface/model/optimizer differs from chosen recovery contract")
        _verify_recovery_files(contract)
        _verify_recovery_environment(contract)
        science = canonical_hash(contract)
        budget = contract['work_budget']
        if (not isinstance(work_ledger, WorkLedger) or not work_ledger.bound
                or work_ledger.contract_sha256 != science or work_ledger.limits != budget['limits']
                or work_ledger.max_bytes != budget['max_bytes']
                or not isinstance(io_budget, SnapshotIOBudget)
                or io_budget.science_sha256 != science or io_budget.contract != contract['snapshot_io_budget']):
            raise ValueError("Independently bound chosen work/I/O journals required")
        self.recovery_contract = deepcopy(contract)
        self._recovery_hash = science
        self.work_ledger, self.io_budget = work_ledger, io_budget
        self._recovery_tokenizer = tokenizer

    def _check_recovery(self, *, files=False):
        if self.recovery_contract is None:
            if self.work_ledger is not None or self.io_budget is not None:
                raise ValueError("Chosen accounting cannot be adopted without its recovery contract")
            return
        contract = self.recovery_contract
        if (canonical_hash(contract) != self._recovery_hash or self._recovery_layout() != contract['loop']
                or self.work_ledger.contract_sha256 != self._recovery_hash
                or self.work_ledger.limits != contract['work_budget']['limits']
                or self.work_ledger.max_bytes != contract['work_budget']['max_bytes']
                or self.io_budget.science_sha256 != self._recovery_hash
                or self.io_budget.contract != contract['snapshot_io_budget']
                or contract['interface_sha256'] != canonical_hash(tokenizer_interface(self._recovery_tokenizer,
                    template=self._recovery_tokenizer.chat_template, stop_ids=self.stop_ids))):
            raise ValueError("Bound chosen recovery contract/interface/live layout changed")
        if files:
            _verify_recovery_files(contract)
            _verify_recovery_environment(contract)

    def update_upper(self):
        """Admit the complete replacement-draw window BEFORE any active draw."""
        rows = self._sample_work()
        return dict(train_updates=1, sampled_examples=self.accumulation,
            sampler_draws=self.accumulation,
            valid_targets=self.accumulation*max(row['chosen_targets'] for row in rows),
            logical_sequence_tokens=self.accumulation*max(row['logical_sequence_tokens'] for row in rows),
            policy_forward_calls=self.accumulation,
            policy_forward_positions=self.accumulation*max(row['policy_forward_positions'] for row in rows))

    def completed_update(self):
        if self.poisoned:
            raise RuntimeError("Interrupted chosen-SFT update requires restart from the original parent")
        if self.completed >= self.updates:
            raise ValueError("Fixed chosen-SFT update horizon exhausted")
        self._check_recovery()
        sft, dpo = native_runners()
        accounting = dpo._WorkAttempt(self.work_ledger, self.update_upper(), 'chosen-sft-update')
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
                accounting.before(sampled_examples=1, sampler_draws=1)
                index = int(torch.randint(len(self.train), (), generator=self.sampler))
                accounting.after(sampled_examples=1, sampler_draws=1)
                attempted["indices"].append(index)
                row = chosen_record(self.train[index], f"draw-{self.completed}-{slot}-{index}")
                selected = dict(valid_targets=sum(self.train[index][0][1][1:]),
                                logical_sequence_tokens=len(row['ids']))
                accounting.before(**selected); accounting.after(**selected)
                window.append(sft.collate([row], self.pad_id, self.device))
            targets = sum(int((batch[1][:, 1:] != -100).sum()) for batch in window)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for batch in window:
                self.guard()
                positions = batch[0].numel()
                accounting.before(policy_forward_calls=1, policy_forward_positions=positions)
                attempted["policy_forward_calls_entered"] += 1
                attempted["policy_forward_positions_entered"] += positions
                with self.autocast_factory():
                    loss = sft.summed_loss(self.model, batch)
                attempted["policy_forward_calls_completed"] += 1
                attempted["policy_forward_positions_completed"] += positions
                accounting.after(policy_forward_calls=1, policy_forward_positions=positions)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite chosen-token NLL")
                (loss / targets).backward()
                total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
            self.guard()
            accounting.before(train_updates=1)
            self.optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters):
                raise RuntimeError("Nonfinite chosen-SFT updated policy")
            work = dict(sampled_pairs=self.accumulation, sampler_draws=self.accumulation,
                        chosen_targets=targets, rejected_targets=0,
                        logical_sequence_tokens=sum(batch[0].numel() for batch in window),
                        policy_forward_calls=self.accumulation, reference_forward_calls=0,
                        policy_forward_positions=sum(batch[0].numel() for batch in window),
                        reference_forward_positions=0)
            accounting.after(train_updates=1); accounting.complete()
            self.completed += 1
            row = dict(update=self.completed, loss=total_loss/targets, gradient_norm=float(norm),
                       indices=list(attempted["indices"]), work=work)
            self.history.append(deepcopy(row))
            attempted["completed"] = True
            self.poisoned = False
            return deepcopy(row)
        except BaseException as error:
            self.poisoned = True
            accounting.fail(error)
            attempted["error"] = dict(type=type(error).__name__, message=str(error)[:512],
                scope="known entered/successful forward geometry; unknown backend partial work is not invented")
            raise

    def snapshot_state(self):
        self._check_recovery(files=True)
        if self.recovery_contract is None or self.poisoned:
            raise RuntimeError("Chosen recovery saves only bound completed numerical state")
        return dict(schema='dongxi-chosen-sft-state-v1', model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(), sampler_rng=self.sampler.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            completed=self.completed, history=deepcopy(self.history),
            work_ledger=self.work_ledger.snapshot())

    def validate_payload(self, payload):
        """Reserve semantic replay before scanning values; never apply bad state."""
        self._check_recovery()
        if self.recovery_contract is None or type(payload) is not dict or type(payload.get('state')) is not dict:
            raise ValueError("Bound chosen completed payload/state required")
        state = payload['state']; completed = state.get('completed')
        _integer(completed, 'saved chosen cursor', 0, self.updates)
        if (type(payload.get('completed_updates')) is not int or payload['completed_updates'] != completed
                or payload.get('phase') != 'completed' or type(state.get('history')) is not list
                or len(state['history']) != completed or len(state) != 9):
            raise ValueError("Chosen completed state/history structure changed")
        self.work_ledger.validate_snapshot(state.get('work_ledger'))
        _, dpo = native_runners()
        costs = chosen_validation_upper(self.recovery_contract, completed)
        attempt = dpo._WorkAttempt(self.work_ledger, costs, 'chosen-sft-recovery-validation')
        try:
            attempt.before(**costs)
            self._validate_state(state)
            attempt.after(**costs); attempt.complete()
        except BaseException as error:
            attempt.fail(error); raise

    def _validate_state(self, state):
        layout = self.recovery_contract['loop']
        if (set(state) != {'schema', 'model', 'optimizer', 'sampler_rng', 'torch_rng',
                'cuda_rng', 'completed', 'history', 'work_ledger'}
                or state['schema'] != 'dongxi-chosen-sft-state-v1'):
            raise ValueError("Unknown chosen completed-state schema")
        completed = state['completed']
        tensors = state['model']
        if type(tensors) not in (dict, OrderedDict) or set(tensors) != set(layout['policy_shapes']):
            raise ValueError("Chosen model tensor keys changed")
        def tensor(value, shape, dtype):
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_quantized
                    or value.is_complex() or list(value.shape) != shape or str(value.dtype) != dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Chosen tensor shape/dtype/finite state changed")
        for name, value in tensors.items():
            expected = layout['policy_shapes'][name]
            tensor(value, expected['shape'], expected['dtype'])
        _, dpo = native_runners()
        saved = state['optimizer']
        if (type(saved) is not dict or set(saved) != {'state', 'param_groups'}
                or type(saved['state']) is not dict or type(saved['param_groups']) is not list
                or len(saved['param_groups']) != 1 or type(saved['param_groups'][0]) is not dict
                or len(saved['param_groups'][0]) > 32):
            raise ValueError("Chosen AdamW schema changed")
        group = saved['param_groups'][0]
        if not dpo.optimizer_metadata_equal({key: value for key, value in group.items() if key != 'params'},
                                           layout['optimizer_group']):
            raise ValueError("Chosen AdamW recipe changed")
        ids = list(range(len(layout['optimizer_shapes'])))
        if (type(group.get('params')) is not list or any(type(value) is not int for value in group['params'])
                or group['params'] != ids or any(type(key) is not int for key in saved['state'])
                or set(saved['state']) != (set(ids) if completed else set())):
            raise ValueError("Chosen AdamW named-order/moment coverage changed")
        for index, moment in saved['state'].items():
            if type(moment) is not dict or set(moment) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError("Chosen AdamW moment schema changed")
            step = moment['step']
            tensor(step, [], 'torch.float32')
            if float(step) != completed:
                raise ValueError("Chosen AdamW step differs from completed cursor")
            for key in ('exp_avg', 'exp_avg_sq'):
                shape = layout['optimizer_shapes'][index]
                tensor(moment[key], shape['shape'], shape['dtype'])
            if bool((moment['exp_avg_sq'] < 0).any()):
                raise ValueError("Chosen AdamW second moment must be nonnegative")
        replay = torch.Generator().manual_seed(layout['seed'])
        numerical_work = dpo.zero_work()
        for number, row in enumerate(state['history'], 1):
            if (type(row) is not dict or set(row) != {'update', 'loss', 'gradient_norm', 'indices', 'work'}
                    or type(row['update']) is not int or row['update'] != number
                    or type(row['indices']) is not list or len(row['indices']) != self.accumulation
                    or any(type(value) is not int for value in row['indices'])):
                raise ValueError("Chosen history/cursor structure changed")
            expected = [int(torch.randint(len(self.train), (), generator=replay)) for _ in range(self.accumulation)]
            work = dpo.zero_work()
            for index in expected:
                dpo.add_work(work, layout['sample_work'][index])
            if (row['indices'] != expected or type(row['work']) is not dict
                    or set(row['work']) != set(dpo.WORK_KEYS)
                    or any(type(value) is not int for value in row['work'].values()) or row['work'] != work):
                raise ValueError("Chosen history draw/target/forward geometry changed")
            dpo.add_work(numerical_work, work)
            if any(type(row[key]) is not float or not math.isfinite(row[key]) or row[key] < 0
                   for key in ('loss', 'gradient_norm')):
                raise ValueError("Chosen metrics must be finite nonnegative scalars")
        # Numerical history can rewind; physical spending cannot. Its retained
        # prefix must nevertheless cover every claimed completed draw/forward.
        covered = state['work_ledger']['completed']
        required = dict(train_updates=completed, sampled_examples=numerical_work['sampled_pairs'],
            sampler_draws=numerical_work['sampler_draws'], valid_targets=numerical_work['chosen_targets'],
            logical_sequence_tokens=numerical_work['logical_sequence_tokens'],
            policy_forward_calls=numerical_work['policy_forward_calls'],
            policy_forward_positions=numerical_work['policy_forward_positions'])
        if any(covered[key] < value for key, value in required.items()):
            raise ValueError("Retained chosen work prefix does not cover numerical history")
        for key in ('sampler_rng', 'torch_rng'):
            tensor(state[key], [layout['cpu_rng_bytes']], 'torch.uint8')
            torch.Generator().set_state(state[key].cpu())
        if not torch.equal(state['sampler_rng'].cpu(), replay.get_state()):
            raise ValueError("Chosen sampler RNG differs from completed replacement-draw cursor")
        if type(state['cuda_rng']) is not list or len(state['cuda_rng']) != len(layout['cuda_rng_bytes']):
            raise ValueError("Chosen CUDA RNG device coverage changed")
        for index, value in enumerate(state['cuda_rng']):
            tensor(value, [layout['cuda_rng_bytes'][index]], 'torch.uint8')
            torch.Generator(device=f'cuda:{index}').set_state(value.cpu())

    def save(self, path, *, parent_invocation):
        """Exclusive shared save, followed by an independently retained receipt."""
        state = self.snapshot_state()
        self.validate_payload(dict(state=state, phase='completed', completed_updates=self.completed))
        state['work_ledger'] = self.work_ledger.snapshot()
        return save_snapshot(path, contract=self.recovery_contract, state=state,
            completed_updates=self.completed, parent_invocation=parent_invocation,
            max_bytes=self.io_budget.contract['envelope']['max_payload_bytes'], guard=self.guard,
            io_budget=self.io_budget, work_receipt_path=Path(str(path)+'.work.json'))

    def restore(self, path, *, expected_sha256, expected_bytes):
        """Restore after byte, receipt, both-prefix and semantic checks complete."""
        self._check_recovery(files=True)
        if self.recovery_contract is None:
            raise ValueError("Chosen restore requires bound independently retained contracts")
        payload = load_snapshot(path, expected_sha256=expected_sha256, expected_bytes=expected_bytes,
            expected_contract=self.recovery_contract,
            max_bytes=self.io_budget.contract['envelope']['max_payload_bytes'],
            validate_payload=self.validate_payload, guard=self.guard, io_budget=self.io_budget)
        state = payload['state']
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.sampler.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.optimizer.zero_grad(set_to_none=True)
        self.completed, self.history = state['completed'], deepcopy(state['history'])
        self.poisoned = False
        return payload


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
                autocast_factory=autocast_factory,
                terminal_suffix_ids=dataset.get('terminal_suffix_ids', []))
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

"""Actual local-random RLVR reader microscope, not a model-scale launcher."""
from copy import deepcopy
import hashlib
from pathlib import Path
import platform
import random
import tempfile

import torch

from . import qwen_rlvr_lab as runner
from .batched_cache_lab import digest
from .snapshot_io_budget import IO_KEYS, io_budget_contract, read_work_receipt
from .training_snapshot import inspect_snapshot, load_snapshot


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_BOUND = 16 * 1024**2
TRAIN = [dict(source_id='cpu-source-0', prompt='Authored integer 0', prompt_ids=[1, 2], expected=0),
         dict(source_id='cpu-source-1', prompt='Authored integer 1', prompt_ids=[1, 3, 2], expected=1),
         dict(source_id='cpu-source-2', prompt='Authored integer 2', prompt_ids=[1, 4, 3, 2], expected=2)]
EVALUATION = [dict(source_id='cpu-heldout-0', prompt='Authored heldout 4', prompt_ids=[1, 5, 2], expected=4)]


def _decode(ids):
    return ''.join(str(i-8) if 8 <= i <= 15 else '?' for i in ids)


def lesson_io_contract():
    """Explicit predeclared tiny-CPU capacities, never production defaults."""
    limits = dict(zip(IO_KEYS, (16, 16, 32, 1024**3, 6000000, 48000000,
                               64*1024**2, 256*1024**2, 512*1024**2)))
    envelope = dict(max_payload_bytes=PAYLOAD_BOUND, max_tree_nodes=100000,
                    max_tensor_elements=1000000, max_tensor_bytes=8*1024**2,
                    max_primitive_bytes=1024**2)
    return io_budget_contract(limits, envelope, 2*1024**2)


def _fixture(seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM, __version__
    random.seed(seed); torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=0, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).eval()
    reference = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=0.)
    work = runner.work_budget_contract(dict.fromkeys(runner.BUDGET_KEYS, 100000), 1024**2)
    io = lesson_io_contract()
    loop = runner.RLVRLoop(model, reference, optimizer, torch.Generator().manual_seed(seed),
        TRAIN, _decode, eos_id=0, stop_ids=[0, 7], group_size=3, max_new_tokens=4,
        updates=4, seed=seed, context_length=32, beta=.02, work_budget=work)
    paths = ['src/dongxi_llms/qwen_rlvr_lab.py', 'src/dongxi_llms/grpo_lab.py',
             'src/dongxi_llms/batched_cache_lab.py', 'src/dongxi_llms/training_snapshot.py',
             'src/dongxi_llms/work_budget.py', 'src/dongxi_llms/snapshot_io_budget.py',
             'src/dongxi_llms/rlvr_snapshot_io_lab.py']
    observed = runner.observable_contract(
        parent_files={'authored-random-initial-policy': digest(model.state_dict())},
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        inputs={'authored-token-fixture':digest((TRAIN, EVALUATION))},
        environment={'python':platform.python_version(), 'torch':str(torch.__version__),
            'transformers':__version__, 'lock_sha256':hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest()},
        interface={'source':{'tokenizer_id':'authored-integer-ID-alphabet', 'tokenizer_revision':'1'*40},
            'encoding':'explicit original token sequences; no downloaded tokenizer', 'stops':[0, 7]},
        train=TRAIN, evaluation=EVALUATION, seed=seed, updates=4, group_size=3,
        max_new_tokens=4, lr=.008, beta=.02, eos_id=0, stop_ids=[0, 7], context_length=32,
        dtype='torch.float32', device={'mode':'cpu', 'name':'CPU'}, work_budget=work,
        snapshot_io_contract=io)
    return loop, runner.full_contract(observed, loop), work, io


def _open(root, loop, contract, work, io, receipt=None, header=None, invocation='lesson-A'):
    accounts = runner.open_snapshot_budgets(contract=contract, budget=work, io_contract=io,
        receipt=receipt, work_path=root/'model-work.jsonl', io_path=root/'io-work.jsonl',
        invocation_id=invocation, expected_sha256=None if header is None else header['payload_sha256'],
        expected_bytes=None if header is None else header['payload_bytes'])
    loop.bind_work_ledger(accounts[0], contract)
    return accounts


def _numerical(loop):
    state = deepcopy(loop.snapshot_state())
    state.pop('work_ledger')
    return state


def _arm(root, phase, seed, expected, expected_history):
    loop, contract, work, io = _fixture(seed)
    model_work, io_work, hook = _open(root, loop, contract, work, io)
    try:
        loop.apply_pending(); loop.apply_pending()
        if phase == 'pending': loop.collect()
        saved = deepcopy(loop.snapshot_state())
        snapshot = root/'boundary.pt'; receipt_path = root/'retained.work.json'
        header = loop.save(snapshot, contract=contract, parent_invocation='lesson-A',
            max_bytes=PAYLOAD_BOUND, io_budget=hook, work_receipt_path=receipt_path)
        receipt = read_work_receipt(receipt_path); hook.bind_receipt(receipt)
        def reject(_): raise ValueError('declared rejecting callback; no state application')
        try:
            load_snapshot(snapshot, expected_sha256=header['payload_sha256'],
                expected_bytes=header['payload_bytes'], expected_contract=contract,
                max_bytes=PAYLOAD_BOUND, io_budget=hook, validate_payload=reject)
        except ValueError as error: failure = str(error)
        else: raise AssertionError('The deliberately rejecting callback must refuse')
        if 'declared rejecting callback' not in failure: raise AssertionError(failure)
        model_before = model_work.snapshot(); io_before = io_work.snapshot()
    finally:
        io_work.close(); model_work.close()
    restored, restored_contract, restored_work, restored_io = _fixture(seed)
    if restored_contract != contract: raise AssertionError('Frozen lesson science changed')
    model_work, io_work, hook = _open(root, restored, contract, restored_work, restored_io,
        receipt, header, invocation='lesson-B')
    try:
        retained = model_before == model_work.snapshot() and io_before == io_work.snapshot()
        arguments = dict(expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'],
                         max_bytes=PAYLOAD_BOUND, io_budget=hook)
        measured = inspect_snapshot(snapshot, expected_contract=contract, **arguments)
        restored.restore(snapshot, contract=contract, **arguments)
        restored_boundary = deepcopy(restored.snapshot_state())
        pending_equal = digest(saved['pending']) == digest(restored.pending)
        before_collections = model_work.snapshot()['reserved']['collections']
        rng_before = restored.generator.get_state().clone()
        first_record = restored.apply_pending()
        after_collections = model_work.snapshot()['reserved']['collections']
        sampling_advanced = not torch.equal(rng_before, restored.generator.get_state())
        while restored.completed < restored.updates: restored.apply_pending()
        exact = digest(_numerical(restored)) == expected
        first_exact = digest(first_record) == digest(expected_history[2])
        if not retained or not exact or not first_exact or not pending_equal:
            raise AssertionError('Native lesson recovery must preserve its original trajectory')
        if (after_collections-before_collections) != int(phase == 'completed'):
            raise AssertionError('Pending must apply without new collection')
        return dict(seed=seed, phase=phase, completed_updates=header['completed_updates'],
            source_cursor=saved['cursor'], pending_present=saved['pending'] is not None,
            pending_ids=None if saved['pending'] is None else saved['pending']['responses'].tolist(),
            pending_ids_equal=pending_equal, exact_recovery=exact, first_record_equal=first_exact,
            later_work_retained=retained, first_action='collect then apply' if phase=='completed' else 'apply retained pool',
            new_collections_at_first_step=after_collections-before_collections,
            sampling_rng_advanced_at_first_step=sampling_advanced,
            snapshot_bytes=measured['payload_bytes'],
            saved_runner_prefix_sequence=receipt['runner_work_prefix']['sequence'],
            saved_io_prefix_sequence=receipt['io_prefix']['sequence'],
            runner_sequence_after_save_and_rejection=model_before['sequence'],
            io_sequence_after_save_and_rejection=io_before['sequence'],
            restored_completed_updates=len(restored_boundary['history']),
            final_history=runner.jsonable(restored.history), final_numerical_sha256=expected,
            io_reserved=io_work.snapshot()['reserved'], io_completed=io_work.snapshot()['completed'],
            failed_io_tickets=io_work.snapshot()['failed_tickets'],
            work_reserved=model_work.snapshot()['reserved'], work_completed=model_work.snapshot()['completed'],
            deliberate_failure=failure)
    finally:
        io_work.close(); model_work.close()


def phase_recovery_example(seed=2323):
    """Two actual native boundaries; owned temp files and global RNG restored.

    This controlled lesson saves one boundary, not the complete runner lifecycle.
    It measures logical operations, not wall-time, physical quota or language skill.
    """
    if type(seed) is not int or seed not in (2323, 2324):
        raise ValueError('Use one of the two predeclared original CPU fixture seeds')
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix='dongxi-rlvr-reader-lesson-') as directory:
            root = Path(directory); baseline = root/'baseline'; baseline.mkdir(mode=0o700)
            loop, contract, work, io = _fixture(seed)
            model_work, io_work, _ = _open(baseline, loop, contract, work, io)
            try:
                while loop.completed < loop.updates: loop.apply_pending()
                expected = digest(_numerical(loop)); expected_history = deepcopy(loop.history)
            finally:
                io_work.close(); model_work.close()
            rows = []
            for phase in ('completed', 'pending'):
                path = root/phase; path.mkdir(mode=0o700)
                rows.append(_arm(path, phase, seed, expected, expected_history))
            return dict(scope='actual local-random native RLVR CPU; manually published boundary, not model-scale lifecycle',
                        seed=seed, model_work_keys=len(runner.BUDGET_KEYS), io_work_keys=len(IO_KEYS),
                        envelope=io['envelope'], rows=rows)
    finally:
        random.setstate(python_state)

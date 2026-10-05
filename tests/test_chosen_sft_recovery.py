"""Bounded original local-Qwen chosen-control replay; no pretrained/GPU/network."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from test_chosen_sft_control import fixture, save_parent, fixture_rows
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.chosen_sft_control import (ChosenSFTLoop, RECOVERY_SOURCES,
    chosen_work_budget_contract, make_chosen_recovery_contract,
    open_chosen_resume_budgets, native_runners, encode_dataset)
from dongxi_llms.run_identity import artifact_hashes, canonical_hash, environment_identity, file_digest
from dongxi_llms.snapshot_io_budget import (IO_KEYS, SnapshotIOBudget, io_budget_contract,
    io_ledger_contract_sha256, read_work_receipt)
from dongxi_llms.work_budget import WorkLedger, WorkBudgetExceeded, WorkLedgerError

ROOT = Path(__file__).resolve().parents[1]
MiB = 1024**2


def exclusive_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def limits():
    _, dpo = native_runners()
    return dict.fromkeys(dpo.BUDGET_KEYS, 100_000_000)


def io_contract():
    caps = dict.fromkeys(IO_KEYS, 100_000_000)
    caps.update(snapshot_inspect_operations=20, snapshot_load_operations=20, snapshot_save_operations=20)
    return io_budget_contract(caps, dict(max_payload_bytes=MiB, max_tree_nodes=20_000,
        max_tensor_elements=200_000, max_tensor_bytes=MiB//2, max_primitive_bytes=MiB//2), MiB)


def loop_for(parent, data, settings):
    model = deepcopy(parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.01)
    return ChosenSFTLoop(model, optimizer, torch.Generator().manual_seed(settings['seed']), data['train'],
        pad_id=data['pad_id'], stop_ids=data['stop_ids'], accumulation=settings['accumulation'],
        updates=settings['updates'])


def observed_contract(loop, tokenizer, data, settings, parent_path, caps=None):
    return make_chosen_recovery_contract(loop, dataset=data, tokenizer=tokenizer, settings=settings,
        parent_files={str(parent_path/name): sha for name, sha in artifact_hashes(parent_path).items()},
        inputs={name: file_digest(ROOT/name) for name in (
            'fixtures/chapter11/train.jsonl', 'fixtures/chapter11/validation.jsonl',
            'fixtures/chapter11/evaluation.jsonl', 'fixtures/matched-chosen-sft/protocol.json')},
        sources={name: file_digest(ROOT/name) for name in RECOVERY_SOURCES},
        environment=environment_identity(ROOT/'uv.lock'),
        work_budget=chosen_work_budget_contract(caps or limits(), MiB), snapshot_io_budget=io_contract())


def create_binding(loop, tokenizer, contract, root):
    science = canonical_hash(contract); work = io = None
    try:
        work = WorkLedger.create(root/'work.jsonl', limits=contract['work_budget']['limits'],
            contract_sha256=science, max_bytes=MiB, invocation_id='initial')
        io = WorkLedger.create(root/'io.jsonl', limits=contract['snapshot_io_budget']['limits'],
            contract_sha256=io_ledger_contract_sha256(contract['snapshot_io_budget'], science),
            max_bytes=MiB, invocation_id='initial')
        hook = SnapshotIOBudget(io, contract=contract['snapshot_io_budget'], scientific_contract_sha256=science)
        loop.bind_recovery(contract, work_ledger=work, io_budget=hook, tokenizer=tokenizer)
        return work, io, hook
    except BaseException:
        if work is not None: work.close()
        if io is not None: io.close()
        raise


def setup(root, *, seed=1818, updates=6, caps=None):
    parent, tokenizer, data, settings = fixture(seed, updates)
    parent_path = root/'parent'
    save_parent(parent_path, parent, tokenizer)
    loop = loop_for(parent, data, settings)
    contract = observed_contract(loop, tokenizer, data, settings, parent_path, caps)
    work, io, hook = create_binding(loop, tokenizer, contract, root)
    return parent, tokenizer, data, settings, loop, contract, work, io, hook


def numerical(loop):
    clone = torch.Generator().set_state(loop.sampler.get_state())
    return dict(completed=loop.completed, model=digest(loop.model.state_dict()),
        optimizer=digest(loop.optimizer.state_dict()), sampler=digest(loop.sampler.get_state()),
        torch_rng=digest(torch.get_rng_state()), history=deepcopy(loop.history),
        next_indices=[int(torch.randint(len(loop.train), (), generator=clone)) for _ in range(loop.accumulation)])


def fail_later(loop):
    original = loop.model.forward; calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('authored later second-forward failure')
        return original(*args, **kwargs)
    with patch.object(loop.model, 'forward', side_effect=failure):
        try:
            loop.completed_update()
        except RuntimeError as error:
            if 'authored later' not in str(error):
                raise
        else:
            raise AssertionError('Expected later failure not observed')
    return deepcopy(loop.attempts[-1])


def child_resume(root):
    """Actual fresh CPU process; use only retained local parent and two journals."""
    torch.set_num_threads(1)
    packet = json.loads((root/'packet.json').read_text())
    contract = packet['contract']
    parent = Qwen3ForCausalLM.from_pretrained(root/'parent', local_files_only=True)
    parent.config._attn_implementation = 'sdpa'
    tokenizer = AutoTokenizer.from_pretrained(root/'parent', local_files_only=True)
    _, splits = fixture_rows()
    data = encode_dataset(tokenizer, *splits, max_length=64)
    loop = loop_for(parent, data, contract['recipe'])
    if observed_contract(loop, tokenizer, data, contract['recipe'], root/'parent') != contract:
        raise AssertionError('Fresh actual observations differ from retained contract')
    work, io, hook = open_chosen_resume_budgets(contract=contract,
        receipt=packet['independent_receipt'], work_path=root/'work.jsonl', io_path=root/'io.jsonl',
        invocation_id='actual-fresh-child', expected_sha256=packet['header']['payload_sha256'],
        expected_bytes=packet['header']['payload_bytes'])
    try:
        before_load = work.snapshot()
        loop.bind_recovery(contract, work_ledger=work, io_budget=hook, tokenizer=tokenizer)
        loop.restore(root/'completed-2.pt', expected_sha256=packet['header']['payload_sha256'],
                     expected_bytes=packet['header']['payload_bytes'])
        while loop.completed < loop.updates:
            loop.completed_update()
        outcome = numerical(loop)
        if outcome != packet['uninterrupted']:
            raise AssertionError('Fresh restored numerical state/history/next IDs differ')
        final = work.snapshot()
        failed = [value for value in work.tickets.values() if value['status'] == 'fail']
        if len(failed) != 1 or final['reserved']['train_updates'] != 7 or final['completed']['train_updates'] != 6:
            raise AssertionError('Later failed capacity was refunded or completed cursor changed')
        result = dict(status='completed-exact-replay', seed=contract['recipe']['seed'],
            restored_update=2, final_update=6, numerical=outcome,
            before_load=before_load, final_work=final, final_io=io.snapshot(),
            failed_tickets=failed, scope='actual local random tiny CPU model; not pretrained/CUDA/hosted')
        exclusive_json(root/'child-result.json', result)
    finally:
        work.close(); io.close()
    return result


def collect_replay(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    for seed in (1818, 1819):
        run = root/f'seed-{seed}'; run.mkdir(mode=0o700)
        parent, tokenizer, data, settings, split, contract, work, io, hook = setup(run, seed=seed)
        try:
            uninterrupted = loop_for(parent, data, settings)
            torch.manual_seed(seed+7000)
            for _ in range(settings['updates']):
                uninterrupted.completed_update()
            expected = numerical(uninterrupted)
            torch.manual_seed(seed+7000)
            for _ in range(2): split.completed_update()
            header = split.save(run/'completed-2.pt', parent_invocation='initial')
            receipt = deepcopy(hook.last_receipt)
            if receipt != read_work_receipt(run/'completed-2.pt.work.json'):
                raise AssertionError('Independent returned receipt differs from retained side artifact')
            failure = fail_later(split)
            if (failure['completed'] or failure['policy_forward_calls_entered'] != 2
                    or failure['policy_forward_calls_completed'] != 1):
                raise AssertionError('Partial entered/completed geometry lost')
            packet = dict(contract=contract, header=header, independent_receipt=receipt,
                uninterrupted=expected, later_failure=failure, after_failure_work=work.snapshot(),
                after_save_io=io.snapshot())
            exclusive_json(run/'packet.json', packet)
        finally:
            work.close(); io.close()
        started = time.monotonic()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--resume-child', str(run)],
            cwd=ROOT, capture_output=True, timeout=60,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'tests'),
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        # Retain raw output even when the child fails; never overwrite an attempt.
        with (run/'child.stdout').open('xb') as handle: handle.write(child.stdout)
        with (run/'child.stderr').open('xb') as handle: handle.write(child.stderr)
        record = dict(seed=seed, exit_code=child.returncode, elapsed_seconds=time.monotonic()-started,
            command=child.args, stdout_sha256=file_digest(run/'child.stdout'),
            stderr_sha256=file_digest(run/'child.stderr'), parent_sha256=contract['parent_sha256'])
        exclusive_json(run/'process.json', record)
        if child.returncode != 0:
            raise AssertionError('Fresh child failed; retained logs: '+str(run))
        result = json.loads((run/'child-result.json').read_text())
        rows.append(dict(process=record, result=result))
    exclusive_json(root/'replay.json', dict(schema='dongxi-chosen-sft-recovery-verification-v1', arms=rows))
    return rows


class ChosenRecoveryTests(unittest.TestCase):
    def test_audited_suffix_bound_into_recovery_contract_and_live_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);parent, tokenizer, _, settings=fixture(1818, 2)
            _, splits=fixture_rows()
            tokenizer.chat_template=tokenizer.chat_template.replace("' <END> '", "' <END> <USER> '")
            data=encode_dataset(tokenizer,*splits,max_length=64,terminal_suffix_ids=[2])
            parent_path=root/'parent';save_parent(parent_path,parent,tokenizer)
            model=deepcopy(parent);optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.01)
            loop=ChosenSFTLoop(model,optimizer,torch.Generator().manual_seed(1818),data['train'],
                pad_id=data['pad_id'],stop_ids=data['stop_ids'],accumulation=settings['accumulation'],updates=2,
                terminal_suffix_ids=[2])
            contract=observed_contract(loop,tokenizer,data,settings,parent_path)
            self.assertEqual(contract['loop']['terminal_suffix_ids'],[2])
            self.assertEqual(contract['dataset']['terminal_suffix_ids'],[2])
            self.assertEqual(contract['interface_sha256'],data['interface_sha256'])
            work,io,_=create_binding(loop,tokenizer,contract,root)
            try:
                row=loop.completed_update();loop.save(root/'one.pt',parent_invocation='authored-suffix')
                self.assertEqual(row['work']['chosen_targets'],sum(sum(data['train'][i][0][1][1:]) for i in row['indices']))
                before=digest(model.state_dict());draw=loop.sampler.get_state().clone()
                loop.terminal_suffix_ids=[3]
                with self.assertRaisesRegex(ValueError,'layout changed'):loop.completed_update()
                self.assertEqual(digest(model.state_dict()),before);self.assertTrue(torch.equal(draw,loop.sampler.get_state()))
            finally:io.close();work.close()
            model2=deepcopy(parent)
            fresh=ChosenSFTLoop(model2,torch.optim.AdamW(model2.parameters(),lr=settings['learning_rate'],weight_decay=.01),
                torch.Generator().manual_seed(1818),data['train'],pad_id=data['pad_id'],stop_ids=data['stop_ids'],
                accumulation=settings['accumulation'],updates=2,terminal_suffix_ids=[2])
            receipt=read_work_receipt(str(root/'one.pt')+'.work.json');header=receipt['snapshot']
            resumed_work,resumed_io,hook=open_chosen_resume_budgets(contract=contract,receipt=receipt,
                work_path=root/'work.jsonl',io_path=root/'io.jsonl',invocation_id='authored-suffix-fresh',
                expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes'])
            try:
                fresh.bind_recovery(contract,work_ledger=resumed_work,io_budget=hook,tokenizer=tokenizer)
                fresh.restore(root/'one.pt',expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes'])
                self.assertEqual(fresh.completed,1);self.assertEqual(fresh.history,[row])
                self.assertEqual(digest(fresh.model.state_dict()),before)
                changed=deepcopy(contract);changed['loop']['terminal_suffix_ids']=[3]
                self.assertNotEqual(canonical_hash(changed),canonical_hash(contract))
            finally:resumed_io.close();resumed_work.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dongxi-chosen-recovery-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def configured(self, **kwargs):
        result = setup(self.root, **kwargs)
        self.addCleanup(result[-2].close); self.addCleanup(result[-3].close)
        return result

    def test_whole_window_cap_refuses_before_draw_or_forward(self):
        caps = limits(); caps['policy_forward_calls'] = 1
        *_, loop, contract, work, io, hook = self.configured(caps=caps)
        before = digest(loop.sampler.get_state())
        with patch.object(loop.model, 'forward', side_effect=AssertionError('must not execute')):
            with self.assertRaises(WorkBudgetExceeded): loop.completed_update()
        self.assertEqual(digest(loop.sampler.get_state()), before)
        self.assertEqual(work.snapshot()['sequence'], 0)
        self.assertFalse(loop.poisoned)
        self.assertEqual(loop.attempts, [])

    def test_original_update_parity_and_full_chosen_geometry(self):
        parent, tokenizer, data, settings, actual, contract, work, io, hook = self.configured()
        expected = loop_for(parent, data, settings)
        self.assertEqual(actual.completed_update(), expected.completed_update())
        self.assertEqual(digest(actual.model.state_dict()), digest(expected.model.state_dict()))
        complete = work.snapshot()['completed']
        row = actual.history[0]
        self.assertEqual(complete['valid_targets'], row['work']['chosen_targets'])
        self.assertEqual(complete['policy_forward_positions'], row['work']['logical_sequence_tokens'])
        self.assertEqual(complete['reference_forward_calls'], 0)

    def test_partial_failure_poisoning_save_refusal_and_retained_costs(self):
        *_, loop, contract, work, io, hook = self.configured()
        row = fail_later(loop)
        summary = work.snapshot()
        self.assertEqual(summary['reserved']['train_updates'], 1)
        self.assertEqual(summary['completed']['train_updates'], 0)
        self.assertEqual(summary['known_partial']['policy_forward_calls'], 1)
        self.assertEqual(summary['attempted_upper']['policy_forward_calls'], 2)
        self.assertTrue(loop.poisoned)
        with self.assertRaises(RuntimeError): loop.save(self.root/'forbidden.pt', parent_invocation='bad')
        self.assertFalse((self.root/'forbidden.pt').exists())

    def test_reservation_fsync_failure_precedes_draw_and_backend(self):
        *_, loop, contract, work, io, hook = self.configured()
        before = numerical(loop)
        with patch('dongxi_llms.work_budget.os.fsync', side_effect=OSError('authored reservation fsync failure')):
            with patch.object(loop.model, 'forward', side_effect=AssertionError('must not execute')):
                with self.assertRaises(WorkLedgerError): loop.completed_update()
        self.assertEqual(numerical(loop), before)
        self.assertTrue(work.poisoned)
        self.assertTrue((self.root/'work.jsonl').stat().st_size > 0)

    def test_optimizer_applied_then_failure_restores_without_refunding(self):
        *_, loop, contract, work, io, hook = self.configured()
        before = numerical(loop)
        header = loop.save(self.root/'zero.pt', parent_invocation='initial')
        receipt = deepcopy(hook.last_receipt)
        original = loop.optimizer.step
        def after_apply(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('authored post-optimizer failure')
        with patch.object(loop.optimizer, 'step', side_effect=after_apply):
            with self.assertRaisesRegex(RuntimeError, 'post-optimizer'): loop.completed_update()
        self.assertTrue(loop.poisoned)
        self.assertEqual(loop.completed, 0)
        self.assertNotEqual(digest(loop.model.state_dict()), before['model'])
        hook.bind_receipt(receipt)
        loop.restore(self.root/'zero.pt', expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'])
        self.assertEqual(numerical(loop), before)
        self.assertEqual(work.snapshot()['reserved']['train_updates'], 1)
        self.assertEqual(len(work.snapshot()['failed_tickets']), 1)

    def test_failed_save_retains_prior_snapshot_and_io_reservation(self):
        *_, loop, contract, work, io, hook = self.configured()
        loop.completed_update()
        loop.save(self.root/'good.pt', parent_invocation='initial')
        prior = file_digest(self.root/'good.pt')
        with patch('dongxi_llms.training_snapshot.torch.save', side_effect=RuntimeError('authored serializer failure')):
            with self.assertRaisesRegex(RuntimeError, 'serializer failure'):
                loop.save(self.root/'failed.pt', parent_invocation='failed-save')
        self.assertEqual(file_digest(self.root/'good.pt'), prior)
        self.assertEqual(io.snapshot()['reserved']['snapshot_save_operations'], 2)
        self.assertEqual(io.snapshot()['completed']['snapshot_save_operations'], 1)
        self.assertEqual(len(io.snapshot()['failed_tickets']), 1)

    def test_retained_work_prefix_must_cover_claimed_numerical_history(self):
        *_, loop, contract, work, io, hook = self.configured()
        loop.completed_update()
        state = loop.snapshot_state()
        state['work_ledger'] = deepcopy(work.prefixes[0])
        with self.assertRaisesRegex(ValueError, 'does not cover numerical history'):
            loop.validate_payload(dict(state=state, phase='completed', completed_updates=1))

    def test_old_boundary_keeps_later_failed_cap_and_refuses_retry(self):
        caps = limits(); caps['train_updates'] = 3
        *_, loop, contract, work, io, hook = self.configured(caps=caps)
        loop.completed_update(); loop.completed_update()
        header = loop.save(self.root/'two.pt', parent_invocation='initial')
        receipt = deepcopy(hook.last_receipt)
        fail_later(loop)
        hook.bind_receipt(receipt)
        loop.restore(self.root/'two.pt', expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'])
        before = numerical(loop)
        with patch.object(loop.model, 'forward', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(WorkBudgetExceeded, 'train_updates'):
                loop.completed_update()
        self.assertEqual(numerical(loop), before)
        self.assertEqual(loop.completed, 2)
        self.assertEqual(work.snapshot()['reserved']['train_updates'], 3)

    def test_malformed_saved_history_rng_moments_refuse_before_application(self):
        *_, loop, contract, work, io, hook = self.configured()
        loop.completed_update()
        state = loop.snapshot_state()
        mutations = []
        bad = deepcopy(state); bad['history'][0]['indices'][0] = True; mutations.append(bad)
        bad = deepcopy(state); bad['history'][0]['work']['chosen_targets'] += 1; mutations.append(bad)
        bad = deepcopy(state); bad['sampler_rng'] = torch.Generator().manual_seed(999).get_state(); mutations.append(bad)
        bad = deepcopy(state); next(iter(bad['optimizer']['state'].values()))['step'] += 1; mutations.append(bad)
        bad = deepcopy(state); next(iter(bad['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1); mutations.append(bad)
        bad = deepcopy(state); bad['optimizer']['param_groups'][0]['lr'] *= 2; mutations.append(bad)
        bad = deepcopy(state); bad['completed'] = True; mutations.append(bad)
        before = numerical(loop)
        for index, bad in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises((ValueError, RuntimeError)):
                loop.validate_payload(dict(state=bad, phase='completed', completed_updates=1))
        self.assertEqual(numerical(loop), before)
        self.assertEqual(len(work.snapshot()['failed_tickets']), 6)

    def test_live_encoded_interface_optimizer_and_model_changes_refuse(self):
        *_, loop, contract, work, io, hook = self.configured()
        cases = [('mask', lambda: loop.train[0][0][1].__setitem__(-2, False)),
                 ('interface', lambda: setattr(loop._recovery_tokenizer, 'chat_template', 'changed')),
                 ('optimizer', lambda: loop.optimizer.param_groups[0]['params'].reverse()),
                 ('model', lambda: setattr(loop.model.config, 'attention_dropout', .1)),
                 ('precision', lambda: setattr(loop, 'autocast_dtype', torch.bfloat16))]
        for name, mutate in cases:
            with self.subTest(name=name):
                original_train = deepcopy(loop.train)
                original_template = loop._recovery_tokenizer.chat_template
                original_params = list(loop.optimizer.param_groups[0]['params'])
                mutate()
                with patch.object(loop.model, 'forward', side_effect=AssertionError('must not execute')):
                    with self.assertRaises(ValueError): loop.completed_update()
                loop.train = original_train
                loop._recovery_tokenizer.chat_template = original_template
                loop.optimizer.param_groups[0]['params'] = original_params
                loop.model.config.attention_dropout = 0.
                loop.autocast_dtype = None

    def test_changed_source_identity_refuses_save(self):
        *_, loop, contract, work, io, hook = self.configured()
        # This is a controlled expected-hash mismatch; no project source is changed.
        with patch('dongxi_llms.chosen_sft_control.file_digest', return_value='f'*64):
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                loop.save(self.root/'not-written.pt', parent_invocation='bad-source')
        self.assertFalse((self.root/'not-written.pt').exists())

    def test_observed_environment_mismatch_refuses_before_payload(self):
        *_, loop, contract, work, io, hook = self.configured()
        observed = environment_identity(ROOT/'uv.lock')
        observed['packages']['torch'] = 'authored-wrong-version'
        with patch('dongxi_llms.chosen_sft_control.environment_identity', return_value=observed):
            with self.assertRaisesRegex(ValueError, 'numerical environment'):
                loop.save(self.root/'bad-environment.pt', parent_invocation='bad-env')
        self.assertFalse((self.root/'bad-environment.pt').exists())

    def test_bad_receipt_and_copied_journal_refuse_before_payload(self):
        parent, tokenizer, data, settings, loop, contract, work, io, hook = self.configured()
        loop.completed_update()
        header = loop.save(self.root/'saved.pt', parent_invocation='initial')
        receipt = deepcopy(hook.last_receipt)
        work.close(); io.close()
        kwargs = dict(contract=contract, receipt=receipt, work_path=self.root/'work.jsonl',
            io_path=self.root/'io.jsonl', invocation_id='resume',
            expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'])
        with self.assertRaisesRegex(ValueError, 'receipt/payload'):
            open_chosen_resume_budgets(**dict(kwargs, expected_sha256='f'*64))
        shutil.copyfile(self.root/'work.jsonl', self.root/'copy.jsonl')
        (self.root/'copy.jsonl').chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'Copied/replaced'):
            open_chosen_resume_budgets(**dict(kwargs, work_path=self.root/'copy.jsonl'))

    def test_actual_fresh_process_replay_and_later_failed_spending(self):
        rows = collect_replay(self.root/'actual-fresh-replay')
        self.assertEqual([row['process']['exit_code'] for row in rows], [0, 0])
        self.assertEqual([row['result']['seed'] for row in rows], [1818, 1819])


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--resume-child':
        child_resume(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--collect-replay':
        collect_replay(Path(sys.argv[2]))
    else:
        unittest.main()

"""Separate original CPU activation-checkpointing gate; released files unchanged."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_dpo_runner_recovery as base

r = base.runner
ROOT = base.ROOT
ATOL = RTOL = 1e-6  # premeasurement spec; never tuned after a result


def make_mode(enabled=True):
    values = list(base.build_fixture())
    model, reference, optimizer, _, _, options = values
    model.config.use_cache = False
    reference.config.use_cache = False
    if enabled:
        model.gradient_checkpointing_enable()  # actual production call/defaults
    options = deepcopy(options)
    options['sources']['tests/test_dpo_runner_recovery.py'] = r.file_digest(ROOT/'tests/test_dpo_runner_recovery.py')
    options['sources']['tests/test_dpo_checkpoint_mode.py'] = r.file_digest(Path(__file__))
    contract = r.make_recovery_contract(model=model, reference=reference, optimizer=optimizer, **options)
    return model, reference, optimizer, values[3], contract, options


def completed(output, values, initial=None):
    return base.train_fixture(output, values, initial=initial)


def interrupted(output, values):
    real_append = r.append_metric
    def failure(path, row):
        if row.get('update') == 3:
            raise OSError('authored checkpoint-on metric failure after durable update3')
        return real_append(path, row)
    try:
        with patch.object(r, 'append_metric', side_effect=failure):
            completed(output, values)
    except OSError as error:
        record = dict(type=type(error).__name__, message=str(error), deliberately_injected=True)
    else:
        raise AssertionError('Declared post-commit interruption did not happen')
    return base.checkpoint_receipt(Path(output)/'checkpoints/completed-000003.pt'), record


def comparison(expected, actual):
    """Report tolerances and observed maxima rather than claiming bitwise parity."""
    maxima = dict(policy=0., reference=0., adam=0., loss=0., margin=0., gradient_norm=0.)
    for role in ('policy', 'reference'):
        for name, value in expected['state'][role].items():
            other = actual['state'][role][name]
            torch.testing.assert_close(value, other, atol=ATOL, rtol=RTOL)
            maxima[role] = max(maxima[role], float((value-other).abs().max()))
    for index, moments in expected['state']['optimizer']['state'].items():
        for name, value in moments.items():
            other = actual['state']['optimizer']['state'][index][name]
            torch.testing.assert_close(value, other, atol=ATOL, rtol=RTOL)
            maxima['adam'] = max(maxima['adam'], float((value-other).abs().max()))
    for first, second in zip(expected['state']['history'], actual['state']['history']):
        if first['indices'] != second['indices'] or first['work'] != second['work'] or first['update'] != second['update']:
            raise AssertionError('Checkpointing changed discrete training work or sample stream')
        for key in ('loss', 'margin', 'gradient_norm'):
            torch.testing.assert_close(torch.tensor(first[key], dtype=torch.float64),
                torch.tensor(second[key], dtype=torch.float64), atol=ATOL, rtol=RTOL)
            maxima[key] = max(maxima[key], abs(first[key]-second[key]))
    for key in ('sampler_rng', 'torch_rng'):
        if not torch.equal(expected['state'][key], actual['state'][key]):
            raise AssertionError('Checkpointing changed retained RNG')
    if expected['state']['counters'] != actual['state']['counters']:
        raise AssertionError('Checkpointing changed cumulative work')
    return dict(atol=ATOL, rtol=RTOL, max_absolute_difference=maxima,
                discrete_work_and_rng_exact=True, tolerance_pass=True,
                scope='measured CPU FP32 parity; tolerance test is not a bitwise claim')


def child(receipt_file, contract_file, output):
    values = make_mode(True)
    expected = json.loads(Path(contract_file).read_text())
    if values[4] != expected:
        raise AssertionError('Fresh checkpoint-on science contract changed')
    receipt = json.loads(Path(receipt_file).read_text())
    initial = base.restore_fixture(values, receipt)
    Path(output).mkdir()
    result = completed(output, values, initial)
    r.write_exclusive_json(Path(output)/'result.json', result)


class DPOCheckpointModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dongxi-dpo-checkpoint-mode-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def directory(self, name):
        output = self.root/name; output.mkdir(); return output

    def full(self, name='on', enabled=True):
        values = make_mode(enabled)
        result = completed(self.directory(name), values)
        return values, result, base.payload_for(result['checkpoint'], values[4])

    def test_actual_production_mode_observation(self):
        values = make_mode(True)
        self.assertTrue(values[0].is_gradient_checkpointing)
        self.assertFalse(values[1].is_gradient_checkpointing)
        self.assertFalse(values[0].config.use_cache)
        self.assertFalse(values[1].config.use_cache)
        self.assertTrue(values[4]['policy_model']['gradient_checkpointing'])
        self.assertFalse(values[4]['reference_model']['gradient_checkpointing'])
        self.assertFalse(values[4]['policy_model']['config']['use_cache'])

    def test_actual_original_equation_update_with_checkpointing(self):
        values = make_mode(True)
        actual = r.completed_dpo_update(*values[:4], base.ENCODED, pad_id=1, accumulation=2,
            beta=.2, device='cpu', update=1)
        independent = make_mode(True)
        model, reference, optimizer, sampler = independent[:4]
        model.train(); reference.eval(); optimizer.zero_grad(set_to_none=True)
        losses, indices = [], []
        for _ in range(2):
            index = int(torch.randint(3, (), generator=sampler)); indices.append(index)
            scores = []
            for network in (model, reference):
                pair = []
                for ids, mask in base.ENCODED[index]:
                    tokens = torch.tensor([ids])
                    logits = network(input_ids=tokens[:, :-1], attention_mask=torch.ones_like(tokens[:, :-1]), use_cache=False).logits.float()
                    selected = logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
                    pair.append(selected[:, torch.tensor(mask[1:])].sum())
                scores.append(pair)
            margin = .2*(scores[0][0]-scores[0][1]-scores[1][0].detach()+scores[1][1].detach())
            loss = torch.nn.functional.softplus(-margin)
            (loss/2).backward(); losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assertEqual(actual['indices'], indices)
        self.assertTrue(torch.isfinite(norm)); self.assertGreater(float(norm), 0.)
        self.assertAlmostEqual(actual['loss'], sum(losses)/2, places=6)
        for first, second in zip(values[0].parameters(), model.parameters()):
            torch.testing.assert_close(first, second, atol=ATOL, rtol=RTOL)
            self.assertTrue(torch.isfinite(first.grad).all())
        self.assertEqual(base.digest(values[2].state_dict()), base.digest(optimizer.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in values[1].parameters()))

    def test_checkpoint_on_completed_replay_exact(self):
        _, full, expected = self.full()
        original = make_mode(True)
        receipt, _ = interrupted(self.directory('interrupted'), original)
        resumed = make_mode(True)
        result = completed(self.directory('resumed'), resumed, base.restore_fixture(resumed, receipt))
        actual = base.payload_for(result['checkpoint'], resumed[4])
        self.assertEqual(base.digest(base.numerical_state(expected)), base.digest(base.numerical_state(actual)))
        self.assertEqual(full['history'][3], result['history'][3])
        self.assertEqual(actual['state']['counters']['sampled_pairs'], 12)
        self.assertEqual(base.digest(actual['state']['reference']), resumed[4]['reference_sha256'])
        self.assertNotEqual(base.digest(actual['state']['reference']), base.digest(actual['state']['policy']))

    def test_checkpoint_on_fresh_process_exact(self):
        _, _, expected = self.full()
        original = make_mode(True)
        receipt, _ = interrupted(self.directory('interrupted'), original)
        r.write_exclusive_json(self.root/'receipt.json', receipt)
        r.write_exclusive_json(self.root/'contract.json', original[4])
        command = [sys.executable, str(Path(__file__).resolve()), '--child', str(self.root/'receipt.json'),
                   str(self.root/'contract.json'), str(self.root/'fresh')]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        result = json.loads((self.root/'fresh/result.json').read_text())
        actual = base.payload_for(result['checkpoint'], original[4])
        self.assertEqual(base.digest(base.numerical_state(expected)), base.digest(base.numerical_state(actual)))

    def test_checkpoint_off_on_predeclared_tolerance(self):
        _, _, first = self.full('off', False)
        _, _, second = self.full('on', True)
        comparison(first, second)

    def test_wrong_mode_contract_rejected_before_application(self):
        values, result, _ = self.full()
        wrong = make_mode(False)
        self.assertNotEqual(values[4], wrong[4])
        self.assertEqual(values[4]['policy_shapes'], wrong[4]['policy_shapes'])
        with patch.object(wrong[0], 'load_state_dict') as apply, self.assertRaises(ValueError):
            base.restore_fixture(wrong, result['checkpoint'])
        apply.assert_not_called()

    def test_eos_prompt_and_padding_preserved_in_checkpoint_mode(self):
        ids, attention, mask = r.collate(base.ENCODED[:2], 0, 1, 'cpu')
        self.assertFalse(mask[0,-1]); self.assertTrue(mask[0,-2]); self.assertFalse(mask[:,:2].any())
        values = make_mode(True)
        row = r.completed_dpo_update(*values[:4], base.ENCODED, pad_id=1, accumulation=2,
            beta=.2, device='cpu', update=1)
        self.assertEqual(row['work']['chosen_targets'], 6)
        self.assertEqual(row['work']['rejected_targets'], 5)


def collect(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    names = ['scripts/run_chapter11_spark_dpo.py', 'tests/test_dpo_runner_recovery.py',
        'tests/test_dpo_checkpoint_mode.py', 'src/dongxi_llms/training_snapshot.py',
        'src/dongxi_llms/dpo_lab.py', 'src/dongxi_llms/batched_cache_lab.py', 'uv.lock',
        'experiments/specs/2026-10-05-dpo-activation-checkpointing.md',
        'experiments/reports/2026-10-05-dpo-runner-recovery.md']
    sources = {name:r.file_digest(ROOT/name) for name in names}
    r.write_exclusive_json(output/'source-identities-before.json', sources)
    r.write_exclusive_json(output/'authored-fixture.json', base.FIXTURE)
    command = [sys.executable, '-m', 'unittest', 'test_dpo_checkpoint_mode', '-v']
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    tests = dict(command=command, exit_code=process.returncode, seconds=time.perf_counter()-started,
                 stdout=process.stdout, stderr=process.stderr)
    r.write_exclusive_json(output/'focused-tests.json', tests)
    if process.returncode:
        raise RuntimeError('Frozen checkpoint-mode tests failed; this output directory remains diagnostic')
    def directory(name):
        path = output/name; path.mkdir(); return path
    on = make_mode(True)
    r.write_exclusive_json(output/'recovery-contract-on.json', on[4])
    full = completed(directory('on-uninterrupted'), on)
    expected = base.payload_for(full['checkpoint'], on[4])
    initial = make_mode(True)
    receipt, failure = interrupted(directory('on-interrupted'), initial)
    r.write_exclusive_json(output/'on-interrupted/failure.json', failure)
    r.write_exclusive_json(output/'independent-receipt.json', receipt)
    resumed = make_mode(True)
    result = completed(directory('on-resumed'), resumed, base.restore_fixture(resumed, receipt))
    actual = base.payload_for(result['checkpoint'], resumed[4])
    command = [sys.executable, str(Path(__file__).resolve()), '--child', str(output/'independent-receipt.json'),
               str(output/'recovery-contract-on.json'), str(output/'on-fresh-process')]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    fresh_execution = dict(command=command, exit_code=process.returncode, seconds=time.perf_counter()-started,
                           stdout=process.stdout, stderr=process.stderr)
    r.write_exclusive_json(output/'fresh-process-execution.json', fresh_execution)
    if process.returncode:
        raise RuntimeError('Frozen checkpoint-on fresh process failed')
    fresh = json.loads((output/'on-fresh-process/result.json').read_text())
    fresh_payload = base.payload_for(fresh['checkpoint'], on[4])
    equal = dict(same_process=base.digest(base.numerical_state(expected))==base.digest(base.numerical_state(actual)),
        fresh_process=base.digest(base.numerical_state(expected))==base.digest(base.numerical_state(fresh_payload)),
        next_row=full['history'][3]==result['history'][3]==fresh['history'][3],
        original_reference=base.digest(expected['state']['reference'])==on[4]['reference_sha256'],
        reference_not_policy=base.digest(expected['state']['reference'])!=base.digest(expected['state']['policy']))
    if not all(equal.values()):
        raise AssertionError('Checkpoint-on exact recovery failed')
    off = make_mode(False)
    r.write_exclusive_json(output/'recovery-contract-off.json', off[4])
    off_result = completed(directory('off-uninterrupted'), off)
    off_payload = base.payload_for(off_result['checkpoint'], off[4])
    parity = comparison(off_payload, expected)
    for name,value in (('on-uninterrupted/result.json', full),('on-resumed/result.json',result),('off-uninterrupted/result.json',off_result)):
        r.write_exclusive_json(output/name,value)
    after = {name:r.file_digest(ROOT/name) for name in names}
    r.write_exclusive_json(output/'source-identities-after.json', after)
    if after != sources:
        raise AssertionError('Released or new source bytes changed during collection')
    artifacts = {str(path.relative_to(ROOT)):r.file_digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    verification = dict(schema='dongxi-dpo-cpu-activation-checkpointing-v1', status='pass',
        command=sys.orig_argv, tests=tests, test_count=7, equality=equal, off_on_comparison=parity,
        on_numeric_state_digest=base.digest(base.numerical_state(expected)),
        on_snapshot_contract_sha256=full['checkpoint']['contract_sha256'],
        off_snapshot_contract_sha256=off_result['checkpoint']['contract_sha256'],
        source_sha256=sources, artifact_sha256=artifacts, fixture_file_sha256=r.file_digest(output/'authored-fixture.json'),
        full_history=full['history'], cumulative_work=full['counters'], observed_interruption=failure,
        measured_environment=on[4]['environment'], mode=on[4]['policy_model'],
        accepted_scope='actual DPO runner functions; random local CPU FP32 activation checkpointing only',
        pending=['CUDA/BF16 recovery and parity','pretrained-model compatibility','model-scale overhead or memory savings',
                 'cross-machine determinism','whole-job resource containment','language quality'])
    r.write_exclusive_json(output/'verification.json',verification)
    print(json.dumps(dict(status='pass',tests=7,verification=str(output/'verification.json'),equality=equal,comparison=parity)))


if __name__ == '__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--child':
        child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':
        collect(sys.argv[2])
    else:
        unittest.main()

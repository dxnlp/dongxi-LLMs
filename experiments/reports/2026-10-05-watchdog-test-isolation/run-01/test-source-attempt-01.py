"""Fixed native command preparation and the same watchdog's inert CPU controls."""
from copy import deepcopy
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import psutil

from dongxi_llms import native_profile_supervisor as lab
from dongxi_llms.snapshot_io_budget import IO_KEYS, io_budget_contract


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def _low(exclude):
    value = lab.injected_cpu_clearance(exclude); value['available_bytes'] = 25*lab.GIB-1
    return value


def _conflict(exclude):
    value = lab.injected_cpu_clearance(exclude)
    value['conflict_scan']['conflicts'] = [dict(pid=123456789, rss_bytes=0,
        reasons=['known-model-process-signature'])]
    return value


def _incomplete(exclude):
    value = lab.injected_cpu_clearance(exclude); value['conflict_scan']['unreadable'] = 1
    return value


def _bad_memory(exclude):
    value = lab.injected_cpu_clearance(exclude); value['available_bytes'] = True
    return value


class AfterPreflight:
    def __init__(self, mode): self.mode, self.count = mode, 0
    def __call__(self, exclude):
        self.count += 1
        if self.count > 1:
            if self.mode == 'block': time.sleep(10)
            elif self.mode == 'error': raise RuntimeError('authored observer failure')
            elif self.mode == 'interrupt': raise KeyboardInterrupt('authored observer interruption')
        return lab.injected_cpu_clearance(exclude)


def _log_failure(row):
    if row['event'] == 'child-started': raise OSError('authored logging failure')


def _blocked_logging(row):
    if row['event'] == 'child-started': time.sleep(10)


def _partial_observer(requests, results, probe):
    first = requests.get()
    results.send(('observation', first[0], lab.injected_cpu_clearance(()), []))
    requests.get()
    # Datagram truncation is rejected atomically, never a framed blocking recv.
    results.endpoint.send(b'{"partial":')
    time.sleep(10)


SUPPORTED = (os.name == 'posix' and 'fork' in mp.get_all_start_methods()
             and hasattr(os, 'waitid') and hasattr(os, 'WNOWAIT'))
ISOLATED_WORKER = 'DONGXI_WATCHDOG_ISOLATED_WORKER'


def _isolated_worker(method):
    """Run one existing inert case in a fresh interpreter, not a threaded fork."""
    if os.environ.get(ISOLATED_WORKER) != '1':
        raise RuntimeError('Explicit isolated test worker required')
    if method not in unittest.defaultTestLoader.getTestCaseNames(WatchdogTests):
        raise ValueError('Only an existing WatchdogTests method is accepted')
    initial_threads = threading.active_count()
    suite = unittest.TestSuite([WatchdogTests(method)])
    result = unittest.TestResult()
    suite.run(result)
    payload = dict(tests_run=result.testsRun, successful=result.wasSuccessful(),
        initial_thread_count=initial_threads,
        errors=[dict(test=case.id(), traceback=trace) for case, trace in result.errors],
        failures=[dict(test=case.id(), traceback=trace) for case, trace in result.failures],
        skipped=[dict(test=case.id(), reason=reason) for case, reason in result.skipped])
    print(json.dumps(payload, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)


def _run_isolated_case(method, *, evidence=None):
    """Test-only subprocess adapter; production refuses threaded callers unchanged."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1')
    environment[ISOLATED_WORKER] = '1'
    environment['PYTHONPATH'] = os.pathsep.join((str(lab.ROOT/'src'), str(lab.ROOT/'tests')))
    if evidence is not None:
        evidence.mkdir(mode=0o700, exist_ok=True)
        environment['DONGXI_WATCHDOG_EVIDENCE'] = str(evidence)
    started = time.monotonic()
    command = [sys.executable, '-c',
        'import sys; import test_native_profile_supervisor as tests; tests._isolated_worker(sys.argv[1])',
        method]
    completed = subprocess.run(command, cwd=lab.ROOT, env=environment,
        capture_output=True, text=True, timeout=30)
    if len(completed.stdout)+len(completed.stderr) > 1024**2:
        raise RuntimeError('Isolated inert-test diagnostics exceed1MiB')
    transport = dict(method=method, actual_exit_code=completed.returncode,
        elapsed_seconds=time.monotonic()-started,
        parent_thread_count=threading.active_count(), stdout=completed.stdout, stderr=completed.stderr)
    retained = environment.get('DONGXI_WATCHDOG_EVIDENCE')
    if retained:
        directory = Path(retained)/'isolated-transport'
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory/(method+'.json')).open('x') as handle:
            handle.write(json.dumps(transport, indent=2, allow_nan=False)+'\n')
    try:
        transport['worker'] = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('Isolated worker did not retain a structured result: '+completed.stderr[:4096]) from error
    return transport


class _IsolatedWatchdogCase(unittest.TestCase):
    """Preserve each outer case count and standard unittest failure/skip mapping."""
    def __init__(self, original):
        super().__init__('runTest')
        self.original = original
    def id(self): return self.original.id()
    def shortDescription(self): return self.original.shortDescription()
    def runTest(self):
        value = _run_isolated_case(self.original._testMethodName)
        worker = value['worker']
        self.assertEqual(worker['tests_run'], 1)
        self.assertEqual(worker['initial_thread_count'], 1)
        if worker['errors']:
            raise RuntimeError('\n'.join(row['traceback'] for row in worker['errors']))
        if worker['failures']:
            self.fail('\n'.join(row['traceback'] for row in worker['failures']))
        if worker['skipped']:
            self.skipTest('; '.join(row['reason'] for row in worker['skipped']))
        self.assertTrue(worker['successful'])
        self.assertEqual(value['actual_exit_code'], 0)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        generator = _module(lab.ROOT/'scripts/prepare_chapter09_instruction_fixture.py', 'native_profile_generator')
        for split, start, count in [('train', 0, 80), ('dev', 80, 20)]:
            rows = generator.records(split, start, count)
            (self.root/(split+'.jsonl')).write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
        (self.root/'lock').write_text('authored CPU input-binding fixture; not the platform environment\n')
        (self.root/'caps.json').write_text(json.dumps(dict.fromkeys(lab.WORK_KEYS, 1000000)))
        io = io_budget_contract(dict.fromkeys(IO_KEYS, 1000000),
            dict(max_payload_bytes=4096, max_tree_nodes=100, max_tensor_elements=512,
                 max_tensor_bytes=1024, max_primitive_bytes=100), 65536)
        (self.root/'io.json').write_text(json.dumps(io))
        self.request = dict(stage_id=lab.STAGE, model=lab.MODEL, revision=lab.REVISION,
            tokenizer_revision=lab.REVISION, recipe=dict(lab.RECIPE), interpreter=sys.executable,
            train=str(self.root/'train.jsonl'), dev=str(self.root/'dev.jsonl'),
            template=str(lab.ROOT/'experiments/data/instruction_interface_v1.jinja'),
            environment_lock=str(self.root/'lock'), work_limits=str(self.root/'caps.json'),
            snapshot_io_limits=str(self.root/'io.json'), snapshot_max_bytes=4096,
            work_journal_max_bytes=65536, work_journal=str(self.root/'work.jsonl'),
            snapshot_io_ledger=str(self.root/'io-ledger.jsonl'), output=str(self.root/'native-output'))
    def tearDown(self): self.temp.cleanup()

    def test_preparation_constructs_exact_native_route_without_spawn_or_outputs(self):
        with patch.object(subprocess, 'Popen') as spawn:
            value = lab.prepare_native_profile(self.request)
        spawn.assert_not_called()
        self.assertFalse(Path(self.request['output']).exists())
        self.assertFalse(value['launch_authorized']); self.assertEqual(value['jobs_started'], 0)
        self.assertIsNone(value['actual_external_result']); self.assertTrue(value['unknown'])
        self.assertEqual(value['argv'][:2], [sys.executable, str(lab.ROOT/'scripts/run_chapter09_spark_sft.py')])
        argv = value['argv'][2:]; observed = dict(zip(argv[::2], argv[1::2]))
        for key, item in lab.RECIPE.items(): self.assertEqual(observed['--'+key.replace('_', '-')], str(item))
        self.assertEqual(observed['--model'], lab.MODEL)
        self.assertEqual(observed['--revision'], lab.REVISION)
        self.assertNotIn('--allow-download', argv); self.assertNotIn('--resume', argv)
        self.assertEqual(value['known_snapshot_io_requirements']['snapshot_save_operations'], 2)
        self.assertEqual(value['known_snapshot_io_requirements']['snapshot_load_operations'], 0)
        self.assertNotIn('encoded_train_targets', value)

    def test_fixed_stage_revision_model_recipe_and_boolean_aliases_refuse(self):
        for key, value in [('stage_id', 'assistant-sft-full-pilot'), ('model', 'Qwen/Qwen3-0.6B'),
                           ('revision', '0'*40), ('tokenizer_revision', '1'*40)]:
            changed = deepcopy(self.request); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): lab.prepare_native_profile(changed)
        for key, value in [('updates', 21), ('microbatch', True), ('mode', 'lora'), ('runtime_seconds', 901)]:
            changed = deepcopy(self.request); changed['recipe'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): lab.prepare_native_profile(changed)

    def test_extra_argv_resume_and_acquisition_fields_refuse(self):
        for key in ('argv', 'allow_download', 'resume', 'shell'):
            changed = deepcopy(self.request); changed[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): lab.prepare_native_profile(changed)

    def test_missing_inputs_and_existing_or_shared_destinations_refuse(self):
        changed = deepcopy(self.request); changed['train'] = str(self.root/'missing')
        with self.assertRaises(FileNotFoundError): lab.prepare_native_profile(changed)
        changed = deepcopy(self.request); changed['work_journal'] = changed['snapshot_io_ledger']
        with self.assertRaises(ValueError): lab.prepare_native_profile(changed)
        (self.root/'native-output').mkdir()
        with self.assertRaises(FileExistsError): lab.prepare_native_profile(self.request)

    def test_nonprivate_journal_parent_refuses(self):
        parent = self.root/'public'; parent.mkdir(mode=0o755); parent.chmod(0o755)
        for key in ('work_journal', 'snapshot_io_ledger'):
            changed = deepcopy(self.request); changed[key] = str(parent/'journal')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'private'): lab.prepare_native_profile(changed)

    def test_original_input_population_and_template_required(self):
        (self.root/'train.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'population'): lab.prepare_native_profile(self.request)

    def test_incomplete_caps_unknown_zero_and_snapshot_schedule_refuse(self):
        caps = dict.fromkeys(lab.WORK_KEYS, 1000000)
        for changed in ({k:v for k,v in caps.items() if k != 'valid_targets'},
                        dict(caps, valid_targets=0), dict(caps, generation_tokens=1023),
                        dict(caps, train_updates=True)):
            (self.root/'caps.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError): lab.prepare_native_profile(self.request)
        (self.root/'caps.json').write_text(json.dumps(caps))
        io = json.loads((self.root/'io.json').read_text()); io['limits']['snapshot_save_operations'] = 1
        (self.root/'io.json').write_text(json.dumps(io))
        with self.assertRaisesRegex(ValueError, 'two complete'): lab.prepare_native_profile(self.request)

    def test_consumed_caps_and_io_bytes_must_match_recorded_binding(self):
        original = lab.read_bounded_json
        for key in ('work_limits', 'snapshot_io_limits'):
            def changed_reader(path):
                value, identity = original(path)
                if path == self.request[key]: identity['sha256'] = '0'*64
                return value, identity
            with self.subTest(key=key), patch.object(lab, 'read_bounded_json', changed_reader):
                with self.assertRaisesRegex(ValueError, 'between reads'): lab.prepare_native_profile(self.request)

    def test_explicit_execution_wiring_and_changed_binding_refuse_before_spawn(self):
        prepared = lab.prepare_native_profile(self.request)
        with self.assertRaises(ValueError):
            lab.execute_native_profile(prepared, operator_declaration='yes', supervision_output=str(self.root/'evidence'))
        with patch.object(lab, '_supervise', return_value={'status':'not-launched-test-spy'}) as engine:
            result = lab.execute_native_profile(prepared,
                operator_declaration='Authored wiring control only; no actual profile permission.',
                supervision_output=str(self.root/'evidence'))
        self.assertEqual(result['status'], 'not-launched-test-spy')
        self.assertEqual(engine.call_args.args[0], prepared['argv'])
        self.assertTrue(engine.call_args.kwargs['native']); self.assertEqual(engine.call_args.kwargs['seconds'], 900)
        (self.root/'lock').write_text('changed actual bytes\n')
        with patch.object(lab, '_supervise') as engine:
            with self.assertRaisesRegex(ValueError, 'changed'):
                lab.execute_native_profile(prepared, operator_declaration='Authored not-authorized declaration.',
                    supervision_output=str(self.root/'evidence'))
        engine.assert_not_called()

    def test_default_cli_prepares_without_launch_and_rejects_execution_omissions(self):
        cli = _module(lab.ROOT/'scripts/prepare_native_sft_profile.py', 'native_profile_cli')
        argv = []
        for key in lab.PATH_KEYS + ('snapshot_max_bytes', 'work_journal_max_bytes'):
            argv.extend(['--'+key.replace('_', '-'), str(self.request[key])])
        with patch.object(cli, 'execute_native_profile') as execute, patch('builtins.print'):
            self.assertEqual(cli.main(argv), 0)
        execute.assert_not_called()
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            cli.main(argv+['--execute'])

    def test_helper_signal_refusal_retains_unknown_cleanup(self):
        class Refuses:
            pid = 12345
            exitcode = None
            def is_alive(self): return True
            def terminate(self): raise PermissionError('authored refusal')
            def kill(self): raise PermissionError('authored refusal')
            def join(self, timeout): pass
        result = lab._finish_helper(Refuses(), seconds=.01)
        self.assertTrue(result['still_alive']); self.assertFalse(result['cleanup_complete'])
        self.assertIsNone(result['actual_exit_code']); self.assertEqual(len(result['errors']), 2)

    def test_threaded_caller_is_not_silently_treated_as_isolated_controller(self):
        output = self.root/'not-created'
        with patch.object(lab.threading, 'active_count', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'isolated'):
                lab.supervise_cpu_control('success', output)
        self.assertFalse(output.exists())
        if SUPPORTED:
            # Full discovery can leave a legitimate tqdm/other auxiliary thread.
            # The adapter isolates the test; it must not weaken the live guard or
            # attempt to stop someone else's thread. Reuse an existing real case.
            stop = threading.Event()
            auxiliary = threading.Thread(target=stop.wait, name='authored-parent-test-thread', daemon=True)
            auxiliary.start()
            parent = os.environ.get('DONGXI_WATCHDOG_EVIDENCE')
            evidence = (Path(parent)/'parent-auxiliary-regression') if parent else self.root/'auxiliary-evidence'
            try:
                self.assertTrue(auxiliary.is_alive())
                value = _run_isolated_case('test_success_and_nonzero_actual_exits', evidence=evidence)
                self.assertGreaterEqual(value['parent_thread_count'], 2)
                self.assertEqual(value['worker']['initial_thread_count'], 1)
                self.assertTrue(value['worker']['successful'])
                self.assertEqual(value['actual_exit_code'], 0)
                self.assertTrue(auxiliary.is_alive())
            finally:
                stop.set(); auxiliary.join(timeout=1)
            self.assertFalse(auxiliary.is_alive())


@unittest.skipUnless(SUPPORTED, 'Owned Linux/POSIX kernel waitid/fork control; not an observed Mac pass')
class WatchdogTests(unittest.TestCase):
    def run(self, result=None):
        if os.environ.get(ISOLATED_WORKER) == '1' or not SUPPORTED:
            return super().run(result)
        return _IsolatedWatchdogCase(self).run(result)
    def setUp(self):
        evidence = os.environ.get('DONGXI_WATCHDOG_EVIDENCE')
        if evidence:
            self.temp = None
            self.root = Path(evidence).resolve()/self._testMethodName
            self.root.mkdir(mode=0o700)
        else:
            self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.counter = 0
    def tearDown(self):
        if self.temp is not None: self.temp.cleanup()
    def run_control(self, mode, **kwargs):
        self.counter += 1; output = self.root/('control-'+str(self.counter))
        before = time.monotonic(); result = lab.supervise_cpu_control(mode, output, **kwargs)
        self.assertLess(time.monotonic()-before, 3)
        self.assertFalse(result['native_model_stage']); self.assertFalse(result['actual_native_profile_executed'])
        if result['child_pid'] is not None:
            self.assertIsNotNone(result['actual_exit_code'])
            self.assertFalse(psutil.pid_exists(result['child_pid']))
        self.assertTrue(result['final_record_retained'])
        self.assertTrue(all(not row['still_alive'] for row in result['queue_feeder_shutdown']))
        raw = json.loads((output/'result.json').read_text())
        for key in ('actual_exit_code', 'stop_reason', 'failure', 'status'):
            self.assertEqual(raw[key], result[key])
        self.assertTrue(result['in_memory_events'])
        return result

    def test_success_and_nonzero_actual_exits(self):
        good = self.run_control('success', seconds=1)
        self.assertEqual(good['status'], 'completed'); self.assertEqual(good['actual_exit_code'], 0)
        bad = self.run_control('exit7', seconds=1)
        self.assertEqual(bad['status'], 'failed'); self.assertEqual(bad['actual_exit_code'], 7)

    def test_external_deadline_reaps_term_ignoring_child(self):
        value = self.run_control('ignore-term', seconds=.5, term_seconds=.05)
        self.assertEqual(value['stop_reason'], 'external-deadline')
        self.assertEqual(value['actual_exit_code'], -signal.SIGKILL)

    def test_preflight_low_conflict_and_incomplete_refuse_without_signalling_conflicts(self):
        for probe, expected in ((_low, 'host-reserve-below-threshold'), (_conflict, 'conflicting-process'),
                                (_incomplete, 'incomplete-clearance')):
            with self.subTest(expected=expected), patch.object(os, 'killpg') as group_signal:
                value = self.run_control('success', probe=probe)
            self.assertEqual(value['status'], 'refused'); self.assertIsNone(value['child_pid'])
            self.assertEqual(value['stop_reason'], expected); group_signal.assert_not_called()

    def test_malformed_or_missing_measurement_refuses(self):
        value = self.run_control('success', probe=_bad_memory)
        self.assertEqual(value['status'], 'refused'); self.assertEqual(value['stop_reason'], 'observer-failure')
        self.assertEqual(value['failure']['type'], 'ValueError')

    def test_observer_error_and_interruption_retained_after_spawn(self):
        for mode, expected in [('error', 'observer-failure'), ('interrupt', 'observer-interrupted')]:
            with self.subTest(mode=mode): value = self.run_control('sleep', seconds=1, probe=AfterPreflight(mode))
            self.assertIsNotNone(value['child_pid']); self.assertEqual(value['stop_reason'], expected)
            self.assertIn('authored', value['failure']['message'])

    def test_blocking_observer_cannot_stall_deadline(self):
        value = self.run_control('sleep', seconds=.15, probe_seconds=1, probe=AfterPreflight('block'))
        self.assertEqual(value['stop_reason'], 'external-deadline')

    def test_stale_observer_stops_before_longer_deadline(self):
        value = self.run_control('sleep', seconds=1, probe_seconds=.1, probe=AfterPreflight('block'))
        self.assertEqual(value['stop_reason'], 'observer-timeout')

    def test_logging_failure_retained_and_child_stopped(self):
        value = self.run_control('sleep', seconds=1, journal_hook=_log_failure)
        self.assertEqual(value['stop_reason'], 'journal-failure')
        self.assertEqual(value['journal_error']['type'], 'OSError')

    def test_blocking_fsync_callback_cannot_stall_deadline(self):
        value = self.run_control('sleep', seconds=.15, logging_seconds=1, journal_hook=_blocked_logging)
        self.assertEqual(value['stop_reason'], 'external-deadline')
        self.assertEqual(value['journal_error']['type'], 'TimeoutError')

    def test_atomic_truncated_observer_datagram_refuses_and_reaps(self):
        with patch.object(lab, '_probe_worker', _partial_observer):
            value = self.run_control('sleep', seconds=.5)
        self.assertEqual(value['stop_reason'], 'controller-failure')
        self.assertEqual(value['failure']['type'], 'JSONDecodeError')

    def test_second_helper_start_failure_cleans_first(self):
        context = mp.get_context('fork'); process_type = type(context.Process(target=time.sleep, args=(0,)))
        original = process_type.start; started = []
        def start(worker):
            if len(started) == 1: started.append(None); raise OSError('authored second-start failure')
            original(worker); started.append(worker)
        with patch.object(process_type, 'start', start):
            value = self.run_control('success')
        self.assertEqual(value['stop_reason'], 'controller-failure')
        self.assertEqual(value['failure']['type'], 'OSError')
        self.assertFalse(started[0].is_alive())

    def test_natural_exit_and_term_cooperative_leader_drain_ignoring_workers(self):
        for mode in ('exit-with-worker', 'term-with-worker'):
            with self.subTest(mode=mode): value = self.run_control(mode, seconds=.5, term_seconds=.05)
            text = (Path(value['output'])/'stdout.txt').read_text()
            worker_pid = int(text.split('owned-worker=')[1].splitlines()[0])
            try: status = psutil.Process(worker_pid).status()
            except psutil.NoSuchProcess: status = 'gone'
            self.assertIn(status, ('gone', psutil.STATUS_ZOMBIE))
            if mode == 'exit-with-worker': self.assertEqual(value['actual_exit_code'], 0)

    def test_unknown_mode_and_invalid_deadline_refuse_before_creation(self):
        output = self.root/'invalid'
        for mode, seconds in [('arbitrary command', .2), ('success', 901), ('success', True)]:
            with self.subTest(mode=mode, seconds=seconds), self.assertRaises(ValueError):
                lab.supervise_cpu_control(mode, output, seconds=seconds)
        self.assertFalse(output.exists())

    def test_denied_term_does_not_skip_kill_reap_or_helper_cleanup(self):
        original = os.killpg
        def deny_term(group, signum):
            if signum == signal.SIGTERM: raise PermissionError('authored TERM refusal')
            return original(group, signum)
        with patch.object(os, 'killpg', deny_term):
            value = self.run_control('sleep', seconds=.2, term_seconds=.05)
        self.assertEqual(value['actual_exit_code'], -signal.SIGKILL)
        self.assertIn('TERM: PermissionError', value['cleanup_errors'])
        self.assertEqual(value['status'], 'failed')


if __name__ == '__main__': unittest.main()

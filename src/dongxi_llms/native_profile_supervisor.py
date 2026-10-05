"""One prepared native profile and its external, owned-process watchdog.

Preparation does not load a tokenizer/model or grant launch authority.  Probes
and fsynced logging run in owned helper processes: neither can stall the parent
deadline.  This is sampled, cooperative supervision, not a cgroup, quota,
hostile-descendant sandbox, authenticated approval service or GPU-idle proof.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

import psutil

from .campaign_supervisor import GIB, host_memory, conflicts
from .run_identity import canonical_hash
from .snapshot_io_budget import IO_KEYS, read_bounded_json, validate_io_contract

ROOT = Path(__file__).resolve().parents[2]
STAGE = 'assistant-profile-base06'
MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
RECIPE = dict(mode='full', updates=20, microbatch=1, accumulation=4,
              max_length=256, learning_rate=2e-5, runtime_seconds=900,
              reserve_gib=25, seed=1212, checkpoint_every=20)
WORK_KEYS = ('train_updates', 'sampled_examples', 'selector_steps', 'valid_targets',
    'logical_sequence_tokens', 'policy_forward_calls', 'policy_forward_positions',
    'evaluation_calls', 'evaluation_positions', 'generation_calls',
    'generation_position_upper_bound', 'generation_tokens',
    'recovery_validation_operations', 'recovery_history_rows',
    'recovery_tensor_elements', 'recovery_rng_states')
PATH_KEYS = ('interpreter', 'train', 'dev', 'template', 'environment_lock',
    'work_limits', 'snapshot_io_limits', 'work_journal', 'snapshot_io_ledger', 'output')
REQUEST_KEYS = set(PATH_KEYS) | {'stage_id', 'model', 'revision',
    'tokenizer_revision', 'recipe', 'snapshot_max_bytes', 'work_journal_max_bytes'}
SOURCES = ('scripts/run_chapter09_spark_sft.py',
    'scripts/prepare_chapter09_instruction_fixture.py',
    'src/dongxi_llms/native_profile_supervisor.py',
    'src/dongxi_llms/run_identity.py', 'src/dongxi_llms/training_snapshot.py',
    'src/dongxi_llms/snapshot_io_budget.py', 'src/dongxi_llms/work_budget.py')
INERT_MODES = frozenset(('success', 'exit7', 'ignore-term', 'sleep', 'worker',
                         'exit-with-worker', 'term-with-worker'))
NATIVE_DEADLINE_ENVELOPES = {
    'profile': 900,
    'story-pilot': 14400,
    'assistant-pilot': 3600,
    'preference-pilot': 1800,
    'reasoning-pilot': 1800,
}


def _integer(value, label, minimum=0, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Exact bounded integer required: ' + label)
    return value


def _path(value):
    if type(value) is not str or not value or any(ord(c) < 32 for c in value):
        raise ValueError('Explicit plain path required')
    if not os.path.isabs(value):
        raise ValueError('Paths must be explicit and absolute')
    return Path(os.path.abspath(value))


def _digest(path, *, executable=False, maximum=64*1024**2):
    path = _path(str(path))
    # A venv executable is legitimately a symlink. Bind both the invoked path
    # and its actual executable bytes; inputs themselves require regular files.
    actual = path.resolve(strict=True) if executable else path
    fd = os.open(actual, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError('Bounded nonempty regular input required: ' + str(path))
        if executable and not os.access(path, os.X_OK):
            raise ValueError('Declared interpreter is not executable')
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024*1024)
            if not block: break
            digest.update(block)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Input changed during inspection')
    finally:
        os.close(fd)
    return dict(path=str(path), actual_path=str(actual), bytes=before.st_size,
                sha256=digest.hexdigest())


def _new_target(path, *, private=False):
    path = _path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('New output/journal required: ' + str(path))
    # Do not create anything during preparation. Refuse missing/symlinked
    # parents rather than silently choosing a new filesystem/ownership domain.
    parent = path.parent
    if not parent.is_dir() or any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError('Existing no-symlink parent required: ' + str(path))
    if parent.stat().st_uid != os.getuid():
        raise ValueError('Output/journal parent must belong to this operator')
    if private and stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ValueError('Journal parent must be private to its operator')
    return path


def prepare_native_profile(request):
    """Hash current inputs and construct ONLY the declared native profile argv.

    Full allowances are caller declarations. No encoded count, tokenizer
    compatibility, snapshot fit, authority or successful profile is inferred.
    """
    if type(request) is not dict or set(request) != REQUEST_KEYS:
        raise ValueError('Exact fixed native-profile fields required; no extra argv/resume/acquisition')
    request = deepcopy(request)
    if (request['stage_id'] != STAGE or request['model'] != MODEL or
            request['revision'] != REVISION or request['tokenizer_revision'] != REVISION):
        raise ValueError('Only the pinned assistant-profile-base06 is supported')
    recipe = request['recipe']
    if (type(recipe) is not dict or set(recipe) != set(RECIPE) or any(
            type(recipe[k]) is not type(v) or recipe[k] != v for k, v in RECIPE.items())):
        raise ValueError('The fixed20-update profile recipe cannot be changed')
    for key in PATH_KEYS:
        request[key] = str(_path(request[key]))
    _integer(request['snapshot_max_bytes'], 'snapshot_max_bytes', 1)
    _integer(request['work_journal_max_bytes'], 'work_journal_max_bytes', 1, 64*1024**2)
    targets = [_new_target(request[k], private=k != 'output')
               for k in ('output', 'work_journal', 'snapshot_io_ledger')]
    if len(set(targets)) != 3:
        raise ValueError('Output and physical journals must be distinct')
    if any(targets[0] in p.parents or p in targets[0].parents for p in targets[1:]):
        raise ValueError('Use existing separate private journal parents, not the future native output')
    files = {key: _digest(request[key], executable=(key == 'interpreter'))
             for key in PATH_KEYS if key not in ('output', 'work_journal', 'snapshot_io_ledger')}
    limits, consumed_limits = read_bounded_json(request['work_limits'])
    if (consumed_limits['sha256'], consumed_limits['bytes']) != (
            files['work_limits']['sha256'], files['work_limits']['bytes']):
        raise ValueError('Consumed SFT cap bytes changed between reads')
    if type(limits) is not dict or set(limits) != set(WORK_KEYS):
        raise ValueError('All sixteen explicit SFT v2 caps are required')
    for key, value in limits.items(): _integer(value, key)
    # These are known schedule counts, not tokenizer-derived target estimates.
    minima = dict(train_updates=20, sampled_examples=80, selector_steps=80,
        generation_calls=16, generation_tokens=1024,
        recovery_validation_operations=2, recovery_history_rows=20)
    if any(limits[k] < amount for k, amount in minima.items()):
        raise ValueError('Caps omit the complete known profile/observer/save schedule')
    if any(limits[k] == 0 for k in WORK_KEYS if k not in minima):
        raise ValueError('Explicit positive allowances required; unknown work is not zero')
    io, consumed_io = read_bounded_json(request['snapshot_io_limits'])
    if (consumed_io['sha256'], consumed_io['bytes']) != (
            files['snapshot_io_limits']['sha256'], files['snapshot_io_limits']['bytes']):
        raise ValueError('Consumed snapshot I/O bytes changed between reads')
    io = validate_io_contract(io)
    envelope = io['envelope']
    if envelope['max_payload_bytes'] != request['snapshot_max_bytes']:
        raise ValueError('Snapshot and I/O payload envelopes differ')
    required = dict.fromkeys(IO_KEYS, 0)
    required.update(snapshot_save_operations=2,
        snapshot_hash_bytes=2*envelope['max_payload_bytes'],
        snapshot_tree_nodes=2*envelope['max_tree_nodes'],
        snapshot_tensor_elements=2*envelope['max_tensor_elements'],
        snapshot_primitive_bytes=2*envelope['max_primitive_bytes'],
        snapshot_clone_bytes=2*envelope['max_tensor_bytes'],
        snapshot_serialization_bytes=2*envelope['max_payload_bytes'])
    if any(io['limits'][key] < amount for key, amount in required.items()):
        raise ValueError('I/O caps must reserve the two complete native saves')
    # The original named source population/template, not an easier substitute.
    card = json.loads((ROOT/'experiments/data/instruction-interface-v1-data-card.json').read_text())
    for key in ('train', 'dev'):
        if files[key]['sha256'] != card['splits'][key]['sha256']:
            raise ValueError('Profile input differs from the original declared ' + key + ' population')
    if files['template']['sha256'] != _digest(ROOT/'experiments/data/instruction_interface_v1.jinja')['sha256']:
        raise ValueError('The original declared template bytes are required')
    sources = {name: _digest(ROOT/name) for name in SOURCES}
    argv = [request['interpreter'], str(ROOT/'scripts/run_chapter09_spark_sft.py'),
        '--model', MODEL, '--revision', REVISION, '--tokenizer-revision', REVISION]
    for key in ('template', 'train', 'dev', 'output'):
        argv.extend(['--'+key, request[key]])
    for key, value in RECIPE.items():
        argv.extend(['--'+key.replace('_', '-'), str(value)])
    for key in ('snapshot_max_bytes', 'work_limits', 'work_journal_max_bytes',
                'work_journal', 'snapshot_io_limits', 'snapshot_io_ledger', 'environment_lock'):
        argv.extend(['--'+key.replace('_', '-'), str(request[key])])
    record = dict(schema='dongxi-native-sft-profile-preparation-v1', stage_id=STAGE,
        request=request, argv=argv, cwd=str(ROOT), input_bindings=files,
        source_bindings=sources, sft_limits=limits, snapshot_io_contract=io,
        known_snapshot_io_requirements=required,
        status='prepared-source-only-not-launch-authorized', launch_authorized=False,
        jobs_started=0, actual_external_result=None,
        unknown=['actual pinned model bytes/interface', 'tokenizer-derived target/position requirements',
                 'allowance/snapshot physical sufficiency', 'operator authority',
                 'actual CUDA/BF16/profile and GPU clearance'],
        scope='fixed native command; explicit allowances are declarations, not encoded measurements or authenticated approval')
    record['preparation_sha256'] = canonical_hash(record)
    return record


def _native_probe(exclude):
    return dict(**host_memory(), conflict_scan=conflicts(exclude=exclude))


def _validate_observation(value):
    if type(value) is not dict or set(value) != {'available_bytes', 'source', 'conflict_scan'}:
        raise ValueError('Explicit labelled memory/conflict observation required')
    _integer(value['available_bytes'], 'available_bytes')
    if type(value['source']) is not str or not 1 <= len(value['source']) <= 256:
        raise ValueError('Resource source label required')
    scan = value['conflict_scan']
    fields = {'source', 'conflicts', 'unreadable', 'scanned', 'raw_command_lines_retained', 'environments_read'}
    if type(scan) is not dict or set(scan) != fields:
        raise ValueError('Complete sanitized conflict clearance required')
    if type(scan['source']) is not str or not 1 <= len(scan['source']) <= 256:
        raise ValueError('Conflict source label required')
    for key in ('unreadable', 'scanned'): _integer(scan[key], key)
    if scan['raw_command_lines_retained'] is not False or scan['environments_read'] is not False:
        raise ValueError('Do not retain arbitrary command lines/environments')
    if type(scan['conflicts']) is not list or len(scan['conflicts']) > 1024:
        raise ValueError('Bounded conflict observations required')
    for row in scan['conflicts']:
        if type(row) is not dict or set(row) != {'pid', 'rss_bytes', 'reasons'}:
            raise ValueError('Sanitized conflict identity required')
        _integer(row['pid'], 'pid', 1); _integer(row['rss_bytes'], 'rss_bytes')
        if type(row['reasons']) is not list or not row['reasons'] or not set(row['reasons']) <= {
                'known-model-process-signature', 'large-resident-process'}:
            raise ValueError('Known conflict reasons required')
    return value


def _owned_descendants(pid):
    if pid is None: return []
    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
        if len(children) > 128: raise ValueError('Owned observation exceeds128descendants')
        return [dict(pid=p.pid, create_time=p.create_time()) for p in children]
    except psutil.NoSuchProcess:
        return []


class _Channel:
    """Atomic <=32KiB datagrams; no blocking framed/pickle recv in controller."""
    def __init__(self, endpoint):
        self.endpoint = endpoint
        endpoint.setblocking(False)

    def send(self, value):
        raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) > 32768: raise ValueError('Owned IPC observation exceeds32KiB')
        self.endpoint.send(raw)

    def poll(self, timeout=0):
        return bool(select.select([self.endpoint], [], [], timeout)[0])

    def recv(self):
        raw = self.endpoint.recv(32769)
        if not raw or len(raw) > 32768: raise ValueError('Malformed/oversized owned datagram')
        return json.loads(raw)

    def close(self):
        self.endpoint.close()


def _channel_pair():
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    return _Channel(first), _Channel(second)


def _leader_exited(process):
    # WNOWAIT retains the owned leader PID until the final session drain. A
    # poll()/wait() here would release that identity before signalling workers.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _probe_worker(requests, results, probe):
    while True:
        request = requests.get()
        if request is None: return
        identifier, excluded, leader = request
        try:
            value = _validate_observation(probe(tuple(excluded) + (os.getpid(),)))
            results.send(('observation', identifier, value, _owned_descendants(leader)))
        except BaseException as error:
            results.send(('error', identifier, dict(type=type(error).__name__, message=str(error)[:512])))
            return


def _logger_worker(requests, results, output, hook):
    handle = None
    try:
        handle = (Path(output)/'events.jsonl').open('x', encoding='utf-8')
        results.send(('ready',))
        while True:
            message = requests.get()
            if message is None: return
            kind, identifier, value = message
            if kind == 'event':
                handle.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
                handle.flush()
                if hook is not None: hook(value)
                os.fsync(handle.fileno())
            else:
                with (Path(output)/'result.json').open('x', encoding='utf-8') as result:
                    result.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
                    result.flush(); os.fsync(result.fileno())
            results.send(('ack', identifier))
    except BaseException as error:
        results.send(('error', dict(type=type(error).__name__, message=str(error)[:512])))
    finally:
        if handle is not None: handle.close()


def _finish_helper(worker, seconds=.15):
    if worker is None or worker.pid is None:
        return dict(pid=None, started=False, actual_exit_code=None, still_alive=False,
                    errors=[], cleanup_complete=True)
    errors = []
    for name in ('terminate', 'kill'):
        try:
            if not worker.is_alive(): break
            getattr(worker, name)()
        except OSError as error:
            errors.append(name+': '+type(error).__name__)
        try: worker.join(timeout=seconds)
        except OSError as error: errors.append('join: '+type(error).__name__)
    try: alive = worker.is_alive()
    except OSError as error:
        alive = None; errors.append('state: '+type(error).__name__)
    return dict(pid=worker.pid, started=True, actual_exit_code=worker.exitcode,
                still_alive=alive, errors=errors,
                cleanup_complete=alive is False and worker.exitcode is not None)


def _cleanup_descendants(rows, result):
    """Signal only retained owned PID/create-time handles, never conflict PIDs."""
    for row in rows:
        try:
            member = psutil.Process(row['pid'])
            if member.create_time() != row['create_time']: continue
            member.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            result.send(('error', type(error).__name__))
    result.send(('done',))


def _supervise(argv, output, *, native, seconds, probe, journal_hook=None,
               probe_seconds=.25, sampling_seconds=.1, term_seconds=.1,
               reap_seconds=.5, logging_seconds=.25, operator_declaration=None,
               native_stage='profile'):
    """Private engine. Public routes supply closed native or inert CPU argv."""
    for name, amount in dict(seconds=seconds, probe_seconds=probe_seconds,
            sampling_seconds=sampling_seconds, term_seconds=term_seconds,
            reap_seconds=reap_seconds, logging_seconds=logging_seconds).items():
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0:
            raise ValueError('Finite positive watchdog bound required: '+name)
    if native_stage not in NATIVE_DEADLINE_ENVELOPES or (not native and native_stage != 'profile'):
        raise ValueError('Unknown or nonnative stage deadline envelope')
    maximum_seconds = NATIVE_DEADLINE_ENVELOPES[native_stage] if native else 2
    if seconds > maximum_seconds or max(probe_seconds, logging_seconds, reap_seconds) > 2 or term_seconds > 1:
        raise ValueError('Watchdog timing envelope exceeded')
    if (os.name != 'posix' or 'fork' not in mp.get_all_start_methods()
            or not hasattr(os, 'waitid') or not hasattr(os, 'WNOWAIT')):
        raise RuntimeError('This owned-session adapter requires POSIX fork and non-reaping waitid')
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError('Use the isolated nonthreaded CLI/controller; arbitrary threaded callers are unsupported')
    output = _new_target(str(_path(str(output))))
    output.mkdir(mode=0o700)
    ctx = mp.get_context('fork')
    # Initial helpers start before any Queue.put creates feeder threads. Late
    # cleanup/writer targets use spawn, not a fork of that threaded controller.
    late = mp.get_context('spawn')
    logs = ctx.Queue(maxsize=64); probes = ctx.Queue(maxsize=2)
    logs.cancel_join_thread(); probes.cancel_join_thread()
    log_receive, log_send = _channel_pair()
    probe_receive, probe_send = _channel_pair()
    logger = ctx.Process(target=_logger_worker, args=(logs, log_send, str(output), journal_hook), daemon=True)
    observer = ctx.Process(target=_probe_worker, args=(probes, probe_send, probe), daemon=True)
    began = time.monotonic()
    events, observations, descendants = [], [], {}
    process = None; out = err = None; reason = failure = journal_error = None
    pending_probe = None; pending_logs = {}; event_id = 0; probe_id = 0
    native_started = None; minimum = None; exit_code = None; cleanup_errors = []
    helper_cleanup = []

    def emit(kind, **fields):
        nonlocal event_id, journal_error
        row = dict(event=kind, utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-began, **fields)
        events.append(row); event_id += 1
        try:
            logs.put_nowait(('event', event_id, row))
            pending_logs[event_id] = time.monotonic()
        except queue.Full:
            journal_error = dict(type='Full', message='Bounded journal queue exhausted')

    def poll_logs():
        nonlocal journal_error
        while log_receive.poll():
            message = log_receive.recv()
            if message[0] == 'ack': pending_logs.pop(message[1], None)
            elif message[0] == 'error': journal_error = message[1]
        if pending_logs and time.monotonic()-min(pending_logs.values()) > logging_seconds:
            journal_error = dict(type='TimeoutError', message='Owned journal/fsync worker timed out')
        if not logger.is_alive() and journal_error is None:
            journal_error = dict(type='ProcessExit', message='Owned journal worker exited')

    def request_probe():
        nonlocal pending_probe, probe_id
        probe_id += 1
        excluded = [os.getpid(), logger.pid, observer.pid]
        if process is not None: excluded.append(process.pid)
        probes.put_nowait((probe_id, excluded, None if process is None else process.pid))
        pending_probe = (probe_id, time.monotonic())

    def take_probe():
        nonlocal pending_probe, reason, failure, minimum
        if not probe_receive.poll(): return False
        message = probe_receive.recv()
        if pending_probe is None or message[1] != pending_probe[0]:
            raise RuntimeError('Unexpected owned observer response')
        pending_probe = None
        if message[0] == 'error':
            failure = message[2]
            reason = 'observer-interrupted' if failure['type'] == 'KeyboardInterrupt' else 'observer-failure'
            emit('observer-error', failure=failure); return True
        value = message[2]; observations.append(value)
        minimum = value['available_bytes'] if minimum is None else min(minimum, value['available_bytes'])
        for row in message[3]: descendants[(row['pid'], row['create_time'])] = row
        emit('resource-sample', observation=value)
        if value['available_bytes'] < 25*GIB: reason = 'host-reserve-below-threshold'
        elif value['conflict_scan']['conflicts']: reason = 'conflicting-process'
        elif value['conflict_scan']['unreadable']: reason = 'incomplete-clearance'
        return True

    try:
        # Both starts are inside ownership cleanup: a second-start failure must
        # not strand the already started first helper.
        logger.start(); observer.start()
        if not log_receive.poll(logging_seconds) or log_receive.recv()[0] != 'ready':
            reason = 'journal-failure'; journal_error = dict(type='TimeoutError', message='Journal bootstrap failed')
        emit('preflight', native_model_stage=native, command=argv, operator_declaration=operator_declaration)
        request_probe()
        while reason is None and (pending_probe is not None or pending_logs):
            poll_logs(); take_probe()
            if journal_error: reason = 'journal-failure'
            elif pending_probe and time.monotonic()-pending_probe[1] >= probe_seconds:
                reason = 'observer-timeout'; failure = dict(type='TimeoutError', message='Preflight observer timed out')
            time.sleep(.002)
        if reason is None:
            out = (output/'stdout.txt').open('x')
            err = (output/'stderr.txt').open('x')
            environment = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            if not native: environment['CUDA_VISIBLE_DEVICES'] = ''
            native_started = time.monotonic()
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=out, stderr=err,
                                       start_new_session=True, shell=False)
            emit('child-started', child_pid=process.pid, owned_session=process.pid)
            request_probe(); next_sample = time.monotonic()+sampling_seconds
            while reason is None:
                now = time.monotonic()
                # Deadline evaluation precedes all response processing and never
                # calls a resource probe, fsync or logging callback itself.
                if now-native_started >= seconds:
                    reason = 'external-deadline'; break
                if _leader_exited(process): break
                poll_logs()
                if journal_error: reason = 'journal-failure'; break
                take_probe()
                if reason: break
                if pending_probe and now-pending_probe[1] >= probe_seconds:
                    reason = 'observer-timeout'; failure = dict(type='TimeoutError', message='Owned observer timed out'); break
                if pending_probe is None and now >= next_sample:
                    request_probe(); next_sample = now+sampling_seconds
                time.sleep(min(.005, max(0., seconds-(time.monotonic()-native_started))))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error)[:512])
        reason = 'controller-interrupted' if isinstance(error, KeyboardInterrupt) else 'controller-failure'
        emit('controller-error', failure=failure)
    finally:
        if process is not None:
            # Drain even when the leader exits0 or promptly obeys TERM. Keep
            # its unreaped PID anchor until both session signals are issued;
            # an unrelated new session cannot reuse that PGID meanwhile.
            cleanup_reason = reason or 'normal-leader-exit-session-drain'
            emit('owned-signal', child_pid=process.pid, signal=signal.SIGTERM, reason=cleanup_reason)
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            except OSError as error: cleanup_errors.append('TERM: '+type(error).__name__)
            time.sleep(term_seconds)
            emit('owned-signal', child_pid=process.pid, signal=signal.SIGKILL, reason=cleanup_reason)
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            except OSError as error: cleanup_errors.append('KILL: '+type(error).__name__)
            try: exit_code = process.wait(timeout=reap_seconds)
            except subprocess.TimeoutExpired: cleanup_errors.append('Owned leader reap timed out')
            except OSError as error: cleanup_errors.append('REAP: '+type(error).__name__)
        # Escaped *observed* descendants use PID/create-time identity checks in
        # a bounded helper. Unobserved/reparented hostile workers are not covered.
        if descendants:
            received, sent = _channel_pair()
            cleanup = late.Process(target=_cleanup_descendants, args=(list(descendants.values()), sent), daemon=True)
            try:
                cleanup.start()
                if not received.poll(reap_seconds): cleanup_errors.append('Observed descendant cleanup timed out')
                else:
                    message = received.recv()
                    if message[0] != 'done': cleanup_errors.append(str(message))
            except BaseException as error:
                cleanup_errors.append('DESCENDANTS: '+type(error).__name__)
            finally:
                try: helper_cleanup.append(_finish_helper(cleanup))
                except OSError as error: cleanup_errors.append('CLEANUP HELPER: '+type(error).__name__)
                received.close(); sent.close()
        try: helper_cleanup.append(_finish_helper(observer))
        except OSError as error: cleanup_errors.append('OBSERVER HELPER: '+type(error).__name__)
        if out is not None: out.close()
        if err is not None: err.close()
    poll_logs()
    if any(not row['cleanup_complete'] for row in helper_cleanup):
        cleanup_errors.append('Owned helper cleanup incomplete/unknown')
    emit('actual-exit', child_pid=None if process is None else process.pid,
         actual_exit_code=exit_code, stop_reason=reason, failure=failure,
         journal_error=journal_error, cleanup_errors=cleanup_errors)
    result = dict(schema='dongxi-native-profile-watchdog-result-v1', native_model_stage=native,
        actual_native_profile_executed=native and process is not None,
        child_command=list(argv), child_pid=None if process is None else process.pid,
        actual_exit_code=exit_code, signal=-exit_code if exit_code is not None and exit_code < 0 else None,
        stop_reason=reason, failure=failure, journal_error=journal_error,
        cleanup_errors=cleanup_errors, observed_owned_descendants=list(descendants.values()),
        helper_cleanup=helper_cleanup,
        observations=observations, minimum_sampled_available_bytes=minimum,
        resource_measurement='labelled point samples, not continuous memory enforcement or GPU-idle proof',
        seconds=time.monotonic()-began, child_seconds=None if native_started is None else time.monotonic()-native_started,
        limits=dict(external_seconds=seconds, reserve_bytes=25*GIB, probe_seconds=probe_seconds,
            logging_seconds=logging_seconds, term_seconds=term_seconds, reap_seconds=reap_seconds),
        status='refused' if process is None else ('failed' if reason or exit_code != 0 or cleanup_errors or journal_error else 'completed'),
        in_memory_events=events, operator_declaration=operator_declaration,
        operator_declaration_scope='caller text is NOT authenticated authorization',
        final_record_acknowledgment='unknown-at-write; returned final_record_retained reports the separate observed writer ACK',
        output=str(output), scope='owned native/inert process only; no quota/cgroup/hostile-tree containment or model success inference')
    # Finish logging with a bounded wait. A failed logger never becomes success.
    stop = time.monotonic()+logging_seconds
    while pending_logs and journal_error is None and time.monotonic() < stop:
        poll_logs(); time.sleep(.002)
    if journal_error is not None:
        result['journal_error'] = journal_error; result['status'] = 'refused' if process is None else 'failed'
        helper_cleanup.append(_finish_helper(logger))
        # Preserve a separate final record using the same isolated writer but no
        # injected callback. Failure of this attempt is returned, not concealed.
        log_receive.close(); log_send.close()
        log_receive, log_send = _channel_pair()
        logger = late.Process(target=_final_writer, args=(str(output), result, log_send), daemon=True)
        logger.start()
        result['final_record_retained'] = bool(log_receive.poll(logging_seconds) and log_receive.recv()[0] == 'done')
    else:
        event_id += 1; logs.put_nowait(('result', event_id, result))
        done = False; stop = time.monotonic()+logging_seconds
        while time.monotonic() < stop and not done:
            if log_receive.poll(.002):
                message = log_receive.recv()
                if message[0] == 'ack' and message[1] == event_id: done = True
                elif message[0] == 'error': result['journal_error'] = message[1]; break
        result['final_record_retained'] = done
        if not done:
            result['status'] = 'refused' if process is None else 'failed'
            result['journal_error'] = result['journal_error'] or dict(type='TimeoutError', message='Final logging timed out')
    helper_cleanup.append(_finish_helper(logger))
    if any(not row['cleanup_complete'] for row in helper_cleanup):
        result['cleanup_errors'].append('Owned helper cleanup incomplete/unknown')
        result['status'] = 'refused' if process is None else 'failed'
    log_receive.close(); log_send.close(); probe_receive.close(); probe_send.close()
    feeder_shutdown = []
    for name, owned_queue in (('logging', logs), ('observer', probes)):
        owned_queue.close()
        feeder = getattr(owned_queue, '_thread', None)
        if feeder is not None: feeder.join(timeout=logging_seconds)
        feeder_shutdown.append(dict(queue=name, still_alive=feeder is not None and feeder.is_alive()))
    result['queue_feeder_shutdown'] = feeder_shutdown
    if any(row['still_alive'] for row in feeder_shutdown):
        result['cleanup_errors'].append('Owned queue feeder shutdown incomplete; do not reuse controller')
        result['status'] = 'refused' if process is None else 'failed'
    return result


def _final_writer(output, value, result):
    try:
        with (Path(output)/'result.json').open('x') as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
            handle.flush(); os.fsync(handle.fileno())
        result.send(('done',))
    except BaseException as error:
        result.send(('error', type(error).__name__))


def execute_native_profile(prepared, *, operator_declaration, supervision_output):
    """Explicit operator route; NEVER called by preparation or course verification.

    Caller text is retained, not authenticated. The operator remains responsible
    for separately authorized acquisition, encoded sizing and actual GPU clearance.
    """
    if type(operator_declaration) is not str or not 12 <= len(operator_declaration) <= 4096:
        raise ValueError('Explicit separately scoped operator declaration required')
    if type(prepared) is not dict or 'request' not in prepared:
        raise ValueError('Current preparation required')
    current = prepare_native_profile(prepared['request'])
    if canonical_hash(current) != canonical_hash(prepared):
        raise ValueError('Prepared source/inputs/command changed; prepare and approve again')
    return _supervise(current['argv'], supervision_output, native=True, seconds=900,
                      probe=_native_probe, operator_declaration=operator_declaration)


def injected_cpu_clearance(exclude):
    """Declared inert-test control, NOT an actual host/GPU measurement."""
    return dict(available_bytes=26*GIB, source='injected-inert-CPU-control', conflict_scan=dict(
        source='injected-complete-clearance', conflicts=[], unreadable=0, scanned=0,
        raw_command_lines_retained=False, environments_read=False))


def supervise_cpu_control(mode, output, *, seconds=.25, probe=injected_cpu_clearance,
                          journal_hook=None, **timing):
    """Same engine, only closed inert children; no arbitrary argv/model imports."""
    if mode not in INERT_MODES: raise ValueError('Unknown inert CPU control')
    argv = [sys.executable, '-m', 'dongxi_llms.native_profile_supervisor', '--inert-child', mode]
    return _supervise(argv, output, native=False, seconds=seconds, probe=probe,
                      journal_hook=journal_hook, **timing)


def _inert_child(mode):
    if mode == 'success': print('inert CPU child completed', flush=True); return 0
    if mode == 'exit7': print('intentional inert exit7', flush=True); return 7
    if mode == 'ignore-term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode in ('worker', 'exit-with-worker', 'term-with-worker'):
        worker = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(5)'])
        print('owned-worker='+str(worker.pid), flush=True)
        time.sleep(.1)  # Give this authored worker time to install its handler.
        if mode == 'exit-with-worker': return 0
    time.sleep(5)
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fixed inert CPU test children only')
    parser.add_argument('--inert-child', choices=sorted(INERT_MODES), required=True)
    raise SystemExit(_inert_child(parser.parse_args().inert_child))

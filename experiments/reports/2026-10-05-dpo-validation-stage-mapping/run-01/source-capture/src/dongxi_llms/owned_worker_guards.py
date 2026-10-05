"""Fixed CPU worker/disk controls, not a production or arbitrary-command launcher.

Owned descendants are sampled with PID/create-time handles. A new process group
is also stopped after leader exit. This is not a cgroup or hostile-tree sandbox.
File size is hard per child-written file; aggregate bytes/free space are sampled.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

import psutil

from .campaign_supervisor import GIB, Journal, Limits, conflicts, host_memory
from .run_identity import file_digest

MODES = frozenset(("success", "blocked-tree", "escaped-worker",
                   "parent-exits-with-worker", "single-file-overflow", "aggregate-file-growth"))


@dataclass(frozen=True)
class DiskLimits:
    per_file_bytes: int = 64 * 1024
    aggregate_bytes: int = 256 * 1024
    minimum_free_bytes: int = 16 * 1024 * 1024

    def validate(self):
        if any(type(x) is not int or x <= 0 for x in asdict(self).values()):
            raise ValueError("Positive integer disk bounds required")
        if not 4096 <= self.per_file_bytes <= 1024**2:
            raise ValueError("CPU fixture per-file cap must be 4 KiB to 1 MiB")
        if not self.per_file_bytes <= self.aggregate_bytes <= 8 * 1024**2:
            raise ValueError("CPU fixture aggregate cap must be file cap to 8 MiB")
        return self


def disk_sample(output):
    """FD-anchored sample: refuse links, count inode payload once, label sampling.

    This is not a hard filesystem quota. Concurrent growth can change a sample;
    swapped child directories are opened relative to their verified parent FD.
    """
    seen, total = set(), 0
    def visit(fd):
        nonlocal total
        with os.scandir(fd) as entries:
            for entry in entries:
                value = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(value.st_mode):
                    raise ValueError("Symlinks are forbidden in the fixture output")
                if stat.S_ISDIR(value.st_mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try: visit(child)
                    finally: os.close(child)
                elif stat.S_ISREG(value.st_mode):
                    identity = (value.st_dev, value.st_ino)
                    if identity not in seen:
                        total += value.st_size
                        seen.add(identity)
                else:
                    raise ValueError("Only regular fixture files/directories are allowed")
    if Path(output).is_symlink():
        raise ValueError("Symlinked fixture root is forbidden")
    root = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(root)
        filesystem = os.fstatvfs(root)
        free = filesystem.f_bavail * filesystem.f_frsize
    finally:
        os.close(root)
    return {"source": "actual-sampled-private-files-and-filesystem-free",
            "artifact_bytes": total, "free_bytes": free}


def _live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        # Unknown is not proven dead. Try identity-safe signaling and report it.
        return True


class OwnedWorkers:
    """Only self-spawned leader descendants; discovery results are never adopted."""
    def __init__(self, process):
        self.leader = process
        self.signal_errors = []
        try:
            self.root_handle = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            self.root_handle = None
        except psutil.Error as error:
            self.root_handle = None
            self.signal_errors.append({"during": "leader-identity", "type": type(error).__name__})
        self.handles = {}
        self.identities = {}

    def sample(self):
        try:
            leader = self.root_handle
            members = [] if leader is None or not leader.is_running() else [leader, *leader.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        except psutil.Error as error:
            self.signal_errors.append({"during": "worker-discovery", "type": type(error).__name__})
            members = []
        for member in members:
            try:
                identity = (member.pid, member.create_time())
                if identity not in self.handles:
                    info = {"pid": member.pid, "create_time": identity[1],
                            "observed_pgid": None, "observed_sid": None}
                    try:
                        info.update(observed_pgid=os.getpgid(member.pid), observed_sid=os.getsid(member.pid))
                    except ProcessLookupError:
                        continue
                    except OSError as error:
                        self.signal_errors.append({"during": "worker-group-observation", "type": type(error).__name__})
                    self.handles[identity] = member
                    self.identities[identity] = info
            except psutil.NoSuchProcess:
                continue
            except (OSError, psutil.Error) as error:
                self.signal_errors.append({"during": "worker-identity", "type": type(error).__name__})
        return [dict(self.identities[key]) for key in self.handles]

    def active(self):
        return [member for member in self.handles.values() if _live(member)]

    def stop(self, limits, journal, reason):
        def record(kind, **fields):
            try:
                journal.event(kind, **fields)
            except BaseException as error:
                self.signal_errors.append({"during": "cleanup-journal", "type": type(error).__name__})

        self.sample()
        for signum, grace in ((signal.SIGTERM, limits.term_grace_seconds),
                              (signal.SIGKILL, limits.kill_reap_seconds)):
            # The group belongs to this start_new_session invocation. An exited
            # leader's same-group workers can still be signalled. Never use a
            # conflict-scan PID here. psutil protects retained handles from reuse.
            own_group = False
            for member in self.active():
                try:
                    own_group = own_group or (member.is_running() and os.getpgid(member.pid) == self.leader.pid)
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass
                except (OSError, psutil.Error) as error:
                    self.signal_errors.append({"during": "group-identity", "type": type(error).__name__})
            if own_group:
                record("owned-group-signal", group=self.leader.pid, signal=signum, reason=reason)
                try:
                    os.killpg(self.leader.pid, signum)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    self.signal_errors.append({"during": "group-signal", "type": type(error).__name__})
            for member in self.active():
                record("owned-worker-signal", pid=member.pid, signal=signum, reason=reason)
                try:
                    member.send_signal(signum)
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as error:
                    self.signal_errors.append({"during": "worker-signal", "pid": member.pid,
                                               "type": type(error).__name__})
            # Popen retains ownership until waitpid reaps it; no psutil discovery
            # failure may strand this invocation's direct child.
            if self.leader.poll() is None:
                try: self.leader.send_signal(signum)
                except ProcessLookupError: pass
                except OSError as error:
                    self.signal_errors.append({"during": "leader-signal", "type": type(error).__name__})
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                self.leader.poll()  # Reap the owned leader, not an unrelated PID.
                if not self.active():
                    break
                self.sample()
                time.sleep(min(.01, max(0., deadline - time.monotonic())))
        try:
            self.leader.wait(timeout=limits.kill_reap_seconds)
        except subprocess.TimeoutExpired:
            self.signal_errors.append({"during": "leader-reap", "type": "TimeoutExpired"})
        return self.final_states()

    def final_states(self):
        rows = []
        for key, member in self.handles.items():
            try:
                state = member.status() if member.is_running() else "gone"
            except psutil.NoSuchProcess:
                state = "gone"
            except psutil.Error as error:
                state = "unknown-" + type(error).__name__
            rows.append({**self.identities[key], "state": state,
                         "live": state not in ("gone", psutil.STATUS_ZOMBIE)})
        return rows


def supervise_owned_fixture(mode, output, *, limits=None, disk_limits=None,
                            memory_probe=host_memory, conflict_probe=conflicts,
                            disk_probe=disk_sample):
    """Supervise fixed cooperative CPU fixtures; no model command can be supplied."""
    if sys.platform != "linux":
        raise RuntimeError("These actual process/disk controls require Linux")
    limits, disk_limits = (limits or Limits()).validate(), (disk_limits or DiskLimits()).validate()
    if mode not in MODES:
        raise ValueError("Only fixed CPU worker fixtures are allowed")
    output = Path(output)
    # A caller cannot redirect writes through an existing symlink component.
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("Symlinked output paths are forbidden")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "events.jsonl")
    process, owned, reason, failure = None, None, None, None
    observations, final_states = [], []
    fallback_errors = []
    command = [sys.executable, "-m", "dongxi_llms.owned_worker_guards", "--fixture", mode,
               "--output", str(output), "--per-file-bytes", str(disk_limits.per_file_bytes)]
    out = err = None
    started = time.monotonic()

    def observe():
        workers = [] if owned is None else owned.sample()
        memory = memory_probe()
        disk = disk_probe(output)
        if (set(memory) != {"source", "available_bytes"} or not isinstance(memory["source"], str)
            or type(memory["available_bytes"]) is not int or memory["available_bytes"] < 0
            or set(disk) != {"source", "artifact_bytes", "free_bytes"}
            or not isinstance(disk["source"], str)
            or any(type(disk[key]) is not int or disk[key] < 0 for key in ("artifact_bytes", "free_bytes"))):
            raise ValueError("Invalid labelled memory/disk sample")
        scan = conflict_probe(exclude=[row["pid"] for row in workers])
        if (set(scan) != {"source", "conflicts", "unreadable", "scanned", "raw_command_lines_retained", "environments_read"}
            or not isinstance(scan["source"], str) or not isinstance(scan["conflicts"], list)
            or any(type(scan[key]) is not int or scan[key] < 0 for key in ("unreadable", "scanned"))
            or scan["raw_command_lines_retained"] is not False or scan["environments_read"] is not False
            or any(set(row) != {"pid", "rss_bytes", "reasons"} or type(row["pid"]) is not int
                   or type(row["rss_bytes"]) is not int or row["rss_bytes"] < 0
                   or not isinstance(row["reasons"], list)
                   or not set(row["reasons"]) <= {"known-model-process-signature", "large-resident-process"}
                   for row in scan["conflicts"])):
            raise ValueError("Invalid sanitized conflict observation")
        row = {"memory": memory, "disk": disk, "conflict_scan": scan, "owned_workers": workers}
        observations.append(row)
        journal.event("resource-sample", **row)
        if owned is not None and owned.signal_errors: return "owned-discovery-incomplete"
        if memory["available_bytes"] < limits.reserve_bytes: return "host-reserve-below-threshold"
        if disk["free_bytes"] < disk_limits.minimum_free_bytes: return "filesystem-free-below-threshold"
        if disk["artifact_bytes"] >= disk_limits.aggregate_bytes: return "sampled-artifact-byte-cap"
        if scan["conflicts"] or scan["unreadable"]: return "conflicting-or-uninspectable-process"
        return None

    try:
        journal.event("preflight", fixture=mode, limits=asdict(limits), disk_limits=asdict(disk_limits))
        reason = observe()
        if reason is None:
            out, err = (output / "stdout.txt").open("x"), (output / "stderr.txt").open("x")
            env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "LANG": "C.UTF-8",
                   "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1"}
            process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            owned = OwnedWorkers(process)
            journal.event("child-started", child_pid=process.pid, command=command)
            while True:
                reason = observe()
                if reason is not None: break
                if process.poll() is not None:
                    # Successful leader exit does not imply workers stopped.
                    if owned.active(): reason = "leader-exited-with-owned-workers"
                    break
                if time.monotonic() - started >= limits.seconds:
                    reason = "external-deadline"
                    break
                time.sleep(limits.sampling_seconds)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:512]}
        reason = "observer-interrupted" if isinstance(error, KeyboardInterrupt) else "supervisor-failure"
    finally:
        try:
            if owned is not None:
                final_states = owned.stop(limits, journal, reason or "normal-shutdown-check")
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)[:512]}
            reason = "cleanup-failure"
        finally:
            # Covers failure before OwnedWorkers was assigned, and unexpected
            # discovery/cleanup exceptions. Never signal a naked inspected PID.
            try:
                if process is not None and process.poll() is None:
                    for signum, grace in ((signal.SIGTERM, limits.term_grace_seconds), (signal.SIGKILL, limits.kill_reap_seconds)):
                        try:
                            process.send_signal(signum)
                            process.wait(timeout=grace)
                            break
                        except ProcessLookupError: break
                        except subprocess.TimeoutExpired: continue
                        except OSError as error:
                            fallback_errors.append({"during": "leader-fallback", "signal": signum, "type": type(error).__name__})
                            continue
            finally:
                for handle in (out, err):
                    if handle is not None:
                        try: handle.close()
                        except OSError as error:
                            fallback_errors.append({"during": "output-close", "type": type(error).__name__})
        code = None if process is None else process.poll()
        if any(row["live"] for row in final_states): reason = "owned-workers-still-live"
        if process is not None and code is None: reason = "owned-leader-still-live"
        result = {"fixture": mode, "limits": asdict(limits), "disk_limits": asdict(disk_limits),
                  "actual_leader_exit_code": code, "stop_reason": reason, "failure": failure,
                  "status": "gate-rejected" if process is None else "failed" if reason or failure or code != 0 or fallback_errors or (owned is not None and owned.signal_errors) else "completed",
                  "owned_final_states": final_states, "observations": observations,
                  "cleanup_errors": [*fallback_errors, *([] if owned is None else owned.signal_errors)],
                  "minimum_sampled_available_bytes": min((r["memory"]["available_bytes"] for r in observations), default=None),
                  "minimum_sampled_free_bytes": min((r["disk"]["free_bytes"] for r in observations), default=None),
                  "maximum_sampled_artifact_bytes": max((r["disk"]["artifact_bytes"] for r in observations), default=None),
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "fixed CPU controls; sampled owned discovery/group cleanup; not cgroup or production containment"}
        try:
            journal.event("actual-exit", exit_code=code, stop_reason=reason, owned_final_states=final_states, failure=failure)
        except BaseException as error:
            result["journal_error"] = {"type": type(error).__name__}
            result["status"] = "failed"
        finally:
            journal.close()
    result["journal"] = {"path": str(journal.path), "sha256": file_digest(journal.path)}
    with (output / "result.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _fixture(mode, output, per_file_bytes):
    resource.setrlimit(resource.RLIMIT_FSIZE, (per_file_bytes, per_file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    if mode == "success": return 0
    if mode in ("single-file-overflow", "aggregate-file-growth"):
        index = 0
        while True:
            with (output / f"fixture-{index:04d}.bin").open("xb", buffering=0) as handle:
                chunks = per_file_bytes // 4096 + 1 if mode == "single-file-overflow" else 1
                for _ in range(chunks): handle.write(b"x" * 4096)
            index += 1
            time.sleep(.005)
    # Workers can optionally leave the leader's process group. Their identity
    # is retained through sampled ancestry before that known parent exits.
    worker = subprocess.Popen([sys.executable, "-m", "dongxi_llms.owned_worker_guards", "--worker"],
                              start_new_session=(mode == "escaped-worker"))
    print(json.dumps({"worker_pid": worker.pid}), flush=True)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "parent-exits-with-worker":
        time.sleep(.25)
        return 0
    while True: time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(MODES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-file-bytes", type=int)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True: time.sleep(1)
    if args.fixture is None or args.output is None or args.per_file_bytes is None:
        parser.error("Fixed fixture, private output and file cap required")
    DiskLimits(per_file_bytes=args.per_file_bytes, aggregate_bytes=max(256 * 1024, args.per_file_bytes)).validate()
    raise SystemExit(_fixture(args.fixture, args.output, args.per_file_bytes))

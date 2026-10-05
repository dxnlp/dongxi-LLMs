"""Independent mocked ownership/failure checks; never signal a real process.

The handles below are authored controls, not measured PID-reuse incidents or
production containment evidence. Temporary disk controls contain no user data.
"""
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import psutil

from dongxi_llms import owned_worker_guards as guards


class Handle:
    def __init__(self, pid, *, alive=True, created=10., status_error=None,
                 discovery_error=None, signal_error=None):
        self.pid, self.alive, self.created = pid, alive, created
        self.status_error, self.discovery_error = status_error, discovery_error
        self.signal_error = signal_error
        self.members, self.signals = [], []

    def create_time(self):
        return self.created

    def is_running(self):
        return self.alive

    def status(self):
        if self.status_error is not None:
            raise self.status_error
        return psutil.STATUS_SLEEPING

    def children(self, recursive=False):
        if self.discovery_error is not None:
            raise self.discovery_error
        return list(self.members)

    def send_signal(self, signum):
        self.signals.append(signum)
        if self.signal_error is not None:
            raise self.signal_error
        if signum == signal.SIGKILL:
            self.alive = False


class Child:
    """Mocked direct Popen ownership; TERM ignored, KILL exits."""
    def __init__(self, pid=1001, *, alive=False):
        self.pid, self.alive, self.signals = pid, alive, []

    def poll(self):
        return None if self.alive else -signal.SIGKILL

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired("authored-fixed-child-control", timeout)
        return self.poll()

    def send_signal(self, signum):
        self.signals.append(signum)
        if signum == signal.SIGKILL:
            self.alive = False


def memory():
    return {"available_bytes": 100 * guards.GIB,
            "source": "injected-independent-memory-control"}


def scan(**kwargs):
    return {"source": "injected-independent-clearance-control", "conflicts": [],
            "unreadable": 0, "scanned": 0, "raw_command_lines_retained": False,
            "environments_read": False}


@unittest.skipUnless(guards.sys.platform == "linux", "Linux-only owned worker contract")
class OwnershipFailureTests(unittest.TestCase):
    def limits(self):
        return guards.Limits(seconds=.1, term_grace_seconds=.01, kill_reap_seconds=.01)

    def owned(self, *members, root=None, child=None):
        child = child or Child()
        root = root or Handle(child.pid, alive=False)
        with patch.object(guards.psutil, "Process", return_value=root):
            owned = guards.OwnedWorkers(child)
        for member in members:
            identity = (member.pid, member.create_time())
            owned.handles[identity] = member
            owned.identities[identity] = {"pid": member.pid, "create_time": identity[1]}
        return owned

    def test_reused_leader_pid_is_never_reacquired_or_adopted(self):
        original = Handle(1001, alive=False, created=10.)
        unrelated = Handle(1001, alive=True, created=99.)
        unrelated.members = [Handle(9999)]
        with patch.object(guards.psutil, "Process", side_effect=[original, unrelated]) as factory, \
             patch.object(guards.os, "killpg") as group:
            owned = guards.OwnedWorkers(Child())
            owned.sample()
            states = owned.stop(self.limits(), Mock(), "reaped-original-leader")
        self.assertEqual(factory.call_count, 1)
        self.assertNotIn(9999, [row["pid"] for row in states])
        self.assertFalse(any(row["create_time"] == 99. for row in states))
        group.assert_not_called()

    def test_vanished_owned_group_cannot_receive_term_or_kill(self):
        owned = self.owned(Handle(1002, alive=False))
        with patch.object(guards.os, "getpgid") as pgid, \
             patch.object(guards.os, "killpg") as group:
            owned.stop(self.limits(), Mock(), "gone-group-control")
        pgid.assert_not_called()
        group.assert_not_called()

    def test_escaped_identity_is_signalled_without_unowned_group(self):
        worker = Handle(1002)
        owned = self.owned(worker)
        with patch.object(guards.os, "getpgid", return_value=777), \
             patch.object(guards.os, "killpg") as group:
            states = owned.stop(self.limits(), Mock(), "escaped-control")
        group.assert_not_called()
        self.assertEqual(worker.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertFalse(any(row["live"] for row in states))

    def test_only_verified_same_group_members_authorize_group_signal(self):
        worker = Handle(1002)
        owned = self.owned(worker)
        with patch.object(guards.os, "getpgid", return_value=1001), \
             patch.object(guards.os, "killpg") as group:
            owned.stop(self.limits(), Mock(), "same-group-control")
        self.assertEqual([call.args for call in group.call_args_list],
                         [(1001, signal.SIGTERM), (1001, signal.SIGKILL)])

    def test_worker_gone_before_kill_does_not_authorize_reused_group(self):
        worker = Handle(1002)
        owned = self.owned(worker)
        def stop_on_term(group_id, signum):
            self.assertEqual(group_id, 1001)
            worker.alive = False
        with patch.object(guards.os, "getpgid", return_value=1001), \
             patch.object(guards.os, "killpg", side_effect=stop_on_term) as group:
            owned.stop(self.limits(), Mock(), "group-vanishes-after-term")
        self.assertEqual([call.args for call in group.call_args_list], [(1001, signal.SIGTERM)])

    def test_missing_or_permission_denied_group_does_not_abort_other_cleanup(self):
        for error in (ProcessLookupError("injected race"), PermissionError("injected denied pgid")):
            with self.subTest(error=type(error).__name__):
                worker = Handle(1002)
                owned = self.owned(worker)
                with patch.object(guards.os, "getpgid", side_effect=error), \
                     patch.object(guards.os, "killpg") as group:
                    states = owned.stop(self.limits(), Mock(), "group-inspection-control")
                group.assert_not_called()
                self.assertEqual(worker.signals, [signal.SIGTERM, signal.SIGKILL])
                self.assertFalse(any(row["live"] for row in states))

    def test_access_denied_discovery_retains_known_handles_and_cleanup(self):
        root = Handle(1001, discovery_error=psutil.AccessDenied(pid=1001))
        worker = Handle(1002)
        owned = self.owned(worker, root=root)
        with patch.object(guards.os, "getpgid", return_value=777), \
             patch.object(guards.os, "killpg") as group:
            states = owned.stop(self.limits(), Mock(), "discovery-denied-control")
        group.assert_not_called()
        self.assertEqual(worker.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertTrue(owned.signal_errors)
        self.assertFalse(any(row["live"] for row in states))

    def test_uninspectable_unsignalable_worker_is_unknown_live_not_success(self):
        blocked = Handle(1002, status_error=psutil.AccessDenied(pid=1002),
                         signal_error=psutil.AccessDenied(pid=1002))
        healthy = Handle(1003)
        owned = self.owned(blocked, healthy)
        with patch.object(guards.os, "getpgid", return_value=777), \
             patch.object(guards.os, "killpg") as group:
            states = owned.stop(self.limits(), Mock(), "status-and-signal-denied-control")
        group.assert_not_called()
        self.assertEqual(healthy.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(blocked.signals, [signal.SIGTERM, signal.SIGKILL])
        row = next(row for row in states if row["pid"] == blocked.pid)
        self.assertTrue(row["live"])
        self.assertNotIn(row["state"], ("gone", psutil.STATUS_ZOMBIE))
        self.assertTrue(owned.signal_errors)

    def run_mock_supervisor(self, child, *, process_error=None, cleanup_error=None):
        journals = []
        original = guards.Journal
        class TrackingJournal(original):
            def __init__(self, path):
                super().__init__(path)
                journals.append(self)
        with tempfile.TemporaryDirectory(prefix="dongxi-owned-failure-control-") as directory:
            with patch.object(guards, "Journal", TrackingJournal), \
                 patch.object(guards.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(guards.psutil, "Process", side_effect=process_error,
                              return_value=Handle(child.pid, alive=False)), \
                 patch.object(guards.os, "killpg") as group:
                if cleanup_error is None:
                    result = guards.supervise_owned_fixture("success", Path(directory) / "run",
                        limits=self.limits(), memory_probe=memory, conflict_probe=scan)
                else:
                    with patch.object(guards.OwnedWorkers, "stop", side_effect=cleanup_error):
                        result = guards.supervise_owned_fixture("success", Path(directory) / "run",
                            limits=self.limits(), memory_probe=memory, conflict_probe=scan)
            self.assertEqual(spawn.call_count, 1)
            group.assert_not_called()
            self.assertTrue(journals[0].handle.closed)
            self.assertTrue(Path(result["journal"]["path"]).is_file())
            self.assertTrue((Path(directory) / "run" / "result.json").is_file())
            return result

    def test_leader_identity_access_denied_still_stops_direct_owned_popen(self):
        child = Child(alive=True)
        result = self.run_mock_supervisor(child, process_error=psutil.AccessDenied(pid=child.pid))
        self.assertEqual(child.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(result["actual_leader_exit_code"], -signal.SIGKILL)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["cleanup_errors"])

    def test_unexpected_cleanup_failure_still_closes_journal_and_reaps_leader(self):
        child = Child(alive=True)
        result = self.run_mock_supervisor(child, cleanup_error=RuntimeError("injected cleanup control"))
        self.assertEqual(child.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(result["actual_leader_exit_code"], -signal.SIGKILL)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["type"], "RuntimeError")

    def test_fallback_term_permission_failure_still_attempts_kill_and_closes(self):
        class DeniedTermChild(Child):
            def send_signal(self, signum):
                if signum == signal.SIGTERM:
                    self.signals.append(signum)
                    raise PermissionError("injected direct-owned TERM denial")
                return super().send_signal(signum)
        child = DeniedTermChild(alive=True)
        result = self.run_mock_supervisor(child, cleanup_error=RuntimeError("injected cleanup control"))
        self.assertEqual(child.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(result["actual_leader_exit_code"], -signal.SIGKILL)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(row.get("type") == "PermissionError"
                            for row in result["cleanup_errors"]))

    def test_direct_symlink_root_is_not_followed(self):
        with tempfile.TemporaryDirectory(prefix="dongxi-owned-disk-control-") as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "real" / "payload").write_bytes(b"outside sampled tree")
            (root / "link").symlink_to(root / "real", target_is_directory=True)
            with self.assertRaises((ValueError, OSError)):
                guards.disk_sample(root / "link")

    def test_child_directory_swapped_to_symlink_is_never_followed(self):
        with tempfile.TemporaryDirectory(prefix="dongxi-owned-disk-race-control-") as directory:
            root = Path(directory)
            sample = root / "sample"; sample.mkdir()
            child = sample / "child"; child.mkdir()
            outside = root / "outside"; outside.mkdir()
            (outside / "payload").write_bytes(b"not part of the sampled private tree")
            original_open = guards.os.open
            swapped = False
            def swap_then_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "child" and kwargs.get("dir_fd") is not None and not swapped:
                    swapped = True
                    child.rename(sample / "old-child")
                    child.symlink_to(outside, target_is_directory=True)
                return original_open(path, flags, *args, **kwargs)
            with patch.object(guards.os, "open", side_effect=swap_then_open):
                with self.assertRaises((ValueError, OSError)):
                    guards.disk_sample(sample)
            self.assertTrue(swapped)


if __name__ == "__main__":
    unittest.main()

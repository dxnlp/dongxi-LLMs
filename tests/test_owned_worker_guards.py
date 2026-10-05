"""Real bounded CPU descendants/file caps; injections remain labelled controls."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms import owned_worker_guards as guards


def memory():
    return {"available_bytes": 100 * guards.GIB, "source": "injected-test-not-actual-memory"}


def scan(**kwargs):
    return {"source": "injected-clearance-not-actual-scan", "conflicts": [], "unreadable": 0,
            "scanned": 1, "raw_command_lines_retained": False, "environments_read": False}


@unittest.skipUnless(guards.sys.platform == "linux", "actual Linux worker/file-limit controls")
class OwnedGuardTests(unittest.TestCase):
    def run_fixture(self, mode, **kwargs):
        with tempfile.TemporaryDirectory(prefix="dongxi-owned-worker-test-") as directory:
            kwargs.setdefault("memory_probe", memory)
            kwargs.setdefault("conflict_probe", scan)
            result = guards.supervise_owned_fixture(mode, Path(directory) / "evidence",
                                                     **kwargs)
            events = [json.loads(line) for line in Path(result["journal"]["path"]).read_text().splitlines()]
            text = (Path(directory) / "evidence" / "stdout.txt").read_text() if result["owned_final_states"] else ""
            return result, events, text

    def test_success_has_actual_exit_and_no_live_owned_process(self):
        result, events, _ = self.run_fixture("success")
        self.assertEqual(result["actual_leader_exit_code"], 0)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(any(row["live"] for row in result["owned_final_states"]))
        self.assertEqual(events[-1]["event"], "actual-exit")

    def test_deadline_stops_same_group_and_escaped_term_ignoring_workers(self):
        for mode in ("blocked-tree", "escaped-worker"):
            result, events, text = self.run_fixture(mode, limits=guards.Limits(seconds=.8))
            self.assertEqual(result["stop_reason"], "external-deadline")
            self.assertEqual(result["actual_leader_exit_code"], -signal.SIGKILL)
            worker_pid = json.loads(text.splitlines()[0])["worker_pid"]
            self.assertIn(worker_pid, [row["pid"] for row in result["owned_final_states"]])
            self.assertFalse(any(row["live"] for row in result["owned_final_states"]))
            requests = [row for row in events if row["event"] == "owned-worker-signal"]
            self.assertTrue(any(row["pid"] == worker_pid and row["signal"] == signal.SIGKILL for row in requests))
            self.assertLess(result["elapsed_seconds"], 3)

    def test_successful_leader_with_running_worker_is_not_safe_success(self):
        result, _, text = self.run_fixture("parent-exits-with-worker")
        self.assertEqual(result["actual_leader_exit_code"], 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stop_reason"], "leader-exited-with-owned-workers")
        worker_pid = json.loads(text.splitlines()[0])["worker_pid"]
        self.assertIn(worker_pid, [row["pid"] for row in result["owned_final_states"]])
        self.assertFalse(any(row["live"] for row in result["owned_final_states"]))

    def test_real_file_cap_is_signal_exit_not_whole_directory_quota(self):
        result, _, _ = self.run_fixture("single-file-overflow")
        self.assertEqual(result["actual_leader_exit_code"], -signal.SIGXFSZ)
        self.assertEqual(result["status"], "failed")
        self.assertLess(result["maximum_sampled_artifact_bytes"], result["disk_limits"]["aggregate_bytes"])

    def test_actual_aggregate_stop_counts_all_files_and_is_labelled_sampled(self):
        result, _, _ = self.run_fixture("aggregate-file-growth",
                                        disk_limits=guards.DiskLimits(per_file_bytes=8192, aggregate_bytes=48 * 1024))
        self.assertEqual(result["stop_reason"], "sampled-artifact-byte-cap")
        self.assertGreaterEqual(result["maximum_sampled_artifact_bytes"], 48 * 1024)
        self.assertTrue(all(row["disk"]["source"].startswith("actual-sampled") for row in result["observations"]))
        self.assertFalse(any(row["live"] for row in result["owned_final_states"]))

    def test_resource_gate_refuses_without_launch_or_signal(self):
        def disk(output):
            return {"source": "injected-low-free-test", "artifact_bytes": 0, "free_bytes": 0}
        with patch.object(subprocess, "Popen") as spawn, patch.object(guards.os, "killpg") as kill:
            result, _, _ = self.run_fixture("success", disk_probe=disk)
        self.assertEqual(result["status"], "gate-rejected")
        self.assertIsNone(result["actual_leader_exit_code"])
        self.assertEqual(result["stop_reason"], "filesystem-free-below-threshold")
        spawn.assert_not_called(); kill.assert_not_called()

    def test_unknown_command_and_invalid_bounds_never_spawn(self):
        with patch.object(subprocess, "Popen") as spawn:
            with self.assertRaises(ValueError): guards.supervise_owned_fixture("train", Path("unused"))
            for limits in (guards.DiskLimits(per_file_bytes=True), guards.DiskLimits(aggregate_bytes=100),
                           guards.DiskLimits(minimum_free_bytes=-1), guards.DiskLimits(per_file_bytes=2 * 1024**2)):
                with self.assertRaises(ValueError): limits.validate()
        spawn.assert_not_called()

    def test_symlinks_and_existing_output_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir(); (root / "link").symlink_to(root / "real", target_is_directory=True)
            with self.assertRaises(ValueError): guards.supervise_owned_fixture("success", root / "link" / "new")
            with self.assertRaises(FileExistsError): guards.supervise_owned_fixture("success", root / "real")
            (root / "real" / "file").symlink_to(root / "missing")
            with self.assertRaises(ValueError): guards.disk_sample(root / "real")

    def test_hardlinks_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "a").write_bytes(b"123456")
            guards.os.link(root / "a", root / "b")
            self.assertEqual(guards.disk_sample(root)["artifact_bytes"], 6)

    def test_cleanup_journal_failure_does_not_strand_workers(self):
        original = guards.Journal.event
        def event(journal, kind, **fields):
            if kind.startswith("owned-"): raise OSError("injected cleanup-journal failure")
            return original(journal, kind, **fields)
        with patch.object(guards.Journal, "event", event):
            result, _, _ = self.run_fixture("escaped-worker", limits=guards.Limits(seconds=.8))
        self.assertTrue(result["cleanup_errors"])
        self.assertFalse(any(row["live"] for row in result["owned_final_states"]))

    def test_observer_failure_cleanup_retains_actual_exit(self):
        count = 0
        def probe(output):
            nonlocal count
            count += 1
            if count > 12: raise OSError("injected runtime disk-probe failure")
            return guards.disk_sample(output)
        result, _, _ = self.run_fixture("escaped-worker", disk_probe=probe)
        self.assertEqual(result["stop_reason"], "supervisor-failure")
        self.assertEqual(result["failure"]["type"], "OSError")
        self.assertIsNotNone(result["actual_leader_exit_code"])
        self.assertFalse(any(row["live"] for row in result["owned_final_states"]))

    def test_conflict_pid_is_not_adopted_as_owned_and_raw_arguments_are_rejected(self):
        conflict = scan(); conflict["conflicts"] = [{"pid": 987654321, "rss_bytes": 5 * guards.GIB,
                                                     "reasons": ["large-resident-process"]}]
        with patch.object(guards.os, "killpg") as kill:
            result, _, _ = self.run_fixture("success", conflict_probe=lambda **kw: deepcopy(conflict))
        kill.assert_not_called(); self.assertEqual(result["owned_final_states"], [])
        conflict["private_args"] = "SECRET-DO-NOT-RETAIN"
        result, events, _ = self.run_fixture("success", conflict_probe=lambda **kw: deepcopy(conflict))
        self.assertEqual(result["failure"]["type"], "ValueError")
        self.assertNotIn("SECRET", json.dumps(events))


if __name__ == "__main__": unittest.main()

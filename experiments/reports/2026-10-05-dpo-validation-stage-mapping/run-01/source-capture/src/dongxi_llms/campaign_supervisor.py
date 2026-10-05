"""External direct-child CPU supervision; no model launcher or acquisition API.

Only fixed self-spawned fixtures can run. Process inspection never signals a
discovered PID and never retains command lines or environments. Sampled memory
and direct-child containment are deliberately narrower than production safety.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .run_identity import file_digest

GIB = 1024**3
MODES = frozenset(("success", "exit7", "self-signal", "ignore-term"))
MODEL_SIGNATURES = ("vllm", "sglang", "tritonserver", "ollama", "torchrun",
                    "train_stories.py", "run_chapter09_spark_sft.py",
                    "run_chapter11_spark_dpo.py", "dongxi_llms.qwen_rlvr_lab")


@dataclass(frozen=True)
class Limits:
    seconds: float = 2.
    reserve_bytes: int = 25 * GIB
    sampling_seconds: float = .02
    term_grace_seconds: float = .1
    kill_reap_seconds: float = 1.

    def validate(self):
        for value in (self.seconds, self.sampling_seconds, self.term_grace_seconds, self.kill_reap_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Finite positive CPU fixture timing bounds required")
        if self.seconds > 5 or self.sampling_seconds > .1 or self.term_grace_seconds > 1 or self.kill_reap_seconds > 2:
            raise ValueError("Only the bounded CPU fixture supervisor is implemented")
        if type(self.reserve_bytes) is not int or self.reserve_bytes < 25 * GIB:
            raise ValueError("Host reserve must remain at least 25 GiB")
        return self


def host_memory():
    text = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("Actual Linux MemAvailable is required")
    return {"available_bytes": int(match[1]) * 1024, "source": "actual-linux-MemAvailable"}


def conflicts(*, exclude=(), proc_root=Path("/proc"), large_rss_bytes=4 * GIB):
    """Conservative sanitized signature/RSS scan, not a complete GPU-idle check.

    Raw argument text is inspected only in memory and is not returned. Permission
    failures remain an incomplete clearance, while vanished processes are races.
    """
    excluded = set(exclude) | {os.getpid()}
    found, unreadable, scanned = [], 0, 0
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            status = (directory / "status").read_text()
            # Limit inspection and do not persist arbitrary process arguments.
            with (directory / "cmdline").open("rb") as handle:
                arguments = handle.read(65537)
            if len(arguments) > 65536:
                unreadable += 1
                continue
            arguments = [x.decode("utf-8", errors="replace") for x in arguments.split(b"\0") if x]
            rss = re.search(r"^VmRSS:\s+(\d+) kB$", status, re.MULTILINE)
            rss_bytes = int(rss[1]) * 1024 if rss else 0
            matches = [name for name in MODEL_SIGNATURES if any(
                Path(arg).name == name or arg == name or arg.startswith(name + ".") for arg in arguments)]
            reasons = (["known-model-process-signature"] if matches else [])
            if rss_bytes >= large_rss_bytes:
                reasons.append("large-resident-process")
            if reasons:
                found.append({"pid": int(directory.name), "rss_bytes": rss_bytes, "reasons": reasons})
            scanned += 1
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, UnicodeError):
            unreadable += 1
    return {"source": "actual-read-only-proc-scan", "conflicts": found, "unreadable": unreadable,
            "scanned": scanned, "raw_command_lines_retained": False, "environments_read": False}


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open("x", encoding="utf-8")
        self.started = time.monotonic()
        self.errors = []

    def event(self, kind, **fields):
        row = {"event": kind, "utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - self.started, **fields}
        self.handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def _signal_owned(process, value, journal, reason):
    if process.poll() is not None:
        return
    # A logging failure cannot strand a running child. Only this Popen object is
    # signalled; discovered process IDs are never a signal target.
    try:
        journal.event("signal-requested", child_pid=process.pid, signal=value, reason=reason)
    except BaseException as error:
        journal.errors.append({"type": type(error).__name__, "during": "signal-journal", "message": str(error)[:512]})
    try:
        process.send_signal(value)
    except ProcessLookupError:
        pass


def _stop_owned(process, limits, journal, reason):
    _signal_owned(process, signal.SIGTERM, journal, reason)
    try:
        return process.wait(timeout=limits.term_grace_seconds)
    except subprocess.TimeoutExpired:
        _signal_owned(process, signal.SIGKILL, journal, reason)
        return process.wait(timeout=limits.kill_reap_seconds)


def supervise_fixture(mode, output, *, limits=None, memory_probe=host_memory, conflict_probe=conflicts):
    """Supervise one fixed leaf fixture. No arbitrary command/model run is accepted.

    Injected probes are controls and are labelled in their observations. A future
    production integration, authorization and descendant containment do not exist.
    """
    limits = (limits or Limits()).validate()
    if mode not in MODES:
        raise ValueError("Only fixed CPU fixture children are authorized here")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "events.jsonl")
    process, exit_code, reason, failure, minimum = None, None, None, None, None
    command = [sys.executable, "-m", "dongxi_llms.campaign_supervisor", "--fixture-child", mode]
    result = {"fixture": mode, "limits": asdict(limits), "child_command": command,
              "child_pid": None, "actual_exit_code": None, "status": "preflight",
              "resource_measurement": "sampled, not continuous", "observations": [],
              "scope": "self-spawned leaf CPU child only; no model job or descendant supervision"}
    out = err = None
    def observe():
        nonlocal minimum
        value = memory_probe()
        if set(value) != {"available_bytes", "source"} or type(value["available_bytes"]) is not int or value["available_bytes"] < 0:
            raise ValueError("Invalid labelled resource observation")
        scan = conflict_probe(exclude=(() if process is None else (process.pid,)))
        if (not isinstance(scan, dict) or set(scan) != {"source", "conflicts", "unreadable", "scanned", "raw_command_lines_retained", "environments_read"}
            or not isinstance(scan.get("conflicts"), list) or type(scan.get("unreadable")) is not int
            or type(scan.get("scanned")) is not int or min(scan["unreadable"], scan["scanned"]) < 0
            or scan["raw_command_lines_retained"] is not False or scan["environments_read"] is not False
            or any(set(row) != {"pid", "rss_bytes", "reasons"} or type(row["pid"]) is not int
                   or type(row["rss_bytes"]) is not int or row["rss_bytes"] < 0
                   or not set(row["reasons"]) <= {"known-model-process-signature", "large-resident-process"}
                   for row in scan["conflicts"])):
            raise ValueError("Invalid sanitized conflict observation")
        row = {**value, "conflict_scan": scan}
        result["observations"].append(row)
        minimum = value["available_bytes"] if minimum is None else min(minimum, value["available_bytes"])
        journal.event("resource-sample", **row)
        if value["available_bytes"] < limits.reserve_bytes:
            return "host-reserve-below-threshold"
        if scan["conflicts"] or scan["unreadable"]:
            return "conflicting-or-uninspectable-process"
        return None
    try:
        journal.event("preflight", fixture=mode, limits=asdict(limits))
        reason = observe()
        if reason is None:
            out = (output / "stdout.txt").open("x")
            err = (output / "stderr.txt").open("x")
            # Fixed fixtures need no credentials or inherited service settings.
            environment = {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "LANG": "C.UTF-8",
                           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=out, stderr=err, env=environment)
            result["child_pid"] = process.pid
            journal.event("child-started", child_pid=process.pid, command=command)
            while process.poll() is None:
                if time.monotonic() - started >= limits.seconds:
                    reason = "external-deadline"
                    break
                reason = observe()
                if reason is not None:
                    break
                time.sleep(min(limits.sampling_seconds, max(0., limits.seconds - (time.monotonic() - started))))
            exit_code = _stop_owned(process, limits, journal, reason) if process.poll() is None else process.wait()
            result["supervised_seconds"] = time.monotonic() - started
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:512]}
        reason = "observer-interrupted" if isinstance(error, KeyboardInterrupt) else "supervisor-failure"
        try:
            journal.event("failure", **failure)
        finally:
            if process is not None and process.poll() is None:
                exit_code = _stop_owned(process, limits, journal, reason)
    finally:
        # This path also covers journal errors; it never touches another process.
        if process is not None:
            if process.poll() is None:
                exit_code = _stop_owned(process, limits, journal, "finally-cleanup")
            else:
                exit_code = process.wait()
        if out is not None: out.close()
        if err is not None: err.close()
        try:
            journal.event("actual-exit", child_pid=None if process is None else process.pid,
                          exit_code=exit_code, signal=(-exit_code if exit_code is not None and exit_code < 0 else None),
                          stop_reason=reason, failure=failure)
        finally:
            journal.close()
    result.update(actual_exit_code=exit_code, stop_reason=reason, failure=failure,
                  signal_journal_errors=journal.errors,
                  minimum_sampled_available_bytes=minimum,
                  status="gate-rejected" if process is None and reason else
                         "interrupted" if reason == "observer-interrupted" else
                         "failed" if failure or (exit_code is not None and exit_code != 0) else "completed",
                  journal={"path": str(output / "events.jsonl"), "sha256": file_digest(output / "events.jsonl")})
    with (output / "result.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _fixture_child(mode):
    if mode == "success":
        print("original CPU fixture completed", flush=True)
        return 0
    if mode == "exit7":
        print("deliberate CPU fixture failure", flush=True)
        return 7
    if mode == "self-signal":
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGUSR1)
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("CPU fixture installed ignore-TERM handler", flush=True)
        while True: time.sleep(60)
    raise ValueError("Unknown CPU fixture")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-child", choices=sorted(MODES), required=True)
    raise SystemExit(_fixture_child(parser.parse_args().fixture_child))

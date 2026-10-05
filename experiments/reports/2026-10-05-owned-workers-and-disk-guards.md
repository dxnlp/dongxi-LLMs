# Owned workers and disk limits: what the controls prove

Eight fixed CPU controls pass, with25 focused guard regressions and14 unchanged
campaign tests. The [predeclared specification](../specs/2026-10-05-owned-workers-and-disk-guards.md)
and [final raw acceptance](2026-10-05-owned-worker-final/acceptance.json)
retain actual exits, observations, worker identities, signals, failures, command,
environment and seven stable source hashes. This extends DXI-03 source readiness;
it does not launch a model or implement a production supervisor.

## Observed outcomes

| Control | Actual leader exit | Outcome |
|---|---:|---|
| Ordinary completion |0|No retained live worker; completed|
| TERM-ignoring same-group worker |−9|External deadline; KILL and cleanup|
| TERM-ignoring independent-session worker |−9|External deadline; retained identity-safe worker cleanup|
| Leader exits while its worker continues |0|Shutdown criterion fails; worker stopped|
| Single growing file |−25|Actual SIGXFSZ at the child file-size limit|
| Many growing files |−15|Sampled aggregate-byte threshold; cleanup|
| Injected low filesystem free space |null|Preflight refused; no child started|
| Injected runtime observer failure |−9|Failure retained; owned worker/leader stopped|

The final panel leaves no retained live worker. Its real minimum sampled
MemAvailable is118.0607567GiB, comfortably above the unchanged25GiB threshold;
this is sampled, not continuous. The aggregate48KiB control records60502 logical
artifact bytes at its largest sample, demonstrating overshoot rather than
pretending the sample is a hard quota. Final journal/result writes are outside
that last sample. Per-file and aggregate constraints have different meanings.

Worker PID/create-time identities and observed process groups/sessions are
recorded. The independent-session control actually leaves the leader's group.
The cleanup retains its discovered handle rather than treating a conflict-scan
PID as authority to signal. Zombies are distinguished from live execution;
these fixtures ended with retained identities classified gone. The implementation
is not a subreaper. Filesystem sampling uses FD-relative `O_NOFOLLOW` directories
and anchored filesystem free-space reads; logical inode payload is counted once.

## Fault review and preserved diagnostics

Independent mocked controls exposed unsafe naked-PID reacquisition, signals to
a vanished group, and cleanup interruptions after denied inspection. The source
now retains the original leader handle, requires a retained live group member,
keeps unknown status conservative, and continues individual cleanup. A further
mock exposed denied TERM in the direct-child fallback aborting before KILL and
file/journal closure. Its denial is now retained while cleanup continues. These
are authored fault controls, not measured PID reuse or actual permission incidents.

The [first focused test attempt](2026-10-05-owned-worker-first-test-attempt.json)
retains a test-helper duplicate-keyword error. The
[first collection](2026-10-05-owned-worker-fixtures/acceptance.json) failed and
preserves its raw diagnostics/source drift rather than being overwritten. An
[intermediate passing collection](2026-10-05-owned-worker-release/acceptance.json)
is historical for its then-current test bytes; a subsequent independent
assertion changed that test file. The final collection reruns all eight controls
and39 tests against unchanged current sources. It exits0 in8.921758 seconds;
its SHA256 is`a9aa4a81d7c2eeab3aa0fa22e2961f3544c3382f4814790867d8a87ec25bda50`.

Reproduce only into a new unused directory, using the existing CPU environment:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  scripts/verify_owned_worker_guards.py --output /tmp/UNUSED-owned-worker-evidence
```

The script accepts an evidence destination, not arbitrary child commands. It
does not install an environment, start a server or interact with model weights.

## Limits and next implementation gate

The hard limit is per child-written file; aggregate disk and memory guards are
sampled. Discovery can miss rapid reparenting, and group checks cannot eliminate
every check/signal race. Trusted cooperative output paths are not a hostile
same-user namespace sandbox. Supervisor crashes, cgroup integration, total disk
quotas and real trainer-worker behavior remain production gates. The
[recovery plan](../../docs/PRODUCTION_RECOVERY_PLAN.md) names the unfinished actual
runner interfaces. All45 model-scale campaign rows remain unexecuted. No GPU,
acquisition, installation, service, publication, media or Git operation occurred.

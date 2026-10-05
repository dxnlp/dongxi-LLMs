# Owned workers and disk guards: bounded CPU contract

This specification precedes its fixture execution. It extends DXI-03 source
readiness, not permission to start a model campaign. All subprocesses are fixed
standard-library fixtures in a new private output directory, using the existing
isolated CPU interpreter. No arbitrary command, external PID, model, download,
service, environment installation or Git action is accepted.

## Questions and predictions

A leader can exit while an owned worker remains alive. Therefore a successful
leader exit is insufficient evidence of shutdown. A worker may also start its
own session, so a group signal alone need not reach it. Periodically retain
identity-checked descendant handles before the leader exits, signal only that
owned set and the group created by this invocation, and verify surviving live
processes afterward. The finite fixture keeps its escaped worker alive long
enough for discovery; this does not test arbitrary rapid reparenting or hostile
daemonization. Such production containment still requires a platform/cgroup
contract and recovery validation.

A hard POSIX file-size limit applies separately to each child-written file and
is inherited by its workers. It is not a whole-directory quota. Sample aggregate
regular-file bytes in the private output, reject symlinks, and stop at a declared
aggregate threshold; retain the largest sample. This sampled stop can overshoot
between observations and does not reserve filesystem space. Independently
sample filesystem free bytes and keep their minimum. A model checkpoint is not
automatically safe merely because these fixture checks pass.

## Fixed controls and acceptance

The source accepts only success, blocked-tree, escaped-worker,
parent-exits-with-worker, single-file-overflow and aggregate-file-growth modes.
The deadline is at most five seconds, sampling at most 0.1 seconds, TERM grace
at most one second and KILL grace at most two seconds. Keep the host reserve at
least 25 GiB, using the existing labelled memory and sanitized conflict probes.
Use separate exclusive stdout, stderr and fsynced journal files; preserve old
outputs. CPU fixture file caps stay between 4 KiB and 1 MiB and aggregate caps
between that file cap and 8 MiB. The fixed writer produces 4 KiB chunks.

Acceptance requires actual CPU success, deadline cleanup of TERM-ignoring
same-group and independently-sessioned workers, cleanup after a successful
leader leaves a worker, the real SIGXFSZ boundary for one growing file, an actual
sampled aggregate stop, and preflight low-free-space rejection with no spawn.
Additional tests cover invalid limits, journal/observer failure cleanup, missing
process races, output preservation and refusal of symlinked outputs. Conflict
inspection never supplies a signal target; probes that inject low resources
must remain labelled controls, not actual host shortages.

Record actual leader exit separately from safety status, every owned worker's
PID/create-time identity, signal attempts/errors, final live/zombie classification,
sampled host/free/artifact extrema, source hashes and invocation. Zombies count
as exited execution but remain reported; the supervisor is not a subreaper.
Never persist arbitrary process command lines or environments.

## Evidence boundary

Passing these checks proves bounded cleanup and guards for these cooperative
CPU fixtures only. Discovery is sampled, process groups can be escaped, the
supervisor itself can fail, and aggregate disk/memory readings are not hard
quotas. The module deliberately exposes no production launcher. Actual
approved runner integration, platform containment, pretrained recovery,
profile/pilot evidence and every external campaign outcome stay pending.

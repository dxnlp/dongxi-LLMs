# Native G4 recovery run01: retained cleanup failure

Date:2026-10-05. Scope: independent read-only diagnosis from saved source,
supervision events, terminal receipts, native reports and bounded metadata.
No models, tensor/weight-body rehashes, tests, new children, signals, GPU jobs,
producer edits or acceptance changes were performed by this review.

## What actually completed, and what did not

The original fixed G4 source2/completed1→2/pending1→2 invocation sequence reached
native exit0 for all three children. Its pending child's supervisor nevertheless
failed the **owned-descendant cleanup acknowledgment** gate. The adapter stops
before its final numerical comparison and acceptance. It is not a passed
recovery prerequisite for a pilot.

| Original role | Native PID / exit | Supervised status | Child seconds | Supervisor seconds | Minimum sampled available bytes |
| --- | --- | --- | ---: | ---: | ---: |
| Source2 | 301052 / 0 | completed | 179.35619652300375 | 179.40738152200356 | 109,313,617,920 |
| Completed1→2 | 301516 / 0 | completed | 131.65557820600225 | 131.7237069410039 | 103,996,592,128 |
| Pending1→2 | 301672 / 0 | failed | 107.62926277797669 | 107.69894294301048 | 103,994,474,496 |

The third [returned receipt](native-rlvr-g4-recovery-20261005-run-01/returned-supervision-3.json)
has null signal, stop reason, native failure and journal error, but contains
`cleanup_errors=["Observed descendant cleanup timed out"]`. It retains observed
owned descendant PID301696/create-time1791204632.62. Its three cleanup/observer/
logger helpers all eventually have actual exit−15, `still_alive=false` and
`cleanup_complete=true`. Those helper outcomes establish their later bounded
termination, **not** receipt of the descendant-cleanup worker's success message.

The event journal records normal-leader-exit session TERM at elapsed107.0339348
and KILL at107.1345269, followed by actual-exit at107.6989098 with the cleanup
error. No external600-second deadline, memory-reserve or conflict stop occurred.
The root separately witnessed an empty GPU query and absence of the retained
owned PIDs at12:53:40UTC; this later observation does not retroactively satisfy
the failed acknowledgment deadline. The owner-observed outer adapter interval,
456.23976223001955 seconds/exit1, overlaps preparation, native supervision and
retained-input checks. It is not a fourth model invocation.

Every native `report.json` says `completed`, with completed cursor2 and a final
completed2 snapshot/marker/independent receipt and full policy export present.
The pending/completed continuation reports each have one update2 row; they
exactly equal the source report's update2 row, including retained responses,
old likelihoods, pool/RNG observations and numerical metrics. That bounded JSON
comparison is not an independently established final policy/reference/Adam/RNG
tensor-storage equality.

The stage's [failure record](native-rlvr-g4-recovery-20261005-run-01/failure.json)
is retained. `acceptance.json` and `closing-bindings.json` do not exist. Source
`execute` checks each supervised status before calling `check_results`, so its
final streamed symbolic-storage comparison has **not been run** for this failed
stage. Three native exit0 results do not replace that comparison or the failed
supervisor gate.

## Retained spending and negative behavior

All three native reports identify the same physical main journal
`48acf3064a9641b0949db2788abdf9ec`/file identity[66306,5901344], and I/O journal
`cc986044bf6545a88080982809cd577a`/[66306,5901346]. The cumulative reports advance
main completed `(collections,train_updates)` from(2,2) to(3,3) to(3,4): the pending
continuation applies its retained pool without another collection. I/O completed
`(inspect,load,save)` advances(0,0,5)→(1,1,8)→(2,2,10). Later source/replay work
is not reset to a checkpoint prefix or refunded because the adapter failed.
Their native operation journals have no open/failed tickets; the later external
supervisor cleanup failure is a different boundary, not an invented failed
training call.

Both original source updates have all-zero strict-integer rewards and relative
advantages. Update1 naturally emits equation text rather than only the integer;
update2 caps all four responses. The stored first gradient is nonzero at finite
precision, and update2 includes nonzero exact reference KL and gradient. Thus
completed optimizer calls are not four positive-reward learning observations,
and zero advantages alone do not imply unchanged numerical parameters. These
two-update smoke records do not establish quality improvement or common20
heldout performance.

## Code-supported startup hypothesis, not a captured child stack

The [supervisor](../../src/dongxi_llms/native_profile_supervisor.py) deliberately
uses a late **spawn** worker, not fork after its controller feeder threads start.
The `received.poll(reap_seconds)` acknowledgment limit is0.5 seconds; lack of a
message appends the exact observed timeout, and cleanup errors keep the result
failed. That behavior is intentional fail-closed supervision.

The original [RLVR controller](native-rlvr-g4-recovery-20261005-run-01/source-capture/scripts/run_native_rlvr_stages.py) imports
`qwen_rlvr_lab` at module scope, which immediately imports Torch and numerical
modules. Python spawn prepares the controller's `__main__` module before running
the cleanup target. Consequently heavy controller import startup is inside the
same0.5-second cleanup acknowledgment window. The first two native children had
no observed descendants and never needed that late worker; the third did.
This provides a specific plausible cause of the third cleanup timeout. The
saved events do not capture the helper's stack or separate import/target timings,
so this review does not state that startup latency is experimentally proven.

The controller and its original test were reconstructed after the failure by
an exact inverse patch, not captured before launch. This review independently
checked their retained bytes against run01's preparation: controller38,738 bytes/
SHA256`7a8322d31a025e219268b6e79b3549e03f5dbb222f7ada2b235297b57b1b82ce`,
test21,480 bytes/SHA256
`3bc3b0460082651a1232e54ff388ee1b4af3d14ebfac619e0220d7389213482d`.
That establishes a content-address-matched postfailure archive, not a new
prelaunch-capture claim.

The narrow proposed repair is to keep model-heavy numerical imports out of the
controller's module-import path, loading them only in functions that need them.
That can let the same spawn helper begin promptly without increasing the
acknowledgment/native deadlines, replacing spawn with unsafe threaded fork,
changing owned PID/create-time checks, signaling unrelated processes, changing
weights/sampler/reference/caps/save schedule or waiving any gate. A bounded
fresh-process regression is needed before a separately retained authorized
continuation. The parent owns any implementation and closed retry selection.

Run01, its three native receipts, original source identities and all physical
journals remain unchanged and failed. G8 recovery and every dependent pilot or
common20 evaluation remain unaccepted until their own actual gates and receipts.
This diagnosis creates no new mandatory successful Base/chat baseline gate and
does not sign DXI-03 or the whole goal complete.

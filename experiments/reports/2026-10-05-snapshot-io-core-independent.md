# Independent checkpoint IO admission review

The final shared trusted-local core passed 60 focused tests and all six separately
reproduced failure controls on 2026-10-05 UTC. Admission now precedes the tested
payload reads, finite scans, clones and serializer calls. Later spending remains
charged to the independently bound physical journal. This is a scoped source/CPU
review, not evidence of runner CLI ordering, numerical model continuation,
pretrained execution, adversarial authentication or physical containment.

## Preserve failures before interpreting the pass

The first passing 43-test panel did not include the controls that exposed these
gaps. The reviewer preserved each actual result and its source hashes before the
owner changed the shared code:

| Reproduced gap | Initial observation | Final targeted observation |
| --- | --- | --- |
| Save tensor-byte limit | An 8-byte float64 tensor was finite-scanned under a 1-byte envelope before clone refusal | Refused before any finite scan |
| Load tensor-byte limit | Authored 16-byte state returned by a patched restricted deserializer was accepted under an 8-byte envelope | Refused before any finite scan |
| Mutable bound receipt | Changing prefix sequence 0 to `False` after binding was accepted | Exact-type refusal before any marker/payload read |
| Mixed reference format | Marker version 2 with an unchanged version 1 payload was accepted by the unhooked loader | Refused before restricted loading |
| Mutable scientific identity | A changed public scientific hash allowed a save under an old-science journal | Refused before a tree walk or reservation |
| Mutable physical journal | Replacing the public journal pointer allowed the parent to load while charging a new journal | Refused before any read; both journals unchanged |

The first three results remain in
[run-01 diagnostics](2026-10-05-snapshot-io-core-independent/run-01/diagnostics.json).
The format mismatch remains in
[run-02](2026-10-05-snapshot-io-core-independent/run-02/schema-diagnostic.json).
The next two identity results remain in
[run-03](2026-10-05-snapshot-io-core-independent/run-03/identity-mutation-diagnostics.json).
The deserializer patch is an explicit malformed-state control, not a claim that
an actual measured checkpoint was contaminated. Public-object mutation controls
test accidental identity changes; they do not authenticate same-UID writers or
constitute a malicious-process sandbox.

## Final actual checks

The final child used the existing isolated CPU interpreter, hidden CUDA, offline
flags and one thread:

```bash
env PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest tests.test_snapshot_io_budget tests.test_training_snapshot tests.test_snapshot_io_schedule -v
```

It exited 0. The 60 tests comprise 25 current IO-core controls, nine legacy snapshot
controls and 26 separate pure schedule controls. The collected child duration was
1.812304093 seconds. Eight source/protocol files were byte-identical before and
after the final panel and targeted diagnostics. The full
[test log](2026-10-05-snapshot-io-core-independent/run-04/tests.json) and
[six diagnostic rechecks](2026-10-05-snapshot-io-core-independent/run-04/diagnostics.json)
are retained. The final
[verification record](2026-10-05-snapshot-io-core-independent/run-04/verification.json)
has SHA256 `5273b368ee3ac9dc7f6f1c2fa84104b6a797e531cfc2325bf85cab355c6abbce`.

Read-only source inspection confirms separate inspect/load/save reservations
before shared contract walking and payload processing. Expected contract,
observed contract and state visits share the whole-operation counters. Tensor
bytes and expanded-view elements refuse before finite scans; save clone and
serialization costs retain conservative reservations on failure. Actual entered
and successful visits are distinct from reserved upper bounds. Legacy unhooked
version 1 references remain explicit; accounted artifacts use version 2, and mode
or marker version mismatch does not reach deserialization.

The independently retained bounded receipt binds exact bytes, science, IO
contract and the checkpoint's pre-save journal prefix. Opening the same journal
from that prefix retains all later failed charges; copying a journal refuses
its physical identity. Receipt mutation, scientific identity mutation and journal
replacement now refuse before processing. A valid checkpoint cannot include its
own future save completion; the pre-save prefix is deliberate, not a refund.

## Scope of the result

Payload hash bytes do not include every metadata, contract or journal hash.
Logical visit units do not measure CPU instructions, allocation peaks or elapsed
time. Bounded bootstrap parsing and journal verification, caller state capture,
runner semantic validation/application, artifact inventory, other outputs and
restricted-deserializer internals remain separately scoped. The loader still
requires trusted local artifacts; explicit envelopes do not sandbox hostile
tensor metadata or arbitrary child processes.

The separate pure supplier was aligned to the consumer's one-million-node
ceiling under a predeclared additive protocol. Its original 25-test records and
the unchanged nineteen DPO dimensions remain historical evidence; the
[26-test hardening report](2026-10-05-snapshot-io-schedule-node-bound.md) is new.
No runner, shared module or canonical book/tracker was edited by this reviewer.
Actual runner fresh-process replay, pre-model CLI admission, physical quotas,
GPU/pretrained/Mac/hosted CI evidence and broader output budgeting are not
established by this review. Day 9 and external-run status are unchanged.

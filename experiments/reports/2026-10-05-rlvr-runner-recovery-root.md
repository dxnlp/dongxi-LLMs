# Actual-loop RLVR recovery: independent root acceptance

The final bounded CPU source gate passes:19 actual-loop RLVR tests and15
identity/merge tests, both test-process exits0. All16 scoped source/lock hashes
remain unchanged during the10.529404-second panel. The
[final acceptance record](2026-10-05-rlvr-runner-recovery-root-final/acceptance.json)
has SHA256 `02aa799a3a67543302f678bd20740419228596da1000fb8ea0e1d5e9f07f6365`.
It retains actual commands, raw logs, package/platform identities and four Linux
MemAvailable observations. Their minimum is125,807,996,928 bytes, above25GiB;
this is not a continuous interval minimum or model-scale resource profile.

## What was tested

The fixed [premeasurement specification](../specs/2026-10-05-rlvr-runner-recovery.md)
uses two original random local Qwen3 configurations/seeds2323/2324, one layer,
width16, vocabulary16, two query/one KV heads and context32. Four CPU FP32 updates
use group3, a four-token cap, AdamW0.008, no weight decay, clipping1 and beta0.02.
No pretrained weight/tokenizer download, GPU or service is involved.

Each seed resumes after two completed updates and, separately, after fully
collecting the third group. Exact state/tail comparisons include policy,
**original** reference, Adam, source cursor, Python/Torch/sampling RNG, full
numerical history, collected/applied work and actual pools. Separate Python
processes replay both phases. A no-collection spy proves that pending restoration
applies the retained pool before new sampling. Multi-update comparisons with the
retained original `update_model` equations match the wrapper without changing
sampling, loss, rewards, learning rate or schedule.

Naturally sampled groups are retained even when all task rewards are zero.
Seed2323's tiny nonzero KL-roundoff-driven gradients/updates must not be labeled
an exact zero-update arm. Seed2324 obtains a naturally sampled successful answer
on update4; it is not selected as a replacement seed. Separately authored
positive/stop/cap/zero-signal controls are explicitly not unforced sampled results.
This is recovery/implementation evidence, not arithmetic or language capability.

Failure controls reject stale/malformed pools, changed bytes/contracts, wrong
cursor/history/reference lineage, incompatible layouts/aliases and malformed RNG
before state application. Partial backward/optimizer state is poisoned, not
committed. Initial/pending/completed receipts precede observers; the final
completed receipt precedes evaluation/export. Status records the authoritative
durable count/phase separately from this invocation's appended rows. Additive
evaluation telemetry retains stop-inclusive raw IDs, known work and partial
error rows; legacy decoded-token semantics are not silently changed. Guards
precede old-policy/reference scoring and known successful partial scoring work
is retained without claiming unknown failed-call costs.

The unchanged eight-token identity fixture still proves local HF/nonzero-LoRA
merge/reload. Its arithmetic CLI path now correctly fails encoded train/evaluation
collision checks instead of passing because all words alias to UNK. The
[initial migration diagnostic](2026-10-05-rlvr-identity-encoded-collision.json)
preserves that actual failure. A separately authored21-token random-HF fixture
then exercises the positive CPU CLI identity path. No check or original merge
fixture was weakened to manufacture a pass.

## Historical collections and limits

The [first root candidate panel](2026-10-05-rlvr-runner-recovery-root/acceptance.json)
passed17 RLVR plus15 identity tests before final callback/durable-status/guard
hardening. It remains historical, not final-source acceptance. Owner collections
retain their first13, later17/18 and final19 controls separately; numerical
histories from the fixed two-seed recipe are not replaced by favorable results.

The final module SHA256 is
`c2d00a3eb8881291f04572645d2e0bf1b52cf2c17536887657b004ca27b6f3bf`;
the focused test SHA256 is
`384b1eafb00bd47cb79cdfa1e91824e39e0b56103f36d9eff9242480fa575b92`.
The accepted format remains trusted-local, not authentication or a hostile tensor
allocation sandbox. Mid-generation recovery, real power loss, arbitrary-process
containment, hard aggregate disk quota, pretrained/CUDA/BF16 recovery and
cross-machine determinism remain unverified.

Chapter13§13.7.1, its question/answer18, lab route and cross-day boundary artifact
place this evidence in the coherent course. The learner remains Day9;13of18
bounded packages and all45 null external campaign outcomes remain unchanged.
Continue the [production source plan](../../docs/PRODUCTION_RECOVERY_PLAN.md),
not an automatic GPU/profile/pilot. This panel is not a full-course or notebook
verification; those runs keep their own identities. No Git write was performed.

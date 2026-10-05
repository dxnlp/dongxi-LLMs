# Cumulative work and numerical replay in the RLVR runner

The actual random local HF CPU runner now reserves complete collection,
application, evaluation and recovery-validation operations. Its 39-test current
panel passes while preserving the archived equations, original reference and
completed/pending numerical replay. Later failed spending survives a restore;
an older optimizer cursor does not create a new allowance.

This is source readiness under cooperative logical accounting, not pretrained
Qwen training, CUDA/BF16 acceptance, a FLOP/backward-dispatch bound, physical
containment or a storage quota. All 45 external campaign outcomes and Day 9
learning status remain unchanged. No shared WorkLedger or snapshot source was
edited, and no downloads, installs, services, GPU jobs or Git writes occurred.

## Protocol and actual verification

The [original protocol](../specs/2026-10-05-rlvr-cumulative-work.md) predates tests.
Two later source-only protocols cover [parsed cap identity](../specs/2026-10-05-rlvr-cap-byte-binding.md)
and [bounded identity rechecks](../specs/2026-10-05-rlvr-bounded-cap-identity.md).
Neither changes a training seed, cap, data record or numerical objective.

The final [run05 raw record](2026-10-05-rlvr-cumulative-work/run-05/verification.json)
has actual exit 0, 39 passing tests (20 new work controls plus 19 original recovery
checks), 18.431 seconds unittest and 19.348793 seconds child process. All 12
source/spec/lock hashes remain equal before and after. Its adjacent
[observations](2026-10-05-rlvr-cumulative-work/run-05/observations.json) retain actual
random outputs, authored failures, every replay result and complete work summaries.
The [acceptance bindings](2026-10-05-rlvr-cumulative-work-verification.json)
identify current sources and historical evidence separately.

Execution uses the existing isolated Linux ARM64 Python 3.12.14/Torch 2.14.1+cpu
environment, hidden CUDA, offline Hugging Face and one numerical thread. The
collector bounds its test child at 120 seconds; replay children have 60-second
bounds and the no-peer FIFO CLI has 10 seconds. These verification durations are
not model-scale performance measurements or a continuous memory guarantee.

## What is reserved before work

The exact 23 dimensions distinguish update/collection/source counts,
rectangular generated slots, stop-inclusive sampled valid response tokens,
real multinomial draws, application response targets, uncached growing-prefix
positions and policy/reference/old-policy calls. Evaluation has its own
examples/calls/positions/output tokens. Recovery tracks validation operations,
policy likelihood calls/positions and uniform draws used to check historical
RNG consumption. Many dimensions overlap; adding them is not a FLOP estimate.

Collection reserves the full group and cap before its first forward or draw.
Finished rows still occupy rectangular forward/sampling slots under the
unchanged implementation, but their EOS fill is not counted as a valid action.
Application reserves validation plus current/reference scoring and the update
before its existing likelihood check. Save/restore reserve the complete retained
history's RNG checks and any pending likelihood forward. Evaluation encodes its
whole frozen panel first, then reserves that panel before its first forward.

An entered failing call and known successful work have different records.
Injected second-generation failure retains one completed generation forward,
three successful sample draws and nine logical positions of uncertain entered
forward work. Backward or optimizer internals are not invented from that input
count. Conservative reservations never refund early stopping or failures.

## Numerical parity and the difference between slots and valid tokens

Both declared seeds run four actual updates with the same original tiny model,
full-support temperature 1 sampling, population advantages, clipping and exact
reference KL. Budgeted results match both the unbudgeted loop and the archived
pre-recovery update equations. Policy/Adam bytes and rollout RNG agree at every
update; the original reference remains unchanged.

| Four-update seed | Reserved rectangular slots | Actual rectangular slots | Valid sampled response tokens | Reserved policy positions | Actual policy positions |
|---|---:|---:|---:|---:|---:|
|2323|48|48|41|411|411|
|2324|48|45|38|405|387|

Policy positions include uncached generation, old-policy scoring and the
existing pending-policy validation/current branch. The latter two branch lengths
are known after the complete pool is collected. Reference scoring uses 69 and 66
positions respectively. These particular four-update rows do not include
checkpoint validation or evaluation; lifecycle/replay observations separately
retain that additional work.

Seed 2323 produces only zero rewards. Seed 2324's fourth unforced group has
rewards `[1,0,0]`; its nonzero learning signal and the resulting policy/Adam update
are preserved. This distinction prevents a trivial all-zero replay from being
presented as the only numerical acceptance. No incorrect sampled answer is
gold-repaired or replaced.

## Recovery retains spending without resampling

Same-process replay covers completed and pending boundaries for both seeds.
Fresh-process replay independently covers both phases at seed 2323. Policy,
original reference, Adam, global/rollout RNG, source cursor, raw pools and full
numerical history agree. The pending child's first application runs under a
spy that forbids collection.

The explicit rollback control takes a snapshot after two updates, retains
the original continuation through four updates in the same physical journal,
then restores the older snapshot and replays its last two updates. The numerical
trajectory still ends at update 4, while the resource ledger correctly retains
six update reservations. The completed branch retains six collections; the
pending branch retains five because its saved group is consumed without a
replacement. Extra pending likelihood/RNG checks remain visible rather than
being folded into training costs.

A separate later-failed-collection control uses a one-collection cap. Restoring
the initial numerical snapshot retains that failed reservation and refuses the
retry before any forward or sample. Changed limits, copied journals, a new
unrelated allowance and a tampered snapshot prefix also refuse. A new evidence
output folder cannot refill the anchored journal. Same-UID malicious editing
and privileged rollback remain outside this trusted-local contract.

## Refusals and lifecycle failures

Whole collection/application/evaluation/payload-validation cap refusal is
verified with forward, sampling and optimizer spies. The response cap and group
are never reduced to fit. A reservation fsync failure also stops before model
work. Injected reference/optimizer failures retain charged applications and
restore the prior pending pool before retrying it.

Baseline/final evaluation, metric, commit-observer, export and pending-save
failures leave the last durable numerical receipt readable. A failed pending
save does not erase the already collected group's spending. The work journal
does not bound arbitrary file exports: snapshot/log/export/scratch byte routing
and actual physical containment remain separate unfinished gates.

The greedy evaluation cost probe reserves eight calls and 36 positions for two
length-three prefixes with cap 4. This actual random model emits an immediate
stop for both, completing two calls/six positions and two stop tokens. Neither
answer is correct. Both prompts intentionally use the same authored encoded
prefix in this accounting probe; it is not a source-independent capability or
generalization evaluation. The preserved actual production CLI uses its own
disjoint frozen arithmetic prompts. Observer/forward failures preserve partial
rows without assigning unknown failed-call tokens or GPU time.

## Cap bytes and bounded identity reads

The CLI requires explicit `--work-limits` and `--work-journal-max-bytes`; resume also
requires the same physical `--work-journal` together with independently retained
snapshot contract/SHA/exact size. The optional CPU API remains backward compatible
when no ledger is supplied. Declared budgets cannot claim an unrelated active
journal or snapshot contract.

Cap reads are bounded at 64 KiB, walk no-follow ancestors, open nonblocking before
checking regular-file type and reject changed bytes, duplicate JSON keys,
nonfinite values, unknown dimensions and bool aliases. The exact initially parsed
SHA is retained. Identity capture and closure hashing use the same bounded
reader, not a generic file open. Actual no-peer FIFO CLI refusal exits 1 in
0.604970 seconds, creates no output and starts no tokenizer/model load. Replacing
a previously valid cap input with a no-peer FIFO also refuses at the identity
closure check. This narrow input protection is not a sandbox for arbitrary
hostile model artifacts.

## Historical records and source identity

The [first command failure](2026-10-05-rlvr-cumulative-work-first-command.json)
stopped before tests because the collector's parent directory did not exist.
It is not reclassified as a pass. Directory creation was corrected without
weakening runner assertions.

The earlier [36-test run02](2026-10-05-rlvr-cumulative-work/run-02/verification.json),
[37-test run03](2026-10-05-rlvr-cumulative-work/run-03/verification.json) and
[38-test run04](2026-10-05-rlvr-cumulative-work/run-04/verification.json) remain
unchanged with their own source/spec identities. Additive active-contract and
cap-input checks explain the later revisions; they do not rewrite earlier
observations or select a favorable seed. Original runner-recovery evidence also
remains historical and unchanged.

Current runner SHA256 is
`819cd8bbfc1701416d050b14888cbc11d4128e143cb7c80266da9678b3f27df1`;
new test source SHA256 is
`6407a1c2d73a2509a5b0170e3307d691e32e43b48eae700c36f8e9f813dbd91a`.
Final raw verification SHA256 is
`7ae4cfe20e749377e276478f770782221446c0bd7aea4275a9e8c690471817d7`;
observations SHA256 is
`c41d9fcbfc45a9563a73fc4e8b2d365cfd861de43399232d04788b6846d83c2d`.
Temporary CPU checkpoint/journal paths in controls are cleaned after tests;
retained receipts are historical observations, not reusable deployed artifacts.

Next source work remains compiler-to-actual-cap mapping, other runner work and
artifact routing. Real private quota/containment/watchdog/GPU clearance and
pretrained/Spark replay require separately authorized actual evidence.

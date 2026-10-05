# Actual-loop SFT and DPO recovery: root acceptance

The bounded source gate passes: 78 tests execute through the shared format and
actual Chapter 9/11 update functions, with all four test-process exits 0. This is
not a pretrained compatibility, CUDA/BF16 replay, model-quality or learner pass.
The [final raw root panel](2026-10-05-runner-recovery-root-final/acceptance.json) retains exact
commands, logs, package identities and 17 scoped source hashes. All 17 remain
unchanged during the 13.850665-second panel. Its SHA256 is
`9bddee249e8a1ba374a04d1d96175d31fffc14d844100d27b434e1b859f68f11`.

| Gate | Root tests | What was exercised |
|---|---:|---|
| Shared core |8|Dense scalar/empty/noncontiguous/BF16 roundtrip, independent bytes/contract and exclusive commit|
| Independent format failures |31|No-load rejection, verified-buffer path replacement, bounds, failed writes/fsyncs and retained earlier commits|
| Actual SFT loop |18|Full/LoRA same-recipe replay, observed environment rejection, full history/counters/RNG, original equations, fresh process and poisoned/failure boundaries|
| Actual DPO loop |21|Original frozen reference, pair sampler/work/history, original equations, fresh process, contract/state and metric/export failures|

The scope is the fixed shared/SFT/DPO source and imported local dependencies.
Concurrent RLVR source construction is excluded, not implicitly accepted by
these 78 tests. No new notebook is executed in this panel; the historical
all-76 reference and later nine-notebook checks keep their earlier identities.

## Actual local models and observations

SFT constructs random Qwen3 models from configuration, not Hub weights: one
layer, width 16, vocabulary 32, FFN width 32, two query/KV heads, head dimension 8,
context 64 and attention dropout 0.1. Full mode trains 3,136 scalars; rank-two Q/V
LoRA trains 128. Both fixed seeds 1212/1213 execute four updates and restore after
update 2. Every arm reproduces the two subsequent numerical rows and whole
policy/Adam/RNG/order/cursor/work/history identity exactly, in the same process
and a separate Python process. Final target/physical-position totals are 27/118
and 25/110 by seed. Short LoRA losses can fluctuate; this is not a favorable-seed
selection or capability comparison. See the [specification](../specs/2026-10-05-sft-runner-recovery.md)
and [final four-arm raw collector](2026-10-05-sft-runner-recovery-environment-hardening-reference.json).

DPO uses the separately fixed seed 1818 random one-layer Qwen3 configuration:
vocabulary 16, width 16, FFN width 32, two query/one KV head and context 32. The unchanged
objective uses sequence sums, one shift, two pair microbatches, beta 0.2,
AdamW 0.008/weight-decay 0.01 and clipping 1. Six updates consume 12 pairs, 32 chosen
and 30 rejected targets, with 86 positions for each of policy and original
reference. Restoration at update 3 exactly replays the remaining selected
indices, metric rows and complete policy/reference/Adam/RNG/work state, including
a fresh process. The reference remains distinct from the trained policy.
These are authored symbolic pairs, not preference alignment or English quality.
See the [specification](../specs/2026-10-05-dpo-runner-recovery.md) and
[final raw collection](2026-10-05-dpo-runner-recovery/run-02/verification.json).

Root collection uses the unchanged isolated CPU environment, CUDA hidden,
HF offline flags and one numerical thread. All four external 60-second command
deadlines remain untriggered. Actual Linux MemAvailable observations before/after
commands reach a minimum 126,094,258,176 bytes, above the 25GiB reserve. This is a
minimum of those observations, not a continuous peak/minimum or production
memory profile. Snapshot file bounds do not predict full-model save/load overhead.

## Failed attempts and durability limits

The initial unsupported pytest invocation is [retained](2026-10-05-training-snapshot-first-command.json);
no test ran and no package was installed. SFT's initial real-tokenizer tests
exposed a changed chat-template return container; explicit `return_dict=False`
preserves the intended ID-list contract. The failed attempt is retained beside
its unchanged numerical recipe. Hugging Face documents this v5 return change
in its [migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md).
DPO's initial architecture-mutation test used a nonexistent configuration
attribute; its actual failure is retained rather than erased by the final pass.
Neither correction tunes the optimizer or chooses a favorable outcome.

The [first root panel](2026-10-05-runner-recovery-root/acceptance.json) passed
77 tests before SFT added its observed Python/platform/package contract check.
It remains historical, not current-source acceptance. The final 78-test panel
above follows the separate SFT environment-hardening collector. Its four-arm
numerical history is unchanged. DPO's first successful collector also remains
historical: it mislabeled a typed state digest as a canonical JSON contract
digest. Its retained identity clarification and separately named `run-02`
correct the receipt labels without changing the runner or recipe.

The format is trusted-local. Expected byte/size/contract inputs come from an
independently retained record, never an unchecked marker. Data-only loading
does not sandbox hostile tensor metadata. A post-publication fsync failure can
leave a readable marker with unconfirmed durability and must remain failed
evidence. No real power-loss behavior is measured. Immutable earlier states,
snapshot-carried history and new invocation outputs protect the stated replay
boundary, not all possible filesystem or process failures.

Raw numerical reports contain portable receipts and metrics. SFT states are in
the recorded temporary directory; DPO tensor states are local collector outputs.
The reports do not make those weight files available on another machine via Git
or establish a portable pretrained checkpoint.

## Coherent course placement and remaining work

Chapters 9 and 11 integrate the committed-boundary/reference explanation,
new worked answers and corrected lab CLI/output contracts. The writing workflow
keeps the lesson in its existing narrative rather than a detached note. The
[cross-day artifact](../../learning_artifacts/day-12-sft-mechanics/durable-training-boundaries.md)
extends CAND-ANIM-019 as a Mac proposal only; no rendering or publication.

The active goal remains 13 of 18 bounded packages complete, with DXI-01/02/03/17/18
partial. This panel left RLVR as a separate gate; the later
[RLVR root panel](2026-10-05-rlvr-runner-recovery-root.md) now passes its
completed/pending replay and identity checks. Current safe source work is
constrained stage/work/artifact/containment integration under the
[source plan](../../docs/PRODUCTION_RECOVERY_PLAN.md). All 45 external
campaign outcomes remain null. Approved pretrained compatibility, current-source
Spark numerical recovery/profile, actual Mac/hosted checks and dependent final
evidence require separate gates. The learner remains Day 9 and Git is not written.
